import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import openinference as oi

OI_KIND = {"generation": "LLM", "function": "TOOL", "handoff": "TOOL", "agent": "AGENT", "custom": "CHAIN"}


def stamp(t):
    # the SDK and the OpenTelemetry spans write the same start time in different formats
    return round(seconds(t), 6)


def answer(output):
    # the text of an assistant message from a generation output
    try:
        msgs = json.loads(output) if isinstance(output, str) else output
    except ValueError:
        return None
    return " ".join(m.get("content") or "" for m in msgs or [] if isinstance(m, dict)).strip() or None


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    trace = next((e for e in ev if e["kind"] == "trace_start"), None)
    if agents is None or trace is None:
        raise SystemExit("the run has no agent definitions or no trace start")
    defs, group = agents["definitions"], trace.get("group_id")
    spans = [e for e in ev if e["kind"] == "span"]
    parent = {s["id"]: s.get("parent_id") for s in spans}

    def depth(sid):
        n = 0
        while parent.get(sid):
            sid, n = parent[sid], n + 1
        return n

    spans.sort(key=lambda s: (seconds(s["started_at"]), depth(s["id"])))
    by_id = {s["id"]: s for s in spans}
    usage = defaultdict(list)
    for e in ev:
        if e["kind"] == "model_end":
            usage[e["agent"]].append(e.get("usage") or {})

    otel = {}
    for s in read(run_dir / "spans.jsonl"):
        if s["parent"] is not None:
            otel.setdefault((stamp(s["start"]), oi.kind(s)), []).append(s)

    def otel_of(s):
        key = (stamp(s["started_at"]), OI_KIND[s["span_data"]["type"]])
        found = otel[key].pop(0) if otel.get(key) else None
        if otel.get(key) == []:
            del otel[key]
        if found is None:
            raise SystemExit(f"native span {s['span_data'].get('name')} has no OpenTelemetry span")
        return found

    rec = Record(f"openai-agents {v['openai-agents']} with openinference-instrumentation-openai-agents "
                 f"{v['openinference-instrumentation-openai-agents']}", head["run_id"], True)
    if group:
        rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, group, group))
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    steps, results = {}, {}
    invocations, model_calls = Counter(), Counter()
    model_steps, tool_steps = [], defaultdict(list)
    calls_of_turn = {}
    pending, failed = {}, {}
    answers, agent_tool_results, answered = [], [], []

    def up(sid, kind):
        while sid:
            if by_id[sid]["span_data"]["type"] == kind:
                return sid
            sid = parent.get(sid)

    def agent_name(sid):
        a = up(sid, "agent")
        return a and by_id[a]["span_data"]["name"]

    def status(step, s):
        err = s.get("error")
        rec.set(step, PAI.status, "failed" if err else "completed")
        if err:
            rec.set(step, PAI.errorType, (err.get("data") or {}).get("error") or err.get("message"))

    for s in spans:
        d, sid = s["span_data"], s["id"]
        kind = d["type"]
        name = d.get("name")
        up_step = steps.get(s.get("parent_id"))
        if kind == "custom" and name in ("task", "turn"):
            data = d["data"]
            label = f"turn {data['turn']}" if name == "turn" else f"runner run: {data['name']}"
            step = rec.step(sid, PAI.Step, up_step, label, s["started_at"], s["ended_at"])
            if name == "turn":
                rec.set(step, PAI.iteration, data["turn"])
            status(step, s)
            steps[sid] = step
            otel_of(s)

        elif kind == "agent":
            invocations[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, up_step, name, s["started_at"], s["ended_at"])
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            status(step, s)
            spec = defs.get(name, {})
            agent_tools = [t["origin"]["agent_name"] for t in spec.get("tools", [])
                           if (t.get("origin") or {}).get("type") == "agent_as_tool"]
            if agent_tools or spec.get("handoffs"):
                rec.g.add((step, RDF.type, PAI.Routing))
                for a in agent_tools + spec.get("handoffs", []):
                    rec.link(step, PAI.candidate, rec.agent(a))
            steps[sid] = step
            otel_of(s)

        elif kind == "generation":
            span = otel_of(s)
            agent = agent_name(sid)
            step = rec.step(sid, PAI.ModelCall, up_step, "generation", s["started_at"], s["ended_at"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            status(step, s)
            inp, out = oi.model_call(rec, step, span["attrs"])
            cost = None
            if not s.get("error"):
                # the run hooks report the usage of each call that returned, in order
                cost = (usage[agent][model_calls[agent]] if model_calls[agent] < len(usage[agent]) else {}).get("cost")
                model_calls[agent] += 1
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            called = oi.tool_calls(span["attrs"])
            calls_of_turn[s.get("parent_id")] = list(called)
            model_steps.append((step, out, [c["id"] for c in called], oi.tool_results(span["attrs"])))
            text = answer(span["attrs"].get("output.value"))
            if text and not called:
                answers.append((sid, out, text))
                answered.append((seconds(s["ended_at"]), agent, out, text))
            steps[sid] = step

        elif kind == "function":
            span = otel_of(s)
            call = (name, same(d.get("input")))
            held = next((p for p in pending.values() if p["call"] == call), None)
            # a call approved by a person runs from the saved state, outside the agent span
            agent = agent_name(sid) or (held and held["agent"])
            step = rec.step(sid, PAI.ToolCall, up_step, name, s["started_at"], s["ended_at"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            status(step, s)
            a, res = oi.tool_call(rec, step, span["attrs"])
            turn_calls = calls_of_turn.get(s.get("parent_id"), [])
            match = next((c for c in turn_calls if c["name"] == name), None)
            if match:
                turn_calls.remove(match)
                cid = match["id"]
            elif held:
                cid = next(c for c, p in pending.items() if p is held)
                del pending[cid]
                rec.used(step, held["arguments"])
                rec.link(step, PROV.wasInformedBy, held.get("check"))
            else:
                cid = None
            tool_steps[cid].append((step, res))
            if d.get("output") is None and not s.get("error"):
                pending[cid] = {"step": step, "call": call, "arguments": a, "agent": agent}
            if agent in defs:
                tool = next((t for t in defs[agent].get("tools", []) if t["name"] == name), None)
                if tool and (tool.get("origin") or {}).get("type") == "agent_as_tool":
                    selected = rec.agent(tool["origin"]["agent_name"])
                    routing = steps.get(up(sid, "agent"))
                    rec.link(routing, PAI.selected, selected)
                    rec.link(selected, PROV.actedOnBehalfOf, rec.agent(agent))
                    rec.link(rec.agent(agent), PROV.actedOnBehalfOf, user)
                    agent_tool_results.append((sid, res, d.get("output")))
            attempt = 1
            key = (agent,) + call
            if key in failed:
                before, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, before)
                rec.set(step, PAI.attempt, attempt)
            if s.get("error"):
                failed[key] = (step, attempt)
            steps[sid], results[sid] = step, res

        elif kind == "handoff":
            otel_of(s)
            step = rec.step(sid, PAI.Handoff, up_step, "handoff", s["started_at"], s["ended_at"])
            status(step, s)
            src, dst = rec.agent(d["from_agent"]), rec.agent(d["to_agent"])
            rec.link(step, PROV.wasAssociatedWith, src)
            rec.link(step, PAI.fromAgent, src)
            rec.link(step, PAI.toAgent, dst)
            rec.link(dst, PROV.actedOnBehalfOf, src)
            routing = steps.get(up(sid, "agent"))
            rec.link(routing, PAI.selected, dst)
            steps[sid] = step

        elif kind == "custom":
            otel_of(s)
            e = d["data"]
            if name in ("memory_write", "memory_read"):
                access = rec.step(sid, PAI.MemoryAccess, up_step, name.replace("_", " "), s["started_at"],
                                  s["ended_at"], "completed")
                rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
                if name == "memory_write":
                    record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                    value = (e.get("value") or "").strip()
                    found = latest([(end, out) for end, who, out, text in answered if who == e["by"] and text == value],
                                   seconds(s["started_at"]))
                    if found is not None:
                        rec.derived(record, found, "inferred")
                else:
                    record = rec.memory_read(e["memory"], e["key"], access)
                    tool = s.get("parent_id")
                    value = rec.g.value(record, PROV.value) if record is not None else None
                    if results.get(tool) is not None and value is not None and \
                            by_id[tool]["span_data"].get("output") == value.toPython():
                        rec.derived(results[tool], record, "inferred")
                steps[sid] = access
            elif name in ("check", "approval"):
                check = rec.step(sid, PAI.Check, up_step, name, s["started_at"], s["ended_at"], "completed")
                rec.link(check, PROV.wasAssociatedWith, approver if name == "approval" else rec.agent(e["by"]))
                if name == "approval":
                    held = pending.get(e["call_id"])
                    if held is None:
                        raise SystemExit(f"approval of call {e['call_id']} has no held call")
                    held["check"] = check
                    rec.link(check, PAI.checked, held["arguments"])
                    rec.g.remove((held["step"], PAI.status, None))
                    rec.set(held["step"], PAI.status, "interrupted")
                else:
                    # a check judges the latest answer before it that another agent gave
                    rec.link(check, PAI.checked, latest([(end, out) for end, who, out, _ in answered if who != e["by"]],
                                                        seconds(s["started_at"])))
                    rec.set(check, PAI.iteration, e.get("iteration"))
                rec.set(check, PAI.outcome, e["outcome"])
                rec.set(check, PAI.reason, e.get("reason"))
                steps[sid] = check
            elif name == "effect":
                effect = rec.entity(f"{sid}-effect", PAI.Effect, "effect", e["target"])
                rec.set(effect, PAI.target, e["target"])
                rec.generated(effect, up_step)
                rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    oi.link_tool_calls(rec, model_steps, tool_steps)

    def inside(sid, top):
        while sid:
            if sid == top:
                return True
            sid = parent.get(sid)

    # an agent used as a tool returns its final answer as the tool result
    for fid, res, output in agent_tool_results:
        for sid, out, text in answers:
            if inside(sid, fid) and text == (output or "").strip():
                rec.derived(res, out, "inferred")
    if otel:
        left = [s["name"] for group in otel.values() for s in group]
        raise SystemExit(f"{len(left)} spans had no matching native span: {left}")
    return rec
