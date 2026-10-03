import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og


def answer(msgs):
    # the text of a model answer that calls no tool
    parts = [p for m in msgs for p in m.get("parts", [])]
    if any(p.get("type") == "tool_call" for p in parts):
        return None
    return " ".join(p.get("content", "") for p in parts if p.get("type") == "text").strip() or None


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit("the run has no agent definitions")
    spans = read(run_dir / "spans.jsonl")
    parent = {s["id"]: s["parent"] for s in spans}

    rec = Record(f"pydantic-ai {v['pydantic-ai-slim']} with its OpenTelemetry instrumentation (version 6)",
                 head["run_id"], True)
    og.session(rec, spans)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    model_end = {e["response_id"]: e for e in ev if e["kind"] == "model_end"}
    runs = {e["run_id"]: e for e in ev if e["kind"] == "run_start"}
    by_call = defaultdict(list)
    for e in ev:
        if e.get("tool_call_id") and e["kind"] not in ("tool_call", "tool_result"):
            by_call[e["tool_call_id"]].append(e)
    retried = {e["tool_call_id"] for e in ev if e["kind"] == "tool_result" and e["part"] == "retry-prompt"}

    steps, invocation_of_run = {}, {}
    invocations = Counter()
    model_steps, tool_steps, results = [], {}, {}
    answers, agent_tools = [], []
    retry_of, requested_by, held, written = {}, {}, {}, []

    def outcome(s):
        return s["status"], s["error"] if s["status"] == "failed" else None

    for s in spans:
        a, sid, op = s["attrs"], s["id"], og.op(s)
        up = steps.get(s["parent"])
        agent = rec.agent(a.get("gen_ai.agent.name"))
        run_id = a.get("gen_ai.agent.call.id")
        if op == "invoke_agent":
            name = a["gen_ai.agent.name"]
            runs.pop(run_id)
            invocations[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, up, name, s["start"], s["end"], *outcome(s))
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, agent)
            candidates = {**agents["delegates"].get(name, {}), **agents["hands_off"].get(name, {})}
            if candidates:
                rec.g.add((step, RDF.type, PAI.Routing))
                for c in candidates.values():
                    rec.link(step, PAI.candidate, rec.agent(c))
            invocation_of_run[run_id] = step
            steps[sid] = step

        elif op == "chat":
            step = rec.step(sid, PAI.ModelCall, up, s["name"], s["start"], s["end"], *outcome(s))
            rec.link(step, PROV.wasAssociatedWith, agent)
            inp, out = og.model_call(rec, step, a)
            native = model_end.pop(a.get("gen_ai.response.id"), None)
            if native is None:
                # the app logs a model response only when the request succeeded
                if s["status"] == "completed":
                    raise SystemExit(f"chat span {sid} has no model response in the events")
                native = {}
            rec.set(step, PAI.iteration, native.get("run_step"))
            cost = (native.get("provider_details") or {}).get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            model_steps.append((step, inp, out))
            text = answer(out)
            if text:
                msg = rec.iri("entity", f"{sid}-output")
                answers.append((sid, msg, text, a["gen_ai.agent.name"]))
            for c in og.tool_call_ids(out, "tool_call"):
                requested_by.setdefault(c, step)
            steps[sid] = step

        elif op == "execute_tool":
            name, cid = a["gen_ai.tool.name"], a.get("gen_ai.tool.call.id")
            step = rec.step(sid, PAI.ToolCall, up, name, s["start"], s["end"], *outcome(s))
            rec.link(step, PROV.wasAssociatedWith, agent)
            og.tool_call(rec, step, a)
            tool_steps[cid] = step
            if a.get("gen_ai.tool.call.result") is not None:
                results[sid] = rec.iri("entity", f"{sid}-result")
            routing = invocation_of_run.get(run_id)
            target = agents["delegates"].get(a["gen_ai.agent.name"], {}).get(name)
            if target:
                rec.link(routing, PAI.selected, rec.agent(target))
                rec.link(rec.agent(target), PROV.actedOnBehalfOf, agent)
                rec.link(agent, PROV.actedOnBehalfOf, user)
                agent_tools.append((sid, a.get("gen_ai.tool.call.result")))
            # a failed call sends the model back; its next call of the same tool in the same run is the retry
            key = (run_id, name, same(a.get("gen_ai.tool.call.arguments")))
            if key in retry_of:
                prior, attempt = retry_of.pop(key)
                rec.link(step, PAI.retryOf, prior)
                rec.set(step, PAI.attempt, attempt + 1)
                if cid in retried:
                    retry_of[key] = (step, attempt + 1)
            elif cid in retried:
                retry_of[key] = (step, 1)
            for e in by_call.pop(cid, []):
                if e["kind"] == "handoff":
                    rec.g.add((step, RDF.type, PAI.Handoff))
                    src, dst = rec.agent(e["source"]), rec.agent(e["target"])
                    rec.link(step, PAI.fromAgent, src)
                    rec.link(step, PAI.toAgent, dst)
                    rec.link(dst, PROV.actedOnBehalfOf, src)
                    rec.link(routing, PAI.selected, dst)
                    agent_tools.append((sid, a.get("gen_ai.tool.call.result")))
                elif e["kind"] in ("memory_write", "memory_read"):
                    access = rec.step(f"{sid}-{e['kind']}-{e['time']}", PAI.MemoryAccess, step, e["kind"].replace("_", " "),
                                      e["time"], e["time"], "completed")
                    rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
                    if e["kind"] == "memory_write":
                        record = rec.memory_write(e["memory"], e["key"], e["value"], access)
                        written.append((record, e["by"], e["value"].strip(), sid, seconds(e["time"])))
                    else:
                        record = rec.memory_read(e["memory"], e["key"], access)
                        value = rec.g.value(record, PROV.value) if record is not None else None
                        if value is not None and a.get("gen_ai.tool.call.result") == str(value):
                            rec.derived(results.get(sid), record, "inferred")
                elif e["kind"] == "effect":
                    effect = rec.entity(f"{sid}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
                    rec.set(effect, PAI.target, e["target"])
                    rec.generated(effect, step)
                    rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))
            steps[sid] = step

    for e in ev:
        if e["kind"] == "deferred_requests":
            run = invocation_of_run.get(e["run_id"])
            for c in e["approvals"]:
                step = rec.step(f"{c['tool_call_id']}-held", PAI.ToolCall, run, c["tool"], e["time"], e["time"],
                                "interrupted")
                rec.link(step, PROV.wasAssociatedWith, rec.agent(e["agent"]))
                rec.link(step, PAI.usedTool, rec.tool(c["tool"]))
                args = rec.entity(f"{c['tool_call_id']}-arguments", PROV.Entity, f"{c['tool']} arguments",
                                  json.dumps(c["args"]))
                rec.used(step, args)
                if c["tool_call_id"] in requested_by:
                    model = rec.local(requested_by[c["tool_call_id"]])
                    rec.used(step, rec.iri("entity", f"{model}-output"))
                held[c["tool_call_id"]] = args
        elif e["kind"] in ("check", "approval"):
            check = rec.step(f"{e['run_id']}-{e['kind']}-{e['time']}", PAI.Check, invocation_of_run.get(e["run_id"]), e["kind"],
                             e["time"], e["time"], "completed")
            rec.set(check, PAI.outcome, e["outcome"])
            if e["kind"] == "approval":
                rec.link(check, PROV.wasAssociatedWith, approver)
                if e["approved_call"] not in held or e["approved_call"] not in tool_steps:
                    raise SystemExit(f"approved call {e['approved_call']} was not held or did not run")
                rec.link(check, PAI.checked, held[e["approved_call"]])
                done = tool_steps[e["approved_call"]]
                rec.used(done, held[e["approved_call"]])
                rec.link(done, PROV.wasInformedBy, check)
            else:
                rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
                rec.link(check, PAI.checked, draft_before(answers, e["time"], spans, e["by"]))
                rec.set(check, PAI.iteration, e["iteration"])
                rec.set(check, PAI.reason, e["reason"])

    og.link_tool_calls(rec, model_steps, tool_steps)

    def inside(sid, top):
        while sid:
            if sid == top:
                return True
            sid = parent.get(sid)

    # an agent run inside a tool returns its final answer as the tool result
    for fid, output in agent_tools:
        for sid, out, text, _ in answers:
            if inside(sid, fid) and text == (output or "").strip():
                rec.derived(results.get(fid), out, "inferred")

    # a note written from an agent's answer given earlier inside the same tool call
    ends = {s["id"]: seconds(s["end"]) for s in spans if s["end"]}
    for record, by, value, fid, at in written:
        out = latest([(ends.get(sid), out) for sid, out, text, agent in answers
                      if agent == by and text == value and inside(sid, fid)], at)
        if out is not None:
            rec.derived(record, out, "inferred")

    leftover = list(runs) + list(model_end) + list(by_call)
    if leftover:
        raise SystemExit(f"{len(leftover)} native events had no matching span: {leftover}")
    return rec


def draft_before(answers, time, spans, checker):
    # the draft a check looked at: the last answer before the check by an agent other than the checker
    end = {s["id"]: seconds(s["end"]) for s in spans if s["end"]}
    return latest([(end.get(sid), msg) for sid, msg, _, agent in answers if agent != checker], seconds(time))
