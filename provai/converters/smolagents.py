import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import read, same, seconds
from . import openinference as oi


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    spec = next((e for e in ev if e["kind"] == "agents"), None)
    if spec is None:
        raise SystemExit("the run has no agent definitions")
    defs, settings, handoff = spec["definitions"], spec.get("settings") or {}, spec.get("handoff") or {}
    spans = read(run_dir / "spans.jsonl")
    by_id = {s["id"]: s for s in spans}

    rec = Record(f"smolagents {v['smolagents']} with openinference-instrumentation-smolagents "
                 f"{v['openinference-instrumentation-smolagents']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    # the steps smolagents keeps in each agent's memory, one per Step span, joined by step number and start time
    native = [e for e in ev if e["kind"] == "action_step"]
    answers = defaultdict(list)
    for e in ev:
        if e["kind"] == "final_answer":
            answers[e["agent"]].append(e["output"])

    nodes, agent_of, runs = {}, {}, Counter()
    step_native, requested, model_steps = {}, defaultdict(list), []
    tool_steps, failed, results, invocations = defaultdict(list), {}, {}, defaultdict(list)
    last_model, pending_results, model_out = {}, {}, {}

    def up(sid, kind):
        while sid in by_id:
            if oi.kind(by_id[sid]) == kind:
                return sid
            sid = by_id[sid]["parent"]

    def native_step(s):
        n = int(s["name"].split()[1])
        start = seconds(s["start"])
        agent = agent_of.get(s["parent"])
        # agents running in parallel can shift a native step's start, so a known agent gets a wider window
        limit = 0.05 if agent is None else 0.5
        found = sorted((e for e in native if e["step"] == n and abs(e["start"] - start) < limit
                        and (agent is None or e["agent"] == agent)), key=lambda e: abs(e["start"] - start))
        if not found:
            raise SystemExit(f"no native step for span {s['name']} at {s['start']}")
        native.remove(found[0])
        return found[0]

    for s in spans:
        kind, sid, a = oi.kind(s), s["id"], s["attrs"]
        parent = nodes.get(s["parent"])
        if kind == "AGENT" and s["name"].endswith(".run"):
            name = s["name"][:-len(".run")]
            runs[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, runs[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            agent_of[sid] = name
            invocations[name].append((sid, step))
            managed = (defs.get(name) or {}).get("managed_agents") or []
            if managed:
                rec.g.add((step, RDF.type, PAI.Routing))
                for m in managed:
                    rec.link(step, PAI.candidate, rec.agent(m))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, user)
            caller = s["parent"] and agent_of.get(s["parent"])
            if caller and name in ((defs.get(caller) or {}).get("managed_agents") or []):
                routing = invocations[caller][-1][1]
                rec.link(routing, PAI.selected, rec.agent(name))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, rec.agent(caller))
                rec.used(step, last_model.get(caller))
                if handoff.get("from") == caller and handoff.get("to") == name:
                    h = rec.step(f"{sid}-handoff", PAI.Handoff, parent, "handoff", s["start"], s["start"], "completed")
                    rec.link(h, PROV.wasAssociatedWith, rec.agent(caller))
                    rec.link(h, PAI.fromAgent, rec.agent(caller))
                    rec.link(h, PAI.toAgent, rec.agent(name))
            nodes[sid] = step

        elif kind == "CHAIN" and s["name"].startswith("Step "):
            n = native_step(s)
            agent_of[sid] = n["agent"]
            step = rec.step(sid, PAI.Step, parent, s["name"], s["start"], s["end"],
                            "failed" if n["error"] else "completed", n["error"] and n["error"]["type"])
            rec.set(step, PAI.iteration, n["step"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(n["agent"]))
            step_native[sid] = n
            nodes[sid] = step

        elif kind == "LLM":
            owner = up(s["parent"], "CHAIN")
            agent = agent_of.get(owner)
            step = rec.step(sid, PAI.ModelCall, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            inp, out = oi.model_call(rec, step, a)
            model_out[sid] = out
            # the app logs the model's settings in case the span leaves them out
            if (step, PAI.temperature, None) not in rec.g:
                rec.set(step, PAI.temperature, settings.get("temperature"))
            if (step, PAI.maxTokens, None) not in rec.g:
                rec.set(step, PAI.maxTokens, settings.get("max_tokens"))
            usage = (step_native.get(owner) or {}).get("usage") or {}
            rec.set(step, PAI.cost, None if usage.get("cost") is None else float(usage["cost"]))
            calls = oi.tool_calls(a)
            requested[owner].extend(calls)
            # the next model step of an agent run reads the results of the tools the previous step called
            for res in pending_results.get(agent) or []:
                rec.used(step, res)
            pending_results[agent] = []
            model_steps.append((step, out, [c["id"] for c in calls], []))
            last_model[agent] = out
            nodes[sid] = step

        elif kind == "TOOL" and a.get("tool.name") and a.get("tool.name") != "final_answer":
            name = a["tool.name"]
            owner = up(s["parent"], "CHAIN")
            agent = agent_of.get(owner)
            step = rec.step(sid, PAI.ToolCall, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            args, res = oi.tool_call(rec, step, a)
            match = next((c for c in requested[owner] if c["name"] == name), None)
            if match:
                requested[owner].remove(match)
                tool_steps[match["id"]].append((step, res))
            pending_results.setdefault(agent, []).append(res)
            key = (agent, name, same(a.get("input.value")))
            attempt = 1
            if key in failed:
                before, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, before)
                rec.set(step, PAI.attempt, attempt)
            if s["status"] == "failed":
                failed[key] = (step, attempt)
            results[sid] = (step, res, args)
            nodes[sid] = step

    # the step an agent takes when it reaches max_steps has no span; only its memory keeps it
    for n in list(native):
        if (n.get("error") or {}).get("type") != "AgentMaxStepsError":
            continue
        run_span = next((s for s in sorted(spans, key=lambda s: seconds(s["start"]), reverse=True) if s["name"] == f"{n['agent']}.run"
                         and seconds(s["start"]) <= n["start"] <= seconds(s["end"]) + 0.05), None)
        step = rec.step(f"{n['agent']}-{n['start']}", PAI.Step, run_span and nodes[run_span["id"]], f"Step {n['step']}",
                        iso(n["start"]), iso(n["end"]), "failed", "AgentMaxStepsError")
        rec.set(step, PAI.iteration, n["step"])
        rec.link(step, PROV.wasAssociatedWith, rec.agent(n["agent"]))
        native.remove(n)
    if native:
        raise SystemExit(f"{len(native)} native steps had no Step span")
    oi.link_tool_calls(rec, model_steps, tool_steps)

    def inside_run(sid, run):
        sid = by_id[sid]["parent"]
        while sid in by_id:
            if sid == run:
                return True
            if oi.kind(by_id[sid]) == "AGENT":
                return False
            sid = by_id[sid]["parent"]
        return False

    # each agent run ends with its final answer, which the caller's next model step reads
    answer_of = {}
    for name, agent_runs in invocations.items():
        for k, (sid, step) in enumerate(agent_runs):
            if k < len(answers[name]):
                ans = rec.entity(f"{sid}-answer", PAI.Message, f"{name} answer", answers[name][k])
                rec.generated(ans, step)
                answer_of[sid] = (ans, answers[name][k])
                # the answer is the argument of the final_answer call that the run's last model call made
                inside = [m for m in model_out if model_out[m] is not None and m in by_id and inside_run(m, sid)]
                if inside:
                    rec.derived(ans, model_out[max(inside, key=lambda m: seconds(by_id[m]["end"]))], "inferred")
                caller = by_id[sid]["parent"] and agent_of.get(by_id[sid]["parent"])
                nxt = next((ms for ms in spans if oi.kind(ms) == "LLM"
                            and agent_of.get(up(ms["parent"], "CHAIN")) == caller
                            and seconds(ms["start"]) > seconds(by_id[sid]["end"])),
                           None)
                if nxt:
                    rec.used(nodes[nxt["id"]], ans)

    def node(s):
        # covering() can return a span that made no step, such as the final_answer tool
        return s and nodes.get(s["id"])

    def covering(t, kind=None, agent=None):
        found = None
        for s in spans:
            if (kind is None or oi.kind(s) == kind) and seconds(s["start"]) <= t <= seconds(s["end"]) + 0.01:
                if agent and agent_of.get(s["id"] if oi.kind(s) == "AGENT" else up(s["parent"], "CHAIN")) != agent \
                        and not (oi.kind(s) == "AGENT" and s["name"] == f"{agent}.run"):
                    continue
                if found is None or seconds(s["start"]) > seconds(found["start"]):
                    found = s
        return found

    drafts = 0
    for e in ev:
        t = seconds(e["time"])
        if e["kind"] in ("memory_write", "memory_read"):
            key = f"{e['memory']}-{e['key']}"
            if e["kind"] == "memory_write":
                run_span = covering(t, "AGENT", e["by"]) or covering(t - 0.05, "AGENT", e["by"])
                access = rec.step(f"{key}-write-{e['time']}", PAI.MemoryAccess, node(run_span),
                                  "memory write", e["time"], e["time"], "completed")
                record = rec.memory_write(e["memory"], e["key"], e["value"], access)
                ans = answer_of.get(run_span["id"]) if run_span else None
                if ans and ans[1].strip() == (e["value"] or "").strip():
                    rec.derived(record, ans[0], "inferred")
            else:
                tool = covering(t, "TOOL", e["by"])
                access = rec.step(f"{key}-read-{e['time']}", PAI.MemoryAccess, node(tool),
                                  "memory read", e["time"], e["time"], "completed")
                record = rec.memory_read(e["memory"], e["key"], access)
                if tool and tool["id"] in results and record is not None:
                    step, res, _ = results[tool["id"]]
                    value = rec.g.value(record, PROV.value)
                    if res is not None and value is not None and str(rec.g.value(res, PROV.value)).strip() == str(value).strip():
                        rec.derived(res, record, "inferred")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))

        elif e["kind"] == "check":
            drafts += 1
            # the agent object that runs final answer checks is the one whose draft was checked
            checked = [covering(t, "CHAIN", a) for a, d in defs.items() if d.get("final_answer_checks")]
            step_span = max((x for x in checked if x), key=lambda x: seconds(x["start"]), default=None)
            draft = rec.entity(f"draft-{drafts}", PAI.Message, f"draft {drafts}", e.get("draft"))
            rec.generated(draft, node(step_span))
            check = rec.step(f"check-{drafts}-{e['time']}", PAI.Check, node(step_span), "check",
                             e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            rec.link(check, PAI.checked, draft)
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
            rec.set(check, PAI.iteration, e["iteration"])

        elif e["kind"] == "approval":
            tool = covering(t, "TOOL")
            check = rec.step(f"approval-{e['time']}", PAI.Check, node(tool), "approval",
                             e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver)
            rec.link(check, PAI.checked, tool and tool["id"] in results and results[tool["id"]][2] or None)
            rec.set(check, PAI.outcome, e["outcome"])
            rec.link(node(tool), PROV.wasInformedBy, check)

        elif e["kind"] == "effect":
            tool = covering(t, "TOOL", e["by"])
            effect = rec.entity(f"effect-{e['time']}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, node(tool))
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))
    return rec
