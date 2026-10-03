import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

PLUMBING = ("workflow.build", "edge_group.process", "message.send")


def text_of(msgs):
    # the answer is the last text part; reasoning models can put their thinking in earlier parts
    texts = [p.get("content") or "" for m in msgs for p in m.get("parts", []) if p.get("type") == "text"]
    return texts[-1].strip() if texts else ""


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit(f"{run_dir}: no agents event")
    defs = agents["definitions"]
    spans = read(run_dir / "spans.jsonl")
    by_sid = {s["id"].removeprefix("0x"): s for s in spans}
    by_id = {s["id"]: s for s in spans}

    rec = Record(f"agent-framework-core {v['agent-framework-core']} with agent-framework-orchestrations "
                 f"{v['agent-framework-orchestrations']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    calls = defaultdict(list)
    for e in ev:
        if e["kind"] == "function_start" and not e["name"].startswith("handoff_to_"):
            calls[e["name"]].append(e)
    declared = defaultdict(list)
    for e in ev:
        if e["kind"] in ("routing", "memory_write", "memory_read", "check", "effect"):
            declared[e["span"]].append(e)
    handoffs = [e["data"] for e in ev if e["kind"] == "workflow_event" and e["type"] == "handoff_sent"]
    approvals = {e["call_id"]: e for e in ev if e["kind"] == "approval"}

    steps, invocations, runs_of = {}, Counter(), Counter()
    model_steps, tool_steps, answers, outputs = [], defaultdict(list), [], {}
    failed, held, executed = {}, {}, {}
    workflow_of, definition, routes = {}, {}, {}

    def up(span, test):
        while span:
            if test(span):
                return span
            span = by_id.get(span["parent"])

    def agent_name(span):
        a = up(span, lambda x: og.op(x) == "invoke_agent")
        return a and a["attrs"]["gen_ai.agent.name"]

    def contained(entity, value, start, invocation):
        # an answer of another agent turn that appears word for word in this input
        whole = json.dumps(og.messages(value), ensure_ascii=False) if value else ""
        for end, ent, answer, owner in answers:
            if owner != invocation and end <= seconds(start) and len(answer) >= 16 and \
                    json.dumps(answer, ensure_ascii=False)[1:-1] in whole:
                rec.derived(entity, ent, "inferred")

    for s in spans:
        a, sid, op, name = s["attrs"], s["id"], og.op(s), s["name"]
        up_step = steps.get(s["parent"])
        if name.startswith("workflow.build"):
            d = json.loads(a["workflow.definition"])
            definition[d["name"]] = d
            continue
        if name.startswith(PLUMBING):
            continue

        if name.startswith("workflow.run"):
            label = a["workflow.name"]
            step = rec.step(sid, PAI.Step, None, f"workflow {label}", s["start"], s["end"], s["status"], s["error"])
            workflow_of[sid] = label

        elif name.startswith("executor.process"):
            wf = workflow_of[up(s, lambda x: x["name"].startswith("workflow.run"))["id"]]
            ex = a["executor.id"]
            runs_of[(wf, ex)] += 1
            step = rec.step(sid, PAI.Step, up_step, f"{wf}: {ex}", s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, runs_of[(wf, ex)])
            agent = defs["workflows"][wf]["executors"].get(ex)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            for e in declared.get(sid.removeprefix("0x"), []):
                if e["kind"] == "routing":
                    group = next((g for g in definition[wf]["edge_groups"] if g.get("selection_func_name")
                                  and any(x["source_id"] == ex for x in g["edges"])), None)
                    if group is None:
                        raise SystemExit(f"routing from {ex} has no selection edge group")
                    rec.g.add((step, RDF.type, PAI.Routing))
                    for edge in group["edges"]:
                        rec.link(step, PAI.candidate, rec.agent(defs["workflows"][wf]["executors"][edge["target_id"]]))
                    rec.set(step, PAI.reason, e["reason"])
                    rec.link(rec.agent(agent), PROV.actedOnBehalfOf, user)
                    routes[sid] = [x["target_id"] for x in group["edges"]]

        elif op == "invoke_agent":
            agent = a["gen_ai.agent.name"]
            invocations[agent] += 1
            step = rec.step(sid, PAI.AgentInvocation, up_step, agent, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, invocations[agent])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))

        elif op == "chat":
            step = rec.step(sid, PAI.ModelCall, up_step, "chat", s["start"], s["end"], s["status"], s["error"])
            agent = agent_name(s)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            og.model_call(rec, step, a)
            # the sampling settings are on the agent span, the call's own span leaves them out
            turn = up(s, lambda x: og.op(x) == "invoke_agent") or {"id": None, "attrs": {}}
            rec.set(step, PAI.temperature, turn["attrs"].get("gen_ai.request.temperature"))
            rec.set(step, PAI.maxTokens, turn["attrs"].get("gen_ai.request.max_tokens"))
            inp, out = og.messages(a.get("gen_ai.input.messages")), og.messages(a.get("gen_ai.output.messages"))
            model_steps.append((step, inp, out))
            contained(rec.iri("entity", f"{sid}-input"), a.get("gen_ai.input.messages"), s["start"], turn["id"])
            said, asked = text_of(out), [p for m in out for p in m.get("parts", []) if p.get("type") == "tool_call"]
            if said and not asked:
                answers.append((seconds(s["end"]), rec.iri("entity", f"{sid}-output"), said, turn["id"]))
                outputs.setdefault(agent, []).append((seconds(s["end"]), rec.iri("entity", f"{sid}-output"), said))
            for p in asked:
                # a call held for a person's approval runs later, in another workflow run, or never
                if p["id"] in approvals:
                    hold = rec.step(f"{sid}-{p['id']}", PAI.ToolCall, up_step, p["name"], s["end"], s["end"],
                                    "interrupted")
                    rec.link(hold, PROV.wasAssociatedWith, rec.agent(agent))
                    og.tool_call(rec, hold, {"gen_ai.tool.name": p["name"]},
                                 json.dumps(p.get("arguments"), ensure_ascii=False))
                    held[p["id"]] = (hold, rec.iri("entity", f"{rec.local(hold)}-arguments"))
                    tool_steps[p["id"]].append(hold)

        elif op == "execute_tool":
            tool = a["gen_ai.tool.name"]
            if not calls.get(tool):
                raise SystemExit(f"tool span {tool} has no matching function event")
            native = calls[tool].pop(0)
            agent = agent_name(s)
            step = rec.step(sid, PAI.ToolCall, up_step, tool, s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            og.tool_call(rec, step, a)
            cid = a.get("gen_ai.tool.call.id")
            tool_steps[cid].append(step)
            if cid in held:
                rec.used(step, held[cid][1])
                executed[cid] = step
            key = (agent, tool, same(a.get("gen_ai.tool.call.arguments", native.get("arguments"))))
            attempt = 1
            if key in failed:
                previous, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if s["status"] == "failed":
                failed[key] = (step, attempt)
        else:
            raise SystemExit(f"span {name} has no mapping")
        steps[sid] = step

    # what each routing selected is the set of its candidates that processed a message in the same workflow run
    for s in spans:
        if s["id"] not in routes:
            continue
        run = up(s, lambda x: x["name"].startswith("workflow.run"))["id"]
        wf = workflow_of[run]
        router = rec.agent(defs["workflows"][wf]["executors"][s["attrs"]["executor.id"]])
        for x in spans:
            if x["name"].startswith("executor.process") and x["attrs"]["executor.id"] in routes[s["id"]] and \
                    x["parent"] == run:
                chosen = rec.agent(defs["workflows"][wf]["executors"][x["attrs"]["executor.id"]])
                rec.link(steps[s["id"]], PAI.selected, chosen)
                rec.link(chosen, PROV.actedOnBehalfOf, router)

    for h in handoffs:
        edge = next((s for s in spans if s["name"].startswith("edge_group.process")
                     and s["attrs"].get("message.source_id") == h["source"]
                     and s["attrs"].get("message.target_id") == h["target"]), None)
        if edge is None:
            raise SystemExit(f"handoff {h['source']} to {h['target']} has no edge span")
        run = up(edge, lambda x: x["name"].startswith("workflow.run"))["id"]
        executors = defs["workflows"][workflow_of[run]]["executors"]
        src, dst = rec.agent(executors[h["source"]]), rec.agent(executors[h["target"]])
        step = rec.step(edge["id"], PAI.Handoff, steps[run], "handoff", edge["start"], edge["end"], "completed")
        rec.link(step, PROV.wasAssociatedWith, src)
        rec.link(step, PAI.fromAgent, src)
        rec.link(step, PAI.toAgent, dst)
        rec.link(dst, PROV.actedOnBehalfOf, src)

    def source_of(agent, text, before):
        # a written value comes from the latest answer of its writer with the same text
        return latest([(x[0], x[1]) for x in outputs.get(agent, []) if x[2] == text], before)

    def judged(by, before):
        # a check judges the latest answer before it that another agent gave
        return latest([(x[0], x[1]) for agent, xs in outputs.items() if agent != by for x in xs], before)

    for e in sorted((e for es in declared.values() for e in es), key=lambda e: seconds(e["time"])):
        span = by_sid[e["span"]]
        step = steps[span["id"]]
        at = e["time"]
        when = seconds(at)
        if e["kind"] in ("memory_write", "memory_read"):
            access = rec.step(f"{span['id']}-{e['kind']}-{at}", PAI.MemoryAccess, step, e["kind"].replace("_", " "),
                              at, at, "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            if e["kind"] == "memory_write":
                record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                source = source_of(e["by"], (e.get("value") or "").strip(), when)
                if source is not None:
                    rec.derived(record, source, "inferred")
            else:
                record = rec.memory_read(e["memory"], e["key"], access)
                value = rec.g.value(record, PROV.value) if record is not None else None
                result = span["attrs"].get("gen_ai.tool.call.result")
                if value is not None and result is not None and result == str(value):
                    rec.derived(rec.iri("entity", f"{span['id']}-result"), record, "inferred")
        elif e["kind"] == "check":
            check = rec.step(f"{span['id']}-check-{at}", PAI.Check, step, "check", at, at, "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            rec.link(check, PAI.checked, judged(e["by"], when))
            rec.set(check, PAI.iteration, e.get("iteration"))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
        elif e["kind"] == "effect":
            effect = rec.entity(f"{span['id']}-effect-{at}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, step)
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    for cid, e in approvals.items():
        if cid not in held:
            raise SystemExit(f"approved call {cid} was never held")
        hold, arguments = held[cid]
        check = rec.step(f"approval-{cid}", PAI.Check, rec.g.value(hold, PAI.partOf), "approval", e["time"],
                         e["time"], "completed")
        rec.link(check, PROV.wasAssociatedWith, approver)
        rec.link(check, PAI.checked, arguments)
        rec.set(check, PAI.outcome, e["outcome"])
        rec.link(executed.get(cid), PROV.wasInformedBy, check)

    leftover = [e["name"] for es in calls.values() for e in es]
    if leftover:
        raise SystemExit(f"{len(leftover)} function events had no matching span: {leftover}")
    og.link_tool_calls(rec, model_steps, tool_steps)
    return rec
