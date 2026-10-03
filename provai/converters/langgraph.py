import json
from collections import Counter, defaultdict

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

PLUMBING = {"model", "tools", "RunnableCallable"}
CLASSES = {"routing": PAI.Routing, "handoff": PAI.Handoff}
# events an agent declares from inside its own node; an approval comes from a person
BY_AGENT = ("routing", "memory_write", "memory_read", "check", "effect")


def text_of(msgs):
    return " ".join(p.get("content", "") for m in msgs for p in m.get("parts", []) if p.get("type") == "text").strip()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    spans = read(run_dir / "spans.jsonl")
    head = ev[0]
    v = head["versions"]
    rec = Record(f"langgraph {v['langgraph']} with opentelemetry-instrumentation-genai-langchain "
                 f"{v['opentelemetry-instrumentation-genai-langchain']}", head["run_id"], True)
    og.session(rec, spans)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    starts = {e["run_id"]: e for e in ev if e["kind"] in ("chain_start", "model_start", "tool_start")}
    ends = {e["run_id"]: e for e in ev if e["kind"] in ("chain_end", "model_end", "tool_end")}
    errors = {e["run_id"]: e for e in ev if e["kind"] in ("chain_error", "tool_error")}
    steps, agent_of, runs_of_node, turns = {}, {}, Counter(), Counter()
    state, state_entity = Counter(), {}
    model_steps, tool_steps, node_output, answers = [], {}, {}, []
    node_class = {e["run_id"]: CLASSES[e["kind"]] for e in ev if e["kind"] in CLASSES}
    node_agent = {e["run_id"]: e["by"] for e in ev if e["kind"] in BY_AGENT and e.get("by")}

    def inside(run_id, node_run):
        while run_id in starts:
            if run_id == node_run:
                return True
            run_id = starts[run_id].get("parent_run_id")
        return False

    def kept(run_id):
        while run_id in starts and run_id not in steps:
            run_id = starts[run_id].get("parent_run_id")
        return steps.get(run_id)

    def finish(step, run_id):
        e = errors.get(run_id)
        if e:
            rec.set(step, PAI.status, "interrupted" if e["error"] == "GraphInterrupt" else "failed")
            if e["error"] != "GraphInterrupt":
                rec.set(step, PAI.errorType, e["error"])
        else:
            rec.set(step, PAI.status, "completed")

    def node_step(e):
        if e.get("run_id") not in steps:
            raise SystemExit(f"the {e['kind']} event names no recorded node run")
        return steps[e["run_id"]]

    def agent_for(run_id):
        while run_id in starts:
            if run_id in agent_of:
                return agent_of[run_id]
            run_id = starts[run_id].get("parent_run_id")

    span_by_id = {s["id"]: s for s in spans}

    def span_agent(s):
        s = span_by_id.get(s["parent"])
        while s:
            if og.op(s) == "invoke_agent":
                return s["attrs"].get("gen_ai.agent.name")
            s = span_by_id.get(s["parent"])

    chats = iter([s for s in spans if og.op(s) in og.MODEL_OPS])
    tool_spans = defaultdict(list)
    for s in spans:
        if og.op(s) == "execute_tool":
            tool_spans[s["attrs"].get("gen_ai.tool.name")].append(s)
    tool_spans = {k: iter(v) for k, v in tool_spans.items()}
    failed_node = {}

    for e in ev:
        rid, kind = e.get("run_id"), e["kind"]
        if kind == "chain_start" and e.get("name") not in PLUMBING:
            name, node = e["name"], e.get("langgraph_node")
            parent = kept(e.get("parent_run_id"))
            end = (ends.get(rid) or errors.get(rid) or {}).get("time")
            if node is None:
                step = rec.step(rid, PAI.Step, None, "graph invocation", e["time"], end)
                for key in e.get("reads") or []:
                    state[key] += 1
                    ent = rec.entity(f"state-{key}-{state[key]}", PROV.Entity, f"{key} v{state[key]}")
                    rec.link(ent, PROV.wasAttributedTo, user)
                    state_entity[key] = ent
            elif name == node:
                runs_of_node[node] += 1
                step = rec.step(rid, node_class.get(rid, PAI.Step), parent, node, e["time"], end)
                rec.set(step, PAI.sequence, e.get("langgraph_step"))
                rec.set(step, PAI.iteration, runs_of_node[node])
                if rid in node_agent:
                    agent_of[rid] = rec.agent(node_agent[rid])
                    rec.link(step, PROV.wasAssociatedWith, agent_of[rid])
                attempt = 1
                prior = failed_node.get((node, e.get("langgraph_step")))
                if prior is not None:
                    attempt = prior[1] + 1
                    rec.link(step, PAI.retryOf, prior[0])
                    rec.set(step, PAI.attempt, attempt)
                for key in e.get("reads") or []:
                    rec.used(step, state_entity.get(key))
            else:
                step = rec.step(rid, PAI.AgentInvocation, parent, name, e["time"], end)
                turns[name] += 1
                rec.set(step, PAI.iteration, turns[name])
                agent_of[rid] = rec.agent(name)
                rec.link(step, PROV.wasAssociatedWith, agent_of[rid])
            steps[rid] = step
            finish(step, rid)
            if node and name == node and rid in errors and errors[rid]["error"] != "GraphInterrupt":
                failed_node[(node, e.get("langgraph_step"))] = (step, attempt)

        elif kind == "chain_end" and rid in steps and starts[rid].get("name") == starts[rid].get("langgraph_node"):
            for key in e.get("writes") or []:
                state[key] += 1
                ent = rec.entity(f"state-{key}-{state[key]}", PROV.Entity, f"{key} v{state[key]}")
                rec.generated(ent, steps[rid])
                if rid in node_output:
                    rec.derived(ent, node_output[rid], "inferred")
                state_entity[key] = ent

        elif kind == "model_start":
            span = next(chats, None)
            if span is None:
                raise SystemExit(f"no model span for the model call {rid}")
            parent = kept(e.get("parent_run_id"))
            end = (ends.get(rid) or {}).get("time")
            agent = agent_for(e.get("parent_run_id"))
            # chat spans are paired with model calls in order, which parallel agents could interleave
            if span_agent(span) and rec.agent(span_agent(span)) != agent:
                raise SystemExit(f"the model call {rid} and its chat span belong to different agents")
            step = rec.step(rid, PAI.ModelCall, parent, span["name"], e["time"], end, span["status"], span["error"])
            rec.link(step, PROV.wasAssociatedWith, agent)
            inp, out = og.model_call(rec, step, span["attrs"])
            cost = ((ends.get(rid) or {}).get("usage") or {}).get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            model_steps.append((step, inp, out))
            if out:
                msg = rec.iri("entity", f"{rid}-output")
                node_run = e.get("parent_run_id")
                while node_run in starts:
                    if node_run in steps and starts[node_run].get("name") == starts[node_run].get("langgraph_node"):
                        node_output[node_run] = msg
                    node_run = starts[node_run].get("parent_run_id")
                answers.append((seconds(end) if end else None, agent, text_of(out), msg))

        elif kind == "tool_start":
            span = next(tool_spans[e["name"]], None) if e["name"] in tool_spans else None
            if span is None:
                raise SystemExit(f"no tool span for the {e['name']} call {rid}")
            parent = kept(e.get("parent_run_id"))
            end = (ends.get(rid) or errors.get(rid) or {}).get("time")
            step = rec.step(rid, PAI.ToolCall, parent, e["name"], e["time"], end, span["status"], span["error"])
            rec.link(step, PROV.wasAssociatedWith, agent_for(e.get("parent_run_id")))
            og.tool_call(rec, step, span["attrs"], e.get("input"), (ends.get(rid) or {}).get("output"))
            cid = span["attrs"].get("gen_ai.tool.call.id")
            if cid:
                tool_steps[cid] = step
            steps[rid] = step

        elif kind == "routing":
            step = node_step(e)
            by = rec.agent(e["by"])
            for name in e["candidates"]:
                rec.link(step, PAI.candidate, rec.agent(name))
            for name in e["selected"]:
                rec.link(step, PAI.selected, rec.agent(name))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, by)
            rec.link(by, PROV.actedOnBehalfOf, user)
            rec.set(step, PAI.reason, e["reason"])

        elif kind == "handoff":
            step = node_step(e)
            rec.link(step, PAI.fromAgent, rec.agent(e["from"]))
            rec.link(step, PAI.toAgent, rec.agent(e["to"]))
            rec.link(rec.agent(e["to"]), PROV.actedOnBehalfOf, rec.agent(e["from"]))

        elif kind in ("memory_write", "memory_read"):
            access = rec.step(f"{rid}-{kind}-{e['time']}", PAI.MemoryAccess, node_step(e), kind.replace("_", " "),
                              e["time"], e["time"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            if kind == "memory_write":
                record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                # the writer's latest answer with the written text
                source = latest([(t, msg) for t, who, text, msg in answers
                                 if who == rec.agent(e["by"]) and text == (e.get("value") or "").strip()],
                                seconds(e["time"]))
                rec.derived(record, source, "inferred")
            else:
                rec.memory_read(e["memory"], e["key"], access)

        elif kind in ("check", "approval"):
            check = rec.step(f"{rid}-{kind}-{e['time']}", PAI.Check, node_step(e), kind, e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver if kind == "approval" else rec.agent(e["by"]))
            rec.link(check, PAI.checked, state_entity.get(e["checked"]))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
            rec.set(check, PAI.iteration, e.get("iteration"))

        elif kind == "effect":
            effect = rec.entity(f"{rid}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            # the tool call that the node declaring the effect ran last
            sent = [s for r, s in steps.items() if starts[r]["kind"] == "tool_start" and inside(r, rid)]
            rec.generated(effect, sent[-1] if sent else node_step(e))
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    og.link_tool_calls(rec, model_steps, tool_steps)
    failed = {}
    for rid, step in steps.items():
        e = starts[rid]
        if e["kind"] != "tool_start":
            continue
        key, attempt = (agent_for(e.get("parent_run_id")), e["name"], same(e.get("input"))), 1
        if key in failed:
            prior, attempt = failed.pop(key)
            attempt += 1
            rec.link(step, PAI.retryOf, prior)
            rec.set(step, PAI.attempt, attempt)
        if rid in errors:
            failed[key] = (step, attempt)
    leftover = list(chats) + [s for it in tool_spans.values() for s in it]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native event: {[s['name'] for s in leftover]}")
    return rec
