import json
from collections import Counter, defaultdict, deque

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, seconds
from . import otel_genai as og


def text(value):
    return " ".join(p["content"] for m in og.messages(value) for p in m.get("parts", [])
                    if p.get("type") == "text" and isinstance(p.get("content"), str)).strip()


def short(span_id):
    return span_id[2:] if span_id and span_id.startswith("0x") else span_id


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    graph = next((e for e in ev if e["kind"] == "graph"), None)
    if graph is None:
        raise SystemExit("the run has no graph definition")
    defs = graph["definitions"]
    models = next((e["config"] for e in ev if e["kind"] == "models"), {})
    spans = read(run_dir / "spans.jsonl")
    for s in spans:
        s["id"], s["parent"] = short(s["id"]), short(s["parent"])
    by_id = {s["id"]: s for s in spans}

    native = defaultdict(list)
    for e in ev:
        if e.get("span_id"):
            if e["span_id"] not in by_id:
                raise SystemExit(f"native event {e['kind']} names span {e['span_id']}, which is not in spans.jsonl")
            native[e["span_id"]].append(e)
    queued = defaultdict(deque)
    for e in ev:
        if e["kind"] == "BeforeNodeCallEvent":
            queued[e["span_id"]].append(e["node_id"])
    held_ids = {i["id"] for e in ev if e["kind"] == "AfterInvocationEvent" for i in e.get("interrupts") or []}

    rec = Record(f"strands-agents {head['versions']['strands-agents']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)
    og.session(rec, spans)

    steps, node_of, ran = {}, {}, defaultdict(dict)
    invocations, cycles = Counter(), Counter()
    outputs, last_chat, last_search = [], {}, {}
    model_steps, tool_steps, held = [], defaultdict(list), {}
    records, writes, handoffs = {}, Counter(), {}
    memories = {}

    def up(sid, op):
        while sid in by_id:
            if og.op(by_id[sid]) == op:
                return sid
            sid = by_id[sid]["parent"]

    def agent_of(sid):
        a = up(sid, "invoke_agent")
        return a and by_id[a]["attrs"]["gen_ai.agent.name"]

    def contained(entity, value, start):
        # an earlier agent answer that appears word for word in this input
        whole = json.dumps(og.messages(value), ensure_ascii=False) if value else ""
        for end, ent, answer, _ in outputs:
            if end <= seconds(start) and len(answer) >= 16 and json.dumps(answer, ensure_ascii=False)[1:-1] in whole:
                rec.derived(entity, ent, "inferred")

    for s in spans:
        sid, a, op = s["id"], s["attrs"], og.op(s)
        parent = steps.get(s["parent"])
        if s["parent"] is None:
            steps[sid] = rec.run
            continue
        if op in ("invoke_graph", "invoke_swarm", "invoke_agent") and queued[s["parent"]]:
            node_of[sid] = queued[s["parent"]].popleft()
            ran[s["parent"]][node_of[sid]] = sid
        if op in ("invoke_graph", "invoke_swarm"):
            label = f"{op.split('_')[1]} {node_of.get(sid, '')}".strip()
            step = rec.step(sid, PAI.Step, parent, label, s["start"], s["end"], s["status"], s["error"])

        elif op == "invoke_agent":
            name = a["gen_ai.agent.name"]
            node = node_of.get(sid)
            if node is not None:
                members = defs["nodes"].get(node_of.get(s["parent"]), {})
                expected = members.get(node) if og.op(by_id[s["parent"]]) == "invoke_swarm" else defs["nodes"].get(node)
                if expected != name:
                    raise SystemExit(f"node {node} ran agent {name}, the graph says {expected}")
            invocations[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            if node is not None and og.op(by_id[s["parent"]]) == "invoke_graph":
                rec.set(step, PAI.branch, node)
            if a.get("gen_ai.input.messages"):
                inp = rec.entity(f"{sid}-input", PAI.Message, f"{name} input", a["gen_ai.input.messages"])
                rec.used(step, inp)
                contained(inp, a["gen_ai.input.messages"], s["start"])
            if a.get("gen_ai.output.messages"):
                out = rec.entity(f"{sid}-output", PAI.Message, f"{name} answer", a["gen_ai.output.messages"])
                rec.generated(out, step)
                outputs.append((seconds(s["end"]), out, text(a["gen_ai.output.messages"]), name))
            sent = handoffs.pop((s["parent"], name), None)
            if sent is not None:
                rec.used(step, sent)

        elif op == "execute_event_loop_cycle":
            owner = up(s["parent"], "invoke_agent")
            cycles[owner] += 1
            step = rec.step(sid, PAI.Step, parent, "event loop cycle", s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, cycles[owner])

        elif op == "chat":
            step = rec.step(sid, PAI.ModelCall, parent, "chat", s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(sid)))
            inp, out = og.model_call(rec, step, a)
            # the agent objects give the settings and finish reasons that the chat span leaves out
            params = (models.get(agent_of(sid)) or {}).get("params") or {}
            if (step, PAI.temperature, None) not in rec.g:
                rec.set(step, PAI.temperature, params.get("temperature"))
            if (step, PAI.maxTokens, None) not in rec.g:
                rec.set(step, PAI.maxTokens, params.get("max_tokens"))
            if (step, PAI.finishReason, None) not in rec.g:
                reasons = sorted({m.get("finish_reason") for m in out if m.get("finish_reason")})
                rec.set(step, PAI.finishReason, ",".join(reasons))
            if inp:
                contained(rec.iri("entity", f"{sid}-input"), a.get("gen_ai.input.messages"), s["start"])
            rec.used(step, last_search.get(s["parent"]))
            model_steps.append((step, inp, out))
            if out:
                last_chat[up(sid, "invoke_agent")] = (rec.iri("entity", f"{sid}-output"), text(a["gen_ai.output.messages"]))

        elif op == "execute_tool":
            name, cid = a["gen_ai.tool.name"], a.get("gen_ai.tool.call.id")
            agent = rec.agent(agent_of(sid))
            step = rec.step(sid, PAI.ToolCall, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, agent)
            og.tool_call(rec, step, a)
            args = rec.iri("entity", f"{sid}-arguments")
            tool_steps[cid].append(step)
            befores = [e for e in native[sid] if e["kind"] == "BeforeToolCallEvent"]
            afters = [e for e in native[sid] if e["kind"] == "AfterToolCallEvent"]
            # a retry asked for by a hook runs inside the same span; each failed attempt becomes its own call
            previous = None
            for n, (b, e) in enumerate(zip(befores, afters), 1):
                if not e["retry"]:
                    break
                attempt = rec.step(f"{sid}-attempt-{n}", PAI.ToolCall, parent, name, b["time"], e["time"], "failed",
                                   e["exception"])
                rec.link(attempt, PAI.usedTool, rec.tool(name))
                rec.link(attempt, PROV.wasAssociatedWith, agent)
                rec.used(attempt, args)
                if n > 1:
                    rec.set(attempt, PAI.attempt, n)
                rec.link(attempt, PAI.retryOf, previous)
                tool_steps[cid].append(attempt)
                previous = attempt
            if previous is not None:
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, len([e for e in afters if e["retry"]]) + 1)
            if not afters and cid and any(cid in i.split(":") for i in held_ids):
                rec.g.remove((step, PAI.status, None))
                rec.set(step, PAI.status, "interrupted")
                held[cid] = args
            for e in native[sid]:
                if e["kind"] == "approval":
                    check = rec.step(f"{sid}-approval", PAI.Check, step, "approval", e["time"], e["time"], "completed")
                    rec.link(check, PROV.wasAssociatedWith, approver)
                    rec.link(check, PAI.checked, held.get(e["tool_use_id"]))
                    rec.set(check, PAI.outcome, e["outcome"])
                    rec.link(step, PROV.wasInformedBy, check)
                    rec.used(step, held.get(e["tool_use_id"]))
            if name == "handoff_to_agent":
                target = json.loads(a["gen_ai.tool.call.arguments"])["agent_name"]
                rec.g.add((step, RDF.type, PAI.Handoff))
                rec.link(step, PAI.fromAgent, agent)
                rec.link(step, PAI.toAgent, rec.agent(target))
                rec.link(rec.agent(target), PROV.actedOnBehalfOf, agent)
                handoffs[(up(sid, "invoke_swarm"), target)] = args

        elif op == "memory.add":
            step = rec.step(sid, PAI.MemoryAccess, parent, "memory add", s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(sid)))
            for store in json.loads(a["memory.store.names"]):
                writes[store] += 1
                record = rec.entity(f"{store}-{writes[store]}", PAI.MemoryRecord, store, a.get("content"))
                rec.generated(record, step)
                memories.setdefault(store, rec.node("memory", PAI.Memory, store, store))
                rec.link(memories[store], PROV.hadMember, record)
                records[(store, a.get("content"))] = record
                owner = up(sid, "invoke_agent")
                answer = owner and by_id[owner]["attrs"].get("gen_ai.output.messages")
                if answer and text(answer) == (a.get("content") or "").strip():
                    rec.derived(record, rec.iri("entity", f"{owner}-output"), "inferred")

        elif op in ("memory.inject", "memory.search"):
            label = op.replace(".", " ")
            step = rec.step(sid, PAI.MemoryAccess, parent, label, s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(sid)))
            if op == "memory.search":
                found = rec.entity(f"{sid}-result", PROV.Entity, "memory search result", a.get("content"))
                rec.generated(found, step)
                for entry in json.loads(a.get("content") or "[]"):
                    record = records.get((entry.get("store_name"), entry.get("content")))
                    rec.used(step, record)
                    if record is not None:
                        rec.derived(found, record, "inferred")
                last_search[up(sid, "execute_event_loop_cycle")] = found

        elif s["name"] == "check":
            step = rec.step(sid, PAI.Check, parent, "check", s["start"], s["end"], "completed")
            rec.link(step, PROV.wasAssociatedWith, rec.agent(a["app.by"]))
            # the latest answer before the check by an agent other than the checker
            rec.link(step, PAI.checked, latest([(end, ent) for end, ent, _, who in outputs if who != a["app.by"]],
                                               seconds(s["start"])))
            rec.set(step, PAI.outcome, a["app.outcome"])
            rec.set(step, PAI.iteration, a.get("app.iteration"))
            rec.set(step, PAI.reason, a.get("app.reason"))

        elif s["name"] == "effect":
            effect = rec.entity(f"{sid}-effect", PAI.Effect, "effect", a["app.target"])
            rec.set(effect, PAI.target, a["app.target"])
            rec.generated(effect, parent)
            rec.link(effect, PROV.wasAttributedTo, rec.agent(a["app.by"]))
            continue

        else:
            raise SystemExit(f"span {s['name']} has no mapping")
        steps[sid] = step

    for invocation, (out, answer) in last_chat.items():
        if answer and answer == text(by_id[invocation]["attrs"].get("gen_ai.output.messages")):
            rec.derived(rec.iri("entity", f"{invocation}-output"), out, "inferred")

    for graph, nodes in ran.items():
        if og.op(by_id[graph]) != "invoke_graph":
            continue
        for src, dst in defs["edges"]:
            if src in nodes and dst in nodes:
                rec.used(steps[nodes[dst]], rec.iri("entity", f"{nodes[src]}-output"))
        for node, sid in nodes.items():
            targets = [dst for src, dst in defs["edges"] if src == node]
            if len(targets) < 2:
                continue
            routing, by = steps[sid], rec.agent(defs["nodes"][node])
            rec.g.add((routing, RDF.type, PAI.Routing))
            rec.set(routing, PAI.reason, text(by_id[sid]["attrs"].get("gen_ai.output.messages")))
            rec.link(by, PROV.actedOnBehalfOf, user)
            for dst in targets:
                if isinstance(defs["nodes"][dst], str):
                    rec.link(routing, PAI.candidate, rec.agent(defs["nodes"][dst]))
                    if dst in nodes:
                        rec.link(routing, PAI.selected, rec.agent(defs["nodes"][dst]))
                        rec.link(rec.agent(defs["nodes"][dst]), PROV.actedOnBehalfOf, by)

    og.link_tool_calls(rec, model_steps, tool_steps)
    return rec
