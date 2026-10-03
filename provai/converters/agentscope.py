import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og


def short(span_id):
    return span_id[2:] if span_id and span_id.startswith("0x") else span_id


def texts(msgs):
    return " ".join(p.get("content", "") for m in msgs for p in m.get("parts", []) if p.get("type") == "text").strip()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit(f"{run_dir}: no agents event")
    defs = agents["definitions"]
    spans = read(run_dir / "spans.jsonl")
    for s in spans:
        s["id"], s["parent"] = short(s["id"]), short(s["parent"])
    by_id = {s["id"]: s for s in spans}
    rec = Record(f"agentscope {head['versions']['agentscope']} with its built-in OpenTelemetry tracing",
                 head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)
    leader = rec.agent(defs["leader"])
    rec.link(leader, PROV.actedOnBehalfOf, user)

    events = [e for e in ev if e["kind"] == "event"]
    replies = {e["reply_id"]: e["name"] for e in events if e["event_type"] == "REPLY_START"}
    reply_order = [(seconds(e["time"]), e["reply_id"], e["name"]) for e in events if e["event_type"] == "REPLY_START"]
    model_calls = {e["span_id"]: e for e in ev if e["kind"] == "model_call"}
    tool_starts = {e["tool_call_id"]: e for e in events if e["event_type"] == "TOOL_CALL_START"}
    tool_ends = {e["tool_call_id"]: e for e in events if e["event_type"] == "TOOL_RESULT_END"}
    held = [(t, e["reply_id"]) for e in events if e["event_type"] == "REQUIRE_USER_CONFIRM" for t in e["tool_calls"]]
    settings = defs.get("settings") or {}

    steps, outputs, model_steps, tool_steps = {}, [], [], defaultdict(list)
    invocations, failed, previous_reply, first_of_reply = Counter(), {}, {}, {}
    leftover = []

    def agent_of(s):
        while s:
            if og.op(s) == "invoke_agent":
                return s["attrs"].get("gen_ai.agent.name")
            s = by_id.get(s["parent"])

    for s in spans:
        a, o, sid = s["attrs"], og.op(s), s["id"]
        parent = steps.get(s["parent"])
        if o is None:
            step = rec.step(sid, PAI.Step, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
            if a.get("session.id"):
                rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, a["session.id"], a["session.id"]))
        elif o == "invoke_agent":
            reply = a.get("agentscope.agent.reply_id")
            if reply not in replies:
                leftover.append(s["name"])
                continue
            name = a["gen_ai.agent.name"]
            first = reply not in previous_reply
            if first:
                invocations[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            # a reply that resumes after its members answered or a person confirmed is a new span of the same reply
            rec.link(step, PROV.wasInformedBy, previous_reply.get(reply))
            previous_reply[reply] = step
            if first:
                first_of_reply[reply] = step
            if name == defs["leader"] and first:
                rec.g.add((step, RDF.type, PAI.Routing))
                for m in defs["members"]:
                    rec.link(step, PAI.candidate, rec.agent(m))
        elif o == "chat":
            # the application logs a model call only when it succeeded
            if sid not in model_calls and s["status"] != "failed":
                leftover.append(s["name"])
                continue
            step = rec.step(sid, PAI.ModelCall, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(s)))
            inp, out = og.model_call(rec, step, a)
            rec.set(step, PAI.temperature, settings.get("temperature"))
            rec.set(step, PAI.maxTokens, settings.get("max_tokens"))
            model_steps.append((step, inp, out))
            if out:
                outputs.append((s["end"], agent_of(s), rec.iri("entity", f"{sid}-output"), texts(out), out))
        elif o == "execute_tool":
            cid = a.get("gen_ai.tool.call.id")
            if cid not in tool_starts:
                leftover.append(s["name"])
                continue
            step = rec.step(sid, PAI.ToolCall, parent, a.get("gen_ai.tool.name"), s["start"], s["end"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(s)))
            og.tool_call(rec, step, a)
            # the span ends OK for a failed call; the native result state says it failed
            state = (tool_ends.get(cid) or {}).get("state")
            rec.set(step, PAI.status, "failed" if state == "error" else "completed")
            tool_steps[cid].append(step)
            key = (agent_of(s), a.get("gen_ai.tool.name"), same(a.get("gen_ai.tool.call.arguments")))
            attempt = 1
            if key in failed:
                previous, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if state == "error":
                failed[key] = (step, attempt)
                rec.set(step, PAI.errorType, "tool error")
        else:
            step = rec.step(sid, PAI.Step, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
        steps[sid] = step

    # TeamAssign: which member each assignment went to is in the leader's model output
    assigned = {}
    for _, agent, out_iri, _, out in outputs:
        for m in out:
            for p in m.get("parts", []):
                if p.get("type") == "tool_call" and p.get("name") == "TeamAssign":
                    assigned[p["id"]] = (p.get("arguments") or {}).get("member"), out_iri
    member_replies = defaultdict(list)
    for t, reply, name in sorted(reply_order):
        member_replies[name].append((t, reply))
    for cid, (member, out_iri) in assigned.items():
        if cid not in tool_starts:
            continue
        start = seconds(tool_starts[cid]["time"])
        reply = next((r for t, r in member_replies.get(member, []) if t >= start), None)
        member_step = next((steps.get(s["id"]) for s in spans if reply is not None and og.op(s) == "invoke_agent"
                            and s["attrs"].get("agentscope.agent.reply_id") == reply), None)
        rec.link(member_step, PROV.used, out_iri)
        selected = rec.agent(member)
        rec.link(selected, PROV.actedOnBehalfOf, leader)
        # the routing is the leader's reply that made this assignment
        tool_span = next((s for s in spans if s["attrs"].get("gen_ai.tool.call.id") == cid), None)
        while tool_span is not None and og.op(tool_span) != "invoke_agent":
            tool_span = by_id.get(tool_span["parent"])
        routing = first_of_reply.get((tool_span or {}).get("attrs", {}).get("agentscope.agent.reply_id"))
        rec.link(routing, PAI.selected, selected)
        done = seconds(tool_ends[cid]["time"]) if cid in tool_ends else float("inf")
        # the assignment returns the member's final answer as its result
        for st in tool_steps.get(cid, []):
            result = rec.iri("entity", f"{rec.local(st)}-result")
            value = str(rec.g.value(result, PROV.value))
            # an assignment whose result the leader gives as its own later answer hands the work on
            if any(agent == defs["leader"] and text and seconds(t) >= start and text in value
                   for t, agent, _, text, _ in outputs):
                rec.g.add((st, RDF.type, PAI.Handoff))
                rec.link(st, PAI.fromAgent, leader)
                rec.link(st, PAI.toAgent, selected)
            answer = latest([(seconds(t), o) for t, agent, o, text, _ in outputs if agent == member and text
                             and text in value], done)
            if answer is not None:
                rec.derived(result, answer, "inferred")

    # the held call never ran under its own span; it is recorded from the confirmation request
    held_args, held_steps = {}, {}
    for t, reply in held:
        step = rec.step(f"held-{t['id']}", PAI.ToolCall, None, t["name"], None, None, "interrupted")
        rec.link(step, PAI.usedTool, rec.tool(t["name"]))
        rec.link(step, PROV.wasAssociatedWith, rec.agent(replies.get(reply)))
        args = rec.entity(f"held-{t['id']}-arguments", PROV.Entity, f"{t['name']} arguments", json.dumps(t["input"]))
        rec.used(step, args)
        held_args[t["id"]] = args
        held_steps[t["id"]] = step
        tool_steps[t["id"]].insert(0, step)

    og.link_tool_calls(rec, model_steps, tool_steps)

    # an earlier answer that appears word for word in a later model input
    for _, _, out_iri, text, _ in outputs:
        if len(text) < 16:
            continue
        for step, inp, _ in model_steps:
            sid = rec.local(step)
            entity = rec.iri("entity", f"{sid}-input")
            if out_iri != rec.iri("entity", f"{sid}-output") and inp and text in json.dumps(inp, ensure_ascii=False) \
                    and (entity, None, None) in rec.g:
                rec.derived(entity, out_iri, "inferred")

    for e in ev:
        if e["kind"] == "memory_write":
            access = rec.step(f"{e['time']}-memory-write", PAI.MemoryAccess, steps.get(e["span_id"]), "memory write",
                              e["time"], e["time"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_write(e["memory"], e["key"], e["value"], access)
            source = latest([(seconds(t), o) for t, agent, o, text, _ in outputs if agent == e["by"]
                             and text == (e["value"] or "").strip()], seconds(e["time"]))
            if source is not None:
                rec.derived(record, source, "inferred")
        elif e["kind"] == "memory_read":
            access = rec.step(f"{e['time']}-memory-read", PAI.MemoryAccess, steps.get(e["span_id"]), "memory read",
                              e["time"], e["time"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_read(e["memory"], e["key"], access)
            result = rec.iri("entity", f"{e['span_id']}-result")
            value = rec.g.value(record, PROV.value) if record is not None else None
            if value is not None and str(value) in str(rec.g.value(result, PROV.value)):
                rec.derived(result, record, "inferred")
        elif e["kind"] == "check":
            check = rec.step(f"{e['time']}-check", PAI.Check, None, "check", e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            draft = latest([(seconds(t), o) for t, agent, o, text, _ in outputs
                            if text and text == (e.get("draft") or "").strip()], seconds(e["time"]))
            rec.link(check, PAI.checked, draft)
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
            rec.set(check, PAI.iteration, e.get("iteration"))
        elif e["kind"] == "approval":
            check = rec.step(f"{e['time']}-approval", PAI.Check, None, "approval", e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver)
            rec.set(check, PAI.outcome, e["outcome"])
            for cid in e["tool_call_ids"]:
                rec.link(check, PAI.checked, held_args.get(cid))
                for step in (st for st in tool_steps.get(cid, []) if st != held_steps.get(cid)):
                    rec.link(step, PROV.wasInformedBy, check)
                    rec.used(step, held_args.get(cid))
        elif e["kind"] == "effect":
            effect = rec.entity(f"{e['time']}-effect", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, steps.get(e["span_id"]))
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native event: {leftover}")
    return rec
