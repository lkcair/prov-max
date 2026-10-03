import json
from collections import defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og


def plain(value):
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        pass
    return str(value if value is not None else "").strip()


def around(span, at):
    # the app's clock has millisecond resolution, so a time inside a span can read up to 1 ms off
    ms = 0.001
    return seconds(span["start"]) - ms <= seconds(at) <= seconds(span["end"]) + ms


def answer(out):
    return " ".join(p.get("content", "") for m in out for p in m.get("parts", []) if p.get("type") == "text").strip()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit("the run has no agent definitions")
    defs = agents["definitions"]
    spans = read(run_dir / "spans.jsonl")
    by_id = {s["id"]: s for s in spans}
    rec = Record(f"vercel-ai {v['ai']} with @ai-sdk/otel {v['@ai-sdk/otel']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    # native events join the spans by response id and tool call id
    model_end = {e["response_id"]: e for e in ev if e["kind"] == "model_end" and e.get("response_id")}
    tool_end = defaultdict(list)
    for e in ev:
        if e["kind"] == "tool_end":
            tool_end[e["tool_call_id"]].append(e)

    def agent_of(s):
        while s:
            if og.op(s) == "invoke_agent":
                return s["attrs"].get("gen_ai.agent.name")
            s = by_id.get(s["parent"])

    steps, model_steps, tool_steps = {}, [], defaultdict(list)
    turns, answers, failed = defaultdict(int), [], {}
    for s in spans:
        o, a = og.op(s), s["attrs"]
        parent = steps.get(s["parent"])
        name = agent_of(s)
        if o == "invoke_agent":
            turns[name] += 1
            step = rec.step(s["id"], PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, turns[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            tools = defs.get(name, {}).get("agent_tools", [])
            if tools:
                rec.g.add((step, RDF.type, PAI.Routing))
                for t in tools:
                    rec.link(step, PAI.candidate, rec.agent(t))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, user)
        elif o == "agent_step":
            step = rec.step(s["id"], PAI.Step, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
        elif o in og.MODEL_OPS:
            step = rec.step(s["id"], PAI.ModelCall, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            inp, out = og.model_call(rec, step, a)
            usage = (model_end.get(a.get("gen_ai.response.id")) or {}).get("openrouter") or {}
            cost = (usage.get("usage") or {}).get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            model_steps.append((step, inp, out))
            text = answer(out)
            # a turn that requested tools is not an answer
            if text and not og.tool_call_ids(out, "tool_call"):
                answers.append((name, seconds(s["end"]), rec.iri("entity", f"{s['id']}-output"), text))
        elif o == "execute_tool":
            tool = a.get("gen_ai.tool.name")
            step = rec.step(s["id"], PAI.ToolCall, parent, tool, s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            og.tool_call(rec, step, a)
            cid = a.get("gen_ai.tool.call.id")
            tool_steps[cid].append(step)
            ended = (tool_end.get(cid) or [{}])[-1]
            if ended.get("type") == "tool-error":
                rec.g.remove((step, PAI.status, None))
                rec.g.remove((step, PAI.errorType, None))
                rec.set(step, PAI.status, "failed")
                rec.set(step, PAI.errorType, (ended.get("error") or {}).get("name"))
            if tool in defs.get(name, {}).get("agent_tools", []):
                up = s
                while up and og.op(up) != "invoke_agent":
                    up = by_id.get(up["parent"])
                if up:
                    rec.link(steps.get(up["id"]), PAI.selected, rec.agent(tool))
                    rec.link(rec.agent(tool), PROV.actedOnBehalfOf, rec.agent(name))
            key = (name, tool, same(a.get("gen_ai.tool.call.arguments")))
            failed_now = ended.get("type") == "tool-error"
            attempt = 1
            if key in failed:
                before, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, before)
                rec.set(step, PAI.attempt, attempt)
            if failed_now:
                failed[key] = (step, attempt)
        else:
            step = rec.step(s["id"], PAI.Step, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
        steps[s["id"]] = step
    og.link_tool_calls(rec, model_steps, tool_steps)

    def inside(sid, top):
        while sid:
            if sid == top:
                return True
            sid = by_id[sid]["parent"] if sid in by_id else None

    # a subagent's final answer is the result of the tool that ran it (matched by content, so inferred)
    for s in spans:
        a = s["attrs"]
        if og.op(s) == "execute_tool" and a.get("gen_ai.tool.name") in defs.get(agent_of(s), {}).get("agent_tools", []):
            result = plain(a.get("gen_ai.tool.call.result"))
            for c in spans:
                if og.op(c) in og.MODEL_OPS and inside(c["id"], s["id"]):
                    text = answer(og.messages(c["attrs"].get("gen_ai.output.messages")))
                    if text and text == result:
                        rec.derived(rec.iri("entity", f"{s['id']}-result"), rec.iri("entity", f"{c['id']}-output"),
                                    "inferred")
    # a model input that holds an earlier answer word for word was built from it (inferred)
    for s in spans:
        if og.op(s) in og.MODEL_OPS:
            msgs = og.messages(s["attrs"].get("gen_ai.input.messages"))
            text = " ".join(str(p.get("content", "")) for m in msgs for p in m.get("parts", []))
            for _, end, out, said in answers:
                if len(said) >= 16 and end <= seconds(s["start"]) and said in text:
                    rec.derived(rec.iri("entity", f"{s['id']}-input"), out, "inferred")
    unmatched = [s["name"] for s in spans if og.op(s) in og.MODEL_OPS and s["attrs"].get("gen_ai.response.id")
                 and s["attrs"]["gen_ai.response.id"] not in model_end]
    unmatched += [s["name"] for s in spans if og.op(s) == "execute_tool" and s["attrs"].get("gen_ai.tool.call.id")
                  and s["attrs"]["gen_ai.tool.call.id"] not in tool_end]
    if unmatched:
        raise SystemExit(f"{len(unmatched)} spans had no matching native event: {unmatched}")

    def tool_around(at):
        # the innermost tool call whose span holds the moment
        made = [s for s in spans if og.op(s) == "execute_tool" and around(s, at)]
        return steps[max(made, key=lambda s: seconds(s["start"]))["id"]] if made else None

    # facts the application declares: handoff, memory, checks, approval, effect
    for e in ev:
        kind, at = e["kind"], e["time"]
        if kind == "handoff":
            h = rec.step(f"handoff-{at}", PAI.Handoff, None, "handoff", at, at, "completed")
            src, dst = rec.agent(e["from"]), rec.agent(e["to"])
            rec.link(h, PROV.wasAssociatedWith, src)
            rec.link(h, PAI.fromAgent, src)
            rec.link(h, PAI.toAgent, dst)
            rec.link(dst, PROV.actedOnBehalfOf, src)
        elif kind == "memory_write":
            access = rec.step(f"write-{at}", PAI.MemoryAccess, tool_around(at), "memory write", at, at, "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_write(e["memory"], e["record"], e.get("value"), access)
            value = (e.get("value") or "").strip()
            found = latest([(end, out) for who, end, out, said in answers if who == e["by"] and said == value], seconds(at))
            if found is not None:
                rec.derived(record, found, "inferred")
        elif kind == "memory_read":
            access = rec.step(f"read-{at}", PAI.MemoryAccess, tool_around(at), "memory read", at, at, "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_read(e["memory"], e["record"], access)
            # the tool call around the read returns the record's content
            value = rec.g.value(record, PROV.value) if record is not None else None
            for s in spans:
                a = s["attrs"]
                if og.op(s) == "execute_tool" and around(s, at):
                    if value is not None and plain(a.get("gen_ai.tool.call.result")) == str(value):
                        rec.derived(rec.iri("entity", f"{s['id']}-result"), record, "inferred")
        elif kind in ("check", "approval"):
            check = rec.step(f"{kind}-{at}", PAI.Check, None, kind, at, at, "completed")
            rec.link(check, PROV.wasAssociatedWith, approver if kind == "approval" else rec.agent(e["by"]))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
            rec.set(check, PAI.iteration, e.get("iteration"))
            if kind == "approval":
                for step in tool_steps.get(e.get("tool_call_id"), []):
                    rec.link(check, PAI.checked, rec.iri("entity", f"{rec.local(step)}-arguments"))
                    rec.link(step, PROV.wasInformedBy, check)
            else:
                # a check judges the latest answer before it that another agent gave
                rec.link(check, PAI.checked, latest([(end, out) for who, end, out, _ in answers if who != e["by"]],
                                                    seconds(at)))
        elif kind == "effect":
            # the effect is declared inside the tool call that made it, the innermost span around its time
            effect = rec.entity(f"effect-{at}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, tool_around(at))
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))
    return rec
