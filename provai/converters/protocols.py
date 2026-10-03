import json
from collections import defaultdict

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, seconds
from . import a2a, mcp


def at(stamp):
    return seconds(stamp) if stamp else None


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    spans = read(run_dir / "spans.jsonl")
    by_id = {s["id"]: s for s in spans}
    logs = defaultdict(list)
    for line in open(run_dir / "logs.jsonl"):
        r = json.loads(line)
        logs[r["span_id"]].append(r)
    rec = Record(f"mcp {v['mcp']} and a2a-sdk {v['a2a-sdk']} with opentelemetry-instrumentation-openai-v2 "
                 f"{v['opentelemetry-instrumentation-openai-v2']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    kind = defaultdict(list)
    for e in ev:
        kind[e["kind"]].append(e)
    # the agent that routed the request is the A2A client that sends the messages
    if not kind["routing"]:
        raise SystemExit("the run has no routing event")
    routing = kind["routing"][0]
    router = routing["by"]
    turns, artifacts, messages = a2a.add(rec, kind["a2a"], kind["task_start"], router)

    def turn_of(span):
        found = turns.get(span["trace"].removeprefix("0x"))
        return found and found[0]

    # the router's choice among the agent cards it resolved, as the application reports it
    found_cards = a2a.cards(kind["a2a"])
    route = rec.step("routing", PAI.Routing, None, "routing", routing["time"], routing["time"], "completed")
    rec.link(route, PROV.wasAssociatedWith, rec.agent(router))
    for name in routing["candidates"]:
        if name in found_cards:
            rec.link(route, PAI.candidate, rec.agent(name))
    for name in routing["selected"]:
        rec.link(route, PAI.selected, rec.agent(name))
        rec.link(rec.agent(name), PROV.actedOnBehalfOf, rec.agent(router))
    rec.link(rec.agent(router), PROV.actedOnBehalfOf, user)

    usage = {e["response_id"]: e for e in kind["model_end"]}
    model_steps, answers, route_reason = [], [], None
    for s in spans:
        a = s["attrs"]
        if a.get("gen_ai.operation.name") != "chat":
            continue
        end = usage.get(a.get("gen_ai.response.id"))
        if end is None:
            if s["status"] == "completed":
                raise SystemExit(f"chat span {s['id']} has no model response in the events")
            end = {}
        s["matched"] = True
        agent = end.get("agent")
        parent = turn_of(s)
        sid = s["id"].removeprefix("0x")
        step = rec.step(sid, PAI.ModelCall, parent, "chat", s["start"], s["end"], s["status"],
                        s["error"])
        rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
        rec.link(step, PAI.usedModel, rec.model(a.get("gen_ai.response.model") or a.get("gen_ai.request.model"),
                                                 a.get("gen_ai.system")))
        rec.set(step, PAI.temperature, a.get("gen_ai.request.temperature"))
        rec.set(step, PAI.maxTokens, a.get("gen_ai.request.max_tokens"))
        rec.set(step, PAI.inputTokens, a.get("gen_ai.usage.input_tokens"))
        rec.set(step, PAI.outputTokens, a.get("gen_ai.usage.output_tokens"))
        reasons = a.get("gen_ai.response.finish_reasons")
        rec.set(step, PAI.finishReason, ",".join(reasons) if isinstance(reasons, list) else reasons)
        cost = (end.get("usage") or {}).get("cost")
        rec.set(step, PAI.cost, None if cost is None else float(cost))
        events = logs.get(s["id"], [])
        inputs = [r["body"] for r in events if r["event_name"] != "gen_ai.choice"]
        choice = next((r["body"]["message"] for r in events if r["event_name"] == "gen_ai.choice"), {})
        inp = rec.entity(f"{sid}-input", PAI.Message, "model input", json.dumps(inputs))
        rec.used(step, inp)
        out = rec.entity(f"{sid}-output", PAI.Message, "model output", json.dumps(choice))
        rec.generated(out, step)
        called = [c["id"] for c in choice.get("tool_calls") or []]
        read_ids = [b["id"] for b in inputs if "id" in b and "content" in b]
        model_steps.append((step, out, called, read_ids, s["start"], inp, inputs))
        if choice.get("content") and not called:
            answers.append((s["end"], out, choice["content"].strip(), parent, agent))
        if agent == router and at(s["start"]) <= at(routing["time"]):
            rec.link(step, PAI.partOf, route)
            rec.g.remove((step, PAI.partOf, rec.run))
            route_reason = choice.get("content") or route_reason
    rec.set(route, PAI.reason, route_reason)

    def parent_for(span):
        while span is not None:
            found = turn_of(span)
            if found is not None:
                return found
            span = by_id.get(span["parent"])

    tool_steps, records = mcp.add(rec, kind["mcp"], kind["elicitation"], by_id, parent_for, approver)
    for step, out, called, read_ids, _, _, _ in model_steps:
        for cid in called:
            for tool, _ in tool_steps.get(cid, []):
                rec.used(tool, out)
        for cid in read_ids:
            for _, res in tool_steps.get(cid, []):
                rec.used(step, res)

    # an artifact or a memory record whose value equals an agent's answer is derived from it, and a message
    # or a model input that quotes an earlier answer or artifact word for word is derived from it (knownBy inferred)
    for _, art, value, turn in artifacts:
        for _, out, answer, parent, _ in answers:
            if parent == turn and answer == value:
                rec.derived(art, out, "inferred")
    sources = [(at(t), e, x) for t, e, x, _ in artifacts] + [(at(t), e, x) for t, e, x, _, _ in answers]
    for recs in records.values():
        for record in recs:
            value = str(rec.g.value(record, PROV.value) or "")
            access = rec.g.value(record, PROV.wasGeneratedBy)
            writer = rec.g.value(access, PROV.wasAssociatedWith)
            written = at(str(rec.g.value(access, PROV.startedAtTime)))
            out = latest([(at(t), o) for t, o, answer, _, who in answers
                          if value and answer == value and rec.agent(who) == writer], written)
            if out is not None:
                rec.derived(record, out, "inferred")
    targets = [(at(t), e, x) for t, e, x in messages]
    targets += [(at(start), inp, whole) for _, _, _, _, start, inp, whole in model_steps]
    for start, entity, whole in targets:
        for when, src, value in sources:
            if when and when <= start and len(value) >= 16 and json.dumps(value)[1:-1] in json.dumps(whole):
                rec.derived(entity, src, "inferred")

    for e in kind["check"]:
        check = rec.step(f"check-{e['time']}", PAI.Check, None, "check", e["time"], e["time"], "completed")
        rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
        draft = next((art for _, art, value, _ in reversed(artifacts) if value == e["checked"]), None)
        rec.link(check, PAI.checked, draft)
        rec.set(check, PAI.iteration, e["iteration"])
        rec.set(check, PAI.outcome, e["outcome"])
        rec.set(check, PAI.reason, e["reason"])

    leftover = [s["name"] for s in spans if not s.get("matched") and
                (s["attrs"].get("gen_ai.operation.name") or s["attrs"].get("mcp.method.name") in
                 ("tools/call", "resources/read", "elicitation/create"))
                and not s["name"].startswith("MCP send")]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching protocol message: {leftover}")
    return rec
