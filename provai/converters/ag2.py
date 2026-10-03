import json
from collections import defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

DELEGATION = "task_"


def called_ids(attrs):
    # AG2 writes OpenAI-style messages: the model's tool calls and the results it read carry the call id
    out = [c.get("id") for m in og.messages(attrs.get("gen_ai.output.messages")) for c in m.get("tool_calls") or []]
    read_ids = [m.get("tool_call_id") for m in og.messages(attrs.get("gen_ai.input.messages")) if m.get("role") == "tool"]
    return [i for i in out if i], [i for i in read_ids if i]


def answer_text(attrs):
    return " ".join(m.get("content") or "" for m in og.messages(attrs.get("gen_ai.output.messages"))
                    if m.get("role") == "assistant" and not m.get("tool_calls")).strip()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit(f"{run_dir}: no agents event")
    defs = agents["definitions"]
    models = next((e["config"] for e in ev if e["kind"] == "models"), {})
    spans = read(run_dir / "spans.jsonl")
    by_id = {s["id"]: s for s in spans}
    rec = Record(f"ag2 {v['ag2']} with its built-in OpenTelemetry middleware", head["run_id"], True)
    og.session(rec, spans)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    def op(s):
        return s["attrs"].get("gen_ai.operation.name")

    def agent_span(s):
        while s:
            if op(s) == "invoke_agent":
                return s
            s = by_id.get(s["parent"])

    def agent_of(s):
        a = agent_span(s)
        return a and a["attrs"].get("gen_ai.agent.name")

    candidates = {d["name"]: [t[len(DELEGATION):] for t in d["tools"] if t.startswith(DELEGATION)]
                  for d in defs.values()}
    calls = {e["call_id"]: e for e in ev if e["kind"] == "ToolCallEvent"}
    errors = {e["call_id"]: e for e in ev if e["kind"] == "ToolErrorEvent"}
    responses = defaultdict(list)
    for e in ev:
        if e["kind"] == "ModelResponse":
            responses[e["agent"]].append(e)
    used_responses = defaultdict(int)

    steps, tool_steps, model_steps, results, answers = {}, defaultdict(list), [], {}, []
    turns = defaultdict(int)
    delegations, failed = [], {}
    for s in spans:
        sid, a = s["id"], s["attrs"]
        parent = steps.get(s["parent"])
        o = op(s)
        if o == "invoke_agent":
            name = a.get("gen_ai.agent.name")
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            turns[name] += 1
            rec.set(step, PAI.iteration, turns[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            if candidates.get(name):
                rec.g.add((step, RDF.type, PAI.Routing))
                for c in candidates[name]:
                    rec.link(step, PAI.candidate, rec.agent(c))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, user)
        elif o == "chat":
            name = agent_of(s)
            step = rec.step(sid, PAI.ModelCall, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            og.model_call(rec, step, a)
            settings = models.get(name) or {}
            rec.set(step, PAI.temperature, settings.get("temperature"))
            rec.set(step, PAI.maxTokens, settings.get("max_tokens"))
            # AG2 logs a model response only for a call that succeeded
            native = None
            if s["status"] == "completed":
                native = responses[name][used_responses[name]] if used_responses[name] < len(responses[name]) else None
                used_responses[name] += 1
                if native is None:
                    raise SystemExit(f"chat span {sid} of {name} has no native model response")
            out_ids, read_ids = called_ids(a)
            output = rec.iri("entity", f"{sid}-output")
            model_steps.append((step, output, out_ids, read_ids))
            text = answer_text(a) or ((native or {}).get("message") or "").strip()
            if text and not out_ids:
                answers.append((s, output, text))
        elif o == "execute_tool":
            name = a.get("gen_ai.tool.name")
            cid = a.get("gen_ai.tool.call.id")
            if cid not in calls:
                raise SystemExit(f"tool span {sid} ({name}) has no native tool call")
            step = rec.step(sid, PAI.ToolCall, parent, name, s["start"], s["end"], s["status"],
                            errors[cid]["error"] if cid in errors else s["error"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of(s)))
            args = calls[cid].get("arguments")
            result = None if cid in errors else (a.get("gen_ai.tool.call.result") or "")
            og.tool_call(rec, step, {"gen_ai.tool.name": name}, args, result)
            res = rec.iri("entity", f"{sid}-result")
            results[sid] = res if (res, None, None) in rec.g else None
            tool_steps[cid].append(step)
            key = (agent_of(s), name, same(args or "{}"))
            attempt = 1
            if key in failed:
                previous, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if cid in errors:
                failed[key] = (step, attempt)
            if name.startswith(DELEGATION):
                target = name[len(DELEGATION):]
                owner = agent_of(s)
                routing = steps.get((agent_span(s) or {}).get("id"))
                delegations.append((s, step, owner, target))
                rec.link(routing, PAI.selected, rec.agent(target))
                rec.link(rec.agent(target), PROV.actedOnBehalfOf, rec.agent(owner))
        elif o == "await_human_input":
            step = rec.step(sid, PAI.Check, parent, "approval", s["start"], s["end"], "completed")
            rec.link(step, PROV.wasAssociatedWith, approver)
        else:
            step = rec.step(sid, PAI.Step, parent, s["name"], s["start"], s["end"], s["status"], s["error"])
        steps[sid] = step

    for step, output, out_ids, read_ids in model_steps:
        for cid in out_ids:
            for t in tool_steps.get(cid, []):
                rec.used(t, output)
        for cid in read_ids:
            for t in tool_steps.get(cid, []):
                rec.used(step, results.get(rec.local(t)))

    # an agent run as a delegation returns its final answer as the tool result
    for s in spans:
        if op(s) == "execute_tool" and (s["attrs"].get("gen_ai.tool.name") or "").startswith(DELEGATION):
            res = results.get(s["id"])
            text = (s["attrs"].get("gen_ai.tool.call.result") or "").strip()
            for a_span, output, a_text in answers:
                inner = a_span
                while inner and inner["id"] != s["id"]:
                    inner = by_id.get(inner["parent"])
                if inner and a_text == text and res is not None:
                    rec.derived(res, output, "inferred")

    # a delegation whose result the delegating agent gives as its own final answer hands the work on
    for s, step, owner, target in delegations:
        text = (s["attrs"].get("gen_ai.tool.call.result") or "").strip()
        owner_span = agent_span(s)
        if text and any(agent_span(a_span) is owner_span and a_text == text and seconds(a_span["start"]) >= seconds(s["end"])
                        for a_span, _, a_text in answers):
            rec.g.add((step, RDF.type, PAI.Handoff))
            rec.link(step, PAI.fromAgent, rec.agent(owner))
            rec.link(step, PAI.toAgent, rec.agent(target))

    def tool_at(e):
        # the tool call of the declaring agent that was running when the application logged the event
        t = seconds(e["time"])
        inside = [s for s in spans if op(s) == "execute_tool" and seconds(s["start"]) <= t <= seconds(s["end"])
                  and agent_of(s) == e["by"]]
        return inside[-1] if inside else None

    # an earlier answer that appears word for word in a later model input was passed on to it (knownBy inferred)
    for s in spans:
        if op(s) != "chat":
            continue
        inp = rec.iri("entity", f"{s['id']}-input")
        text = " ".join(m.get("content") or "" for m in og.messages(s["attrs"].get("gen_ai.input.messages"))
                        if m.get("role") == "user")
        for a_span, output, a_text in answers:
            if len(a_text) >= 16 and seconds(a_span["end"]) <= seconds(s["start"]) and a_text in text and (inp, None, None) in rec.g:
                rec.derived(inp, output, "inferred")

    for e in ev:
        kind = e["kind"]
        if kind == "memory_write":
            span = next((s for s in spans if s["attrs"].get("gen_ai.tool.call.id") == e.get("call_id")), None)
            if span is None:
                raise SystemExit(f"memory write {e.get('call_id')} has no tool span")
            access = rec.step(f"{span['id']}-memory-write-{e['time']}", PAI.MemoryAccess, steps[span["id"]],
                              "memory write", span["end"], span["end"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
            # a written value comes from the latest answer of its writer with the same text
            source = latest([(seconds(a_span["end"]), output) for a_span, output, a_text in answers
                             if a_text == (e.get("value") or "").strip() and agent_of(a_span) == e["by"]], seconds(e["time"]))
            if source is not None:
                rec.derived(record, source, "inferred")
        elif kind == "memory_read":
            span = tool_at(e)
            if span is None:
                raise SystemExit("a memory read has no tool call around it")
            tool, tid = steps[span["id"]], span["id"]
            access = rec.step(f"{tid}-memory-read-{e['time']}", PAI.MemoryAccess, tool, "memory read", e["time"], e["time"],
                              "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            record = rec.memory_read(e["memory"], e["key"], access)
            value = rec.g.value(record, PROV.value) if record is not None else None
            res = results.get(tid)
            if res is not None and value is not None and str(rec.g.value(res, PROV.value)) == str(value):
                rec.derived(res, record, "inferred")
        elif kind == "check":
            check = rec.step(f"check-{e['time']}", PAI.Check, None, "check", e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            rec.link(check, PAI.checked, latest([(seconds(a_span["end"]), output) for a_span, output, _ in answers
                                                 if agent_of(a_span) != e["by"]], seconds(e["time"])))
            rec.set(check, PAI.iteration, e["iteration"])
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
        elif kind == "approval":
            gated = next((s for s in spans if s["attrs"].get("gen_ai.tool.call.id") == e["call_id"]), None)
            if gated is None:
                raise SystemExit(f"approval of {e['call_id']} has no tool span")
            check = next((steps[s["id"]] for s in spans if op(s) == "await_human_input" and s["parent"] == gated["id"]),
                         None)
            if check is None:
                raise SystemExit("the approval has no await_human_input span")
            rec.link(check, PAI.checked, rec.iri("entity", f"{gated['id']}-arguments"))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.link(steps[gated["id"]], PROV.wasInformedBy, check)
        elif kind == "effect":
            span = tool_at(e)
            if span is None:
                raise SystemExit("an effect has no tool call around it")
            effect = rec.entity(f"{span['id']}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, steps[span["id"]])
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    leftover = [n for n, rs in responses.items() if used_responses[n] != len(rs)]
    if leftover:
        raise SystemExit(f"native model responses without a chat span: {leftover}")
    return rec
