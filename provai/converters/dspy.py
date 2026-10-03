import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import openinference as oi

PLUMBING = ("Adapter.acall", "Adapter.__call__", "Predict(StringSignature).forward")


def text_of(value):
    return " ".join(str(value).split())


def span_key(s):
    kind, name = oi.kind(s), s["name"]
    if kind == "LLM":
        return ("lm", None)
    if kind == "TOOL":
        return ("tool", name.rsplit(".", 1)[0])
    return ("module", name.split(".", 1)[0])


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    rec = Record(f"dspy {v['dspy']} with openinference-instrumentation-dspy {v['openinference-instrumentation-dspy']}",
                 head["run_id"], True)
    spans = read(run_dir / "spans.jsonl")
    oi.session(rec, spans)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    # spans join native calls by kind, name, and start order; adapter spans sit inside their Predict call
    queues = defaultdict(list)
    for s in spans:
        if s["name"].endswith(PLUMBING):
            continue
        queues[span_key(s)].append(s)
    queues = {k: iter(q) for k, q in queues.items()}

    starts = {e["call_id"]: e for e in ev if e["kind"].endswith("_start")}
    ends = {e["call_id"]: e for e in ev if e["kind"].endswith("_end")}
    steps = {}
    answers, lm_outputs, tool_results, tool_args = [], {}, {}, {}
    invocations, react_turns = Counter(), Counter()

    def span_for(cid, key):
        s = next(queues[key], None) if key in queues else None
        if s is None:
            raise SystemExit(f"no span for native call {cid} {key}")
        return s

    def agent(cid):
        while cid in starts:
            if starts[cid].get("agent"):
                return starts[cid]["agent"]
            cid = starts[cid].get("parent")

    for e in ev:
        kind = e["kind"]
        if not kind.endswith("_start"):
            continue
        cid, parent = e["call_id"], steps.get(e.get("parent"))
        end = ends.get(cid)
        if end is None:
            raise SystemExit(f"native call {cid} has no end event")
        if kind == "lm_start":
            s = span_for(cid, ("lm", None))
        elif kind == "tool_start":
            s = span_for(cid, ("tool", e["tool"]))
        else:
            s = span_for(cid, ("module", e["cls"]))
        start, finish = s["start"], s["end"]
        status = "failed" if end.get("error") else "completed"
        name = agent(cid)

        if kind == "module_start" and e.get("agent"):
            invocations[name] += 1
            step = rec.step(cid, PAI.AgentInvocation, parent, name, start, finish, status, end.get("error"))
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            if end.get("outputs"):
                out = rec.entity(f"{cid}-answer", PAI.Message, f"{name} answer", end["outputs"])
                rec.generated(out, step)
                answers.append((seconds(end["time"]), name, out, text_of(end["outputs"])))
        elif kind == "module_start":
            step = rec.step(cid, PAI.Step, parent, e["cls"], start, finish, status, end.get("error"))
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            outer = starts.get(e.get("parent"), {})
            if e["cls"] == "Predict" and outer.get("cls") == "ReAct":
                react_turns[e["parent"]] += 1
                rec.set(step, PAI.iteration, react_turns[e["parent"]])
        elif kind == "lm_start":
            step = rec.step(cid, PAI.ModelCall, parent, "LM call", start, finish, status, end.get("error"))
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            inp, out = oi.model_call(rec, step, s["attrs"])
            usage = end.get("usage") or {}
            rec.set(step, PAI.inputTokens, usage.get("prompt_tokens"))
            rec.set(step, PAI.outputTokens, usage.get("completion_tokens"))
            cost = end.get("cost") if end.get("cost") is not None else usage.get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            lm_outputs[cid] = (seconds(e["time"]), name, inp, out, s["attrs"].get("input.value") or "")
        else:
            step = rec.step(cid, PAI.ToolCall, parent, e["tool"], start, finish, status, end.get("error"))
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            rec.link(step, PAI.usedTool, rec.tool(e["tool"]))
            args = rec.entity(f"{cid}-arguments", PROV.Entity, f"{e['tool']} arguments", json.dumps(e.get("inputs")))
            rec.used(step, args)
            tool_args[cid] = args
            if end.get("outputs") is not None and not end.get("error"):
                res = rec.entity(f"{cid}-result", PROV.Entity, f"{e['tool']} result", end["outputs"])
                rec.generated(res, step)
                tool_results[cid] = res
        steps[cid] = step

    # a ReAct turn's model call chose the tool that follows it; the next turn and the extract read its result
    for react in [c for c, e in starts.items() if e.get("cls") == "ReAct"]:
        children = sorted(((c, e) for c, e in starts.items() if e.get("parent") == react),
                          key=lambda ce: seconds(ce[1]["time"]))
        last_lm, results = None, []
        for c, e in children:
            inner = [lm for lm, x in starts.items() if x["kind"] == "lm_start" and in_call(starts, lm, c)]
            if e["kind"] == "tool_start":
                if last_lm is not None:
                    rec.used(steps[c], lm_outputs[last_lm][3])
                if c in tool_results:
                    results.append(tool_results[c])
            elif inner:
                for lm in inner:
                    for r in results:
                        rec.used(steps[lm], r)
                last_lm = inner[-1]

    # an agent's answer is a field DSPy parsed from the last model output inside the agent's call (inferred)
    for c, e in starts.items():
        if e["kind"] == "module_start" and e.get("agent") and ends[c].get("outputs"):
            inner = sorted((seconds(x["time"]), lm) for lm, x in starts.items()
                           if x["kind"] == "lm_start" and in_call(starts, lm, c))
            answer = rec.iri("entity", f"{c}-answer")
            for _, lm in reversed(inner):
                out = lm_outputs[lm][3]
                if out is not None and text_of(ends[c]["outputs"]) in text_of(ends[lm]["outputs"] or ""):
                    rec.derived(answer, out, "inferred")
                    break

    # a failed tool call sends the model back; its next call of the same tool and arguments is the retry
    failed = {}
    for e in ev:
        if e["kind"] != "tool_start":
            continue
        key = (agent(e["call_id"]), e["tool"], same(e.get("inputs")))
        step, attempt = steps[e["call_id"]], 1
        if key in failed:
            prior, attempt = failed.pop(key)
            attempt += 1
            rec.link(step, PAI.retryOf, prior)
            rec.set(step, PAI.attempt, attempt)
        if ends[e["call_id"]].get("error"):
            failed[key] = (step, attempt)

    # an earlier agent answer that appears word for word in a model input reached that call (inferred)
    for _, (when, name, inp, out, prompt) in lm_outputs.items():
        flat = text_of(json.loads(prompt) if prompt.startswith("{") else prompt)
        for t, who, answer, text in answers:
            if t < when and who != name and len(text) >= 16 and text in flat and inp is not None:
                rec.derived(inp, answer, "inferred")

    routing = None
    for e in ev:
        kind = e["kind"]
        if kind == "routing":
            # the router's own call that the routing was declared from, else its latest call before it
            c = e.get("parent")
            while c in starts and starts[c].get("agent") != e["by"]:
                c = starts[c].get("parent")
            routing = steps.get(c) if c in starts else latest(
                [(seconds(x["time"]), steps[c]) for c, x in starts.items() if x.get("agent") == e["by"]],
                seconds(e["time"]))
            if routing is None:
                raise SystemExit(f"no agent call of the router {e['by']}")
            rec.g.add((routing, RDF.type, PAI.Routing))
            for a in e["candidates"]:
                rec.link(routing, PAI.candidate, rec.agent(a))
            for a in e["selected"]:
                rec.link(routing, PAI.selected, rec.agent(a))
                rec.link(rec.agent(a), PROV.actedOnBehalfOf, rec.agent(e["by"]))
            rec.link(rec.agent(e["by"]), PROV.actedOnBehalfOf, user)
            rec.set(routing, PAI.reason, e.get("reason"))
        elif kind == "handoff":
            step = rec.step(f"handoff-{e['time']}", PAI.Handoff, None, "handoff", e["time"], e["time"], "completed")
            src, dst = rec.agent(e["by"]), rec.agent(e["to"])
            rec.link(step, PROV.wasAssociatedWith, src)
            rec.link(step, PAI.fromAgent, src)
            rec.link(step, PAI.toAgent, dst)
            rec.link(dst, PROV.actedOnBehalfOf, src)
            # the handoff passes on what the other agents answered before it
            for t, who, answer, _ in answers:
                if t < seconds(e["time"]) and who not in (e["by"], e["to"]):
                    rec.used(step, answer)
        elif kind in ("memory_write", "memory_read"):
            access = rec.step(f"{e['parent']}-{kind}-{e['time']}", PAI.MemoryAccess, steps.get(e["parent"]),
                              kind.replace("_", " "), e["time"], e["time"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            if kind == "memory_write":
                record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                # the writer's own answer with the same text; its call ends just after it writes the note
                source = latest([(t, answer) for t, who, answer, text in answers
                                 if who == e["by"] and text == text_of(e.get("value") or "")], seconds(e["time"]), 1)
                rec.derived(record, source, "inferred")
            else:
                record = rec.memory_read(e["memory"], e["key"], access)
                res = tool_results.get(e["parent"])
                value = rec.g.value(record, PROV.value) if record is not None else None
                if res is not None and value is not None and text_of(ends[e["parent"]]["outputs"]) == text_of(value):
                    rec.derived(res, record, "inferred")
        elif kind in ("check", "approval"):
            check = rec.step(f"{e.get('parent') or 'run'}-{kind}-{e['time']}", PAI.Check, steps.get(e.get("parent")),
                             kind, e["time"], e["time"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver if kind == "approval" else rec.agent(e["by"]))
            if kind == "approval":
                rec.link(check, PAI.checked, tool_args.get(e["parent"]))
                rec.link(steps.get(e["parent"]), PROV.wasInformedBy, check)
            else:
                # the latest answer before the check by an agent other than the checker
                rec.link(check, PAI.checked, latest([(t, a) for t, who, a, _ in answers if who != e["by"]],
                                                    seconds(e["time"])))
                rec.set(check, PAI.iteration, e.get("iteration"))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
        elif kind == "effect":
            if e["parent"] not in steps:
                raise SystemExit(f"the effect names an unknown call {e['parent']}")
            effect = rec.entity(f"{e['parent']}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            rec.generated(effect, steps[e["parent"]])
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    # an earlier agent answer that appears word for word in a tool's arguments was passed to it (inferred)
    for c, e in starts.items():
        if e["kind"] == "tool_start":
            given = text_of(" ".join(str(v) for v in (e.get("inputs") or {}).values()))
            source = latest([(t, a) for t, who, a, text in answers if len(text) >= 16 and text in given],
                            seconds(e["time"]))
            rec.derived(tool_args[c], source, "inferred")

    leftover = [s["name"] for q in queues.values() for s in q]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native call: {leftover}")
    return rec


def in_call(starts, cid, top):
    while cid in starts:
        if cid == top:
            return True
        cid = starts[cid].get("parent")
    return False
