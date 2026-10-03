import json
import re
from collections import Counter, defaultdict

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import openinference as oi


def event_type(value):
    found = re.findall(r"'([^']+)'", value or "")
    name = (found[-1] if found else value or "").rsplit(".", 1)[-1]
    return None if name in ("None", "NoneType", "") else name


def text_of(value):
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return value
    if isinstance(data, dict) and data.get("blocks"):
        return " ".join(b.get("text", "") for b in data["blocks"])
    return value


def squash(text):
    # letters and digits only, so an answer matches the escaped and shortened repr of the event that carries it
    return re.sub(r"[^a-z0-9]", "", (text or "").replace("\\u202f", "").lower())


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    rec = Record(f"llama-index-core {v['llama-index-core']} with llama-index-workflows {v['llama-index-workflows']} "
                 f"and openinference-instrumentation-llama-index {v['openinference-instrumentation-llama-index']}",
                 head["run_id"], True)
    settings = next((e for e in ev if e["kind"] == "llm"), None)
    workflow = next((e for e in ev if e["kind"] == "workflow"), None)
    if settings is None or workflow is None:
        raise SystemExit("the run has no model settings or no workflow definition")
    defs = workflow["definitions"]
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    spans = read(run_dir / "spans.jsonl")
    oi.session(rec, spans)
    enters = [e for e in ev if e["kind"] == "span_enter"]
    if len(enters) != len(spans):
        raise SystemExit(f"{len(spans)} spans for {len(enters)} native spans")
    otel = {}
    for e, s in zip(enters, spans):
        if s["name"] != e["id"].partition("-")[0]:
            raise SystemExit(f"span {s['name']} has no matching native span (found {e['id']})")
        otel[e["id"]] = s
    dropped = {e["id"]: e for e in ev if e["kind"] == "span_drop"}
    native = {e["id"]: e for e in enters}
    chat_end = {e["span_id"]: e for e in ev if e["kind"] == "LLMChatEndEvent"}
    # one model call is an achat span inside another; the inner one is the call
    wrappers = set()
    for e in enters:
        if e["id"].partition("-")[0].endswith(".achat"):
            p = e["parent"]
            while p:
                if p.partition("-")[0].endswith(".achat"):
                    wrappers.add(p)
                p = native[p]["parent"] if p in native else None
    turns = [e["id"] for e in enters if e["id"].partition("-")[0] == "AgentWorkflow.run_agent_step"]
    # agent turns and AgentInput events come in the same order
    turn_agent = dict(zip(turns, (e["current_agent_name"] for e in ev if e["kind"] == "agent_AgentInput")))
    flow = next((e["instance"] for e in enters if e["parent"] is None), None)
    if flow is None:
        raise SystemExit("no root span for the workflow run")

    def step_of(sid):
        # the workflow step a span runs in, by the step names of the workflow definition
        while sid:
            cls, _, method = sid.partition("-")[0].rpartition(".")
            if (cls == flow or cls.endswith("." + flow)) and method in defs:
                return method
            sid = native[sid]["parent"] if sid in native else None

    # the agent of a step: the one agent that runs inside it, else the agent that declared facts from it,
    # else the step itself
    agents_in, declared = defaultdict(set), defaultdict(set)
    for e in enters:
        kind = e["id"].partition("-")[0]
        name = e["name"] if kind == "FunctionAgent.run" else turn_agent.get(e["id"])
        if name and step_of(e["parent"]):
            agents_in[step_of(e["parent"])].add(name)
    for e in ev:
        if e["kind"] in ("memory_write", "memory_read", "check", "effect") and step_of(e.get("span_id")):
            declared[step_of(e["span_id"])].add(e["by"])

    def step_agent(method):
        if agents_in[method]:
            return next(iter(agents_in[method])) if len(agents_in[method]) == 1 else None
        return next(iter(declared[method])) if len(declared[method]) == 1 else method
    handoffs = iter(e for e in ev if e["kind"] == "agent_ToolCall" and e["tool_name"] == "handoff")

    steps, agent_of, runs, turns = {}, {}, Counter(), Counter()
    step_runs, last_attempt, used_by = defaultdict(list), {}, {}
    model_steps, tool_steps, requested = [], defaultdict(list), defaultdict(list)
    answers, failed_tools, turn = [], {}, {}

    def inside(sid, top):
        while sid:
            if sid == top:
                return True
            sid = native[sid]["parent"] if sid in native else None

    def kept(sid):
        while sid and sid not in steps:
            sid = native[sid]["parent"] if sid in native else None
        return steps.get(sid)

    def agent_for(sid):
        while sid:
            if sid in agent_of:
                return agent_of[sid]
            sid = native[sid]["parent"] if sid in native else None

    def targets_of(method):
        # the steps that accept an event this step can return
        emits = set(defs[method]["returns"])
        return [n for n, d in defs.items() if n != method and emits & set(d["accepts"])]

    def context_of(sid):
        # the agent run a span belongs to, where model calls and the tool calls they request meet
        while sid:
            kind = sid.partition("-")[0]
            if kind in ("FunctionAgent.run", "AgentWorkflow.run"):
                return sid
            sid = native[sid]["parent"] if sid in native else None

    for e in enters:
        sid, s = e["id"], otel[e["id"]]
        kind = sid.partition("-")[0]
        cls, method = (kind.rsplit(".", 1) + [""])[:2]
        parent = kept(e["parent"])
        error = (dropped.get(sid) or {}).get("error")
        failed = bool(error)
        status = "failed" if failed else "completed"

        if e["parent"] is None:
            steps[sid] = rec.step(sid, PAI.Step, None, "workflow run", s["start"], s["end"], status, error)

        elif step_of(sid) == method and (cls == flow or cls.endswith("." + flow)):
            runs[method] += 1
            step = rec.step(sid, PAI.Routing if len(targets_of(method)) > 1 else PAI.Step, parent, method,
                            s["start"], s["end"], status, error)
            rec.set(step, PAI.iteration, runs[method])
            agent = rec.agent(step_agent(method))
            rec.link(step, PROV.wasAssociatedWith, agent)
            agent_of[sid] = agent
            earlier = step_runs[method]
            last_attempt[method] = last_attempt.get(method, 1) + 1 if earlier and earlier[-1][2] else 1
            if last_attempt[method] > 1:
                rec.link(step, PAI.retryOf, earlier[-1][1])
                rec.set(step, PAI.attempt, last_attempt[method])
            step_runs[method].append((sid, step, failed))
            steps[sid] = step

        elif kind == "FunctionAgent.run":
            step = rec.step(sid, PAI.AgentInvocation, parent, e["name"], s["start"], s["end"], status, error)
            turns[e["name"]] += 1
            rec.set(step, PAI.iteration, turns[e["name"]])
            agent_of[sid] = rec.agent(e["name"])
            rec.link(step, PROV.wasAssociatedWith, agent_of[sid])
            steps[sid] = step

        elif kind == "AgentWorkflow.run":
            steps[sid] = rec.step(sid, PAI.Step, parent, "agent workflow", s["start"], s["end"], status, error)

        elif kind == "AgentWorkflow.run_agent_step":
            name = turn_agent.get(sid)
            if name is None:
                raise SystemExit(f"no AgentInput event for the agent turn {sid}")
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], status, error)
            turns[name] += 1
            rec.set(step, PAI.iteration, turns[name])
            agent_of[sid] = rec.agent(name)
            rec.link(step, PROV.wasAssociatedWith, agent_of[sid])
            # tool calls of an agent workflow run beside the turn; they belong to the agent whose turn asked
            turn[context_of(sid)] = (agent_of[sid], step)
            steps[sid] = step

        elif method == "achat" and (sid in chat_end or (failed and sid not in wrappers)):
            end = chat_end.get(sid, {})
            step = rec.step(sid, PAI.ModelCall, parent, "chat", s["start"], s["end"], status, error)
            agent = agent_for(sid)
            rec.link(step, PROV.wasAssociatedWith, agent)
            _, out = oi.model_call(rec, step, s["attrs"])
            rec.set(step, PAI.temperature, settings.get("temperature"))
            rec.set(step, PAI.maxTokens, settings.get("max_tokens"))
            cost = (end.get("usage") or {}).get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            called = oi.tool_calls(s["attrs"])
            requested[context_of(sid)].extend(called)
            model_steps.append((step, out, [c["id"] for c in called], oi.tool_results(s["attrs"])))
            text = (end.get("content") or "").strip()
            if text and text != "None" and not called:
                answers.append((sid, agent, out, text, seconds(s["end"]) if s["end"] else None))
            steps[sid] = step

        elif kind == "FunctionTool.acall":
            name = e["name"]
            calls = requested[context_of(sid)]
            match = next((c for c in calls if c["name"] == name), None)
            if match:
                calls.remove(match)
            agent = agent_for(sid)
            if context_of(sid) in turn:
                agent, parent = turn[context_of(sid)]
            if name == "handoff":
                h = next(handoffs, None)
                if h is None:
                    raise SystemExit(f"no handoff tool call event for {sid}")
                step = rec.step(sid, PAI.Handoff, parent, "handoff", s["start"], s["end"], status, error)
                src, dst = agent, rec.agent(h["tool_kwargs"]["to_agent"])
                rec.link(step, PROV.wasAssociatedWith, src)
                rec.link(step, PAI.fromAgent, src)
                rec.link(step, PAI.toAgent, dst)
                rec.link(dst, PROV.actedOnBehalfOf, src)
                rec.set(step, PAI.reason, h["tool_kwargs"].get("reason"))
                if match:
                    tool_steps[match["id"]].append((step, None))
            else:
                step = rec.step(sid, PAI.ToolCall, parent, name, s["start"], s["end"], status, error)
                rec.link(step, PROV.wasAssociatedWith, agent)
                attrs = dict(s["attrs"])
                if attrs.get("output.value") is not None:
                    attrs["output.value"] = text_of(attrs["output.value"])
                _, res = oi.tool_call(rec, step, attrs)
                if match:
                    tool_steps[match["id"]].append((step, res))
                key, attempt = (agent, name, same(attrs.get("input.value"))), 1
                if key in failed_tools:
                    prior, attempt = failed_tools.pop(key)
                    attempt += 1
                    rec.link(step, PAI.retryOf, prior)
                    rec.set(step, PAI.attempt, attempt)
                if failed:
                    failed_tools[key] = (step, attempt)
            steps[sid] = step

    oi.link_tool_calls(rec, model_steps, tool_steps)

    # routing: a step whose events go to several steps; the candidates are the agents of those steps
    routers = [m for m in defs if m in step_runs and len(targets_of(m)) > 1]
    for m in routers:
        routing = step_runs[m][0][1]
        router = rec.agent(step_agent(m))
        for n in targets_of(m):
            rec.link(routing, PAI.candidate, rec.agent(step_agent(n)))
        rec.link(router, PROV.actedOnBehalfOf, user)
        reason = next((t for sid, _, _, t, _ in answers if kept(native[sid]["parent"]) == routing), None)
        rec.set(routing, PAI.reason, reason)

    # data flow: each step run used the workflow event that triggered it and generated the event it returned
    states = [e for e in ev if e["kind"] == "step_state"]
    started = defaultdict(list)
    for e in states:
        started[(e["name"], e["state"])].append(e)
    queue, buffered, made = defaultdict(list), defaultdict(list), Counter()
    # an event that no workflow step accepts goes out to the person, such as a request for approval
    accepted = {a for d in defs.values() for a in d["accepts"]}
    run_index = Counter()
    order = sorted(((seconds(otel[sid]["start"]), name, sid, step, bad) for name, rs in step_runs.items()
                    for sid, step, bad in rs))
    last_run, approval_draft = {}, None
    for _, name, sid, step, bad in order:
        i = run_index[name]
        run_index[name] += 1
        if len(started[(name, "not_running")]) <= i:
            raise SystemExit(f"no state change event for run {i + 1} of the step {name}")
        begin = started[(name, "running")][i]
        end = started[(name, "not_running")][i]
        need = event_type(begin["input"])
        if need and need not in ("StartEvent", "HumanResponseEvent"):
            if not queue[need]:
                producer = next((last_run[n] for n in reversed(list(last_run)) if need in defs[n]["returns"]), None)
                if producer is None:
                    raise SystemExit(f"no earlier step returns the {need} that {name} accepts")
                made[need] += 1
                ent = rec.entity(f"{need}-{made[need]}", PROV.Entity, need)
                rec.generated(ent, producer)
                queue[need].append(ent)
            used = queue[need].pop(0) if not bad else queue[need][0]
            rec.used(step, used)
            used_by[step] = used
            buffered[name].append(used)
        out = event_type(end["output"])
        if out and not bad and out != "StopEvent":
            for ent in buffered[name][:-1]:
                rec.used(step, ent)
            buffered[name] = []
            made[out] += 1
            ent = rec.entity(f"{out}-{made[out]}", PROV.Entity, out, otel[sid]["attrs"].get("output.value"))
            rec.generated(ent, step)
            queue[out].append(ent)
            # the event a step returns holds the final answer of the agent that ran inside it
            inner = [a for a in answers if inside(a[0], sid)]
            shown = re.search(r"text=['\"](.*?)(?:\.\.\.)?['\"][,)]", otel[sid]["attrs"].get("output.value") or "")
            core = squash(shown.group(1)) if shown else ""
            if inner and len(core) >= 20 and squash(inner[-1][3]).startswith(core):
                rec.derived(ent, inner[-1][2], "inferred")
            if out not in accepted:
                approval_draft = used_by.get(step)
        last_run[name] = step

    # selected agents: the steps that ran on what the router emitted
    for m in routers:
        routing, router = step_runs[m][0][1], rec.agent(step_agent(m))
        for n in targets_of(m):
            if n in step_runs:
                rec.link(routing, PAI.selected, rec.agent(step_agent(n)))
                rec.link(rec.agent(step_agent(n)), PROV.actedOnBehalfOf, router)

    # facts the application declared through the dispatcher
    checks = []
    for e in ev:
        kind = e["kind"]
        if kind in ("memory_write", "memory_read"):
            parent = steps.get(e["span_id"])
            access = rec.step(f"{e['span_id']}-{kind}-{e['time']}", PAI.MemoryAccess, parent, kind.replace("_", " "),
                              e["time"], e["time"], "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
            if kind == "memory_write":
                record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                # the writer's latest answer with the written text, ended by the write
                source = latest([(end, out) for _, a, out, text, end in answers
                                 if a == rec.agent(e["by"]) and text == (e.get("value") or "").strip()],
                                seconds(e["time"]))
                rec.derived(record, source, "inferred")
            elif (record := rec.memory_read(e["memory"], e["key"], access)) is not None:
                result = rec.iri("entity", f"{e['span_id']}-result")
                value = rec.g.value(record, PROV.value)
                if value is not None and str(rec.g.value(result, PROV.value)) == str(value):
                    rec.derived(result, record, "inferred")
        elif kind == "check":
            step = steps.get(e["span_id"])
            if step is None:
                raise SystemExit(f"the check names an unknown span {e['span_id']}")
            check = rec.step(f"{e['span_id']}-check-{e['time']}", PAI.Check, step, "check", e["time"], e["time"],
                             "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            rec.link(check, PAI.checked, used_by.get(step) or rec.g.value(step, PROV.used))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))
            rec.set(check, PAI.iteration, int(e["iteration"]))
            checks.append(check)
        elif kind == "human_response":
            check = rec.step(f"approval-{e['time']}", PAI.Check, rec.run, "approval", e["time"], e["time"],
                             "completed")
            rec.link(check, PROV.wasAssociatedWith, approver)
            rec.link(check, PAI.checked, approval_draft)
            rec.set(check, PAI.outcome, e["response"])
            answered = [n for n, d in defs.items() if "HumanResponseEvent" in d["accepts"]]
            for sid, step, _ in (r for n in answered for r in step_runs[n]):
                rec.link(step, PROV.wasInformedBy, check)
                for child, cstep in steps.items():
                    if native[child]["parent"] == sid:
                        rec.link(cstep, PROV.wasInformedBy, check)
        elif kind == "effect":
            effect = rec.entity(f"{e['span_id']}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
            rec.set(effect, PAI.target, e["target"])
            if e["span_id"] not in steps:
                raise SystemExit(f"the effect names an unknown span {e['span_id']}")
            rec.generated(effect, steps[e["span_id"]])
            rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))
    return rec
