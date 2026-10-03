import json
from collections import Counter, defaultdict

from rdflib import XSD, Literal

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import openinference as oi


def answer(attrs):
    return " ".join(m.get("content") or "" for m in oi.indexed(attrs, "llm.output_messages")).strip()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    models = next((e["config"] for e in ev if e["kind"] == "models"), {})
    definition = next((e["definition"] for e in ev if e["kind"] == "workflow"), None)
    if definition is None:
        raise SystemExit("the run has no workflow definition")
    routing = next((e for e in ev if e["kind"] == "routing"), {})
    spans = read(run_dir / "spans.jsonl")

    rec = Record(f"agno {v['agno']} with openinference-instrumentation-agno "
                 f"{v['openinference-instrumentation-agno']}", head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)
    start = next((e for e in ev if e["kind"] == "WorkflowStarted"), None)
    if start is None:
        raise SystemExit("the run has no WorkflowStarted event")
    rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, start["session_id"], start["session_id"]))

    # model spans join native model requests by their token counts; tool spans join tool calls by name and order
    llm = defaultdict(list)
    tools = defaultdict(list)
    by_id = {s["id"]: s for s in spans}
    for s in spans:
        a = s["attrs"]
        # a failed request has no native event and no token counts; it is recorded from its span below
        if oi.kind(s) == "LLM" and s["status"] != "failed":
            llm[(a.get("llm.token_count.prompt"), a.get("llm.token_count.completion"))].append(s)
        elif oi.kind(s) == "TOOL":
            tools[a.get("tool.name")].append(s)
    agent_spans = {s["attrs"].get("agno.run.id"): s for s in spans if oi.kind(s) == "AGENT"}
    used_spans = set()
    # a team delegates by member id; the runs name each id's agent
    agent_names = {e["agent_id"]: e["agent_name"] for e in ev if e["kind"] == "RunStarted" and e.get("agent_id")}
    accesses, effects = {}, {}

    steps, open_steps, step_count = {}, {}, Counter()
    invocations, runs_of_agent, agent_of_run = {}, Counter(), {}
    model_steps, tool_steps, results = [], defaultdict(list), {}
    tool_step, held, failed = {}, {}, {}
    iteration_step = {}
    access_record, step_parent, router, router_id = {}, {}, None, None
    outputs, inputs, answered = [], [], []

    def ended(kinds, key, value):
        return next((e["time"] for e in ev if e["kind"] in kinds and e.get(key) == value), None)

    def status(step, value, error=None):
        rec.g.remove((step, PAI.status, None))
        rec.set(step, PAI.status, value)
        rec.set(step, PAI.errorType, error)

    def under(sid, top):
        while sid:
            if sid == top:
                return True
            sid = step_parent.get(sid)

    def where(e):
        if e.get("parent_run_id") in invocations:
            return invocations[e["parent_run_id"]]
        return open_steps.get(e.get("step_id"))

    def model_call(step, span, agent):
        used_spans.add(span["id"])
        inp, out = oi.model_call(rec, step, span["attrs"])
        if inp is not None:
            said = " ".join(m.get("content") or "" for m in oi.indexed(span["attrs"], "llm.input_messages"))
            inputs.append((seconds(span["start"]), inp, said))
        params = models.get(agent) or {}
        rec.set(step, PAI.temperature, params.get("temperature"))
        rec.set(step, PAI.maxTokens, params.get("max_tokens"))
        called = oi.tool_calls(span["attrs"])
        model_steps.append((step, out, [c["id"] for c in called], oi.tool_results(span["attrs"])))
        text = answer(span["attrs"])
        if out is not None and text and not called:
            answered.append((agent, out, seconds(span["end"]), text))
            outputs.append((seconds(span["end"]), out, text))
        for c in called:
            # a delegated task travels as a tool argument and reaches the member's prompt
            try:
                values = json.loads(c["arguments"] or "{}").values()
            except (ValueError, AttributeError):
                values = []
            outputs.extend((seconds(span["end"]), out, x) for x in values if isinstance(x, str))

    for i, e in enumerate(ev):
        k, t = e["kind"], e["time"]
        if k in ("RouterExecutionStarted", "ParallelExecutionStarted", "LoopExecutionStarted", "StepStarted"):
            sid = e["step_id"]
            step_parent[sid] = e.get("parent_step_id")
            step_count[sid] += 1
            local = f"{sid}-{step_count[sid]}"
            cls = PAI.Routing if k == "RouterExecutionStarted" else PAI.Step
            parent = iteration_step.get(e.get("parent_step_id")) or open_steps.get(e.get("parent_step_id"))
            step = rec.step(local, cls, parent, e["step_name"], t)
            status(step, "completed")
            steps[local] = step
            open_steps[sid] = step
            if k == "RouterExecutionStarted":
                router, router_id = step, sid
                spec = next((c for c in definition["steps"] if c.get("name") == e["step_name"]), None)
                if spec is None or not routing:
                    raise SystemExit(f"router {e['step_name']} has no definition or no declared routing")
                for choice in spec["choices"]:
                    rec.link(step, PAI.candidate, rec.agent(routing["agents"][choice["name"]]))
                rec.set(step, PAI.reason, routing.get("answer"))
        elif k in ("RouterExecutionCompleted", "ParallelExecutionCompleted", "LoopExecutionCompleted", "StepCompleted"):
            step = open_steps.get(e["step_id"])
            if step is not None:
                rec.g.add((step, PROV.endedAtTime, Literal(t, datatype=XSD.dateTime)))
        elif k == "LoopIterationStarted":
            local = f"{e['step_id']}-iteration-{e['iteration']}"
            step = rec.step(local, PAI.Step, open_steps[e["step_id"]], f"iteration {e['iteration']}", t)
            rec.set(step, PAI.iteration, e["iteration"])
            status(step, "completed")
            iteration_step[e["step_id"]] = step
        elif k == "RunStarted":
            name = e.get("agent_name") or e.get("team_name")
            runs_of_agent[name] += 1
            end = ended(("RunCompleted", "RunError"), "run_id", e["run_id"])
            step = rec.step(e["run_id"], PAI.AgentInvocation, where(e), name, t, end)
            rec.set(step, PAI.iteration, runs_of_agent[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            error = next((x.get("error") or x.get("content") for x in ev
                          if x["kind"] == "RunError" and x.get("run_id") == e["run_id"]), None)
            status(step, "failed" if error else "completed", error)
            invocations[e["run_id"]] = step
            agent_of_run[e["run_id"]] = name
            if router_id and under(e.get("step_id"), router_id):
                rec.link(router, PAI.selected, rec.agent(name))
                rec.link(rec.agent(name), PROV.actedOnBehalfOf, rec.agent(routing["by"]))
                rec.link(rec.agent(routing["by"]), PROV.actedOnBehalfOf, user)
        elif k == "ModelRequestCompleted":
            started = next((x["time"] for x in reversed(ev[:i])
                            if x["kind"] == "ModelRequestStarted" and x.get("run_id") == e["run_id"]), None)
            if started is None:
                raise SystemExit(f"model request of run {e['run_id']} has no start")
            agent = agent_of_run[e["run_id"]]
            step = rec.step(f"{e['run_id']}-model-{started}", PAI.ModelCall, invocations[e["run_id"]], "model request",
                            started, t)
            status(step, "completed")
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            matches = llm.get((e.get("input_tokens"), e.get("output_tokens"))) or []
            if not matches:
                raise SystemExit(f"no model span with {e.get('input_tokens')} and {e.get('output_tokens')} tokens")
            model_call(step, matches.pop(0), agent)
        elif k == "RunPaused":
            for tool in e.get("tools") or []:
                if tool.get("requires_confirmation") and not tool.get("confirmed"):
                    cid = tool["tool_call_id"]
                    step = rec.step(f"{cid}-held", PAI.ToolCall, invocations[e["run_id"]], tool["tool_name"], t, t)
                    status(step, "interrupted")
                    rec.link(step, PAI.usedTool, rec.tool(tool["tool_name"]))
                    rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of_run[e["run_id"]]))
                    args = rec.entity(f"{cid}-held-arguments", PROV.Entity, f"{tool['tool_name']} arguments",
                                      json.dumps(tool.get("tool_args")))
                    rec.used(step, args)
                    held[cid] = {"step": step, "arguments": args}
                    tool_steps[cid].append((step, None))
        elif k == "ToolCallStarted":
            tool = e["tool"]
            cid, name = tool["tool_call_id"], tool["tool_name"]
            local = f"{cid}-run" if cid in held else cid
            done = next((x for x in ev[i:] if x["kind"] == "ToolCallCompleted"
                         and x["tool"]["tool_call_id"] == cid), None)
            step = rec.step(local, PAI.ToolCall, invocations[e["run_id"]], name, t, done and done["time"])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent_of_run[e["run_id"]]))
            span = tools[name].pop(0) if tools.get(name) else None
            if span is None:
                raise SystemExit(f"no tool span for {name}")
            used_spans.add(span["id"])
            args, res = oi.tool_call(rec, step, span["attrs"])
            error = done and done["tool"].get("tool_call_error")
            status(step, "failed" if error else "completed", done["tool"].get("result") if error else None)
            tool_steps[cid].append((step, res))
            tool_step[cid] = step
            results[cid] = res
            call = (agent_of_run[e["run_id"]], name, same(tool.get("tool_args")))
            attempt = 1
            if call in failed:
                previous, attempt = failed.pop(call)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if error:
                failed[call] = (step, attempt)
            if cid in held:
                rec.used(step, held[cid]["arguments"])
                rec.link(step, PROV.wasInformedBy, held[cid].get("check"))
            if name == "delegate_task_to_member":
                member_id = (tool.get("tool_args") or {}).get("member_id")
                member = agent_names.get(member_id, member_id)
                handoff = rec.step(f"{cid}-handoff", PAI.Handoff, invocations[e["run_id"]], "handoff", t, t)
                status(handoff, "completed")
                source, target = rec.agent(agent_of_run[e["run_id"]]), rec.agent(member)
                rec.link(handoff, PROV.wasAssociatedWith, source)
                rec.link(handoff, PAI.fromAgent, source)
                rec.link(handoff, PAI.toAgent, target)
                rec.link(target, PROV.actedOnBehalfOf, source)
                rec.used(handoff, args)
        elif k == "RunContinued":
            for cid, h in held.items():
                if "check" in h:
                    continue
                completed = next((x for x in ev[i:] if x["kind"] == "ToolCallCompleted"
                                  and x["tool"]["tool_call_id"] == cid), None)
                decision = completed and completed["tool"].get("confirmed")
                if decision is not None:
                    check = rec.step(f"{cid}-approval", PAI.Check, invocations[e["run_id"]], "approval", t, t)
                    status(check, "completed")
                    rec.link(check, PROV.wasAssociatedWith, approver)
                    rec.link(check, PAI.checked, h["arguments"])
                    rec.set(check, PAI.outcome, "approve" if decision else "reject")
                    h["check"] = check
        elif k == "Custom":
            name = e.get("name")
            if name == "memory_write":
                access = rec.step(f"{e['step_id']}-memory-write-{t}", PAI.MemoryAccess, open_steps.get(e["step_id"]),
                                  "memory write", t, t)
                status(access, "completed")
                rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
                record = rec.memory_write(e["memory"], e["key"], e.get("value"), access)
                # Agno session state lives as long as the workflow session
                rec.set(record, PAI.scope, "session")
                # a written value comes from the latest answer of its writer with the same text
                source = latest([(end, out) for agent, out, end, text in answered
                                 if agent == e["by"] and text == (e.get("value") or "").strip()], seconds(t))
                if source is not None:
                    rec.derived(record, source, "inferred")
            elif name == "memory_read":
                cid = e["tool_call_id"]
                access = rec.step(f"{cid}-memory-read-{t}", PAI.MemoryAccess, None, "memory read", t, t)
                status(access, "completed")
                rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
                access_record[i] = rec.memory_read(e["memory"], e["key"], access)
                accesses[i] = access
            elif name == "check":
                local = f"{e['step_id']}-{step_count[e['step_id']]}"
                check = rec.step(f"{local}-check-{t}", PAI.Check, steps.get(local), "check", t, t)
                status(check, "completed")
                rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
                # a check judges the latest answer before it that another agent gave
                rec.link(check, PAI.checked, latest([(end, out) for agent, out, end, _ in answered if agent != e["by"]],
                                                     seconds(t)))
                rec.set(check, PAI.outcome, e["outcome"])
                rec.set(check, PAI.reason, e.get("reason"))
                rec.set(check, PAI.iteration, e.get("iteration"))
            elif name == "effect":
                effect = rec.entity(f"{e['tool_call_id']}-effect-{t}", PAI.Effect, "effect", e["target"])
                rec.set(effect, PAI.target, e["target"])
                effects[i] = effect
                rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))

    # memory reads and effects happen inside their tool call, whose step exists once the call is recorded
    for i, e in enumerate(ev):
        if e["kind"] == "Custom" and e.get("name") == "memory_read":
            cid = e["tool_call_id"]
            if cid in tool_step:
                rec.g.remove((accesses[i], PAI.partOf, None))
                rec.link(accesses[i], PAI.partOf, tool_step[cid])
            record = access_record.get(i)
            value = record is not None and rec.g.value(record, PROV.value)
            res = results.get(cid)
            if value and res is not None and str(rec.g.value(res, PROV.value)) == str(value):
                rec.derived(res, record, "inferred")
        if e["kind"] == "Custom" and e.get("name") == "effect":
            rec.generated(effects[i], tool_step.get(e["tool_call_id"]))

    # the router agent's own call runs outside the event stream; its spans are the only record of it
    for run_id, span in agent_spans.items():
        name = span["attrs"].get("agent.name")
        if run_id in invocations or not routing or name != routing.get("by"):
            continue
        step = rec.step(run_id, PAI.AgentInvocation, router, name, span["start"], span["end"], span["status"])
        runs_of_agent[name] += 1
        rec.set(step, PAI.iteration, runs_of_agent[name])
        rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
        rec.link(router, PROV.wasAssociatedWith, rec.agent(name))
        for s in spans:
            if oi.kind(s) == "LLM" and s["parent"] == span["id"] and s["id"] not in used_spans:
                call = rec.step(s["id"], PAI.ModelCall, step, "model request", s["start"], s["end"], s["status"])
                rec.link(call, PROV.wasAssociatedWith, rec.agent(name))
                model_call(call, s, name)
                key = (s["attrs"].get("llm.token_count.prompt"), s["attrs"].get("llm.token_count.completion"))
                if s in llm.get(key, []):
                    llm[key].remove(s)

    for s in spans:
        if oi.kind(s) == "LLM" and s["status"] == "failed" and s["id"] not in used_spans:
            agent_span = by_id.get(s["parent"])
            while agent_span is not None and oi.kind(agent_span) != "AGENT":
                agent_span = by_id.get(agent_span["parent"])
            name = agent_span and agent_span["attrs"].get("agent.name")
            parent = agent_span and invocations.get(agent_span["attrs"].get("agno.run.id"))
            call = rec.step(s["id"], PAI.ModelCall, parent, "model request", s["start"], s["end"], s["status"], s["error"])
            rec.link(call, PROV.wasAssociatedWith, rec.agent(name))
            model_call(call, s, name)

    oi.link_tool_calls(rec, model_steps, tool_steps)
    # an answer passed on inside a later prompt is matched by its text, at least 16 characters long
    for started, inp, value in inputs:
        for done, out, text in outputs:
            if done <= started and len(text) >= 16 and text in value:
                rec.derived(inp, out, "inferred")
    leftover = [s["name"] for group in list(llm.values()) + list(tools.values()) for s in group]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native event: {leftover}")
    return rec
