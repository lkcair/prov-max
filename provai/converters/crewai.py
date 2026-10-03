import ast
import json
import re
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

AGENT_RUNS = {"LiteAgentExecutionStartedEvent", "AgentExecutionStartedEvent"}
DELEGATION = "delegate_work_to_coworker"
MIN_MATCH = 16


def parse(value):
    if not isinstance(value, str):
        return value or {}
    for load in (json.loads, ast.literal_eval):
        try:
            return load(value)
        except (ValueError, SyntaxError):
            pass
    return {}


def texts(msgs):
    return " ".join(p.get("content", "") for m in og.messages(msgs) for p in m.get("parts", [])
                    if p.get("type") == "text")


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def convert(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = rows[0]
    flow_def = next((r for r in rows if r["kind"] == "flow"), None)
    if flow_def is None:
        raise SystemExit("the run has no flow definition")
    method_agents = flow_def["method_agents"]
    methods = flow_def["definition"]["methods"]
    ev = [r for r in rows if r["kind"] == "event"]
    by_id = {e["event_id"]: e for e in ev}
    end_of, error_of = {}, {}
    for e in ev:
        if e.get("started_event_id"):
            if e["event_class"].endswith(("ErrorEvent", "FailedEvent")):
                error_of[e["started_event_id"]] = e
            elif e["event_class"].endswith(("CompletedEvent", "FinishedEvent")):
                end_of[e["started_event_id"]] = e
    span_of, scope_of, id_of = {}, {}, {}
    for s in read(run_dir / "spans.jsonl"):
        span_of[s["attrs"].get("event_id")] = s
        for hit in json.loads(s["attrs"].get("crewai.memory.results") or "[]"):
            scope_of[norm(hit["record"]["content"])] = hit["record"]["scope"].strip("/")
            id_of[norm(hit["record"]["content"])] = hit["record"]["id"]
    used_spans = set()

    v = head["versions"]
    rec = Record(f"crewai {v['crewai']} with its built-in OpenTelemetry tracing", head["run_id"], True)
    flow_id = next((s["attrs"]["crewai.flow.id"] for s in span_of.values() if s["attrs"].get("crewai.flow.id")), None)
    if flow_id:
        rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, flow_id, flow_id))
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    main_flow = next((e for e in ev if e["event_class"] == "FlowStartedEvent" and not e.get("parent_event_id")), None)
    if main_flow is None:
        raise SystemExit("the run has no top-level FlowStartedEvent")
    steps, agent_of_step, finished_method = {}, {}, {}
    runs = Counter()
    last_model, pending = {}, defaultdict(list)
    failed, sources, answers = {}, [], {}
    gated = set()
    model_inputs, tool_args = [], []

    def kept(eid):
        while eid and eid not in steps:
            eid = by_id.get(eid, {}).get("parent_event_id")
        return eid

    def end_time(eid):
        return (end_of.get(eid) or error_of.get(eid) or {}).get("timestamp")

    def inside(eid, ancestors):
        while eid:
            if eid in ancestors:
                return True
            eid = by_id.get(eid, {}).get("parent_event_id")
        return False

    def span(e):
        s = span_of.get(e["event_id"])
        if s:
            used_spans.add(e["event_id"])
        return s

    def role_inside(eid):
        for e in ev:
            if e.get("agent_role") and e["event_id"] != eid:
                p = e.get("parent_event_id")
                while p:
                    if p == eid:
                        return e["agent_role"]
                    p = by_id.get(p, {}).get("parent_event_id")

    def candidates(router):
        found = []
        for label in methods[router].get("emit") or []:
            for name, m in methods.items():
                if m.get("listen") == label and method_agents.get(name):
                    found.append((label, method_agents[name]))
        return found

    def trigger_agent(method):
        listen = methods[method].get("listen")
        return method_agents.get(method) or (method_agents.get(listen) if isinstance(listen, str) else None)

    def text_entity(entity, text, when):
        if entity is not None and len(norm(text)) >= MIN_MATCH:
            sources.append((entity, norm(text), when))

    for e in ev:
        cls, eid = e["event_class"], e["event_id"]
        parent = steps.get(kept(e.get("parent_event_id")))

        if cls == "FlowStartedEvent" and e is main_flow:
            steps[eid] = rec.step(eid, PAI.Step, None, f"flow {e['flow_name']}", e["timestamp"], end_time(eid),
                                  "completed")
            span(e)

        elif cls == "MethodExecutionStartedEvent":
            name = e["method_name"]
            runs[name] += 1
            options = candidates(name) if methods[name].get("router") else []
            step = rec.step(eid, PAI.Step, parent, name, e["timestamp"], end_time(eid),
                            "failed" if eid in error_of else "completed")
            rec.set(step, PAI.iteration, runs[name])
            rec.set(step, PAI.sequence, e.get("emission_sequence"))
            agent = rec.agent(trigger_agent(name))
            rec.link(step, PROV.wasAssociatedWith, agent)
            agent_of_step[step] = agent
            if len({a for _, a in options}) > 1:
                rec.g.add((step, RDF.type, PAI.Routing))
                for _, a in options:
                    rec.link(step, PAI.candidate, rec.agent(a))
                label = (end_of.get(eid) or {}).get("result")
                for chosen_label, a in options:
                    if chosen_label == label:
                        rec.link(step, PAI.selected, rec.agent(a))
                        rec.link(rec.agent(a), PROV.actedOnBehalfOf, agent)
                rec.link(agent, PROV.actedOnBehalfOf, user)
            trigger = finished_method.get(e.get("triggered_by_event_id"))
            rec.link(step, PROV.wasInformedBy, trigger)
            if end_of.get(eid):
                finished_method[end_of[eid]["event_id"]] = step
            steps[eid] = step
            span(e)

        elif cls in AGENT_RUNS:
            role = (re.search(r"'role': '([^']+)'", str(e.get("agent_info"))) or [None, None])[1] or role_inside(eid)
            runs[role] += 1
            step = rec.step(eid, PAI.AgentInvocation, parent, role, e["timestamp"], end_time(eid),
                            "failed" if eid in error_of else "completed")
            rec.set(step, PAI.iteration, runs[role])
            agent = rec.agent(role)
            rec.link(step, PROV.wasAssociatedWith, agent)
            agent_of_step[step] = agent
            output = (end_of.get(eid) or {}).get("output")
            if output:
                answer = rec.entity(f"{eid}-answer", PROV.Entity, f"{role} answer", output)
                rec.generated(answer, step)
                answers[eid] = (answer, role, output)
                text_entity(answer, output, end_time(eid))
            steps[eid] = step
            span(e)

        elif cls in ("CrewKickoffStartedEvent", "TaskStartedEvent"):
            label = "crew" if cls.startswith("Crew") else f"task: {str(e.get('task_name'))[:60]}"
            steps[eid] = rec.step(eid, PAI.Step, parent, label, e["timestamp"], end_time(eid),
                                  "failed" if eid in error_of else "completed")
            span(e)

        elif cls == "LLMCallStartedEvent":
            s = span(e)
            step = rec.step(eid, PAI.ModelCall, parent, "chat", e["timestamp"], end_time(eid),
                            "failed" if eid in error_of else "completed")
            agent = rec.agent(e.get("agent_role"))
            rec.link(step, PROV.wasAssociatedWith, agent)
            loop = e.get("parent_event_id")
            for res in pending.pop(loop, []):
                rec.used(step, res)
            inp, out = og.model_call(rec, step, s["attrs"]) if s else ([], [])
            sid = rec.local(step)
            out_entity = rec.iri("entity", f"{sid}-output") if out else None
            last_model[loop] = out_entity
            if inp:
                model_inputs.append((rec.iri("entity", f"{sid}-input"), norm(texts(inp)), e["timestamp"]))
            text_entity(out_entity, texts(out), end_time(eid))
            answer_text = norm(texts(out))
            for answer, role, output in answers.values():
                if norm(output) == answer_text and role == e.get("agent_role"):
                    rec.derived(answer, out_entity, "inferred")
            steps[eid] = step

        elif cls == "ToolUsageStartedEvent":
            s = span(e)
            name = e["tool_name"]
            err = error_of.get(eid)
            kind = err and re.match(r"\w+", err.get("error") or "")
            step = rec.step(eid, PAI.ToolCall, parent, name, e["timestamp"], end_time(eid),
                            "failed" if err else "completed", kind.group(0) if kind else err and err.get("error"))
            role = e.get("agent_role")
            agent = rec.agent(role)
            rec.link(step, PROV.wasAssociatedWith, agent)
            agent_of_step[step] = agent
            done = end_of.get(eid) or {}
            args = parse(e.get("tool_args"))
            og.tool_call(rec, step, {"gen_ai.tool.name": name}, json.dumps(args, sort_keys=True), done.get("output"))
            sid = rec.local(step)
            args_entity = rec.iri("entity", f"{sid}-arguments")
            result = rec.iri("entity", f"{sid}-result") if done.get("output") is not None else None
            loop = e.get("parent_event_id")
            rec.used(step, last_model.get(loop))
            if result is not None:
                pending[loop].append(result)
                text_entity(result, done.get("output"), end_time(eid))
            tool_args.append((args_entity, norm(json.dumps(args)), e["timestamp"]))
            key = (role, name, same(args))
            attempt = 1
            if key in failed:
                previous, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if err:
                failed[key] = (step, attempt)
                for f in ev:
                    if f["event_class"] == "ToolFailureDetectedEvent" and f.get("parent_event_id") == loop \
                            and f.get("tool_name") == name and f["event_id"] in span_of:
                        used_spans.add(f["event_id"])
            if name == DELEGATION:
                rec.g.add((step, RDF.type, PAI.Handoff))
                coworker = rec.agent(args.get("coworker"))
                rec.link(step, PAI.fromAgent, agent)
                rec.link(step, PAI.toAgent, coworker)
                rec.link(coworker, PROV.actedOnBehalfOf, agent)
            # a tool call in a method that a person's approval started is the action the person approved
            if not err and done.get("output") and inside(eid, gated):
                effect = rec.entity(f"{eid}-effect-{e['timestamp']}", PAI.Effect, "effect", done["output"])
                rec.set(effect, PAI.target, done["output"])
                rec.generated(effect, step)
                rec.link(effect, PROV.wasAttributedTo, agent)
            steps[eid] = step

        elif cls in ("MemorySaveStartedEvent", "MemoryQueryStartedEvent"):
            s = span(e)
            write = cls == "MemorySaveStartedEvent"
            done = end_of.get(eid) or {}
            access = rec.step(eid, PAI.MemoryAccess, parent, "memory write" if write else "memory read",
                              e["timestamp"], end_time(eid), "failed" if eid in error_of else "completed")
            agent = rec.agent(done.get("agent_role")) if write else agent_of_step.get(parent)
            rec.link(access, PROV.wasAssociatedWith, agent)
            if write:
                record = rec.entity(id_of.get(norm(e["value"]), eid), PAI.MemoryRecord,
                                    parse(e.get("metadata")).get("key"), e["value"])
                rec.generated(record, access)
                # a written value comes from the latest answer of its writer with the same text
                source = latest([(seconds(end_time(a_eid)), answer) for a_eid, (answer, role, output) in answers.items()
                                 if role == done.get("agent_role") and norm(output) == norm(e["value"])
                                 and end_time(a_eid)], seconds(e["timestamp"]))
                if source is not None:
                    rec.derived(record, source, "inferred")
                name = scope_of.get(norm(e["value"]), "memory")
                rec.link(rec.node("memory", PAI.Memory, name, name), PROV.hadMember, record)
                text_entity(record, e["value"], end_time(eid))
            else:
                results = json.loads((s or {}).get("attrs", {}).get("crewai.memory.results") or "[]")
                for hit in results:
                    r = hit["record"]
                    record = rec.iri("entity", r["id"])
                    memory = rec.node("memory", PAI.Memory, r["scope"].strip("/"), r["scope"].strip("/"))
                    rec.link(memory, PROV.hadMember, record)
                    rec.used(access, record)
                    tool = parent
                    out = rec.iri("entity", f"{rec.local(tool)}-result")
                    if (out, PROV.value, None) in rec.g and norm(rec.g.value(out, PROV.value)) == norm(r["content"]):
                        rec.derived(out, record, "inferred")
            steps[eid] = access

        elif cls == "CheckEvent":
            check = rec.step(eid, PAI.Check, parent, "check", e["timestamp"], e["timestamp"], "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            # the latest answer before the check by an agent other than the checker
            rec.link(check, PAI.checked, latest([(seconds(end_time(a_eid)), answer)
                                                 for a_eid, (answer, role, _) in answers.items()
                                                 if role != e["by"] and end_time(a_eid)], seconds(e["timestamp"])))
            rec.set(check, PAI.iteration, int(e["iteration"]) if e.get("iteration") is not None else None)
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))

        elif cls == "HumanFeedbackRequestedEvent":
            span(e)
            reviewed = rec.entity(f"{eid}-output", PROV.Entity, f"{e['method_name']} output", e.get("output"))
            rec.generated(reviewed, parent)
            for answer, role, output in answers.values():
                if norm(output) == norm(e.get("output")):
                    rec.derived(reviewed, answer, "inferred")
            text_entity(reviewed, e.get("output"), e["timestamp"])

        elif cls == "HumanFeedbackReceivedEvent":
            span(e)
            check = rec.step(eid, PAI.Check, parent, "approval", e["timestamp"], e["timestamp"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver)
            asked = next((f for f in ev if f["event_class"] == "HumanFeedbackRequestedEvent"
                          and f.get("parent_event_id") == e.get("parent_event_id")), None)
            if asked is None:
                raise SystemExit("human feedback was received without a request")
            rec.link(check, PAI.checked, rec.iri("entity", f"{asked['event_id']}-output"))
            # the flow goes on only when the feedback is "approve"; any other answer rejects the draft
            rec.set(check, PAI.outcome, "approve" if (e.get("feedback") or "").strip().lower() == "approve" else "reject")
            finished = end_of.get(e.get("parent_event_id"))
            if finished:
                for later in ev:
                    if later["event_class"] == "MethodExecutionStartedEvent" and \
                            later.get("triggered_by_event_id") == finished["event_id"]:
                        rec.link(rec.iri("step", later["event_id"]), PROV.wasInformedBy, check)
                        gated.add(later["event_id"])

    # what a model read or a tool was given can contain earlier outputs; matched by content, so inferred
    for target, text, when in model_inputs + tool_args:
        for source, piece, made in sources:
            if piece in text and made and seconds(made) <= seconds(when):
                rec.derived(target, source, "inferred")

    leftover = [s["name"] for k, s in span_of.items() if k not in used_spans]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native event: {leftover}")
    return rec
