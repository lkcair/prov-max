import json
from collections import defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import openinference as oi

# Haystack internals that OpenInference traces and Haystack's own tracer does not
PLUMBING = {"ChatPromptBuilder.run", "ChatPromptBuilder.run_async", "Pipeline.run_async_generator"}
OUTCOME = {"confirm": "approve", "modify": "edit", "reject": "reject"}


def text(message):
    return " ".join(c.get("text", "") for c in (message or {}).get("content", []) if "text" in c).strip()


def calls(message):
    return [c["tool_call"] for c in (message or {}).get("content", []) if "tool_call" in c]


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    pipe = next((e for e in ev if e["kind"] == "pipeline"), None)
    if pipe is None:
        raise SystemExit("the run has no pipeline definition")
    edges, models = pipe["definition"]["edges"], pipe["models"]
    native = {e["id"]: e for e in ev if e["kind"] == "span"}

    def depth(e):
        n = 0
        while e["parent"] in native:
            e, n = native[e["parent"]], n + 1
        return n
    spans = sorted(native.values(), key=lambda e: (seconds(e["start"]), depth(e)))
    for s in spans:
        s["otel_span"] = "0x" + s["otel_span"]
    by_id = {s["id"]: s for s in spans}
    otel = read(run_dir / "spans.jsonl")
    used_otel = set()

    rec = Record(f"haystack {v['haystack-ai']} with openinference-instrumentation-haystack "
                 f"{v['openinference-instrumentation-haystack']}", head["run_id"], True)
    oi.session(rec, otel)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    steps, component_of = {}, {}
    latest_output, last_answer, answer_of_agent = {}, {}, {}
    model_steps, tool_steps = [], defaultdict(list)
    requested, failed, approvals = {}, {}, []
    final_answer, answers = {}, []
    chains = defaultdict(list)
    for s in otel:
        if oi.kind(s) == "CHAIN" and s["name"] not in PLUMBING:
            chains[s["name"].rsplit(".", 1)[0]].append(s)
        if s["name"] in PLUMBING:
            used_otel.add(s["id"])

    def visit_of_span(sid):
        while sid in by_id and by_id[sid]["name"] != "haystack.component.run":
            sid = by_id[sid]["parent"]
        return sid if sid in by_id else None

    def component_run(sid):
        visit = visit_of_span(sid)
        return by_id[visit]["tags"] if visit else None

    def oi_text(oi_span):
        return " ".join(str(m.get("content", "")) for m in oi.indexed(oi_span["attrs"], "llm.output_messages")).strip()

    visit_of = {}

    # what the application declared from inside each component: the agent it acts for, a routing, a handoff
    declared_by, declared = defaultdict(set), defaultdict(set)
    types = {}
    for s in spans:
        if s["name"] == "haystack.component.run":
            types[s["tags"]["haystack.component.name"]] = s["tags"]["haystack.component.type"]
        elif s["name"].startswith("app.") and component_run(s["id"]):
            component = component_run(s["id"])["haystack.component.name"]
            declared[component].add(s["name"][4:])
            if s["name"] != "app.approval" and s["tags"].get("by"):
                declared_by[component].add(s["tags"]["by"])

    def agent_name(component):
        if len(declared_by[component]) == 1:
            return next(iter(declared_by[component]))
        return component if types.get(component, "Agent") == "Agent" else None

    def agent(component):
        return rec.agent(agent_name(component) or component)

    def model_call(sid, parent, component, oi_span, native_output=None, native_input=None):
        step = rec.step(sid, PAI.ModelCall, parent, "chat", oi_span["start"], oi_span["end"],
                        oi_span["status"], oi_span["error"])
        rec.link(step, PROV.wasAssociatedWith, agent(component))
        _, out = oi.model_call(rec, step, oi_span["attrs"])
        settings = models.get(component) or {}
        rec.set(step, PAI.temperature, settings.get("temperature"))
        rec.set(step, PAI.maxTokens, settings.get("max_tokens"))
        used_otel.add(oi_span["id"])
        replies = (native_output or {}).get("replies") or []
        if replies:
            cost = ((replies[0].get("meta") or {}).get("usage") or {}).get("cost")
            rec.set(step, PAI.cost, None if cost is None else float(cost))
            rec.set(step, PAI.finishReason, (replies[0].get("meta") or {}).get("finish_reason"))
        made = [c for r in replies for c in calls(r)]
        read_ids = [r["tool_call_result"]["origin"]["id"] for m in (native_input or {}).get("messages", [])
                    for r in m.get("content", []) if "tool_call_result" in r]
        model_steps.append((step, out, [c["id"] for c in made], read_ids))
        answer = " ".join(text(r) for r in replies).strip() or oi_text(oi_span)
        if answer and not made:
            last_answer[component] = (rec.iri("entity", f"{sid}-output"), answer)
            answer_of_agent[agent_name(component) or component] = last_answer[component]
            answers.append((seconds(oi_span["end"]), agent_name(component) or component, last_answer[component][0]))
            visit = visit_of.get(sid)
            if visit:
                final_answer[visit] = last_answer[component]
        return step, made

    for s in spans:
        name, tags, sid = s["name"], s["tags"], s["id"]
        parent = steps.get(s["parent"])
        if name == "haystack.pipeline.run":
            steps[sid] = rec.step(sid, PAI.Step, None, "pipeline run", s["start"], s["end"],
                                  "failed" if s["error"] else "completed", s["error"])
            used_otel.update(o["id"] for o in otel if o["name"] == "Pipeline.run_async")

        elif name == "haystack.component.run":
            component, kind = tags["haystack.component.name"], tags["haystack.component.type"]
            routes, acts = "routing" in declared[component], agent_name(component) is not None
            cls = PAI.Routing if routes else PAI.Handoff if "handoff" in declared[component] else \
                PAI.AgentInvocation if acts else PAI.Step
            step = rec.step(sid, cls, parent, component, s["start"], s["end"],
                            "failed" if s["error"] else "completed", s["error"])
            rec.set(step, PAI.iteration, int(tags["haystack.component.visits"]))
            if acts:
                rec.link(step, PROV.wasAssociatedWith, agent(component))
            if routes:
                rec.g.add((step, RDF.type, PAI.AgentInvocation))
                for u, target, _, _ in edges:
                    if u == component:
                        rec.link(step, PAI.candidate, agent(target))
                for target in tags.get("haystack.component.output") or {}:
                    rec.link(step, PAI.selected, agent(target))
                    rec.link(agent(target), PROV.actedOnBehalfOf, agent(component))
                rec.link(agent(component), PROV.actedOnBehalfOf, user)
            # data flow: a component run uses the latest output of each component connected to it
            for u, target, _, _ in edges:
                if target == component and u in latest_output:
                    rec.used(step, latest_output[u])
            out = rec.entity(f"{sid}-output", PROV.Entity, f"{component} output",
                             json.dumps(tags.get("haystack.component.output"), default=str))
            rec.generated(out, step)
            latest_output[component] = out
            steps[sid] = step
            component_of[sid] = component
            if kind != "Agent":
                if not chains[kind]:
                    raise SystemExit(f"no OpenInference span for the {component} run")
                chain = chains[kind].pop(0)
                used_otel.add(chain["id"])
                # model calls a custom component makes itself, seen only by OpenInference
                for o in otel:
                    if o["parent"] == chain["id"] and oi.kind(o) == "LLM":
                        model_call(o["id"], step, component, o)

        elif name == "haystack.agent.run":
            steps[sid], component_of[sid] = parent, component_of[s["parent"]]
            used_otel.add(s["otel_span"])
            chains["Agent"] = [c for c in chains["Agent"] if c["id"] != s["otel_span"]]

        elif name == "haystack.agent.step":
            component = component_of[s["parent"]]
            step = rec.step(sid, PAI.Step, parent, f"step {tags['haystack.agent.step']}", s["start"], s["end"],
                            "failed" if s["error"] else "completed", s["error"])
            rec.set(step, PAI.iteration, int(tags["haystack.agent.step"]) + 1)
            steps[sid], component_of[sid] = step, component

        elif name == "haystack.agent.step.llm":
            component = component_of[s["parent"]]
            visit_of[sid] = visit_of_span(sid)
            run = by_id[s["parent"]]["parent"]
            oi_span = next((o for o in otel if o["parent"] == by_id[run]["otel_span"] and oi.kind(o) == "LLM"
                            and o["id"] not in used_otel), None)
            if oi_span is None:
                raise SystemExit(f"no OpenInference model span for the agent step {sid}")
            _, made = model_call(sid, parent, component, oi_span, tags.get("haystack.agent.step.llm.output"),
                                 tags.get("haystack.agent.step.llm.input"))
            requested[s["parent"]] = made
            component_of[sid] = component

        elif name == "haystack.agent.step.tool":
            component = component_of[s["parent"]]
            tool, arguments = tags["haystack.tool.name"], tags.get("haystack.agent.step.tool.input")
            result = tags.get("haystack.agent.step.tool.output")
            error = result.get("error") if isinstance(result, dict) else None
            step = rec.step(sid, PAI.ToolCall, parent, tool, s["start"], s["end"], "failed" if error else "completed",
                            error)
            rec.link(step, PROV.wasAssociatedWith, agent(component))
            rec.link(step, PAI.usedTool, rec.tool(tool))
            args = rec.entity(f"{sid}-arguments", PROV.Entity, f"{tool} arguments", same(arguments))
            rec.used(step, args)
            res = rec.entity(f"{sid}-result", PROV.Entity, f"{tool} result", str(result))
            rec.generated(res, step)
            match = next((c for c in requested.get(s["parent"], []) if c["tool_name"] == tool
                          and same(c["arguments"]) == same(arguments)), None)
            if match:
                requested[s["parent"]].remove(match)
                tool_steps[match["id"]].append((step, res))
            key, attempt = (component, tool, same(arguments)), 1
            if key in failed:
                prior, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, prior)
                rec.set(step, PAI.attempt, attempt)
            if error:
                failed[key] = (step, attempt)
            steps[sid], component_of[sid] = step, component

        elif name == "haystack.agent.hook":
            component = component_of[s["parent"]]
            step = rec.step(sid, PAI.Step, parent, f"{tags['haystack.agent.hook.point']} hook", s["start"], s["end"],
                            "failed" if s["error"] else "completed", s["error"])
            steps[sid], component_of[sid] = step, component

        elif name == "haystack.agent.hook.human_in_the_loop.strategy":
            check = rec.step(sid, PAI.Check, parent, "approval", s["start"], s["end"], "completed")
            rec.link(check, PROV.wasAssociatedWith, approver)
            decision = tags["haystack.agent.hook.human_in_the_loop.strategy.decision"]
            if decision not in OUTCOME:
                raise SystemExit(f"unknown confirmation decision {decision}")
            rec.set(check, PAI.outcome, OUTCOME[decision])
            approvals.append((check, tags["haystack.tool.call.id"]))

        elif name == "app.routing":
            rec.set(parent, PAI.reason, tags.get("reason"))

        elif name == "app.handoff":
            rec.link(parent, PAI.fromAgent, rec.agent(tags["source"]))
            rec.link(parent, PAI.toAgent, rec.agent(tags["target"]))
            rec.link(rec.agent(tags["target"]), PROV.actedOnBehalfOf, rec.agent(tags["source"]))

        elif name in ("app.memory_write", "app.memory_read"):
            access = rec.step(sid, PAI.MemoryAccess, parent, name[4:].replace("_", " "), s["start"], s["end"],
                              "completed")
            rec.link(access, PROV.wasAssociatedWith, rec.agent(tags["by"]))
            if name == "app.memory_write":
                record = rec.memory_write(tags["memory"], tags["key"], tags.get("value"), access)
                source = answer_of_agent.get(tags["by"])
                if source and source[1] == (tags.get("value") or "").strip():
                    rec.derived(record, source[0], "inferred")
            else:
                record = rec.memory_read(tags["memory"], tags["key"], access)
                tool = by_id[s["parent"]]
                value = rec.g.value(record, PROV.value) if record is not None else None
                if value is not None and str(tool["tags"].get("haystack.agent.step.tool.output")) == str(value):
                    rec.derived(rec.iri("entity", f"{tool['id']}-result"), record, "inferred")

        elif name == "app.check":
            check = rec.step(sid, PAI.Check, parent, "check", s["start"], s["end"], "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(tags["by"]))
            # the check judged the latest answer of an agent other than the one checking
            rec.link(check, PAI.checked, latest([(t, a) for t, who, a in answers if who != tags["by"]],
                                                seconds(s["start"])))
            rec.set(check, PAI.outcome, tags["outcome"])
            rec.set(check, PAI.reason, tags.get("reason"))
            rec.set(check, PAI.iteration, int(tags["iteration"]))

        elif name == "app.effect":
            effect = rec.entity(f"{sid}-effect-{s['start']}", PAI.Effect, "effect", tags["target"])
            rec.set(effect, PAI.target, tags["target"])
            rec.generated(effect, parent)
            rec.link(effect, PROV.wasAttributedTo, rec.agent(tags["by"]))

    # an agent component returns its final answer as last_message
    for visit, (answer, answer_text) in final_answer.items():
        output = by_id[visit]["tags"].get("haystack.component.output") or {}
        if text(output.get("last_message")) == answer_text:
            rec.derived(rec.iri("entity", f"{visit}-output"), answer, "inferred")

    # a person's decision on a held call comes before the call runs
    for check, cid in approvals:
        for step, _ in tool_steps.get(cid, []):
            rec.link(check, PAI.checked, rec.iri("entity", f"{rec.local(step)}-arguments"))
            rec.link(step, PROV.wasInformedBy, check)
    oi.link_tool_calls(rec, model_steps, tool_steps)
    leftover = [s["name"] for s in otel if s["id"] not in used_otel]
    if leftover:
        raise SystemExit(f"{len(leftover)} spans had no matching native span: {leftover}")
    return rec
