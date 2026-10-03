import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

SKIP = {"mapping", "workflow_conditional_eval", "model_generation"}
STATUS = {"success": "completed", "failed": "failed", "suspended": "interrupted", "canceled": "cancelled"}


def flat(messages):
    # the visible text of the messages a model step read, without the reasoning around tool calls
    out = []
    for m in messages or []:
        c = m.get("content")
        out.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
    return "\n".join(out)


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    flow = next((e for e in ev if e["kind"] == "workflow"), None)
    if flow is None:
        raise SystemExit("the run has no workflow definition")
    native = {e["id"]: e for e in ev if e["kind"] == "span"}
    usage = [e for e in ev if e["kind"] == "model_step"]
    spans = read(run_dir / "spans.jsonl")
    for s in spans:
        s["sid"] = s["id"].removeprefix("0x")
        if s["sid"] not in native:
            raise SystemExit(f"span {s['name']} has no matching native span")
        s["native"] = native[s["sid"]]
    by_id = {s["id"]: s for s in spans}
    if len(spans) != len(native):
        raise SystemExit(f"{len(native) - len(spans)} native spans had no OpenTelemetry span")

    rec = Record(f"mastra @mastra/core {v['@mastra/core']} with @mastra/otel-exporter {v['@mastra/otel-exporter']}",
                 head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    branch_agents, conditionals = {}, []
    for node in flow["definition"]:
        for x in node.get("steps") or []:
            branch_agents[x["step"]["id"]] = (x["step"].get("metadata") or {}).get("agent")
        if node.get("type") == "conditional":
            conditionals.append([x["step"]["id"] for x in node["steps"]])
    scope = (flow.get("memory") or {}).get("scope")

    steps, runs_of, invocations = {}, Counter(), Counter()
    answers, tool_steps, failed = [], defaultdict(list), {}
    requested, generations = {}, defaultdict(list)
    approval, flow_steps, said = None, [], []

    def up(span, test):
        while span:
            if test(span):
                return span
            span = by_id.get(span["parent"])

    def kind(span):
        return span["native"]["type"]

    def agent_of(span):
        a = up(span, lambda x: kind(x) == "agent_run")
        return a and a["native"]["entity_name"]

    def contained(entity, text, start, owner):
        for end, ent, answer, who in answers:
            if who != owner and end <= seconds(start) and len(answer) >= 16 and answer in text:
                rec.derived(entity, ent, "inferred")

    def judged(span, at, by):
        # the answer whose text the step that declared a check got as input, else the latest answer of another agent
        step = up(span, lambda x: kind(x) == "workflow_step")
        given = step and step["native"].get("input")
        values = {v.strip() for v in (given.values() if isinstance(given, dict) else []) if isinstance(v, str)}
        t = seconds(at)
        return latest([(x[0], x[1]) for x in said if x[2] in values], t) or \
            latest([(x[0], x[1]) for x in said if x[3] != by], t)

    def agent_run(span):
        run = up(span, lambda x: kind(x) == "agent_run")
        if run is None:
            raise SystemExit(f"span {span['name']} runs outside an agent run")
        return run

    def agents_in(step_span):
        return [x["native"]["entity_name"] for x in spans if kind(x) == "agent_run"
                and up(by_id.get(x["parent"]), lambda y: kind(y) == "workflow_step") is step_span]

    def cost_of(agent, inference):
        u = inference["attributes"].get("usage") or {}
        for e in usage:
            got = e.get("usage") or {}
            if e["agent"] == agent and got.get("inputTokens") == u.get("inputTokens") and \
                    got.get("outputTokens") == u.get("outputTokens"):
                usage.remove(e)
                return (((e.get("provider_metadata") or {}).get("openrouter") or {}).get("usage") or {}).get("cost")

    for s in spans:
        n, a, sid = s["native"], s["attrs"], s["id"]
        t = n["type"]
        kept = up(by_id.get(s["parent"]), lambda x: x["id"] in steps)
        up_step = kept and steps[kept["id"]]
        if t in SKIP:
            continue

        if t == "workflow_run":
            step = rec.step(sid, PAI.Step, up_step, s["name"], s["start"], s["end"])
            rec.set(step, PAI.status, STATUS.get((n["attributes"] or {}).get("status"), "completed"))

        elif t == "workflow_step":
            name = n["entity_id"]
            runs_of[name] += 1
            status = STATUS.get((n["attributes"] or {}).get("status"), "completed")
            step = rec.step(sid, PAI.Step, up_step, f"step {name}", s["start"], s["end"], status, s["error"])
            rec.set(step, PAI.iteration, runs_of[name])
            if name in branch_agents:
                rec.set(step, PAI.branch, name)
                rec.link(step, PROV.wasAssociatedWith, rec.agent(branch_agents[name]))
            flow_steps.append((s, step))

        elif t == "workflow_conditional":
            step = rec.step(sid, PAI.Routing, up_step, "routing", s["start"], s["end"], "completed")
            # the step that ran just before the branch decided it, through the agent that ran inside it
            before = [(x, st) for x, st in flow_steps
                      if x["parent"] == s["parent"] and seconds(x["end"]) <= seconds(s["start"])]
            if not before:
                raise SystemExit(f"no workflow step ran before the branch {sid}")
            decided, decided_step = before[-1]
            router = rec.agent((agents_in(decided) or [decided["native"]["entity_id"]])[-1])
            rec.link(step, PROV.wasAssociatedWith, router)
            rec.link(step, PROV.wasInformedBy, decided_step)
            rec.link(router, PROV.actedOnBehalfOf, user)
            # the candidates are the branches of this conditional in the workflow definition
            own = next((c for c in conditionals if set(n["attributes"]["selectedSteps"]) <= set(c)), None)
            if own is None:
                raise SystemExit(f"no conditional of the workflow has the branches selected at {sid}")
            for x in own:
                rec.link(step, PAI.candidate, rec.agent(branch_agents[x]))
            for x in n["attributes"]["selectedSteps"]:
                rec.link(step, PAI.selected, rec.agent(branch_agents[x]))
                rec.link(rec.agent(branch_agents[x]), PROV.actedOnBehalfOf, router)
            rec.set(step, PAI.reason, (n.get("input") or {}).get("reason"))

        elif t == "workflow_loop":
            step = rec.step(sid, PAI.Step, up_step, f"loop {n['attributes'].get('loopType')}", s["start"], s["end"],
                            "completed")

        elif t == "agent_run":
            agent = n["entity_name"]
            invocations[agent] += 1
            step = rec.step(sid, PAI.AgentInvocation, up_step, agent, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, invocations[agent])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))

        elif t == "model_step":
            run = agent_run(s)["id"]
            step = rec.step(sid, PAI.Step, steps.get(run),
                            f"agent step {n['attributes'].get('stepIndex')}", s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, n["attributes"].get("stepIndex", 0) + 1)
            agent = agent_of(s)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            inp = rec.entity(f"{sid}-input", PAI.Message, "model input", json.dumps(n.get("input"), ensure_ascii=False))
            out = rec.entity(f"{sid}-output", PAI.Message, "model output", json.dumps(n.get("output"), ensure_ascii=False))
            contained(inp, flat(n.get("input")), s["start"], run)
            calls = (n.get("output") or {}).get("toolCalls") or []
            generations[run].append((s, inp, out, [c["toolCallId"] for c in calls]))
            for c in calls:
                requested[c["toolCallId"]] = out
            text = ((n.get("output") or {}).get("text") or "").strip()
            if text and not calls:
                answers.append((seconds(s["end"]), out, text, run))
                said.append((seconds(s["end"]), out, text, agent))

        elif t == "model_inference":
            step_span = by_id[s["parent"]]
            step = rec.step(sid, PAI.ModelCall, steps[step_span["id"]], "chat", s["start"], s["end"], s["status"],
                            s["error"])
            agent = agent_of(s)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            og.model_call(rec, step, {k: v for k, v in a.items() if k != "gen_ai.output.messages"})
            # the call reads and writes the messages of its model step; its own span leaves the input out
            rec.used(step, rec.iri("entity", f"{step_span['id']}-input"))
            rec.generated(rec.iri("entity", f"{step_span['id']}-output"), step)
            cost = cost_of(agent, n)
            rec.set(step, PAI.cost, None if cost is None else float(cost))

        elif t == "tool_call":
            tool = n["entity_name"]
            agent = agent_of(s)
            # the exported span says unknown; Mastra's own span keeps the error's name
            error = (n.get("error") or {}).get("name") or s["error"]
            step = rec.step(sid, PAI.ToolCall, up_step, tool, s["start"], s["end"], s["status"], error)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            args = json.dumps(n.get("input"), ensure_ascii=False)
            output = n.get("output")
            og.tool_call(rec, step, {"gen_ai.tool.name": tool}, args,
                         None if output is None else (output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)))
            cid = n["attributes"].get("toolCallId")
            tool_steps[cid].append(step)
            rec.used(step, requested.get(cid))
            if tool.startswith("agent-"):
                # a supervisor delegates to a subagent through a tool named after it
                rec.g.add((step, RDF.type, PAI.Handoff))
                src, dst = rec.agent(agent), rec.agent(tool.removeprefix("agent-"))
                rec.link(step, PAI.fromAgent, src)
                rec.link(step, PAI.toAgent, dst)
                rec.link(dst, PROV.actedOnBehalfOf, src)
            key, attempt = (agent, tool, same(args)), 1
            if key in failed:
                prior, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, prior)
                rec.set(step, PAI.attempt, attempt)
            if s["status"] == "failed":
                failed[key] = (step, attempt)

        elif t == "memory_operation":
            step = rec.step(sid, PAI.MemoryAccess, up_step, s["name"], s["start"], s["end"], s["status"], s["error"])
            steps[sid] = step
            continue

        elif t == "generic":
            data = n.get("output") or {}
            at = s["start"]
            if n["name"] == "memory_write":
                access = next((steps[x["id"]] for x in spans if x["parent"] == s["parent"] and kind(x) == "memory_operation"),
                              None)
                if access is None:
                    raise SystemExit(f"no memory operation span for the declared write {sid}")
                rec.link(access, PROV.wasAssociatedWith, rec.agent(data["by"]))
                record = rec.memory_write(data["memory"], data["record"], data.get("value"), access)
                rec.set(record, PAI.scope, scope)
                # the writer's latest answer with the written text, ended by the write
                source = latest([(x[0], x[1]) for x in said
                                 if x[3] == data["by"] and x[2] == (data.get("value") or "").strip()], seconds(at))
                rec.derived(record, source, "inferred")
                continue
            if n["name"] == "memory_read":
                step = rec.step(sid, PAI.MemoryAccess, up_step, "memory read", at, s["end"], "completed")
                rec.link(step, PROV.wasAssociatedWith, rec.agent(data["by"]))
                record = rec.memory_read(data["memory"], data.get("record"), step)
                if record is not None:
                    tool = by_id.get(s["parent"])
                    if tool is not None and tool["native"].get("output") == str(rec.g.value(record, PROV.value)):
                        rec.derived(rec.iri("entity", f"{tool['id']}-result"), record, "inferred")
            elif n["name"] in ("check", "approval"):
                step = rec.step(sid, PAI.Check, up_step, n["name"], at, s["end"], "completed")
                rec.link(step, PROV.wasAssociatedWith, approver if n["name"] == "approval" else rec.agent(data["by"]))
                rec.link(step, PAI.checked, judged(s, at, data["by"]))
                rec.set(step, PAI.outcome, data["outcome"])
                rec.set(step, PAI.reason, data.get("reason"))
                rec.set(step, PAI.iteration, data.get("iteration"))
                if n["name"] == "approval":
                    approval = (step, up(s, lambda x: kind(x) == "workflow_step"))
            elif n["name"] == "effect":
                effect = rec.entity(f"{sid}-effect-{at}", PAI.Effect, "effect", data["target"])
                rec.set(effect, PAI.target, data["target"])
                rec.generated(effect, up_step)
                rec.link(effect, PROV.wasAttributedTo, rec.agent(data["by"]))
                continue
            else:
                raise SystemExit(f"declared span {n['name']} has no mapping")

        else:
            raise SystemExit(f"span type {t} has no mapping")
        steps[sid] = step

    # the next model step of an agent run reads the results of the tools the previous step called
    for turns in generations.values():
        for (_, _, _, called), (s1, _, _, _) in zip(turns, turns[1:]):
            if not called:
                continue
            model = next((x for x in spans if x["parent"] == s1["id"] and kind(x) == "model_inference"), None)
            if model is None:
                raise SystemExit(f"no model call inside the model step {s1['id']}")
            for cid in called:
                for tool in tool_steps.get(cid, []):
                    rec.used(steps[model["id"]], rec.iri("entity", f"{rec.local(tool)}-result"))
    # a subagent's answer is the result of the delegation tool that ran it
    for s in spans:
        if kind(s) == "tool_call" and s["native"]["entity_name"].startswith("agent-"):
            text = ((s["native"].get("output") or {}).get("text") or "").strip()
            for _, out, answer, run in answers:
                if up(by_id[run], lambda x: x["id"] == s["id"]) and answer == text:
                    rec.derived(rec.iri("entity", f"{s['id']}-result"), out, "inferred")
    # the person's approval lets the next step of the workflow and its tool calls run
    if approval and approval[1] is not None:
        check, held = approval
        later = [(x, st) for x, st in flow_steps
                 if x["parent"] == held["parent"] and seconds(x["start"]) >= seconds(held["end"])]
        for x, send in later[:1]:
            rec.link(send, PROV.wasInformedBy, check)
            for s in spans:
                if kind(s) == "tool_call" and up(s, lambda y: y is x):
                    rec.link(steps[s["id"]], PROV.wasInformedBy, check)
    if usage:
        raise SystemExit(f"{len(usage)} model step events had no matching model call")
    return rec
