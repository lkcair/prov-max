import json
from collections import Counter, defaultdict

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds


def text_of(blocks):
    if isinstance(blocks, dict) and "content" in blocks:
        blocks = blocks["content"]
    if isinstance(blocks, list):
        return " ".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
    return None if blocks is None else str(blocks)


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    spans = read(run_dir / "spans.jsonl")
    logs = [json.loads(line) for line in open(run_dir / "otel_logs.jsonl")]
    by_id = {s["id"]: s for s in spans}

    rec = Record(f"claude-agent-sdk {v['claude-agent-sdk']} with Claude Code {v['claude-code']} OpenTelemetry traces",
                 head["run_id"], True)
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    messages = [e for e in ev if e["kind"] == "message"]
    main_agent, candidates = {}, {}
    for e in messages:
        if e.get("subtype") == "init":
            main_agent[e["data"]["session_id"]] = e["label"]
            candidates[e["data"]["session_id"]] = e["data"].get("agents") or []
    outputs = defaultdict(list)
    for e in messages:
        if e["type"] == "AssistantMessage":
            outputs[e["message_id"]] += [b for b in e["content"] if "thinking" not in b]
    cost = {log["attributes"]["request_id"]: log["attributes"] for log in logs if log["body"] == "claude_code.api_request"}
    responses = {log["attributes"]["request_id"]: log["attributes"]["response"] for log in logs
                 if log["body"] == "claude_code.assistant_response"}
    hooks = defaultdict(dict)
    subagent_type, subagent_time = {}, defaultdict(dict)
    for e in ev:
        if e["kind"] != "hook":
            continue
        name = e["hook_event_name"]
        if e.get("tool_use_id"):
            hooks[e["tool_use_id"]][name] = e
        if name in ("SubagentStart", "SubagentStop"):
            subagent_type[e["agent_id"]] = e["agent_type"]
            subagent_time[e["agent_id"]][name] = e["time"]
    declared = defaultdict(list)
    for e in ev:
        if e.get("tool_use_id") and e["kind"] in ("memory_write", "memory_read", "approval", "effect"):
            declared[e["tool_use_id"]].append(e)

    for sid in {s["attrs"].get("session.id") for s in spans} - {None}:
        rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, sid, sid))

    steps, tools, invocations = {}, {}, Counter()
    routing, subagent_step, requested = {}, {}, {}
    model_calls = defaultdict(list)
    previous = {}
    answers, agent_results, failed, memory_events = [], [], {}, []
    pending = set(hooks)

    def ancestor(s, name):
        while s:
            if s["name"] == name:
                return s
            s = by_id.get(s["parent"])

    def owner(s):
        # the agent a span belongs to: a subagent by its id, otherwise the main agent of the session
        agent_id = s["attrs"].get("agent_id")
        if agent_id:
            if agent_id not in subagent_type:
                raise SystemExit(f"no SubagentStart hook for the subagent {agent_id}")
            return agent_id, subagent_type[agent_id]
        session = s["attrs"]["session.id"]
        if session not in main_agent:
            raise SystemExit(f"no init message for session {session}")
        interaction = ancestor(s, "claude_code.interaction")
        # the CLI can make a model call for a session before the session's first interaction
        key = interaction["id"] if interaction else session
        return key, main_agent[session]

    def parent_step(s):
        agent_id = s["attrs"].get("agent_id")
        if agent_id:
            return subagent(s, agent_id)
        p = by_id.get(s["parent"])
        while p and p["id"] not in steps:
            p = by_id.get(p["parent"])
        return steps.get(p["id"]) if p else None

    def subagent(s, agent_id):
        if agent_id not in subagent_step:
            call = ancestor(s, "claude_code.tool")
            if call is None:
                raise SystemExit(f"subagent {agent_id} runs under no tool call")
            name = subagent_type[agent_id]
            invocations[name] += 1
            times = subagent_time[agent_id]
            step = rec.step(agent_id, PAI.AgentInvocation, steps[call["id"]], name, times.get("SubagentStart"),
                            times.get("SubagentStop"), "completed")
            rec.set(step, PAI.iteration, invocations[name])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(name))
            # the CLI starts each subagent afresh for the Agent call that asks for it
            rec.link(rec.agent(name), PAI.spawnedBy, steps[call["id"]])
            subagent_step[agent_id] = step
        return subagent_step[agent_id]

    for s in spans:
        a, sid, name = s["attrs"], s["id"], s["name"]
        if name == "claude_code.interaction":
            agent = main_agent[a["session.id"]]
            invocations[agent] += 1
            step = rec.step(sid, PAI.AgentInvocation, None, agent, s["start"], s["end"], "completed")
            rec.set(step, PAI.iteration, invocations[agent])
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            steps[sid] = step

        elif name == "claude_code.llm_request":
            key, agent = owner(s)
            step = rec.step(sid, PAI.ModelCall, parent_step(s), "llm_request", s["start"], s["end"],
                            "completed" if a.get("success") in (True, "true", "True") else "failed")
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            rec.link(step, PAI.usedModel, rec.model(a.get("gen_ai.request.model") or a.get("model"),
                                                     a.get("gen_ai.system")))
            rec.set(step, PAI.inputTokens, a.get("input_tokens"))
            rec.set(step, PAI.outputTokens, a.get("output_tokens"))
            reasons = a.get("gen_ai.response.finish_reasons")
            rec.set(step, PAI.finishReason, ",".join(reasons) if isinstance(reasons, list) else reasons)
            usage = cost.pop(a.get("request_id"), {})
            rec.set(step, PAI.cost, None if usage.get("cost_usd") is None else float(usage["cost_usd"]))
            # the stream carries tool calls and main-agent text; a subagent's final text reaches only the log
            response = responses.pop(a.get("gen_ai.response.id"), None)
            blocks = outputs.pop(a.get("gen_ai.response.id"), None) or ([{"text": response}] if response else [])
            out = None
            if blocks:
                out = rec.entity(f"{sid}-output", PAI.Message, "model output", json.dumps(blocks))
                rec.generated(out, step)
                for b in blocks:
                    if "name" in b:
                        requested[b["id"]] = out
                text = " ".join(b["text"] for b in blocks if b.get("text")).strip()
                if text and not any("name" in b for b in blocks):
                    answers.append((key, out, text))
            model_calls[key].append((seconds(s["start"]), step))
            # the CLI sends the whole transcript with every request, so each call reads the previous answer
            thread = a.get("agent_id") or a["session.id"]
            if thread in previous:
                rec.used(step, previous[thread])
            if out is not None:
                previous[thread] = out
            steps[sid] = step

        elif name == "claude_code.tool":
            key, agent = owner(s)
            tool, cid = a["tool_name"], a["gen_ai.tool.call.id"]
            run = [k for k in spans if k["parent"] == sid and k["name"] == "claude_code.tool.execution"]
            blocked = [k for k in spans if k["parent"] == sid and k["name"] == "claude_code.tool.blocked_on_user"]
            if run:
                status = "completed" if run[0]["attrs"].get("success") in (True, "true", "True") else "failed"
            else:
                status = "blocked" if blocked and blocked[0]["attrs"].get("decision") == "reject" else "failed"
            step = rec.step(sid, PAI.ToolCall, parent_step(s), tool, s["start"], s["end"], status,
                            run[0]["attrs"].get("error") if run and status == "failed" else None)
            rec.link(step, PROV.wasAssociatedWith, rec.agent(agent))
            rec.link(step, PAI.usedTool, rec.tool(tool))
            for k in run + blocked:
                steps[k["id"]] = step
            pending.discard(cid)
            h = hooks[cid]
            if "PreToolUse" not in h:
                raise SystemExit(f"no PreToolUse hook for tool call {cid}")
            args = rec.entity(f"{sid}-arguments", PROV.Entity, f"{tool} arguments", json.dumps(h["PreToolUse"]["tool_input"]))
            rec.used(step, args)
            rec.used(step, requested.get(cid))
            result = None
            if "PostToolUse" in h:
                result = rec.entity(f"{sid}-result", PROV.Entity, f"{tool} result", text_of(h["PostToolUse"]["tool_response"]))
            elif "PostToolUseFailure" in h:
                result = rec.entity(f"{sid}-result", PROV.Entity, f"{tool} result", h["PostToolUseFailure"]["error"])
            rec.generated(result, step)
            tools[cid] = (step, result, key, seconds(s["end"]))
            if tool in ("Agent", "Task"):
                # the agent turn that starts subagents chose them among the agents its session offers
                session = a["session.id"]
                if session not in routing:
                    turn = ancestor(s, "claude_code.interaction")
                    if turn is None:
                        raise SystemExit(f"Agent call {cid} runs outside an agent turn")
                    routing[session] = steps[turn["id"]]
                    rec.g.add((routing[session], RDF.type, PAI.Routing))
                    for c in candidates[session]:
                        rec.link(routing[session], PAI.candidate, rec.agent(c))
                selected = rec.agent(a.get("subagent_type"))
                rec.link(routing.get(a["session.id"]), PAI.selected, selected)
                rec.link(selected, PROV.actedOnBehalfOf, rec.agent(agent))
                rec.link(rec.agent(agent), PROV.actedOnBehalfOf, user)
                agent_results.append((sid, result, rec.g.value(result, PROV.value) if result is not None else None))
            # a failed call sends the model back; its next call of the same tool and arguments is the retry
            same_call = (agent, tool, same(h["PreToolUse"]["tool_input"]))
            attempt = 1
            if same_call in failed:
                prior, attempt = failed.pop(same_call)
                attempt += 1
                rec.link(step, PAI.retryOf, prior)
                rec.set(step, PAI.attempt, attempt)
            if status == "failed":
                failed[same_call] = (step, attempt)
            for e in declared.pop(cid, []):
                if e["kind"] in ("memory_write", "memory_read"):
                    memory_events.append((e, sid, step, result))
                elif e["kind"] == "approval":
                    check = rec.step(f"{sid}-approval-{e['time']}", PAI.Check, step, "approval", e["time"], e["time"],
                                     "completed")
                    rec.link(check, PROV.wasAssociatedWith, approver)
                    rec.link(check, PAI.checked, args)
                    rec.set(check, PAI.outcome, e["outcome"])
                    rec.link(step, PROV.wasInformedBy, check)
                elif e["kind"] == "effect":
                    effect = rec.entity(f"{sid}-effect-{e['time']}", PAI.Effect, "effect", e["target"])
                    rec.set(effect, PAI.target, e["target"])
                    rec.generated(effect, step)
                    rec.link(effect, PROV.wasAttributedTo, rec.agent(e["by"]))
            steps[sid] = step

    # memory accesses in the order they happened, so a read sees the versions written before it
    for e, sid, step, result in sorted(memory_events, key=lambda x: seconds(x[0]["time"])):
        access = rec.step(f"{sid}-{e['kind']}-{e['time']}", PAI.MemoryAccess, step,
                          e["kind"].replace("_", " "), e["time"], e["time"], "completed")
        rec.link(access, PROV.wasAssociatedWith, rec.agent(e["by"]))
        if e["kind"] == "memory_write":
            record = rec.memory_write(e["memory"], e["key"], e["value"], access)
            if result is not None and str(rec.g.value(result, PROV.value)) == e["value"]:
                rec.derived(record, result, "inferred")
        else:
            record = rec.memory_read(e["memory"], e["key"], access)
            if record is not None and result is not None and \
                    rec.g.value(result, PROV.value) == rec.g.value(record, PROV.value):
                rec.derived(result, record, "inferred")

    # the model call that follows a tool call in the same agent reads its result
    for cid, (step, result, key, end) in tools.items():
        later = [m for start, m in sorted(model_calls[key], key=lambda x: x[0]) if start >= end]
        if later and result is not None:
            rec.used(later[0], result)

    # an agent run as a tool returns its final answer as the tool result
    def inside(step, top):
        return (step, PAI.partOf, top) in rec.g or any(inside(p, top) for p in rec.g.objects(step, PAI.partOf))

    for sid, result, value in agent_results:
        for key, out, text in answers:
            if key in subagent_step and inside(subagent_step[key], steps[sid]) and value is not None and text == str(value):
                rec.derived(result, out, "inferred")

    handoff = next((e for e in ev if e["kind"] == "handoff"), None)
    received = next((e for e in ev if e["kind"] == "handoff_received"), None)
    if handoff and received:
        prompt = next((e for e in ev if e["kind"] == "query" and e["label"] == handoff["target"]), None)
        if prompt is None:
            raise SystemExit(f"no query for the handoff target {handoff['target']}")
        step = rec.step("handoff", PAI.Handoff, None, "handoff", handoff["time"], prompt["time"], "completed")
        src, dst = rec.agent(handoff["source"]), rec.agent(handoff["target"])
        rec.link(step, PROV.wasAssociatedWith, src)
        rec.link(step, PAI.fromAgent, src)
        rec.link(step, PAI.toAgent, dst)
        rec.link(dst, PROV.actedOnBehalfOf, src)
        rec.used(step, last_answer_of(rec, answers, by_id, main_agent, handoff["time"],
                                      lambda n: n == handoff["source"]))
        message = rec.entity("handoff-message", PAI.Message, "handoff message", prompt["prompt"])
        rec.generated(message, step)
        first = min((s for s in spans if s["name"] == "claude_code.interaction"
                     and s["attrs"]["session.id"] == received["to_session"]),
                    key=lambda s: seconds(s["start"]), default=None)
        if first is None:
            raise SystemExit(f"no agent turn in the session {received['to_session']} that received the handoff")
        rec.used(steps[first["id"]], message)

    for e in ev:
        if e["kind"] == "check":
            check = rec.step(f"check-{e.get('iteration')}-{e['time']}", PAI.Check, None, "check", e["time"], e["time"],
                             "completed")
            rec.link(check, PROV.wasAssociatedWith, rec.agent(e["by"]))
            rec.link(check, PAI.checked, last_answer_of(rec, answers, by_id, main_agent, e["time"],
                                                         lambda n: n != e["by"]))
            rec.set(check, PAI.iteration, None if e.get("iteration") is None else int(e["iteration"]))
            rec.set(check, PAI.outcome, e["outcome"])
            rec.set(check, PAI.reason, e.get("reason"))

    leftover = list(outputs) + list(pending) + list(declared)
    if leftover:
        raise SystemExit(f"{len(leftover)} native records had no matching span: {leftover}")
    return rec


def last_answer_of(rec, answers, by_id, main_agent, time, keep):
    # the last text answer before a moment, given by a main agent that keep accepts, such as the draft a check judged
    done = []
    for key, out, text in answers:
        span = by_id.get(key)
        if span is None or not keep(main_agent[span["attrs"]["session.id"]]):
            continue
        end = rec.g.value(rec.g.value(out, PROV.wasGeneratedBy), PROV.endedAtTime)
        if end is not None:
            done.append((seconds(str(end)), out))
    return latest(done, seconds(time))
