import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from rdflib import RDF

from ..config import PAI, PROV
from ..record import Record
from ..spans import latest, read, same, seconds
from . import otel_genai as og

CONFIRM = "adk_request_confirmation"
QUOTED = re.compile(r"\[(?P<agent>[^\]]+)\] `(?P<tool>[^`]+)` tool returned result:")


def text_of(msgs):
    # the final answer is the last text part; the model's reasoning comes as earlier text parts
    texts = [p.get("content", "") for m in msgs for p in m.get("parts", []) if p.get("type") == "text"]
    return texts[-1].strip() if texts else ""


def plain(value):
    return value if isinstance(value, str) or value is None else json.dumps(value)


def when(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def convert(run_dir):
    ev = [json.loads(line) for line in open(run_dir / "events.jsonl")]
    head = ev[0]
    v = head["versions"]
    agents = next((e for e in ev if e["kind"] == "agents"), None)
    if agents is None:
        raise SystemExit(f"{run_dir}: no agents event")
    defs = agents["definitions"]
    spans = read(run_dir / "spans.jsonl")
    by_id = {s["id"]: s for s in spans}
    children = defaultdict(list)
    for s in spans:
        children[s["parent"]].append(s)

    rec = Record(f"google-adk {v['google-adk']} with litellm {v['litellm']}", head["run_id"], True)
    rec.link(rec.run, PAI.partOf, rec.node("session", PAI.Session, head["run_id"], head["run_id"]))
    user = rec.node("agent", PROV.Person, "user", "user")
    approver = rec.node("agent", PROV.Person, "approver", "approver")
    rec.link(rec.run, PROV.wasAssociatedWith, user)

    def calls(kind):
        out = defaultdict(list)
        for e in ev:
            if e["kind"] == kind:
                out[e["call_id"]].append(e)
        return out

    tool_start, tool_end, tool_error = calls("tool_start"), calls("tool_end"), calls("tool_error")
    costs = [e for e in ev if e["kind"] == "litellm"]
    # a person's answers reach the plugin as user messages, not as runner events
    native = [e if e["kind"] == "event" else
              {"session_id": e["session_id"], "event": {"content": e["content"], "timestamp": seconds(e["time"]),
                                                        "id": f"user-{e['time']}", "author": "user"}}
              for e in ev if e["kind"] in ("event", "user_message")]

    steps, agent_of, invocations = {}, {}, Counter()
    model_steps, tool_steps, answers = [], defaultdict(list), []
    failed, agent_tool_results, held, quoted = {}, [], {}, []
    done = []
    consumed, span_of_call = set(), {}

    def agent_for(span):
        while span:
            if span["id"] in agent_of:
                return agent_of[span["id"]]
            span = by_id.get(span["parent"])

    def inside(span, top):
        while span:
            if span["id"] == top:
                return True
            span = by_id.get(span["parent"])

    for s in spans:
        op, a, sid = og.op(s), s["attrs"], s["id"]
        parent = steps.get(s["parent"])
        if s["name"] == "invocation":
            steps[sid] = rec.step(sid, PAI.Step, parent, "invocation", s["start"], s["end"], s["status"])

        elif op == "invoke_agent":
            name = a["gen_ai.agent.name"]
            invocations[name] += 1
            step = rec.step(sid, PAI.AgentInvocation, parent, name, s["start"], s["end"], s["status"], s["error"])
            rec.set(step, PAI.iteration, invocations[name])
            agent_of[sid] = rec.agent(name)
            rec.link(step, PROV.wasAssociatedWith, agent_of[sid])
            spec = defs.get(name, {})
            agent_tools = [t["agent"] for t in spec.get("tools", []) if t.get("agent")]
            if agent_tools or spec.get("sub_agents"):
                rec.g.add((step, RDF.type, PAI.Routing))
                for c in agent_tools + spec.get("sub_agents", []):
                    rec.link(step, PAI.candidate, rec.agent(c))
                rec.link(agent_of[sid], PROV.actedOnBehalfOf, user)
            steps[sid] = step

        elif s["name"] == "call_llm":
            gen = next((c for c in children[sid] if og.op(c) == "generate_content"), None)
            if gen is not None:
                consumed.add(gen["id"])
            ga = gen["attrs"] if gen else a
            step = rec.step(sid, PAI.ModelCall, parent, "call_llm", s["start"], s["end"], s["status"], s["error"])
            agent = agent_for(s)
            rec.link(step, PROV.wasAssociatedWith, agent)
            og.model_call(rec, step, ga)
            # ADK keeps the sampling settings only in its own request attribute
            config = json.loads(a.get("gcp.vertex.agent.llm_request") or "{}").get("config") or {}
            rec.set(step, PAI.temperature, config.get("temperature"))
            rec.set(step, PAI.topP, config.get("top_p"))
            rec.set(step, PAI.seed, config.get("seed"))
            if not ga.get("gen_ai.request.max_tokens"):
                rec.set(step, PAI.maxTokens, config.get("max_output_tokens"))
            inp, out = og.messages(ga.get("gen_ai.input.messages")), og.messages(ga.get("gen_ai.output.messages"))
            called = og.tool_call_ids(out, "tool_call")
            said = text_of(out)
            # LiteLLM reports cost per completion; join by the tool calls it returned, else by its answer
            match = next((c for c in costs if called and c["tool_call_ids"] == called), None) or \
                next((c for c in costs if not called and not c["tool_call_ids"] and said and c["text"].endswith(said)),
                     None)
            if match:
                costs.remove(match)
                cost = match["usage"].get("cost")
                rec.set(step, PAI.cost, None if cost is None else float(cost))
            model_steps.append((step, inp, out))
            # other agents' tool results reach this call as quoted text, without their call ids
            for m in inp:
                for p in m.get("parts", []):
                    q = QUOTED.match(p.get("content", "")) if p.get("type") == "text" else None
                    if q:
                        quoted.append((step, q["agent"], q["tool"], seconds(s["start"])))
            if said and not called:
                answers.append((s, rec.iri("entity", f"{sid}-output"), said, rec.local(agent)))
            steps[sid] = step
            if gen is not None:
                steps[gen["id"]] = step

        elif op == "execute_tool":
            name, cid = a.get("gen_ai.tool.name"), a.get("gen_ai.tool.call.id")
            if name == "(merged tools)":
                continue
            agent = agent_for(s)
            if not tool_start.get(cid):
                raise SystemExit(f"tool span {name} {cid} has no matching native event")
            start = tool_start[cid].pop(0)
            end = tool_end[cid].pop(0) if tool_end.get(cid) else {}
            err = tool_error[cid].pop(0) if tool_error.get(cid) else None
            if name == "transfer_to_agent":
                step = rec.step(sid, PAI.Handoff, parent, "transfer_to_agent", s["start"], s["end"], "completed")
                dst = rec.agent(start["args"]["agent_name"])
                rec.link(step, PROV.wasAssociatedWith, agent)
                rec.link(step, PAI.fromAgent, agent)
                rec.link(step, PAI.toAgent, dst)
                rec.link(dst, PROV.actedOnBehalfOf, agent)
                rec.link(parent, PAI.selected, dst)
                steps[sid] = step
                continue
            result = end.get("result")
            waiting = isinstance(result, dict) and "requires confirmation" in str(result.get("error", ""))
            status = "failed" if err else "interrupted" if waiting else "completed"
            step = rec.step(sid, PAI.ToolCall, parent, name, s["start"], s["end"], status, err and err["error"])
            rec.link(step, PROV.wasAssociatedWith, agent)
            args = json.dumps(start["args"], ensure_ascii=False)
            og.tool_call(rec, step, {"gen_ai.tool.name": name}, args, None if waiting else plain(result))
            res = None if waiting else rec.iri("entity", f"{sid}-result")
            tool_steps[cid].append(step)
            if waiting:
                held[cid] = {"step": step, "arguments": rec.iri("entity", f"{sid}-arguments")}
            elif cid in held:
                rec.used(step, held[cid]["arguments"])
                held[cid]["ran"] = step
            tool = next((t for t in defs.get(rec.local(agent), {}).get("tools", []) if t["name"] == name), {})
            if tool.get("agent"):
                selected = rec.agent(tool["agent"])
                rec.link(parent, PAI.selected, selected)
                rec.link(selected, PROV.actedOnBehalfOf, agent)
                agent_tool_results.append((s, res, plain(result)))
            key = (rec.local(agent), name, same(start["args"]))
            attempt = 1
            if key in failed:
                previous, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, previous)
                rec.set(step, PAI.attempt, attempt)
            if err:
                failed[key] = (step, attempt)
            steps[sid] = step
            span_of_call[cid] = s
            if res is not None:
                done.append((rec.local(agent), name, seconds(s["end"]), res))

        elif op == "generate_content" and sid in consumed:
            continue
        else:
            raise SystemExit(f"span {s['name']} has no mapping")

    def source_of(agent, text, before, slack=0):
        # a written value comes from the latest answer of its writer with the same text
        return latest([(seconds(x[0]["end"]), x[1]) for x in answers if x[3] == agent and x[2] == text], before, slack)

    def judged(by, before):
        # a check judges the latest answer before it that another agent gave
        return latest([(seconds(x[0]["end"]), x[1]) for x in answers if x[3] != by], before)

    def one(responses, text=None):
        # a memory read or an effect belongs to the tool response of its event that carries its text
        found = [r for r in responses if r["name"] != CONFIRM]
        if len(found) > 1 and text:
            found = [r for r in found if text in plain((r["response"] or {}).get("result"))]
        if len(found) != 1:
            raise SystemExit(f"an event declares a fact beside {len(found)} tool responses")
        return found[0]

    def tool_step(cid):
        return tool_steps[cid][-1] if tool_steps.get(cid) else None

    def agent_span(session, agent):
        return next((s for s in spans if og.op(s) == "invoke_agent" and s["attrs"].get("gen_ai.agent.name") == agent
                     and s["attrs"].get("gen_ai.conversation.id") == session), None)

    for step, agent, tool, start in quoted:
        found = [res for a, t, end, res in done if a == agent and t == tool and end <= start]
        if found:
            rec.used(step, found[-1])

    # ADK state keys carry their scope as a prefix: app:, user:, or temp:
    scopes = {}
    for row in native:
        for k in ((row["event"].get("actions") or {}).get("state_delta") or {}):
            if ":" in k:
                scopes[k.split(":", 1)[1]] = k.split(":", 1)[0]
    for row in native:
        e = row["event"]
        delta = (e.get("actions") or {}).get("state_delta") or {}
        at = when(e["timestamp"])
        responses = [p["function_response"] for p in (e.get("content") or {}).get("parts") or [] if "function_response" in p]
        for r in responses:
            if r["name"] == CONFIRM:
                ask = next((p["function_call"] for x in native for p in (x["event"].get("content") or {}).get("parts") or []
                            if "function_call" in p and p["function_call"].get("id") == r["id"]), None)
                if ask is None:
                    raise SystemExit(f"confirmation {r['id']} has no request")
                cid = ask["args"]["originalFunctionCall"]["id"]
                if cid not in held:
                    raise SystemExit(f"confirmed call {cid} was never held")
                check = rec.step(f"{e['id']}-approval", PAI.Check, None, "approval", at, at, "completed")
                rec.link(check, PROV.wasAssociatedWith, approver)
                rec.link(check, PAI.checked, held[cid]["arguments"])
                rec.set(check, PAI.outcome, "approve" if r["response"].get("confirmed") else "reject")
                rec.link(held[cid].get("ran"), PROV.wasInformedBy, check)
        for name, data in delta.items():
            if name in ("memory_write", "memory_read") and data.get("by") == e.get("author"):
                if name == "memory_write":
                    span = agent_span(row["session_id"], data["by"])
                    access = rec.step(f"{e['id']}-{name}", PAI.MemoryAccess, steps.get(span and span["id"]),
                                      "memory write", at, at, "completed")
                    record = rec.memory_write(data["memory"], data["key"], data.get("value"), access)
                    rec.set(record, PAI.scope, data.get("scope") or scopes.get(f"{data['memory']}/{data['key']}"))
                    # the event that carries the write can be stamped just before the answer's span closes
                    source = source_of(data["by"], (data.get("value") or "").strip(), e["timestamp"], slack=1)
                    if source is not None:
                        rec.derived(record, source, "inferred")
                else:
                    n = rec.versions.get((data["memory"], data["key"]))
                    stored = n and rec.g.value(rec.iri("entity", f"{data['memory']}/{data['key']}/{n}"), PROV.value)
                    response = one(responses, stored and str(stored))
                    cid = response["id"]
                    access = rec.step(f"{e['id']}-{name}", PAI.MemoryAccess, tool_step(cid), "memory read", at, at,
                                      "completed")
                    record = rec.memory_read(data["memory"], data["key"], access)
                    span = span_of_call.get(cid)
                    if span is None:
                        raise SystemExit(f"memory read {cid} has no tool span")
                    value = rec.g.value(record, PROV.value) if record is not None else None
                    if value is not None and plain((response["response"] or {}).get("result")) == str(value):
                        rec.derived(rec.iri("entity", f"{span['id']}-result"), record, "inferred")
                rec.link(access, PROV.wasAssociatedWith, rec.agent(data["by"]))
            elif name == "check":
                check = rec.step(f"{e['id']}-check", PAI.Check, None, "check", at, at, "completed")
                rec.link(check, PROV.wasAssociatedWith, rec.agent(data["by"]))
                rec.link(check, PAI.checked, judged(data["by"], e["timestamp"]))
                rec.set(check, PAI.iteration, data.get("iteration"))
                rec.set(check, PAI.outcome, data["outcome"])
                rec.set(check, PAI.reason, data.get("reason"))
            elif name == "effect" and e.get("author") == data["by"]:
                effect = rec.entity(f"{e['id']}-effect", PAI.Effect, "effect", data["target"])
                rec.set(effect, PAI.target, data["target"])
                rec.generated(effect, tool_step(one(responses, data["target"])["id"]))
                rec.link(effect, PROV.wasAttributedTo, rec.agent(data["by"]))

    leftover = [e["tool"] for calls_ in tool_start.values() for e in calls_]
    if leftover:
        raise SystemExit(f"{len(leftover)} native tool calls had no matching span: {leftover}")
    og.link_tool_calls(rec, model_steps, {cid: s[-1] for cid, s in tool_steps.items()})

    # an agent used as a tool returns its final answer as the tool result
    for span, res, output in agent_tool_results:
        for s, out, said, _ in answers:
            if inside(s, span["id"]) and said == (output or "").strip():
                rec.derived(res, out, "inferred")
    return rec
