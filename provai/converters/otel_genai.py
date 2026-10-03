import json

from ..config import PAI, PROV
from ..record import Record
from ..spans import read

MODEL_OPS = {"chat", "generate_content", "text_completion"}
MEMORY_OPS = {"search_memory", "create_memory", "update_memory", "upsert_memory", "delete_memory"}


def op(span):
    return span["attrs"].get("gen_ai.operation.name")


def messages(value):
    try:
        return json.loads(value) if isinstance(value, str) else (value or [])
    except ValueError:
        return []


def tool_call_ids(msgs, kind):
    return [p.get("id") for m in msgs for p in m.get("parts", []) if p.get("type") == kind and p.get("id")]


def model_call(rec, step, attrs):
    a = attrs
    rec.link(step, PAI.usedModel, rec.model(a.get("gen_ai.response.model") or a.get("gen_ai.request.model"),
                                             a.get("gen_ai.provider.name") or a.get("gen_ai.system")))
    rec.set(step, PAI.temperature, a.get("gen_ai.request.temperature"))
    rec.set(step, PAI.topP, a.get("gen_ai.request.top_p"))
    rec.set(step, PAI.maxTokens, a.get("gen_ai.request.max_tokens"))
    rec.set(step, PAI.seed, a.get("gen_ai.request.seed"))
    rec.set(step, PAI.inputTokens, a.get("gen_ai.usage.input_tokens"))
    rec.set(step, PAI.outputTokens, a.get("gen_ai.usage.output_tokens"))
    reasons = a.get("gen_ai.response.finish_reasons")
    rec.set(step, PAI.finishReason, ",".join(reasons) if isinstance(reasons, list) else reasons)
    sid = rec.local(step)
    inp, out = messages(a.get("gen_ai.input.messages")), messages(a.get("gen_ai.output.messages"))
    if inp:
        rec.used(step, rec.entity(f"{sid}-input", PAI.Message, "model input", json.dumps(inp)))
    if out:
        rec.generated(rec.entity(f"{sid}-output", PAI.Message, "model output", json.dumps(out)), step)
    return inp, out


def tool_call(rec, step, attrs, arguments=None, result=None):
    a = attrs
    name = a.get("gen_ai.tool.name")
    tool = rec.tool(name)
    rec.link(step, PAI.usedTool, tool)
    sid = rec.local(step)
    args = a.get("gen_ai.tool.call.arguments", arguments)
    if args is not None:
        rec.used(step, rec.entity(f"{sid}-arguments", PROV.Entity, f"{name} arguments", args))
    res = a.get("gen_ai.tool.call.result", result)
    if res is not None:
        rec.generated(rec.entity(f"{sid}-result", PROV.Entity, f"{name} result", res), step)
    return tool


def link_tool_calls(rec, model_steps, tool_steps):
    # model_steps: [(step, input messages, output messages)], tool_steps: {call id: step or [steps]}
    def steps(cid):
        found = tool_steps.get(cid, [])
        return found if isinstance(found, list) else [found]

    for step, inp, out in model_steps:
        sid = rec.local(step)
        for cid in tool_call_ids(out, "tool_call"):
            for tool in steps(cid):
                rec.used(tool, rec.iri("entity", f"{sid}-output"))
        for cid in tool_call_ids(inp, "tool_call_response"):
            for tool in steps(cid):
                result = rec.iri("entity", f"{rec.local(tool)}-result")
                if (result, None, None) in rec.g:
                    rec.used(step, result)


def session(rec, spans):
    for s in spans:
        cid = s["attrs"].get("gen_ai.conversation.id") or s["attrs"].get("session.id")
        if cid:
            n = rec.node("session", PAI.Session, cid, cid)
            rec.link(rec.run, PAI.partOf, n)
            return n


CLASSES = {"invoke_agent": PAI.AgentInvocation, "execute_tool": PAI.ToolCall, **{o: PAI.MemoryAccess for o in MEMORY_OPS}}


def convert(run_dir):
    # any OpenTelemetry GenAI trace, with no framework file
    spans = read(run_dir / "spans.jsonl")
    captured = any(s["attrs"].get("gen_ai.input.messages") for s in spans)
    rec = Record("otel-genai spans", run_dir.name, captured)
    session(rec, spans)
    by_id = {s["id"]: s for s in spans}
    steps, model_steps, tool_steps = {}, [], {}

    def agent_of(s):
        while s:
            if op(s) == "invoke_agent" and s["attrs"].get("gen_ai.agent.name"):
                return rec.agent(s["attrs"]["gen_ai.agent.name"])
            s = by_id.get(s["parent"])

    for s in spans:
        o = op(s)
        cls = PAI.ModelCall if o in MODEL_OPS else CLASSES.get(o, PAI.Step)
        if cls == PAI.ToolCall and not s["attrs"].get("gen_ai.tool.name"):
            cls = PAI.Step
        steps[s["id"]] = rec.step(s["id"], cls, None, s["name"], s["start"], s["end"], s["status"], s["error"])
    for s in spans:
        step = steps[s["id"]]
        if s["parent"] in steps:
            rec.g.remove((step, PAI.partOf, None))
            rec.link(step, PAI.partOf, steps[s["parent"]])
        rec.link(step, PROV.wasAssociatedWith, agent_of(s))
        o = op(s)
        if o in MODEL_OPS:
            inp, out = model_call(rec, step, s["attrs"])
            model_steps.append((step, inp, out))
        elif o == "execute_tool" and s["attrs"].get("gen_ai.tool.name"):
            tool_call(rec, step, s["attrs"])
            if s["attrs"].get("gen_ai.tool.call.id"):
                tool_steps[s["attrs"]["gen_ai.tool.call.id"]] = step
    link_tool_calls(rec, model_steps, tool_steps)
    return rec
