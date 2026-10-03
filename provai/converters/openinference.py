import json
import re

from ..config import PAI, PROV
from ..record import Record
from ..spans import read

KIND = "openinference.span.kind"


def kind(span):
    return span["attrs"].get(KIND)


def indexed(attrs, prefix):
    # llm.output_messages.0.message.tool_calls.1.tool_call.id -> {(0, 1): {"id": ...}}
    out = {}
    pat = re.compile(re.escape(prefix) + r"\.(\d+)\.message\.(.+)")
    for k, v in attrs.items():
        m = pat.fullmatch(k)
        if m:
            out.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return [out[i] for i in sorted(out)]


def tool_calls(attrs, prefix="llm.output_messages"):
    calls = []
    for msg in indexed(attrs, prefix):
        found = {}
        for k, v in msg.items():
            m = re.fullmatch(r"tool_calls\.(\d+)\.tool_call\.(.+)", k)
            if m:
                found.setdefault(int(m.group(1)), {})[m.group(2)] = v
        calls += [{"id": c.get("id"), "name": c.get("function.name"), "arguments": c.get("function.arguments")}
                  for _, c in sorted(found.items())]
    return calls


def tool_results(attrs):
    return [m["tool_call_id"] for m in indexed(attrs, "llm.input_messages") if m.get("tool_call_id")]


def model_call(rec, step, attrs):
    a = attrs
    rec.link(step, PAI.usedModel, rec.model(a.get("llm.model_name"), a.get("llm.provider") or a.get("llm.system")))
    try:
        params = json.loads(a.get("llm.invocation_parameters") or "{}")
    except ValueError:
        params = {}
    rec.set(step, PAI.temperature, params.get("temperature"))
    rec.set(step, PAI.topP, params.get("top_p"))
    rec.set(step, PAI.maxTokens, params.get("max_tokens") or params.get("max_completion_tokens"))
    rec.set(step, PAI.seed, params.get("seed"))
    rec.set(step, PAI.inputTokens, a.get("llm.token_count.prompt"))
    rec.set(step, PAI.outputTokens, a.get("llm.token_count.completion"))
    cost = a.get("llm.cost.total")
    rec.set(step, PAI.cost, None if cost is None else float(cost))
    sid = rec.local(step)
    inp = out = None
    if a.get("input.value"):
        inp = rec.entity(f"{sid}-input", PAI.Message, "model input", a["input.value"])
        rec.used(step, inp)
    if a.get("output.value"):
        out = rec.entity(f"{sid}-output", PAI.Message, "model output", a["output.value"])
        rec.generated(out, step)
    return inp, out


def tool_call(rec, step, attrs):
    a = attrs
    name = a.get("tool.name")
    rec.link(step, PAI.usedTool, rec.tool(name))
    sid = rec.local(step)
    args = res = None
    if a.get("input.value") is not None:
        args = rec.entity(f"{sid}-arguments", PROV.Entity, f"{name} arguments", a["input.value"])
        rec.used(step, args)
    if a.get("output.value") is not None:
        res = rec.entity(f"{sid}-result", PROV.Entity, f"{name} result", a["output.value"])
        rec.generated(res, step)
    return args, res


def link_tool_calls(rec, model_steps, tool_steps):
    # model_steps: [(step, output, ids it called, ids whose results it read)], tool_steps: {call id: [(step, result)]}
    for step, out, called, results_read in model_steps:
        for cid in called:
            for tool, _ in tool_steps.get(cid, []):
                rec.used(tool, out)
        for cid in results_read:
            for _, res in tool_steps.get(cid, []):
                rec.used(step, res)


def session(rec, spans):
    for s in spans:
        sid = s["attrs"].get("session.id")
        if sid:
            n = rec.node("session", PAI.Session, sid, sid)
            rec.link(rec.run, PAI.partOf, n)
            return n


CLASSES = {"AGENT": PAI.AgentInvocation, "LLM": PAI.ModelCall, "TOOL": PAI.ToolCall, "RETRIEVER": PAI.Retrieval}


def convert(run_dir):
    # any OpenInference trace, with no framework file
    spans = read(run_dir / "spans.jsonl")
    captured = any(s["attrs"].get("input.value") for s in spans if kind(s) == "LLM")
    rec = Record("openinference spans", run_dir.name, captured)
    session(rec, spans)
    by_id = {s["id"]: s for s in spans}
    steps, model_steps, tool_steps, requested = {}, [], {}, {}

    def agent_of(s):
        while s:
            if kind(s) == "AGENT" and s["attrs"].get("agent.name"):
                return rec.agent(s["attrs"]["agent.name"])
            s = by_id.get(s["parent"])

    def is_call(s):
        # an LLM span that only prepares a request has no tokens, no output, and no error
        a = s["attrs"]
        return (s["status"] == "failed" or a.get("llm.token_count.prompt") is not None
                or any(k.startswith(("llm.output_messages", "llm.input_messages")) for k in a))

    # an LLM span that wraps another LLM span is one call seen twice; the innermost one is the call
    wrappers = set()
    for s in spans:
        if kind(s) == "LLM":
            p = by_id.get(s["parent"])
            while p:
                if kind(p) == "LLM":
                    wrappers.add(p["id"])
                p = by_id.get(p["parent"])
    for s in spans:
        cls = CLASSES.get(kind(s), PAI.Step)
        if cls == PAI.ToolCall and not s["attrs"].get("tool.name"):
            cls = PAI.Step
        if cls == PAI.ModelCall and (s["id"] in wrappers or not is_call(s)):
            cls = PAI.Step
        steps[s["id"]] = rec.step(s["id"], cls, None, s["name"], s["start"], s["end"], s["status"], s["error"])

    def container(sid):
        # the step that holds a model call and the tool calls it requested, above any wrapper LLM spans
        while sid in by_id and kind(by_id[sid]) == "LLM":
            sid = by_id[sid]["parent"]
        return sid

    for s in spans:
        step = steps[s["id"]]
        if s["parent"] in steps:
            rec.g.remove((step, PAI.partOf, None))
            rec.link(step, PAI.partOf, steps[s["parent"]])
        rec.link(step, PROV.wasAssociatedWith, agent_of(s))
        if kind(s) == "LLM" and s["id"] not in wrappers and is_call(s):
            inp, out = model_call(rec, step, s["attrs"])
            called = tool_calls(s["attrs"])
            requested.setdefault(container(s["parent"]), []).extend(called)
            model_steps.append((step, out, [c["id"] for c in called], tool_results(s["attrs"])))
        elif kind(s) == "TOOL" and s["attrs"].get("tool.name"):
            args, res = tool_call(rec, step, s["attrs"])
            # tool spans carry no call id; take the first call of that name the model requested in the same step
            calls = requested.get(container(s["parent"]), [])
            match = next((c for c in calls if c["name"] == s["attrs"]["tool.name"]), None)
            if match:
                calls.remove(match)
                tool_steps.setdefault(match["id"], []).append((step, res))
    link_tool_calls(rec, model_steps, tool_steps)
    return rec
