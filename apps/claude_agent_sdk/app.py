import asyncio
import json
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from claude_agent_sdk import (AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient, HookMatcher,
                              PermissionResultAllow, PermissionResultDeny, create_sdk_mcp_server, tool)
from claude_agent_sdk._cli_version import __cli_version__

from provai import capture, llm

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
weather_calls = 0
STATUS = {0: "UNSET", 1: "OK", 2: "ERROR"}


def value(v):
    if "arrayValue" in v:
        return [value(x) for x in v["arrayValue"].get("values", [])]
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    return None


def attrs(items):
    return {a["key"]: value(a["value"]) for a in items or []}


def stamp(nanos):
    return datetime.fromtimestamp(int(nanos) / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Receiver(BaseHTTPRequestHandler):
    # a local OTLP/HTTP JSON endpoint; the CLI exports its traces and logs here and nowhere else
    spans = logs = None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path.endswith("/v1/traces"):
            for rs in body.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for s in ss.get("spans", []):
                        self.spans.write(json.dumps({
                            "name": s["name"], "context": {"trace_id": "0x" + s["traceId"], "span_id": "0x" + s["spanId"]},
                            "parent_id": "0x" + s["parentSpanId"] if s.get("parentSpanId") else None,
                            "start_time": stamp(s["startTimeUnixNano"]), "end_time": stamp(s["endTimeUnixNano"]),
                            "status": {"status_code": STATUS[(s.get("status") or {}).get("code", 0)],
                                       "description": (s.get("status") or {}).get("message")},
                            "attributes": attrs(s.get("attributes")),
                            "events": [{"name": e["name"], "timestamp": stamp(e["timeUnixNano"]),
                                        "attributes": attrs(e.get("attributes"))} for e in s.get("events", [])],
                            "links": [{"context": {"trace_id": "0x" + k["traceId"], "span_id": "0x" + k["spanId"]},
                                       "attributes": attrs(k.get("attributes"))} for k in s.get("links", [])],
                            "scope": ss.get("scope", {}).get("name")}) + "\n")
            self.spans.flush()
        elif self.path.endswith("/v1/logs"):
            for rl in body.get("resourceLogs", []):
                for sl in rl.get("scopeLogs", []):
                    for r in sl.get("logRecords", []):
                        self.logs.write(json.dumps({
                            "time": stamp(r.get("timeUnixNano") or r.get("observedTimeUnixNano")),
                            "body": value(r.get("body") or {}), "trace_id": r.get("traceId"),
                            "span_id": r.get("spanId"), "attributes": attrs(r.get("attributes"))}) + "\n")
            self.logs.flush()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def plain(x):
    if is_dataclass(x):
        return {k: plain(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)


def hooks(events):
    async def record(data, tool_use_id, context):
        events("hook", **{k: v for k, v in plain(data).items() if k != "transcript_path"})
        name = data.get("tool_name")
        if data["hook_event_name"] == "PostToolUse":
            # the facts agent's answer goes to the shared notes, as in the reference task
            if name in ("Agent", "Task") and data["tool_input"].get("subagent_type") == "facts_agent":
                text = str(answer_of(data["tool_response"]))
                NOTES["Lisbon"] = text
                events("memory_write", tool_use_id=tool_use_id, by="facts_agent", memory="notes", key="Lisbon",
                       value=text)
            elif name == "mcp__brief__read_note":
                events("memory_read", tool_use_id=tool_use_id, by="writer", memory="notes",
                       key="Lisbon")
            elif name == "mcp__brief__send_brief":
                events("effect", tool_use_id=tool_use_id, by="writer", target=str(answer_of(data["tool_response"])))
        return {}

    names = ("PreToolUse", "PostToolUse", "PostToolUseFailure", "SubagentStart", "SubagentStop",
             "UserPromptSubmit", "Stop")
    return {n: [HookMatcher(hooks=[record])] for n in names}


def answer_of(response):
    # the text of a tool response, which the CLI gives as content blocks or a plain value
    if isinstance(response, dict) and "content" in response:
        response = response["content"]
    if isinstance(response, list):
        return " ".join(b.get("text", "") for b in response if isinstance(b, dict)).strip()
    return response


def server(out_dir):
    @tool("get_weather", "Current weather for a city.", {"city": str})
    async def get_weather(args):
        global weather_calls
        weather_calls += 1
        if weather_calls == 1:
            raise TimeoutError("weather service timed out")
        return {"content": [{"type": "text", "text": f"{args['city']}: 21 C, clear sky"}]}

    @tool("search_facts", "Facts about a city from the document store.", {"city": str})
    async def search_facts(args):
        return {"content": [{"type": "text", "text": DOCS.get(args["city"], "no document found")}]}

    @tool("read_note", "Read the shared note about a city.", {"city": str})
    async def read_note(args):
        return {"content": [{"type": "text", "text": NOTES.get(args["city"], "no note")}]}

    @tool("send_brief", "Send the brief to a recipient.", {"text": str, "to": str})
    async def send_brief(args):
        path = out_dir / f"brief-to-{args['to']}.txt"
        path.write_text(args["text"])
        return {"content": [{"type": "text", "text": str(path)}]}

    return create_sdk_mcp_server("brief", "1.0.0", tools=[get_weather, search_facts, read_note, send_brief])


AGENTS = {
    "weather_agent": AgentDefinition(
        description="Finds the current weather of a city.", tools=["mcp__brief__get_weather"], background=False,
        prompt="You do not know the weather. Always call get_weather first. If it fails, call it again. "
               "Then answer in one sentence."),
    "facts_agent": AgentDefinition(
        description="Finds facts about a city.", tools=["mcp__brief__search_facts"], background=False,
        prompt="You do not know facts about cities. Always call search_facts first, then answer in one sentence."),
    "translator_agent": AgentDefinition(
        description="Translates a text into Portuguese.", tools=[], background=False,
        prompt="Translate the text you are given into Portuguese."),
}


async def converse(options, prompts, events, label):
    # one CLI session; every message of its stream goes to the event log
    async with ClaudeSDKClient(options) as client:
        last = None
        for prompt in prompts:
            events("query", label=label, prompt=prompt)
            await client.query(prompt)
            async for m in client.receive_response():
                events("message", label=label, type=type(m).__name__, **plain(m))
                if type(m).__name__ == "ResultMessage":
                    last = m
            yield last


async def run(model, out_dir, run_id, events, endpoint, config_dir):
    env = {"ANTHROPIC_BASE_URL": "https://openrouter.ai/api", "ANTHROPIC_AUTH_TOKEN": llm.key(),
           "ANTHROPIC_API_KEY": "", "CLAUDE_CONFIG_DIR": str(config_dir),
           "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
           "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
           "CLAUDE_CODE_ENABLE_TELEMETRY": "1", "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
           "OTEL_TRACES_EXPORTER": "otlp", "OTEL_LOGS_EXPORTER": "otlp", "OTEL_METRICS_EXPORTER": "none",
           "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json", "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
           "OTEL_LOG_USER_PROMPTS": "1", "OTEL_LOG_TOOL_DETAILS": "1", "OTEL_LOG_TOOL_CONTENT": "1",
           "OTEL_LOG_ASSISTANT_RESPONSES": "1", "OTEL_TRACES_EXPORT_INTERVAL": "1000",
           "OTEL_LOGS_EXPORT_INTERVAL": "1000", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1",
           "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1", "CLAUDE_CODE_SUBAGENT_MODEL": model}
    brief = server(out_dir)
    base = dict(model=model, env=env, cwd=str(out_dir), setting_sources=[], hooks=hooks(events),
                mcp_servers={"brief": brief}, permission_mode="default", max_turns=12)

    async def delegate(tool_name, tool_input, context):
        # the coordinator works through its agents; only a subagent may call the worker tools
        if context.agent_id:
            return PermissionResultAllow()
        return PermissionResultDeny(message="Call this tool through weather_agent or facts_agent.")

    coordinator = ClaudeAgentOptions(
        **base, agents=AGENTS, tools=["Task"], allowed_tools=["Task", "Agent"], can_use_tool=delegate,
        disallowed_tools=["mcp__brief__read_note", "mcp__brief__send_brief"],
        system_prompt="You plan a short brief in English on a city's weather and facts. Your agents are "
                      "weather_agent, facts_agent, and translator_agent. Call only the agents the brief needs. In "
                      "your first reply, call every agent you need at once, in parallel, in one message. Then reply "
                      "with what they found.")
    found = None
    async for result in converse(coordinator, ["Write a short brief on Lisbon's weather and facts."], events,
                                 "coordinator"):
        found = result

    events("handoff", source="coordinator", target="writer", from_session=found.session_id)

    async def approve(tool_name, tool_input, context):
        # a person reviews the send_brief call; the reference app approves on their behalf
        if tool_name == "mcp__brief__send_brief":
            events("approval", tool_use_id=context.tool_use_id, by="person", outcome="approve", tool=tool_name)
        return PermissionResultAllow()

    writer = ClaudeAgentOptions(
        **base, tools=[], allowed_tools=["mcp__brief__read_note"], can_use_tool=approve,
        disallowed_tools=["mcp__brief__get_weather", "mcp__brief__search_facts"],
        system_prompt="You write two-sentence briefs about a city. Always call read_note for the city first. "
                      "Reply with the brief only. When asked to send a brief, call send_brief with the brief and "
                      "the recipient.")
    evaluator = ClaudeAgentOptions(
        **{**base, "max_turns": 2}, tools=[], disallowed_tools=["mcp__brief__get_weather", "mcp__brief__search_facts",
                                                               "mcp__brief__read_note", "mcp__brief__send_brief"],
        system_prompt="Does the brief you are given mention the weather? Answer PASS or FAIL.")

    async with ClaudeSDKClient(writer) as client:
        async def ask(prompt):
            events("query", label="writer", prompt=prompt)
            await client.query(prompt)
            last = None
            async for m in client.receive_response():
                events("message", label="writer", type=type(m).__name__, **plain(m))
                if type(m).__name__ == "ResultMessage":
                    last = m
            return last

        draft = await ask(f"Write a two-sentence brief on Lisbon from the coordinator's findings: {found.result}")
        events("handoff_received", target="writer", to_session=draft.session_id)
        for iteration in range(1, 4):
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict = None
                async for result in converse(evaluator, [draft.result], events, "evaluator"):
                    verdict = result.result or ""
                passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
            events("check", by="evaluator", checked="draft", iteration=iteration,
                   outcome="pass" if passed else "fail", reason=reason)
            if passed:
                break
            draft = await ask(f"The draft failed review: {reason}. Revise it.")
        await ask("Send the brief to client.")


def main():
    model = llm.pick()
    run_id = "cc-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("claude_agent_sdk", run_id)
    config_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}-config-"))
    Receiver.spans, Receiver.logs = open(out / "spans.jsonl", "a"), open(out / "otel_logs.jsonl", "a")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="claude_agent_sdk", versions={
        "claude-agent-sdk": version("claude-agent-sdk"), "claude-code": __cli_version__},
        agents={k: {"description": a.description, "tools": a.tools} for k, a in AGENTS.items()})
    try:
        asyncio.run(run(model, out, run_id, events, f"http://127.0.0.1:{httpd.server_address[1]}", config_dir))
    finally:
        httpd.shutdown()
        Receiver.spans.close()
        Receiver.logs.close()
        events.close()
        shutil.rmtree(config_dir, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    main()
