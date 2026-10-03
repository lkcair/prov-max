import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# this version of the OpenAI instrumentation writes message content as log events, not on the spans
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

import httpx
from a2a.client import A2ACardResolver, ClientFactory
from a2a.client.client import ClientConfig
from a2a.helpers.proto_helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role,
                               SendMessageRequest)
from mcp import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp_types import ElicitResult, Implementation, ResourceLink, ToolAnnotations
from openai import AsyncOpenAI
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult, SimpleLogRecordProcessor
from pydantic import BaseModel
from starlette.applications import Starlette

from provai import capture, llm
from provai.config import OPENROUTER

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
SKILLS = {"weather_agent": "Finds the current weather of a city.",
          "facts_agent": "Finds facts about a city.",
          "translator_agent": "Translates a text into Portuguese.",
          "writer": "Writes a two-sentence brief about a city and sends it."}
events = None
weather_calls = 0


class LogLines(LogRecordExporter):
    def __init__(self, path):
        self.out = open(path, "a")

    def export(self, batch):
        for r in batch:
            self.out.write(r.to_json(indent=None) + "\n")
        self.out.flush()
        return LogRecordExportResult.SUCCESS

    def shutdown(self):
        self.out.close()

    def force_flush(self, timeout_millis=30000):
        return True


def span_ids():
    c = trace.get_current_span().get_span_context()
    return {"trace_id": format(c.trace_id, "032x"), "span_id": format(c.span_id, "016x")} if c.is_valid else {}


class Wire:
    # logs every MCP request the server receives, inside the server span the SDK opens for it
    async def __call__(self, ctx, call_next):
        row = {"id": ctx.request_id, "method": ctx.method, "params": ctx.params, **span_ids()}
        try:
            row["client"] = ctx.session.client_params.client_info.name
        except AttributeError:
            row["client"] = None
        try:
            result = await call_next(ctx)
            dump = getattr(result, "model_dump", None)
            row["result"] = dump(mode="json", by_alias=True, exclude_none=True) if dump else result
            return result
        except Exception as e:
            row["error"] = type(e).__name__
            raise
        finally:
            events("mcp", **row)


class Approval(BaseModel):
    approve: bool


def mcp_server(out_dir):
    server = MCPServer("city-tools", version="1.0.0", middleware=[Wire()])

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        global weather_calls
        weather_calls += 1
        if weather_calls == 1:
            raise TimeoutError("weather service timed out")
        return f"{city}: 21 C, clear sky"

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def search_facts(city: str) -> str:
        """Facts about a city from the document store."""
        return DOCS.get(city, "no document found")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
    def save_note(city: str, text: str) -> ResourceLink:
        """Save a note about a city to the shared notes."""
        NOTES[city] = text
        return ResourceLink(name=f"note on {city}", uri=f"notes://{city}", mime_type="text/plain")

    @server.resource("notes://{city}")
    def note(city: str) -> str:
        return NOTES.get(city, "")

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
    async def send_brief(text: str, to: str, ctx: Context) -> str:
        """Send the brief to a recipient."""
        answer = await ctx.elicit(f"Send this brief to {to}?\n{text}", Approval)
        # a person who refuses declines the elicitation, so the action alone carries the decision
        if answer.action != "accept":
            return "not sent"
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        return str(path)

    return server


async def person(context, params):
    # a person answers the elicitation; the reference app approves on their behalf
    answer = ElicitResult(action="accept", content={"approve": True})
    events("elicitation", client="writer", params=params.model_dump(mode="json", by_alias=True, exclude_none=True),
           answer=answer.model_dump(mode="json", by_alias=True, exclude_none=True), by="person")
    return answer


class Model:
    def __init__(self, model):
        self.model = model
        self.client = AsyncOpenAI(base_url=OPENROUTER, api_key=llm.key())

    async def chat(self, agent, messages, tools=None, required=False):
        kw = {"tools": tools, "tool_choice": "required" if required else "auto"} if tools else {}
        r = await self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2,
                                                      max_tokens=1000, **kw)
        usage = r.usage.model_dump() if r.usage else {}
        events("model_end", agent=agent, response_id=r.id, usage=usage, **span_ids())
        return r.choices[0].message


def answer_of(message):
    return (message.content or "").strip()


async def agent_loop(name, model, mcp, prompt, text, tool_names, require_first=True):
    listed = [t for t in (await mcp.list_tools()).tools if t.name in tool_names]
    tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                               "parameters": t.input_schema}} for t in listed]
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    failed = False
    for turn in range(6):
        reply = await model.chat(name, messages, tools, required=(turn == 0 and require_first) or failed)
        if not reply.tool_calls:
            return answer_of(reply)
        messages.append(reply.model_dump(exclude_none=True))
        failed = False
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            events("tool_dispatch", agent=name, call_id=call.id, tool=call.function.name, arguments=args)
            # the model's call id rides in _meta, so the server's record of the call names it
            result = await mcp.call_tool(call.function.name, args, meta={"toolCallId": call.id})
            content = " ".join(getattr(c, "text", None) or getattr(c, "uri", "") for c in result.content)
            failed = failed or result.is_error
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return answer_of(reply)


class Worker(AgentExecutor):
    def __init__(self, name, model, mcp, prompt, tools):
        self.name, self.model, self.mcp, self.prompt, self.tools = name, model, mcp, prompt, tools

    async def execute(self, context, event_queue):
        if not context.current_task:
            await event_queue.enqueue_event(new_task_from_user_message(context.message))
        task = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task.start_work()
        events("task_start", agent=self.name, task_id=context.task_id, context_id=context.context_id,
               message_id=context.message.message_id, **span_ids())
        text = context.get_user_input()
        answer = await self.run(text)
        await task.add_artifact([new_text_part(answer)], name="answer")
        await task.complete()

    async def run(self, text):
        return await agent_loop(self.name, self.model, self.mcp, self.prompt, text, self.tools)

    async def cancel(self, context, event_queue):
        pass


class Facts(Worker):
    async def run(self, text):
        answer = await super().run(text)
        # the facts agent writes its answer to the shared notes, as in the reference task
        await self.mcp.call_tool("save_note", {"city": "Lisbon", "text": answer})
        return answer


class Writer(Worker):
    async def run(self, text):
        if "Send the brief" in text:
            return await agent_loop(self.name, self.model, self.mcp, self.prompt, text, ["send_brief"])
        note = (await self.mcp.read_resource("notes://Lisbon")).contents[0].text
        reply = await self.model.chat(self.name, [{"role": "system", "content": self.prompt},
                                                  {"role": "user", "content": f"{text}\nNote: {note}"}])
        return answer_of(reply)


def card(name):
    return AgentCard(name=name, description=SKILLS[name], version="1",
                     supported_interfaces=[AgentInterface(url=f"http://{name}/a2a", protocol_binding="JSONRPC",
                                                          protocol_version="1.0")],
                     capabilities=AgentCapabilities(streaming=False), default_input_modes=["text/plain"],
                     default_output_modes=["text/plain"],
                     skills=[AgentSkill(id=name, name=name, description=SKILLS[name], tags=[name])])


def detached(app):
    # each agent runs as if in its own process: a request starts from an empty trace context
    async def wrapped(scope, receive, send):
        token = otel_context.attach(otel_context.Context())
        try:
            await app(scope, receive, send)
        finally:
            otel_context.detach(token)
    return wrapped


def a2a_app(name, executor):
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card(name))
    return detached(Starlette(routes=create_agent_card_routes(card(name)) + create_jsonrpc_routes(handler, "/a2a")))


async def wire_request(r):
    events("a2a", direction="request", agent=r.url.host, method=r.method, path=r.url.path,
           body=json.loads(r.content) if r.content else None)


async def wire_response(r):
    await r.aread()
    events("a2a", direction="response", agent=r.request.url.host, status=r.status_code,
           path=r.request.url.path, body=r.json() if r.content else None)


async def run(model_id, out_dir, run_id):
    model = Model(model_id)
    server = mcp_server(out_dir)
    async with AsyncExitStack() as stack:
        async def mcp_client(name, **kw):
            return await stack.enter_async_context(Client(server, mode="legacy", client_info=Implementation(
                name=name, version="1"), **kw))
        workers = {
            "weather_agent": Worker("weather_agent", model, await mcp_client("weather_agent"),
                                    "You do not know the weather. Always call get_weather first, even if an "
                                    "earlier attempt failed, then answer in one sentence.", ["get_weather"]),
            "facts_agent": Facts("facts_agent", model, await mcp_client("facts_agent"),
                                 "You do not know facts about cities. Always call search_facts first, then "
                                 "answer in one sentence.", ["search_facts"]),
            "translator_agent": Worker("translator_agent", model, None, "Translate the text into Portuguese.", []),
            "writer": Writer("writer", model, await mcp_client("writer", elicitation_callback=person),
                             "You write two-sentence briefs about a city. Reply with the brief only. When asked "
                             "to send the brief, call send_brief with the brief and the recipient.", ["send_brief"]),
        }
        clients, cards = {}, {}
        for name, worker in workers.items():
            hx = await stack.enter_async_context(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=a2a_app(name, worker)), base_url=f"http://{name}",
                event_hooks={"request": [wire_request], "response": [wire_response]}))
            cards[name] = await A2ACardResolver(hx, f"http://{name}").get_agent_card()
            clients[name] = ClientFactory(ClientConfig(httpx_client=hx, streaming=False)).create(cards[name])

        async def send(name, text):
            message = new_text_message(text, context_id=run_id, role=Role.ROLE_USER)
            async for event in clients[name].send_message(SendMessageRequest(message=message)):
                task = event.task
                return " ".join(p.text for a in task.artifacts for p in a.parts)

        candidates = ["weather_agent", "facts_agent", "translator_agent"]
        routing = await model.chat("coordinator", [{"role": "user", "content":
                                   "Agents: " + "; ".join(f"{n}: {cards[n].description}" for n in candidates)
                                   + ". Which agents are needed for a short brief on Lisbon's weather and facts? "
                                   "Answer with a JSON list of agent names."}])
        text = answer_of(routing)
        selected = [n for n in candidates if n in text] or candidates[:2]
        events("routing", by="coordinator", candidates=candidates, selected=selected)
        tasks = {"weather_agent": "Weather in Lisbon?", "facts_agent": "Facts about Lisbon?",
                 "translator_agent": "Translate a brief on Lisbon."}
        results = await asyncio.gather(*(send(n, tasks[n]) for n in selected))
        found = dict(zip(selected, results))
        draft = await send("writer", f"Write a two-sentence brief on Lisbon. Weather: "
                                     f"{found.get('weather_agent', '')} Facts: {found.get('facts_agent', '')}")
        for iteration in range(1, 4):
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict = answer_of(await model.chat("evaluator", [{"role": "user", "content":
                                    f"Does this brief mention the weather? Answer PASS or FAIL.\n{draft}"}]))
                passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
            events("check", by="evaluator", checked=draft, iteration=iteration,
                   outcome="pass" if passed else "fail", reason=reason)
            if passed:
                break
            draft = await send("writer", f"The draft failed review: {reason}. Revise it:\n{draft}")
        await send("writer", f"Send the brief to client. The brief:\n{draft}")


def main():
    model = llm.pick()
    run_id = "pr-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("protocols", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "protocols-reference")
    logs = LoggerProvider()
    logs.add_log_record_processor(SimpleLogRecordProcessor(LogLines(out / "logs.jsonl")))
    OpenAIInstrumentor().instrument(tracer_provider=provider, logger_provider=logs)
    global events
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="protocols", versions={
        p: version(p) for p in ("mcp", "a2a-sdk", "openai", "opentelemetry-sdk",
                                "opentelemetry-instrumentation-openai-v2", "opentelemetry-util-genai")})
    asyncio.run(run(model, out, run_id))
    provider.shutdown()
    logs.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
