import asyncio
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY"
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

import litellm
from litellm.integrations.custom_logger import CustomLogger
from google.adk.agents import LlmAgent
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from provai import capture, llm

APP = "city_brief"
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
weather_calls = 0


def dump(obj):
    return obj.model_dump(mode="json", exclude_none=True) if obj is not None else None


class Native(BasePlugin):
    def __init__(self, events):
        super().__init__(name="native")
        self.events = events

    def ids(self, ctx):
        inv = getattr(ctx, "_invocation_context", ctx)
        return {"invocation_id": inv.invocation_id, "session_id": inv.session.id, "branch": inv.branch}

    async def on_user_message_callback(self, *, invocation_context, user_message):
        self.events("user_message", content=dump(user_message), **self.ids(invocation_context))

    async def on_event_callback(self, *, invocation_context, event):
        self.events("event", event=dump(event), **self.ids(invocation_context))

    async def before_agent_callback(self, *, agent, callback_context):
        self.events("agent_start", agent=agent.name, **self.ids(callback_context))

    async def after_agent_callback(self, *, agent, callback_context):
        self.events("agent_end", agent=agent.name, **self.ids(callback_context))

    async def after_model_callback(self, *, callback_context, llm_response):
        self.events("model_end", agent=callback_context.agent_name, usage=dump(llm_response.usage_metadata),
                    **self.ids(callback_context))

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self.events("tool_start", tool=tool.name, args=tool_args, call_id=tool_context.function_call_id,
                    agent=tool_context.agent_name, **self.ids(tool_context))

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        self.events("tool_end", tool=tool.name, call_id=tool_context.function_call_id, result=result,
                    **self.ids(tool_context))

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self.events("tool_error", tool=tool.name, call_id=tool_context.function_call_id,
                    error=type(error).__name__, message=str(error), **self.ids(tool_context))


def get_weather(city: str) -> str:
    """Current weather for a city."""
    global weather_calls
    weather_calls += 1
    if weather_calls == 1:
        raise TimeoutError("weather service timed out")
    return f"{city}: 21 C, clear sky"


def search_facts(city: str) -> str:
    """Facts about a city from the document store."""
    return DOCS.get(city, "no document found")


def read_note(city: str, tool_context: ToolContext) -> str:
    """Read the shared note about a city."""
    tool_context.state["memory_read"] = {"by": "writer", "memory": "notes", "key": city}
    return tool_context.state.get(f"app:notes/{city}", "no note")


def save_note(callback_context):
    # the facts agent's answer goes to the shared notes, app-scoped state that outlives the session
    events = [e for e in callback_context._invocation_context.session.events if e.author == "facts_agent"]
    text = "".join(p.text or "" for p in events[-1].content.parts if not p.thought).strip() if events else ""
    callback_context.state["app:notes/Lisbon"] = text
    callback_context.state["memory_write"] = {"by": "facts_agent", "memory": "notes", "key": "Lisbon", "value": text}


def build(model, out_dir):
    def send_brief(text: str, to: str, tool_context: ToolContext) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        tool_context.state["effect"] = {"by": "writer", "target": str(path)}
        return str(path)

    config = types.GenerateContentConfig(temperature=0.2, max_output_tokens=1000)
    worker = dict(model=model, generate_content_config=config)
    weather_agent = LlmAgent(name="weather_agent", description="Finds the current weather of a city.",
                             tools=[get_weather], **worker,
                             instruction="You do not know the weather. Always call get_weather first, even if an "
                                         "earlier attempt failed, then answer in one sentence.")
    facts_agent = LlmAgent(name="facts_agent", description="Finds facts about a city.", tools=[search_facts],
                           after_agent_callback=save_note, **worker,
                           instruction="You do not know facts about cities. Always call search_facts first, then "
                                       "answer in one sentence.")
    translator_agent = LlmAgent(name="translator_agent", description="Translates a text into Portuguese.", **worker,
                                instruction="Translate the text you are given into Portuguese.")
    writer = LlmAgent(name="writer", description="Writes the brief.", **worker,
                      tools=[read_note, FunctionTool(send_brief, require_confirmation=True)],
                      instruction="You write two-sentence briefs about a city. Always call read_note for the city "
                                  "first. Reply with the brief only. When asked to send a brief, call send_brief "
                                  "with the brief and the recipient.")
    coordinator = LlmAgent(
        name="coordinator", sub_agents=[writer], **worker,
        tools=[AgentTool(weather_agent), AgentTool(facts_agent), AgentTool(translator_agent)],
        instruction="You plan a short brief on a city's weather and facts. In your first reply, call every agent "
                    "you need at once, in parallel. Then transfer to the writer with what they found.")
    evaluator = LlmAgent(name="evaluator", **worker,
                         instruction="Does the brief you are given mention the weather? Answer PASS or FAIL.")
    return coordinator, evaluator, [coordinator, writer, evaluator, weather_agent, facts_agent, translator_agent]


def definitions(agents):
    # what each agent can call or transfer to, read from the agent objects
    def tool(t):
        return {"name": t.name, "agent": t.agent.name if isinstance(t, AgentTool) else None,
                "require_confirmation": getattr(t, "_require_confirmation", False) is True}
    return {a.name: {"tools": [tool(t) for t in a.tools if hasattr(t, "name")] +
                     [{"name": t.__name__, "agent": None, "require_confirmation": False}
                      for t in a.tools if callable(t) and not hasattr(t, "name")],
                     "sub_agents": [s.name for s in a.sub_agents]} for a in agents}


def text(content):
    return "".join(p.text or "" for p in (content.parts or []) if not p.thought).strip() if content else ""


async def run(model, out_dir, run_id, events):
    plugins = [Native(events), ReflectAndRetryToolPlugin(max_retries=3)]
    coordinator, evaluator, agents = build(model, out_dir)
    events("agents", definitions=definitions(agents))
    sessions = InMemorySessionService()
    runner = Runner(app_name=APP, agent=coordinator, session_service=sessions, plugins=plugins)
    judge = Runner(app_name=APP, agent=evaluator, session_service=sessions, plugins=plugins)
    await sessions.create_session(app_name=APP, user_id="user", session_id=run_id)

    async def say(message, by=runner, sid=run_id):
        return [e async for e in by.run_async(user_id="user", session_id=sid, new_message=message)]

    def answer(turn):
        return next((text(e.content) for e in reversed(turn) if text(e.content)), "")

    def user(t):
        return types.Content(role="user", parts=[types.Part(text=t)])

    async def declare(name, data):
        event = Event(author="app", invocation_id=f"declare-{name}", actions=EventActions(state_delta={name: data}))
        await sessions.append_event(await sessions.get_session(app_name=APP, user_id="user", session_id=run_id), event)
        events("event", event=dump(event), invocation_id=event.invocation_id, session_id=run_id, branch=None)

    draft = answer(await say(user("Write a short brief on Lisbon's weather and facts.")))
    for iteration in range(1, 4):
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            await sessions.create_session(app_name=APP, user_id="user", session_id=f"{run_id}-check-{iteration}")
            verdict = answer(await say(user(draft), judge, f"{run_id}-check-{iteration}"))
            passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
        await declare("check", {"by": "evaluator", "checked": "draft", "iteration": iteration,
                                "outcome": "pass" if passed else "fail", "reason": reason})
        if passed:
            break
        draft = answer(await say(user(f"The draft failed review: {reason}. Revise it."))) or draft
    turn = await say(user("Send the brief to client."))
    asks = [p.function_call for e in turn for p in (e.content.parts if e.content else [])
            if p.function_call and p.function_call.name == "adk_request_confirmation"]
    for ask in asks:
        # a person reviews the pending send_brief call; the reference app approves on their behalf
        await say(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=ask.id, name="adk_request_confirmation", response={"confirmed": True}))]))


def main():
    model_id = llm.pick()
    run_id = "adk-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("adk", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "adk-reference")
    events = capture.Events(out / "events.jsonl")

    class Usage(CustomLogger):
        # ADK keeps token counts only; the provider's cost is visible one layer down, in LiteLLM
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            msg = response_obj.choices[0].message
            events("litellm", start=start_time.isoformat(), end=end_time.isoformat(),
                   usage=response_obj.usage.model_dump(),
                   tool_call_ids=[c.id for c in (msg.tool_calls or [])], text=(msg.content or "").strip())

    litellm.callbacks = [Usage()]
    events("run", run_id=run_id, model=model_id, framework="adk", versions={
        p: version(p) for p in ("google-adk", "litellm", "google-genai", "opentelemetry-sdk")})
    model = LiteLlm(model=f"openrouter/{model_id}", api_key=llm.key())
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
