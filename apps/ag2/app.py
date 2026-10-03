import asyncio
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ag2 import Agent, tool
from ag2.config.openai import OpenAIConfig
from ag2.events import (HumanInputRequest, HumanMessage, ModelResponse, TaskCompleted, TaskFailed, TaskStarted,
                        ToolApprovalRequest, ToolCallEvent, ToolErrorEvent, ToolResultEvent)
from ag2.middleware.builtin import TelemetryMiddleware, approval_required
from ag2.observers import observer
from ag2.tools.subagents import subagent_tool

from provai import capture, llm
from provai.config import OPENROUTER

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
SETTINGS = {"temperature": 0.2, "max_tokens": 1000}
weather_calls = 0


def text(value):
    return value if isinstance(value, str) else json.dumps(value, default=str)


def rendered(result):
    # a tool result is a list of input parts; text parts carry their content
    result = getattr(result, "result", result)
    parts = getattr(result, "parts", None)
    if parts is None:
        return text(result)
    return " ".join(str(getattr(p, "content", p)) for p in parts)


def native(events, agent):
    # every event on an agent's stream, with ids and the fields a record needs; never a config object
    def log(event):
        row = {"agent": agent, "event_id": getattr(event, "id", None), "created_at": getattr(event, "created_at", None)}
        if isinstance(event, ModelResponse):
            u = event.usage
            row.update(response_id=event.response_id, model=event.model, provider=event.provider,
                       finish_reason=event.finish_reason, message=event.message.content if event.message else None,
                       tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in event.tool_calls.calls],
                       usage={"input": u.prompt_tokens, "output": u.completion_tokens})
        elif isinstance(event, ToolCallEvent):
            row.update(call_id=event.id, name=event.name, arguments=event.arguments)
        elif isinstance(event, ToolErrorEvent):
            row.update(call_id=event.parent_id, name=event.name, error=type(event.error).__name__,
                       message=str(event.error))
        elif isinstance(event, ToolResultEvent):
            row.update(call_id=event.parent_id, name=event.name, result=rendered(event))
        elif isinstance(event, (TaskStarted, TaskCompleted, TaskFailed)):
            row.update(task_id=event.task_id, task_agent=event.agent_name, objective=event.objective)
            if isinstance(event, TaskCompleted):
                row.update(result=text(event.result))
            if isinstance(event, TaskFailed):
                row.update(error=type(event.error).__name__)
        elif isinstance(event, ToolApprovalRequest):
            row.update(request_id=event.id, call_id=event.tool_call_id, prompt=event.content)
        elif isinstance(event, HumanInputRequest):
            row.update(request_id=event.id, prompt=event.content)
        elif isinstance(event, HumanMessage):
            row.update(request_id=event.parent_id, content=event.content)
        else:
            return
        events(type(event).__name__, **row)
    return observer(None, log)


def build(model, out_dir, events, provider, run_id):
    config = OpenAIConfig(model=model, api_key=llm.key(), base_url=OPENROUTER, parallel_tool_calls=True, **SETTINGS)

    def agent(name, prompt, tools=(), **kw):
        span = TelemetryMiddleware(tracer_provider=provider, agent_name=name, provider_name="openrouter",
                                   model_name=model, span_attributes={"gen_ai.conversation.id": run_id})
        return Agent(name, prompt, config=config, tools=list(tools), middleware=[span],
                     observers=[native(events, name)], **kw)

    @tool
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        global weather_calls
        weather_calls += 1
        if weather_calls == 1:
            raise TimeoutError("weather service timed out")
        return f"{city}: 21 C, clear sky"

    @tool
    def search_facts(city: str) -> str:
        """Facts about a city from the document store."""
        return DOCS.get(city, "no document found")

    @tool
    def read_note(city: str) -> str:
        """Read the shared note about a city."""
        events("memory_read", by="writer", memory="notes", key=city)
        return NOTES.get(city, "no note")

    @tool(middleware=[approval_required(allow_always=False)])
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        events("effect", by="writer", target=str(path))
        return str(path)

    async def save_note(call_next, event, context):
        # the facts agent's answer goes to the shared notes, as in the reference task
        result = await call_next(event, context)
        answer = rendered(result)
        NOTES["Lisbon"] = answer
        events("memory_write", by="facts_agent", memory="notes", key="Lisbon", value=answer, call_id=event.id)
        return result

    weather_agent = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
                                           "earlier attempt failed, then answer in one sentence.", [get_weather])
    facts_agent = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                       "answer in one sentence.", [search_facts])
    translator_agent = agent("translator_agent", "Translate the text you are given into Portuguese.")
    writer = agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city "
                             "first. Reply with the brief only.", [read_note])
    coordinator = agent(
        "coordinator",
        "You plan a short brief on a city's weather and facts. In your first reply, call task_weather_agent and "
        "task_facts_agent together, in parallel. Then call task_writer with what they found. Then reply with the "
        "writer's brief only.",
        [subagent_tool(weather_agent, description="Finds the current weather of a city."),
         subagent_tool(facts_agent, description="Finds facts about a city.", middleware=[save_note]),
         subagent_tool(translator_agent, description="Translates a text into Portuguese."),
         subagent_tool(writer, description="Writes the brief from the weather and facts it is given.")])
    evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.")

    def person(request):
        # a person reviews the held send_brief call; the reference app approves on their behalf
        events("approval", by="person", checked="draft", outcome="approve",
               call_id=getattr(request, "tool_call_id", None), request_id=request.id)
        return "y"

    sender = agent("writer", "You send briefs. Call send_brief with the brief you are given and the recipient.",
                   [send_brief], hitl_hook=person)
    return coordinator, writer, evaluator, sender, [coordinator, weather_agent, facts_agent, translator_agent, writer,
                                                    evaluator, sender]


def definitions(agents):
    # what each agent can call, read from the agent objects; a task_ tool runs the agent it names
    return {f"{a.name}#{i}": {"name": a.name, "tools": [t.name for t in getattr(a, "tools", ())]}
            for i, a in enumerate(agents)}


async def run(model, out_dir, run_id, events, provider):
    coordinator, writer, evaluator, sender, agents = build(model, out_dir, events, provider, run_id)
    events("agents", definitions=definitions(agents))
    events("models", config={a.name: {"model": model, **SETTINGS} for a in agents})
    draft = (await coordinator.ask("Write a short brief on Lisbon's weather and facts.")).body or ""
    for iteration in range(1, 4):
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = (await evaluator.ask(draft)).body or ""
            passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
        events("check", by="evaluator", checked="draft", iteration=iteration,
               outcome="pass" if passed else "fail", reason=reason)
        if passed:
            break
        draft = (await writer.ask(f"The draft failed review: {reason}. Revise it:\n{draft}")).body or draft
    await sender.ask(f"Send this brief to client:\n{draft}")


def main():
    model = llm.pick()
    run_id = "ag2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("ag2", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "ag2-reference")
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="ag2",
           versions={p: version(p) for p in ("ag2", "openai", "opentelemetry-sdk")})
    asyncio.run(run(model, out, run_id, events, provider))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
