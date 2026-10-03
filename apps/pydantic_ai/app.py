import asyncio
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic_ai import Agent, CustomEvent, DeferredToolRequests, DeferredToolResults, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (DeferredToolRequestsEvent, DeferredToolResultsEvent, FunctionToolCallEvent,
                                  FunctionToolResultEvent)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from provai import capture, llm

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
weather_calls = 0


@dataclass(kw_only=True)
class MemoryWrite(CustomEvent):
    by: str
    memory: str
    key: str
    value: str


@dataclass(kw_only=True)
class MemoryRead(CustomEvent):
    by: str
    memory: str
    key: str


@dataclass(kw_only=True)
class Handoff(CustomEvent):
    source: str
    target: str


@dataclass(kw_only=True)
class Check(CustomEvent):
    by: str
    checked: str
    iteration: int
    outcome: str
    reason: str


@dataclass(kw_only=True)
class Approval(CustomEvent):
    by: str
    outcome: str
    approved_call: str


@dataclass(kw_only=True)
class Effect(CustomEvent):
    by: str
    target: str


class Native(AbstractCapability):
    def __init__(self, events, first_tool=False):
        self.events = events
        self.first_tool = first_tool

    def get_model_settings(self):
        # the first model request must call a tool; later requests may answer
        if self.first_tool:
            return lambda ctx: {"tool_choice": "required"} if ctx.run_step <= 1 and ctx.prompt else {}

    async def before_run(self, ctx):
        self.events("run_start", run_id=ctx.run_id, agent=ctx.agent.name, conversation_id=ctx.conversation_id)

    async def after_run(self, ctx, *, result):
        self.events("run_end", run_id=ctx.run_id, agent=ctx.agent.name, output=result.output)
        return result

    async def on_run_error(self, ctx, *, error):
        self.events("run_error", run_id=ctx.run_id, agent=ctx.agent.name, error=type(error).__name__)
        raise error

    async def after_model_request(self, ctx, *, request_context, response):
        self.events("model_end", run_id=ctx.run_id, agent=ctx.agent.name, run_step=ctx.run_step,
                    response_id=response.provider_response_id, provider=response.provider_name,
                    provider_details=response.provider_details, usage=asdict(response.usage))
        return response

    async def on_event(self, ctx, *, event):
        base = dict(run_id=ctx.run_id, agent=ctx.agent.name, run_step=ctx.run_step)
        if isinstance(event, FunctionToolCallEvent):
            self.events("tool_call", **base, tool_call_id=event.part.tool_call_id, tool=event.part.tool_name,
                        args=event.part.args_as_dict())
        elif isinstance(event, FunctionToolResultEvent):
            self.events("tool_result", **base, tool_call_id=event.part.tool_call_id, tool=event.part.tool_name,
                        part=event.part.part_kind, content=event.part.content)
        elif isinstance(event, DeferredToolRequestsEvent):
            self.events("deferred_requests", **base, approvals=[
                {"tool_call_id": c.tool_call_id, "tool": c.tool_name, "args": c.args_as_dict()}
                for c in event.requests.approvals])
        elif isinstance(event, DeferredToolResultsEvent):
            self.events("deferred_results", **base, approvals={k: str(v) for k, v in event.results.approvals.items()})
        elif isinstance(event, CustomEvent):
            self.events(event.name, **base, tool_call_id=event.tool_call_id, tool=event.tool_name,
                        **{k: v for k, v in asdict(event).items() if k not in ("name", "tool_call_id", "tool_name", "event_kind")})


def fetch_weather(city):
    global weather_calls
    weather_calls += 1
    if weather_calls == 1:
        raise TimeoutError("weather service timed out")
    return f"{city}: 21 C, clear sky"


@dataclass
class Review:
    iteration: int


def must_call(agent, tool):
    # free models sometimes write the tool call as text; send them back until they call the tool
    @agent.output_validator
    def called(ctx: RunContext, output):
        if isinstance(output, str) and not any(p.part_kind == "tool-call" and p.tool_name == tool for m in ctx.messages for p in m.parts):
            raise ModelRetry(f"Call {tool} first.")
        return output


def build(model, out_dir, events):
    settings = OpenRouterModelSettings(temperature=0.2, max_tokens=1000)

    def agent(name, instructions, first_tool=False, **kw):
        return Agent(model, name=name, instructions=instructions, model_settings=settings,
                     capabilities=[Native(events, first_tool)], **kw)

    weather = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
                                     "earlier attempt failed, then answer in one sentence.", True)
    facts = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                 "answer in one sentence.", True)
    translator = agent("translator_agent", "Translate the text you are given into Portuguese.")
    writer = agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city "
                             "first. Reply with the brief only. When asked to send a brief, call send_brief with "
                             "the brief and the recipient.", True, output_type=[str, DeferredToolRequests])
    evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.",
                      deps_type=Review)

    must_call(weather, "get_weather")
    must_call(facts, "search_facts")
    must_call(writer, "read_note")

    @weather.tool_plain
    def get_weather(city: str) -> str:
        """Current weather for a city."""
        try:
            return fetch_weather(city)
        except TimeoutError as e:
            raise ModelRetry(str(e))

    @facts.tool_plain
    def search_facts(city: str) -> str:
        """Facts about a city from the document store."""
        return DOCS.get(city, "no document found")

    @writer.tool
    async def read_note(ctx: RunContext, city: str) -> str:
        """Read the shared note about a city."""
        await ctx.emit(MemoryRead(by="writer", memory="notes", key=city))
        return NOTES.get(city, "no note")

    @writer.tool(requires_approval=True)
    async def send_brief(ctx: RunContext, text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        await ctx.emit(Effect(by="writer", target=str(path)))
        return str(path)

    @evaluator.output_validator
    async def verdict(ctx: RunContext[Review], output: str) -> str:
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if ctx.deps.iteration == 1:
            outcome, reason = "fail", "the first draft is always sent back for revision"
        else:
            failed = "FAIL" in output.upper() and ctx.deps.iteration < 3
            outcome, reason = ("fail" if failed else "pass"), output[:200]
        await ctx.emit(Check(by="evaluator", checked="draft", iteration=ctx.deps.iteration, outcome=outcome,
                             reason=reason))
        return outcome if outcome == "pass" else reason

    state = {}

    async def hand_off_to_writer(ctx: RunContext, weather_report: str, city_facts: str) -> str:
        """Hand the worker results to the writer, who drafts the brief."""
        await ctx.emit(Handoff(source="coordinator", target="writer"))
        result = await writer.run(f"Write a two-sentence brief on Lisbon. Weather: {weather_report}. "
                                  f"Facts: {city_facts}.", usage=ctx.usage, conversation_id=ctx.conversation_id)
        state["writer"] = result
        return result.output

    coordinator = agent("coordinator", "You plan a short brief on a city's weather and facts. In your first reply, "
                                       "call every agent you need at once, in parallel. Then hand off to the "
                                       "writer with what they found.", True, output_type=[hand_off_to_writer])

    @coordinator.tool
    async def weather_agent(ctx: RunContext, city: str) -> str:
        """Finds the current weather of a city."""
        result = await weather.run(f"Weather in {city}?", usage=ctx.usage, conversation_id=ctx.conversation_id)
        return result.output

    @coordinator.tool
    async def facts_agent(ctx: RunContext, city: str) -> str:
        """Finds facts about a city."""
        result = await facts.run(f"Facts about {city}?", usage=ctx.usage, conversation_id=ctx.conversation_id)
        # the facts agent's answer goes to the shared notes, as in the reference task
        NOTES[city] = result.output
        await ctx.emit(MemoryWrite(by="facts_agent", memory="notes", key=city, value=result.output))
        return result.output

    @coordinator.tool
    async def translator_agent(ctx: RunContext, text: str) -> str:
        """Translates a text into Portuguese."""
        result = await translator.run(text, usage=ctx.usage, conversation_id=ctx.conversation_id)
        return result.output

    return coordinator, writer, evaluator, state


async def run(model, out_dir, run_id, events):
    m = OpenRouterModel(model, provider=OpenRouterProvider(api_key=llm.key()))
    coordinator, writer, evaluator, state = build(m, out_dir, events)
    events("agents", delegates={"coordinator": {"weather_agent": "weather_agent", "facts_agent": "facts_agent",
                                                "translator_agent": "translator_agent"}},
           hands_off={"coordinator": {"hand_off_to_writer": "writer"}})
    await coordinator.run("Write a short brief on Lisbon's weather and facts.", conversation_id=run_id)
    draft = state["writer"]
    for iteration in range(1, 4):
        review = await evaluator.run(draft.output, deps=Review(iteration), conversation_id=run_id)
        if review.output == "pass":
            break
        draft = await writer.run(f"The draft failed review: {review.output}. Revise it.",
                                 message_history=draft.all_messages(), conversation_id=run_id)
    held = await writer.run("Send the brief to client.", message_history=draft.all_messages(), conversation_id=run_id)
    approvals = {c.tool_call_id: True for c in held.output.approvals}
    # a person reviews the held send_brief call; the reference app approves on their behalf
    async with writer.iter(message_history=held.all_messages(), conversation_id=run_id,
                           deferred_tool_results=DeferredToolResults(approvals=approvals)) as resumed:
        for call_id in approvals:
            await resumed.emit(Approval(by="person", outcome="approve", approved_call=call_id))
        async for _ in resumed:
            pass


def main():
    model = llm.pick()
    run_id = "pa-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("pydantic_ai", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "pydantic-ai-reference")
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider, version=6))
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="pydantic_ai", versions={
        p: version(p) for p in ("pydantic-ai-slim", "pydantic-graph", "openai", "opentelemetry-sdk")})
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
