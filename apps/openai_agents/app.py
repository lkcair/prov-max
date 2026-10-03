import asyncio
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents import (Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, RunHooks, Runner,
                    add_trace_processor, custom_span, function_tool, trace)
from agents.tool import get_function_tool_origin
from agents.tracing import TracingProcessor
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

from provai import capture, llm
from provai.config import OPENROUTER

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
weather_calls = 0


class Native(TracingProcessor):
    def __init__(self, events):
        self.events = events

    def on_trace_start(self, t):
        self.events("trace_start", **t.export())

    def on_trace_end(self, t):
        self.events("trace_end", trace_id=t.trace_id)

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        self.events("span", **span.export())

    def shutdown(self):
        pass

    def force_flush(self):
        pass


class Usage(RunHooks):
    def __init__(self, events):
        self.events = events

    async def on_llm_end(self, context, agent, response):
        self.events("model_end", agent=agent.name, request_id=response.request_id, usage=response.raw_usage)


@function_tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    global weather_calls
    weather_calls += 1
    if weather_calls == 1:
        raise TimeoutError("weather service timed out")
    return f"{city}: 21 C, clear sky"


@function_tool
def search_facts(city: str) -> str:
    """Facts about a city from the document store."""
    return DOCS.get(city, "no document found")


async def save_note(result):
    # the facts agent's answer goes to the shared notes, as in the reference task
    text = str(result.final_output)
    NOTES["Lisbon"] = text
    with custom_span("memory_write", {"by": "facts_agent", "memory": "notes", "key": "Lisbon", "value": text}):
        pass
    return text


@function_tool
def read_note(city: str) -> str:
    """Read the shared note about a city."""
    with custom_span("memory_read", {"by": "writer", "memory": "notes", "key": city}):
        pass
    return NOTES.get(city, "no note")


def build(model, out_dir, hooks):
    @function_tool(needs_approval=True)
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        with custom_span("effect", {"by": "writer", "target": str(path)}):
            pass
        return str(path)

    settings = ModelSettings(temperature=0.2, max_tokens=1000, preserve_raw_usage=True)
    # the first turn must call a tool; the SDK resets tool_choice to auto after it
    calls = ModelSettings(temperature=0.2, max_tokens=1000, preserve_raw_usage=True, tool_choice="required")
    worker = dict(model=model, model_settings=settings)
    caller = dict(model=model, model_settings=calls)
    weather_agent = Agent(name="weather_agent", tools=[get_weather], **caller,
                          instructions="You do not know the weather. Always call get_weather first, even if an "
                                       "earlier attempt failed, then answer in one sentence.")
    facts_agent = Agent(name="facts_agent", tools=[search_facts], **caller,
                        instructions="You do not know facts about cities. Always call search_facts first, then "
                                     "answer in one sentence.")
    translator_agent = Agent(name="translator_agent", **worker,
                             instructions="Translate the text you are given into Portuguese.")
    writer = Agent(name="writer", tools=[read_note, send_brief], **caller,
                   instructions="You write two-sentence briefs about a city. Always call read_note for the city "
                                "first. Reply with the brief only. When asked to send a brief, call send_brief "
                                "with the brief and the recipient.")
    coordinator = Agent(
        name="coordinator", handoffs=[writer], model=model,
        model_settings=ModelSettings(temperature=0.2, max_tokens=1000, preserve_raw_usage=True,
                                     parallel_tool_calls=True, tool_choice="required"),
        instructions="You plan a short brief on a city's weather and facts. In your first reply, call every "
                     "agent you need at once, in parallel. Then hand off to the writer with what they found.",
        tools=[weather_agent.as_tool("weather_agent", "Finds the current weather of a city.", hooks=hooks),
               facts_agent.as_tool("facts_agent", "Finds facts about a city.", hooks=hooks,
                                   custom_output_extractor=save_note),
               translator_agent.as_tool("translator_agent", "Translates a text into Portuguese.", hooks=hooks)])
    evaluator = Agent(name="evaluator", **worker,
                      instructions="Does the brief you are given mention the weather? Answer PASS or FAIL.")
    return coordinator, writer, evaluator, weather_agent, facts_agent, translator_agent


def definitions(agents):
    # what each agent can call, read from the agent objects; the trace does not say which tools are agents
    def tool(t):
        origin = get_function_tool_origin(t)
        return {"name": t.name, "origin": origin and origin.to_json_dict(), "needs_approval": t.needs_approval is True}
    return {a.name: {"tools": [tool(t) for t in a.tools], "handoffs": [h.name for h in a.handoffs]} for a in agents}


async def run(model, out_dir, run_id, events):
    client = AsyncOpenAI(base_url=OPENROUTER, api_key=llm.key())
    m = OpenAIChatCompletionsModel(model=model, openai_client=client)
    hooks = Usage(events)
    agents = build(m, out_dir, hooks)
    coordinator, writer, evaluator = agents[:3]
    events("agents", definitions=definitions(agents))
    config = RunConfig(workflow_name="city brief", group_id=run_id, trace_include_sensitive_data=True)
    with trace("city brief", group_id=run_id):
        result = await Runner.run(coordinator, "Write a short brief on Lisbon's weather and facts.",
                                  run_config=config, hooks=hooks)
        for iteration in range(1, 4):
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict = (await Runner.run(evaluator, result.final_output, run_config=config, hooks=hooks)).final_output
                passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
            with custom_span("check", {"by": "evaluator", "checked": "draft", "iteration": iteration,
                                       "outcome": "pass" if passed else "fail", "reason": reason}):
                pass
            if passed:
                break
            result = await Runner.run(writer, result.to_input_list() + [
                {"role": "user", "content": f"The draft failed review: {reason}. Revise it."}],
                run_config=config, hooks=hooks)
        result = await Runner.run(writer, result.to_input_list() + [
            {"role": "user", "content": "Send the brief to client."}], run_config=config, hooks=hooks)
        state = result.to_state()
        for item in result.interruptions:
            # a person reviews the pending send_brief call; the reference app approves on their behalf
            with custom_span("approval", {"by": "person", "checked": "draft", "outcome": "approve",
                                          "tool": item.tool_name, "call_id": item.raw_item.call_id}):
                state.approve(item)
        await Runner.run(writer, state, run_config=config, hooks=hooks)


def main():
    model = llm.pick()
    run_id = "oa-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("openai_agents", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "openai-agents-reference")
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    events = capture.Events(out / "events.jsonl")
    add_trace_processor(Native(events))
    events("run", run_id=run_id, model=model, framework="openai_agents", versions={
        p: version(p) for p in ("openai-agents", "openai", "opentelemetry-sdk",
                                "openinference-instrumentation-openai-agents")})
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
