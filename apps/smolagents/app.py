import sys
import threading
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from smolagents import (ActionStep, FinalAnswerStep, LogLevel, OpenAIServerModel, PlanningStep, ToolCallingAgent,
                        tool)

from provai import capture, llm
from provai.config import OPENROUTER

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
lock = threading.Lock()
weather_calls = 0
events = None


@tool
def get_weather(city: str) -> str:
    """Current weather for a city.

    Args:
        city: The city.
    """
    global weather_calls
    with lock:
        weather_calls += 1
        first = weather_calls == 1
    if first:
        raise TimeoutError("weather service timed out")
    return f"{city}: 21 C, clear sky"


@tool
def search_facts(city: str) -> str:
    """Facts about a city from the document store.

    Args:
        city: The city.
    """
    return DOCS.get(city, "no document found")


@tool
def read_note(city: str) -> str:
    """Read the shared note about a city.

    Args:
        city: The city.
    """
    events("memory_read", by="writer", memory="notes", key=city)
    return NOTES.get(city, "no note")


def text_of(message):
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content


def native(agent_name):
    # every step an agent finishes, as smolagents keeps it in the agent's memory
    def log(step, agent=None):
        if isinstance(step, ActionStep):
            raw = getattr(step.model_output_message, "raw", None)
            usage = getattr(raw, "usage", None)
            events("action_step", agent=agent_name, step=step.step_number,
                   start=step.timing.start_time, end=step.timing.end_time,
                   tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in step.tool_calls or []],
                   output_text=text_of(step.model_output_message), observations=step.observations,
                   error={"type": type(step.error).__name__, "message": str(step.error)} if step.error else None,
                   tokens=step.token_usage.dict() if step.token_usage else None,
                   usage=usage.model_dump() if usage is not None else None,
                   response_id=getattr(raw, "id", None), is_final_answer=step.is_final_answer)
        elif isinstance(step, PlanningStep):
            events("planning_step", agent=agent_name, plan=step.plan)
        elif isinstance(step, FinalAnswerStep):
            events("final_answer", agent=agent_name, output=str(step.output))
    return {ActionStep: log, PlanningStep: log, FinalAnswerStep: log}


def build(model, out_dir):
    settings = dict(temperature=0.2, max_tokens=1000)
    m = OpenAIServerModel(model_id=model, api_base=OPENROUTER, api_key=llm.key(), **settings)
    quiet = dict(model=m, verbosity_level=LogLevel.OFF)

    weather_agent = ToolCallingAgent(
        tools=[get_weather], name="weather_agent", description="Finds the current weather of a city.",
        instructions="You do not know the weather. Always call get_weather first, even if an earlier attempt "
                     "failed, then answer in one sentence.", step_callbacks=native("weather_agent"), max_steps=5,
        **quiet)
    facts_agent = ToolCallingAgent(
        tools=[search_facts], name="facts_agent", description="Finds facts about a city.",
        instructions="You do not know facts about cities. Always call search_facts first, then answer in one "
                     "sentence.", step_callbacks=native("facts_agent"), max_steps=5, **quiet)
    translator_agent = ToolCallingAgent(
        tools=[], name="translator_agent", description="Translates a text into Portuguese.",
        step_callbacks=native("translator_agent"), max_steps=3, **quiet)

    def save_note(step, agent=None):
        # the facts agent's answer goes to the shared notes, as in the reference task
        text = str(step.output)
        NOTES["Lisbon"] = text
        events("memory_write", by="facts_agent", memory="notes", key="Lisbon", value=text)
    facts_agent.step_callbacks.register(FinalAnswerStep, save_note)

    evaluator = ToolCallingAgent(
        tools=[], name="evaluator", instructions="Does the brief you are given mention the weather? Answer PASS or "
                                                 "FAIL.", step_callbacks=native("evaluator"), max_steps=2, **quiet)
    reviews = []

    def review(answer, memory, agent=None):
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        iteration = len(reviews) + 1
        if iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = str(evaluator.run(f"Brief: {answer}"))
            passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
        reviews.append(passed)
        events("check", by="evaluator", checked="draft", iteration=iteration, outcome="pass" if passed else "fail",
               reason=reason, draft=str(answer))
        if not passed:
            raise ValueError(f"the draft failed review: {reason}. Revise it")
        return True

    writer = ToolCallingAgent(
        tools=[read_note], name="writer", description="Writes a two-sentence brief about a city from what it is "
                                                     "given.",
        instructions="You write two-sentence briefs about a city. Always call read_note for the city first, then "
                     "give the brief as your final answer.", final_answer_checks=[review],
        step_callbacks=native("writer"), max_steps=8, **quiet)

    coordinator = ToolCallingAgent(
        tools=[], managed_agents=[weather_agent, facts_agent, translator_agent, writer], name="coordinator",
        instructions="You plan a short brief on a city's weather and facts. In your first step, call every team "
                     "member you need at once, in the same step. Then call the writer with what they found, and "
                     "give the writer's brief as your final answer.",
        step_callbacks=native("coordinator"), max_steps=8, **quiet)

    @tool
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient.

        Args:
            text: The brief.
            to: The recipient.
        """
        # a person reviews every send before it happens; the reference app approves on their behalf
        events("approval", by="person", checked="send_brief", outcome="approve", text=text, to=to)
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        events("effect", by="writer", target=str(path))
        return str(path)

    agents = [coordinator, weather_agent, facts_agent, translator_agent, writer, evaluator]
    return coordinator, writer, send_brief, agents, settings


def definitions(agents):
    # what each agent can call and which agents it manages, read from the agent objects
    return {a.name: {"tools": [t for t in a.tools if t != "final_answer"], "managed_agents": list(a.managed_agents),
                     "final_answer_checks": [f.__name__ for f in a.final_answer_checks]} for a in agents}


def main():
    global events
    model = llm.pick()
    run_id = "sm-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("smolagents", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "smolagents-reference")
    SmolagentsInstrumentor().instrument(tracer_provider=provider)
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="smolagents", versions={
        p: version(p) for p in ("smolagents", "openai", "opentelemetry-sdk", "openinference-instrumentation-smolagents")})
    coordinator, writer, send_brief, agents, settings = build(model, out)
    # smolagents has no handoff; the app declares that the coordinator's call of the writer passes the work on
    events("agents", definitions=definitions(agents), settings=settings, handoff={"from": "coordinator", "to": "writer"})
    brief = coordinator.run("Write a short brief on Lisbon's weather and facts.")
    # the writer gets the send tool only once the brief has passed review
    writer.tools["send_brief"] = send_brief
    writer.final_answer_checks = []
    writer.run(f"Send this brief to client with send_brief, then answer with the file path:\n{brief}", reset=False)
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
