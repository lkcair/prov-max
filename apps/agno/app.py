import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["AGNO_TELEMETRY"] = "false"

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openrouter import OpenRouter
from agno.run.agent import CustomEvent as AgentEvent
from agno.run.workflow import CustomEvent
from agno.team import Team
from agno.team.mode import TeamMode
from agno.tools import tool
from agno.workflow import Loop, Parallel, Router, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from openinference.instrumentation.agno import AgnoInstrumentor

from provai import capture, llm

CANDIDATES = ["weather", "facts", "translator"]
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOISE = {"RunContent", "RunIntermediateContent", "ReasoningContentDelta", "TeamRunContent",
         "TeamRunIntermediateContent", "TeamReasoningContentDelta", "StepOutput"}
weather_calls = 0


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


def read_note(run_context, city: str):
    """Read the shared note about a city."""
    yield AgentEvent(name="memory_read", by="writer", memory="notes", key=city, created_at=int(time.time()))
    yield ((run_context.session_state or {}).get("notes") or {}).get(city, "no note")


def text(event):
    content = getattr(event, "content", None)
    return content if isinstance(content, str) else ""


def build(model, out_dir, log):
    def openrouter():
        return OpenRouter(id=model, api_key=llm.key(), temperature=0.2, max_tokens=1000)

    @tool(requires_confirmation=True)
    def send_brief(text: str, to: str):
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        yield AgentEvent(name="effect", by="writer", target=str(path), created_at=int(time.time()))
        yield str(path)

    def agent(name, prompt, tools=()):
        return Agent(name=name, model=openrouter(), instructions=prompt, tools=list(tools), telemetry=False)

    coordinator = agent("coordinator", f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short brief "
                                       "on a city's weather and facts? Answer with a JSON list of agent names.")
    weather_agent = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
                                           "earlier attempt failed, then answer in one sentence.", [get_weather])
    facts_agent = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                       "answer in one sentence.", [search_facts])
    translator_agent = agent("translator_agent", "Translate the text you are given into Portuguese.")
    writer = agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city first. "
                             "Reply with the brief only.", [read_note])
    evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.")
    sender = agent("writer", "You send the brief you are given: call send_brief with the brief and the recipient.",
                   [send_brief])
    # a paused run is continued by its id, which Agno looks up in the agent's database
    sender.db = InMemoryDb()
    log("models", config={a.name: {"id": a.model.id, "temperature": a.model.temperature,
                                   "max_tokens": a.model.max_tokens}
                          for a in (coordinator, weather_agent, facts_agent, translator_agent, writer, evaluator)})

    async def run_agent(a, prompt, need_tool=False, need_word=None, **kw):
        # a free model sometimes skips its tool or loses the topic; ask again, at most three times
        for _ in range(3):
            answer, called = "", False
            async for ev in a.arun(prompt, stream=True, stream_events=True, **kw):
                yield ev
                if type(ev).__name__ == "ToolCallCompletedEvent" and not ev.tool.tool_call_error:
                    called = True
                if type(ev).__name__ == "RunCompletedEvent":
                    answer = text(ev)
            if (called or not need_tool) and (not need_word or need_word in answer):
                break
        yield StepOutput(content=answer)

    async def weather(step_input: StepInput, run_context):
        async for ev in run_agent(weather_agent, "Weather in Lisbon?", need_tool=True,
                                  session_state=run_context.session_state):
            yield ev

    async def facts(step_input: StepInput, run_context):
        answer = ""
        async for ev in run_agent(facts_agent, "Facts about Lisbon?", need_tool=True,
                                  session_state=run_context.session_state):
            if isinstance(ev, StepOutput):
                answer = ev.content.strip()
                continue
            yield ev
        run_context.session_state.setdefault("notes", {})["Lisbon"] = answer
        yield CustomEvent(name="memory_write", by="facts_agent", memory="notes", key="Lisbon", value=answer,
                          created_at=int(time.time()))
        yield StepOutput(content=answer)

    async def translator(step_input: StepInput, run_context):
        async for ev in run_agent(translator_agent, step_input.input or ""):
            yield ev

    steps = {"weather": Step(name="weather", executor=weather), "facts": Step(name="facts", executor=facts),
             "translator": Step(name="translator", executor=translator)}

    async def route(step_input: StepInput):
        answer = (await coordinator.arun(str(step_input.input))).content or ""
        try:
            chosen = [a for a in json.loads(answer[answer.index("["):answer.rindex("]") + 1]) if a in CANDIDATES]
        except ValueError:
            chosen = []
        chosen = chosen or ["weather", "facts"]
        log("routing", by="coordinator", answer=answer[:500], chosen=chosen,
            agents={"weather": "weather_agent", "facts": "facts_agent", "translator": "translator_agent"})
        # a Router runs a list of steps one after another, so the selection runs as one Parallel step
        return Parallel(*[steps[c] for c in chosen], name="workers")

    handoff = Team(name="coordinator", mode=TeamMode.route, members=[writer], model=openrouter(), telemetry=False,
                   instructions="Route the gathered weather and facts to the writer.")

    async def hand_off(step_input: StepInput, run_context):
        prompt = f"Write a two-sentence brief on Lisbon from these findings:\n{step_input.previous_step_content}"
        draft = ""
        async for ev in handoff.arun(prompt, stream=True, stream_events=True, session_state=run_context.session_state):
            yield ev
            # the team run ends with a completion event of its own, named after the team
            if type(ev).__name__.endswith("RunCompletedEvent") and getattr(ev, "team_name", None) == "coordinator":
                draft = text(ev)
        run_context.session_state["draft"] = draft.strip()
        yield StepOutput(content=draft)

    async def evaluate(step_input: StepInput, run_context):
        state = run_context.session_state
        state["iteration"] = state.get("iteration", 0) + 1
        iteration = state["iteration"]
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = ""
            async for ev in run_agent(evaluator, state["draft"]):
                if isinstance(ev, StepOutput):
                    verdict = ev.content
                else:
                    yield ev
            passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
        state["verdict"] = "" if passed else reason
        yield CustomEvent(name="check", by="evaluator", checked="draft", iteration=iteration,
                          outcome="pass" if passed else "fail", reason=reason, created_at=int(time.time()))
        yield StepOutput(content="PASS" if passed else "FAIL")

    async def revise(step_input: StepInput, run_context):
        state = run_context.session_state
        if not state.get("verdict"):
            yield StepOutput(content=state["draft"])
            return
        draft = ""
        prompt = (f"Revise this two-sentence brief on Lisbon. It failed review: {state['verdict']}\n"
                  f"Brief:\n{state['draft']}\nReply with the revised brief on Lisbon only.")
        async for ev in run_agent(writer, prompt, need_word="Lisbon", session_state=state):
            if isinstance(ev, StepOutput):
                draft = ev.content
            else:
                yield ev
        state["draft"] = draft.strip()
        yield StepOutput(content=draft)

    def passed(outputs):
        return any(o.step_name == "evaluate" and o.content == "PASS" for o in outputs)

    workflow = Workflow(name="city brief", telemetry=False, session_state={"notes": {}}, steps=[
        Router(name="coordinator", choices=list(steps.values()), selector=route),
        Step(name="handoff", executor=hand_off),
        Loop(name="review", max_iterations=3, end_condition=passed,
             steps=[Step(name="evaluate", executor=evaluate), Step(name="revise", executor=revise)]),
    ])
    return workflow, sender


async def run(model, out_dir, run_id, log):
    workflow, sender = build(model, out_dir, log)
    log("workflow", definition=workflow.to_dict())

    def record(ev):
        name = type(ev).__name__.removesuffix("Event")
        if name not in NOISE:
            # custom events keep their fields as plain attributes, outside the dataclass fields
            data = vars(ev) if name == "Custom" else ev.to_dict()
            log(name, **json.loads(json.dumps(data, default=str)))

    draft = ""
    async for ev in workflow.arun("Write a short brief on Lisbon's weather and facts.", stream=True,
                                  stream_events=True, session_id=run_id, user_id="user"):
        record(ev)
        if type(ev).__name__ == "WorkflowCompletedEvent":
            draft = text(ev).strip()
    paused = None
    for _ in range(3):
        async for ev in sender.arun(f"Call send_brief with to=client and this text:\n{draft}", stream=True,
                                    stream_events=True, session_id=run_id, user_id="user"):
            record(ev)
            if type(ev).__name__ == "RunPausedEvent":
                paused = ev
        if paused:
            break
    if paused:
        for req in paused.active_requirements:
            # a person reviews the held send_brief call; the reference app approves on their behalf
            req.confirm()
        async for ev in sender.acontinue_run(run_id=paused.run_id, requirements=paused.requirements, stream=True,
                                             stream_events=True, session_id=run_id, user_id="user"):
            record(ev)


def main():
    model = llm.pick()
    run_id = "ag-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("agno", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "agno-reference")
    AgnoInstrumentor().instrument(tracer_provider=provider)
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="agno", versions={
        p: version(p) for p in ("agno", "openai", "opentelemetry-sdk", "openinference-instrumentation-agno")})
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
