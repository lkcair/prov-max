import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.update(CREWAI_DISABLE_TELEMETRY="true", CREWAI_DISABLE_TRACKING="true", CREWAI_TRACING_ENABLED="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crewai import LLM, Agent, Crew, Process, Task
from crewai.events import crewai_event_bus
from crewai.events.base_events import BaseEvent
from crewai.events.types.flow_events import HumanFeedbackReceivedEvent, HumanFeedbackRequestedEvent
from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.flow.human_feedback import human_feedback
from crewai.memory.unified_memory import Memory
from crewai.telemetry.tracing import TraceSession
from crewai.tools import tool
from pydantic import BaseModel

from provai import capture, llm

CANDIDATES = ("weather", "facts", "translator")
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
weather_calls = 0


class CheckEvent(BaseEvent):
    type: str = "check"
    by: str
    checked: str
    iteration: int
    outcome: str
    reason: str


def embed(texts):
    # a local bag-of-words embedding, so memory needs no embedding service
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


def record_all(events):
    def every(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from every(sub)

    def write(source, event):
        data = json.loads(json.dumps(event.to_json(), default=str))
        for key in ("agent", "from_agent", "from_task", "task", "crew", "source"):
            data.pop(key, None)
        events("event", event_class=type(event).__name__, source_class=type(source).__name__, **data)

    for cls in set(every(BaseEvent)):
        crewai_event_bus.on(cls)(write)


@tool("get_weather")
def get_weather(city: str) -> str:
    """Current weather for a city."""
    global weather_calls
    weather_calls += 1
    if weather_calls == 1:
        raise TimeoutError("weather service timed out")
    return f"{city}: 21 C, clear sky"


@tool("search_facts")
def search_facts(city: str) -> str:
    """Facts about a city from the document store."""
    return DOCS.get(city, "no document found")


class Providers:
    # hands CrewAI's trace session our tracer provider, so its spans go to spans.jsonl
    def __init__(self, tracer_provider):
        self.tracer_provider = tracer_provider

    def get_tracer(self, name=None):
        return self.tracer_provider.get_tracer(name or "crewai", version("crewai"))

    def emit_log(self, *args, **kwargs):
        pass

    def flush(self, timeout_millis=30000):
        return self.tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis=30000):
        return True


class Person:
    # a person reviews the draft; the reference app approves on their behalf and reports it
    # with the same events CrewAI's console provider emits
    def request_feedback(self, context, flow):
        crewai_event_bus.emit(flow, HumanFeedbackRequestedEvent(
            flow_name=flow.name or type(flow).__name__, method_name=context.method_name,
            output=context.method_output, message=context.message, emit=context.emit))
        crewai_event_bus.emit(flow, HumanFeedbackReceivedEvent(
            flow_name=flow.name or type(flow).__name__, method_name=context.method_name, feedback="approve"))
        return "approve"


class BriefState(BaseModel):
    city: str = "Lisbon"
    selected: list[str] = []
    reason: str = ""
    weather: str = ""
    facts: str = ""
    draft: str = ""
    iteration: int = 0
    verdict: str = ""
    sent: str = ""


def build(model, out_dir):
    chat = LLM(model=f"openrouter/{model}", api_key=llm.key(), temperature=0.2, max_tokens=1000)
    notes = Memory(llm=chat, embedder=embed, storage=str(out_dir / "notes"))

    @tool("read_note")
    def read_note(city: str) -> str:
        """Read the shared note about a city."""
        hits = notes.recall(f"{city} facts", scope="/notes", depth="shallow")
        return hits[0].record.content if hits else "no note"

    @tool("send_brief")
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        return str(path)

    def agent(role, goal, tools=(), delegation=False):
        return Agent(role=role, goal=goal, backstory=goal, llm=chat, tools=list(tools), allow_delegation=delegation,
                     max_iter=6, verbose=False)

    coordinator = agent("coordinator", "Plan briefs on cities and hand the work to the writer.", delegation=True)
    weather_agent = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
                                           "earlier attempt failed, then answer in one sentence.", [get_weather])
    facts_agent = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                       "answer in one sentence.", [search_facts])
    writer = agent("writer", "Write two-sentence briefs about cities. Always call read_note for the city first.",
                   [read_note])
    sender = agent("writer", "Send approved briefs with send_brief.", [send_brief])
    evaluator = agent("evaluator", "Check whether a brief mentions the weather. Answer PASS or FAIL.")
    translator = agent("translator_agent", "Translate texts into Portuguese.")

    class BriefFlow(Flow[BriefState]):
        @start()
        def coordinate(self):
            answer = coordinator.kickoff(
                f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short brief on {self.state.city}'s "
                "weather and facts? Answer with a JSON list of agent names.").raw
            try:
                chosen = [a for a in json.loads(answer[answer.index("["):answer.rindex("]") + 1]) if a in CANDIDATES]
            except ValueError:
                chosen = []
            self.state.selected = chosen or ["weather", "facts"]
            self.state.reason = answer[:500]

        @router(coordinate, emit=["research", "translation"])
        def route(self):
            return "translation" if self.state.selected == ["translator"] else "research"

        @listen("research")
        async def weather(self):
            out = await weather_agent.kickoff_async(f"Weather in {self.state.city}?")
            self.state.weather = out.raw

        @listen("research")
        async def facts(self):
            out = await facts_agent.kickoff_async(f"Facts about {self.state.city}?")
            self.state.facts = out.raw
            notes.remember(out.raw, scope="/notes", categories=["notes"], importance=0.5,
                           metadata={"key": self.state.city}, agent_role="facts_agent")

        @listen("translation")
        def translate(self):
            self.state.draft = translator.kickoff(f"Translate a brief on {self.state.city} into Portuguese.").raw

        @listen(and_(weather, facts))
        def handoff(self):
            task = Task(description=f"Delegate the drafting of a two-sentence brief on {self.state.city} to the writer "
                                    "with the Delegate work to coworker tool, passing these findings as context. "
                                    f"Weather: {self.state.weather}. Facts: {self.state.facts}. "
                                    "Return the writer's brief only.",
                        expected_output="the writer's two-sentence brief", agent=coordinator)
            crew = Crew(agents=[coordinator, writer], tasks=[task], process=Process.sequential, verbose=False)
            self.state.draft = crew.kickoff().raw

        @listen("revision")
        def revise(self):
            self.state.draft = writer.kickoff(
                f"Write a two-sentence brief on {self.state.city}. Weather: {self.state.weather}. "
                f"Facts: {self.state.facts}. The last draft failed review: {self.state.verdict}. Fix it. "
                f"Last draft: {self.state.draft}. Reply with the brief only.").raw
            return self.state.draft

        @router(or_(handoff, revise), emit=["approval", "revision"])
        def evaluate(self):
            self.state.iteration += 1
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if self.state.iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict = evaluator.kickoff(f"Does this brief mention the weather? Answer PASS or FAIL.\n"
                                            f"{self.state.draft}").raw
                passed, reason = "FAIL" not in verdict.upper() or self.state.iteration == 3, verdict[:200]
            crewai_event_bus.emit(self, CheckEvent(by="evaluator", checked="draft", iteration=self.state.iteration,
                                                   outcome="pass" if passed else "fail", reason=reason))
            self.state.verdict = "" if passed else reason
            return "approval" if passed else "revision"

        @listen("approval")
        @human_feedback(message="Approve sending this brief to the client?", provider=Person())
        def review(self):
            return self.state.draft

        @listen(review)
        def send(self):
            if self.last_human_feedback.feedback.strip().lower() != "approve":
                return
            # CrewAI always sends tool_choice "auto", so the sender is asked again until the brief is sent
            for _ in range(3):
                out = sender.kickoff(f"Call send_brief with to='client' and this brief as text, unchanged:\n"
                                     f"{self.state.draft}")
                if (out_dir / "brief-to-client.txt").exists():
                    break
            self.state.sent = out.raw

    return BriefFlow, [coordinator, weather_agent, facts_agent, translator, writer, evaluator]


def main():
    model = llm.pick()
    run_id = "cr-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("crewai", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "crewai-reference")
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="crewai", versions={
        p: version(p) for p in ("crewai", "openai", "opentelemetry-sdk")})
    record_all(events)
    flow_class, agents = build(model, out)
    flow = flow_class()
    events("flow", definition=flow_class.flow_definition().model_dump(mode="json", exclude_none=True),
           agents={a.role: [t.name for t in a.tools] for a in agents},
           method_agents={"coordinate": "coordinator", "weather": "weather_agent", "facts": "facts_agent",
                          "translate": "translator_agent", "handoff": "coordinator", "revise": "writer",
                          "evaluate": "evaluator", "send": "writer"})
    session = TraceSession(run_id, providers=Providers(provider))
    with session.activate():
        flow.kickoff(inputs={"city": "Lisbon"})
    session.shutdown()
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
