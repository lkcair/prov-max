import asyncio
import contextvars
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline, component, tracing
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.hooks.human_in_the_loop import AlwaysAskPolicy, BlockingConfirmationStrategy, ConfirmationHook
from haystack.hooks.human_in_the_loop.dataclasses import ConfirmationUIResult
from haystack.tools import tool
from haystack.utils import Secret
from openinference.instrumentation import using_session
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry import trace as otel

from provai import capture, llm
from provai.config import OPENROUTER

CANDIDATES = ["weather_agent", "facts_agent", "translator_agent"]
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = InMemoryDocumentStore()
weather_calls = 0
current = contextvars.ContextVar("haystack_span", default=None)


def plain(value):
    if isinstance(value, (ChatMessage, Document)):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return getattr(value, "name", None) or repr(value)


class NativeSpan(tracing.Span):
    def __init__(self, name, parent):
        self.id, self.name, self.parent, self.tags = uuid.uuid4().hex, name, parent, {}

    def set_tag(self, key, value):
        self.tags[key] = value

    def raw_span(self):
        return self


class Native(tracing.Tracer):
    # Haystack's own tracing interface: every pipeline, component, agent, tool, and hook span with its tags
    def __init__(self, events):
        self.events = events

    @contextmanager
    def trace(self, operation_name, tags=None, parent_span=None):
        parent = parent_span or current.get()
        span = NativeSpan(operation_name, parent.id if parent else None)
        span.tags.update(tags or {})
        token = current.set(span)
        otel_span = otel.get_current_span().get_span_context().span_id
        start = datetime.now(timezone.utc).isoformat()
        error = None
        try:
            yield span
        except Exception as e:
            error = type(e).__name__
            raise
        finally:
            current.reset(token)
            self.events("span", id=span.id, parent=span.parent, name=operation_name, start=start,
                        end=datetime.now(timezone.utc).isoformat(), otel_span=format(otel_span, "016x"),
                        error=error, tags=plain(span.tags))

    def current_span(self):
        return current.get()


def declare(name, **tags):
    # facts the application states, through Haystack's own tracer
    with tracing.tracer.trace(f"app.{name}", tags=tags):
        pass


def text_of(message):
    return (message.text or "").strip() if message else ""


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
    declare("memory_read", by="writer", memory="notes", key=city)
    docs = NOTES.filter_documents({"field": "meta.city", "operator": "==", "value": city})
    return docs[0].content if docs else "no note"


class SaveNote:
    # the facts agent's answer goes to the shared notes, as in the reference task
    def run(self, state):
        text = text_of(state.data["messages"][-1])
        NOTES.write_documents([Document(id="notes-Lisbon", content=text, meta={"city": "Lisbon"})],
                              policy=DuplicatePolicy.OVERWRITE)
        declare("memory_write", by="facts_agent", memory="notes", key="Lisbon", value=text)


class Person:
    # a person reviews every send_brief call; the reference app approves on their behalf
    def get_user_confirmation(self, tool_name, tool_description, tool_params):
        declare("approval", by="person", tool=tool_name, outcome="approve")
        return ConfirmationUIResult(action="confirm")

    def to_dict(self):
        return {"type": "Person"}


@component
class Coordinator:
    def __init__(self, chat):
        self.chat = chat

    @component.output_types(weather_agent=list[ChatMessage], facts_agent=list[ChatMessage],
                            translator_agent=list[ChatMessage])
    def run(self, city: str):
        prompt = (f"Agents: weather, facts, translator. Which agents are needed for a short brief on {city}'s "
                  "weather and facts? Answer with a JSON list of agent names.")
        answer = text_of(self.chat.run([ChatMessage.from_user(prompt)])["replies"][0])
        chosen = [a for a in CANDIDATES if a.split("_")[0] in answer.lower()] or CANDIDATES[:2]
        declare("routing", by="coordinator", reason=answer[:500])
        tasks = {"weather_agent": f"Weather in {city}?", "facts_agent": f"Facts about {city}?",
                 "translator_agent": f"Translate a brief on {city}."}
        return {a: [ChatMessage.from_user(tasks[a])] for a in chosen}


@component
class Handoff:
    # the weather and facts answers meet here and the work passes from the coordinator to the writer
    @component.output_types(messages=list[ChatMessage])
    def run(self, weather: ChatMessage, facts: ChatMessage):
        declare("handoff", source="coordinator", target="writer")
        task = (f"Write a two-sentence brief on Lisbon. Weather: {text_of(weather)}. Facts: {text_of(facts)}. "
                "Call read_note for Lisbon first. Reply with the brief only.")
        return {"messages": [ChatMessage.from_user(task)]}


@component
class Evaluator:
    def __init__(self, chat):
        self.chat, self.iteration = chat, 0

    @component.output_types(outcome=str, revise=list[ChatMessage], brief=list[ChatMessage])
    def run(self, draft: ChatMessage):
        self.iteration += 1
        text = text_of(draft)
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if self.iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = text_of(self.chat.run([ChatMessage.from_user(
                f"Does this brief mention the weather? Answer PASS or FAIL.\n{text}")])["replies"][0])
            passed, reason = "FAIL" not in verdict.upper() or self.iteration == 3, verdict[:200]
        declare("check", by="evaluator", checked="draft", iteration=self.iteration,
                outcome="pass" if passed else "fail", reason=reason)
        return {"outcome": "pass" if passed else "fail",
                "revise": [ChatMessage.from_user(f"The draft failed review: {reason}. Revise it, calling "
                                                 f"read_note for Lisbon first:\n{text}")],
                "brief": [ChatMessage.from_user(f"Send this brief to client with send_brief:\n{text}")]}


def build(model, out_dir):
    settings = {"temperature": 0.2, "max_tokens": 1000}

    def chat():
        return OpenAIChatGenerator(api_key=Secret.from_token(llm.key()), model=model, api_base_url=OPENROUTER,
                                   generation_kwargs=settings)

    @tool
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        declare("effect", by="writer", target=str(path))
        return str(path)

    worker = "Answer in one sentence after calling your tool."
    weather_agent = Agent(chat_generator=chat(), tools=[get_weather],
                          system_prompt="You do not know the weather. Always call get_weather first, even if an "
                                        "earlier attempt failed. " + worker)
    facts_agent = Agent(chat_generator=chat(), tools=[search_facts], hooks={"after_run": [SaveNote()]},
                        system_prompt="You do not know facts about cities. Always call search_facts first. " + worker)
    translator_agent = Agent(chat_generator=chat(), system_prompt="Translate the text you are given into Portuguese.")
    writer = Agent(chat_generator=chat(), tools=[read_note],
                   system_prompt="You write two-sentence briefs about a city. Always call read_note for the city "
                                 "first. Reply with the brief only.")
    confirm = ConfirmationHook(confirmation_strategies={
        "send_brief": BlockingConfirmationStrategy(confirmation_policy=AlwaysAskPolicy(), confirmation_ui=Person())})
    # one send per brief: no parallel tool calls, and the run ends once send_brief has run
    sender = Agent(chat_generator=OpenAIChatGenerator(api_key=Secret.from_token(llm.key()), model=model,
                                                      api_base_url=OPENROUTER,
                                                      generation_kwargs={**settings, "parallel_tool_calls": False}),
                   tools=[send_brief], hooks={"before_tool": [confirm]}, exit_conditions=["send_brief"],
                   system_prompt="You send briefs. Call send_brief exactly once, with the brief you are given and "
                                 "to=client.")
    review = ConditionalRouter(routes=[
        {"condition": "{{ outcome == 'fail' }}", "output": "{{ revise }}", "output_name": "revise",
         "output_type": list[ChatMessage]},
        {"condition": "{{ outcome == 'pass' }}", "output": "{{ brief }}", "output_name": "release",
         "output_type": list[ChatMessage]}], unsafe=True)

    p = Pipeline(max_runs_per_component=4)
    p.add_component("coordinator", Coordinator(chat()))
    p.add_component("weather_agent", weather_agent)
    p.add_component("facts_agent", facts_agent)
    p.add_component("translator_agent", translator_agent)
    p.add_component("handoff", Handoff())
    p.add_component("draft_input", BranchJoiner(list[ChatMessage]))
    p.add_component("writer", writer)
    p.add_component("evaluator", Evaluator(chat()))
    p.add_component("review", review)
    p.add_component("sender", sender)
    for a in CANDIDATES:
        p.connect(f"coordinator.{a}", f"{a}.messages")
    p.connect("weather_agent.last_message", "handoff.weather")
    p.connect("facts_agent.last_message", "handoff.facts")
    p.connect("handoff.messages", "draft_input")
    p.connect("review.revise", "draft_input")
    p.connect("draft_input", "writer.messages")
    p.connect("writer.last_message", "evaluator.draft")
    p.connect("evaluator.outcome", "review.outcome")
    p.connect("evaluator.revise", "review.revise")
    p.connect("evaluator.brief", "review.brief")
    p.connect("review.release", "sender.messages")
    return p


def main():
    model = llm.pick()
    run_id = "hs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("haystack", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "haystack-reference")
    HaystackInstrumentor().instrument(tracer_provider=provider)
    events = capture.Events(out / "events.jsonl")
    tracing.enable_tracing(Native(events))
    events("run", run_id=run_id, model=model, framework="haystack", versions={
        p: version(p) for p in ("haystack-ai", "openai", "opentelemetry-sdk", "openinference-instrumentation-haystack")})
    pipeline = build(model, out)
    # the components, their agents, and the connections, read from the pipeline object
    definition = {"components": {n: type(pipeline.get_component(n)).__name__ for n in pipeline.graph.nodes},
                  "edges": [[u, v, d["from_socket"].name, d["to_socket"].name]
                            for u, v, d in pipeline.graph.edges(data=True)],
                  "tools": {n: [t.name for t in (pipeline.get_component(n).tools or [])]
                            for n in pipeline.graph.nodes if isinstance(pipeline.get_component(n), Agent)}}
    models = {name: {"model": model, "temperature": 0.2, "max_tokens": 1000}
              for name in ("coordinator", *CANDIDATES, "writer", "evaluator", "sender")}
    models["sender"]["parallel_tool_calls"] = False
    events("pipeline", definition=definition, models=models)
    with using_session(run_id):
        asyncio.run(pipeline.run_async({"coordinator": {"city": "Lisbon"}}))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
