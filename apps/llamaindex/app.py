import asyncio
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from openinference.instrumentation import using_session
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from workflows import Context, Workflow, step
from workflows.events import (Event, HumanResponseEvent, InputRequiredEvent, StartEvent, StepStateChanged,
                              StopEvent)
from workflows.retry_policy import retry_if_exception_type, retry_policy, stop_after_attempt, wait_fixed

from provai import capture, llm
from provai.config import OPENROUTER

CANDIDATES = ["weather", "facts", "translator"]
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
dispatcher = get_dispatcher(__name__)
weather_calls = 0
EVENTS = None


class Declared(BaseEvent):
    kind: str
    data: dict

    @classmethod
    def class_name(cls):
        return "Declared"


def declare(kind, **data):
    dispatcher.event(Declared(kind=kind, data=data))


def plain(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


class NativeSpans(BaseSpanHandler[SimpleSpan]):
    def new_span(self, id_, bound_args, instance=None, parent_span_id=None, tags=None, **kw):
        meta = getattr(instance, "metadata", None)
        EVENTS("span_enter", id=id_, parent=parent_span_id, instance=type(instance).__name__ if instance else None,
               name=getattr(instance, "name", None) or getattr(meta, "name", None), tags=plain(tags or {}))
        return SimpleSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})

    def prepare_to_exit_span(self, id_, bound_args, instance=None, result=None, **kw):
        EVENTS("span_exit", id=id_)
        return self.open_spans.get(id_)

    def prepare_to_drop_span(self, id_, bound_args, instance=None, err=None, **kw):
        EVENTS("span_drop", id=id_, error=type(err).__name__ if err else None, message=str(err)[:300] if err else None)
        return self.open_spans.get(id_)


class NativeEvents(BaseEventHandler):
    @classmethod
    def class_name(cls):
        return "NativeEvents"

    def handle(self, event, **kw):
        name = event.class_name()
        if name.endswith("InProgressEvent"):
            return
        row = {"span_id": event.span_id}
        if name == "LLMChatStartEvent":
            row["model"] = plain(event.model_dict)
        elif name == "LLMChatEndEvent" and event.response is not None:
            raw = event.response.raw
            usage = getattr(raw, "usage", None)
            row["usage"] = usage.model_dump() if usage is not None else None
            row["response_id"] = getattr(raw, "id", None)
            msg = event.response.message
            row["content"] = msg.content
            row["tool_calls"] = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                                 for c in msg.additional_kwargs.get("tool_calls") or []]
        elif name == "AgentToolCallEvent":
            row.update(tool=event.tool.name, arguments=event.arguments)
        elif name == "ExceptionEvent":
            row["error"] = type(event.exception).__name__
        elif name == "Declared":
            name = event.kind
            row.update(event.data)
        EVENTS(name, **row)


class WeatherEvent(Event):
    city: str


class FactsEvent(Event):
    city: str


class TranslateEvent(Event):
    text: str


class WeatherDone(Event):
    text: str


class FactsDone(Event):
    text: str


class TranslatorDone(Event):
    text: str


class Draft(Event):
    text: str
    iteration: int


class Revise(Event):
    text: str
    reason: str
    iteration: int


class ApprovalRequest(InputRequiredEvent):
    draft: str


def describe(agent):
    # what an agent can call and whom it can hand off to, read from the agent object
    EVENTS("agent", name=agent.name, tools=[t.metadata.name for t in agent.tools or []],
           can_handoff_to=agent.can_handoff_to)


def build(model, out_dir, log_agent_event):
    chat = OpenAILike(model=model, api_base=OPENROUTER, api_key=llm.key(), is_chat_model=True,
                      is_function_calling_model=True, temperature=0.2, max_tokens=1000, context_window=128000,
                      max_retries=10)
    # the sampling settings live on the LLM object; the chat events carry only its metadata
    EVENTS("llm", **{k: getattr(chat, k) for k in ("model", "temperature", "max_tokens", "api_base")})

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

    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        declare("effect", by="writer", target=str(path))
        return str(path)

    def agent(name, prompt, tools, **kw):
        kw.setdefault("initial_tool_choice", "required" if tools else None)
        made = FunctionAgent(name=name, description=f"The {name}.", system_prompt=prompt, tools=tools, llm=chat,
                             streaming=False, **kw)
        describe(made)
        return made

    weather_agent = agent("weather_agent", "You do not know the weather. Call get_weather exactly once. If it "
                                           "returns an error, reply with the single word ERROR. Otherwise answer "
                                           "in one sentence.", [get_weather])
    facts_agent = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                       "answer in one sentence.", [search_facts])
    sender = FunctionTool.from_defaults(send_brief)

    def writer_for(ctx):
        async def read_note(city: str) -> str:
            """Read the shared note about a city."""
            notes = await ctx.store.get("notes", default={})
            declare("memory_read", by="writer", memory="notes", key=city)
            return notes.get(city, "no note")
        return agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city "
                               "first. Reply with the brief only.", [read_note])

    class CityBrief(Workflow):
        @step
        async def coordinator(self, ctx: Context, ev: StartEvent) -> WeatherEvent | FactsEvent | TranslateEvent | None:
            prompt = (f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short brief on "
                      f"{ev.city}'s weather and facts? Answer with a JSON list of agent names.")
            answer = (await chat.achat([ChatMessage(role="user", content=prompt)])).message.content or ""
            try:
                selected = [a for a in json.loads(answer[answer.index("["):answer.rindex("]") + 1]) if a in CANDIDATES]
            except ValueError:
                selected = []
            selected = selected or ["weather", "facts"]
            await ctx.store.set("city", ev.city)
            await ctx.store.set("selected", selected)
            for name in selected:
                ctx.send_event({"weather": WeatherEvent(city=ev.city), "facts": FactsEvent(city=ev.city),
                                "translator": TranslateEvent(text=ev.city)}[name])
            return None

        @step(retry_policy=retry_policy(retry=retry_if_exception_type(TimeoutError), wait=wait_fixed(0.5),
                                       stop=stop_after_attempt(3)))
        async def weather(self, ctx: Context, ev: WeatherEvent) -> WeatherDone:
            out = await weather_agent.run(user_msg=f"Weather in {ev.city}?")
            failed = [c for c in out.tool_calls if c.tool_name == "get_weather" and c.tool_output.is_error]
            ok = [c for c in out.tool_calls if c.tool_name == "get_weather" and not c.tool_output.is_error]
            if failed and not ok:
                # the tool failed, so the step fails and the workflow's retry policy runs it again
                raise TimeoutError("weather service timed out")
            return WeatherDone(text=str(out.response.content))

        @step
        async def facts(self, ctx: Context, ev: FactsEvent) -> FactsDone:
            out = await facts_agent.run(user_msg=f"Facts about {ev.city}?")
            text = str(out.response.content).strip()
            notes = await ctx.store.get("notes", default={})
            notes[ev.city] = text
            await ctx.store.set("notes", notes)
            declare("memory_write", by="facts_agent", memory="notes", key=ev.city, value=text)
            return FactsDone(text=text)

        @step
        async def translator(self, ctx: Context, ev: TranslateEvent) -> TranslatorDone:
            return TranslatorDone(text=ev.text)

        @step
        async def merge(self, ctx: Context, ev: WeatherDone | FactsDone | TranslatorDone) -> Draft | None:
            done = {"weather": WeatherDone, "facts": FactsDone, "translator": TranslatorDone}
            got = ctx.collect_events(ev, [done[n] for n in await ctx.store.get("selected")])
            if got is None:
                return None
            found = {type(e): e.text for e in got}
            weather, facts = found.get(WeatherDone, ""), found.get(FactsDone, "")
            city = await ctx.store.get("city")
            coordinator = agent("coordinator", "You pass the gathered weather and facts to the writer. Call "
                                               "handoff with to_agent writer and a short reason.", [],
                                initial_tool_choice="required", can_handoff_to=["writer"])
            team = AgentWorkflow(agents=[coordinator, writer_for(ctx)], root_agent="coordinator")
            handler = team.run(user_msg=f"Write a two-sentence brief on {city}. Weather: {weather}. Facts: {facts}.")
            async for e in handler.stream_events():
                log_agent_event(e)
            result = await handler
            return Draft(text=str(result.response.content).strip(), iteration=1)

        @step
        async def evaluate(self, ctx: Context, ev: Draft) -> Revise | ApprovalRequest:
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if ev.iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict = (await chat.achat([ChatMessage(
                    role="user", content=f"Does this brief mention the weather? Answer PASS or FAIL.\n{ev.text}")]
                )).message.content or ""
                passed, reason = "FAIL" not in verdict.upper() or ev.iteration == 3, verdict[:200]
            declare("check", by="evaluator", checked="draft", iteration=ev.iteration,
                    outcome="pass" if passed else "fail", reason=reason)
            if not passed:
                return Revise(text=ev.text, reason=reason, iteration=ev.iteration)
            await ctx.store.set("draft", ev.text)
            return ApprovalRequest(prefix="Approve sending the brief to the client?", draft=ev.text)

        @step
        async def revise(self, ctx: Context, ev: Revise) -> Draft:
            city = await ctx.store.get("city")
            out = await writer_for(ctx).run(user_msg=f"The draft brief on {city} failed review: {ev.reason}. "
                                                     f"Revise it:\n{ev.text}")
            return Draft(text=str(out.response.content).strip(), iteration=ev.iteration + 1)

        @step
        async def send(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
            if ev.response != "approve":
                return StopEvent(result="not sent")
            draft = await ctx.store.get("draft")
            out = await sender.acall(text=draft, to="client")
            return StopEvent(result=str(out))

    return CityBrief(timeout=900)


def definitions(wf):
    # the steps and the events each accepts and returns, read from the workflow object
    def names(types):
        return [getattr(t, "__name__", str(t)) for t in types]
    return {name: {"accepts": names(cfg.accepted_events), "returns": names(cfg.return_types),
                   "retry": cfg.retry_policy is not None}
            for name, cfg in wf._step_configs().items()}


async def run(model, out_dir, run_id, events):
    def agent_event(e):
        name = type(e).__name__
        if name == "AgentStream":
            return
        row = {k: plain(getattr(e, k)) for k in ("current_agent_name", "tool_name", "tool_kwargs", "tool_id")
               if hasattr(e, k)}
        if hasattr(e, "tool_output"):
            row.update(output=str(e.tool_output.content), is_error=e.tool_output.is_error)
        if hasattr(e, "response") and hasattr(e.response, "content"):
            row["response"] = e.response.content
        events("agent_" + name, **row)

    wf = build(model, out_dir, agent_event)
    events("workflow", definitions=definitions(wf))
    with using_session(run_id):
        handler = wf.run(city="Lisbon")
        async for ev in handler.stream_events(expose_internal=True):
            if isinstance(ev, StepStateChanged):
                events("step_state", name=ev.name, state=ev.step_state.value, worker_id=ev.worker_id,
                       input=ev.input_event_name, output=ev.output_event_name)
            elif isinstance(ev, ApprovalRequest):
                events("input_required", prefix=ev.prefix, draft=ev.draft)
                # a person approves on the reference app's behalf
                events("human_response", by="person", response="approve")
                handler.ctx.send_event(HumanResponseEvent(response="approve"))
        await handler


def main():
    model = llm.pick()
    run_id = "li-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("llamaindex", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "llamaindex-reference")
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)
    global EVENTS
    events = EVENTS = capture.Events(out / "events.jsonl")
    root = get_dispatcher()
    root.add_span_handler(NativeSpans())
    root.add_event_handler(NativeEvents())
    events("run", run_id=run_id, model=model, framework="llamaindex", versions={
        p: version(p) for p in ("llama-index-core", "llama-index-workflows", "llama-index-llms-openai-like",
                                "openai", "opentelemetry-sdk", "openinference-instrumentation-llama-index")})
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
