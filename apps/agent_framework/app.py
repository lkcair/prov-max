import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental"

from agent_framework import (Agent, AgentExecutor, AgentExecutorRequest, Executor, Message, WorkflowBuilder,
                             WorkflowContext, agent_middleware, function_middleware, handler, tool)
from agent_framework.observability import enable_instrumentation
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_orchestrations import HandoffBuilder
from openai import AsyncOpenAI
from opentelemetry import trace

from provai import capture, llm
from provai.config import OPENROUTER

CANDIDATES = ["weather", "facts", "translator"]
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
weather_calls = 0
events = None


def span_id():
    return format(trace.get_current_span().get_span_context().span_id, "016x")


def plain(value):
    try:
        return value if isinstance(value, (str, int, float, bool, type(None))) else json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


@agent_middleware
async def agent_log(context, call_next):
    events("agent_start", agent=context.agent.name, span=span_id())
    await call_next()
    events("agent_end", agent=context.agent.name)


@function_middleware
async def function_log(context, call_next):
    events("function_start", name=context.function.name, arguments=plain(dict(context.arguments)), span=span_id())
    try:
        await call_next()
    except Exception as e:
        events("function_error", name=context.function.name, error=type(e).__name__, message=str(e))
        raise
    events("function_end", name=context.function.name, result=plain(context.result))


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
    events("memory_read", by="writer", memory="notes", key=city, span=span_id())
    return NOTES.get(city, "no note")


@dataclass
class Task:
    city: str
    selected: list


@dataclass
class Draft:
    city: str
    text: str
    iteration: int


@dataclass
class Revision:
    city: str
    text: str
    reason: str
    iteration: int


class Coordinator(Executor):
    def __init__(self, agent):
        super().__init__(id="coordinator")
        self.agent = agent

    @handler
    async def route(self, city: str, ctx: WorkflowContext[Task]) -> None:
        answer = (await self.agent.run(f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short brief on "
                                       f"{city}'s weather and facts? Answer with a JSON list of agent names.")).text
        try:
            selected = [a for a in json.loads(answer[answer.index("["):answer.rindex("]") + 1]) if a in CANDIDATES]
        except ValueError:
            selected = []
        selected = selected or ["weather", "facts"]
        events("routing", by="coordinator", reason=answer[:500], span=span_id())
        await ctx.send_message(Task(city=city, selected=selected))


def called_tool(response):
    return any(c.type == "function_call" for m in response.messages for c in m.contents)


class Worker(Executor):
    def __init__(self, id, agent, question, remember=False):
        super().__init__(id=id)
        self.agent, self.question, self.remember = agent, question, remember

    @handler
    async def work(self, task: Task, ctx: WorkflowContext[str]) -> None:
        question = self.question.format(city=task.city)
        response = await self.agent.run(question)
        for _ in range(2):
            # free models sometimes answer without the tool the task depends on; ask again, as a new run
            if called_tool(response):
                break
            response = await self.agent.run(f"{question} You must call your tool before answering.")
        text = response.text.strip()
        if self.remember:
            # the facts agent's answer goes to the shared notes, as in the reference task
            NOTES[task.city] = text
            events("memory_write", by=self.agent.name, memory="notes", key=task.city, value=text, span=span_id())
        await ctx.send_message(text)


class Gather(Executor):
    @handler
    async def gather(self, found: list[str], ctx: WorkflowContext[None, str]) -> None:
        await ctx.yield_output("\n".join(found))


class Evaluator(Executor):
    def __init__(self, agent):
        super().__init__(id="evaluator")
        self.agent = agent

    @handler
    async def review(self, draft: Draft, ctx: WorkflowContext[Revision | AgentExecutorRequest]) -> None:
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if draft.iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = (await self.agent.run(draft.text)).text
            passed, reason = "FAIL" not in verdict.upper() or draft.iteration == 3, verdict[:200]
        events("check", by="evaluator", checked="draft", iteration=draft.iteration,
               outcome="pass" if passed else "fail", reason=reason, span=span_id())
        if passed:
            await ctx.send_message(AgentExecutorRequest(messages=[Message(
                role="user", contents=[f"Send this brief on {draft.city} to client: {draft.text}"])]),
                target_id="send")
        else:
            await ctx.send_message(Revision(draft.city, draft.text, reason, draft.iteration), target_id="revise")


class Reviser(Executor):
    def __init__(self, agent):
        super().__init__(id="revise")
        self.agent = agent

    @handler
    async def revise(self, revision: Revision, ctx: WorkflowContext[Draft]) -> None:
        text = (await self.agent.run(f"Your brief on {revision.city}: {revision.text}\nThe draft failed review: "
                                     f"{revision.reason}. Revise it and reply with the brief only.")).text.strip()
        await ctx.send_message(Draft(revision.city, text or revision.text, revision.iteration + 1))


def build(model, out_dir):
    @tool(approval_mode="always_require")
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        events("effect", by="writer", target=str(path), span=span_id())
        return str(path)

    # free models get rate-limited upstream for a few seconds at a time
    openai = AsyncOpenAI(api_key=llm.key(), base_url=OPENROUTER, max_retries=8)
    client = OpenAIChatCompletionClient(model=model, async_client=openai, middleware=[function_log])
    settings = {"temperature": 0.2, "max_tokens": 1000}
    # the first turn must call a tool; the framework resets tool_choice to auto after it
    calls = {**settings, "tool_choice": "required"}

    def agent(name, instructions, tools=None, options=settings):
        return Agent(client, instructions, name=name, tools=tools, default_options=options, middleware=[agent_log],
                     require_per_service_call_history_persistence=True)

    agents = {
        "coordinator": agent("coordinator", "You plan a short brief on a city's weather and facts. When you are given "
                                            "what the agents found, hand off to the writer."),
        "weather_agent": agent("weather_agent", "You do not know the weather. Always call get_weather first, even if "
                                                "an earlier attempt failed, then answer in one sentence.",
                               [get_weather], calls),
        "facts_agent": agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                            "answer in one sentence.", [search_facts], calls),
        "translator_agent": agent("translator_agent", "Translate the text you are given into Portuguese."),
        "writer": agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city "
                                  "first. Reply with the brief only.", [read_note], calls),
        # the same writer, with the tool that acts outside the system, for the release step only
        # a resumed run starts again with the default options, so a required tool would be called twice
        "sender": agent("writer", "You send city briefs. Call send_brief with the brief you are given and the "
                                  "recipient.", [send_brief]),
        "evaluator": agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL."),
    }
    return agents


def definitions(agents, workflows):
    # what each agent can call and how each workflow is wired, read from the objects
    return {"agents": {key: {"name": a.name, "tools": [{"name": t.name, "approval_mode": getattr(t, "approval_mode", None)}
                                                      for t in a.default_options.get("tools", [])]}
                       for key, a in agents.items()},
            "workflows": {name: {"executors": {i: getattr(getattr(e, "agent", None), "name", None)
                                               for i, e in w.executors.items()}, "edge_groups": [
                {"type": type(g).__name__, "sources": list(g.source_executor_ids),
                 "targets": list(g.target_executor_ids)} for g in w.edge_groups]} for name, w in workflows.items()}}


async def run_workflow(workflow, label, **kwargs):
    # non-streaming runs report one output event per agent answer, streaming runs one per token update
    result = await workflow.run(**kwargs)
    for e in result:
        data = e.data
        events("workflow_event", workflow=label, type=e.type, executor=e.executor_id, iteration=e.iteration,
               request_id=e.request_id if e.type == "request_info" else None, state=e.state and e.state.value,
               data_type=type(data).__name__ if data is not None else None,
               data=plain({"source": data.source, "target": data.target}) if e.type == "handoff_sent" else
               plain(getattr(data, "text", None)) if e.type == "output" else None)
    return list(result)


async def run(model, out_dir):
    agents = build(model, out_dir)
    a = agents
    coordinator = Coordinator(a["coordinator"])
    workers = [Worker("weather", a["weather_agent"], "Weather in {city}?"),
               Worker("facts", a["facts_agent"], "Facts about {city}?", remember=True),
               Worker("translator", a["translator_agent"], "Translate a note on {city}.")]
    route = (WorkflowBuilder(name="route", start_executor=coordinator)
             .add_multi_selection_edge_group(coordinator, workers,
                                             lambda task, targets: [t for t in targets if t in task.selected])
             .add_fan_in_edges(workers[:2], Gather(id="gather"))
             .build())
    handoff = (HandoffBuilder(name="handoff", participants=[a["coordinator"], a["writer"]])
               .with_start_agent(a["coordinator"])
               .add_handoff(a["coordinator"], [a["writer"]])
               .with_termination_condition(lambda conversation: any(
                   m.author_name == "writer" and m.role == "assistant" and m.text for m in conversation))
               .build())
    evaluator, reviser = Evaluator(a["evaluator"]), Reviser(a["writer"])
    review = (WorkflowBuilder(name="review", start_executor=evaluator)
              .add_edge(evaluator, reviser)
              .add_edge(reviser, evaluator)
              .add_edge(evaluator, AgentExecutor(a["sender"], id="send"))
              .build())
    events("agents", definitions=definitions(agents, {"route": route, "handoff": handoff, "review": review}))

    found = [e.data for e in await run_workflow(route, "route", message="Lisbon") if e.type == "output"]
    turn = await run_workflow(handoff, "handoff", message=f"What the agents found about Lisbon:\n{found[-1]}\n"
                                                          "Hand off to the writer.")
    draft = next(e.data.text for e in reversed(turn) if e.type == "output" and getattr(e.data, "text", None)
                 and e.executor_id == "writer")
    turn = await run_workflow(review, "review", message=Draft("Lisbon", draft, 1))
    asks = [e for e in turn if e.type == "request_info"]
    for ask in asks:
        # a person reviews the pending send_brief call; the reference app approves on their behalf
        events("approval", by="person", checked="draft", outcome="approve", request_id=ask.request_id,
               call_id=ask.data.function_call.call_id, tool=ask.data.function_call.name)
    if asks:
        await run_workflow(review, "review", responses={
            ask.request_id: ask.data.to_function_approval_response(approved=True) for ask in asks})


def main():
    global events
    model = llm.pick()
    run_id = "maf-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("agent_framework", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "agent-framework-reference")
    enable_instrumentation(enable_sensitive_data=True)
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="agent_framework", versions={
        p: version(p) for p in ("agent-framework-core", "agent-framework-openai", "agent-framework-orchestrations",
                                "openai", "opentelemetry-sdk")})
    asyncio.run(run(model, out))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
