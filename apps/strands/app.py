import asyncio
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_latest_experimental,gen_ai_span_attributes_only"

from opentelemetry import trace

from provai import capture, llm
from provai.config import OPENROUTER

CANDIDATES = ["weather", "facts", "translator"]
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
weather_calls = 0


def span_id():
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx.is_valid else None


def declare(name, **attrs):
    with trace.get_tracer("city-brief").start_as_current_span(name, attributes={f"app.{k}": v for k, v in attrs.items()}):
        pass


def build(model, out_dir, events, run_id):
    from strands import Agent, tool
    from strands.hooks import (AfterInvocationEvent, AfterModelCallEvent, AfterNodeCallEvent, AfterToolCallEvent,
                               BeforeInvocationEvent, BeforeModelCallEvent, BeforeNodeCallEvent, BeforeToolCallEvent,
                               HookProvider)
    from strands.memory import MemoryManager
    from strands.memory.types import MemoryInjectionConfig
    from strands.models.openai import OpenAIModel
    from strands.multiagent import GraphBuilder, Swarm
    from strands.vended_memory_stores.test_memory_store import TestMemoryStore

    class Native(HookProvider):
        def register_hooks(self, registry, **kw):
            for kind in (BeforeInvocationEvent, AfterInvocationEvent, BeforeModelCallEvent, AfterModelCallEvent,
                         BeforeToolCallEvent, AfterToolCallEvent, BeforeNodeCallEvent, AfterNodeCallEvent):
                registry.add_callback(kind, self.log)

        def log(self, event):
            row = {"span_id": span_id()}
            agent = getattr(event, "agent", None)
            if agent is not None:
                row["agent"] = agent.name
            if hasattr(event, "node_id"):
                row["node_id"] = event.node_id
                row["source"] = type(event.source).__name__
            tool_use = getattr(event, "tool_use", None)
            if tool_use:
                row.update(tool_use_id=tool_use["toolUseId"], tool=tool_use["name"], input=tool_use["input"])
            if isinstance(event, AfterToolCallEvent):
                row.update(status=event.result.get("status"), retry=event.retry,
                           exception=type(event.exception).__name__ if event.exception else None)
            if isinstance(event, AfterModelCallEvent):
                stop = event.stop_response
                row.update(stop_reason=str(stop.stop_reason) if stop else None,
                           exception=type(event.exception).__name__ if event.exception else None)
            if isinstance(event, AfterInvocationEvent) and event.result is not None:
                row.update(stop_reason=str(event.result.stop_reason),
                           interrupts=[i.to_dict() for i in event.result.interrupts or []])
            events(type(event).__name__, **row)

    class RetryTimeouts(HookProvider):
        def register_hooks(self, registry, **kw):
            registry.add_callback(AfterToolCallEvent, self.retry)

        def retry(self, event):
            if isinstance(event.exception, TimeoutError):
                event.retry = True

    class Approval(HookProvider):
        # a person reviews every send_brief call before it runs
        def register_hooks(self, registry, **kw):
            registry.add_callback(BeforeToolCallEvent, self.ask)

        def ask(self, event):
            if event.tool_use["name"] != "send_brief":
                return
            answer = event.interrupt("approval", reason={"draft": event.tool_use["input"].get("text")})
            events("approval", span_id=span_id(), tool_use_id=event.tool_use["toolUseId"], by="person",
                   outcome=answer)
            if answer != "approve":
                event.cancel_tool = "the person rejected the brief"

    class SaveNote(HookProvider):
        # the facts agent's answer goes to the shared notes, as in the reference task
        def register_hooks(self, registry, **kw):
            registry.add_callback(AfterInvocationEvent, self.save)

        async def save(self, event):
            text = str(event.result).strip()
            await memory.add(text)

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
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        declare("effect", by="writer", target=str(path))
        return str(path)

    def openrouter():
        return OpenAIModel(client_args={"api_key": llm.key(), "base_url": OPENROUTER}, model_id=model,
                           params={"temperature": 0.2, "max_tokens": 1000})

    notes = TestMemoryStore(name="notes", writable=True, persist=False)
    memory = MemoryManager(stores=[notes], search_tool_config=False,
                           injection=MemoryInjectionConfig(query=lambda context, **kw: "Lisbon"))
    session = {"session.id": run_id, "user.id": "user"}
    hooks = [Native(), RetryTimeouts()]

    def agent(name, prompt, tools=(), extra=(), **kw):
        return Agent(name=name, model=openrouter(), system_prompt=prompt, tools=list(tools), hooks=hooks + list(extra),
                     trace_attributes=session, callback_handler=None, **kw)

    coordinator = agent("coordinator", f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short "
                                       "brief on a city's weather and facts? Answer with a JSON list of agent names.")
    weather = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
                                     "earlier attempt failed, then answer in one sentence.", [get_weather])
    facts = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
                                 "answer in one sentence.", [search_facts], [SaveNote()])
    translator = agent("translator_agent", "Translate the text you are given into Portuguese.")
    relay = agent("coordinator", "You pass the gathered weather and facts to the writer. Call handoff_to_agent "
                                 "with agent_name writer and the gathered text as the message.")
    writer = agent("writer", "You write two-sentence briefs about a city from the weather, the facts, and the "
                             "notes you are given. Reply with the brief only.", [], [Approval()],
                   memory_manager=memory)
    evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.")
    # the sampling settings live in each agent's model config; the chat spans do not carry them
    events("models", config={a.name: {k: a.model.get_config().get(k) for k in ("model_id", "params")}
                             for a in (coordinator, weather, facts, translator, relay, writer, evaluator)})

    def selected(name):
        def condition(state):
            text = str(state.results["coordinator"].result)
            chosen = [a for a in CANDIDATES if a in text] or ["weather", "facts"]
            return name in chosen
        return condition

    def both_done(state):
        return all(n in state.results for n in ("weather", "facts"))

    handoff = Swarm([relay, writer], entry_point=relay, max_handoffs=2, hooks=[Native()], id="handoff")
    g = GraphBuilder()
    g.add_node(coordinator, "coordinator")
    g.add_node(weather, "weather")
    g.add_node(facts, "facts")
    g.add_node(translator, "translator")
    g.add_node(handoff, "handoff")
    for name in CANDIDATES:
        g.add_edge("coordinator", name, condition=selected(name))
    g.add_edge("weather", "handoff", condition=both_done)
    g.add_edge("facts", "handoff", condition=both_done)
    g.set_entry_point("coordinator")
    g.set_hook_providers([Native()])
    g.set_graph_id("city-brief")
    graph = g.build()
    return graph, writer, evaluator, send_brief


async def run(model, out_dir, run_id, events):
    from strands.types.exceptions import MaxTokensReachedException

    graph, writer, evaluator, send_brief = build(model, out_dir, events, run_id)
    # node ids and the agents behind them, read from the graph object; the spans name agents only
    def executor(node):
        members = getattr(node.executor, "nodes", None)
        return {n: m.executor.name for n, m in members.items()} if members else node.executor.name
    defs = {"nodes": {n: executor(node) for n, node in graph.nodes.items()},
            "edges": [[e.from_node.node_id, e.to_node.node_id] for e in graph.edges]}
    events("graph", definitions=defs)
    with trace.get_tracer("city-brief").start_as_current_span("city brief", attributes={"session.id": run_id}):
        result = await graph.invoke_async("Write a short brief on Lisbon's weather and facts.")
        draft = str(result.results["handoff"].result.results["writer"].result).strip()
        for iteration in range(1, 4):
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                try:
                    verdict = str(await evaluator.invoke_async(draft))
                except MaxTokensReachedException:
                    # other frameworks hand back the cut answer; Strands raises and keeps it in the history
                    verdict = " ".join(c.get("text", "") for c in evaluator.messages[-1]["content"])
                passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
            declare("check", by="evaluator", checked="draft", iteration=iteration,
                    outcome="pass" if passed else "fail", reason=reason)
            if passed:
                break
            draft = str(await writer.invoke_async(f"The draft failed review: {reason}. Revise it:\n{draft}")).strip()
        # the writer gets the send tool only once the draft has passed
        writer.tool_registry.register_tool(send_brief)
        answer = await writer.invoke_async("Send the brief to client: call send_brief with the brief and to=client.")
        if answer.stop_reason == "interrupt":
            # the person approves on the reference app's behalf
            await writer.invoke_async([{"interruptResponse": {"interruptId": i.id, "response": "approve"}}
                                       for i in answer.interrupts])


def main():
    model = llm.pick()
    run_id = "st-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("strands", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "strands-reference")
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="strands", versions={
        p: version(p) for p in ("strands-agents", "openai", "opentelemetry-sdk")})
    asyncio.run(run(model, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
