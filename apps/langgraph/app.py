import json
import os
from importlib.metadata import version
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "SPAN_ONLY"

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, RetryPolicy, Send, interrupt
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor

from provai import capture, llm
from provai.config import OPENROUTER

CANDIDATES = ["weather", "facts", "translator"]
AGENTS = {"weather": "weather_agent", "facts": "facts_agent", "translator": "translator_agent"}
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
weather_calls = 0


class Native(BaseCallbackHandler):
    def __init__(self, events):
        self.events = events

    def meta(self, metadata):
        keep = ("langgraph_node", "langgraph_step", "langgraph_triggers", "langgraph_checkpoint_ns", "thread_id")
        return {k: v for k, v in (metadata or {}).items() if k in keep}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, name=None, **kw):
        self.events("chain_start", run_id=run_id, parent_run_id=parent_run_id, name=name or kw.get("run_name"),
                    reads=sorted(inputs) if isinstance(inputs, dict) else None, **self.meta(metadata))

    def on_chain_end(self, outputs, *, run_id, **kw):
        self.events("chain_end", run_id=run_id, writes=sorted(outputs) if isinstance(outputs, dict) else None)

    def on_chain_error(self, error, *, run_id, **kw):
        self.events("chain_error", run_id=run_id, error=type(error).__name__)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kw):
        self.events("model_start", run_id=run_id, parent_run_id=parent_run_id, **self.meta(metadata))

    def on_llm_end(self, response, *, run_id, **kw):
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.events("model_end", run_id=run_id, usage=usage)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, **kw):
        self.events("tool_start", run_id=run_id, parent_run_id=parent_run_id, name=(serialized or {}).get("name"),
                    input=input_str, **self.meta(metadata))

    def on_tool_end(self, output, *, run_id, **kw):
        self.events("tool_end", run_id=run_id, output=str(getattr(output, "content", output)))

    def on_tool_error(self, error, *, run_id, **kw):
        self.events("tool_error", run_id=run_id, error=type(error).__name__)

    def on_custom_event(self, name, data, *, run_id, metadata=None, **kw):
        self.events(name, run_id=run_id, **data, **self.meta(metadata))


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


class State(TypedDict, total=False):
    city: str
    selected: list
    weather: str
    facts: str
    draft: str
    iteration: int
    verdict: str
    approved: str
    sent: str


def build(model, out_dir):
    chat = ChatOpenAI(model=model, base_url=OPENROUTER, api_key=llm.key(), temperature=0.2, max_tokens=1000)
    weather_agent = create_agent(chat, [get_weather], name="weather_agent",
                                 system_prompt="You do not know the weather. Always call get_weather first, "
                                               "even if an earlier attempt failed, then answer in one sentence.")
    facts_agent = create_agent(chat, [search_facts], name="facts_agent",
                               system_prompt="You do not know facts about cities. Always call search_facts "
                                             "first, then answer in one sentence.")

    @tool
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        return str(path)

    def coordinator(state: State):
        prompt = (f"Agents: {', '.join(CANDIDATES)}. Which agents are needed for a short brief on "
                  f"{state['city']}'s weather and facts? Answer with a JSON list of agent names.")
        answer = chat.invoke(prompt).content
        try:
            selected = [a for a in json.loads(answer[answer.index("["):answer.rindex("]") + 1]) if a in CANDIDATES]
        except ValueError:
            selected = []
        selected = selected or ["weather", "facts"]
        dispatch_custom_event("routing", {"by": "coordinator", "candidates": [AGENTS[a] for a in CANDIDATES],
                                          "selected": [AGENTS[a] for a in selected], "reason": answer[:500]})
        return {"selected": selected}

    def weather(state: State):
        out = weather_agent.invoke({"messages": [("user", f"Weather in {state['city']}?")]})
        return {"weather": out["messages"][-1].content}

    def facts(state: State, *, store: BaseStore):
        out = facts_agent.invoke({"messages": [("user", f"Facts about {state['city']}?")]})
        text = out["messages"][-1].content
        store.put(("notes",), state["city"], {"text": text})
        dispatch_custom_event("memory_write", {"by": "facts_agent", "memory": "notes", "key": state["city"],
                                               "value": text})
        return {"facts": text}

    def handoff(state: State):
        dispatch_custom_event("handoff", {"from": "coordinator", "to": "writer"})
        return Command(goto="writer")

    def writer(state: State, *, store: BaseStore):
        note = store.get(("notes",), state["city"])
        dispatch_custom_event("memory_read", {"by": "writer", "memory": "notes", "key": state["city"]})
        feedback = f" The last draft failed review: {state['verdict']}. Fix it." if state.get("verdict") else ""
        draft = chat.invoke(f"Write a two-sentence brief on {state['city']}. Weather: {state.get('weather')}. "
                            f"Facts: {state.get('facts')}. Note: {note.value['text'] if note else ''}.{feedback} "
                            "Reply with the brief only.").content
        return {"draft": draft, "iteration": state.get("iteration", 0) + 1}

    def evaluator(state: State):
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if state["iteration"] == 1:
            outcome, reason = "fail", "the first draft is always sent back for revision"
        else:
            verdict = chat.invoke(f"Does this brief mention the weather? Answer PASS or FAIL.\n{state['draft']}").content
            outcome = "fail" if "FAIL" in verdict.upper() and state["iteration"] < 3 else "pass"
            reason = verdict[:200]
        dispatch_custom_event("check", {"by": "evaluator", "checked": "draft", "iteration": state["iteration"],
                                        "outcome": outcome, "reason": reason})
        return {"verdict": "" if outcome == "pass" else reason}

    def approval(state: State):
        decision = interrupt({"draft": state["draft"]})
        dispatch_custom_event("approval", {"by": "person", "checked": "draft", "outcome": decision})
        return {"approved": decision}

    def send(state: State):
        target = send_brief.invoke({"text": state["draft"], "to": "client"})
        dispatch_custom_event("effect", {"by": "writer", "target": target})
        return {"sent": target}

    g = StateGraph(State)
    g.add_node("coordinator", coordinator)
    g.add_node("weather", weather, retry_policy=RetryPolicy(retry_on=TimeoutError, max_attempts=3))
    g.add_node("facts", facts)
    g.add_node("translator", lambda state: {})
    g.add_node("handoff", handoff)
    g.add_node("writer", writer)
    g.add_node("evaluator", evaluator)
    g.add_node("approval", approval)
    g.add_node("send", send)
    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", lambda s: [Send(a, s) for a in s["selected"]], CANDIDATES)
    g.add_edge(["weather", "facts"], "handoff")
    g.add_edge("writer", "evaluator")
    g.add_conditional_edges("evaluator", lambda s: "writer" if s["verdict"] else "approval", ["writer", "approval"])
    g.add_edge("approval", "send")
    g.add_edge("send", END)
    return g.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


def main():
    model = llm.pick()
    run_id = "lg-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("langgraph", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "langgraph-reference")
    LangChainInstrumentor().instrument(tracer_provider=provider)
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model, framework="langgraph", versions={
        p: version(p) for p in ("langgraph", "langchain", "langchain-openai", "opentelemetry-sdk",
                                "opentelemetry-instrumentation-genai-langchain")})
    app = build(model, out)
    config = {"configurable": {"thread_id": run_id}, "callbacks": [Native(events)]}
    app.invoke({"city": "Lisbon"}, config)
    app.invoke(Command(resume="approve"), config)
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
