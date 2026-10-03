import asyncio
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import dspy
from dspy.utils.exceptions import AdapterParseError
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from openinference.instrumentation import using_session
from openinference.instrumentation.dspy import DSPyInstrumentor

from provai import capture, llm

CANDIDATES = {"weather": "weather_agent", "facts": "facts_agent", "translator": "translator_agent"}
DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
EVENTS = None
weather_calls = 0


def settings(lm):
    # only the sampling settings; the LM's kwargs also hold the API key
    return {k: lm.kwargs.get(k) for k in ("temperature", "max_tokens")}


def agent_of(instance):
    return getattr(instance, "agent_name", None)


class Native(BaseCallback):
    # start handlers run before DSPy marks the new call as active, so the active call is the parent
    def __init__(self, events):
        self.events = events
        self.lms = {}

    def start(self, kind, call_id, instance, **fields):
        self.events(f"{kind}_start", call_id=call_id, parent=ACTIVE_CALL_ID.get(), cls=type(instance).__name__,
                    agent=agent_of(instance), **fields)

    def end(self, kind, call_id, outputs, exception, **fields):
        self.events(f"{kind}_end", call_id=call_id, outputs=str(outputs)[:4000] if outputs is not None else None,
                    error=type(exception).__name__ if exception else None,
                    message=str(exception)[:500] if exception else None, **fields)

    def on_module_start(self, call_id, instance, inputs):
        self.start("module", call_id, instance, inputs={k: str(v)[:2000] for k, v in inputs.items()})

    def on_module_end(self, call_id, outputs, exception=None):
        self.end("module", call_id, outputs, exception)

    def on_lm_start(self, call_id, instance, inputs):
        self.lms[call_id] = instance
        self.start("lm", call_id, instance, model=instance.model, settings=settings(instance))

    def on_lm_end(self, call_id, outputs, exception=None):
        entry = self.lms.pop(call_id).history[-1] if not exception else {}
        self.end("lm", call_id, outputs, exception, usage=entry.get("usage"), cost=entry.get("cost"),
                 response_model=entry.get("response_model"), uuid=entry.get("uuid"))

    def on_tool_start(self, call_id, instance, inputs):
        self.start("tool", call_id, instance, tool=instance.name, inputs=inputs.get("kwargs", inputs))

    def on_tool_end(self, call_id, outputs, exception=None):
        self.end("tool", call_id, outputs, exception)


def declare(kind, **fields):
    # facts DSPy has no construct for, reported by the application inside the call they belong to
    EVENTS(kind, parent=ACTIVE_CALL_ID.get(), **fields)


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


def save_note(city: str, text: str) -> str:
    """Save a note about a city to the shared notes."""
    NOTES[city] = text
    declare("memory_write", by="facts_agent", memory="notes", key=city, value=text)
    return "saved"


def read_note(city: str) -> str:
    """Read the shared note about a city."""
    declare("memory_read", by="writer", memory="notes", key=city)
    return NOTES.get(city, "no note")


class Agent(dspy.Module):
    def __init__(self, name):
        super().__init__()
        self.agent_name = name


class Coordinator(Agent):
    def __init__(self):
        super().__init__("coordinator")
        self.route = dspy.Predict("request, candidates -> agents_needed: str")

    async def aforward(self, request):
        answer = (await self.route.acall(request=request, candidates=", ".join(CANDIDATES))).agents_needed
        selected = [a for a in CANDIDATES if a in answer.lower()] or ["weather", "facts"]
        declare("routing", by="coordinator", candidates=list(CANDIDATES.values()),
                selected=[CANDIDATES[a] for a in selected], reason=answer[:500])
        return selected


class Worker(Agent):
    def __init__(self, name, signature, tools):
        super().__init__(name)
        self.react = dspy.ReAct(signature, tools=tools, max_iters=4)

    async def aforward(self, city):
        return (await self.react.acall(city=city)).answer


class Facts(Worker):
    async def aforward(self, city):
        answer = await super().aforward(city)
        # the facts agent's answer goes to the shared notes, as in the reference task
        await self.note.acall(city=city, text=answer)
        return answer


class Writer(Agent):
    def __init__(self):
        super().__init__("writer")
        self.read = dspy.Tool(read_note)
        self.draft = dspy.Predict("city, weather, facts, note, feedback -> brief")

    async def aforward(self, city, weather, facts, feedback=""):
        note = await self.read.acall(city=city)
        return (await self.draft.acall(city=city, weather=weather, facts=facts, note=note,
                                       feedback=feedback)).brief


class Evaluator(Agent):
    def __init__(self):
        super().__init__("evaluator")
        self.judge = dspy.Predict("brief -> verdict: str")

    async def aforward(self, brief):
        return (await self.judge.acall(brief=brief)).verdict


class Release(Agent):
    def __init__(self, send_brief):
        super().__init__("writer")
        self.send = dspy.Tool(send_brief)

    async def aforward(self, brief):
        return await self.send.acall(text=brief, to="client")


async def ask(module, **kwargs):
    # a free model sometimes answers in a shape DSPy cannot parse; ask again, at most three times
    for attempt in range(3):
        try:
            return await module.acall(**kwargs)
        except AdapterParseError:
            if attempt == 2:
                raise


def make_lm(model):
    return dspy.LM(f"openrouter/{model}", api_key=llm.key(), temperature=0.2, max_tokens=1000, cache=False)


async def run(model, out_dir):
    def send_brief(text: str, to: str) -> str:
        """Send the brief to a recipient."""
        # a person reviews the brief before it leaves; the reference app approves on their behalf
        declare("approval", by="person", checked="draft", outcome="approve")
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        declare("effect", by="writer", target=str(path))
        return str(path)

    coordinator = Coordinator()
    weather = Worker("weather_agent", "city -> answer",
                     [dspy.Tool(get_weather)])
    facts = Facts("facts_agent", "city -> answer", [dspy.Tool(search_facts)])
    facts.note = dspy.Tool(save_note)
    translator = Worker("translator_agent", "city -> answer", [])
    # the writer gets the tool that acts outside the system for the release step only
    writer, evaluator, release = Writer(), Evaluator(), Release(send_brief)
    agents = {"weather": weather, "facts": facts, "translator": translator}
    for m in (coordinator, weather, facts, translator, writer, evaluator, release):
        m.set_lm(make_lm(model))
    definitions = {}
    for m in (coordinator, weather, facts, translator, writer, evaluator, release):
        tools = [t.name for _, p in m.named_sub_modules() for t in getattr(p, "tools", {}).values()
                 if hasattr(t, "name")] + [t.name for t in vars(m).values() if isinstance(t, dspy.Tool)]
        entry = definitions.setdefault(m.agent_name, {"classes": [], "tools": []})
        entry["classes"].append(type(m).__name__)
        entry["tools"] += [t for t in tools if t not in entry["tools"]]
    EVENTS("agents", definitions=definitions,
           models={m.agent_name: {"model": m.get_lm().model, **settings(m.get_lm())} for m in (coordinator, writer)})

    city = "Lisbon"
    selected = await ask(coordinator, request=f"Write a short brief on {city}'s weather and facts.")
    results = await asyncio.gather(*(ask(agents[a], city=city) for a in selected))
    found = dict(zip(selected, results))
    declare("handoff", by="coordinator", to="writer")
    feedback, draft = "", ""
    for iteration in range(1, 4):
        draft = await ask(writer, city=city, weather=found.get("weather", ""), facts=found.get("facts", ""),
                                   feedback=feedback)
        # the reference task fails the first draft on purpose, so every framework runs the loop twice
        if iteration == 1:
            passed, reason = False, "the first draft is always sent back for revision"
        else:
            verdict = await ask(evaluator, brief=draft)
            passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
        declare("check", by="evaluator", checked="draft", iteration=iteration,
                outcome="pass" if passed else "fail", reason=reason)
        if passed:
            break
        feedback = f"The last draft failed review: {reason}. Fix it."
    await release.acall(brief=draft)


def main():
    global EVENTS
    model = llm.pick()
    run_id = "ds-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("dspy", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "dspy-reference")
    DSPyInstrumentor().instrument(tracer_provider=provider)
    EVENTS = capture.Events(out / "events.jsonl")
    EVENTS("run", run_id=run_id, model=model, framework="dspy", versions={
        p: version(p) for p in ("dspy", "litellm", "opentelemetry-sdk", "openinference-instrumentation-dspy")})
    dspy.configure(callbacks=[Native(EVENTS)], track_usage=True)
    with using_session(run_id):
        asyncio.run(run(model, out))
    provider.shutdown()
    EVENTS.close()
    print(out)


if __name__ == "__main__":
    main()
