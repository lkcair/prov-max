import asyncio
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opentelemetry import trace

from provai import capture, llm
from provai.config import OPENROUTER

DOCS = {"Lisbon": "Lisbon is the capital of Portugal and lies on the Tagus estuary."}
NOTES = {}
SKIP = {"TEXT_BLOCK_DELTA", "THINKING_BLOCK_DELTA", "TOOL_CALL_DELTA", "TOOL_RESULT_TEXT_DELTA",
        "DATA_BLOCK_DELTA", "TOOL_RESULT_DATA_DELTA"}
weather_calls = 0


def span_id():
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx.is_valid else None


def text_of(msg):
    return " ".join(b.text for b in msg.content if getattr(b, "type", None) == "text" and b.text).strip() \
        if not isinstance(msg.content, str) else msg.content.strip()


def build(model_id, out_dir, events):
    from agentscope.agent import Agent, ReActConfig
    from agentscope.credential import OpenAICredential
    from agentscope.message import TextBlock
    from agentscope.middleware import MiddlewareBase, TracingMiddleware
    from agentscope.model import OpenAIChatModel
    from agentscope.permission import PermissionBehavior, PermissionDecision
    from agentscope.pipeline import TeamMember, TeamPipeline
    from agentscope.tool import FunctionTool, ToolChoice, ToolChunk, Toolkit

    def chunk(text):
        return ToolChunk(content=[TextBlock(type="text", text=text)])

    class Native(MiddlewareBase):
        # the response id and usage of each model call, which the stream events leave out
        async def on_model_call(self, agent, input_kwargs, next_handler):
            res = await next_handler(**input_kwargs)
            events("model_call", agent=agent.name, reply_id=agent.state.reply_id, span_id=span_id(),
                   response_id=getattr(res, "id", None),
                   usage={k: getattr(res.usage, k, None) for k in ("input_tokens", "output_tokens", "time")}
                   if getattr(res, "usage", None) else None)
            return res

    class FirstTurnTool(MiddlewareBase):
        # the first reasoning step of a reply must call a tool; later steps may answer
        async def on_reasoning(self, agent, input_kwargs, next_handler):
            if agent.state.reply_context.cur_iter == 0:
                input_kwargs = {**input_kwargs, "tool_choice": ToolChoice(mode="required")}
            async for item in next_handler(**input_kwargs):
                yield item

    class SaveNote(MiddlewareBase):
        # the facts agent's answer goes to the shared notes, as in the reference task
        async def on_reply(self, agent, input_kwargs, next_handler):
            last = None
            async for item in next_handler(**input_kwargs):
                if type(item).__name__ == "Msg":
                    last = item
                yield item
            if last is not None:
                NOTES["Lisbon"] = text_of(last)
                events("memory_write", by="facts_agent", memory="notes", key="Lisbon", value=NOTES["Lisbon"],
                       reply_id=agent.state.reply_id, span_id=span_id())

    async def get_weather(city: str) -> ToolChunk:
        """Current weather for a city.

        Args:
            city (str): The city.
        """
        global weather_calls
        weather_calls += 1
        if weather_calls == 1:
            raise TimeoutError("weather service timed out")
        return chunk(f"{city}: 21 C, clear sky")

    async def search_facts(city: str) -> ToolChunk:
        """Facts about a city from the document store.

        Args:
            city (str): The city.
        """
        return chunk(DOCS.get(city, "no document found"))

    async def read_note(city: str) -> ToolChunk:
        """Read the shared note about a city.

        Args:
            city (str): The city.
        """
        events("memory_read", by="writer", memory="notes", key=city, span_id=span_id())
        return chunk(NOTES.get(city, "no note"))

    async def send_brief(text: str, to: str) -> ToolChunk:
        """Send the brief to a recipient.

        Args:
            text (str): The brief.
            to (str): The recipient.
        """
        path = out_dir / f"brief-to-{to}.txt"
        path.write_text(text)
        events("effect", by="writer", target=str(path), span_id=span_id())
        return chunk(str(path))

    def model():
        return OpenAIChatModel(
            credential=OpenAICredential(api_key=llm.key(), base_url=OPENROUTER), model=model_id, stream=False,
            parameters=OpenAIChatModel.Parameters(temperature=0.2, max_tokens=1000))

    async def agent(name, prompt, tools=(), extra=()):
        kit = Toolkit()
        for t in tools:
            await kit.add_tool(t)
        return Agent(name=name, system_prompt=prompt, model=model(), toolkit=kit,
                     middlewares=[TracingMiddleware(), Native(), *extra],
                     react_config=ReActConfig(max_iters=6))

    allow = PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Reads change nothing outside the run.")

    def reader(func):
        return FunctionTool(func, is_read_only=True, permission=allow)

    async def make():
        weather = await agent("weather_agent", "You do not know the weather. Always call get_weather first, even if "
                                               "an earlier attempt failed, then answer in one sentence.",
                              [reader(get_weather)], [FirstTurnTool()])
        facts = await agent("facts_agent", "You do not know facts about cities. Always call search_facts first, "
                                           "then answer in one sentence.", [reader(search_facts)],
                            [FirstTurnTool(), SaveNote()])
        translator = await agent("translator_agent", "Translate the text you are given into Portuguese.")
        writer = await agent("writer", "You write two-sentence briefs about a city from the weather and facts you "
                                       "are given and the note you read. Always call read_note for the city first. "
                                       "Reply with the brief only.", [reader(read_note)], [FirstTurnTool()])
        coordinator = await agent(
            "coordinator", "You plan a short brief on a city's weather and facts. In your first reply, assign the "
                           "weather_agent and the facts_agent at once with TeamAssign, both in the same reply. When "
                           "both have answered, assign the writer with everything they found. Then reply with the "
                           "writer's brief only.", [], [FirstTurnTool()])
        evaluator = await agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.")
        team = TeamPipeline(coordinator, [
            TeamMember(agent=weather, description="Finds the current weather of a city."),
            TeamMember(agent=facts, description="Finds facts about a city."),
            TeamMember(agent=translator, description="Translates a text into Portuguese."),
            TeamMember(agent=writer, description="Writes the brief from the weather and the facts.")])
        sender = FunctionTool(send_brief, permission=PermissionDecision(
            behavior=PermissionBehavior.ASK, message="A person approves every brief before it is sent."))
        # what the coordinator can assign and what each agent can call, read from the objects
        tools = {}
        for a in (weather, facts, translator, writer, coordinator):
            tools[a.name] = [s["function"]["name"] for s in await a.toolkit.get_tool_schemas()]
        events("agents", definitions={"leader": coordinator.name, "members": list(team.members), "tools": tools,
                                      "settings": {"temperature": 0.2, "max_tokens": 1000}})
        return team, writer, evaluator, sender

    return make


async def drain(stream, events, who):
    from agentscope.event import RequireUserConfirmEvent
    last, pending = None, []
    async for item in stream:
        kind = getattr(item, "type", None)
        if kind is None:
            last = item
            calls = [{"id": b.id, "name": b.name, "input": b.input} for b in item.content
                     if not isinstance(item.content, str) and getattr(b, "type", None) == "tool_call"]
            events("message", stream=who, id=item.id, name=item.name, text=text_of(item), tool_calls=calls)
            continue
        if str(kind) in SKIP:
            continue
        row = item.model_dump(mode="json")
        if isinstance(item, RequireUserConfirmEvent):
            pending.append(item)
        events("event", stream=who, **{k: v for k, v in row.items() if k != "type"}, event_type=str(kind))
    return last, pending


async def run(model_id, out_dir, run_id, events):
    from agentscope.event import ConfirmResult, UserConfirmResultEvent
    from agentscope.message import UserMsg
    team, writer, evaluator, sender = await build(model_id, out_dir, events)()
    with trace.get_tracer("city-brief").start_as_current_span("city brief", attributes={"session.id": run_id}):
        await drain(team.reply_stream(UserMsg("user", "Write a short brief on Lisbon's weather and facts.")),
                    events, "team")
        # the team stream ends with events; the leader's answer is the last message of its context
        last = team.leader.state.context[-1]
        events("message", stream="team", id=last.id, name=last.name, text=text_of(last), tool_calls=[])
        draft = text_of(last)
        for iteration in range(1, 4):
            # the reference task fails the first draft on purpose, so every framework runs the loop twice
            if iteration == 1:
                passed, reason = False, "the first draft is always sent back for revision"
            else:
                verdict, _ = await drain(evaluator.reply_stream(UserMsg("user", draft), yield_final_msg=True), events, "evaluator")
                verdict = text_of(verdict)
                passed, reason = "FAIL" not in verdict.upper() or iteration == 3, verdict[:200]
            events("check", by="evaluator", checked="draft", iteration=iteration, outcome="pass" if passed else "fail",
                   reason=reason, draft=draft)
            if passed:
                break
            revised, _ = await drain(writer.reply_stream(UserMsg("user", f"The draft failed review: {reason}. Revise "
                                                                         f"it:\n{draft}"), yield_final_msg=True), events, "writer")
            draft = text_of(revised)
        # the writer gets the send tool only once the draft has passed
        await writer.toolkit.add_tool(sender)
        _, pending = await drain(writer.reply_stream(UserMsg("user", "Send this brief to client: call send_brief "
                                                                     f"with text and to=client.\n{draft}"), yield_final_msg=True),
                                 events, "writer")
        for ask in pending:
            # a person reviews the held send_brief call; the reference app approves on their behalf
            events("approval", by="person", outcome="approve", reply_id=ask.reply_id,
                   tool_call_ids=[t.id for t in ask.tool_calls])
            await drain(writer.reply_stream(UserConfirmResultEvent(
                reply_id=ask.reply_id, confirm_results=[ConfirmResult(confirmed=True, tool_call=t)
                                                        for t in ask.tool_calls]), yield_final_msg=True), events, "writer")


def main():
    model_id = llm.pick()
    run_id = "as-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = capture.run_dir("agentscope", run_id)
    provider = capture.tracer_provider(out / "spans.jsonl", "agentscope-reference")
    events = capture.Events(out / "events.jsonl")
    events("run", run_id=run_id, model=model_id, framework="agentscope",
           versions={p: version(p) for p in ("agentscope", "openai", "opentelemetry-sdk")})
    asyncio.run(run(model_id, out, run_id, events))
    provider.shutdown()
    events.close()
    print(out)


if __name__ == "__main__":
    main()
