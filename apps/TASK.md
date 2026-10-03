# Reference task

Every framework runs the same small task, written the way that framework is meant to be used, so
that each record exercises the same behaviors.

A user asks for a short brief about a city.

1. A coordinator agent routes the request. Its candidates are a weather agent, a facts agent, and a
   translator agent. It selects the weather and facts agents.
2. The weather agent and the facts agent run in parallel.
   - The weather agent calls the tool `get_weather(city)`. The first call fails with a timeout error;
     the agent retries and the second call succeeds.
   - The facts agent calls `search_facts(city)`, which looks the city up in a small document store,
     and writes one note to the shared memory `notes`.
3. A writer agent receives the work by handoff. It reads the note from `notes`, merges the two
   worker outputs, and drafts the brief.
4. The application sends the first draft back, and the writer revises it; the application records this
   first rejection as a check of the evaluator. An evaluator agent checks each revision, and the loop
   stops when it passes a draft. At the third iteration, the application stops the loop and records a pass.
5. A person approves the release. In the reference app the approval is a function that answers
   "approve" on behalf of a person, and it is recorded as a human decision.
6. The writer calls `send_brief(text, to)`, which writes the brief to a file. This is the effect
   outside the system.

Model answers come from OpenRouter, picked by `provai.llm.pick()` from a fixed list of models.
Answers only need to be short and plausible, since the task tests the record.

Each run writes `data/runs/<framework>/<run id>/spans.jsonl` (OpenTelemetry spans as the framework
or its instrumentor emits them), `events.jsonl` (what the framework's own callbacks, hooks, or event
stream report), and, after conversion, `record.ttl` (PROV-AI).
