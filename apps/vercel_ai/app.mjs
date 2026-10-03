import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { context } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { BasicTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OpenTelemetry } from "@ai-sdk/otel";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { ToolLoopAgent, registerTelemetry, stepCountIs, tool } from "ai";
import { z } from "zod";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const ENV = path.resolve(ROOT, "../.env");
const PYTHON = process.env.PROVAI_PYTHON || "python3";
const OPENROUTER = "https://openrouter.ai/api/v1";

const CANDIDATES = ["weather_agent", "facts_agent", "translator_agent"];
const DOCS = { Lisbon: "Lisbon is the capital of Portugal and lies on the Tagus estuary." };
const NOTES = {};
let weatherCalls = 0;

function key() {
  if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;
  for (const env of [path.resolve(ROOT, ".env"), ENV].filter((f) => fs.existsSync(f)))
    for (const line of fs.readFileSync(env, "utf8").split("\n"))
      if (line.startsWith("OPENROUTER_API_KEY=")) return line.slice("OPENROUTER_API_KEY=".length).trim().replace(/^['"]|['"]$/g, "");
  throw new Error("set OPENROUTER_API_KEY or put it in a .env file");
}

function pick() {
  // the model order, the low-cost fallbacks, and PROVAI_SKIP live in provai.llm
  const [model, paid] = JSON.parse(execFileSync(PYTHON, ["-c",
    "import json; from provai import llm; from provai.config import PAID; print(json.dumps([llm.pick(), list(PAID)]))"],
    { cwd: ROOT, encoding: "utf8" }).trim().split("\n").pop());
  if (!model.endsWith(":free") && !paid.includes(model)) throw new Error(`${model} is neither free nor a listed low-cost model`);
  return model;
}

function version(pkg) {
  return JSON.parse(fs.readFileSync(path.join(HERE, "node_modules", pkg, "package.json"), "utf8")).version;
}

function plain(value) {
  const seen = new WeakSet();
  return JSON.parse(JSON.stringify(value ?? null, (k, v) => {
    if (typeof v === "bigint") return Number(v);
    if (v instanceof Date) return v.toISOString();
    if (v instanceof Error) return { name: v.name, message: v.message };
    if (v && typeof v === "object") {
      if (seen.has(v)) return undefined;
      seen.add(v);
    }
    return v;
  }));
}

function events(file) {
  return (kind, fields = {}) =>
    fs.appendFileSync(file, JSON.stringify({ time: new Date().toISOString(), kind, ...plain(fields) }) + "\n");
}

function iso([sec, nano]) {
  return new Date(sec * 1000).toISOString().replace(/\.\d+Z$/, "") + "." + String(nano).padStart(9, "0").slice(0, 6) + "Z";
}

// writes OpenTelemetry spans as JSON lines, in the shape the Python SDK's to_json gives
function spanFile(file) {
  const code = { 0: "UNSET", 1: "OK", 2: "ERROR" };
  return {
    export(spans, done) {
      for (const s of spans) {
        const ctx = s.spanContext();
        const parent = s.parentSpanContext?.spanId ?? s.parentSpanId;
        fs.appendFileSync(file, JSON.stringify({
          name: s.name,
          context: { trace_id: "0x" + ctx.traceId, span_id: "0x" + ctx.spanId },
          kind: s.kind,
          parent_id: parent ? "0x" + parent : null,
          start_time: iso(s.startTime),
          end_time: iso(s.endTime),
          status: { status_code: code[s.status.code], description: s.status.message },
          attributes: s.attributes,
          events: s.events.map(e => ({ name: e.name, timestamp: iso(e.time), attributes: e.attributes })),
          links: [],
          resource: { attributes: s.resource?.attributes ?? {} },
        }) + "\n");
      }
      done({ code: 0 });
    },
    shutdown: async () => {},
    forceFlush: async () => {},
  };
}

// the SDK's own callbacks, logged field by field; start events also hold headers and provider options
function callbacks(name, log) {
  const output = o => o?.type === "tool-error"
    ? { type: o.type, error: plain(o.error) }
    : { type: o?.type, output: o?.output };
  return {
    onStart: e => log("agent_start", { agent: name, call_id: e.callId, model_id: e.modelId,
      tools: Object.keys(e.tools ?? {}), tool_choice: e.toolChoice }),
    onStepStart: e => log("step_start", { agent: name, call_id: e.callId, step: e.stepNumber }),
    onLanguageModelCallEnd: e => log("model_end", { agent: name, call_id: e.callId, response_id: e.responseId,
      finish_reason: e.finishReason, usage: e.usage, openrouter: e.providerMetadata?.openrouter,
      content: e.content }),
    onToolExecutionStart: e => log("tool_start", { agent: name, call_id: e.callId, tool_call_id: e.toolCall?.toolCallId,
      tool: e.toolCall?.toolName, input: e.toolCall?.input }),
    onToolExecutionEnd: e => log("tool_end", { agent: name, call_id: e.callId, tool_call_id: e.toolCall?.toolCallId,
      tool: e.toolCall?.toolName, ms: e.toolExecutionMs, ...output(e.toolOutput) }),
    onStepEnd: e => log("step_end", { agent: name, call_id: e.callId, step: e.stepNumber,
      finish_reason: e.finishReason, tool_calls: (e.toolCalls ?? []).map(c => ({ id: c.toolCallId, tool: c.toolName })),
      response_id: e.response?.id }),
    onEnd: e => log("agent_end", { agent: name, call_id: e.callId, text: e.text, finish_reason: e.finishReason }),
  };
}

function build(model, outDir, log) {
  const openrouter = createOpenRouter({ apiKey: key(), baseURL: OPENROUTER });
  const llm = openrouter(model, { usage: { include: true } });
  const settings = { temperature: 0.2, maxOutputTokens: 1000 };
  // the first step must call a tool; later steps may answer
  const firstCall = ({ stepNumber }) => (stepNumber === 0 ? { toolChoice: "required" } : {});
  const agent = (name, instructions, tools = {}, extra = {}) => new ToolLoopAgent({
    id: name, model: llm, instructions, tools, ...settings, stopWhen: stepCountIs(6),
    telemetry: { functionId: name }, ...callbacks(name, log), ...extra,
  });

  const getWeather = tool({
    description: "Current weather for a city.",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => {
      weatherCalls += 1;
      if (weatherCalls === 1) {
        const e = new Error("weather service timed out");
        e.name = "TimeoutError";
        throw e;
      }
      return `${city}: 21 C, clear sky`;
    },
  });
  const searchFacts = tool({
    description: "Facts about a city from the document store.",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => DOCS[city] ?? "no document found",
  });
  const readNote = tool({
    description: "Read the shared note about a city.",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => {
      log("memory_read", { by: "writer", memory: "notes", record: city });
      return NOTES[city] ?? "no note";
    },
  });
  const sendBrief = tool({
    description: "Send the brief to a recipient.",
    inputSchema: z.object({ text: z.string(), to: z.string() }),
    execute: async ({ text, to }) => {
      const target = path.join(outDir, `brief-to-${to}.txt`);
      fs.writeFileSync(target, text);
      log("effect", { by: "writer", target });
      return target;
    },
  });

  const weather = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an "
    + "earlier attempt failed, then answer in one sentence.", { get_weather: getWeather }, { prepareStep: firstCall });
  const facts = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then "
    + "answer in one sentence.", { search_facts: searchFacts }, { prepareStep: firstCall });
  const translator = agent("translator_agent", "Translate the text you are given into Portuguese.");

  // a subagent runs inside the tool that calls it; the facts agent's answer goes to the shared notes
  const asTool = (sub, description, after) => tool({
    description,
    inputSchema: z.object({ task: z.string() }),
    execute: async ({ task }, { abortSignal }) => {
      const result = await sub.generate({ prompt: task, abortSignal });
      if (after) after(result.text);
      return result.text;
    },
  });
  const coordinator = agent("coordinator", "You plan a short brief on a city's weather and facts. In your first "
    + "reply, call every agent you need at once, in parallel. Then answer with what they found.", {
    weather_agent: asTool(weather, "Finds the current weather of a city."),
    facts_agent: asTool(facts, "Finds facts about a city.", text => {
      NOTES.Lisbon = text;
      log("memory_write", { by: "facts_agent", memory: "notes", record: "Lisbon", value: text });
    }),
    translator_agent: asTool(translator, "Translates a text into Portuguese."),
  }, { prepareStep: firstCall });

  const writer = agent("writer", "You write two-sentence briefs about a city. Always call read_note for the city "
    + "first. Reply with the brief only.", { read_note: readNote }, { prepareStep: firstCall });
  // the same writer, with the tool that acts outside the system, for the release step only
  const sender = agent("writer", "Send the brief you are given to the recipient with send_brief.",
    { send_brief: sendBrief }, { stopWhen: stepCountIs(2), toolApproval: { send_brief: "user-approval" },
      // the send must be a tool call; once the person's answer is in the messages, the model may just reply
      prepareStep: ({ messages }) => (messages.some(m => m.role === "tool") ? {} : { toolChoice: "required" }) });
  const evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.");

  const agents = { coordinator, weather_agent: weather, facts_agent: facts, translator_agent: translator, writer, evaluator };
  return { agents, sender };
}

async function main() {
  const model = pick();
  const runId = "va-" + new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
  const outDir = path.join(process.env.PROVAI_RUNS || path.join(ROOT, "data", "runs"), "vercel_ai", runId);
  fs.mkdirSync(outDir, { recursive: true });
  const log = events(path.join(outDir, "events.jsonl"));

  context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
  const provider = new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(spanFile(path.join(outDir, "spans.jsonl")))] });
  registerTelemetry(new OpenTelemetry({ tracer: provider.getTracer("gen_ai") }));

  log("run", { run_id: runId, model, framework: "vercel_ai", versions: Object.fromEntries(
    ["ai", "@ai-sdk/otel", "@openrouter/ai-sdk-provider", "@opentelemetry/sdk-trace-base", "zod"].map(p => [p, version(p)])) });
  const { agents, sender } = build(model, outDir, log);
  // what each agent can call, read from the agent objects; the spans name tools but not which ones run agents
  log("agents", { definitions: Object.fromEntries(Object.entries(agents).map(([n, a]) => [n, {
    tools: Object.keys(a.tools ?? {}), agent_tools: Object.keys(a.tools ?? {}).filter(t => CANDIDATES.includes(t)) }])),
    approvals: { send_brief: "user-approval" }, settings: { temperature: 0.2, max_output_tokens: 1000 } });

  const gathered = await agents.coordinator.generate({ prompt: "Write a short brief on Lisbon's weather and facts." });
  // the SDK has no handoff; the coordinator's answer becomes the writer's task
  log("handoff", { from: "coordinator", to: "writer" });
  let draft = (await agents.writer.generate({ prompt: `Write a two-sentence brief on Lisbon from this: ${gathered.text}` })).text;
  for (let iteration = 1; iteration <= 3; iteration++) {
    let passed, reason;
    // the reference task fails the first draft on purpose, so every framework runs the loop twice
    if (iteration === 1) {
      [passed, reason] = [false, "the first draft is always sent back for revision"];
    } else {
      const verdict = (await agents.evaluator.generate({ prompt: draft })).text;
      [passed, reason] = [!verdict.toUpperCase().includes("FAIL") || iteration === 3, verdict.slice(0, 200)];
    }
    log("check", { by: "evaluator", checked: "draft", iteration, outcome: passed ? "pass" : "fail", reason });
    if (passed) break;
    draft = (await agents.writer.generate({ prompt: `The draft failed review: ${reason}. Revise it:\n${draft}` })).text;
  }

  const messages = [{ role: "user", content: `Send this brief to client:\n${draft}` }];
  const held = await sender.generate({ messages });
  messages.push(...held.responseMessages);
  const responses = [];
  for (const part of held.content) {
    if (part.type === "tool-approval-request" && !part.isAutomatic) {
      // a person reviews the held send_brief call; the reference app approves on their behalf
      log("approval", { by: "person", checked: "draft", outcome: "approve", approval_id: part.approvalId,
        tool_call_id: part.toolCall?.toolCallId });
      responses.push({ type: "tool-approval-response", approvalId: part.approvalId, approved: true });
    }
  }
  messages.push({ role: "tool", content: responses });
  await sender.generate({ messages });
  await provider.shutdown();
  console.log(outDir);
}

await main();
