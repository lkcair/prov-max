import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Mastra reports feature usage to its own servers unless this is set
process.env.MASTRA_TELEMETRY_DISABLED = "1";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../..");
const ENV = path.resolve(ROOT, "../.env");
const PYTHON = process.env.PROVAI_PYTHON || "python3";
const OPENROUTER = "https://openrouter.ai/api/v1";
const SCOPE = "resource";

const { Mastra } = await import("@mastra/core/mastra");
const { Agent } = await import("@mastra/core/agent");
const { createTool } = await import("@mastra/core/tools");
const { createWorkflow, createStep } = await import("@mastra/core/workflows");
const { InMemoryStore } = await import("@mastra/core/storage");
const { Memory } = await import("@mastra/memory");
const { Observability } = await import("@mastra/observability");
const { OtelExporter } = await import("@mastra/otel-exporter");
const { createOpenRouter } = await import("@openrouter/ai-sdk-provider");
const { z } = await import("zod");

const CANDIDATES = ["weather", "facts", "translator"];
const DOCS = { Lisbon: "Lisbon is the capital of Portugal and lies on the Tagus estuary." };
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

// Mastra's own spans, with their types, entities, attributes, and metadata
function nativeSpans(log) {
  return {
    name: "provai-native",
    init() {},
    async exportTracingEvent(event) {
      if (event.type !== "span_ended") return;
      const s = event.exportedSpan;
      log("span", {
        id: s.id, trace_id: s.traceId, parent_id: s.parentSpanId ?? null, type: s.type, name: s.name,
        entity_type: s.entityType, entity_id: s.entityId, entity_name: s.entityName, is_event: s.isEvent,
        start: s.startTime, end: s.endTime, attributes: s.attributes, metadata: s.metadata,
        input: s.input, output: s.output, error: s.errorInfo,
      });
    },
    async flush() {},
    async shutdown() {},
  };
}

function declare(tracingContext, name, metadata) {
  tracingContext.currentSpan.createEventSpan({ type: "generic", name, metadata, output: metadata });
}

function build(model, outDir, log, runId, storage) {
  const openrouter = createOpenRouter({ apiKey: key(), baseURL: OPENROUTER });
  const llm = openrouter(model, { usage: { include: true } });
  const settings = { temperature: 0.2, maxOutputTokens: 1000 };
  const usage = name => step => log("model_step", {
    agent: name, response_id: step.response?.id, finish_reason: step.finishReason, usage: step.usage,
    provider_metadata: step.providerMetadata,
  });
  // the first model step must call a tool; later steps may answer
  const firstCall = ({ stepNumber }) => (stepNumber === 0 ? { toolChoice: "required" } : {});
  const options = (name, extra = {}) => ({ modelSettings: settings, onStepFinish: usage(name), maxSteps: 6, ...extra });

  const memory = new Memory({ storage, options: { workingMemory: { enabled: true, scope: SCOPE }, lastMessages: false } });

  const getWeather = createTool({
    id: "get_weather", description: "Current weather for a city.",
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
  const searchFacts = createTool({
    id: "search_facts", description: "Facts about a city from the document store.",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }) => DOCS[city] ?? "no document found",
  });
  const readNote = createTool({
    id: "read_note", description: "Read the shared note about a city.",
    inputSchema: z.object({ city: z.string() }),
    execute: async ({ city }, context) => {
      const note = await memory.getWorkingMemory({ threadId: runId, resourceId: "notes" });
      if (context?.tracingContext?.currentSpan)
        declare(context.tracingContext, "memory_read", { by: "writer", memory: "notes", record: city });
      return note ?? "no note";
    },
  });
  const sendBrief = createTool({
    id: "send_brief", description: "Send the brief to a recipient.",
    inputSchema: z.object({ text: z.string(), to: z.string() }),
    execute: async ({ text, to }, context) => {
      const target = path.join(outDir, `brief-to-${to}.txt`);
      fs.writeFileSync(target, text);
      declare(context.tracingContext, "effect", { by: "writer", target });
      return target;
    },
  });

  const agent = (id, instructions, tools = {}, extra = {}) =>
    new Agent({ id, name: id, model: llm, instructions, tools, defaultOptions: options(id, extra.defaults), ...extra.config });
  const coordinator = agent("coordinator", `Agents: ${CANDIDATES.join(", ")}. Which agents are needed for a short brief on a city's weather and facts? Answer with a JSON list of agent names.`);
  const weatherAgent = agent("weather_agent", "You do not know the weather. Always call get_weather first, even if an earlier attempt failed, then answer in one sentence.",
    { get_weather: getWeather }, { defaults: { prepareStep: firstCall } });
  const factsAgent = agent("facts_agent", "You do not know facts about cities. Always call search_facts first, then answer in one sentence.",
    { search_facts: searchFacts }, { defaults: { prepareStep: firstCall } });
  const translatorAgent = agent("translator_agent", "Translate the text you are given into Portuguese.");
  const writer = agent("writer", "You write two-sentence briefs about a city from the weather and facts you are given. Always call read_note for the city first. Reply with the brief only.",
    { read_note: readNote }, { defaults: { prepareStep: firstCall } });
  const relay = agent("coordinator", "You pass the gathered weather and facts to the writer. Call the agent-writer tool with maxSteps 5 and the gathered text as the prompt, then reply with the writer's brief.",
    {}, { defaults: { prepareStep: firstCall }, config: { agents: { writer } } });
  const evaluator = agent("evaluator", "Does the brief you are given mention the weather? Answer PASS or FAIL.");
  const sender = agent("writer", "When asked to send a brief, call send_brief with the brief and the recipient.",
    { send_brief: sendBrief }, { defaults: { prepareStep: firstCall } });

  const City = z.object({ city: z.string(), selected: z.array(z.string()), reason: z.string() });
  const Answer = z.object({ text: z.string() });
  const route = createStep({
    id: "coordinator", inputSchema: z.object({ city: z.string() }), outputSchema: City,
    execute: async ({ inputData, tracingContext }) => {
      const out = await coordinator.generate(`Agents: ${CANDIDATES.join(", ")}. Which agents are needed for a short brief on ${inputData.city}'s weather and facts? Answer with a JSON list of agent names.`, { tracingContext });
      let selected = [];
      try {
        const t = out.text;
        selected = JSON.parse(t.slice(t.indexOf("["), t.lastIndexOf("]") + 1)).filter(a => CANDIDATES.includes(a));
      } catch {}
      return { city: inputData.city, selected: selected.length ? selected : ["weather", "facts"], reason: out.text.slice(0, 500) };
    },
  });
  const worker = (id, agentOf, prompt, after) => createStep({
    id, inputSchema: City, outputSchema: Answer, metadata: { agent: agentOf.name },
    execute: async ({ inputData, tracingContext }) => {
      const out = await agentOf.generate(prompt(inputData.city), { tracingContext });
      if (after) await after(out.text, tracingContext);
      return { text: out.text };
    },
  });
  const weather = worker("weather", weatherAgent, c => `Weather in ${c}?`);
  const facts = worker("facts", factsAgent, c => `Facts about ${c}?`, async (text, tracingContext) => {
    await memory.updateWorkingMemory({ threadId: runId, resourceId: "notes", workingMemory: text,
      observabilityContext: { tracingContext, tracing: tracingContext } });
    declare(tracingContext, "memory_write", { by: "facts_agent", memory: "notes", record: "Lisbon", value: text });
  });
  const translator = worker("translator", translatorAgent, c => `Translate the name ${c}.`);
  const Draft = z.object({ draft: z.string(), gathered: z.string(), iteration: z.number(), passed: z.boolean() });
  const handoff = createStep({
    id: "handoff", inputSchema: z.any(), outputSchema: Draft,
    execute: async ({ inputData, tracingContext }) => {
      const gathered = `Weather: ${inputData.weather?.text}. Facts: ${inputData.facts?.text}.`;
      const out = await relay.generate(`Have the writer write a two-sentence brief on Lisbon. ${gathered}`, { tracingContext });
      // the draft is the writer's answer that the delegation returned
      const results = (out.toolResults ?? []).map(r => r.payload ?? r).filter(r => r.toolName === "agent-writer" && !r.result?.error);
      const draft = results.length ? (results.at(-1).result?.text ?? JSON.stringify(results.at(-1).result)) : out.text;
      return { draft: draft.trim(), gathered, iteration: 0, passed: false };
    },
  });
  const review = createStep({
    id: "review", inputSchema: Draft, outputSchema: Draft,
    execute: async ({ inputData, tracingContext }) => {
      const iteration = inputData.iteration + 1;
      let passed, reason;
      // the reference task fails the first draft on purpose, so every framework runs the loop twice
      if (iteration === 1) [passed, reason] = [false, "the first draft is always sent back for revision"];
      else {
        const verdict = (await evaluator.generate(inputData.draft, { tracingContext })).text;
        [passed, reason] = [!verdict.toUpperCase().includes("FAIL") || iteration === 3, verdict.slice(0, 200)];
      }
      declare(tracingContext, "check", { by: "evaluator", checked: "draft", iteration, outcome: passed ? "pass" : "fail", reason });
      let draft = inputData.draft;
      if (!passed)
        draft = (await writer.generate(`The draft failed review: ${reason}. Revise it with this ${inputData.gathered}\n${draft}`, { tracingContext })).text.trim();
      return { draft, gathered: inputData.gathered, iteration, passed };
    },
  });
  const approve = createStep({
    id: "approve", inputSchema: Draft,
    outputSchema: z.object({ draft: z.string() }),
    suspendSchema: z.object({ draft: z.string() }), resumeSchema: z.object({ decision: z.string() }),
    execute: async ({ inputData, resumeData, suspend, tracingContext }) => {
      if (!resumeData) return await suspend({ draft: inputData.draft });
      declare(tracingContext, "approval", { by: "person", checked: "draft", outcome: resumeData.decision });
      return { draft: inputData.draft };
    },
  });
  const send = createStep({
    id: "send", inputSchema: z.object({ draft: z.string() }), outputSchema: z.object({ sent: z.string() }),
    execute: async ({ inputData, tracingContext }) => {
      const out = await sender.generate(`Send the brief to client: call send_brief with to=client and this text:\n${inputData.draft}`, { tracingContext });
      return { sent: out.text };
    },
  });

  const workflow = createWorkflow({ id: "city-brief", inputSchema: z.object({ city: z.string() }), outputSchema: z.object({ sent: z.string() }) })
    .then(route)
    .branch(CANDIDATES.map(name => [async ({ inputData }) => inputData.selected.includes(name), { weather, facts, translator }[name]]))
    .then(handoff)
    .dountil(review, async ({ inputData }) => inputData.passed)
    .then(approve)
    .then(send)
    .commit();
  const agents = { coordinator, weatherAgent, factsAgent, translatorAgent, writer, relay, evaluator, sender };
  return { workflow, agents, memory };
}

async function main() {
  const model = pick();
  const runId = "ma-" + new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
  const outDir = path.join(process.env.PROVAI_RUNS || path.join(ROOT, "data", "runs"), "mastra", runId);
  fs.mkdirSync(outDir, { recursive: true });
  const log = events(path.join(outDir, "events.jsonl"));
  log("run", { run_id: runId, model, framework: "mastra", versions: Object.fromEntries(
    ["@mastra/core", "@mastra/memory", "@mastra/observability", "@mastra/otel-exporter", "@openrouter/ai-sdk-provider"]
      .map(p => [p, version(p)])) });

  const otel = new OtelExporter({
    // the custom provider is required by the exporter; our span file replaces its network exporter
    provider: { custom: { endpoint: "http://127.0.0.1:9/unused", protocol: "http/json" } },
    exporter: spanFile(path.join(outDir, "spans.jsonl")), signals: { traces: true, logs: false },
  });
  const observability = new Observability({
    configs: { provai: { serviceName: "mastra-reference", exporters: [otel, nativeSpans(log)],
                          excludeSpanTypes: ["processor_run", "model_chunk"] } },
  });
  const storage = new InMemoryStore();
  const { workflow, agents, memory } = build(model, outDir, log, runId, storage);
  const mastra = new Mastra({ workflows: { cityBrief: workflow }, agents, storage, observability, logger: false });
  const wf = mastra.getWorkflow("cityBrief");
  // what each agent can call and which agents it can delegate to, from the agent objects
  log("agents", { definitions: Object.fromEntries(await Promise.all(Object.entries(agents).map(async ([role, a]) => [role, {
    name: a.name, tools: Object.keys(await a.listTools?.() ?? await a.getTools?.() ?? {}),
    agents: Object.keys(await a.listAgents?.() ?? {}),
  }]))) });
  log("workflow", { definition: wf.serializedStepGraph ?? wf.stepGraph, memory: { scope: SCOPE, resource: "notes" } });

  const run = await wf.createRun({ resourceId: "user" });
  run.watch(event => log("watch", { event }));
  let result = await run.start({ inputData: { city: "Lisbon" } });
  log("run_status", { status: result.status, suspended: result.suspended });
  if (result.status === "suspended") {
    // the person approves on the reference app's behalf
    log("person", { step: "approve", decision: "approve" });
    result = await run.resume({ step: "approve", resumeData: { decision: "approve" } });
    log("run_status", { status: result.status });
  }
  await observability.shutdown?.();
  await otel.shutdown();
  console.log(outDir, result.status);
}

await main();
