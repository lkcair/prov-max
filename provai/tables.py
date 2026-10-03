import csv
import json

from rdflib import OWL, RDF, RDFS, Graph

from .config import ONTOLOGY, PAI, PROV, ROOT, RUNS

PAPER = ROOT / "paper"
NAMES = {
    "langgraph": "LangGraph",
    "claude_agent_sdk": "Claude Agent SDK",
    "strands": "Strands Agents",
    "pydantic_ai": "Pydantic AI",
    "openai_agents": "OpenAI Agents SDK",
    "adk": "Google ADK",
    "mastra": "Mastra",
    "llamaindex": "LlamaIndex",
    "dspy": "DSPy",
    "crewai": "CrewAI",
    "agno": "Agno",
    "agent_framework": "Microsoft Agent Framework",
    "haystack": "Haystack",
    "smolagents": "smolagents",
    "agentscope": "AgentScope",
    "ag2": "AG2",
    "vercel_ai": "Vercel AI SDK",
    "protocols": "Protocol flow",
}
ORDER = list(NAMES)
MAIN = {"langgraph": "langgraph", "claude_agent_sdk": "claude-agent-sdk", "strands": "strands-agents",
        "pydantic_ai": "pydantic-ai-slim", "openai_agents": "openai-agents", "adk": "google-adk",
        "mastra": "@mastra/core", "llamaindex": "llama-index-core", "crewai": "crewai", "agno": "agno",
        "agent_framework": "agent-framework-core", "protocols": "mcp", "dspy": "dspy", "haystack": "haystack-ai",
        "smolagents": "smolagents", "agentscope": "agentscope", "ag2": "ag2", "vercel_ai": "ai"}
SPANS = {"langgraph": "OTel GenAI, OpenTelemetry LangChain instrumentor",
         "openai_agents": "OpenInference instrumentor",
         "pydantic_ai": "OTel GenAI, built in",
         "strands": "OTel GenAI, built in",
         "adk": "OTel GenAI, built in",
         "claude_agent_sdk": "OTel, beta enhanced telemetry",
         "agent_framework": "OTel GenAI, built in",
         "crewai": "OTel GenAI, built in",
         "llamaindex": "OpenInference instrumentor",
         "protocols": "MCP and A2A SDKs, OpenTelemetry OpenAI instrumentor",
         "mastra": "OTel GenAI, Mastra exporter",
         "agno": "OpenInference instrumentor",
         "dspy": "OpenInference instrumentor",
         "haystack": "OpenInference instrumentor",
         "ag2": "OTel GenAI, built in",
         "smolagents": "OpenInference instrumentor",
         "agentscope": "OTel GenAI, built in",
         "vercel_ai": "OTel GenAI, AI SDK integration"}
NATIVE = {"langgraph": "callbacks, custom events",
          "openai_agents": "trace processor, run hooks, agent objects",
          "pydantic_ai": "capability hooks, stream events",
          "strands": "hooks, graph and swarm objects",
          "adk": "plugin, event stream, LiteLLM logger",
          "claude_agent_sdk": "hooks, message stream, permission callback",
          "agent_framework": "workflow events, middleware",
          "crewai": "event bus, flow definition",
          "llamaindex": "dispatcher events, workflow stream and objects",
          "protocols": "JSON-RPC messages, A2A tasks and Agent Cards",
          "mastra": "Mastra spans, step callbacks, agent and workflow objects",
          "agno": "run events, workflow and team objects",
          "dspy": "callbacks, LM history",
          "haystack": "tracer interface, confirmation hook",
          "ag2": "observers, events, approval hook, agent objects",
          "smolagents": "agent memory, step callbacks, agent objects",
          "agentscope": "reply event stream, middleware, team objects",
          "vercel_ai": "lifecycle callbacks, tool approval, agent objects"}
GROUPS = {
    "Executions": "Session Run Step partOf iteration attempt retryOf branch sequence status errorType",
    "Agents": "Agent AgentInvocation spawnedBy accountableAgent",
    "Routing and handoff": "Routing candidate selected reason Handoff fromAgent toAgent Message",
    "Model and tool calls": "ModelCall Model usedModel provider temperature topP maxTokens seed finishReason inputTokens "
                            "outputTokens cost PromptTemplate version ToolCall Tool usedTool readOnly destructive "
                            "Retrieval Document",
    "Memory": "Memory MemoryRecord MemoryAccess validAt scope",
    "Checks and effects": "Check checked outcome Vote votedFor trustLevel Effect target irreversible",
    "Failure attribution": "FailureAttribution blamedStep blamedAgent",
    "Dependence and record": "dependsOn derivedFrom informedBy knownBy source contentCaptured",
}
WORDS = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def needs():
    with open(ROOT / "queries" / "needs.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(r["need"], [g.split("|") for g in r["terms"].split()], r["note"], r["name"]) for r in rows]


def recorded(groups, terms):
    return bool(groups) and all(any(t in terms for t in g) for g in groups)


def vocabulary():
    g = Graph().parse(ONTOLOGY)
    own = lambda kind: sum(1 for s in g.subjects(RDF.type, kind) if str(s).startswith(str(PAI)))
    return own(OWL.Class), own(OWL.ObjectProperty), own(OWL.DatatypeProperty)


def prov_terms(g, terms):
    # the PROV terms a group specializes, through PROV-AI parents, and those it closes
    up, down = [], []
    for t in terms:
        todo = [PAI[t]]
        while todo:
            node = todo.pop()
            for p in (*g.objects(node, RDFS.subClassOf), *g.objects(node, RDFS.subPropertyOf)):
                if str(p).startswith(str(PROV)):
                    up.append(str(p)[len(str(PROV)):])
                elif str(p).startswith(str(PAI)):
                    todo.append(p)
        down += [str(c)[len(str(PROV)):] for c in g.subjects(RDFS.subPropertyOf, PAI[t]) if str(c).startswith(str(PROV))]
    shown = list(dict.fromkeys(up))
    if down:
        shown.append("closes " + ", ".join(dict.fromkeys(down)))
    return ", ".join(shown)


def ordered(runs):
    return sorted(runs, key=lambda r: (ORDER.index(r["framework"]) if r["framework"] in ORDER else 99, r["path"]))


def write():
    PAPER.mkdir(exist_ok=True)
    out = json.loads((ROOT / "data" / "evaluation.json").read_text())
    runs = ordered(out["runs"])
    full = [r for r in runs if r["path"] == "full"]
    spans = {r["framework"]: r for r in runs if r["path"] == "spans"}
    cqs = list(out["questions"])
    need_rows = needs()
    covered = [n for n, groups, _, _ in need_rows if groups]
    in_any = [n for n, groups, _, _ in need_rows if any(recorded(groups, r["terms"]) for r in full)]
    in_all = [n for n, groups, _, _ in need_rows if groups and all(recorded(groups, r["terms"]) for r in full)]
    in_spans = [n for n, groups, _, _ in need_rows if any(recorded(groups, spans[r["framework"]]["terms"]) for r in full)]
    classes, objects, datas = vocabulary()
    answered = lambda r: sum(r["answers"].values())

    m = [
        macro("nFrameworks", len([r for r in full if r["framework"] != "protocols"])),
        macro("nNeeds", len(need_rows)),
        macro("nNeedsCovered", len(covered)),
        macro("nNeedsRecorded", len(in_any)),
        macro("nNeedsRecordedAll", len(in_all)),
        macro("nNeedsSpans", len(in_spans)),
        macro("nClasses", classes),
        macro("nObjectProps", objects),
        macro("nDataProps", datas),
        macro("cqSpansMin", min(answered(spans[r["framework"]]) for r in full)),
        macro("cqSpansMax", max(answered(spans[r["framework"]]) for r in full)),
        macro("cqFullMin", min(answered(r) for r in full)),
        macro("cqFullMax", max(answered(r) for r in full)),
        macro("taskFullAll", sum(all(r["task"].values()) for r in full if r["framework"] != "protocols")),
        macro("taskProtocols", "all" if all(all(r["task"].values()) for r in full if r["framework"] == "protocols")
              else "not all"),
        macro("taskSpansMax", WORDS[max(sum(spans[r["framework"]]["task"].values()) for r in full)]),
        macro("costMissing", WORDS[sum(not r["answers"]["CQ11"] for r in full)]),
        macro("mergedTriples", f"{out['merged_triples']:,}"),
        macro("queryMax", f"{max(out['query_s'].values()):.1f}"),
        macro("convertMax", f"{max(r['convert_s'] for r in full):.2f}"),
        macro("closeMin", f"{min(r['close_s'] for r in full):.0f}"),
        macro("closeMax", f"{max(r['close_s'] for r in full):.0f}"),
        macro("ontologyClose", f"{out['ontology_close_s']:.0f}"),
        macro("closeTotal", f"{sum(r['close_s'] for r in runs):.0f}"),
        macro("queryTotal", f"{sum(out['query_s'].values()):.0f}"),
        macro("triplesMin", min(r["triples"] for r in full)),
        macro("triplesMax", max(r["triples"] for r in full)),
    ]
    (PAPER / "gen-macros.tex").write_text("".join(m))

    head = " & ".join(f"\\textbf{{{c[2:]}}}" for c in cqs)
    lines = ["\\begin{tabular}{@{}l" + "c" * len(cqs) + "cc@{}}", "\\toprule",
             f"\\textbf{{Framework}} & {head} & \\textbf{{S/9}} & \\textbf{{I/9}} \\\\", "\\midrule"]
    for r in full:
        s = spans[r["framework"]]
        cells = ["S" if s["answers"][c] else ("I" if r["answers"][c] else "") for c in cqs]
        lines.append(f"{NAMES[r['framework']]} & " + " & ".join(cells)
                     + f" & {sum(s['task'].values())} & {sum(r['task'].values())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "gen-matrix.tex").write_text("\n".join(lines) + "\n")

    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "\\textbf{Framework} & \\textbf{Spans} & \\textbf{Record} & \\textbf{Closed} & "
             "\\textbf{Convert (s)} & \\textbf{Close (s)} \\\\", "\\midrule"]
    fast_convert = f"{min(r['convert_s'] for r in full):.2f}"
    fast_close = f"{min(r['close_s'] for r in full):.0f}"
    bold = lambda text, best: f"\\textbf{{{text}}}" if text == best else text
    for r in full:
        s = spans[r["framework"]]
        convert = bold(f"{r['convert_s']:.2f}", fast_convert)
        closing = bold(f"{r['close_s']:.0f}", fast_close)
        lines.append(f"{NAMES[r['framework']]} & {s['triples']} & {r['triples']} & {r['closed_triples']} & "
                     f"{convert} & {closing} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "gen-cost.tex").write_text("\n".join(lines) + "\n")

    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.2\\textwidth}>{\\raggedright\\arraybackslash}p{0.11\\textwidth}"
             ">{\\raggedright\\arraybackslash}p{0.29\\textwidth}>{\\raggedright\\arraybackslash}p{0.32\\textwidth}@{}}", "\\toprule",
             "\\textbf{Framework} & \\textbf{Version} & \\textbf{Spans} & \\textbf{Native interface} \\\\",
             "\\midrule"]
    for r in full:
        fw = r["framework"]
        with open(RUNS / fw / r["run"] / "events.jsonl") as f:
            versions = json.loads(f.readline()).get("versions", {})
        shown = versions.get(MAIN.get(fw), "")
        if fw == "protocols":
            shown = f"{versions.get('mcp', '')}, {versions.get('a2a-sdk', '')}"
        lines.append(f"{NAMES[fw]} & {shown} & {SPANS.get(fw, '')} & "
                     f"{NATIVE.get(fw, '')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "gen-frameworks.tex").write_text("\n".join(lines) + "\n")

    with open(ROOT / "data" / "declared.tsv") as f:
        declared = {r["framework"]: r for r in csv.DictReader(f, delimiter="\t")}
    facts = ["routing", "handoff", "memory", "verdict", "approval", "effect"]
    lines = ["\\begin{tabular}{@{}l" + "c" * len(facts) + "@{}}", "\\toprule",
             "\\textbf{Framework} & " + " & ".join(f"\\textbf{{{x.capitalize()}}}" for x in facts) + " \\\\",
             "\\midrule"]
    for r in full:
        row = declared.get(r["framework"])
        if row:
            lines.append(f"{NAMES[r['framework']]} & " + " & ".join({"yes": "D", "converter": "C"}.get(row[x], "N") for x in facts)
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "gen-declared.tex").write_text("\n".join(lines) + "\n")

    g = Graph().parse(ONTOLOGY)
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.14\\textwidth}>{\\raggedright\\arraybackslash}p{0.45\\textwidth}"
             ">{\\raggedright\\arraybackslash}p{0.2\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}@{}}", "\\toprule",
             "\\textbf{Group} & \\textbf{Terms} & \\textbf{PROV terms} & \\textbf{Needs} \\\\", "\\midrule"]
    for group, terms in GROUPS.items():
        terms = terms.split()
        served = [n for n, groups, _, _ in need_rows if any(t in terms for alt in groups for t in alt)]
        lines.append(f"{group} & {', '.join(terms)} & {prov_terms(g, terms)} & {', '.join(served)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "gen-terms.tex").write_text("\n".join(lines) + "\n")

    return m


if __name__ == "__main__":
    print("".join(write()))
