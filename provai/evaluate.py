import importlib
import json
import re
import time
from collections import defaultdict

from rdflib import RDF, Graph, Literal

from .config import PAI, PROV, QUERIES, ROOT, RUN, RUNS
from .graph import check, close, with_ontology

REPEATS = 5
PREFIXES = ("PREFIX provai: <https://w3id.org/prov-ai#>\nPREFIX prov: <http://www.w3.org/ns/prov#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")


def questions():
    parts = re.split(r"(?m)^# (CQ\d+) ", QUERIES.read_text())[1:]
    out = []
    for name, body in zip(parts[::2], parts[1::2]):
        text, needs, query = body.split("\n", 2)
        out.append((name, text, needs.removeprefix("# needs: ").replace("?", "").split(), query))
    return out


def span_format(run_dir):
    with open(run_dir / "spans.jsonl") as f:
        return "openinference" if any("openinference.span.kind" in line for line in f) else "otel_genai"


def tail(term):
    return str(term).rsplit("/", 1)[-1] if term is not None else None


def filled(value):
    if value is None:
        return False
    if isinstance(value, Literal) and value.toPython() in (0, "", False):
        return False
    return True


def task(answers):
    # the reference task's behaviors, read from the answers to the questions
    def rows(cq):
        return answers.get(cq, [])

    def has(cq, var, word):
        return any(word in (tail(r.get(var)) or "") for r in rows(cq))

    checks = sorted((int(r["iteration"]), str(r["outcome"])) for r in rows("CQ8") if r.get("iteration"))
    return {
        "routing": has("CQ4", "selected", "weather") and has("CQ4", "selected", "facts")
                   and has("CQ4", "candidate", "translator") and not has("CQ4", "selected", "translator"),
        "parallel": any({"weather", "facts"} <= {w for w in ("weather", "facts")
                                                 if w in tail(r["a"]) or w in tail(r["b"])} for r in rows("CQ9")),
        "retry": any(r.get("retry") for r in rows("CQ8")),
        "handoff": has("CQ3", "to", "writer"),
        "memory": has("CQ5", "writer", "facts") and has("CQ5", "reader", "writer"),
        "loop": bool(checks) and checks[0][1] == "fail" and checks[-1][1] == "pass",
        "approval": any(str(r["outcome"]) == "approve" for r in rows("CQ6")),
        "lineage": has("CQ7", "tool", "get_weather"),
        "accountable": any(r.get("person") for r in rows("CQ3")),
    }


def terms_used(g):
    # the PROV-AI and PROV terms a record uses, as classes of its nodes or as predicates
    used = set()
    for s, p, o in g:
        if not str(s).startswith(RUN):
            continue
        term = o if p == RDF.type else p
        for ns, prefix in ((str(PAI), ""), (str(PROV), "prov:")):
            if str(term).startswith(ns):
                used.add(prefix + str(term)[len(ns):])
    return sorted(used)


def evaluate(frameworks=None):
    qs = questions()
    merged, runs = Graph(), {}
    converters = ROOT / "provai" / "converters"
    for fw_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and (converters / f"{p.name}.py").exists()):
        if frameworks and fw_dir.name not in frameworks:
            continue
        for run_dir in sorted(p for p in fw_dir.iterdir() if (p / "spans.jsonl").exists()):
            for path, module in (("spans", span_format(run_dir)), ("full", fw_dir.name)):
                converter = importlib.import_module(f"provai.converters.{module}")
                converted, closed = [], []
                for _ in range(REPEATS):
                    t = time.perf_counter()
                    rec = converter.convert(run_dir)
                    converted.append(time.perf_counter() - t)
                    t = time.perf_counter()
                    g = close(with_ontology(rec.g))
                    closed.append(time.perf_counter() - t)
                ok, _ = check(rec.g)
                merged += g
                runs[str(rec.run)] = {"framework": fw_dir.name, "run": run_dir.name, "path": path,
                                      "converter": module, "triples": len(rec.g), "conforms": ok,
                                      "convert_s": round(min(converted), 3), "close_s": round(min(closed), 3),
                                      "closed_triples": len(g), "terms": terms_used(g), "answers": {}}
    baseline = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        close(with_ontology())
        baseline.append(time.perf_counter() - t)
    timings = {}
    for name, text, needs, query in qs:
        spent = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            result = merged.query(PREFIXES + query)
            rows = [{str(v): r[v] for v in result.vars if r[v] is not None} for r in result]
            spent.append(time.perf_counter() - t)
        timings[name] = round(min(spent), 3)
        by_run = defaultdict(list)
        for r in rows:
            by_run[str(r["run"])].append(r)
        for run, info in runs.items():
            got = by_run.get(run, [])
            info["answers"][name] = all(any(filled(r.get(n)) for r in got) for n in needs)
            info.setdefault("rows", {})[name] = got
    for info in runs.values():
        info["task"] = task(info.pop("rows"))
    out = {"questions": {n: t for n, t, _, _ in qs}, "query_s": timings, "merged_triples": len(merged),
           "ontology_close_s": round(min(baseline), 3),
           "runs": list(runs.values())}
    (ROOT / "data" / "evaluation.json").write_text(json.dumps(out, indent=1))
    return out


def show(out):
    names = list(out["questions"])
    print("framework       path   triples conf  " + " ".join(f"{n[2:]:>3}" for n in names) + "  task")
    for r in out["runs"]:
        cells = " ".join(f"{'y' if r['answers'][n] else '.':>3}" for n in names)
        done = sum(r["task"].values())
        print(f"{r['framework']:<15} {r['path']:<6} {r['triples']:>7} {'yes' if r['conforms'] else 'NO':<5} "
              f"{cells}  {done}/{len(r['task'])} {[k for k, v in r['task'].items() if not v]}")
    print(f"merged graph {out['merged_triples']} triples; query seconds {out['query_s']}")
