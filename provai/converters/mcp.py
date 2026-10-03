import json
from collections import defaultdict
from urllib.parse import urlsplit

from ..config import PAI, PROV
from ..spans import same, seconds


def resource(uri):
    # notes://Lisbon is the record Lisbon of the memory notes
    parts = urlsplit(uri)
    return parts.netloc + parts.path


def content_text(result):
    return " ".join(c.get("text") or c.get("uri") or "" for c in (result or {}).get("content") or []).strip()


def annotations(rows):
    # tools/list results give each tool's declared hints
    hints = {}
    for r in rows:
        if r["method"] == "tools/list":
            for t in (r.get("result") or {}).get("tools", []):
                hints[t["name"]] = t.get("annotations") or {}
    return hints


def server_version(rows):
    # the server states its version when a client initializes the session
    for r in rows:
        if r["method"] == "initialize":
            return ((r.get("result") or {}).get("serverInfo") or {}).get("version")


def add(rec, rows, elicitations, spans_by_id, parent_for, approver):
    # rows are the MCP requests the server received; returns tool steps by the model's call id and records by URI
    hints, version = annotations(rows), server_version(rows)
    tool_steps, records = defaultdict(list), {}
    failed = {}
    calls_by_span = {}
    effects, outcomes = [], {}
    for r in sorted(rows, key=lambda r: seconds(r["time"])):
        if r["method"] not in ("tools/call", "resources/read"):
            continue
        span = spans_by_id.get("0x" + (r.get("span_id") or ""))
        if span is None:
            raise SystemExit(f"MCP request {r['client']} {r['id']} has no server span")
        span["matched"] = True
        agent = rec.agent(r["client"])
        parent = parent_for(span)
        params = r.get("params") or {}
        local = f"mcp-{r['client']}-{r['id']}"
        if r["method"] == "tools/call":
            name = params["name"]
            result = r.get("result") or {}
            failed_now = bool(result.get("isError")) or "error" in r
            step = rec.step(local, PAI.ToolCall, parent, name, span["start"], span["end"],
                            "failed" if failed_now else "completed", span["error"] if failed_now else None)
            rec.link(step, PROV.wasAssociatedWith, agent)
            tool = rec.tool(name)
            rec.link(step, PAI.usedTool, tool)
            hint = hints.get(name, {})
            rec.set(tool, PAI.readOnly, hint.get("readOnlyHint"))
            rec.set(tool, PAI.destructive, hint.get("destructiveHint"))
            rec.set(tool, PAI.version, version)
            args = rec.entity(f"{local}-arguments", PROV.Entity, f"{name} arguments", json.dumps(params.get("arguments")))
            rec.used(step, args)
            res = None
            if result:
                res = rec.entity(f"{local}-result", PROV.Entity, f"{name} result", content_text(result))
                rec.generated(res, step)
            cid = (params.get("_meta") or {}).get("toolCallId")
            if cid:
                tool_steps[cid].append((step, res))
            key = (r["client"], name, same(params.get("arguments")))
            attempt = 1
            if key in failed:
                before, attempt = failed.pop(key)
                attempt += 1
                rec.link(step, PAI.retryOf, before)
                rec.set(step, PAI.attempt, attempt)
            if failed_now:
                failed[key] = (step, attempt)
            calls_by_span[span["id"]] = (step, args, agent)
            if failed_now:
                continue
            # a resource link returned by a tool that is not read-only names the resource the call wrote
            for c in result.get("content") or []:
                if c.get("type") == "resource_link" and not hint.get("readOnlyHint"):
                    uri = c["uri"]
                    access = rec.step(f"{local}-write", PAI.MemoryAccess, step, "memory write", span["start"],
                                      span["end"], "completed")
                    rec.link(access, PROV.wasAssociatedWith, agent)
                    record = rec.memory_write(urlsplit(uri).scheme, resource(uri), None, access)
                    records.setdefault(uri, []).append(record)
            if hint.get("destructiveHint"):
                effects.append((step, local, name, content_text(result), agent))
        else:
            uri = params["uri"]
            step = rec.step(local, PAI.MemoryAccess, parent, "memory read", span["start"], span["end"],
                            "failed" if "error" in r else "completed")
            rec.link(step, PROV.wasAssociatedWith, agent)
            text = " ".join(c.get("text", "") for c in (r.get("result") or {}).get("contents", []))
            out = rec.entity(f"{local}-result", PROV.Entity, f"{uri} contents", text)
            rec.generated(out, step)
            record = rec.memory_read(urlsplit(uri).scheme, resource(uri), step)
            if record is not None:
                if rec.g.value(record, PROV.value) is None:
                    rec.set(record, PROV.value, text)
                rec.derived(out, record)

    for e in elicitations:
        # the elicitation request's trace context names the server span that sent it, inside the tool call
        span_id = "0x" + e["params"]["_meta"]["traceparent"].split("-")[2]
        sender = spans_by_id.get(span_id)
        if sender is None or sender["parent"] not in calls_by_span:
            raise SystemExit("an elicitation has no tool call around it")
        sender["matched"] = True
        step, args, agent = calls_by_span[sender["parent"]]
        check = rec.step(f"{span_id}-elicitation", PAI.Check, step, "approval", e["time"], e["time"], "completed")
        rec.link(check, PROV.wasAssociatedWith, approver)
        rec.link(check, PAI.checked, args)
        # the tool call goes on only after the person answers
        rec.link(step, PROV.wasInformedBy, check)
        answer = e["answer"]
        # MCP answers an elicitation with accept, decline, or cancel
        outcome = {"accept": "approve", "cancel": "cancel"}.get(answer.get("action"), "reject")
        rec.set(check, PAI.outcome, outcome)
        rec.set(check, PAI.reason, e["params"].get("message"))
        outcomes[step] = outcome

    for step, local, name, text, agent in effects:
        # a destructive tool changes something outside the run, unless the person it asked did not approve
        if outcomes.get(step, "approve") != "approve":
            continue
        effect = rec.entity(f"{local}-effect", PAI.Effect, f"{name} effect", text)
        rec.set(effect, PAI.target, text)
        rec.generated(effect, step)
        rec.link(effect, PROV.wasAttributedTo, agent)
    return tool_steps, records
