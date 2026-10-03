import json
from datetime import datetime


def exception_type(span):
    for e in span.get("events") or []:
        if e.get("name") == "exception":
            return (e.get("attributes") or {}).get("exception.type")


def seconds(t):
    # spans write "Z" and Python events "+00:00", so times are compared as numbers
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def same(arguments):
    # one form for tool arguments, so a retry with the same arguments is recognized
    try:
        return json.dumps(json.loads(arguments) if isinstance(arguments, str) else arguments, sort_keys=True)
    except (TypeError, ValueError):
        return str(arguments)


def latest(items, t, slack=0):
    # items are (end in seconds, value); the value of the latest one that ended by t
    found = [(end, value) for end, value in items if end is not None and end <= t + slack]
    return max(found, key=lambda x: x[0])[1] if found else None


def read(path):
    spans = []
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for s in rows:
        attrs = s.get("attributes") or {}
        status = (s.get("status") or {}).get("status_code", "UNSET")
        spans.append({
            "id": s["context"]["span_id"],
            "trace": s["context"]["trace_id"],
            "parent": s.get("parent_id"),
            "name": s.get("name", ""),
            "start": s.get("start_time"),
            "end": s.get("end_time"),
            "status": "failed" if status == "ERROR" else "completed",
            "error": attrs.get("error.type") or (exception_type(s) or (s.get("status") or {}).get("description")
                                                 if status == "ERROR" else None),
            "attrs": attrs,
            "events": s.get("events") or [],
            "links": s.get("links") or [],
        })
    by_id = {s["id"]: s for s in spans}

    def depth(s):
        n = 0
        while s["parent"] in by_id:
            s, n = by_id[s["parent"]], n + 1
        return n
    # clocks with millisecond resolution can give a child its parent's start time
    return sorted(spans, key=lambda s: (seconds(s["start"]) if s["start"] else 0, depth(s)))
