from collections import Counter

from ..config import PAI, PROV
from ..spans import seconds

STATES = {"TASK_STATE_COMPLETED": "completed", "TASK_STATE_FAILED": "failed", "TASK_STATE_CANCELED": "cancelled",
          "TASK_STATE_REJECTED": "blocked", "TASK_STATE_INPUT_REQUIRED": "interrupted",
          "TASK_STATE_AUTH_REQUIRED": "interrupted"}


def text(parts):
    return " ".join(p.get("text", "") for p in parts or []).strip()


def cards(rows):
    return {r["agent"]: r["body"] for r in rows
            if r["direction"] == "response" and r["path"].endswith("agent-card.json") and r.get("body")}


def add(rec, rows, starts, client):
    # rows are the HTTP messages an A2A client sent and received; starts are the remote agents' task starts
    session = None
    requests = {r["body"]["id"]: r for r in rows if r["direction"] == "request" and r.get("body")}
    started = {s["message_id"]: s for s in starts}
    turns, artifacts, messages = {}, [], []
    runs = Counter()
    by = rec.agent(client)
    for r in sorted(rows, key=lambda r: seconds(r["time"])):
        if r["direction"] != "response" or not r.get("body") or "id" not in r["body"]:
            continue
        req = requests[r["body"]["id"]]
        if req["body"].get("method") != "SendMessage":
            continue
        sent = req["body"]["params"]["message"]
        task = r["body"]["result"]["task"]
        if session is None:
            session = rec.node("session", PAI.Session, task["contextId"], task["contextId"])
            rec.link(rec.run, PAI.partOf, session)
        remote = rec.agent(r["agent"])
        handoff = rec.step(f"a2a-{sent['messageId']}", PAI.Handoff, None, f"message to {r['agent']}",
                           req["time"], r["time"], "completed")
        rec.link(handoff, PROV.wasAssociatedWith, by)
        rec.link(handoff, PAI.fromAgent, by)
        rec.link(handoff, PAI.toAgent, remote)
        rec.link(remote, PROV.actedOnBehalfOf, by)
        message = rec.entity(f"a2a-{sent['messageId']}", PAI.Message, f"message to {r['agent']}", text(sent["parts"]))
        rec.generated(message, handoff)
        rec.link(message, PROV.wasAttributedTo, by)
        messages.append((req["time"], message, text(sent["parts"])))

        start = started[sent["messageId"]]
        runs[r["agent"]] += 1
        state = task["status"]["state"]
        turn = rec.step(f"a2a-task-{task['id']}", PAI.AgentInvocation, None, r["agent"], start["time"],
                        task["status"].get("timestamp"), STATES.get(state, "completed"))
        rec.set(turn, PAI.iteration, runs[r["agent"]])
        rec.link(turn, PROV.wasAssociatedWith, remote)
        rec.used(turn, message)
        rec.link(turn, PROV.wasInformedBy, handoff)
        turns[start["trace_id"]] = (turn, r["agent"])
        for a in task.get("artifacts") or []:
            art = rec.entity(f"a2a-{a['artifactId']}", PAI.Message, a.get("name") or "artifact", text(a["parts"]))
            rec.generated(art, turn)
            rec.link(art, PROV.wasAttributedTo, remote)
            artifacts.append((task["status"].get("timestamp"), art, text(a["parts"]), turn))
    return turns, artifacts, messages
