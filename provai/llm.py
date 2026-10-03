import json
import os
import urllib.error
import urllib.request

from .config import BUDGET_USD, MODELS, OPENROUTER, PAID, ROOT


def key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for env in (ROOT / ".env", ROOT.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit("set OPENROUTER_API_KEY or put it in a .env file")


def allowed(model):
    if not (model.endswith(":free") or model in PAID):
        raise SystemExit(f"{model} is neither free nor a listed low-cost model")
    return model


def spent():
    req = urllib.request.Request(f"{OPENROUTER}/credits", headers={"Authorization": f"Bearer {key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]["total_usage"]


def answers(model, tools=True):
    body = {"model": allowed(model), "max_tokens": 64,
            "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]}
    if tools:
        body["tools"] = [{"type": "function", "function": {
            "name": "get_weather", "description": "weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    req = urllib.request.Request(f"{OPENROUTER}/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return "choices" in json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit("OpenRouter refused the key (401)")
        if e.code == 402:
            raise SystemExit("OpenRouter has no credit left (402)")
        return False
    except Exception:
        return False


def pick(tools=True):
    # free models first, in order of preference; models listed in PROVAI_SKIP are skipped, e.g. after a rate limit
    skip = set(filter(None, os.environ.get("PROVAI_SKIP", "").split(",")))
    for model in MODELS:
        if model not in skip and answers(model, tools):
            return model
    start = os.environ.get("PROVAI_USAGE_START")
    if start is None:
        raise SystemExit("no free model answered; set PROVAI_USAGE_START to the credits used so far to allow paid models")
    if spent() - float(start) > BUDGET_USD:
        raise SystemExit(f"the paid budget of {BUDGET_USD} USD is spent")
    for model in PAID:
        if model not in skip and answers(model, tools):
            return model
    raise SystemExit("no model answered; try again later")
