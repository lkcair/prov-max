import os
from pathlib import Path

from rdflib import Namespace

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "ontology" / "prov-ai.ttl"
SHAPES = ROOT / "ontology" / "prov-ai-shapes.ttl"
PROVO = ROOT / "ontology" / "prov-o.ttl"
RUNS = Path(os.environ.get("PROVAI_RUNS", ROOT / "data" / "runs"))
QUERIES = ROOT / "queries" / "questions.rq"

PAI = Namespace("https://w3id.org/prov-ai#")
PROV = Namespace("http://www.w3.org/ns/prov#")
RUN = "https://w3id.org/prov-ai/run/"

OPENROUTER = "https://openrouter.ai/api/v1"
MODELS = (
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "cohere/north-mini-code:free",
    "qwen/qwen3.8-27b:free",
)
# paid fallbacks once the free daily quota is spent, each under 5 USD per million output tokens
PAID = (
    "meta-llama/llama-4-scout",
    "google/gemma-4-26b-a4b-it",
    "openai/gpt-oss-120b",
)
BUDGET_USD = 3.0
