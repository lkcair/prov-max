import os
import re
import subprocess
import tempfile
from pathlib import Path

from rdflib import Graph
from rdflib.compare import isomorphic

from .config import ONTOLOGY, PAI, PROVO, ROOT, RUN
from .graph import close, with_ontology

JAR = Path(os.environ.get("HERMIT_JAR", "HermiT.jar"))


def export():
    g = Graph().parse(ONTOLOGY)
    owl = ONTOLOGY.with_suffix(".owl")
    g.serialize(owl, format="xml")
    return isomorphic(g, Graph().parse(owl, format="xml"))


INFERRED = ("dependsOn", "derivedFrom", "accountableAgent")


def compare(run_dir, jar=JAR):
    # HermiT reads at most three digits of fractional seconds
    text = re.sub(r"(T\d\d:\d\d:\d\d\.\d{3})\d+", r"\1", (run_dir / "record.ttl").read_text())
    text += "\n<https://w3id.org/prov-ai/check> a <http://www.w3.org/2002/07/owl#Ontology> ;\n" \
            "    <http://www.w3.org/2002/07/owl#imports> <https://w3id.org/prov-ai> .\n"
    with tempfile.NamedTemporaryFile("w", suffix=".ttl", delete=False) as f:
        f.write(text)
    try:
        out = subprocess.run(["java", "-Xmx1g", "-cp", str(jar), str(ROOT / "hermit" / "Check.java"),
                              str(PROVO), str(ONTOLOGY), f.name], capture_output=True, text=True)
    finally:
        Path(f.name).unlink()
    if out.returncode:
        raise SystemExit(out.stdout + out.stderr)
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    hermit = {tuple(line.split()) for line in lines if line.split()[0] in INFERRED}
    g = close(with_ontology(run_dir / "record.ttl"))
    rl = {(p, str(s), str(o)) for p in INFERRED
          for s, o in g.subject_objects(PAI[p]) if str(s).startswith(RUN)}
    return [line for line in lines if line.split()[0] not in INFERRED], \
        len(hermit), len(rl), sorted(hermit - rl), sorted(rl - hermit)
