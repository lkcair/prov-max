from decimal import Decimal
from urllib.parse import quote, unquote

from rdflib import RDF, RDFS, XSD, Graph, Literal, URIRef

from .config import PAI, PROV, RUN


def literal(value):
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return Literal(format(Decimal(repr(value)), "f"), datatype=XSD.decimal)
    return Literal(str(value))


class Record:
    def __init__(self, source, run_id, content_captured):
        self.g = Graph()
        self.g.bind("provai", PAI)
        self.g.bind("prov", PROV)
        self.base = f"{RUN}{quote(source.split()[0].lower(), safe='')}/{quote(str(run_id), safe='')}/"
        self.run = self.node("run", PAI.Run)
        self.set(self.run, PAI.source, source)
        self.set(self.run, PAI.contentCaptured, content_captured)
        self.versions = {}

    def local(self, node):
        # the local name of a node, unquoted, to build the names of the entities it owns
        return unquote(str(node).rsplit("/", 1)[-1])

    def iri(self, kind, local):
        return URIRef(f"{self.base}{kind}/{quote(str(local), safe='')}")

    def node(self, kind, cls, local="", label=None):
        n = self.iri(kind, local) if local else URIRef(self.base + kind)
        self.g.add((n, RDF.type, cls))
        if label:
            self.g.add((n, RDFS.label, Literal(label)))
        return n

    def set(self, s, p, value):
        lit = literal(value) if value is not None and value != "" else None
        if lit is not None:
            self.g.add((s, p, lit))

    def link(self, s, p, o):
        if s is not None and o is not None:
            self.g.add((s, p, o))

    def step(self, local, cls, parent=None, label=None, start=None, end=None, status=None, error=None):
        n = self.node("step", cls, local, label)
        self.link(n, PAI.partOf, parent if parent is not None else self.run)
        if start:
            self.g.add((n, PROV.startedAtTime, Literal(start, datatype=XSD.dateTime)))
        if end:
            self.g.add((n, PROV.endedAtTime, Literal(end, datatype=XSD.dateTime)))
        self.set(n, PAI.status, status)
        self.set(n, PAI.errorType, error)
        return n

    def agent(self, name):
        return self.node("agent", PAI.Agent, name, name) if name else None

    def model(self, name, provider=None):
        if not name:
            return None
        n = self.node("model", PAI.Model, f"{provider or ''}/{name}", name)
        self.set(n, PAI.provider, provider)
        return n

    def memory_write(self, memory, key, value, access):
        # each write to a key makes a new record that revises the one before it
        n = self.versions[(memory, key)] = self.versions.get((memory, key), 0) + 1
        record = self.entity(f"{memory}/{key}/{n}", PAI.MemoryRecord, key, value)
        self.generated(record, access)
        self.link(self.node("memory", PAI.Memory, memory, memory), PROV.hadMember, record)
        if n > 1:
            self.link(record, PROV.wasRevisionOf, self.iri("entity", f"{memory}/{key}/{n - 1}"))
        return record

    def memory_read(self, memory, key, access):
        n = self.versions.get((memory, key))
        if not n:
            return None
        record = self.iri("entity", f"{memory}/{key}/{n}")
        self.used(access, record)
        return record

    def tool(self, name):
        return self.node("tool", PAI.Tool, name, name)

    def entity(self, local, cls=PROV.Entity, label=None, value=None):
        n = self.node("entity", cls, local, label)
        self.set(n, PROV.value, value)
        return n

    def used(self, step, entity):
        self.link(step, PROV.used, entity)

    def generated(self, entity, step):
        self.link(entity, PROV.wasGeneratedBy, step)

    def derived(self, entity, source, known=None):
        if entity is None or source is None:
            return
        self.link(entity, PROV.wasDerivedFrom, source)
        if known:
            q = URIRef(f"{entity}/derivation/{quote(self.local(source), safe='')}")
            self.g.add((entity, PROV.qualifiedDerivation, q))
            self.g.add((q, RDF.type, PROV.Derivation))
            self.g.add((q, PROV.entity, source))
            self.set(q, PAI.knownBy, known)

    def turtle(self):
        return self.g.serialize(format="turtle")
