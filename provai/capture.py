import json
from datetime import datetime, timezone

from .config import RUNS


def run_dir(framework, run_id):
    d = RUNS / framework / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def tracer_provider(path, service):
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class JsonLines(SpanExporter):
        def __init__(self):
            self.out = open(path, "a")

        def export(self, spans):
            for s in spans:
                self.out.write(s.to_json(indent=None) + "\n")
            self.out.flush()
            return SpanExportResult.SUCCESS

        def shutdown(self):
            self.out.close()

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(JsonLines()))
    trace.set_tracer_provider(provider)
    return provider


class Events:
    def __init__(self, path):
        self.out = open(path, "a")

    def __call__(self, kind, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        self.out.write(json.dumps(row, default=str) + "\n")
        self.out.flush()

    def close(self):
        self.out.close()
