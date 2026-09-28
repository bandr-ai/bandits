"""Pytest plugin: also write every span an upstream test exports as OTLP/JSON.

Runs an instrumentation library's own test suite (recorded API responses) and
keeps the spans its real instrumentor produced, for `scripts/audit_source.py`.

    cd openllmetry/packages/opentelemetry-instrumentation-openai
    UV_PROJECT_ENVIRONMENT=.venv-dump uv sync --group test
    VIRTUAL_ENV=.venv-dump uv pip install opentelemetry-exporter-otlp-proto-common protobuf
    SPAN_DUMP=OUT PYTHONPATH=BANDITS/scripts/replay OPENAI_API_KEY=test \
        .venv-dump/bin/pytest tests -p span_dump --record-mode=none

Span ids are written as OTLP/JSON hex. One JSONL file per test that exported spans.
"""

import base64
import json
import os
import re
from pathlib import Path

import pytest
from google.protobuf.json_format import MessageToDict
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_OUT = Path(os.environ["SPAN_DUMP"])
_OUT.mkdir(parents=True, exist_ok=True)
_current: dict[str, str] = {}
_original = InMemorySpanExporter.export


def _export(self, spans):
    name = _current.get("test")
    if name and spans:
        request = _hex_ids(MessageToDict(encode_spans(spans)))
        path = _OUT / f"{name}.jsonl"
        with path.open("a") as out:
            out.write(json.dumps(request) + "\n")
    return _original(self, spans)


def _hex_ids(request):
    # protobuf's JSON mapping writes bytes as base64; OTLP/JSON requires hex ids.
    for resource in request.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                for item in (span, *span.get("links", [])):
                    for key in ("traceId", "spanId", "parentSpanId"):
                        if key in item:
                            item[key] = base64.b64decode(item[key]).hex()
    return request


InMemorySpanExporter.export = _export


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["test"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.nodeid)[-150:]
