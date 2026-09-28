#!/usr/bin/env python3
"""Check a public OpenInference span JSONL against Bandits' OTLP reader.

The input is the OpenTelemetry-style span JSONL published by
inference-net/SearchAgentDemoTraces. It is not an OTLP envelope. This script
wraps each recorded span in OTLP/JSON without changing its attributes, then
compares the decoded model spans back to the source. It reports counts only;
trace content is never printed.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bandits.ingest import load_corpus
from bandits.ingest.native import _ns
from bandits.redact import RedactionRuleset
from bandits.traces import SpanKind, SpanStatus, WorkflowDeclaration

NO_REDACTION = RedactionRuleset("none-openinference-validation", ())
MESSAGE_KEY = re.compile(r"^llm\.(input|output)_messages\.(\d+)\.message\.(role|content)$")
TOOL_NAME_KEY = re.compile(
    r"^llm\.(input|output)_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.function\.name$"
)


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_value(item) for item in value]}}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, ensure_ascii=False)}


def _attrs(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": _value(value)} for key, value in values.items() if value is not None
    ]


def _otlp(source: dict[str, Any]) -> dict[str, Any]:
    span = {
        "traceId": source["trace_id"],
        "spanId": source["span_id"],
        "name": source["name"],
        "kind": 3,
        "startTimeUnixNano": str(_ns(source["start_time"])),
        "endTimeUnixNano": str(_ns(source["end_time"])),
        "attributes": _attrs(source.get("attributes") or {}),
        "status": {
            "code": 2 if (source.get("status") or {}).get("code") == "STATUS_CODE_ERROR" else 1
        },
    }
    if source.get("parent_span_id"):
        span["parentSpanId"] = source["parent_span_id"]
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attrs((source.get("resource") or {}).get("attributes") or {})
                },
                "scopeSpans": [{"scope": source.get("scope") or {}, "spans": [span]}],
            }
        ]
    }


def _batches(path: Path, traces_per_batch: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    traces: set[str] = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_id = row["trace_id"]
            if batch and trace_id not in traces and len(traces) >= traces_per_batch:
                yield batch
                batch, traces = [], set()
            batch.append(row)
            traces.add(trace_id)
    if batch:
        yield batch


def _source_messages(attrs: dict[str, Any], direction: str) -> list[tuple[str | None, str | None]]:
    found: dict[int, dict[str, str]] = {}
    for key, value in attrs.items():
        match = MESSAGE_KEY.match(key)
        if match and match[1] == direction:
            found.setdefault(int(match[2]), {})[match[3]] = value
    return [(found[i].get("role"), found[i].get("content") or "") for i in sorted(found)]


def _decoded_messages(value: Any) -> list[tuple[str | None, str | None]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    return [
        (
            message.get("role"),
            "".join(
                part.get("content", "") if part.get("type") == "text" else part.get("result", "")
                for part in message.get("parts", [])
                if part.get("type") in ("text", "tool_call_response")
            ),
        )
        for message in value
        if isinstance(message, dict)
    ]


def _source_tool_names(attrs: dict[str, Any], direction: str) -> list[str]:
    found = []
    for key, value in attrs.items():
        match = TOOL_NAME_KEY.match(key)
        if match and match[1] == direction:
            found.append((int(match[2]), int(match[3]), value))
    return [name for _, _, name in sorted(found)]


def _decoded_tool_names(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    return [
        part["name"]
        for message in value
        if isinstance(message, dict)
        for part in message.get("parts", [])
        if isinstance(part, dict) and part.get("type") == "tool_call" and "name" in part
    ]


def validate(path: Path, traces_per_batch: int = 50) -> Counter[str]:
    counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="bandits-openinference-") as directory:
        sample = Path(directory) / "spans.otlp.jsonl"
        for rows in _batches(path, traces_per_batch):
            with sample.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(_otlp(row), ensure_ascii=False) + "\n")
            corpus = load_corpus(
                sample,
                "otlp-std",
                NO_REDACTION,
                workflow=WorkflowDeclaration(task_fields=()),
            )
            decoded = {
                (trace.trace_id, span.span_id): span
                for trace in corpus.traces
                for span in trace.spans
                if span.kind == SpanKind.MODEL
            }
            all_spans = {
                (trace.trace_id, span.span_id): span
                for trace in corpus.traces
                for span in trace.spans
            }
            nodes = {
                (trace.trace_id, node.span_id): node
                for trace in corpus.traces
                for node in trace.workflow_nodes
            }
            invocations = {
                (trace.trace_id, trace.request.source_span_id): trace.request
                for trace in corpus.traces
                if trace.request and trace.request.source_span_id
            }
            wrappers = {
                (trace.trace_id, node.span_id): node
                for trace in corpus.traces
                for node in trace.workflow_nodes
                if node.attributes.get("bandits.duplicate_model_of")
            }
            counts["traces"] += len(corpus.traces)
            counts["source_spans"] += len(rows)
            counts["model_actions"] += len(decoded)
            counts["duplicate_wrappers"] += len(wrappers)
            for issue in corpus.issues:
                counts[f"issue:{issue.kind}"] += 1
            for row in rows:
                key = (row["trace_id"], row["span_id"])
                represented = all_spans.get(key) or nodes.get(key)
                error = (row.get("status") or {}).get("code") == "STATUS_CODE_ERROR"
                if error:
                    counts["source_error_spans"] += 1
                if represented is None and key not in invocations:
                    counts["missing_structural_record"] += 1
                if represented is not None:
                    if represented.parent_span_id != row.get("parent_span_id"):
                        counts["wrong_parent"] += 1
                    if (represented.status == SpanStatus.ERROR) != error:
                        counts["wrong_status"] += 1
                elif key in invocations and (invocations[key].status == SpanStatus.ERROR) != error:
                    counts["wrong_invocation_status"] += 1
                if (row.get("attributes") or {}).get("openinference.span.kind") != "LLM":
                    continue
                counts["source_model_spans"] += 1
                if key in wrappers:
                    continue
                span = decoded.get(key)
                if span is None:
                    counts["missing_model_span"] += 1
                    continue
                attrs = row.get("attributes") or {}
                if span.attributes.get("bandits.output_unusable_reason"):
                    counts["detected:unusable_output"] += 1
                raw_output = attrs.get("output.value")
                if isinstance(raw_output, str):
                    try:
                        parsed_output = json.loads(raw_output)
                    except json.JSONDecodeError:
                        parsed_output = raw_output
                    if isinstance(parsed_output, str) and parsed_output.startswith(
                        "<APIResponse ["
                    ):
                        counts["source:api_response_placeholder"] += 1
                for field in ("input.value", "output.value"):
                    if field in attrs:
                        counts[f"source:{field}"] += 1
                        if span.attributes.get(field) != attrs[field]:
                            counts[f"lost:{field}"] += 1
                for direction in ("input", "output"):
                    expected = _source_messages(attrs, direction)
                    normalized = span.attributes.get(f"gen_ai.{direction}.messages")
                    if expected:
                        counts[f"source:{direction}_messages"] += 1
                        actual = _decoded_messages(normalized)
                        if actual != expected:
                            counts[f"mismatch:{direction}_messages"] += 1
                    source_tools = _source_tool_names(attrs, direction)
                    if source_tools:
                        counts[f"source:{direction}_tool_calls"] += len(source_tools)
                        if _decoded_tool_names(normalized) != source_tools:
                            counts[f"mismatch:{direction}_tool_calls"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--traces-per-batch", type=int, default=50)
    args = parser.parse_args()
    counts = validate(args.path, args.traces_per_batch)
    print(json.dumps(dict(sorted(counts.items())), indent=2))
    if any(
        key.startswith(("lost:", "mismatch:", "wrong_", "missing_model_span")) and value
        for key, value in counts.items()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
