"""Readers for documented native trace exports, routed through the OTLP reader.

OpenInference is an OTLP convention, not a separate file format. Phoenix's
Span JSON and LangSmith's Run JSON are native formats; Langfuse's bundled
trace/observation JSON is another. This module converts their recorded fields
to OTLP internally and retains each native record in a namespaced attribute.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_bytes
from bandits.traces import TraceCorpus, TraceIssue, WorkflowDeclaration


def _hex_id(value: object, size: int, namespace: str) -> str:
    raw = str(value or "")
    normalized = raw.removeprefix("0x").replace("-", "").lower()
    if len(normalized) == size and all(c in "0123456789abcdef" for c in normalized):
        return normalized
    return hashlib.sha256(f"{namespace}:{raw}".encode()).hexdigest()[:size]


def _ns(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        # datetime only stores microseconds; keep the remaining three digits
        # separately so an ISO timestamp with nanoseconds is not rounded away.
        fraction = re.search(r"\.(\d+)", value)
        extra = 0
        if fraction:
            digits = fraction.group(1)[:9].ljust(9, "0")
            extra = int(digits[6:])
            value = value.replace(fraction.group(0), "." + digits[:6], 1)
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000 + extra


def _av(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, ensure_ascii=False)}


def _attrs(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _av(v)} for k, v in values.items()]


def _records(data: bytes) -> Iterator[dict[str, Any]]:
    stripped = data.lstrip()
    if stripped.startswith(b"["):
        parsed = json.loads(data)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON array")
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("expected objects in JSON array")
            yield item
    elif stripped.startswith(b"{") and (
        b"\n" not in stripped or not stripped.split(b"\n", 1)[0].rstrip().endswith(b"}")
    ):
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        yield parsed
    else:
        for line in io.BytesIO(data):
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("expected one object per JSONL line")
                yield parsed


def _span(
    trace_id: object,
    span_id: object,
    parent_id: object,
    name: object,
    start: object,
    end: object,
    attributes: dict[str, Any],
    *,
    namespace: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    beginning, ending = _ns(start), _ns(end)
    if beginning is None or ending is None or not span_id:
        return None
    result: dict[str, Any] = {
        "traceId": _hex_id(trace_id, 32, namespace + ":trace"),
        "spanId": _hex_id(span_id, 16, namespace + ":span"),
        "name": str(name or "unnamed"),
        "startTimeUnixNano": str(beginning),
        "endTimeUnixNano": str(ending),
        "attributes": _attrs(attributes),
    }
    if parent_id:
        result["parentSpanId"] = _hex_id(parent_id, 16, namespace + ":span")
    if extra:
        result.update(extra)
    return result


def _langfuse(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record.get("observations"), list):
        raise ValueError("Langfuse input requires a trace with observations[]")
    trace_id = record.get("trace_id") or record.get("id")
    if not trace_id:
        raise ValueError("Langfuse trace has no trace_id/id")
    spans: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = [(o, None) for o in reversed(record["observations"])]
    while stack:
        observation, enclosing_id = stack.pop()
        if not isinstance(observation, dict):
            continue
        native_id = observation.get("id")
        if native_id is None or str(native_id) in seen:
            continue
        seen.add(str(native_id))
        attributes = {
            "langfuse.observation.type": observation.get("type"),
            "langfuse.observation.id": native_id,
            "langfuse.observation.parent_id": observation.get("parentObservationId"),
            "input.value": observation.get("input"),
            "output.value": observation.get("output"),
            "gen_ai.request.model": observation.get("model"),
            "langfuse.observation.level": observation.get("level"),
            "bandits.native.record": {k: v for k, v in observation.items() if k != "children"},
        }
        if observation.get("parentObservationId") is None and enclosing_id is None:
            attributes["bandits.native.trace_record"] = {
                k: v for k, v in record.items() if k != "observations"
            }
        metadata = observation.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                attributes[f"metadata.{key}"] = value
        span = _span(
            trace_id,
            native_id,
            observation.get("parentObservationId") or enclosing_id,
            observation.get("name"),
            observation.get("startTime"),
            observation.get("endTime"),
            attributes,
            namespace="langfuse",
        )
        if span is not None:
            if isinstance(metadata, dict):
                if isinstance(metadata.get("resourceAttributes"), dict):
                    span["_native_resource"] = metadata["resourceAttributes"]
                if isinstance(metadata.get("scope"), dict):
                    span["_native_scope"] = metadata["scope"]
            spans.append(span)
        for child in reversed(observation.get("children") or []):
            stack.append((child, native_id))
    return spans


_LANGSMITH_KIND = {
    "llm": "LLM",
    "chat_model": "LLM",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "chain": "CHAIN",
    "prompt": "PROMPT",
}


def _langsmith(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not record.get("id") or not record.get("run_type"):
        raise ValueError("LangSmith run requires id and run_type")
    attributes = {
        "openinference.span.kind": _LANGSMITH_KIND.get(str(record["run_type"]).lower(), "UNKNOWN"),
        "input.value": record.get("inputs"),
        "output.value": record.get("outputs"),
        "bandits.native.record": {k: v for k, v in record.items() if k != "child_runs"},
    }
    inputs, outputs = record.get("inputs"), record.get("outputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("messages"), list):
        attributes["gen_ai.input.messages"] = inputs["messages"]
    if isinstance(outputs, dict) and isinstance(outputs.get("messages"), list):
        attributes["gen_ai.output.messages"] = outputs["messages"]
    status = str(record.get("status") or "").lower()
    extra = (
        {"status": {"code": 2, "message": str(record.get("error") or "")}}
        if status == "error" or record.get("error")
        else None
    )
    span = _span(
        record.get("trace_id") or record["id"],
        record["id"],
        record.get("parent_run_id"),
        record.get("name"),
        record.get("start_time"),
        record.get("end_time"),
        attributes,
        namespace="langsmith",
        extra=extra,
    )
    return [span] if span else []


def _phoenix(record: dict[str, Any]) -> list[dict[str, Any]]:
    context = record.get("context")
    if not isinstance(context, dict) or not context.get("trace_id") or not context.get("span_id"):
        raise ValueError("Phoenix span requires context.trace_id and context.span_id")
    attributes = record.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise ValueError("Phoenix span attributes must be an object")
    attributes = {**attributes, "bandits.native.record": record}
    resource = record.get("resource") or {}
    scope = record.get("instrumentation_scope") or record.get("instrumentationScope") or {}
    status = record.get("status")
    if isinstance(status, dict) and str(
        status.get("status_code") or status.get("code") or ""
    ).upper() in ("ERROR", "STATUS_CODE_ERROR"):
        status = {"code": 2, "message": status.get("message")}
    extra = {
        "events": record.get("events") or [],
        "links": record.get("links") or [],
        "status": status,
    }
    span = _span(
        context["trace_id"],
        context["span_id"],
        record.get("parent_id"),
        record.get("name"),
        record.get("start_time"),
        record.get("end_time"),
        attributes,
        namespace="phoenix",
        extra=extra,
    )
    if span is not None:
        if isinstance(resource, dict):
            span["_native_resource"] = resource.get("attributes") or {}
        if isinstance(scope, dict):
            span["_native_scope"] = scope
    return [span] if span else []


_CONVERTERS = {"langfuse": _langfuse, "langsmith": _langsmith, "phoenix": _phoenix}


def _langsmith_runs(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Flatten documented child_runs without turning nesting into time order."""
    stack = [(record, None, record.get("trace_id") or record.get("id"))]
    seen: set[str] = set()
    while stack:
        run, enclosing, trace_id = stack.pop()
        if not isinstance(run, dict):
            continue
        native_id = str(run.get("id"))
        if native_id in seen:
            continue
        seen.add(native_id)
        if enclosing and not run.get("parent_run_id"):
            run = {**run, "parent_run_id": enclosing}
        if trace_id and not run.get("trace_id"):
            run = {**run, "trace_id": trace_id}
        yield run
        for child in reversed(run.get("child_runs") or []):
            stack.append((child, run.get("id"), trace_id))


def load_native(
    path: Path,
    source_name: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    workflow: WorkflowDeclaration | None = None,
    pipeline_steps: bool = True,
) -> TraceCorpus:
    """Import one native JSON/JSONL file without an external conversion script."""
    if source_name not in _CONVERTERS:
        raise ValueError(f"unknown native source {source_name!r}")
    issues: list[TraceIssue] = []
    source_hash = hashlib.sha256()

    def source_records() -> Iterator[dict[str, Any]]:
        with path.open("rb") as stream:
            first = stream.readline()
            # A complete first-line object signals JSONL. Keep only one trace
            # in memory even when the export is many gigabytes long.
            if first.lstrip().startswith(b"{") and first.rstrip().endswith(b"}"):
                for number, line in enumerate(chain((first,), stream), start=1):
                    source_hash.update(line)
                    if not line.strip():
                        continue
                    safe = redact_bytes(line, f"{path}:record{number}", ruleset)
                    issues.extend(safe.issues)
                    yield from _records(safe.data)
            else:
                original = first + stream.read()
                source_hash.update(original)
                safe = redact_bytes(original, str(path), ruleset)
                issues.extend(safe.issues)
                yield from _records(safe.data)

    with tempfile.TemporaryDirectory(prefix="bandits-native-") as temporary:
        converted = Path(temporary) / "converted.jsonl"
        converted_traces = []
        converted_issues: list[TraceIssue] = []
        with converted.open("w+", encoding="utf-8") as output:
            pending = 0

            def flush() -> None:
                nonlocal pending
                if not pending:
                    return
                output.flush()
                chunk = load_otlp_standard(
                    converted, ruleset, pipeline_steps=pipeline_steps, workflow=workflow
                )
                converted_traces.extend(chunk.traces)
                converted_issues.extend(chunk.issues)
                output.seek(0)
                output.truncate(0)
                pending = 0

            for index, outer in enumerate(source_records(), start=1):
                records = (
                    outer.get("runs")
                    if source_name == "langsmith"
                    else outer.get("spans")
                    if source_name == "phoenix"
                    else None
                )
                if not isinstance(records, list):
                    records = [outer]
                expanded = (
                    (
                        run
                        for record in records
                        if isinstance(record, dict)
                        for run in _langsmith_runs(record)
                    )
                    if source_name == "langsmith"
                    else iter(records)
                )
                for record in expanded:
                    try:
                        if not isinstance(record, dict):
                            raise ValueError("expected an object")
                        spans = _CONVERTERS[source_name](record)
                    except (TypeError, ValueError) as exc:
                        issues.append(
                            TraceIssue(
                                kind="unsupported_native_record",
                                detail=str(exc),
                                location=f"{path}:{index}",
                            )
                        )
                        continue
                    if not spans:
                        issues.append(
                            TraceIssue(
                                kind="unrepresented_record",
                                detail="no span with valid id and timestamps",
                                location=f"{path}:{index}",
                            )
                        )
                        continue
                    for span in spans:
                        native_resource = span.pop("_native_resource", {})
                        native_scope = span.pop("_native_scope", {"name": source_name})
                        if not isinstance(native_scope, dict):
                            native_scope = {"name": source_name}
                        if not isinstance(native_resource, dict):
                            native_resource = {}
                        scope = {k: v for k, v in native_scope.items() if k != "attributes"}
                        scope["attributes"] = (
                            _attrs(native_scope.get("attributes") or {})
                            if isinstance(native_scope.get("attributes"), dict)
                            else native_scope.get("attributes") or []
                        )
                        request = {
                            "resourceSpans": [
                                {
                                    "resource": {"attributes": _attrs(native_resource)},
                                    "scopeSpans": [{"scope": scope, "spans": [span]}],
                                }
                            ]
                        }
                        output.write(json.dumps(request, ensure_ascii=False) + "\n")
                        pending += 1
                if source_name == "langfuse" and pending >= 400:
                    flush()
            flush()
    return TraceCorpus(
        source=source_name,
        traces=tuple(
            trace.replace(source=source_name, source_digest=source_hash.hexdigest())
            for trace in converted_traces
        ),
        issues=tuple(issues) + tuple(converted_issues),
        workflow=workflow,
        redaction_ruleset=ruleset.name,
    )
