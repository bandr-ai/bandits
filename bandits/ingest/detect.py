"""Conservative, deterministic recognition of supported trace file schemas.

Detection reads structure only. It never uses model-role text to infer human
authorship or decides whether an application is a workflow. Unknown and mixed
inputs must be declared explicitly or mapped separately, not guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_SAMPLE = 512 * 1024 * 1024
"""A single JSON document up to this size is parsed whole to recognize it; its
reader loads it whole anyway. JSONL is sampled line by line."""


class DetectionError(ValueError):
    """No unique supported source schema was established."""


@dataclass(frozen=True)
class Detection:
    source: str
    evidence: str
    files_checked: int


def _samples(path: Path) -> list[Any]:
    with path.open("rb") as stream:
        first = stream.readline(_MAX_SAMPLE + 1)
        if not first:
            raise DetectionError(f"empty file: {path}")
        if len(first) > _MAX_SAMPLE:
            raise DetectionError(
                f"first record exceeds {_MAX_SAMPLE} bytes: {path}; declare --source"
            )
        try:
            first_record = json.loads(first)
        except json.JSONDecodeError:
            rest = stream.read(_MAX_SAMPLE - len(first) + 1)
            if len(first) + len(rest) > _MAX_SAMPLE:
                raise DetectionError(
                    f"JSON sample exceeds {_MAX_SAMPLE} bytes: {path}; declare --source"
                ) from None
            try:
                return [json.loads(first + rest)]
            except json.JSONDecodeError as exc:
                raise DetectionError(f"cannot parse first JSON record in {path}: {exc}") from exc
        samples = [first_record]
        for line in stream:
            if not line.strip():
                continue
            if len(samples) == 10:
                break
            if len(line) > _MAX_SAMPLE:
                raise DetectionError(
                    f"sample record exceeds {_MAX_SAMPLE} bytes: {path}; declare --source"
                )
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DetectionError(f"malformed JSONL in {path}: {exc}; declare --source") from exc
        return samples


def _shape(record: Any) -> tuple[str, str] | None:
    if isinstance(record, list):
        if record and all(
            isinstance(item, dict) and isinstance(item.get("role"), str) for item in record
        ):
            return "chat-json", "top-level role/content messages"
        if record and isinstance(record[0], dict):
            return _shape(record[0])
        return None
    if not isinstance(record, dict):
        return None
    if isinstance(record.get("resourceSpans"), list) or (
        isinstance(record.get("batches"), list)
        and record["batches"]
        and all(
            isinstance(item, dict)
            and ("scopeSpans" in item or "instrumentationLibrarySpans" in item)
            for item in record["batches"]
        )
    ):
        return "otlp-std", "OTLP resourceSpans/batches"
    events = record.get("events")
    if (
        isinstance(record.get("schema_version"), str)
        and record.get("session_id")
        and isinstance(events, list)
        and events
        and all(
            isinstance(e, dict)
            and isinstance(e.get("event_type"), str)
            and isinstance(e.get("payload"), dict)
            for e in events[:10]
        )
    ):
        return "failproofai", "FailproofAI session transcript with typed events"
    if isinstance(record.get("observations"), list) and (
        record.get("trace_id")
        or record.get("id")
        or (isinstance(record.get("trace"), dict) and record["trace"].get("id"))
    ):
        return "langfuse", "Langfuse trace with observations"
    context = record.get("context")
    if isinstance(context, dict) and context.get("trace_id") and context.get("span_id"):
        return "phoenix", "Phoenix span context IDs"
    runs = record.get("runs")
    if isinstance(runs, list) and runs and isinstance(runs[0], dict) and runs[0].get("run_type"):
        return "langsmith", "LangSmith runs with run_type"
    spans = record.get("spans")
    if isinstance(spans, list) and spans and isinstance(spans[0], dict):
        if isinstance(spans[0].get("context"), dict) and spans[0]["context"].get("span_id"):
            return "phoenix", "Phoenix spans with context IDs"
        if "span_attributes" in spans[0] and "span_id" in spans[0]:
            return "trail", "TRAIL nested span tree"
    data = record.get("data")
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and isinstance(data[0].get("context"), dict)
        and data[0]["context"].get("trace_id")
        and data[0]["context"].get("span_id")
    ):
        return "phoenix", "Phoenix getSpans data[] with context IDs"
    if (record.get("id") or record.get("run_id")) and record.get("run_type"):
        return "langsmith", "LangSmith Run id or run_id with run_type"
    if (
        record.get("sessionId")
        and record.get("type") in ("user", "assistant")
        and isinstance(record.get("message"), dict)
    ):
        return "claude-code", "Claude Code session event"
    if (
        record.get("trace_id")
        and record.get("span_id")
        and isinstance(record.get("attributes"), dict)
    ):
        return "otlp", "legacy flat OTLP span"
    if (
        isinstance(record.get("messages"), list)
        and record["messages"]
        and all(
            isinstance(item, dict) and isinstance(item.get("role"), str)
            for item in record["messages"]
        )
    ):
        return "chat-json", "conversation messages with roles"
    return None


def detect_source(path: Path) -> Detection:
    """Recognize only unambiguous supported JSON shapes; never call an LLM."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")
        )
    else:
        raise DetectionError(f"source path does not exist: {path}")
    if not files:
        raise DetectionError(f"no JSON or JSONL files under {path}")
    found: dict[str, list[str]] = {}
    for file in files:
        for record in _samples(file):
            result = _shape(record)
            if result is None:
                raise DetectionError(
                    f"unrecognized trace structure in {file}; declare --source if known, "
                    "otherwise a new reader is required"
                )
            source, evidence = result
            found.setdefault(source, []).append(evidence)
    if len(found) != 1:
        raise DetectionError(
            f"mixed source schemas under {path}: {', '.join(sorted(found))}; ingest separately"
        )
    source, evidence = next(iter(found.items()))
    return Detection(source, evidence[0], len(files))
