"""OTLP adapter.

The native format: a trace is already a flat stream of spans, each declaring its
own trace id, span id, optional parent span id, and start/end time. Nothing here
reconstructs anything the export didn't already say directly.

Expected record shape, one JSON object per line::

    {
      "trace_id": "trace-1",
      "span_id": "span-1",
      "parent_span_id": null,
      "name": "gpt-5",
      "start_time": "2026-01-01T00:00:00Z",
      "end_time": "2026-01-01T00:00:01Z",
      "attributes": {
        "gen_ai.operation.name": "chat",
        "task": "Fix the failing test in parser.py"
      }
    }

``attributes["gen_ai.operation.name"]`` decides the span kind: ``"chat"`` is a
MODEL span, ``"execute_tool"`` is a TOOL span. Anything else is a
:class:`~bandits.traces.TraceIssue`, not a guess.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from bandits.ingest.toolsets import parse_toolset
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue

_LINEAGE_KEYS = (
    "gen_ai.conversation.id",
    "session.id",
    "session_id",
    "conversation.id",
    "conversation_id",
    "thread.id",
    "thread_id",
)
"""Attribute names carrying a session grouping, in decreasing order of standardness."""

_TOOLSET_KEYS = ("gen_ai.request.tools", "gen_ai.request.functions", "llm.request.functions")
"""Attribute names carrying the toolset offered to the model, across exporters."""

_SYSTEM_PROMPT_KEYS = ("gen_ai.system_instructions", "gen_ai.request.system", "system_prompt")

_CONTEXT_KEYS = (
    "gen_ai.request.model",
    "gen_ai.request.temperature",
    "gen_ai.request.top_p",
    "gen_ai.request.max_tokens",
    "gen_ai.system",
)
"""Settings the run used. Recorded because a demonstration is only reproducible
against the configuration that produced it."""

_OPERATION_TO_KIND = {
    "chat": SpanKind.MODEL,
    "execute_tool": SpanKind.TOOL,
}


class OtlpFormatError(ValueError):
    """The file itself is not readable as OTLP JSONL. Raised, not swallowed."""


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _first(attributes: dict, keys: tuple[str, ...]) -> object | None:
    return next((attributes[key] for key in keys if attributes.get(key) is not None), None)


def _parse_record(record: dict, *, location: str) -> tuple[Span, str, str | None, str | None]:
    """Returns ``(span, trace_id, task, lineage_id)`` for a valid record.

    Raises :class:`_RecordError` rather than returning a partial span when the
    record cannot be normalized."""
    trace_id = record.get("trace_id")
    span_id = record.get("span_id")
    name = record.get("name")
    attributes = record.get("attributes") or {}
    operation = attributes.get("gen_ai.operation.name")
    kind = _OPERATION_TO_KIND.get(operation)
    started_at = _parse_timestamp(record.get("start_time"))
    ended_at = _parse_timestamp(record.get("end_time"))

    missing = [
        field
        for field, value in (
            ("trace_id", trace_id),
            ("span_id", span_id),
            ("name", name),
        )
        if not isinstance(value, str) or not value
    ]
    if missing:
        raise _RecordError(f"missing or invalid field(s): {', '.join(missing)}", location)
    if kind is None:
        raise _RecordError(
            f"unrecognized gen_ai.operation.name {operation!r}; expected 'chat' or 'execute_tool'",
            location,
        )
    if started_at is None or ended_at is None:
        raise _RecordError("start_time/end_time must be ISO-8601 timestamps", location)

    if kind is SpanKind.TOOL:
        arguments = attributes.get("gen_ai.tool.call.arguments") or {}
        output = attributes.get("gen_ai.tool.call.result")
    else:
        arguments = attributes.get("gen_ai.request.arguments") or {}
        output = attributes.get("gen_ai.completion")

    status = SpanStatus.ERROR if attributes.get("status") == "error" else SpanStatus.OK
    parent_span_id = record.get("parent_span_id") or None
    task = attributes.get("task") if parent_span_id is None else None
    lineage_id = next(
        (
            attributes[key]
            for key in _LINEAGE_KEYS
            if isinstance(attributes.get(key), str) and attributes[key]
        ),
        None,
    )

    span = Span(
        span_id=span_id,
        parent_span_id=parent_span_id,
        kind=kind,
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        arguments=arguments if isinstance(arguments, dict) else {},
        output=output,
        attributes=attributes,
    )
    return span, trace_id, task, lineage_id


def _declared_context(spans: tuple[Span, ...]) -> tuple[object, object, dict]:
    """The toolset, system prompt and settings declared on the episode's root span.

    Read from the root only. A tool listed on a later span is the toolset as it
    stood by then, which is not the same claim as what the episode was offered at
    the start, and treating the two as one would hand a new attempt a toolset
    that had already been narrowed by what the agent learned.
    """
    root = next((span for span in spans if span.parent_span_id is None), None)
    if root is None:
        return None, None, {}
    attributes = root.attributes
    system_prompt = _first(attributes, _SYSTEM_PROMPT_KEYS)
    return (
        parse_toolset(_first(attributes, _TOOLSET_KEYS)),
        system_prompt if isinstance(system_prompt, str) else None,
        {key: attributes[key] for key in _CONTEXT_KEYS if attributes.get(key) is not None},
    )


class _RecordError(Exception):
    def __init__(self, detail: str, location: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.location = location


def load_otlp(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> TraceCorpus:
    """Read one OTLP JSONL export into a :class:`TraceCorpus`."""
    source = redact_source(path, ruleset)
    raw = source.data
    source_digest = source.source_digest

    spans_by_trace: dict[str, list[tuple[int, Span]]] = {}
    task_by_trace: dict[str, str] = {}
    lineage_by_trace: dict[str, str] = {}
    issues: list[TraceIssue] = list(source.issues)

    for index, line in enumerate(raw.split(b"\n")):
        if not line.strip():
            continue
        location = f"{path}:{index + 1}"
        try:
            record = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            issues.append(TraceIssue(kind="malformed_json", detail=str(exc), location=location))
            continue
        if not isinstance(record, dict):
            issues.append(
                TraceIssue(
                    kind="malformed_record",
                    detail=f"expected a JSON object, got {type(record).__name__}",
                    location=location,
                )
            )
            continue
        try:
            span, trace_id, task, lineage_id = _parse_record(record, location=location)
        except _RecordError as exc:
            issues.append(
                TraceIssue(kind="malformed_span", detail=exc.detail, location=exc.location)
            )
            continue
        spans_by_trace.setdefault(trace_id, []).append((index, span))
        if task is not None:
            task_by_trace[trace_id] = task
        if lineage_id is not None:
            lineage_by_trace.setdefault(trace_id, lineage_id)

    traces = []
    for trace_id, collected in sorted(spans_by_trace.items()):
        # Source order breaks ties, not span_id: exporters often stamp a whole
        # episode with one coarse timestamp, and sorting those lexicographically
        # puts 's10' before 's2' and picks the wrong terminal span.
        ordered = tuple(
            span for _, span in sorted(collected, key=lambda pair: (pair[1].started_at, pair[0]))
        )
        tools, system_prompt, context = _declared_context(ordered)
        traces.append(
            Trace(
                trace_id=trace_id,
                source="otlp",
                source_digest=source_digest,
                task=task_by_trace.get(trace_id),
                lineage_id=lineage_by_trace.get(trace_id),
                tools_available=tools,  # type: ignore[arg-type]
                system_prompt=system_prompt,
                runtime_context=context,
                spans=ordered,
            )
        )
    return TraceCorpus(
        source="otlp",
        traces=tuple(traces),
        issues=tuple(issues),
        redaction_ruleset=source.ruleset,
    )
