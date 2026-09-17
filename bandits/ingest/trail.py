"""TRAIL adapter: smolagents OpenInference span trees, one JSON file per trace.

TRAIL (Patronus AI, arXiv:2505.08638) records GAIA and SWE-Bench runs of a
smolagents ``CodeAgent`` as a nested span tree. The shape that matters here:

- An ``AGENT`` span opens a run and declares the task in ``input.value``.
- Each ``CHAIN`` span named ``Step N`` is one agent step. Its child ``LLM``
  span is the model call that decided the step, and the step's own
  ``output.value`` is the execution log of the code that call wrote — the
  environment's reaction to the action.
- ``TOOL`` spans are tool calls the managed search agent made inside a step.

The step's execution log is emitted as a TOOL span named ``execute`` placed
*after* the step's children, so a trace reads action → tool calls → reaction in
the order it happened. Span ids are kept verbatim: TRAIL's human annotations
locate every error by span id, and a renamed id would sever the trace from the
only ground truth it has.

Nothing is inferred as a user turn. The task is the literal text handed to the
agent, and it is recorded as ``task`` only.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue

SOURCE = "trail"

_KIND_KEY = "openinference.span.kind"
_DURATION = re.compile(r"^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")

_MAX_TEXT = 4000
"""Per-field cap on recorded text. Execution logs and page dumps run to tens of
thousands of characters; a turn reader needs the head, and the source file is
still there for anything else."""


def _parse_duration(value: object) -> timedelta:
    if isinstance(value, (int, float)):
        return timedelta(seconds=float(value))
    match = _DURATION.match(str(value or ""))
    if not match:
        return timedelta(0)
    days, hours, minutes, seconds = match.groups()
    return timedelta(
        days=int(days or 0),
        hours=int(hours or 0),
        minutes=int(minutes or 0),
        seconds=float(seconds or 0),
    )


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clip(value: object, limit: int = _MAX_TEXT) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except (TypeError, ValueError):
            value = str(value)
    if len(value) > limit:
        return value[:limit] + "…[truncated]"
    return value


def _json(value: object) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _model_output(attributes: dict[str, Any]) -> Any:
    content = attributes.get("llm.output_messages.0.message.content")
    calls: list[dict[str, Any]] = []
    index = 0
    while True:
        prefix = f"llm.output_messages.0.message.tool_calls.{index}.tool_call.function."
        name = attributes.get(prefix + "name")
        if name is None:
            break
        calls.append({"name": name, "arguments": _json(attributes.get(prefix + "arguments"))})
        index += 1
    if calls:
        return {"content": _clip(content), "tool_calls": calls}
    if content is None:
        output = attributes.get("output.value")
        parsed = _json(output)
        if isinstance(parsed, dict) and "content" in parsed:
            return _clip(parsed.get("content"))
        return _clip(output)
    return _clip(content)


def _last_input(attributes: dict[str, Any]) -> dict[str, Any]:
    """The last message the model was shown: the observation it acted on."""
    index = 0
    last: str | None = None
    while (text := attributes.get(f"llm.input_messages.{index}.message.content")) is not None:
        last = text
        index += 1
    return {"observation": _clip(last, 1500)} if last is not None else {}


def _tool_arguments(attributes: dict[str, Any]) -> dict[str, Any]:
    parsed = _json(attributes.get("input.value"))
    if isinstance(parsed, dict):
        kwargs = parsed.get("kwargs")
        args = parsed.get("args")
        if isinstance(kwargs, dict) and kwargs:
            return {k: _clip(v, 1000) for k, v in kwargs.items()}
        if isinstance(args, list) and args:
            return {"args": [_clip(a, 1000) for a in args]}
        return {k: _clip(v, 1000) for k, v in parsed.items() if k != "sanitize_inputs_outputs"}
    return {"input": _clip(parsed, 1000)} if parsed is not None else {}


def _span(
    raw: dict[str, Any],
    *,
    kind: SpanKind,
    name: str,
    arguments: dict[str, Any],
    output: Any,
    span_id: str | None = None,
    parent: str | None = None,
    started: datetime | None = None,
) -> Span:
    started_at = started or _timestamp(raw.get("timestamp")) or datetime.fromtimestamp(0)
    ended_at = started_at + _parse_duration(raw.get("duration"))
    status = SpanStatus.ERROR if raw.get("status_code") == "Error" else SpanStatus.OK
    return Span(
        span_id=span_id or str(raw["span_id"]),
        parent_span_id=parent if parent is not None else raw.get("parent_span_id"),
        kind=kind,
        name=name,
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        arguments=arguments,
        output=output,
        attributes={"trail.span_name": raw.get("span_name")},
    )


def _walk(
    raw: dict[str, Any], spans: list[Span], task: list[str], issues: list[TraceIssue]
) -> None:
    attributes = raw.get("span_attributes") or {}
    kind = attributes.get(_KIND_KEY)
    children = sorted(raw.get("child_spans") or [], key=lambda c: str(c.get("timestamp", "")))

    if kind == "AGENT" and not task:
        parsed = _json(attributes.get("input.value"))
        if isinstance(parsed, dict) and parsed.get("task"):
            task.append(str(parsed["task"]))

    if kind == "LLM":
        spans.append(
            _span(
                raw,
                kind=SpanKind.MODEL,
                name=str(attributes.get("llm.model_name") or raw.get("span_name") or "model"),
                arguments=_last_input(attributes),
                output=_model_output(attributes),
            )
        )
    elif kind == "TOOL":
        spans.append(
            _span(
                raw,
                kind=SpanKind.TOOL,
                name=str(attributes.get("tool.name") or raw.get("span_name") or "tool"),
                arguments=_tool_arguments(attributes),
                output=_clip(attributes.get("output.value")),
            )
        )

    for child in children:
        _walk(child, spans, task, issues)

    if kind == "CHAIN":
        # The step's execution log is the reaction to everything its children
        # did, so it goes after them. Its id is the step's own: an annotation
        # placed on the step lands on the reaction, which is where it belongs.
        output = attributes.get("output.value")
        if output is None and raw.get("status_code") == "Error":
            output = raw.get("status_message")
        started = _timestamp(raw.get("timestamp"))
        if started is not None:
            started = started + _parse_duration(raw.get("duration"))
        spans.append(
            _span(
                raw,
                kind=SpanKind.TOOL,
                name="execute",
                arguments={"step": str(raw.get("span_name") or "")},
                output=_clip(output),
                started=started,
            )
        )


def _load_file(path: Path, ruleset: RedactionRuleset) -> tuple[Trace | None, list[TraceIssue], str]:
    source = redact_source(path, ruleset)
    issues = list(source.issues)
    try:
        payload = json.loads(source.data.decode("utf-8", "replace"))
    except ValueError as exc:
        issues.append(TraceIssue(kind="malformed_json", detail=str(exc), location=str(path)))
        return None, issues, source.ruleset

    if not isinstance(payload, dict):
        issues.append(
            TraceIssue(
                kind="malformed_record",
                detail=f"trace file is a {type(payload).__name__}, not an object",
                location=str(path),
            )
        )
        return None, issues, source.ruleset

    spans: list[Span] = []
    task: list[str] = []
    try:
        for root in payload.get("spans") or []:
            _walk(root, spans, task, issues)
    except (AttributeError, KeyError, TypeError) as exc:
        # A malformed span tree -- a non-object root or child, a span with no
        # `span_id` -- must quarantine this one file, not abort every other
        # file the directory ingest was about to read.
        issues.append(
            TraceIssue(
                kind="malformed_span",
                detail=f"{type(exc).__name__}: {exc}",
                location=str(path),
            )
        )
        return None, issues, source.ruleset
    if not spans:
        issues.append(
            TraceIssue(kind="empty_trace", detail="no LLM, TOOL or step spans", location=str(path))
        )
        return None, issues, source.ruleset

    trace = Trace(
        trace_id=str(payload.get("trace_id") or path.stem),
        source=SOURCE,
        source_digest=source.source_digest,
        task=task[0] if task else None,
        runtime_context={"framework": "smolagents", "benchmark": "trail"},
        spans=tuple(spans),
    )
    return trace, issues, source.ruleset


def load_trail(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> TraceCorpus:
    """Read one TRAIL trace file, or a directory of them, into a corpus."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    traces: list[Trace] = []
    issues: list[TraceIssue] = []
    ruleset_name = ruleset.name
    for file in files:
        trace, file_issues, ruleset_name = _load_file(file, ruleset)
        issues.extend(file_issues)
        if trace is not None:
            traces.append(trace)
    return TraceCorpus(
        source=SOURCE,
        traces=tuple(traces),
        issues=tuple(issues),
        redaction_ruleset=ruleset_name,
    )
