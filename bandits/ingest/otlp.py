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
from typing import Any

from bandits.ingest.toolsets import parse_toolset
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue, UserTurn

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

_TOOLSET_KEYS = (
    "gen_ai.tool.definitions",
    "gen_ai.request.tools",
    "gen_ai.request.functions",
    "llm.request.functions",
)
"""Attribute names carrying the toolset offered to the model, across exporters."""

_SYSTEM_PROMPT_KEYS = ("gen_ai.system_instructions", "gen_ai.request.system", "system_prompt")

_CONTEXT_KEYS = (
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.provider.name",
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


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _messages(value: object) -> list[dict[str, Any]]:
    """Read the OTel GenAI multipart message representation when present."""
    parsed = _json_value(value)
    if not isinstance(parsed, list):
        return []
    return [message for message in parsed if isinstance(message, dict)]


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("parts")
    return [part for part in raw if isinstance(part, dict)] if isinstance(raw, list) else []


def _message_text(message: dict[str, Any]) -> str | None:
    chunks = [
        part["content"]
        for part in _parts(message)
        if part.get("type") == "text" and isinstance(part.get("content"), str)
    ]
    if chunks:
        return "\n".join(chunks)
    content = message.get("content")
    return content if isinstance(content, str) else None


def _instruction_text(text: str) -> str:
    """Remove an explicitly delimited harness preamble from a user request.

    Some agent harnesses serialize policy/context and the actual task into one
    user part. The original remains in ``gen_ai.input.messages``; this returns
    only the source-declared task section used for grouping.
    """
    marker = "Task from supervisor:\n"
    if marker in text:
        task = text.rsplit(marker, 1)[1].strip()
        if task:
            return task
    return text


def _declared_task(attributes: dict[str, Any]) -> str | None:
    for message in _messages(attributes.get("gen_ai.input.messages")):
        if message.get("role") in ("user", "human"):
            text = _message_text(message)
            if text:
                return _instruction_text(text)
    return None


def _declared_completion(attributes: dict[str, Any]) -> object:
    messages = _messages(attributes.get("gen_ai.output.messages"))
    text = [
        rendered
        for message in messages
        if message.get("role") == "assistant"
        if (rendered := _message_text(message))
    ]
    if text:
        return "\n".join(text)
    return attributes.get("gen_ai.completion")


def _user_turns(spans: tuple[Span, ...]) -> tuple[tuple[UserTurn, ...], int]:
    """Recover user turns from successive GenAI input snapshots.

    Most exporters repeat the complete conversation in every model span. Only
    the suffix added since the preceding snapshot is new; treating every
    snapshot independently would duplicate the opening request once per model
    call. Exporters which record only the current call's input are also valid:
    when snapshots do not overlap, the whole later snapshot is new.

    A user-role message with no textual representation is counted rather than
    silently discarded. Its placement is anchored immediately before the
    model span that consumed it, after the preceding normalized span.
    """
    turns: list[UserTurn] = []
    unrepresented = 0
    previous: list[dict[str, Any]] = []
    preceding_span_id: str | None = None

    for span in spans:
        if span.kind is not SpanKind.MODEL:
            preceding_span_id = span.span_id
            continue

        current = _messages(span.attributes.get("gen_ai.input.messages"))
        overlap = 0
        for size in range(min(len(previous), len(current)), 0, -1):
            if previous[-size:] == current[:size]:
                overlap = size
                break

        for message in current[overlap:]:
            if message.get("role") not in ("user", "human"):
                continue
            content = _message_text(message)
            if content:
                turns.append(UserTurn(text=content, after_span_id=preceding_span_id))
            else:
                unrepresented += 1

        previous = current
        preceding_span_id = span.span_id

    return tuple(turns), unrepresented


def _tool_result(value: object) -> object:
    parsed = _json_value(value)
    return parsed


def _embedded_tool_spans(spans: tuple[Span, ...]) -> tuple[Span, ...]:
    """Recover tool executions recorded only inside cumulative GenAI messages.

    A response in the input of a later model call proves the tool returned, and
    its call id pairs it with the assistant tool-call part. OTel chat spans do
    not timestamp that execution separately, so its zero-width derived time is
    explicitly marked synthetic.
    """
    calls: dict[str, tuple[str, str, dict[str, Any]]] = {}
    emitted: set[str] = set()
    combined: list[Span] = []

    for span in spans:
        for message in _messages(span.attributes.get("gen_ai.input.messages")):
            for part in _parts(message):
                part_type = part.get("type")
                call_id = part.get("id") or part.get("tool_call_id")
                if part_type == "tool_call" and isinstance(call_id, str):
                    name = part.get("name")
                    arguments = _json_value(part.get("arguments"))
                    if isinstance(name, str) and name:
                        calls.setdefault(
                            call_id,
                            (
                                span.span_id,
                                name,
                                arguments if isinstance(arguments, dict) else {"raw": arguments},
                            ),
                        )
                elif part_type == "tool_call_response" and isinstance(call_id, str):
                    if call_id in emitted:
                        continue
                    call = calls.get(call_id)
                    name = part.get("name")
                    recorded = call is not None
                    if call is not None:
                        parent_id, tool_name, arguments = call
                    else:
                        parent_id, arguments = None, {}
                        tool_name = name if isinstance(name, str) and name else "tool"
                    combined.append(
                        Span(
                            span_id=f"{span.span_id}:tool:{call_id}",
                            parent_span_id=parent_id,
                            kind=SpanKind.TOOL,
                            name=tool_name,
                            started_at=span.started_at,
                            ended_at=span.started_at,
                            arguments=arguments,
                            output=_tool_result(part.get("result")),
                            call_recorded=recorded,
                            attributes={"synthetic_time": True, "source": "gen_ai.input.messages"},
                        )
                    )
                    emitted.add(call_id)

        combined.append(span)
        for message in _messages(span.attributes.get("gen_ai.output.messages")):
            for part in _parts(message):
                call_id = part.get("id") or part.get("tool_call_id")
                name = part.get("name")
                if (
                    part.get("type") == "tool_call"
                    and isinstance(call_id, str)
                    and isinstance(name, str)
                    and name
                ):
                    arguments = _json_value(part.get("arguments"))
                    calls.setdefault(
                        call_id,
                        (
                            span.span_id,
                            name,
                            arguments if isinstance(arguments, dict) else {"raw": arguments},
                        ),
                    )
    return tuple(combined)


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
        output = _declared_completion(attributes)

    recorded_status = record.get("status")
    status_code = recorded_status.get("code") if isinstance(recorded_status, dict) else None
    status = (
        SpanStatus.ERROR
        if attributes.get("status") == "error" or status_code in ("STATUS_CODE_ERROR", 2)
        else SpanStatus.OK
    )
    parent_span_id = record.get("parent_span_id") or None
    # A native root may carry the adapter's compact ``task`` field. A filtered
    # chat-only export can retain parent ids for wrapper spans it intentionally
    # omitted, so standard messages are readable on any chat span; the first
    # instruction observed for the trace is selected below.
    task = attributes.get("task") if parent_span_id is None else None
    task = task or _declared_task(attributes)
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
        # Chat-only datasets commonly remove their orchestration/wrapper spans
        # while retaining the chat spans' original parent ids. The first span
        # whose parent is outside the exported set is the observable entry.
        exported_ids = {span.span_id for span in spans}
        root = next((span for span in spans if span.parent_span_id not in exported_ids), None)
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
        ordered = _embedded_tool_spans(ordered)
        tools, system_prompt, context = _declared_context(ordered)
        user_turns, unrepresented_user_turns = _user_turns(ordered)
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
                user_turns=user_turns,
                unrepresented_user_turns=unrepresented_user_turns,
                spans=ordered,
            )
        )
    return TraceCorpus(
        source="otlp",
        traces=tuple(traces),
        issues=tuple(issues),
        redaction_ruleset=source.ruleset,
    )
