"""OTLP adapter: OpenTelemetry GenAI spans -> :class:`Trace`.

This is the good case. An OTLP export that follows the GenAI semantic conventions
already records invocation points explicitly: ``execute_tool`` spans carry the tool
name, the call id, the serialized arguments and the serialized response, and the
span status says whether it failed. That is exactly the record that makes an
environment reconstructible (PLAN.md, "The key concept: invocation points"), so no
recovery heuristics are needed here -- contrast :mod:`bandits.ingest.chat_json`.

Two encodings of span attributes exist in the wild and both are supported:

* the plain-dict form ``{"gen_ai.tool.name": "get_order"}`` (what collectors that
  speak JSON directly emit, and what our fixture uses), and
* the verbose protobuf-JSON form
  ``[{"key": "gen_ai.tool.name", "value": {"stringValue": "get_order"}}]``.

They must parse identically; :func:`normalize_attributes` is the single place that
difference is resolved.
"""

from __future__ import annotations

import json
from typing import Any

from bandits.contracts import (
    CallStatus,
    IngestIssue,
    InvocationPoint,
    JsonObject,
    JsonValue,
    Message,
    Trace,
)
from bandits.ingest.errors import infer_error_kind

#: GenAI semantic-convention attribute names we read.
ATTR_OPERATION = "gen_ai.operation.name"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_TOOL_MESSAGE = "gen_ai.tool.message"
ATTR_PROMPT = "gen_ai.prompt"
ATTR_COMPLETION = "gen_ai.completion"

#: bandits-specific attributes carried by the fixture corpus.
ATTR_EPISODE_ID = "bandits.episode_id"
ATTR_OUTCOME = "bandits.outcome"

#: ``gen_ai.operation.name`` value that marks a tool invocation span.
OP_EXECUTE_TOOL = "execute_tool"
#: ``gen_ai.operation.name`` value that marks the model call carrying the transcript.
OP_CHAT = "chat"

#: OTLP ``status.code``: 0 UNSET, 1 OK, 2 ERROR.
STATUS_CODE_ERROR = 2

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


# --------------------------------------------------------------------------
# attribute decoding
# --------------------------------------------------------------------------


def _decode_any_value(value: Any) -> JsonValue:
    """Unwrap one protobuf-JSON ``AnyValue`` into a plain Python value.

    The verbose OTLP encoding boxes every attribute in a one-key type tag. We unwrap
    the tag rather than keeping it, so that downstream code -- and the equivalence
    test against chat-json -- sees the same Python objects either way. Integers
    arrive as strings in protobuf JSON, hence the explicit ``intValue`` cast.
    """
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        raw = value["intValue"]
        return int(raw) if isinstance(raw, str) else raw
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        values = value["arrayValue"].get("values", []) if isinstance(value["arrayValue"], dict) else []
        return [_decode_any_value(v) for v in values]
    if "kvlistValue" in value:
        pairs = value["kvlistValue"].get("values", []) if isinstance(value["kvlistValue"], dict) else []
        return {p.get("key"): _decode_any_value(p.get("value")) for p in pairs if isinstance(p, dict)}
    return value


def normalize_attributes(attributes: Any) -> JsonObject:
    """Return span attributes as a flat ``name -> value`` dict, whichever form they use.

    Supporting both encodings in one function is the point: every other function in
    this module can then assume the plain-dict form, and the "verbose parses
    identically to plain" guarantee has exactly one implementation to hold up.
    """
    if attributes is None:
        return {}
    if isinstance(attributes, dict):
        return dict(attributes)
    if isinstance(attributes, list):
        out: JsonObject = {}
        for item in attributes:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not isinstance(key, str):
                continue
            out[key] = _decode_any_value(item.get("value"))
        return out
    return {}


def _json_attr(attributes: JsonObject, key: str) -> JsonValue:
    """Decode an attribute that holds a JSON *string*, e.g. arguments or a response.

    GenAI conventions serialize structured payloads into string attributes. If the
    exporter already put a structure there, we take it as-is; if the string is not
    valid JSON we return it verbatim rather than dropping the payload, and the
    caller decides whether that is an issue.
    """
    raw = attributes.get(key)
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


# --------------------------------------------------------------------------
# span walking
# --------------------------------------------------------------------------


def iter_spans(record: JsonObject) -> list[JsonObject]:
    """Flatten ``resourceSpans[].scopeSpans[].spans[]`` into one list, in file order.

    The nesting carries resource/scope metadata we do not need for reconstruction;
    order across the nesting is preserved so that spans without usable timestamps
    still have a deterministic fallback ordering.
    """
    spans: list[JsonObject] = []
    for resource in record.get("resourceSpans") or []:
        if not isinstance(resource, dict):
            continue
        for scope in resource.get("scopeSpans") or resource.get("instrumentationLibrarySpans") or []:
            if not isinstance(scope, dict):
                continue
            for span in scope.get("spans") or []:
                if isinstance(span, dict):
                    spans.append(span)
    return spans


def _span_start(span: JsonObject) -> int:
    """Span start time in nanoseconds, 0 when absent.

    Used only for ordering. OTLP writes these as decimal strings; a span missing a
    start time sorts first and keeps its file position via the stable sort.
    """
    raw = span.get("startTimeUnixNano")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _latency_ms(span: JsonObject) -> float | None:
    """Wall-clock duration of a span in milliseconds, or ``None`` if not derivable."""
    start, end = span.get("startTimeUnixNano"), span.get("endTimeUnixNano")
    try:
        return (int(end) - int(start)) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _span_status(span: JsonObject) -> CallStatus:
    """Map ``status.code`` to a :class:`CallStatus`.

    On OTLP the span status is authoritative -- it is the producer's own statement
    that the call failed -- so we never second-guess it from the response body.
    """
    status = span.get("status")
    code = status.get("code") if isinstance(status, dict) else None
    if code == STATUS_CODE_ERROR or (isinstance(code, str) and code.upper().endswith("ERROR")):
        return CallStatus.ERROR
    return CallStatus.OK


def _coerce_outcome(value: JsonValue) -> bool | None:
    """Read the ground-truth label, accepting bool or the string forms exports use.

    ``None`` means unlabeled, which is a meaningful state (PLAN.md Step 8) and must
    not collapse into ``False``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "pass", "passed", "success", "1"):
            return True
        if lowered in ("false", "fail", "failed", "failure", "0"):
            return False
    return None


# --------------------------------------------------------------------------
# message + invocation extraction
# --------------------------------------------------------------------------


def _messages_from_payload(payload: JsonValue, *, issues: list[IngestIssue], location: str) -> list[Message]:
    """Turn a decoded ``gen_ai.prompt`` / ``gen_ai.completion`` payload into messages.

    The transcript is what carries the task statement and the final answer, which is
    what stage 5 mines tasks from. A turn with an unrecognized role becomes an issue
    rather than being coerced into a role it does not have.
    """
    if payload is None:
        return []
    turns = payload if isinstance(payload, list) else [payload]
    messages: list[Message] = []
    for index, turn in enumerate(turns):
        if isinstance(turn, str):
            messages.append(Message(role="user", content=turn))
            continue
        if not isinstance(turn, dict):
            issues.append(IngestIssue(
                kind="malformed_message",
                detail=f"expected an object, got {type(turn).__name__}",
                location=f"{location}[{index}]",
            ))
            continue
        role = turn.get("role")
        if role not in VALID_ROLES:
            issues.append(IngestIssue(
                kind="unknown_message_role",
                detail=f"role {role!r} is not one of {sorted(VALID_ROLES)}",
                location=f"{location}[{index}]",
            ))
            continue
        content = turn.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, sort_keys=True)
        messages.append(Message(
            role=role,
            content=content,
            tool_call_id=turn.get("tool_call_id"),
        ))
    return messages


def _invocation_from_span(
    span: JsonObject,
    attributes: JsonObject,
    *,
    trace_id: str,
    step: int,
    issues: list[IngestIssue],
    location: str,
) -> InvocationPoint | None:
    """Build one :class:`InvocationPoint` from an ``execute_tool`` span.

    Returns ``None`` (with an issue recorded) only when the span has no tool name,
    which makes it unusable: an invocation point without a tool cannot be replayed
    and must not be invented.
    """
    tool = attributes.get(ATTR_TOOL_NAME)
    if not isinstance(tool, str) or not tool.strip():
        issues.append(IngestIssue(
            kind="tool_span_missing_name",
            detail=f"execute_tool span has no {ATTR_TOOL_NAME}",
            location=location,
        ))
        return None

    arguments = _json_attr(attributes, ATTR_TOOL_ARGUMENTS)
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        issues.append(IngestIssue(
            kind="malformed_arguments",
            detail=f"{ATTR_TOOL_ARGUMENTS} for {tool!r} decoded to {type(arguments).__name__}, not an object",
            location=location,
        ))
        arguments = {"_raw": arguments}

    response = _json_attr(attributes, ATTR_TOOL_MESSAGE)
    status = _span_status(span)
    error_kind = infer_error_kind(response) if status is CallStatus.ERROR else None

    call_id = attributes.get(ATTR_TOOL_CALL_ID)
    if not isinstance(call_id, str) or not call_id:
        call_id = f"{trace_id}:{step}"
        issues.append(IngestIssue(
            kind="synthesized_call_id",
            detail=f"span for {tool!r} declared no {ATTR_TOOL_CALL_ID}; synthesized {call_id!r}",
            location=location,
        ))

    span_id = span.get("spanId")
    return InvocationPoint(
        call_id=call_id,
        trace_id=trace_id,
        step=step,
        tool=tool,
        arguments=arguments,
        response=response,
        status=status,
        error_kind=error_kind,
        latency_ms=_latency_ms(span),
        source_span_id=span_id if isinstance(span_id, str) else None,
    )


def parse_otlp_record(
    record: JsonObject,
    *,
    source_digest: str,
    location: str,
    fallback_trace_id: str,
) -> tuple[Trace | None, list[IngestIssue]]:
    """Normalize one OTLP export record (one JSONL line) into a :class:`Trace`.

    One record is expected to hold one episode. The trace id is the declared
    ``bandits.episode_id`` when present, falling back to the OTLP ``traceId`` --
    the episode id is the identity the corpus is keyed on, and the raw ``traceId``
    is kept in metadata for provenance.
    """
    issues: list[IngestIssue] = []
    spans = iter_spans(record)
    if not spans:
        issues.append(IngestIssue(
            kind="no_spans",
            detail="record contains no resourceSpans[].scopeSpans[].spans[]",
            location=location,
        ))
        return None, issues

    decoded: list[tuple[JsonObject, JsonObject]] = [(s, normalize_attributes(s.get("attributes"))) for s in spans]

    otlp_trace_id: str | None = None
    episode_id: str | None = None
    outcome: bool | None = None
    messages: list[Message] = []

    for span, attributes in decoded:
        if otlp_trace_id is None and isinstance(span.get("traceId"), str):
            otlp_trace_id = span["traceId"]
        if episode_id is None and isinstance(attributes.get(ATTR_EPISODE_ID), str):
            episode_id = attributes[ATTR_EPISODE_ID]
        if outcome is None and ATTR_OUTCOME in attributes:
            outcome = _coerce_outcome(attributes[ATTR_OUTCOME])
        if attributes.get(ATTR_OPERATION) == OP_CHAT:
            messages.extend(_messages_from_payload(
                _json_attr(attributes, ATTR_PROMPT), issues=issues, location=f"{location}:prompt"))
            messages.extend(_messages_from_payload(
                _json_attr(attributes, ATTR_COMPLETION), issues=issues, location=f"{location}:completion"))

    trace_id = episode_id or otlp_trace_id or fallback_trace_id

    tool_spans = [(s, a) for s, a in decoded if a.get(ATTR_OPERATION) == OP_EXECUTE_TOOL]
    # Stable sort on start time: declared order wins ties, so spans that share a
    # timestamp keep their file position instead of being reshuffled per run.
    tool_spans.sort(key=lambda pair: _span_start(pair[0]))

    invocations: list[InvocationPoint] = []
    for step, (span, attributes) in enumerate(tool_spans):
        invocation = _invocation_from_span(
            span,
            attributes,
            trace_id=trace_id,
            step=step,
            issues=issues,
            location=f"{location}:span[{span.get('spanId')}]",
        )
        if invocation is not None:
            invocations.append(invocation)

    # A dropped span would leave a hole in the step sequence, so renumber densely.
    invocations = [inv.model_copy(update={"step": i}) for i, inv in enumerate(invocations)]

    trace = Trace(
        trace_id=trace_id,
        source="otlp",
        source_digest=source_digest,
        messages=tuple(messages),
        invocations=tuple(invocations),
        outcome=outcome,
        metadata={
            "otlp_trace_id": otlp_trace_id,
            "episode_id": episode_id,
            "span_count": len(spans),
            "location": location,
        },
    )
    return trace, issues


__all__ = [
    "ATTR_COMPLETION",
    "ATTR_EPISODE_ID",
    "ATTR_OUTCOME",
    "ATTR_PROMPT",
    "ATTR_TOOL_ARGUMENTS",
    "ATTR_TOOL_CALL_ID",
    "ATTR_TOOL_MESSAGE",
    "ATTR_TOOL_NAME",
    "OP_CHAT",
    "OP_EXECUTE_TOOL",
    "STATUS_CODE_ERROR",
    "iter_spans",
    "normalize_attributes",
    "parse_otlp_record",
]
