"""Readers for documented native trace exports, routed through the OTLP reader.

OpenInference is an OTLP convention, not a separate file format. Phoenix's
Span JSON and LangSmith's Run JSON are native formats; Langfuse's bundled
trace/observation JSON is another. This module converts their recorded fields
to OTLP internally. Each span carries ``bandits.source.record``, a pointer to
its native record in the redacted source archive (``ArtifactStore.read_native_record``),
rather than a copy of it.
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

from bandits.ingest.mapping import IngestMapping
from bandits.ingest.otlp_standard import TRACE_RECORD, iter_otlp_standard, load_otlp_standard
from bandits.ingest.report import EXAMPLES, IngestReport, aggregate_issues
from bandits.jsonarray import iter_array
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_bytes
from bandits.traces import Trace, TraceCorpus, TraceIssue, WorkflowDeclaration

Position = dict[str, Any]
"""Where a native record sits in its file: ``{"line": n}`` (1-based, physical),
plus ``"index"`` when that line holds a JSON array; ``{"index": i}`` in a
whole-file array; ``{"document": True}`` for a single JSON document."""

Skipped = list[tuple[str, str | None]]
"""``(reason, observation id)`` for each observation a converter could not
turn into a span. Reasons starting ``duplicate`` are duplicates; the rest are
unconvertible."""

_LINE_LIMIT = 64 << 20
"""Bytes read looking for the end of the first line; past this, a file
starting with ``[`` is read as one streamed array."""

_DUPLICATE = "duplicate id in this record; the first copy is kept"
_INSIDE_SKIPPED = "inside an observation that could not be converted"
_ON_TRACE = "kept on the trace: "
"""Prefix of a skip reason whose record is kept whole in the trace's
``source_record``: counted as ``kept_on_trace``, not as lost."""


class _Unconvertible(ValueError):
    """One observation that cannot become a span; its record is still read."""


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
    # An unrecorded value is an absent attribute. Encoding None would write the
    # string "null", which the decoder reads as a message saying "null". The
    # null itself stays visible in the source archive.
    return [{"key": k, "value": _av(v)} for k, v in values.items() if v is not None]


Used = dict[str, frozenset[str] | None]
"""The fields a converter consumes: ``None`` for a field consumed whole, or the
keys it reads inside an object (the rest of that object is not consumed)."""


def unmapped_fields(record: dict[str, Any], used: Used) -> dict[str, Any]:
    """What a converter does not consume of *record*, kept under original names.

    A field it never reads is kept whole (nulls and empty values too). Inside
    an object it reads only partly, the keys it does not read are kept. A
    partly read field that is not an object is kept whole: its shape was
    unexpected, so nothing in it was read.
    """
    unmapped: dict[str, Any] = {}
    for key, value in record.items():
        if key not in used:
            unmapped[key] = value
            continue
        inner = used[key]
        if inner is None:
            continue
        if not isinstance(value, dict):
            unmapped[key] = value
            continue
        rest = {k: v for k, v in value.items() if k not in inner}
        if rest:
            unmapped[key] = rest
    return unmapped


def _carry(attributes: dict[str, Any], record: dict[str, Any], used: Used) -> None:
    """Keep :func:`unmapped_fields` on the span as ``bandits.unmapped``.

    Converters name only what they consume; everything else comes along by
    default, so a field nobody listed (usage, cost, a vendor's new field) is
    never silently dropped, and every source puts such fields in the same
    place. Their bytes also stay in the source archive.
    """
    unmapped = unmapped_fields(record, used)
    if unmapped:
        attributes["bandits.unmapped"] = unmapped


def _standard_usage(
    attributes: dict[str, Any], tokens: dict[str, Any], params: dict[str, Any]
) -> None:
    """GenAI names for token counts and request parameters a source recorded
    under its own names, so readers find them in one place."""
    for name, value in tokens.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes.setdefault(f"gen_ai.usage.{name}", value)
    for name in ("temperature", "top_p", "max_tokens", "top_k", "seed"):
        value = params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes.setdefault(f"gen_ai.request.{name}", value)


def _records(data: bytes) -> Iterator[tuple[Position, dict[str, Any]]]:
    stripped = data.lstrip()
    if stripped.startswith(b"["):
        parsed = json.loads(data)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON array")
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError("expected objects in JSON array")
            yield {"index": index}, item
    elif stripped.startswith(b"{") and (
        b"\n" not in stripped or not stripped.split(b"\n", 1)[0].rstrip().endswith(b"}")
    ):
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        yield {"document": True}, parsed
    else:
        for number, line in enumerate(io.BytesIO(data), start=1):
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("expected one object per JSONL line")
                yield {"line": number}, parsed


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
) -> dict[str, Any]:
    """One OTLP span, or :class:`_Unconvertible` naming what is missing."""
    if not span_id:
        raise _Unconvertible("empty id")
    beginning, ending = _ns(start), _ns(end)
    if beginning is None:
        raise _Unconvertible("start time missing" if start is None else "start time unparseable")
    if ending is None:
        raise _Unconvertible(
            "end time missing (still running?)" if end is None else "end time unparseable"
        )
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


def _langfuse(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    if not isinstance(record.get("observations"), list):
        raise ValueError("Langfuse input requires a trace with observations[]")
    trace_record = record.get("trace") if isinstance(record.get("trace"), dict) else record
    trace_id = trace_record.get("trace_id") or trace_record.get("id")
    if not trace_id:
        raise ValueError("Langfuse trace has no trace_id/id")
    spans: list[dict[str, Any]] = []
    skipped: Skipped = []
    seen: set[str] = set()
    # A top step is one whose parent is not in this record (absent from the
    # export, or none at all); each carries the trace-level fields.
    present = {str(o.get("id")) for o in _langfuse_observations(record["observations"])}
    # The flag marks observations nested in one that was skipped whole: they
    # cannot be placed, so they are counted rather than silently lost.
    stack = [(o, None, False) for o in reversed(record["observations"])]
    while stack:
        observation, enclosing_id, orphaned = stack.pop()
        if not isinstance(observation, dict):
            skipped.append(("not an object", None))
            continue
        native_id = observation.get("id")
        reason = (
            _DUPLICATE
            if native_id is not None and str(native_id) in seen
            else _INSIDE_SKIPPED
            if orphaned
            else "empty id"
            if native_id is None or native_id == ""
            else None
        )
        if reason is not None:
            skipped.append((reason, None if native_id is None else str(native_id)))
            stack.extend((c, None, True) for c in reversed(observation.get("children") or []))
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
            "bandits.source.record": {**position, "observation_id": str(native_id)},
        }
        parent = observation.get("parentObservationId") or enclosing_id
        if parent is None or str(parent) not in present:
            # Carried on every top step (any may become the request, which is
            # not stored as a step); the decoder moves it onto the trace.
            attributes[TRACE_RECORD] = {
                k: v for k, v in trace_record.items() if k != "observations"
            }
        metadata = observation.get("metadata")
        if isinstance(metadata, str):
            # Some exports serialize metadata as JSON text; read it the same way.
            try:
                decoded = json.loads(metadata)
            except ValueError:
                decoded = None
            metadata = decoded if isinstance(decoded, dict) else metadata
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key in _PROMOTED_METADATA and isinstance(value, dict):
                    # The span's resource or scope (below), not stored twice;
                    # the marker lets readers find it under its own name.
                    moved = attributes.setdefault("bandits.native.metadata_moved", {})
                    moved[key] = _PROMOTED_METADATA[key]
                    continue
                attributes[f"metadata.{key}"] = value
        # Metadata that is not an object is not consumed: it is carried as-is.
        _carry(
            attributes,
            observation,
            langfuse_used(observation),
        )
        usage = observation.get("usageDetails") or observation.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        params = observation.get("modelParameters")
        _standard_usage(
            attributes,
            {
                "input_tokens": usage.get("input"),
                "output_tokens": usage.get("output"),
                "total_tokens": usage.get("total"),
            },
            params if isinstance(params, dict) else {},
        )
        try:
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
        except _Unconvertible as exc:
            # Its children still have an id and times of their own; they keep
            # this one as their (now absent) parent.
            skipped.append((str(exc), str(native_id)))
        else:
            if isinstance(metadata, dict):
                if isinstance(metadata.get("resourceAttributes"), dict):
                    span["_native_resource"] = metadata["resourceAttributes"]
                if isinstance(metadata.get("scope"), dict):
                    span["_native_scope"] = metadata["scope"]
            spans.append(span)
        for child in reversed(observation.get("children") or []):
            stack.append((child, native_id, False))
    return spans, skipped


_LANGFUSE_USED: Used = {
    "id": None,
    "type": None,
    "name": None,
    "parentObservationId": None,
    "startTime": None,
    "endTime": None,
    "input": None,
    "output": None,
    "model": None,
    "level": None,
    "metadata": None,
    "children": None,
}
"""Langfuse observation fields the converter turns into span fields or the
attributes above; every other field is kept in ``bandits.unmapped``."""


_PROMOTED_METADATA = {"resourceAttributes": "resource", "scope": "scope"}
"""Langfuse metadata keys holding the span's OTel resource and scope: they
become the span's resource and scope, read back as ``metadata.*`` by
``bandits.fields``."""


def langfuse_used(observation: dict[str, Any]) -> Used:
    """Langfuse's consumed fields for *observation*: metadata is consumed only
    when it is an object (or JSON text of one), whose keys become ``metadata.*``."""
    metadata = observation.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            pass
    if isinstance(metadata, dict):
        return _LANGFUSE_USED
    return {k: v for k, v in _LANGFUSE_USED.items() if k != "metadata"}


RAW_IO: dict[str, dict[str, str]] = {
    "langfuse": {"input.value": "input", "output.value": "output"},
    "langsmith": {"input.value": "inputs", "output.value": "outputs"},
}
"""Per source: the raw field each converter writes to an I/O value key."""

USED_FIELDS = {
    "langfuse": langfuse_used,
    "langsmith": lambda record: _LANGSMITH_USED,
    "phoenix": lambda record: _PHOENIX_USED,
    "failproofai": lambda record: failproofai_used(record),
}
"""Per source, the fields its converter consumes for one native record."""


def _langfuse_observations(observations: list[Any]) -> Iterator[dict[str, Any]]:
    stack = list(observations)
    while stack:
        observation = stack.pop()
        if isinstance(observation, dict):
            yield observation
            stack.extend(observation.get("children") or [])


_LANGSMITH_KIND = {
    "llm": "LLM",
    "chat_model": "LLM",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "chain": "CHAIN",
    "prompt": "PROMPT",
}


_LANGSMITH_USED: Used = {
    "id": None,
    "run_id": None,
    "run_type": None,
    "name": None,
    "trace_id": None,
    "parent_run_id": None,
    "start_time": None,
    "end_time": None,
    "inputs": None,
    "outputs": None,
    "status": None,
    "error": None,
    "child_runs": None,
}
"""LangSmith run fields the converter consumes; the rest are kept in ``bandits.unmapped``."""


def _langsmith(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    run_id = record.get("id") or record.get("run_id")
    if not run_id or not record.get("run_type"):
        raise ValueError("LangSmith run requires id/run_id and run_type")
    attributes = {
        "openinference.span.kind": _LANGSMITH_KIND.get(str(record["run_type"]).lower(), "UNKNOWN"),
        "input.value": record.get("inputs"),
        "output.value": record.get("outputs"),
        "bandits.source.record": {**position, "observation_id": str(run_id)},
    }
    _carry(attributes, record, _LANGSMITH_USED)
    extra_fields = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    params = extra_fields.get("invocation_params")
    _standard_usage(
        attributes,
        {
            "input_tokens": record.get("prompt_tokens"),
            "output_tokens": record.get("completion_tokens"),
            "total_tokens": record.get("total_tokens"),
        },
        params if isinstance(params, dict) else {},
    )
    # Keep the producer's structure in input.value/output.value. The shared
    # decoder normalizes LangChain's nested message batches and generations;
    # assigning them directly to gen_ai.* would falsely mark raw lists as
    # already-normalized GenAI messages.
    status = str(record.get("status") or "").lower()
    extra = (
        {"status": {"code": 2, "message": str(record.get("error") or "")}}
        if status == "error" or record.get("error")
        else None
    )
    try:
        span = _span(
            record.get("trace_id") or run_id,
            run_id,
            record.get("parent_run_id"),
            record.get("name"),
            record.get("start_time"),
            record.get("end_time"),
            attributes,
            namespace="langsmith",
            extra=extra,
        )
    except _Unconvertible as exc:
        return [], [(str(exc), str(run_id))]
    return [span], []


_PHOENIX_USED: Used = {
    "context": frozenset({"trace_id", "span_id"}),
    "attributes": None,
    "name": None,
    "parent_id": None,
    "start_time": None,
    "end_time": None,
    "span_kind": None,
    "status": frozenset({"status_code", "code", "message"}),
    "status_code": None,
    "status_message": None,
    "events": None,
    "links": None,
    "resource": frozenset({"attributes"}),
    "instrumentation_scope": None,
    "instrumentationScope": None,
}
"""Phoenix span fields the converter consumes; the rest are kept in ``bandits.unmapped``."""


def _phoenix(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    context = record.get("context")
    if not isinstance(context, dict) or not context.get("trace_id") or not context.get("span_id"):
        raise ValueError("Phoenix span requires context.trace_id and context.span_id")
    attributes = record.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise ValueError("Phoenix span attributes must be an object")
    attributes = {
        **attributes,
        "bandits.source.record": {**position, "observation_id": str(context["span_id"])},
    }
    if "openinference.span.kind" not in attributes and isinstance(record.get("span_kind"), str):
        attributes["openinference.span.kind"] = record["span_kind"]
    _carry(attributes, record, _PHOENIX_USED)
    resource = record.get("resource") or {}
    scope = record.get("instrumentation_scope") or record.get("instrumentationScope") or {}
    status = record.get("status")
    if status is None and record.get("status_code") is not None:
        status = {
            "status_code": record["status_code"],
            "message": record.get("status_message"),
        }
    if isinstance(status, dict) and str(
        status.get("status_code") or status.get("code") or ""
    ).upper() in ("ERROR", "STATUS_CODE_ERROR"):
        status = {"code": 2, "message": status.get("message")}
    extra = {
        "events": record.get("events") or [],
        "links": record.get("links") or [],
        "status": status,
    }
    try:
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
    except _Unconvertible as exc:
        return [], [(str(exc), str(context["span_id"]))]
    if isinstance(resource, dict):
        span["_native_resource"] = resource.get("attributes") or {}
    if isinstance(scope, dict):
        span["_native_scope"] = scope
    return [span], []


FAILPROOFAI_VERSIONS = ("1", "2")
"""FailproofAI session transcript versions read: v1 (the dashboard's evaluator
JSON export, integer event ids) and v2 (string ids, ``event_count``)."""

_FAILPROOFAI_PAIRS = (
    # (opener, closer, the documented id both carry, span kind)
    ("agent_start", "agent_end", "agent_id", "AGENT"),
    ("model_request", "model_response", "request_id", "LLM"),
    ("tool_use", "tool_result", "tool_call_id", "TOOL"),
    ("hook_triggered", "hook_completed", "hook_id", "CHAIN"),
    ("agent_pause", "agent_resume", "pause_id", "UNKNOWN"),
    ("human_wait", "human_input", "input_id", "UNKNOWN"),
)
"""FailproofAI's documented opener/closer pairs (docs/reference/custom-agents:
"give the closing event the same id as its opener"). Agents have no id of
their own; a start and an end are the same agent by ``agent_id``."""

FAILPROOFAI_EVENT_TYPES = frozenset(
    {t for opener, closer, _, _ in _FAILPROOFAI_PAIRS for t in (opener, closer)}
    | {"error", "human_pause", "human_interrupt"}
)
"""All fifteen documented event types; the last three stand alone."""

_FAILPROOFAI_FAILED = frozenset({"failed", "error", "timeout", "rejected"})
"""``agent_end.outcome`` values FailproofAI counts as a failed run; any other
value, including ``"failure"``, counts as success."""

_FAILPROOFAI_IO: dict[str, tuple[str, tuple[str, ...]]] = {
    "agent_start": ("input", ("goal",)),
    "agent_end": ("output", ("outcome", "summary")),
    "model_request": ("input", ("messages", "system", "tools")),
    # tool_calls is not an SDK field, but a model's calls are recorded beside
    # its content in the LangChain/OpenAI message shape; the decoder reads them.
    "model_response": ("output", ("role", "content", "tool_calls")),
    "tool_use": ("input", ("input",)),
    "tool_result": ("output", ("output",)),
    "hook_triggered": ("input", ("input",)),
    "hook_completed": ("output", ("output",)),
    "human_wait": ("input", ("prompt", "options")),
    "human_input": ("output", ("response",)),
}
"""Per event type: which side of the span its payload fills, and the fields."""

_FAILPROOFAI_ATTRIBUTES: dict[str, dict[str, str]] = {
    "agent_start": {"parent_id": "failproofai.agent.parent_id"},
    "model_request": {"model": "gen_ai.request.model", "request_id": "failproofai.request_id"},
    "model_response": {
        "model": "gen_ai.response.model",
        "input_tokens": "gen_ai.usage.input_tokens",
        "output_tokens": "gen_ai.usage.output_tokens",
        "request_id": "failproofai.request_id",
    },
    "tool_use": {"tool_name": "gen_ai.tool.name", "tool_call_id": "gen_ai.tool.call.id"},
    "tool_result": {"tool_name": "gen_ai.tool.name", "tool_call_id": "gen_ai.tool.call.id"},
    "hook_triggered": {
        "hook_name": "failproofai.hook.name",
        "hook_id": "failproofai.hook.id",
        "trigger_event": "failproofai.hook.trigger_event",
    },
    "hook_completed": {
        "hook_name": "failproofai.hook.name",
        "hook_id": "failproofai.hook.id",
        "outcome": "failproofai.hook.outcome",
    },
    "agent_pause": {"pause_id": "failproofai.pause.id", "reason": "failproofai.pause.reason"},
    "agent_resume": {"pause_id": "failproofai.pause.id", "reason": "failproofai.resume.reason"},
    "human_wait": {"input_id": "failproofai.human.input_id", "reason": "failproofai.human.reason"},
    "human_input": {"input_id": "failproofai.human.input_id"},
}
"""Per event type: payload fields kept as span attributes, under these names.
A closer's value lands only where its opener left the name unset."""

_FAILPROOFAI_EVENT_KEYS = frozenset({"id", "ts", "event_type", "payload"})


def failproofai_used(event: dict[str, Any]) -> Used:
    """The fields of one transcript event the converter consumes: its envelope
    and the payload fields it maps. The rest of the payload (the producer's
    own fields, ``agent_id``, ``environment``) stays in ``bandits.unmapped``."""
    event_type = event.get("event_type")
    side = _FAILPROOFAI_IO.get(event_type, ("", ()))[1]
    names = _FAILPROOFAI_ATTRIBUTES.get(event_type, {})
    consumed = {"type", *side, *names}
    if event_type in ("tool_result", "hook_completed"):
        consumed.add("error")
    if event_type == "model_response":
        consumed.add("stop_reason")  # gen_ai.response.finish_reasons
    return {k: None for k in _FAILPROOFAI_EVENT_KEYS - {"payload"}} | {
        "payload": frozenset(consumed)
    }


def failproofai_declared(event: dict[str, Any], key: str) -> tuple[bool, Any] | None:
    """What *event* (a span's opener, which its pointer names) declared under
    I/O attribute *key*, built as the converter builds it: ``(True, value)``,
    ``(False, None)`` when it declared none, or None when the value comes from
    the closer, a record the pointer does not name."""
    side, fields = _FAILPROOFAI_IO.get(event.get("event_type"), ("", ()))
    if key != f"{side}.value":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    values = {f: payload[f] for f in fields if f in payload}
    if not values:
        return False, None
    return True, values[fields[0]] if len(fields) == 1 else values


RAW_DECLARED = {"failproofai": failproofai_declared}
"""Per source whose I/O sits below the record's top level: what a raw record
declared under an I/O attribute (see :func:`failproofai_declared`)."""


def _failproofai_key(value: object) -> str | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    return str(value) if isinstance(value, (str, int)) else None


def _failproofai_pairs(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], dict[int, str]]:
    """Opener/closer pairs with the basis each was matched on, and the reason
    every other pair event (by list position) was left unpaired.

    The documented id comes first. Only events that carry none may pair on
    ``correlation_id``, a field FailproofAI does not define but producers
    write on both halves; that basis is named ``source_extension:…``. Either
    way a pair needs exactly one opener and one closer sharing the value, with
    the same ``tool_name``/``hook_name`` where both record one, and the closer
    not before the opener. Arrival order is never a pairing: FailproofAI's own
    fallback, and wrong whenever two calls overlap.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    unpaired: dict[int, str] = {}
    position = {id(e): i for i, e in enumerate(events)}
    for opener, closer, key, _ in _FAILPROOFAI_PAIRS:
        halves = [e for e in events if e["event_type"] in (opener, closer)]
        for basis in (key, "correlation_id"):
            groups: dict[str, list[dict[str, Any]]] = {}
            for event in halves:
                if position[id(event)] in unpaired:
                    continue
                if basis != key and _failproofai_key(event["payload"].get(key)) is not None:
                    continue  # carries the documented id, so is matched on it alone
                value = _failproofai_key(event["payload"].get(basis))
                if value is not None:
                    groups.setdefault(value, []).append(event)
            for value, group in groups.items():
                openers = [e for e in group if e["event_type"] == opener]
                closers = [e for e in group if e["event_type"] == closer]
                reason = None
                if len(openers) != 1 or len(closers) != 1:
                    reason = (
                        f"{len(openers)} {opener} and {len(closers)} {closer} share "
                        f"{basis} {value!r}"
                    )
                else:
                    first, last = openers[0]["payload"], closers[0]["payload"]
                    started, ended = _ns(openers[0]["ts"]), _ns(closers[0]["ts"])
                    name = next(
                        (n for n in ("tool_name", "hook_name") if first.get(n) and last.get(n)),
                        None,
                    )
                    if name and first[name] != last[name]:
                        reason = f"{basis} {value!r} joins different {name}s"
                    elif started is not None and ended is not None and ended < started:
                        reason = f"{closer} with {basis} {value!r} is before its {opener}"
                if reason is not None:
                    for event in group:
                        unpaired[position[id(event)]] = reason
                    continue
                label = basis if basis == key else f"source_extension:{basis}"
                pairs.append((openers[0], closers[0], label))
                for event in group:
                    unpaired[position[id(event)]] = ""
        for event in halves:
            reason = unpaired.get(position[id(event)])
            if reason is None:
                unpaired[position[id(event)]] = (
                    f"no {opener if event['event_type'] == closer else closer} "
                    f"shares an id with it"
                )
    return pairs, {i: r for i, r in unpaired.items() if r}


def _failproofai(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    """One FailproofAI session transcript: a session span, and a span per
    proven opener/closer pair beneath it (beneath its agent's span when that
    agent's start and end were paired).

    Every event that does not become a span is kept whole in the trace
    record under ``unpaired_events``, with the reason, and counted as
    ``kept_on_trace``: unpaired or ambiguous halves, the standalone
    ``error``/``human_pause``/``human_interrupt`` (FailproofAI names no span
    they belong to), unknown types, and paired pauses and human waits, which
    have no span kind of their own (``human_input`` is kept as recorded; its
    authorship is FailproofAI's to state, not inferred here).
    """
    version = record.get("schema_version")
    if version not in FAILPROOFAI_VERSIONS:
        raise ValueError(f"FailproofAI transcript schema_version {version!r} is not one of 1, 2")
    events = record.get("events")
    session_id = record.get("session_id")
    if not isinstance(events, list) or not session_id:
        raise ValueError("FailproofAI transcript requires session_id and events[]")
    if "event_count" in record and record["event_count"] != len(events):
        raise ValueError(
            f"event_count is {record['event_count']!r} but the transcript has {len(events)} events"
        )
    skipped: Skipped = []
    readable: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []  # events preserved on the trace, not as spans
    seen: set[str] = set()
    for event in events:
        if not (
            isinstance(event, dict)
            and isinstance(event.get("payload"), dict)
            and isinstance(event.get("event_type"), str)
            and _failproofai_key(event.get("id")) is not None
        ):
            kept.append(
                {"reason": "not an event with id, event_type and a payload object", "event": event}
            )
            continue
        native_id = str(event["id"])
        if native_id in seen:
            kept.append({"reason": "duplicate event id; the first is used", "event": event})
            continue
        seen.add(native_id)
        readable.append(event)
    pairs, reasons = _failproofai_pairs(readable)
    paired = {id(e) for opener, closer, _ in pairs for e in (opener, closer)}
    for index, event in enumerate(readable):
        if id(event) in paired:
            continue
        event_type = event["event_type"]
        reason = reasons.get(index) or (
            "standalone event; FailproofAI records no span it belongs to"
            if event_type in FAILPROOFAI_EVENT_TYPES
            else f"event type {event_type!r} is not in FailproofAI's documented catalog"
        )
        kept.append({"reason": reason, "event": event})

    session_span: str | None = f"session:{session_id}"
    if _ns(record.get("started_at")) is None or _ns(record.get("ended_at")) is None:
        skipped.append(("session has no parseable started_at/ended_at", str(session_id)))
        session_span = None
    agents = {
        opener["payload"].get("agent_id"): opener
        for opener, _, _ in pairs
        if opener["event_type"] == "agent_start"
    }
    spans: list[dict[str, Any]] = []
    for opener, closer, basis in pairs:
        kind = next(k for o, _, _, k in _FAILPROOFAI_PAIRS if o == opener["event_type"])
        if kind == "UNKNOWN":
            for event, other in ((opener, closer), (closer, opener)):
                kept.append(
                    {
                        "reason": f"paired by {basis} with event {other['id']}; no span kind "
                        f"represents {opener['event_type']}/{closer['event_type']}",
                        "event": event,
                    }
                )
            continue
        attributes: dict[str, Any] = {
            "openinference.span.kind": kind,
            "failproofai.pairing": basis,
            "failproofai.closer_event_id": closer["id"],
            "bandits.source.record": {**position, "observation_id": str(opener["id"])},
        }
        for event in (opener, closer):
            payload = event["payload"]
            side, fields = _FAILPROOFAI_IO.get(event["event_type"], ("", ()))
            values = {f: payload[f] for f in fields if f in payload}
            if values:
                attributes[f"{side}.value"] = values[fields[0]] if len(fields) == 1 else values
            for field, name in _FAILPROOFAI_ATTRIBUTES.get(event["event_type"], {}).items():
                attributes.setdefault(name, payload.get(field))
        if closer["event_type"] == "model_response" and closer["payload"].get("stop_reason"):
            attributes["gen_ai.response.finish_reasons"] = [closer["payload"]["stop_reason"]]
        _carry(attributes, opener, failproofai_used(opener))
        closer_rest = unmapped_fields(closer, failproofai_used(closer))
        if closer_rest:
            attributes["failproofai.closer_unmapped"] = closer_rest
        error = (
            closer["payload"].get("error")
            if closer["event_type"] in ("tool_result", "hook_completed")
            else None
        )
        outcome = closer["payload"].get("outcome") if closer["event_type"] == "agent_end" else None
        extra = (
            {"status": {"code": 2, "message": str(error or outcome)}}
            if error or (isinstance(outcome, str) and outcome in _FAILPROOFAI_FAILED)
            else None
        )
        agent_id = opener["payload"].get("agent_id")
        owner = agents.get(
            opener["payload"].get("parent_id") if opener["event_type"] == "agent_start" else agent_id
        )
        parent = (
            f"agent_start:{owner['id']}" if owner is not None and owner is not opener else session_span
        )
        name = (
            attributes.get("gen_ai.request.model")
            or attributes.get("gen_ai.tool.name")
            or attributes.get("failproofai.hook.name")
            or (agent_id if kind == "AGENT" else None)
            or opener["event_type"]
        )
        try:
            spans.append(
                _span(
                    session_id,
                    f"{opener['event_type']}:{opener['id']}",
                    parent,
                    name,
                    opener["ts"],
                    closer["ts"],
                    attributes,
                    namespace="failproofai",
                    extra=extra,
                )
            )
        except _Unconvertible as exc:
            kept.extend({"reason": str(exc), "event": event} for event in (opener, closer))
    trace_record = {k: v for k, v in record.items() if k != "events"}
    if kept:
        trace_record["unpaired_events"] = kept
    if session_span is not None:
        spans.insert(
            0,
            _span(
                session_id,
                session_span,
                None,
                f"session {record.get('agent_id') or session_id}",
                record["started_at"],
                record["ended_at"],
                {
                    "openinference.span.kind": "CHAIN",
                    "failproofai.agent_id": record.get("agent_id"),
                    "failproofai.environment": record.get("environment"),
                    TRACE_RECORD: trace_record,
                },
                namespace="failproofai",
            ),
        )
    else:
        # No session span to carry the record: every top span carries it.
        for span in spans:
            if "parentSpanId" not in span:
                span["attributes"].append({"key": TRACE_RECORD, "value": _av(trace_record)})
    # With no span at all there is no trace to keep them on.
    prefix = _ON_TRACE if spans else ""
    for entry in kept:
        event = entry["event"]
        native_id = event.get("id") if isinstance(event, dict) else None
        skipped.append(
            (prefix + entry["reason"], None if native_id is None else str(native_id))
        )
    return spans, skipped


_CONVERTERS = {
    "langfuse": _langfuse,
    "langsmith": _langsmith,
    "phoenix": _phoenix,
    "failproofai": _failproofai,
}


def _langsmith_runs(record: dict[str, Any], skipped: Skipped) -> Iterator[dict[str, Any]]:
    """Flatten documented child_runs without turning nesting into time order.

    Children that are not objects, and repeated run ids, go to *skipped*; so
    do the children of a repeated run, which have no single place to go.
    """
    stack = [
        (record, None, record.get("trace_id") or record.get("id") or record.get("run_id"), False)
    ]
    seen: set[str] = set()
    while stack:
        run, enclosing, trace_id, orphaned = stack.pop()
        if not isinstance(run, dict):
            skipped.append(("not an object", None))
            continue
        native_id = str(run.get("id") or run.get("run_id"))
        if native_id in seen or orphaned:
            skipped.append((_DUPLICATE if native_id in seen else _INSIDE_SKIPPED, native_id))
            stack.extend((c, None, trace_id, True) for c in reversed(run.get("child_runs") or []))
            continue
        seen.add(native_id)
        if enclosing and not run.get("parent_run_id"):
            run = {**run, "parent_run_id": enclosing}
        if trace_id and not run.get("trace_id"):
            run = {**run, "trace_id": trace_id}
        yield run
        for child in reversed(run.get("child_runs") or []):
            stack.append((child, run.get("id") or run.get("run_id"), trace_id, False))


def _where(path: Path, position: Position) -> str:
    if "line" in position:
        return f"{path}:{position['line']}" + (
            f"[{position['index']}]" if "index" in position else ""
        )
    return f"{path}[{position['index']}]" if "index" in position else str(path)


def _origin(path: Path, span: dict[str, Any]) -> str:
    """Where a converted span came from in the native file, for issue locations."""
    for attribute in span.get("attributes") or []:
        if attribute.get("key") == "bandits.source.record":
            pointer = json.loads(attribute["value"]["stringValue"])
            position = {k: v for k, v in pointer.items() if k != "observation_id"}
            return f"{_where(path, position)} observation {pointer['observation_id']}"
    return str(path)


class NativeConversion:
    """One native file converted to OTLP in chunks, counting every observation.

    :meth:`chunks` yields the path of a temporary OTLP/JSONL file holding the
    next batch; whoever consumes it (the loader, or request discovery) must
    finish reading before asking for the next. Langfuse records are batched
    by about 400 spans so a large export never has to fit in memory.
    """

    def __init__(
        self,
        path: Path,
        source_name: str,
        ruleset: RedactionRuleset,
        *,
        scratch_dir: Path | None = None,
    ) -> None:
        if source_name not in _CONVERTERS:
            raise ValueError(f"unknown native source {source_name!r}")
        if path.is_dir():
            raise ValueError(
                f"{source_name} reads one export file; {path} is a directory — ingest each file"
            )
        self.path, self.source_name, self.ruleset = path, source_name, ruleset
        self.scratch_dir = scratch_dir
        self.issues: list[TraceIssue] = []
        self.source_hash = hashlib.sha256()
        # Converter-level counts; a caller adds the decoded chunks' own.
        self.report = IngestReport()
        self.observations = 0
        # Native origin of each line in the current converted chunk.
        self.origins: list[str] = []
        # Records kept whole on each trace's source_record, by OTLP trace id;
        # see :meth:`drop_kept`.
        self.kept_by_trace: dict[str, int] = {}

    def _source_records(self) -> Iterator[tuple[Position, dict[str, Any]]]:
        path, ruleset = self.path, self.ruleset
        with path.open("rb") as stream:
            first = stream.readline(_LINE_LIMIT)
            first_number = 1
            while first and not first.strip():
                self.source_hash.update(first)
                first = stream.readline(_LINE_LIMIT)
                first_number += 1
            # A first line cut at the limit is a single-line array export
            # (streamed below); any other long first line is read whole.
            cut = len(first) == _LINE_LIMIT and not first.endswith(b"\n")
            if cut and not first.lstrip().startswith(b"["):
                first += stream.readline()
                cut = False
            # Complete JSON objects/arrays are independent JSONL records.
            try:
                jsonl = not cut and isinstance(json.loads(first), (dict, list))
            except (UnicodeDecodeError, json.JSONDecodeError):
                jsonl = False
            if jsonl:
                for number, line in enumerate(chain((first,), stream), start=first_number):
                    self.source_hash.update(line)
                    if not line.strip():
                        continue
                    safe = redact_bytes(line, f"{path}:record{number}", ruleset)
                    self.issues.extend(safe.issues)
                    for inner, record in _records(safe.data):
                        index = {"index": inner["index"]} if "index" in inner else {}
                        yield {"line": number, **index}, record
            elif first.lstrip().startswith(b"["):
                # One array element in memory at a time, not the whole file.
                stream.seek(0)
                self.source_hash = hashlib.sha256()
                for index, (_, element, line) in enumerate(iter_array(stream, self.source_hash)):
                    if element is None:
                        break
                    safe = redact_bytes(element, str(path), ruleset, first_line=line)
                    self.issues.extend(safe.issues)
                    record = json.loads(safe.data)
                    if not isinstance(record, dict):
                        raise ValueError("expected objects in JSON array")
                    yield {"index": index}, record
            else:
                stream.seek(0)
                original = stream.read()
                self.source_hash = hashlib.sha256(original)
                safe = redact_bytes(original, str(path), ruleset)
                self.issues.extend(safe.issues)
                yield from _records(safe.data)

    def _skip(self, skipped: Skipped, position: Position) -> None:
        self.observations += len(skipped)
        for reason, observation_id in skipped:
            kept = reason.startswith(_ON_TRACE)
            reason = reason.removeprefix(_ON_TRACE)
            bucket = (
                "kept_on_trace"
                if kept
                else "duplicate_native"
                if reason == _DUPLICATE
                else "unconvertible"
            )
            self.report.buckets[bucket] += 1
            examples = (
                self.report.kept_on_trace_examples if kept else self.report.unconvertible_examples
            )
            if len(examples) < EXAMPLES:
                examples.append(
                    f"{_where(self.path, position)} observation {observation_id or '?'}: {reason}"
                )

    def chunks(self) -> Iterator[Path]:
        source_name = self.source_name
        with tempfile.TemporaryDirectory(
            prefix=".bandits-native-", dir=self.scratch_dir or Path.cwd()
        ) as temporary:
            converted = Path(temporary) / "converted.jsonl"
            with converted.open("w+", encoding="utf-8") as output:
                pending = 0
                for position, outer in self._source_records():
                    records = (
                        outer.get("runs")
                        if source_name == "langsmith"
                        else outer.get("spans") or outer.get("data")
                        if source_name == "phoenix"
                        else None
                    )
                    if not isinstance(records, list):
                        records = [outer]
                    skipped: Skipped = []
                    expanded = (
                        (
                            run
                            for record in records
                            for run in (
                                _langsmith_runs(record, skipped)
                                if isinstance(record, dict)
                                else [record]
                            )
                        )
                        if source_name == "langsmith"
                        else iter(records)
                    )
                    for record in expanded:
                        try:
                            if not isinstance(record, dict):
                                raise ValueError("expected an object")
                            spans, dropped = _CONVERTERS[source_name](record, position)
                        except (TypeError, ValueError) as exc:
                            self.issues.append(
                                TraceIssue(
                                    kind="unsupported_native_record",
                                    detail=str(exc),
                                    location=_where(self.path, position),
                                )
                            )
                            self.report.unreadable_items["unsupported_native_record"] += 1
                            continue
                        self.observations += len(spans)
                        self._skip(dropped, position)
                        on_trace = sum(reason.startswith(_ON_TRACE) for reason, _ in dropped)
                        if on_trace and spans:
                            trace = spans[0]["traceId"]
                            self.kept_by_trace[trace] = self.kept_by_trace.get(trace, 0) + on_trace
                        for span in spans:
                            self.origins.append(_origin(self.path, span))
                            output.write(
                                json.dumps(_request(span, source_name), ensure_ascii=False)
                            )
                            output.write("\n")
                            pending += 1
                    self._skip(skipped, position)
                    if source_name == "langfuse" and pending >= 400:
                        output.flush()
                        yield converted
                        output.seek(0)
                        output.truncate(0)
                        self.origins = []
                        pending = 0
                if pending:
                    output.flush()
                    yield converted

    def relocate(self, issue: TraceIssue, chunk: Path) -> TraceIssue:
        """*issue* located in the native file instead of the converted *chunk*.

        The chunk lives in a random temporary directory, so its path in a
        stored issue would make the corpus id differ on every ingest.
        """
        location = issue.location or ""
        if not location.startswith(str(chunk)):
            return issue
        line = location[len(str(chunk)) :].removeprefix(":").split("#", 1)[0]
        if line.isdigit() and 0 < int(line) <= len(self.origins):
            return issue.replace(location=self.origins[int(line) - 1])
        return issue.replace(location=str(self.path))

    def drop_kept(self, kept_traces: set[str]) -> None:
        """Recount records kept on a trace the decoder then dropped (it had no
        model call, tool call or pipeline step) as unconvertible: the trace's
        source_record went with it."""
        for trace, count in self.kept_by_trace.items():
            if trace in kept_traces:
                continue
            self.report.buckets["kept_on_trace"] -= count
            self.report.buckets["unconvertible"] += count
            if len(self.report.unconvertible_examples) < EXAMPLES:
                self.report.unconvertible_examples.append(
                    f"{self.path} trace {trace}: {count} record(s) kept on it were dropped with "
                    "it; the trace has no model call, tool call or pipeline step"
                )

    def finish(self, decoded: IngestReport) -> IngestReport:
        """Converter counts plus the decoded chunks' (*decoded*), checked to add up."""
        total = IngestReport()
        total.merge(self.report)
        total.merge(decoded)
        lost = sum(
            self.report.buckets[b] for b in ("unconvertible", "duplicate_native", "kept_on_trace")
        )
        if self.observations != lost + decoded.spans_seen:
            total.accounting_errors.append(
                f"{self.path}: {self.observations} observation(s) seen but {lost} skipped and "
                f"{decoded.spans_seen} converted; this is a bandits bug"
            )
        total.spans_seen = self.observations
        return total


def _request(span: dict[str, Any], source_name: str) -> dict[str, Any]:
    """One converted span as its own OTLP export request."""
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
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(native_resource)},
                "scopeSpans": [{"scope": scope, "spans": [span]}],
            }
        ]
    }


def iter_native(
    path: Path,
    source_name: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    workflow: WorkflowDeclaration | None = None,
    pipeline_steps: bool = True,
    report: IngestReport | None = None,
    mapping: IngestMapping | None = None,
    scratch_dir: Path | None = None,
) -> Iterator[Trace | TraceCorpus]:
    """Yield traces, then a trace-free corpus footer containing issues and metadata.

    The report is complete only when the iterator is exhausted. Bundled Langfuse
    releases decoded chunks; interleaved native spans are grouped on disk.

    Every observation the converters see is counted: converted, skipped as
    unconvertible or duplicate, and the converted ones through the OTLP
    reader's own buckets. ``report`` receives the totals.
    """
    conversion = NativeConversion(path, source_name, ruleset, scratch_dir=scratch_dir)
    # Always collected: the per-ingest summary issues are built from it once,
    # not once per chunk.
    decoded = IngestReport()
    with path.open("rb") as original:
        digest = hashlib.file_digest(original, "sha256").hexdigest()
    converted_issues: list[TraceIssue] = []
    seen_ids: set[str] = set()
    for chunk_path in conversion.chunks():
        ids = set()
        reader = iter_otlp_standard if source_name != "langfuse" else None
        if reader is None:
            chunk = load_otlp_standard(
                chunk_path,
                ruleset,
                pipeline_steps=pipeline_steps,
                workflow=workflow,
                report=decoded,
                defer_aggregate_issues=True,
                mapping=mapping,
            )
            items = (*chunk.traces, chunk.replace(traces=()))
        else:
            items = reader(
                chunk_path,
                ruleset,
                pipeline_steps=pipeline_steps,
                workflow=workflow,
                report=decoded,
                defer_aggregate_issues=True,
                mapping=mapping,
                scratch_dir=scratch_dir,
            )
        for item in items:
            if isinstance(item, TraceCorpus):
                converted_issues.extend(conversion.relocate(i, chunk_path) for i in item.issues)
            else:
                ids.add(item.trace_id)
                yield item.replace(source=source_name, source_digest=digest)
        decoded.split_trace_ids += len(ids & seen_ids)
        seen_ids |= ids
        if reader is None:
            del chunk, items
    if conversion.source_hash.hexdigest() != digest:
        raise ValueError(f"source file changed during ingest: {path}")
    conversion.drop_kept(seen_ids)
    internal = conversion.finish(decoded)
    deferred = aggregate_issues(
        internal,
        str(path),
        workflow=workflow is not None,
        mapping_name=workflow.mapping_name if workflow is not None else None,
        step_kinds=mapping.step_kinds if mapping is not None else (),
    )
    if report is not None:
        report.merge(internal)
    yield TraceCorpus(
        source=source_name,
        traces=(),
        issues=tuple(conversion.issues) + tuple(converted_issues) + tuple(deferred),
        workflow=workflow,
        redaction_ruleset=ruleset.name,
    )


def load_native(
    path: Path,
    source_name: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    workflow: WorkflowDeclaration | None = None,
    pipeline_steps: bool = True,
    report: IngestReport | None = None,
    mapping: IngestMapping | None = None,
) -> TraceCorpus:
    """Materialize the streaming native reader for library callers."""
    traces = []
    for item in iter_native(
        path,
        source_name,
        ruleset,
        workflow=workflow,
        pipeline_steps=pipeline_steps,
        report=report,
        mapping=mapping,
    ):
        if isinstance(item, TraceCorpus):
            return item.replace(traces=tuple(traces))
        traces.append(item)
    raise RuntimeError("native reader ended without a footer")
