"""LangSmith adapter: exported run trees -> :class:`Trace`.

This is the adapter the ideal customer actually needs. The profile in
docs/PRODUCT.md is a company with millions of trajectories already sitting in
LangSmith or Arize; if the library cannot open that store, the conversation ends
before the fidelity gate ever runs.

LangSmith records a **run tree**, not a message list. Every node is a run with a
``run_type``, and the two that matter are:

``run_type == "tool"``
    A real invocation point, recorded explicitly. ``inputs`` are the arguments,
    ``outputs`` the response, ``error`` the failure. This is the good case and it
    needs no recovery heuristics -- it is the same quality of record as an OTLP
    ``execute_tool`` span.
``run_type in {"llm", "chat_model"}``
    The transcript. We read messages off it for the task statement and the final
    answer, and nothing else. Tool calls are **not** recovered from LLM outputs
    here even though they often appear there: if a tool run exists we already have
    the response, and if it does not, the call has no response to reconstruct an
    environment from. Recovering the request half alone would produce invocation
    points with ``response=None``, which look like data to every stage downstream
    and are not. That case becomes an :class:`IngestIssue` instead.

Ordering
--------
Invocation ``step`` is the order the runs actually executed, taken from
``start_time`` where every tool run declares one, and from document order
otherwise. Position in sequence is what stage 4 uses to separate the reads that
establish the starting state from everything after the first write, so an
ordering guess would corrupt pre-state reconstruction rather than merely
mislabel it. When timestamps are partial we keep document order and say so.

Envelope unwrapping
-------------------
LangChain wraps tool payloads in a single-key envelope: ``{"input": {...}}`` on
the way in, ``{"output": ...}`` on the way out. Those exact one-key forms are
unwrapped, because the envelope is the framework's, not the tool's, and stage 3
infers entities from response *fields*. Anything else is passed through
untouched -- we unwrap a known wrapper, we never reshape a payload we do not
recognize.
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
from bandits.ingest.errors import infer_error_kind, looks_like_error, normalize_error_kind

#: ``run_type`` values that carry an invocation point.
TOOL_RUN_TYPES = frozenset({"tool", "retriever"})

#: ``run_type`` values that carry transcript messages.
LLM_RUN_TYPES = frozenset({"llm", "chat_model"})

#: Top-level keys, in preference order, that may carry the episode identity.
TRACE_ID_KEYS: tuple[str, ...] = ("trace_id", "id", "run_id", "session_id")

#: Single-key envelopes LangChain puts around tool arguments.
INPUT_ENVELOPE_KEYS: tuple[str, ...] = ("input", "args", "kwargs", "tool_input")

#: Single-key envelopes LangChain puts around tool responses.
OUTPUT_ENVELOPE_KEYS: tuple[str, ...] = ("output", "result", "return_value")

#: LangChain message ``type`` -> contract role. Anything else is dropped with an issue.
ROLE_BY_TYPE: dict[str, str] = {
    "system": "system",
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "systemmessage": "system",
    "humanmessage": "user",
    "aimessage": "assistant",
    "toolmessage": "tool",
    "functionmessage": "tool",
}

#: ``extra.metadata`` keys that may carry the ground-truth outcome label.
OUTCOME_KEYS: tuple[str, ...] = ("outcome", "success", "passed", "bandits.outcome")


def _unwrap(payload: Any, keys: tuple[str, ...]) -> JsonValue:
    """Strip one known single-key LangChain envelope, if that is exactly what this is.

    The guard is deliberately narrow: exactly one key, and that key is a known
    envelope name. A tool whose response genuinely has one field called ``output``
    is indistinguishable from an envelope at this layer, and unwrapping it is
    harmless -- stage 3 sees the same value either way. A two-key payload is never
    touched, because then the choice of which key is "the" payload would be ours
    rather than the framework's.
    """
    if not isinstance(payload, dict) or len(payload) != 1:
        return payload
    (key, value), = payload.items()
    if key not in keys:
        return payload
    return value


def _decode_maybe_json(value: Any) -> JsonValue:
    """Decode a JSON-string payload, keeping the raw text when it is not JSON.

    LangSmith stores tool outputs as whatever the tool returned; a tool that returns
    a serialized string is common. Keeping the raw text rather than discarding it
    means a text-only tool still reaches stage 2 as evidence, even though it will
    not yield an entity.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _arguments(run: JsonObject, *, tool: str, issues: list[IngestIssue], location: str) -> JsonObject:
    """Normalize one tool run's ``inputs`` into an argument object.

    Arguments are the half of an invocation point stage 7 replays, so a payload that
    is not an object is preserved under ``_raw`` and flagged rather than reshaped
    into ``{}`` -- an empty argument set would read downstream as "this tool takes no
    arguments", which is a different and false claim.
    """
    raw = run.get("inputs")
    if raw is None:
        return {}
    value = _decode_maybe_json(_unwrap(raw, INPUT_ENVELOPE_KEYS))
    if isinstance(value, dict):
        return value
    issues.append(IngestIssue(
        kind="malformed_arguments",
        detail=f"inputs for {tool!r} decoded to {type(value).__name__}, not an object",
        location=location,
    ))
    return {"_raw": value}


def _response(run: JsonObject) -> JsonValue:
    """Normalize one tool run's ``outputs`` into the response body."""
    raw = run.get("outputs")
    if raw is None:
        return None
    return _decode_maybe_json(_unwrap(raw, OUTPUT_ENVELOPE_KEYS))


def _error_text(run: JsonObject) -> str | None:
    """The run's declared error string, if it declares one."""
    error = run.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        for key in ("message", "detail", "error"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _status_and_kind(run: JsonObject, response: JsonValue) -> tuple[CallStatus, str | None]:
    """Decide OK/ERROR for one tool run, and the coarse error label when it failed.

    Two independent signals, and either is sufficient: LangSmith's own ``error``
    field, and an error-shaped response body. A run that returned
    ``{"error": "not_found"}`` without setting the run-level error is still a
    failure, and the environment has to be able to reproduce it -- an env that never
    returns ``not_found`` trains an agent that has never seen adversity.
    """
    text = _error_text(run)
    if text is not None:
        kind = infer_error_kind(response) or normalize_error_kind(text)
        return CallStatus.ERROR, kind
    if looks_like_error(response):
        return CallStatus.ERROR, infer_error_kind(response)
    return CallStatus.OK, None


def _latency_ms(run: JsonObject) -> float | None:
    """Wall-clock duration from the run's timestamps, when both parse."""
    start = _timestamp(run.get("start_time"))
    end = _timestamp(run.get("end_time"))
    if start is None or end is None:
        return None
    delta = (end - start) * 1000.0
    return delta if delta >= 0 else None


def _timestamp(value: Any) -> float | None:
    """Parse a LangSmith timestamp into epoch seconds, or ``None``.

    ISO-8601 strings are the documented form; epoch numbers show up in exports that
    went through a warehouse. A trailing ``Z`` is normalized because
    :meth:`datetime.fromisoformat` rejects it before 3.11.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    from datetime import datetime

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _iter_runs(record: JsonObject, *, issues: list[IngestIssue], location: str) -> list[tuple[JsonObject, str]]:
    """Flatten one record into ``(run, location)`` pairs in document order.

    Both export shapes are accepted: a root run carrying ``child_runs``, and a
    ``{"runs": [...]}`` envelope holding an already-flat list (what you get from the
    LangSmith API when you query by ``trace_id``). Nesting depth carries no meaning
    for us -- a tool run is an invocation point wherever it sits in the tree.
    """
    flat = record.get("runs")
    if isinstance(flat, list):
        out: list[tuple[JsonObject, str]] = []
        for index, run in enumerate(flat):
            where = f"{location}:runs[{index}]"
            if isinstance(run, dict):
                out.append((run, where))
            else:
                issues.append(IngestIssue(
                    kind="malformed_run",
                    detail=f"expected a run object, got {type(run).__name__}",
                    location=where,
                ))
        return out

    out = []

    def walk(run: JsonObject, where: str) -> None:
        out.append((run, where))
        children = run.get("child_runs")
        if not isinstance(children, list):
            return
        for index, child in enumerate(children):
            child_where = f"{where}:child_runs[{index}]"
            if isinstance(child, dict):
                walk(child, child_where)
            else:
                issues.append(IngestIssue(
                    kind="malformed_run",
                    detail=f"expected a run object, got {type(child).__name__}",
                    location=child_where,
                ))

    walk(record, location)
    return out


def _order(runs: list[tuple[JsonObject, str]]) -> list[tuple[JsonObject, str]]:
    """Sort tool runs by ``start_time`` when every one of them declares a parseable one.

    All-or-nothing on purpose. A partial sort -- timestamped runs ordered, the rest
    appended -- would silently interleave two different orderings, and step order is
    what stage 4 uses to find the last read before the first write. Document order is
    the honest fallback.
    """
    stamps = [_timestamp(run.get("start_time")) for run, _ in runs]
    if any(stamp is None for stamp in stamps):
        return runs
    return [pair for _, pair in sorted(zip(stamps, runs, strict=True), key=lambda item: item[0])]


def _message_from_lc(raw: Any) -> tuple[Message | None, bool]:
    """Convert one LangChain message into a contract :class:`Message`.

    Returns ``(message, recognized)``. Two encodings occur: the plain
    ``{"type": "human", "content": ...}`` dict, and the serialized
    ``{"id": ["langchain", ..., "HumanMessage"], "kwargs": {...}}`` form that
    ``load``/``dumpd`` round-trips produce. ``recognized`` is False for anything
    else, so the caller can raise an issue rather than drop the turn in silence.
    """
    if not isinstance(raw, dict):
        return None, False

    kwargs = raw.get("kwargs")
    if isinstance(kwargs, dict):
        identity = raw.get("id")
        type_hint = identity[-1] if isinstance(identity, list) and identity else None
        body = kwargs
    else:
        type_hint = None
        body = raw

    declared = body.get("type") or body.get("role") or type_hint
    if not isinstance(declared, str):
        return None, False
    role = ROLE_BY_TYPE.get(declared.strip().lower())
    if role is None:
        return None, False

    content = body.get("content")
    if isinstance(content, list):
        # Multi-part content blocks: keep the text parts, which is where the task
        # statement and the final answer live.
        parts = [
            part.get("text")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        content = "\n".join(parts) if parts else None
    elif content is not None and not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)

    call_id = body.get("tool_call_id")
    return Message(
        role=role,  # type: ignore[arg-type]
        content=content,
        tool_call_id=call_id if isinstance(call_id, str) else None,
    ), True


def _messages(run: JsonObject, *, issues: list[IngestIssue], location: str) -> list[Message]:
    """Read the transcript off one llm/chat_model run.

    Inputs give the prompt, outputs give the completion. LangSmith nests generations
    two deep (``generations[i][j]``) because a single call may produce several
    candidates; we take them in declared order.
    """
    out: list[Message] = []
    inputs = run.get("inputs")
    raw_messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if isinstance(raw_messages, list):
        # ``messages`` is a list per prompt, so a batch call nests one level deeper.
        flat = raw_messages[0] if raw_messages and isinstance(raw_messages[0], list) else raw_messages
        for index, raw in enumerate(flat):
            message, recognized = _message_from_lc(raw)
            if message is not None:
                out.append(message)
            elif not recognized:
                issues.append(IngestIssue(
                    kind="unrecognized_message",
                    detail="message is neither a plain {'type','content'} dict nor a serialized LangChain message",
                    location=f"{location}:inputs.messages[{index}]",
                ))
    elif isinstance(inputs, dict) and isinstance(inputs.get("input"), str):
        out.append(Message(role="user", content=inputs["input"]))

    out.extend(_completion_messages(run))
    return out


def _completion_messages(run: JsonObject) -> list[Message]:
    """Assistant turns recovered from an llm run's ``outputs``."""
    outputs = run.get("outputs")
    if not isinstance(outputs, dict):
        return []
    out: list[Message] = []
    generations = outputs.get("generations")
    if isinstance(generations, list):
        for group in generations:
            candidates = group if isinstance(group, list) else [group]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                message, _ = _message_from_lc(candidate.get("message"))
                if message is not None and message.content:
                    out.append(message)
                elif isinstance(candidate.get("text"), str) and candidate["text"]:
                    out.append(Message(role="assistant", content=candidate["text"]))
    if not out and isinstance(outputs.get("output"), str):
        out.append(Message(role="assistant", content=outputs["output"]))
    return out


def _metadata(record: JsonObject) -> JsonObject:
    """``extra.metadata`` from the root run, which is where callers put their own labels."""
    extra = record.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("metadata"), dict):
        return dict(extra["metadata"])
    return {}


def _coerce_outcome(value: Any) -> bool | None:
    """Read a ground-truth label; ``None`` keeps 'unlabeled' distinct from 'failed'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "pass", "passed", "success", "1"):
            return True
        if lowered in ("false", "fail", "failed", "failure", "0"):
            return False
    return None


def _outcome(record: JsonObject, metadata: JsonObject) -> bool | None:
    """The episode's success label, from record metadata or the top level."""
    for key in OUTCOME_KEYS:
        if key in metadata:
            resolved = _coerce_outcome(metadata[key])
            if resolved is not None:
                return resolved
        if key in record:
            resolved = _coerce_outcome(record[key])
            if resolved is not None:
                return resolved
    return None


def parse_langsmith_record(
    record: JsonObject,
    *,
    source_digest: str,
    location: str,
    fallback_trace_id: str,
) -> tuple[Trace | None, list[IngestIssue]]:
    """Normalize one LangSmith run tree (one JSONL record) into a :class:`Trace`.

    A record with no runs at all is unusable and returns ``None`` plus an issue. A
    record whose runs contain no tool run *is* returned -- a genuinely tool-free
    episode is real data -- but it carries a ``no_tool_runs`` issue, because in
    practice that means the project logged LLM calls only, and that is the single
    fact that decides whether an environment can be built at all. See
    ``bandits triage``.
    """
    issues: list[IngestIssue] = []
    runs = _iter_runs(record, issues=issues, location=location)
    if not runs:
        issues.append(IngestIssue(
            kind="missing_runs",
            detail="record contains no runs; expected a run object or {'runs': [...]}",
            location=location,
        ))
        return None, issues

    trace_id = fallback_trace_id
    for key in TRACE_ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            trace_id = value
            break

    tool_runs = [(run, where) for run, where in runs if str(run.get("run_type", "")).lower() in TOOL_RUN_TYPES]
    ordered = _order(tool_runs)

    invocations: list[InvocationPoint] = []
    for step, (run, where) in enumerate(ordered):
        name = run.get("name")
        if not isinstance(name, str) or not name:
            issues.append(IngestIssue(
                kind="unnamed_tool_run",
                detail="tool run has no 'name'; the action space cannot include an unnamed tool",
                location=where,
            ))
            continue
        response = _response(run)
        status, error_kind = _status_and_kind(run, response)
        call_id = run.get("id")
        invocations.append(InvocationPoint(
            call_id=call_id if isinstance(call_id, str) and call_id else f"{trace_id}:{step}",
            trace_id=trace_id,
            step=step,
            tool=name,
            arguments=_arguments(run, tool=name, issues=issues, location=where),
            response=response,
            status=status,
            error_kind=error_kind,
            latency_ms=_latency_ms(run),
            source_span_id=call_id if isinstance(call_id, str) else None,
        ))

    messages: list[Message] = []
    for run, where in runs:
        if str(run.get("run_type", "")).lower() in LLM_RUN_TYPES:
            messages.extend(_messages(run, issues=issues, location=where))

    if not messages:
        messages.extend(_root_messages(record))

    if not invocations:
        issues.append(IngestIssue(
            kind="no_tool_runs",
            detail=(
                "no run_type='tool' runs in this trace; without invocation points there "
                "is no action space, no state and no verifier"
            ),
            location=location,
        ))

    metadata = _metadata(record)
    trace = Trace(
        trace_id=trace_id,
        source="langsmith",
        source_digest=source_digest,
        messages=tuple(messages),
        invocations=tuple(invocations),
        outcome=_outcome(record, metadata),
        metadata={
            "episode_id": trace_id,
            "run_count": len(runs),
            "tool_run_count": len(invocations),
            "langsmith_metadata": metadata,
        },
    )
    return trace, issues


def _root_messages(record: JsonObject) -> list[Message]:
    """Fall back to the root run's own inputs/outputs for the task statement.

    An agent executor logged without its llm children still carries the instruction
    on the root run, and stage 5 needs an instruction to mine a task from. This
    recovers the task statement and the final answer only -- never invocation points.
    """
    out: list[Message] = []
    inputs = record.get("inputs")
    if isinstance(inputs, dict):
        for key in ("input", "question", "query", "instruction"):
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                out.append(Message(role="user", content=value))
                break
    outputs = record.get("outputs")
    if isinstance(outputs, dict):
        for key in ("output", "answer", "result"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip():
                out.append(Message(role="assistant", content=value))
                break
    return out
