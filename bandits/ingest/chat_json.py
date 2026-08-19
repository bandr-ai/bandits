"""Chat-JSON adapter: OpenAI-style transcripts -> :class:`Trace`.

This is the recovery path (PLAN.md Step 6: "most stacks log LLM calls, not tool
calls"). There are no tool spans here at all -- only a message list. Invocation
points have to be *reconstructed* by pairing each assistant ``tool_calls`` block
with the ``tool``-role message that answered it.

Pairing rules, in order of preference:

1. **By ``tool_call_id``.** The declared id is the only sound join key: it survives
   interleaving, parallel calls and out-of-order results. This is the normal path.
2. **By tool name plus source order**, and only when the result message declares no
   id. Ambiguous by construction, so it always records an :class:`IngestIssue`.
3. **By source order alone**, when there is not even a name. Also always an issue.

An invocation recovered this way must be indistinguishable from the same call read
off an OTLP span -- that equivalence is the module's correctness criterion and is
asserted directly in ``ingest_test.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from bandits.ingest.errors import infer_error_kind, looks_like_error

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

#: Top-level keys, in preference order, that may carry the episode identity.
TRACE_ID_KEYS: tuple[str, ...] = ("conversation_id", "trace_id", "episode_id", "id")


@dataclass
class _PendingCall:
    """One assistant-declared tool call still waiting for its ``tool``-role reply."""

    step: int
    call_id: str | None
    tool: str
    arguments: JsonObject
    location: str
    response: JsonValue = None
    answered: bool = False


def _decode_content(content: Any) -> JsonValue:
    """Decode a ``tool``-role message body, which is a JSON string by convention.

    Kept lenient on purpose: if the body is not JSON we return the raw text instead
    of discarding it, so a text-only tool response still reaches stage 2 rather than
    vanishing between adapters.
    """
    if content is None:
        return None
    if isinstance(content, (dict, list, bool, int, float)):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _decode_arguments(raw: Any, *, tool: str, issues: list[IngestIssue], location: str) -> JsonObject:
    """Decode a ``function.arguments`` payload into an argument object.

    Arguments are the half of an invocation point that stage 4 replays, so a payload
    that is not an object is preserved under ``_raw`` and flagged rather than being
    silently reshaped into ``{}``.
    """
    if raw is None:
        return {}
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            issues.append(IngestIssue(
                kind="malformed_arguments",
                detail=f"arguments for {tool!r} are not valid JSON: {raw[:120]!r}",
                location=location,
            ))
            return {"_raw": raw}
    if isinstance(value, dict):
        return value
    issues.append(IngestIssue(
        kind="malformed_arguments",
        detail=f"arguments for {tool!r} decoded to {type(value).__name__}, not an object",
        location=location,
    ))
    return {"_raw": value}


def _coerce_outcome(value: JsonValue) -> bool | None:
    """Read the top-level ground-truth label; ``None`` keeps 'unlabeled' distinct from 'failed'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "pass", "passed", "success", "1"):
            return True
        if lowered in ("false", "fail", "failed", "failure", "0"):
            return False
    return None


def _to_message(raw: JsonObject, *, issues: list[IngestIssue], location: str) -> Message | None:
    """Convert one raw chat message into a contract :class:`Message`.

    Assistant turns that only carry ``tool_calls`` have ``content: None``; we keep
    them, because dropping them would make the transcript disagree with the
    invocation sequence recovered from it.
    """
    role = raw.get("role")
    if role not in VALID_ROLES:
        issues.append(IngestIssue(
            kind="unknown_message_role",
            detail=f"role {role!r} is not one of {sorted(VALID_ROLES)}",
            location=location,
        ))
        return None
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    tool_call_id = raw.get("tool_call_id")
    return Message(
        role=role,
        content=content,
        tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
    )


def _collect_tool_calls(
    raw: JsonObject,
    *,
    pending: list[_PendingCall],
    issues: list[IngestIssue],
    location: str,
) -> None:
    """Append every tool call declared by one assistant message to ``pending``.

    Step numbers are assigned here, from declaration order, which is what makes the
    recovered sequence line up with the span-ordered OTLP sequence for the same
    episode.
    """
    calls = raw.get("tool_calls")
    if not calls:
        return
    if not isinstance(calls, list):
        issues.append(IngestIssue(
            kind="malformed_tool_calls",
            detail=f"expected a list, got {type(calls).__name__}",
            location=location,
        ))
        return
    for index, call in enumerate(calls):
        where = f"{location}.tool_calls[{index}]"
        if not isinstance(call, dict):
            issues.append(IngestIssue(
                kind="malformed_tool_calls",
                detail=f"expected an object, got {type(call).__name__}",
                location=where,
            ))
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        tool = function.get("name") or call.get("name")
        if not isinstance(tool, str) or not tool.strip():
            issues.append(IngestIssue(
                kind="tool_call_missing_name",
                detail="assistant tool_call declares no function name",
                location=where,
            ))
            continue
        raw_args = function.get("arguments", call.get("arguments"))
        call_id = call.get("id")
        pending.append(_PendingCall(
            step=len(pending),
            call_id=call_id if isinstance(call_id, str) and call_id else None,
            tool=tool,
            arguments=_decode_arguments(raw_args, tool=tool, issues=issues, location=where),
            location=where,
        ))


def _match_result(
    raw: JsonObject,
    *,
    pending: list[_PendingCall],
    issues: list[IngestIssue],
    location: str,
) -> None:
    """Attach one ``tool``-role message to the call it answers.

    Implements the three pairing rules in the module docstring. Rules 2 and 3 are
    order-based guesses, so each records an issue: the corpus must show where a
    reconstruction was inferred rather than read.
    """
    response = _decode_content(raw.get("content"))
    call_id = raw.get("tool_call_id")

    if isinstance(call_id, str) and call_id:
        for call in pending:
            if call.call_id == call_id and not call.answered:
                call.response, call.answered = response, True
                return
        issues.append(IngestIssue(
            kind="orphan_tool_result",
            detail=f"tool message references unknown tool_call_id {call_id!r}",
            location=location,
        ))
        return

    name = raw.get("name")
    if isinstance(name, str) and name:
        for call in pending:
            if call.tool == name and not call.answered:
                call.response, call.answered = response, True
                issues.append(IngestIssue(
                    kind="recovered_by_name_and_order",
                    detail=(
                        f"tool result declared no tool_call_id; paired with the earliest "
                        f"unanswered {name!r} call at step {call.step}"
                    ),
                    location=location,
                ))
                return
        issues.append(IngestIssue(
            kind="orphan_tool_result",
            detail=f"no unanswered call named {name!r} to pair this result with",
            location=location,
        ))
        return

    for call in pending:
        if not call.answered:
            call.response, call.answered = response, True
            issues.append(IngestIssue(
                kind="recovered_by_order",
                detail=(
                    f"tool result declared neither tool_call_id nor name; paired with the "
                    f"earliest unanswered call {call.tool!r} at step {call.step}"
                ),
                location=location,
            ))
            return
    issues.append(IngestIssue(
        kind="orphan_tool_result",
        detail="tool result has no id, no name, and no unanswered call to pair with",
        location=location,
    ))


def _finalize(
    pending: list[_PendingCall],
    *,
    trace_id: str,
    issues: list[IngestIssue],
) -> list[InvocationPoint]:
    """Turn resolved pending calls into :class:`InvocationPoint` records.

    Status is inferred from the response body because a chat transcript carries no
    span status -- this is the one place the two adapters get to the same answer by
    different evidence, and it is why ``looks_like_error`` lives in
    :mod:`bandits.ingest.errors` next to the kind inference.
    """
    invocations: list[InvocationPoint] = []
    for step, call in enumerate(pending):
        if not call.answered:
            issues.append(IngestIssue(
                kind="tool_call_without_result",
                detail=f"no tool message answers call {call.call_id or call.tool!r}; response left null",
                location=call.location,
            ))
        status = CallStatus.ERROR if looks_like_error(call.response) else CallStatus.OK
        error_kind = infer_error_kind(call.response) if status is CallStatus.ERROR else None
        call_id = call.call_id
        if call_id is None:
            call_id = f"{trace_id}:{step}"
            issues.append(IngestIssue(
                kind="synthesized_call_id",
                detail=f"tool_call for {call.tool!r} declared no id; synthesized {call_id!r}",
                location=call.location,
            ))
        invocations.append(InvocationPoint(
            call_id=call_id,
            trace_id=trace_id,
            step=step,
            tool=call.tool,
            arguments=call.arguments,
            response=call.response,
            status=status,
            error_kind=error_kind,
        ))
    return invocations


def parse_chat_record(
    record: JsonObject,
    *,
    source_digest: str,
    location: str,
    fallback_trace_id: str,
) -> tuple[Trace | None, list[IngestIssue]]:
    """Normalize one chat-JSON record (one JSONL line) into a :class:`Trace`.

    A record with no ``messages`` list is unusable and returns ``None`` plus an
    issue: with no transcript there is nothing to recover, and a trace with zero
    invocations would masquerade as a legitimately tool-free episode.
    """
    issues: list[IngestIssue] = []
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        issues.append(IngestIssue(
            kind="missing_messages",
            detail=f"record has no 'messages' list (got {type(raw_messages).__name__})",
            location=location,
        ))
        return None, issues

    trace_id = fallback_trace_id
    for key in TRACE_ID_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            trace_id = value
            break

    messages: list[Message] = []
    pending: list[_PendingCall] = []
    for index, raw in enumerate(raw_messages):
        where = f"{location}:messages[{index}]"
        if not isinstance(raw, dict):
            issues.append(IngestIssue(
                kind="malformed_message",
                detail=f"expected an object, got {type(raw).__name__}",
                location=where,
            ))
            continue
        message = _to_message(raw, issues=issues, location=where)
        if message is not None:
            messages.append(message)
        if raw.get("role") == "assistant":
            _collect_tool_calls(raw, pending=pending, issues=issues, location=where)
        elif raw.get("role") == "tool":
            _match_result(raw, pending=pending, issues=issues, location=where)

    invocations = _finalize(pending, trace_id=trace_id, issues=issues)

    trace = Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest=source_digest,
        messages=tuple(messages),
        invocations=tuple(invocations),
        outcome=_coerce_outcome(record.get("outcome")),
        metadata={
            "episode_id": trace_id,
            "message_count": len(raw_messages),
            "location": location,
        },
    )
    return trace, issues


__all__ = [
    "TRACE_ID_KEYS",
    "parse_chat_record",
]
