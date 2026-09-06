"""Chat JSON adapter: the shape almost any agent framework can dump with zero
observability tooling — a plain list of ``{role, content, tool_calls}`` messages.

An assistant message with ``tool_calls`` becomes one MODEL span per requested
call; the following ``tool``-role message with a matching ``tool_call_id``
becomes that call's child TOOL span. Chat JSON carries no real timestamps, so
spans get sequential synthetic ones — ordinal order only, never treated as
measured latency, and tagged ``attributes={"synthetic_time": True}`` so nothing
downstream mistakes assigned order for real timing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bandits.ingest.toolsets import parse_toolset
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue, UserTurn

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


_LINEAGE_KEYS = ("session_id", "conversation_id", "thread_id", "lineage_id")

_SETTING_KEYS = ("model", "temperature", "top_p", "max_tokens", "scaffold", "agent_version")
"""Configuration a wrapper may declare around its messages, when it declares any."""


def _lineage_of(container: dict) -> str | None:
    return next(
        (
            container[key]
            for key in _LINEAGE_KEYS
            if isinstance(container.get(key), str) and container[key]
        ),
        None,
    )


def _settings_of(container: dict) -> dict:
    return {key: container[key] for key in _SETTING_KEYS if container.get(key) is not None}


def _conversations(payload: object) -> list[tuple[list[dict], str | None, dict]]:
    """A file is either one conversation (a bare message array) or a list of them.

    Each is paired with its declared session id and whatever else the wrapper
    declared around the messages — the offered toolset and the settings the run
    used. A bare message array has nowhere to declare any of that.
    """
    if (
        isinstance(payload, list)
        and payload
        and all(isinstance(item, dict) and "role" in item for item in payload)
    ):
        return [(payload, None, {})]
    if isinstance(payload, list):
        return [(c.get("messages", []), _lineage_of(c), c) for c in payload if isinstance(c, dict)]
    if isinstance(payload, dict):
        return [(payload.get("messages", []), _lineage_of(payload), payload)]
    return []


def _tool_result(content: object) -> object:
    """Parse a tool result that arrived as JSON text, keeping the text otherwise.

    Chat exports serialize tool returns into the ``content`` string. Leaving them
    as text would hide every structured field from outcome extraction, so a JSON
    object or array is parsed back; anything else is genuinely prose and is kept
    exactly as it came.
    """
    if not isinstance(content, str):
        return content
    stripped = content.strip()
    if not stripped or stripped[0] not in "{[":
        return content
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return content


def _tool_calls(message: dict) -> list[dict]:
    raw = message.get("tool_calls")
    return [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []


def _convert_conversation(
    messages: list[dict],
    *,
    trace_id: str,
    source_digest: str,
    lineage_id: str | None,
    wrapper: dict,
    location: str,
    issues: list[TraceIssue],
) -> Trace | None:
    # Every system and developer message, in order. Keeping only the first
    # dropped the developer instruction that narrowed it, which is the half that
    # usually carries the constraint the run was actually held to.
    declared = [
        m["content"]
        for m in messages
        if m.get("role") in ("system", "developer") and isinstance(m.get("content"), str)
    ]
    system_prompt = "\n\n".join(declared) if declared else None

    spans: list[Span] = []
    user_turns: list[UserTurn] = []
    unrepresented = 0
    pending_calls: dict[
        str, str
    ] = {}  # tool_call_id -> span_id of the MODEL span that requested it
    ordinal = 0

    for index, message in enumerate(messages):
        role = message.get("role")
        timestamp = _EPOCH + timedelta(seconds=ordinal)

        if role in ("user", "human"):
            content = message.get("content")
            if isinstance(content, str):
                # Anchored to the last span, because a correction is only an
                # instruction for what comes after it.
                user_turns.append(
                    UserTurn(text=content, after_span_id=spans[-1].span_id if spans else None)
                )
            else:
                # Content blocks, an attachment, or nothing at all. Counted, not
                # skipped: an export that drops one of these teaches the next
                # action as an answer to a request the transcript never shows.
                unrepresented += 1
                issues.append(
                    TraceIssue(
                        kind="unrepresentable_user_turn",
                        detail=f"message {index}: user content is "
                        f"{type(content).__name__}, not text",
                        location=location,
                    )
                )
            continue

        if role == "assistant":
            calls = _tool_calls(message)
            if calls:
                for call in calls:
                    function = (
                        call.get("function") if isinstance(call.get("function"), dict) else call
                    )
                    name = function.get("name")
                    call_id = call.get("id") or call.get("tool_call_id")
                    if not isinstance(name, str) or not name:
                        issues.append(
                            TraceIssue(
                                kind="tool_call_missing_name",
                                detail=f"message {index} has a tool call with no readable name",
                                location=location,
                            )
                        )
                        continue
                    span_id = f"{trace_id}:span-{ordinal}"
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    spans.append(
                        Span(
                            span_id=span_id,
                            parent_span_id=None,
                            kind=SpanKind.MODEL,
                            name=name,
                            started_at=timestamp,
                            ended_at=timestamp,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            attributes={"synthetic_time": True},
                        )
                    )
                    if isinstance(call_id, str) and call_id:
                        pending_calls[call_id] = span_id
                    ordinal += 1
            else:
                content = message.get("content")
                span_id = f"{trace_id}:span-{ordinal}"
                spans.append(
                    Span(
                        span_id=span_id,
                        parent_span_id=None,
                        kind=SpanKind.MODEL,
                        name="assistant",
                        started_at=timestamp,
                        ended_at=timestamp,
                        output=content,
                        attributes={"synthetic_time": True},
                    )
                )
                ordinal += 1

        elif role == "tool":
            call_id = message.get("tool_call_id")
            parent_span_id = pending_calls.get(call_id) if isinstance(call_id, str) else None
            name = message.get("name") or "tool"
            span_id = f"{trace_id}:span-{ordinal}"
            spans.append(
                Span(
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    kind=SpanKind.TOOL,
                    name=name,
                    started_at=timestamp,
                    ended_at=timestamp,
                    output=_tool_result(message.get("content")),
                    status=SpanStatus.OK,
                    attributes={"synthetic_time": True},
                )
            )
            if parent_span_id is None:
                issues.append(
                    TraceIssue(
                        kind="unpaired_tool_result",
                        detail=f"message {index}: no matching assistant tool_call for tool_call_id {call_id!r}",
                        location=location,
                    )
                )
            ordinal += 1

    # The instruction is the first turn that reached us as text. A conversation
    # whose only user content is an image has no readable instruction, and is
    # refused here rather than raising on a task that is not a string.
    task = next((turn.text for turn in user_turns if turn.text), None)
    if task is None:
        issues.append(
            TraceIssue(
                kind="no_user_message",
                detail=(
                    "conversation has no readable user instruction"
                    if unrepresented
                    else "conversation has no user message"
                ),
                location=location,
            )
        )
        return None

    if not spans:
        issues.append(
            TraceIssue(
                kind="empty_conversation",
                detail="conversation has no assistant or tool message",
                location=location,
            )
        )
        return None

    return Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest=source_digest,
        task=task,
        lineage_id=lineage_id,
        # Keyed on presence, not truth: an explicitly empty `tools` is falsey,
        # and reading it as absent turns "no tools were offered" back into
        # "nobody said".
        tools_available=parse_toolset(
            wrapper["tools"] if "tools" in wrapper else wrapper.get("functions")
        ),
        system_prompt=system_prompt,
        runtime_context=_settings_of(wrapper),
        user_turns=tuple(user_turns),
        unrepresented_user_turns=unrepresented,
        spans=tuple(spans),
    )


def load_chat_json(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> TraceCorpus:
    """Read one chat-JSON export (one conversation or an array of them) into a
    :class:`~bandits.traces.TraceCorpus`."""
    source = redact_source(path, ruleset)
    raw = source.data
    source_digest = source.source_digest

    issues: list[TraceIssue] = list(source.issues)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        issues.append(TraceIssue(kind="malformed_json", detail=str(exc), location=str(path)))
        return TraceCorpus(
            source="chat-json",
            traces=(),
            issues=tuple(issues),
            redaction_ruleset=source.ruleset,
        )

    conversations = _conversations(payload)
    traces: list[Trace] = []
    for index, (messages, lineage_id, wrapper) in enumerate(conversations):
        location = f"{path}[{index}]"
        trace = _convert_conversation(
            messages,
            trace_id=f"{path.stem}-{index}",
            source_digest=source_digest,
            lineage_id=lineage_id,
            wrapper=wrapper,
            location=location,
            issues=issues,
        )
        if trace is not None:
            traces.append(trace)

    return TraceCorpus(
        source="chat-json",
        traces=tuple(traces),
        issues=tuple(issues),
        redaction_ruleset=source.ruleset,
    )
