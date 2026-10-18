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

from bandits.ingest.records import read_records
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


_ROLES = {
    "user": "user", "human": "user",
    "assistant": "assistant", "ai": "assistant", "gpt": "assistant", "bot": "assistant",
    "model": "assistant",
    "system": "system", "developer": "developer",
    "tool": "tool", "function": "tool", "observation": "tool",
}  # fmt: skip
"""Role names exports use for the five roles Bandits reads (OpenAI, ShareGPT, Gemini, LangChain)."""
_ROLE_KEYS = ("role", "from", "speaker")
_TYPE_ROLES = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
"""LangChain spells the role as ``type``. ``user`` and ``assistant`` are left out on purpose: a
Claude Code event is ``{"type": "user", ...}`` and must not be taken for a chat message."""
_WRAPPER_KEYS = (
    "messages",
    "conversations",
    "conversation",
    "turns",
    "contents",
    "chat",
    "history",
)
"""Where a wrapper object keeps its list of messages."""
_TEXT_TYPES = (None, "text", "input_text", "output_text")


def _role_of(message: dict) -> tuple[str | None, str | None]:
    """The role as Bandits names it, and the role as the export wrote it (None if it has none)."""
    for key in _ROLE_KEYS:
        raw = message.get(key)
        if isinstance(raw, str):
            return _ROLES.get(raw.strip().lower()), raw
    raw = message.get("type")
    if isinstance(raw, str) and raw in _TYPE_ROLES and "sessionId" not in message:
        return _TYPE_ROLES[raw], raw
    return None, None


def _is_messages(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and _role_of(item)[1] is not None for item in value)
    )


def messages_key(container: dict) -> str | None:
    """The key a wrapper keeps its messages under, if it keeps them under one."""
    return next((k for k in _WRAPPER_KEYS if _is_messages(container.get(k))), None)


def _conversations(
    payload: object, issues: list[TraceIssue], location: str = ""
) -> list[tuple[list[dict], str | None, dict]]:
    """Every shape a chat export commonly takes, each reduced to conversations.

    - a bare message array: one conversation
    - ``{"messages": [...], ...}`` (or ``conversations``, ``turns``, ``contents``...): one
      conversation, the rest being its wrapper
    - an array of either of those, or an array of message arrays
    - JSON lines (``payload`` is then the list of lines): one of the above per
      line, or one message per line, which is one conversation

    Each is paired with its declared session id and whatever else the wrapper
    declared around the messages — the offered toolset and the settings the run
    used. A bare message array has nowhere to declare any of that. An item that
    fits none of these is an issue, never a silent skip.
    """
    found: list[tuple[list[dict], str | None, dict]] = []

    def visit(item: object, where: str, nested: bool) -> None:
        if _is_messages(item):
            found.append((item, None, {}))  # type: ignore[arg-type]
        elif isinstance(item, dict):
            key = messages_key(item)
            found.append((item[key] if key else item.get("messages", []), _lineage_of(item), item))
        elif isinstance(item, list) and not nested:
            for index, inner in enumerate(item):
                visit(inner, f"{where}[{index}]", True)
        else:
            issues.append(
                TraceIssue(
                    kind="unrecognized_record",
                    detail=f"{where}: expected a message array or an object with "
                    f"messages, found {type(item).__name__}",
                    location=location,
                )
            )

    if _is_messages(payload):
        return [(payload, None, {})]  # type: ignore[list-item]
    visit(payload, location or "$", False)
    return found


def _text_of(content: object) -> object:
    """A message or tool-result body as text when it is made of text blocks, else as it came."""
    if not isinstance(content, list):
        return content
    texts = [
        b if isinstance(b, str) else b.get("text")
        for b in content
        if isinstance(b, str) or (isinstance(b, dict) and b.get("type") in _TEXT_TYPES)
    ]
    if all(isinstance(t, str) for t in texts) and len(texts) == len(content):
        return "\n".join(texts)
    return content


def _blocks(sequence: list) -> dict:
    """Split content blocks (OpenAI parts, Anthropic blocks, Gemini parts) by what they are."""
    out: dict = {"texts": [], "calls": [], "results": [], "thinking": [], "other": False}
    for block in sequence:
        if isinstance(block, str):
            out["texts"].append(block)
        elif not isinstance(block, dict):
            out["other"] = True
        elif block.get("type") == "tool_use":
            out["calls"].append(
                {
                    "id": block.get("id"),
                    "function": {"name": block.get("name"), "arguments": block.get("input")},
                }
            )
        elif block.get("type") == "tool_result":
            out["results"].append(
                {
                    "tool_call_id": block.get("tool_use_id"),
                    "content": _text_of(block.get("content")),
                }
            )
        elif isinstance(block.get("functionCall"), dict):
            call = block["functionCall"]
            out["calls"].append(
                {
                    "id": None,
                    "_by_name": True,
                    "function": {"name": call.get("name"), "arguments": call.get("args")},
                }
            )
        elif isinstance(block.get("functionResponse"), dict):
            reply = block["functionResponse"]
            out["results"].append(
                {
                    "name": reply.get("name"),
                    "_by_name": True,
                    "content": json.dumps(reply.get("response")),
                }
            )
        elif block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
            out["thinking"].append(block["thinking"])
        elif block.get("type") == "redacted_thinking":
            continue
        elif block.get("type") in _TEXT_TYPES and isinstance(block.get("text"), str):
            out["texts"].append(block["text"])
        else:
            out["other"] = True
    return out


def _normalize(messages: list, issues: list[TraceIssue], location: str) -> list[dict]:
    """Every message as the OpenAI-style ``{role, content, tool_calls}`` the conversion reads.

    Roles and keys other exports use are mapped (ShareGPT ``from``/``value``, LangChain
    ``type``, Gemini ``role: model`` with ``parts``); Anthropic content blocks become text,
    ``tool_calls`` and ``tool`` messages. A message that cannot be mapped is an issue, never
    a silent drop. ``_index`` keeps the message's place in the export for issue details.
    """
    out: list[dict] = []
    if not isinstance(messages, list):
        issues.append(
            TraceIssue(
                kind="unrecognized_record",
                detail=f"the messages are a {type(messages).__name__}, not a list",
                location=location,
            )
        )
        return out
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            issues.append(
                TraceIssue(
                    kind="unrecognized_record",
                    detail=f"message {index} is a {type(message).__name__}, not an object",
                    location=location,
                )
            )
            continue
        role, raw = _role_of(message)
        if raw is None:
            issues.append(
                TraceIssue(
                    kind="unrecognized_record",
                    detail=f"message {index} has no role",
                    location=location,
                )
            )
            continue
        if role is None:
            issues.append(
                TraceIssue(
                    kind="unrecognized_role",
                    detail=f"message {index} has the role {raw[:40]!r}, which Bandits does not read",
                    location=location,
                )
            )
            continue
        by_name = raw.strip().lower() == "function"
        content = next((message[k] for k in ("content", "value", "text") if k in message), None)
        sequence = content if isinstance(content, list) else message.get("parts")
        plain = {
            **message,
            "role": role,
            "content": _text_of(content) if sequence is None else content,
            "_index": index,
        }
        if isinstance(message.get("function_call"), dict) and not message.get("tool_calls"):
            plain["tool_calls"] = [
                {"id": None, "_by_name": True, "function": message["function_call"]}
            ]
        if by_name:
            plain["_by_name"] = True
        if not isinstance(sequence, list):
            out.append(plain)
            continue
        parts = _blocks(sequence)
        text = "\n".join(parts["texts"]) if parts["texts"] else None
        for result in parts["results"]:
            out.append({**result, "role": "tool", "_index": index})
        if role == "assistant":
            calls = [*(message.get("tool_calls") or []), *parts["calls"]]
            out.append(
                {
                    **plain,
                    "content": text,
                    "tool_calls": calls,
                    "_thinking": "\n".join(parts["thinking"]) or None,
                }
            )
        elif parts["results"] and not (parts["texts"] or parts["other"]):
            continue
        elif parts["other"] and role == "user":
            out.append({**plain, "content": sequence})
        else:
            out.append({**plain, "content": text})
    return out


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
    messages = _normalize(messages, issues, location)
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
    pending_calls: dict[str, tuple[str, str]] = {}
    """tool_call_id -> (span_id, tool name) for a call awaiting one result."""
    pending_by_name: dict[str, list[str]] = {}
    """Calls with no id (the legacy ``function_call``, Gemini's ``functionCall``), by tool name:
    their result names the function instead of an id, and answers the oldest one."""
    ordinal = 0

    for position, message in enumerate(messages):
        index = message.get("_index", position)
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
            text = message.get("content")
            thinking = {"thinking": message["_thinking"]} if message.get("_thinking") else {}
            if calls:
                if isinstance(text, str) and text.strip():
                    # What the assistant said before its calls is a span of its own, ahead of
                    # them, as the Claude Code reader records it. It is part of the turn.
                    spans.append(
                        Span(
                            span_id=f"{trace_id}:span-{ordinal}",
                            parent_span_id=None,
                            kind=SpanKind.MODEL,
                            name="assistant",
                            started_at=timestamp,
                            ended_at=timestamp,
                            output=text,
                            attributes={"synthetic_time": True, **thinking},
                        )
                    )
                    ordinal += 1
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
                        pending_calls[call_id] = (span_id, name)
                    elif call.get("_by_name"):
                        pending_by_name.setdefault(name, []).append(span_id)
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
                        attributes={"synthetic_time": True, **thinking},
                    )
                )
                ordinal += 1

        elif role == "tool":
            call_id = message.get("tool_call_id")
            pending = pending_calls.pop(call_id, None) if isinstance(call_id, str) else None
            declared_name = message.get("name")
            if pending is None and message.get("_by_name") and isinstance(declared_name, str):
                waiting = pending_by_name.get(declared_name)
                if waiting:
                    pending = (waiting.pop(0), declared_name)
            name_mismatch = (
                pending is not None
                and isinstance(declared_name, str)
                and bool(declared_name)
                and declared_name != pending[1]
            )
            parent_span_id = pending[0] if pending is not None and not name_mismatch else None
            name = pending[1] if pending is not None else declared_name or "tool"
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
                    # A result whose call this file never recorded. Kept, so the
                    # observation stays available to analysis, and marked, so an
                    # export cannot reach backwards from it to a call the
                    # assistant was never shown to have made.
                    call_recorded=parent_span_id is not None,
                    attributes={"synthetic_time": True},
                )
            )
            if parent_span_id is None:
                detail = (
                    f"message {index}: tool result names {declared_name!r}, but matching "
                    f"assistant tool_call names {pending[1]!r}"
                    if name_mismatch
                    else f"message {index}: no matching assistant tool_call for "
                    f"tool_call_id {call_id!r}"
                )
                issues.append(
                    TraceIssue(
                        kind="mismatched_tool_result" if name_mismatch else "unpaired_tool_result",
                        detail=detail,
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
    read = read_records(raw, str(path))
    issues.extend(read.issues)
    if not read.records:
        if not read.issues:
            issues.append(
                TraceIssue(kind="empty_file", detail="the file has no content", location=str(path))
            )
        return TraceCorpus(
            source="chat-json",
            traces=(),
            issues=tuple(issues),
            redaction_ruleset=source.ruleset,
        )
    payload = read.records[0].value if read.layout == "document" else read.values

    conversations = _conversations(payload, issues, str(path))
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
