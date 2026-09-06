"""Claude Code / Cursor session-log adapter.

One JSONL file is one session. Records carry ``type: "user"`` or
``"assistant"`` with Anthropic content blocks; a ``tool_use`` block is a tool
call and the ``tool_result`` block quoting its id is that call's result.

Everything else the format emits — queue operations, attachments, file-history
snapshots, titles — is bookkeeping about the session rather than about the
episode, and is skipped. Anything skipped for a reason worth knowing becomes an
issue rather than vanishing.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bandits.ingest.toolsets import parse_toolset
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue, UserTurn

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_WRAPPER_TAGS = (
    "local-command-caveat",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "system-reminder",
    "ide_opened_file",
    "ide_selection",
)
"""Harness scaffolding injected into the user turn, not anything a user asked for.

Left in, these dominate the text: every slash command carries the same caveat,
so a corpus of unrelated sessions groups into one large family whose shared
descriptor is the boilerplate rather than any task.
"""

_WRAPPER = re.compile(
    r"<(" + "|".join(_WRAPPER_TAGS) + r")>.*?</\1>|</?(" + "|".join(_WRAPPER_TAGS) + r")>",
    re.DOTALL,
)


_CONTEXT_KEYS = ("cwd", "model", "permissionMode", "version", "gitBranch")
"""Session configuration, spread across the init record and the turn records."""


def _session_context(records: list[dict]) -> tuple[object, str | None, dict]:
    """The toolset and configuration the session declared, where it declared them.

    A session log written by the CLI opens with an ``init`` record naming every
    tool the run was started with; a log exported without one says nothing about
    the toolset, and that stays unknown rather than being reconstructed from the
    calls that happen to appear.
    """
    init = next(
        (r for r in records if r.get("type") == "system" and r.get("subtype") == "init"),
        None,
    )
    system_prompt = (init or {}).get("systemPrompt")
    context: dict = {}
    for record in (init, *records):
        for key in _CONTEXT_KEYS:
            if isinstance(record, dict) and record.get(key) is not None:
                context.setdefault(key, record[key])
    return (
        parse_toolset((init or {}).get("tools")),
        system_prompt if isinstance(system_prompt, str) else None,
        context,
    )


def _timestamp(record: dict, ordinal: int) -> datetime:
    raw = record.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return _EPOCH + timedelta(seconds=ordinal)


def _blocks(record: dict) -> list[dict]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _text(blocks: list[dict]) -> str:
    """The human-authored part of a turn, with harness scaffolding removed."""
    joined = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return _WRAPPER.sub(" ", joined).strip()


def _unrepresentable(blocks: list[dict]) -> bool:
    """Whether a user record carries content this trace cannot put in a transcript.

    A ``tool_result`` is the harness answering the agent, and a text block that
    is nothing but harness scaffolding strips to empty — neither is a user turn.
    An image or a document is: the agent saw it, the transcript cannot show it,
    and dropping it silently is what lets an export teach the next action as an
    answer to a request nobody can read.
    """
    return any(block.get("type") not in ("tool_result", "text") for block in blocks)


def _is_error(block: dict) -> bool:
    """``is_error`` arrives as a real bool from some writers and as text from others."""
    raw = block.get("is_error")
    return raw is True or (isinstance(raw, str) and raw.strip().lower() == "true")


def _coerce(value: Any) -> Any:
    """Tool input is sometimes a dict and sometimes its repr. Keep whatever survives."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def load_claude_code(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> TraceCorpus:
    """Read a session log, or a directory of them, into one corpus.

    A directory is the useful case: one session is one episode, and families
    only mean anything across many of them.
    """
    if path.is_dir():
        return _load_directory(path, ruleset)
    return _load_session(path, ruleset)


def _load_directory(path: Path, ruleset: RedactionRuleset) -> TraceCorpus:
    traces = []
    issues: list[TraceIssue] = []
    for session in sorted(path.rglob("*.jsonl")):
        corpus = _load_session(session, ruleset)
        traces.extend(corpus.traces)
        issues.extend(corpus.issues)
    return TraceCorpus(
        source="claude-code",
        traces=tuple(traces),
        issues=tuple(issues),
        redaction_ruleset=ruleset.name,
    )


def _load_session(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> TraceCorpus:
    """Read one session log into a single-trace :class:`TraceCorpus`."""
    source = redact_source(path, ruleset)
    issues: list[TraceIssue] = list(source.issues)

    spans: list[Span] = []
    user_turns: list[UserTurn] = []
    unrepresented = 0
    pending: dict[str, str] = {}
    results: dict[str, dict] = {}
    task: str | None = None
    session_id: str | None = None
    ordinal = 0

    records: list[dict] = []
    for index, line in enumerate(source.data.decode("utf-8", "replace").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                TraceIssue(kind="malformed_json", detail=str(exc), location=f"{path}:{index + 1}")
            )
            continue
        if isinstance(record, dict):
            records.append(record)

    # Results are collected first: a tool_result appears in a later record than
    # the call it answers, and a call whose result never arrives must stay
    # unanswered rather than silently reading as an empty success.
    for record in records:
        for block in _blocks(record):
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                results[block["tool_use_id"]] = block

    for record in records:
        if record.get("type") not in ("user", "assistant"):
            continue
        session_id = session_id or record.get("sessionId")
        blocks = _blocks(record)
        started = _timestamp(record, ordinal)

        if record.get("type") == "user" and not record.get("isSidechain"):
            # Only turns a human actually wrote: a user record carrying nothing
            # but tool results is the harness answering the agent.
            text = _text(blocks)
            if text:
                task = task if task is not None else text
                user_turns.append(
                    UserTurn(text=text, after_span_id=spans[-1].span_id if spans else None)
                )
            if _unrepresentable(blocks):
                # Counted even when the record also carried text: the turn that
                # reaches a transcript is then only part of what the user sent.
                unrepresented += 1
                issues.append(
                    TraceIssue(
                        kind="unrepresentable_user_turn",
                        detail="user turn carries content that is not text: "
                        + ", ".join(
                            sorted(
                                {
                                    str(b.get("type"))
                                    for b in blocks
                                    if b.get("type") not in ("tool_result", "text")
                                }
                            )
                        ),
                        location=str(path),
                    )
                )

        for block in blocks:
            if block.get("type") == "text" and record.get("type") == "assistant":
                text = _text([block])
                if not text:
                    continue
                spans.append(
                    Span(
                        span_id=f"span-{ordinal}",
                        kind=SpanKind.MODEL,
                        name=record.get("message", {}).get("model") or "assistant",
                        started_at=started,
                        ended_at=started,
                        output=text,
                    )
                )
                ordinal += 1

            elif block.get("type") == "tool_use":
                call_id = block.get("id")
                span_id = f"span-{ordinal}"
                spans.append(
                    Span(
                        span_id=span_id,
                        kind=SpanKind.MODEL,
                        name=block.get("name") or "tool",
                        started_at=started,
                        ended_at=started,
                        arguments=_coerce(block.get("input")),
                        attributes={"tool_call": True},
                    )
                )
                if isinstance(call_id, str):
                    pending[call_id] = span_id
                ordinal += 1

                result = results.get(call_id) if isinstance(call_id, str) else None
                if result is None:
                    issues.append(
                        TraceIssue(
                            kind="unanswered_tool_call",
                            detail=f"{block.get('name')!r} has no recorded result",
                            location=f"{path}:{call_id}",
                        )
                    )
                    continue
                spans.append(
                    Span(
                        span_id=f"span-{ordinal}",
                        parent_span_id=span_id,
                        kind=SpanKind.TOOL,
                        name=block.get("name") or "tool",
                        started_at=started,
                        ended_at=started,
                        output=_coerce_result(result.get("content")),
                        status=SpanStatus.ERROR if _is_error(result) else SpanStatus.OK,
                        attributes={"synthetic_time": True},
                    )
                )
                ordinal += 1

    if task is None:
        issues.append(
            TraceIssue(
                kind="no_user_message",
                detail="session records no user instruction",
                location=str(path),
            )
        )
    if not spans:
        return TraceCorpus(
            source="claude-code",
            traces=(),
            issues=tuple(issues),
            redaction_ruleset=source.ruleset,
        )

    tools_available, system_prompt, context = _session_context(records)
    trace = Trace(
        trace_id=path.stem,
        source="claude-code",
        source_digest=source.source_digest,
        task=task,
        lineage_id=session_id,
        tools_available=tools_available,  # type: ignore[arg-type]
        system_prompt=system_prompt,
        runtime_context=context,
        user_turns=tuple(user_turns),
        unrepresented_user_turns=unrepresented,
        spans=tuple(spans),
    )
    return TraceCorpus(
        source="claude-code",
        traces=(trace,),
        issues=tuple(issues),
        redaction_ruleset=source.ruleset,
    )


def _coerce_result(content: Any) -> Any:
    """A tool result is JSON when it can be, and text when it genuinely is text."""
    if isinstance(content, list):
        content = "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    if not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return content
    return content
