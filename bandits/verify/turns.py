"""Cut a trace into turns: one action, and everything that reacted to it.

The unit a next-state verifier reads. Each MODEL span is an action; the tool
results and user messages that arrive before the next MODEL span are the
environment's reaction to it. The last action of an episode usually has no
reaction — nothing followed it — and that is recorded as *unobserved*, never
scored as if silence were approval.

Source-agnostic on purpose. A support transcript, a coding session and a GUI
run all reduce to the same sequence; what differs is how a reader interprets
the reactions, and that is the archetype's job, not this module's.
"""

from __future__ import annotations

import json
from typing import Any

from bandits.traces import Contract, SpanKind, SpanStatus, Trace

ACTION_CHARS = 1600
REACTION_CHARS = 1200
"""Per-turn caps. Head and tail are both kept: an execution log's verdict is
usually at the end, and a tool result's error at the start."""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}…[{len(text) - limit} chars omitted]…{text[-tail:]}"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def render_action(output: Any) -> str:
    """An assistant output as one readable block: its text, then any calls."""
    if isinstance(output, dict) and "tool_calls" in output:
        lines = []
        content = output.get("content")
        if content:
            lines.append(_as_text(content))
        for call in output.get("tool_calls") or []:
            if isinstance(call, dict):
                lines.append(f"→ {call.get('name')}({_as_text(call.get('arguments'))})")
        return "\n".join(lines)
    return _as_text(output)


class Reaction(Contract):
    """One thing the environment or user did in response to an action."""

    span_id: str | None
    kind: str
    """``tool`` for a tool result, ``user`` for a user message."""

    name: str
    text: str
    error: bool = False


class Turn(Contract):
    trace_id: str
    index: int
    action_span_id: str
    action: str
    reactions: tuple[Reaction, ...] = ()

    @property
    def observed(self) -> bool:
        return bool(self.reactions)

    @property
    def errored(self) -> bool:
        return any(reaction.error for reaction in self.reactions)

    def next_state(self, limit: int = REACTION_CHARS) -> str | None:
        """The reaction as one block, or None when nothing followed."""
        if not self.reactions:
            return None
        per = max(200, limit // max(1, len(self.reactions)))
        lines = []
        for reaction in self.reactions:
            flag = " [ERROR]" if reaction.error else ""
            lines.append(f"[{reaction.kind}:{reaction.name}]{flag} {_clip(reaction.text, per)}")
        # `per`'s 200-char floor is there so a turn with many reactions does
        # not reduce every one of them to nothing; it means the per-reaction
        # clips alone do not bound the total. This does.
        return _clip("\n".join(lines), limit)

    def as_dict(self, *, task: str | None = None) -> dict[str, Any]:
        """The turn as plain data, for a predicate or a prompt to read."""
        return {
            "trace_id": self.trace_id,
            "index": self.index,
            "task": task,
            "action": self.action,
            "next_state": self.next_state(),
            "reactions": [
                {"kind": r.kind, "name": r.name, "text": r.text, "error": r.error}
                for r in self.reactions
            ],
            "observed": self.observed,
            "errored": self.errored,
        }


def extract_turns(trace: Trace) -> tuple[Turn, ...]:
    """Every action in the trace with the reactions that followed it.

    Tool results before the first model call have no action to answer; they
    are dropped rather than attached to a turn that did not cause them. User
    turns are placed by ``after_span_id`` and count as reactions to the action
    they followed.
    """
    user_after: dict[str | None, list[str]] = {}
    for turn in trace.user_turns:
        user_after.setdefault(turn.after_span_id, []).append(turn.text)

    turns: list[Turn] = []
    current: Turn | None = None
    reactions: list[Reaction] = []

    def close() -> None:
        nonlocal current, reactions
        if current is not None:
            turns.append(current.replace(reactions=tuple(reactions)))
        current = None
        reactions = []

    for span in trace.spans:
        if span.kind is SpanKind.MODEL:
            close()
            current = Turn(
                trace_id=trace.trace_id,
                index=len(turns),
                action_span_id=span.span_id,
                action=_clip(render_action(span.output), ACTION_CHARS),
            )
        elif current is not None:
            reactions.append(
                Reaction(
                    span_id=span.span_id,
                    kind="tool",
                    name=span.name,
                    text=_clip(_as_text(span.output), REACTION_CHARS),
                    error=span.status is SpanStatus.ERROR,
                )
            )
        for text in user_after.pop(span.span_id, ()):
            if current is not None:
                reactions.append(
                    Reaction(
                        span_id=span.span_id,
                        kind="user",
                        name="user",
                        text=_clip(text, REACTION_CHARS),
                    )
                )
    close()
    return tuple(turns)


def turns_by_trace(traces: tuple[Trace, ...]) -> dict[str, tuple[Turn, ...]]:
    return {trace.trace_id: extract_turns(trace) for trace in traces}
