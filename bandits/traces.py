"""Canonical trace model.

A trace is an ordered set of spans. Every ingest adapter's only job is to produce
that list correctly for its source format; nothing downstream ever looks at a raw
export again once it has been turned into a ``TraceCorpus``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    """Base for every model in this module: immutable, no silent extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def replace(self, **updates: Any) -> Self:
        """Copy with changes, re-running validators.

        ``model_copy`` skips validation entirely, so every invariant these models
        declare would hold at construction and then quietly stop holding the
        first time a correction edited one.
        """
        return type(self).model_validate({**self.model_dump(), **updates})


class SpanKind(str, Enum):
    MODEL = "model"
    """One call to a language model."""

    TOOL = "tool"
    """One tool call."""


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class Span(Contract):
    """One node in a trace: either a model call or a tool call."""

    span_id: str
    parent_span_id: str | None = None
    """None marks the root span of the trace."""

    kind: SpanKind
    name: str
    """Tool name for a TOOL span; model name for a MODEL span."""

    started_at: datetime
    ended_at: datetime
    status: SpanStatus = SpanStatus.OK
    arguments: dict[str, Any] = Field(default_factory=dict)
    """Tool call arguments, or the prompt/input for a model call."""

    output: Any = None
    """Tool response, or the completion text for a model call."""

    attributes: dict[str, Any] = Field(default_factory=dict)
    """Anything else the source declared that doesn't have a dedicated field."""


class ToolSchema(Contract):
    """One tool as it was offered to the agent, not as it was called."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    """The parameter schema the source declared. None means the tool was named
    without a definition, so a call to it may not be reproducible anywhere."""


class UserTurn(Contract):
    """One user message, and the point in the trajectory it arrived at."""

    text: str
    after_span_id: str | None = None
    """The span this turn followed. None means it opened the episode."""


class TraceIssue(Contract):
    """One source record that could not be normalized. Never silently dropped."""

    kind: str
    detail: str
    location: str | None = None


class Trace(Contract):
    """One normalized agent episode."""

    trace_id: str
    source: str
    """Declared adapter name that produced this trace, e.g. 'otlp'."""

    source_digest: str
    """sha256 hex of the exact source bytes this trace came from."""

    task: str | None = None
    """The user-facing instruction, when the source declares one."""

    lineage_id: str | None = None
    """Session, ticket, or retry chain this episode belongs to.

    Traces sharing one must never straddle a fit/held-out split: a retry of the
    same request on both sides of the boundary leaks the answer across it. None
    means the source declared no grouping, which is recorded rather than assumed
    to mean independence.
    """

    tools_available: tuple[ToolSchema, ...] | None = None
    """The toolset offered at the start of the episode, when the source says.

    Not the same fact as the tools this episode called: which tool to reach for,
    out of what was on offer, is most of the decision a demonstration is meant to
    teach, and a row showing a call carries neither the alternatives nor the
    schema to reproduce it. None means the source declared no toolset, which is
    recorded as unknown — never as an empty toolset.
    """

    system_prompt: str | None = None
    """The system or developer instruction the episode ran under, when recorded."""

    runtime_context: dict[str, Any] = Field(default_factory=dict)
    """Configuration the episode ran under: model, sampling settings, working
    directory, scaffold version. Empty means the source declared none."""

    user_turns: tuple[UserTurn, ...] = ()
    """Every user message the source recorded, in order, with its position.

    A conversation is not one instruction followed by a monologue: a correction
    or an approval halfway through is why the rest of the episode looks the way
    it does. Empty means the source was not read for turns at all, not that the
    episode had one — ``task`` still carries the opening instruction either way.
    """

    unrepresented_user_turns: int = 0
    """User messages the source recorded and this trace could not represent.

    Above zero, any transcript rebuilt from this trace omits something the agent
    was actually told, so it must be refused rather than exported as if the
    later actions answered only the first instruction.
    """

    spans: tuple[Span, ...]


class TraceCorpus(Contract):
    """The normalized result of one ingest run."""

    source: str
    traces: tuple[Trace, ...]
    issues: tuple[TraceIssue, ...] = ()

    redaction_ruleset: str | None = None
    """Which redaction ruleset produced these bytes.

    Recorded because the same source file under a changed ruleset yields a
    different corpus, and without this there would be nothing to explain why two
    corpora sharing a ``source_digest`` do not match.
    """
