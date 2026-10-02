"""Canonical trace model.

A trace is an ordered set of spans. Every ingest adapter's only job is to produce
that list correctly for its source format; nothing downstream ever looks at a raw
export again once it has been turned into a ``TraceCorpus``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

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

    call_recorded: bool = True
    """Whether the source paired this tool result with a call the agent made.

    False marks a result the source left orphaned: no assistant call it answers,
    so nothing recorded that the agent ever asked for it. The result is kept —
    it happened, and analysis may still read it — but a transcript rebuilt from
    it would have to invent the call that produced it, and a demonstration built
    that way teaches an action inferred only from its own outcome.

    True is what every other span means. A MODEL span is itself the record of
    what the model did, and a TOOL span the source declares as an executed call —
    an OTLP ``execute_tool`` span carrying its own arguments — records the
    action, not merely its result. Only an adapter that can watch a result
    arrive with nothing to answer may set this False.
    """

    attributes: dict[str, Any] = Field(default_factory=dict)
    """Anything else the source declared that doesn't have a dedicated field."""


class ToolSchema(Contract):
    """One tool as it was offered to the agent, not as it was called."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    """The parameter schema the source declared. None means the tool was named
    without a definition, so a call to it may not be reproducible anywhere."""

    output_schema: dict[str, Any] | bool | None = None
    """JSON Schema for the tool result, when the source declared one.

    None means validation is unavailable, never that an arbitrary result was
    validated. Boolean schemas are retained because they are valid JSON Schema,
    including ``false`` for a tool that can produce no valid JSON instance.
    Kept separate from the input parameter schema.
    """

    def offered_projection(self) -> dict[str, Any]:
        """The published shape: what the source declared the agent could call.

        D68. Explicit rather than ``model_dump()``, so a field added to this
        contract cannot reach an export, an eval case, or hashed analysis
        evidence by default. Nulls are kept (``description`` has always
        shipped), which is why this is a projection and not ``exclude_none``.
        Extending the published shape is a versioned migration (D69).
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def simulation_projection(self) -> dict[str, Any]:
        """The offered shape plus the declared result schema.

        Only the simulator needs this: ``output_schema`` is what lets a
        proposed tool result be validated (D63) rather than taken on trust.
        It stays out of the published shape above.
        """
        return {**self.offered_projection(), "output_schema": self.output_schema}


class UserTurn(Contract):
    """One user message, and the point in the trajectory it arrived at."""

    text: str
    after_span_id: str | None = None
    """The span this turn followed. None means it opened the episode."""

    origin: Literal["recorded", "declared"] = "recorded"
    """``recorded``: the source recorded a person saying this. ``declared``: a
    workflow request that ingest was told (``--request-origin human``) came from
    a person; the source itself recorded it as the invocation's input, not as a
    message. Kept apart so a declared request is never mistaken for a recorded
    conversation turn."""


class WorkflowDeclaration(Contract):
    """What was declared at ingest about a workflow source. Never inferred.

    Stored on the corpus so every artifact built from it carries the semantics
    it was built under; a corpus ingested with other selectors is a different
    artifact, not the same one reread.
    """

    task_fields: tuple[str, ...] = ()
    """Paths into the invocation record (``input.query``, ``input.payload.query``);
    the first that resolves supplies the task. Empty: the task stays unresolved."""

    delivered_field: str | None = None
    """Path into the invocation record holding what was delivered (``output.answer``)."""

    request_origin: Literal["human", "machine", "unknown"] = "unknown"
    """Who started the runs. A workflow has no human follow-ups inside a run; that
    says nothing about who started it."""

    derivation_version: int = 1
    """Bumped whenever how requests, nodes or links are derived changes."""


class WorkflowRequest(Contract):
    """The application invocation of a workflow episode: what came in, what went out.

    Distinct from the structural root (an exporter's container span) and from
    the task (the request inside the input). Kept even when no task resolves,
    so analysis can still read the recorded input.
    """

    source_span_id: str | None
    """The invocation span. None when no candidate or several did (see ``invocation_basis``)."""

    invocation_basis: str
    """Why this span is the invocation, or why none was chosen."""

    candidate_span_ids: tuple[str, ...] = ()
    raw_input: Any = None
    raw_output: Any = None
    status: SpanStatus | None = None
    """Recorded status of the invocation span; None when no invocation was selected."""

    task: str | None = None
    task_status: Literal["declared", "unresolved", "conflict"] = "unresolved"
    task_path: str | None = None
    task_reason: str | None = None
    """Why the task is unresolved or in conflict; None when declared."""

    origin: Literal["human", "machine", "unknown"] = "unknown"

    delivered: Any = None
    """The value at the declared delivered field, as recorded; None if undeclared or absent."""


class WorkflowNode(Contract):
    """A recorded workflow step that contains model calls: structure, not an action.

    Nobody chose to call it the way a model chooses a tool; the program ran it.
    It is kept so a model call's enclosing result and execution round survive,
    and it never enters the action/reaction sequence.
    """

    span_id: str
    parent_span_id: str | None = None
    name: str
    started_at: datetime
    ended_at: datetime
    input: Any = None
    output: Any = None
    status: SpanStatus = SpanStatus.OK
    """Whether the source recorded the step as failed. A failed step is not a
    successful one with an odd output."""

    framework: dict[str, Any] = Field(default_factory=dict)
    """Framework metadata the source recorded on the node (e.g. ``langgraph_step``),
    lifted out for convenience; the same keys remain in ``attributes``."""

    attributes: dict[str, Any] = Field(default_factory=dict)
    """Every attribute the source declared on the step (status messages, levels,
    custom metadata), except the input/output values already in ``input``/``output``."""


class EvidenceLink(Contract):
    """One recorded relationship involving a model call.

    ``input_context`` points to an earlier pipeline result whose text appears
    in the call input; other kinds describe structure or later observations.

    A link says what the record supports and on what basis — never that one
    thing caused another. ``ambiguous`` marks a match that could as well belong
    elsewhere; unknown relationships are simply absent.
    """

    call_span_id: str
    kind: Literal[
        "tool_result", "enclosing_result", "text_match", "shared_result", "same_round", "delivery",
        "input_context",
    ]
    target_span_id: str | None = None
    """The span or node the call links to; None for ``delivery`` (the request record).
    For ``input_context``, the target is the earlier source of recorded input text."""

    basis: str
    match_chars: int | None = None
    ambiguous: bool = False
    shared_with: tuple[str, ...] = ()
    """Other calls whose output the same target also contains (``shared_result``).

    Overlap, not contribution: a framework that passes its whole state forward
    repeats every earlier output in every later input. ``shared_result`` says the
    texts appear there together — never that these calls caused that result."""


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

    task_source: str | None = None
    """Where ``task`` was read from and on what basis, when an adapter chose it
    from candidates rather than a declared field. None means no such choice."""

    interaction: Literal["conversation", "workflow"] = "conversation"
    """How the episode was driven, as declared at ingest — never inferred.

    ``conversation``: a person and the agent take turns, so a user-role message
    is someone speaking. ``workflow``: a program called the models, building
    each prompt itself, and nobody spoke inside the run; a user-role message in
    a model's input is the program's instruction to that call. Workflow says
    nothing about who *started* the run — that is ``request.origin``. Reading a
    workflow as a conversation turns every later prompt into a person reacting
    to the call before it.
    """

    request: WorkflowRequest | None = None
    """Workflow only: the application invocation (its recorded input and output)."""

    workflow_nodes: tuple[WorkflowNode, ...] = ()
    """Workflow only: recorded steps containing model calls, kept as structure."""

    evidence: tuple[EvidenceLink, ...] = ()
    """Workflow only: recorded relationships from each model call to later records."""

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

    workflow: WorkflowDeclaration | None = None
    """Set when the corpus was ingested as a workflow, with what was declared."""

    redaction_ruleset: str | None = None
    """Which redaction ruleset produced these bytes.

    Recorded because the same source file under a changed ruleset yields a
    different corpus, and without this there would be nothing to explain why two
    corpora sharing a ``source_digest`` do not match.
    """

    control_markers: tuple[str, ...] = ()
    """Literal tokens the exporting system writes into a message's own text.

    A fact about this specific corpus, declared once at ingest — never
    inferred from ``source``, which names a generic adapter (``chat-json``)
    shared by many unrelated exports and cannot distinguish one benchmark's
    scaffolding from another's. tau2's simulator, for one, appends
    ``###TRANSFER###`` to a user turn's own text on most airline episodes,
    not only the ones that actually escalate — a miner shown it would read
    that token as evidence about what the user asked for. Empty unless the
    ingest that produced this corpus said otherwise: a generic corpus carries
    no assumption that it needs sanitizing.
    """
