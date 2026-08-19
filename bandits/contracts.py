"""Shared data contracts for the whole pipeline.

Every stage reads and writes these types and nothing else. This file is the
integration spine: if two stages disagree about a field here, they disagree
everywhere. Treat it as frozen unless the change is coordinated.

Pipeline shape::

    raw export
      -> [ingest]    Trace(InvocationPoint...)
      -> [surface]   ToolProfile per tool, with ToolClass
      -> [state]     StateSchema (entities, keys, relations)
      -> [task]      TaskCase (instruction + reconstructed pre-state)
      -> [env]       materialized SQLite store + EffectLedger
      -> [verify]    Verifier (assertions over final state + effects)
      -> [fidelity]  FidelityReport (replay the source trace, accept/reject)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JsonValue = Any
JsonObject = dict[str, Any]


class Contract(BaseModel):
    """Base for every contract type: immutable, no silent extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Stage 1 - ingest
# --------------------------------------------------------------------------


class CallStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class InvocationPoint(Contract):
    """One recorded tool call. The atomic unit of the entire system.

    This is the record that makes an environment reconstructible. See
    docs/PLAN.md, "The key concept: invocation points".
    """

    call_id: str
    """Stable id for this call. From the export when declared, else synthesized."""

    trace_id: str
    step: int
    """0-based position within the trace, in declared order."""

    tool: str
    arguments: JsonObject = Field(default_factory=dict)
    response: JsonValue = None
    status: CallStatus = CallStatus.OK
    error_kind: str | None = None
    """Coarse error label when status is ERROR, e.g. 'not_found'. None when unknown."""

    latency_ms: float | None = None
    source_span_id: str | None = None
    """Original vendor span id, kept for provenance. Never used for logic."""


class Message(Contract):
    """One conversational turn. Carries the task statement and the final answer."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None


class Trace(Contract):
    """One normalized agent episode."""

    trace_id: str
    source: str
    """Declared adapter name, e.g. 'otlp'. Never guessed."""

    source_digest: str
    """sha256 of the exact source bytes this trace came from."""

    messages: tuple[Message, ...] = ()
    invocations: tuple[InvocationPoint, ...] = ()
    outcome: bool | None = None
    """Ground-truth success label. None means unlabeled - see PLAN.md Step 8."""

    metadata: JsonObject = Field(default_factory=dict)

    @property
    def instruction(self) -> str | None:
        """First user message, which is the task statement."""
        for m in self.messages:
            if m.role == "user" and m.content:
                return m.content
        return None


class IngestIssue(Contract):
    """A record that could not be normalized. Never silently dropped."""

    kind: str
    detail: str
    location: str | None = None


class TraceCorpus(Contract):
    """The normalized result of one ingest run."""

    source: str
    traces: tuple[Trace, ...]
    issues: tuple[IngestIssue, ...] = ()


# --------------------------------------------------------------------------
# Stage 2 - surface
# --------------------------------------------------------------------------


class ToolClass(str, Enum):
    """How a tool is treated when the environment is rebuilt."""

    READ = "read"
    """Answered by querying the rebuilt store."""

    WRITE = "write"
    """Mutates the rebuilt store. This is what verifiers assert on."""

    EXTERNAL = "external"
    """Irreversible / third-party. Stubbed, but the attempt is written to the effect ledger."""

    UNKNOWN = "unknown"
    """Not enough evidence. Must never be probed (PLAN.md Step 9)."""


class FieldProfile(Contract):
    """One observed field in a tool's arguments or response."""

    name: str
    json_types: tuple[str, ...]
    occurrences: int
    null_count: int = 0
    distinct_values: int = 0
    sample_values: tuple[JsonValue, ...] = ()
    looks_like_identifier: bool = False


class ErrorMode(Contract):
    """An observed failure response the rebuilt environment must be able to return."""

    error_kind: str
    occurrences: int
    example_response: JsonValue = None


class ToolProfile(Contract):
    """Everything learned about one tool from the corpus plus the declared schema."""

    name: str
    declared_schema: JsonObject | None = None
    """From the tool registry when available. Authoritative for the action space."""

    tool_class: ToolClass = ToolClass.UNKNOWN
    class_confidence: float = 0.0
    class_evidence: tuple[str, ...] = ()
    """Human-readable reasons for the classification. Required for review."""

    call_count: int = 0
    argument_fields: tuple[FieldProfile, ...] = ()
    response_fields: tuple[FieldProfile, ...] = ()
    error_modes: tuple[ErrorMode, ...] = ()
    observed_only: bool = False
    """True when the tool appears in traces but not in the declared registry."""

    declared_only: bool = False
    """True when declared but never called. These are the probing candidates."""


class ToolSurface(Contract):
    """The full action space: declared tools unioned with observed ones."""

    tools: tuple[ToolProfile, ...]

    def by_name(self, name: str) -> ToolProfile | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None


# --------------------------------------------------------------------------
# Stage 3 - state schema inference
# --------------------------------------------------------------------------


class ForeignKey(Contract):
    field: str
    references_entity: str
    references_field: str
    confidence: float = 0.0


class WriteEffect(Contract):
    """What a write tool was observed to change on an entity.

    This is evidence collected at schema-inference time and carried forward, so
    the environment runtime never has to re-derive a write's semantics from the
    tool's English name. Guessing ``refund_order -> "refunded"`` from the verb
    works until it meets ``cancel_order -> "cancelled"`` or
    ``approve_return -> "authorized"``.
    """

    tool: str
    key_argument: str | None = None
    """Which argument identifies the row being written."""

    argument_columns: dict[str, str] = Field(default_factory=dict)
    """Argument name -> column name, where the two differ.

    e.g. ``refund_order(amount_cents=...)`` sets ``orders.refund_amount_cents``.
    """

    sets_constants: JsonObject = Field(default_factory=dict)
    """Column -> constant value this tool was observed to set every time.

    e.g. ``{"status": "refunded"}``. Only populated when the value was actually
    observed, never inferred from the tool name.
    """

    response_echoes: tuple[str, ...] = ()
    """Columns the tool's own response echoes back to the caller."""

    evidence_count: int = 0
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    """Human-readable justification. A reviewer reads these."""


class EntitySchema(Contract):
    """One inferred table behind the tool responses."""

    name: str
    primary_key: str | None = None
    fields: tuple[FieldProfile, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    written_by: tuple[str, ...] = ()
    read_by: tuple[str, ...] = ()
    write_effects: tuple[WriteEffect, ...] = ()
    """Observed semantics of each writing tool. See WriteEffect."""

    evidence_count: int = 0
    static_snapshot: bool = False
    """True when the entity is only ever read and never cross-referenced.

    Underdetermined: we materialize the observed rows verbatim and refuse to
    invent structure. See PLAN.md Step 7, "Where this fails".
    """


class StateSchema(Contract):
    """The reconstructed database behind the agent's tools."""

    entities: tuple[EntitySchema, ...] = ()
    unresolved: tuple[str, ...] = ()
    """Tools whose responses could not be attributed to any entity."""

    def entity(self, name: str) -> EntitySchema | None:
        for e in self.entities:
            if e.name == name:
                return e
        return None


# --------------------------------------------------------------------------
# Stage 4 - tasks
# --------------------------------------------------------------------------


class EntityRows(Contract):
    """Seed rows for one entity in a task's starting state."""

    entity: str
    rows: tuple[JsonObject, ...] = ()


class TaskCase(Contract):
    """One task mined from a trace, with the starting state it needs to be solvable."""

    task_id: str
    trace_id: str
    instruction: str
    pre_state: tuple[EntityRows, ...] = ()
    """Reconstructed from reads that happen BEFORE the first write. Never after."""

    tools: tuple[str, ...] = ()
    outcome: bool | None = None
    provenance: JsonObject = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Stage 5 - environment runtime
# --------------------------------------------------------------------------


class Effect(Contract):
    """One attempted external side effect, recorded instead of performed."""

    tool: str
    arguments: JsonObject = Field(default_factory=dict)
    step: int = 0


class Observation(Contract):
    """What the environment returns for one action."""

    response: JsonValue = None
    status: CallStatus = CallStatus.OK
    error_kind: str | None = None


class EnvManifest(Contract):
    """Identity and honest limitations of one materialized environment."""

    env_id: str
    task_id: str
    schema_digest: str
    tool_classes: dict[str, ToolClass] = Field(default_factory=dict)
    static_entities: tuple[str, ...] = ()
    unsupported_tools: tuple[str, ...] = ()
    """Tools we could not reimplement. Calling one raises rather than faking a success."""


# --------------------------------------------------------------------------
# Stage 6 - verification
# --------------------------------------------------------------------------


class AssertionKind(str, Enum):
    STATE_EQUALS = "state_equals"
    """A field in the final state must equal an expected value."""

    STATE_UNCHANGED = "state_unchanged"
    """An entity (or row) must be untouched. Guards against collateral damage."""

    ROW_EXISTS = "row_exists"
    ROW_ABSENT = "row_absent"
    EFFECT_COUNT = "effect_count"
    """An external effect must have been attempted exactly N times."""


class Assertion(Contract):
    kind: AssertionKind
    entity: str | None = None
    row_key: JsonValue = None
    field: str | None = None
    expected: JsonValue = None
    tool: str | None = None
    """For EFFECT_COUNT."""

    description: str = ""


class Verifier(Contract):
    """The reward function. Code, never a judge."""

    verifier_id: str
    task_id: str
    assertions: tuple[Assertion, ...] = ()
    reviewed_by: str | None = None
    """Must be set by a human before this verifier is allowed to grade. PLAN.md Step 11."""


class AssertionResult(Contract):
    assertion: Assertion
    passed: bool
    actual: JsonValue = None


class VerificationResult(Contract):
    task_id: str
    passed: bool
    results: tuple[AssertionResult, ...] = ()
    reward: float = 0.0


# --------------------------------------------------------------------------
# Stage 7 - fidelity gate
# --------------------------------------------------------------------------


class ToolFidelity(Contract):
    """Per-tool replay agreement. Reported individually, never averaged away."""

    tool: str
    replayed: int
    matched: int
    mismatched: int
    unsupported: int = 0
    examples: tuple[JsonObject, ...] = ()

    @property
    def rate(self) -> float:
        return self.matched / self.replayed if self.replayed else 0.0


class FidelityReport(Contract):
    """The accept/reject test for a rebuilt environment. PLAN.md Step 12."""

    env_id: str
    trace_id: str
    per_tool: tuple[ToolFidelity, ...] = ()
    overall_rate: float = 0.0
    accepted: bool = False
    threshold: float = 0.9
    notes: tuple[str, ...] = ()


__all__ = [
    "Assertion",
    "AssertionKind",
    "AssertionResult",
    "CallStatus",
    "Contract",
    "Effect",
    "EntityRows",
    "EntitySchema",
    "EnvManifest",
    "ErrorMode",
    "FidelityReport",
    "FieldProfile",
    "ForeignKey",
    "IngestIssue",
    "InvocationPoint",
    "JsonObject",
    "JsonValue",
    "Message",
    "Observation",
    "StateSchema",
    "TaskCase",
    "ToolClass",
    "ToolFidelity",
    "ToolProfile",
    "ToolSurface",
    "Trace",
    "TraceCorpus",
    "VerificationResult",
    "Verifier",
    "WriteEffect",
]
