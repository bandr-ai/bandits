"""Derived analysis contracts.

These live beside the corpus, never on it. The same trace may be reassessed by a
newer analysis policy or corrected by a reviewer; that must produce a new
artifact rather than rewrite the evidence it was derived from.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from bandits.traces import Contract


class Visibility(str, Enum):
    """When a fact becomes knowable, relative to one episode."""

    AT_START = "at_start"
    """Available to a live request at t=0: the instruction and permitted context."""

    DURING = "during"
    """Observed only as the episode runs."""

    TERMINAL = "terminal"
    """The final output or final observed state."""

    POST_HOC = "post_hoc"
    """Knowable only afterwards: outcome labels, totals, downstream corrections."""


_STRENGTH_ORDER: dict[str, int] = {"strong": 3, "moderate": 2, "weak": 1}


class EvidenceKind(str, Enum):
    """What produced a claim, ordered by how directly it can establish an outcome."""

    LIVE_QUERY = "live_query"
    TERMINAL_STATE_CHECK = "terminal_state_check"
    STRUCTURED_EXTERNAL_RESULT = "structured_external_result"
    TRUSTED_EVALUATOR = "trusted_evaluator"
    USER_FEEDBACK = "user_feedback"
    HUMAN_LABEL = "human_label"
    MODEL_JUDGMENT = "model_judgment"
    OBSERVED_TRACE = "observed_trace"
    """A fact read straight off a span that carries no outcome semantics of its own."""

    AGENT_SELF_REPORT = "agent_self_report"
    """The floor, deliberately. Nothing may rank below what the agent says about itself."""


_EVIDENCE_KIND_ORDER = {
    kind: rank
    for rank, kind in enumerate(
        reversed(tuple(EvidenceKind)),
        start=1,
    )
}


def kind_rank(kind: EvidenceKind) -> int:
    """Where a kind sits in the trust order. Higher wins."""
    return _EVIDENCE_KIND_ORDER[kind]


class Evidence(Contract):
    """One fact read off a trace, carrying where it came from and when it was knowable."""

    evidence_id: str
    claim: str
    """Short stable name for what is being asserted, e.g. 'command_exit_code'."""

    value: Any
    visibility: Visibility
    provenance: Literal["observed", "derived", "model", "human"]
    """'observed' means read directly off a span; nothing here infers."""

    strength: Literal["strong", "moderate", "weak"]
    kind: EvidenceKind = EvidenceKind.OBSERVED_TRACE
    trace_id: str
    span_id: str | None = None

    @property
    def strength_rank(self) -> int:
        """Higher wins when two pieces of evidence disagree."""
        return _STRENGTH_ORDER[self.strength]

    @property
    def trust_rank(self) -> tuple[int, int]:
        """Deterministic conflict ordering: source kind first, stated strength second."""
        return (_EVIDENCE_KIND_ORDER[self.kind], self.strength_rank)


class LeakageError(ValueError):
    """A task prompt was built from a fact that is not knowable at t=0.

    Raised rather than silently dropped: a task set that leaks its own answer
    into the prompt scores suspiciously well and is very hard to notice later.
    """


class TaskCandidate(Contract):
    """One episode, reshaped into something a new attempt could be asked to do."""

    task_id: str
    trace_id: str
    lineage_id: str | None = None
    """Identity metadata, not a fact about the episode, so it is not evidence."""

    instruction: str | None = None
    """None when the source never declared one; see ``limitations``."""

    prompt_evidence_ids: tuple[str, ...] = ()
    """Facts the prompt may be built from. Enforced ``at_start`` — see :func:`build_task_candidate`."""

    trajectory_span_ids: tuple[str, ...] = ()
    terminal_span_ids: tuple[str, ...] = ()
    outcome_evidence_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    """What this extraction could not recover. Never filled in with a guess."""


def build_task_candidate(
    *,
    task_id: str,
    trace_id: str,
    lineage_id: str | None = None,
    instruction: str | None,
    prompt_evidence: tuple[Evidence, ...],
    trajectory_span_ids: tuple[str, ...],
    terminal_span_ids: tuple[str, ...],
    outcome_evidence: tuple[Evidence, ...],
    limitations: tuple[str, ...] = (),
) -> TaskCandidate:
    """Assemble a task candidate, refusing any prompt fact that is not ``at_start``."""
    leaked = [e for e in prompt_evidence if e.visibility is not Visibility.AT_START]
    if leaked:
        detail = ", ".join(f"{e.evidence_id} ({e.visibility.value})" for e in leaked)
        raise LeakageError(f"prompt evidence for {task_id} is not knowable at t=0: {detail}")

    return TaskCandidate(
        task_id=task_id,
        trace_id=trace_id,
        lineage_id=lineage_id,
        instruction=instruction,
        prompt_evidence_ids=tuple(e.evidence_id for e in prompt_evidence),
        trajectory_span_ids=trajectory_span_ids,
        terminal_span_ids=terminal_span_ids,
        outcome_evidence_ids=tuple(e.evidence_id for e in outcome_evidence),
        limitations=limitations,
    )


class CorpusAnalysis(Contract):
    """The deterministic read of one corpus: what was asked, and what is known about how it went."""

    schema_version: int = 1
    corpus_id: str
    """Parent artifact. Every derived artifact records the one it came from."""

    source: str
    tasks: tuple[TaskCandidate, ...]
    evidence: tuple[Evidence, ...]
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> CorpusAnalysis:
        """Reject ambiguous evidence ids and dangling task references at the boundary."""
        evidence_ids = [e.evidence_id for e in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("analysis contains duplicate evidence ids")
        known = set(evidence_ids)
        for task in self.tasks:
            referenced = set(task.prompt_evidence_ids) | set(task.outcome_evidence_ids)
            missing = referenced - known
            if missing:
                raise ValueError(
                    f"task {task.task_id} references missing evidence: {sorted(missing)}"
                )
        return self

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}


_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")


def evidence_id(
    *, trace_id: str, claim: str, span_id: str | None = None, detail: str | None = None
) -> str:
    """Deterministic id, so re-analyzing the same corpus lands on the same ids.

    ``detail`` separates several facts of the same claim on one span — one span
    reporting both an amount and a status is two pieces of evidence, not one
    overwriting the other.

    The readable half flattens punctuation, so two details differing only in it —
    a literal ``a.b`` key and a nested ``a`` inside ``b`` — would share one id
    and the second fact would be dropped as a duplicate of the first. The digest
    is over the exact parts, so the id stays readable and still separates them.
    """
    parts = [trace_id, span_id or "trace", claim]
    if detail is not None:
        parts.append(detail)
    slug = _SLUG_UNSAFE.sub("-", "-".join(parts).lower()).strip("-")
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:8]
    return f"ev-{slug}-{digest}"


class SlotKind(str, Enum):
    """Why a task was reserved a place in a selection.

    A set built only from cluster centres has no failure signal in it, so slots
    are reserved for the tail before the remaining budget is spent.
    """

    MEDOID = "medoid"
    """The most central real trace in a family."""

    RARE_TOOL = "rare_tool"
    KNOWN_FAILURE = "known_failure"
    LONG_EPISODE = "long_episode"
    UNMEASURABLE = "unmeasurable"
    """No evidence stronger than the agent's own claim; kept so gaps stay visible."""

    FARTHEST_FIRST = "farthest_first"
    """Fills the remaining budget by maximizing distance from what is already in."""


class SelectedTask(Contract):
    trace_id: str
    family_id: str
    slot: SlotKind


class MissingSlot(Contract):
    """A reserved slot that could not be filled, and why. Never quietly dropped."""

    slot: str
    reason: str


class FamilyCoherence(Contract):
    """How far apart a family's most distant members are, and what that was judged against.

    The threshold is stored beside the measurement rather than only applied to it:
    grouping links edges and never bounds the span of the component they build, so
    a reader has to be able to recompute the verdict — and re-choose the factor —
    from the artifact alone, without re-mining or knowing the flags that were passed.
    """

    diameter: float = Field(ge=0.0, le=1.0)
    """Widest distance between any two descriptors in this family."""

    link_threshold: float = Field(ge=0.0, le=1.0)
    """``1 - similarity``: the furthest apart any single admitted link could be."""

    diameter_factor: float = Field(gt=0.0)
    widest_pair: tuple[str, str]
    """The two descriptors that span the family, so a reviewer knows what to split."""

    @property
    def over_merged(self) -> bool:
        """Wider than any single legal link explains, so held together only transitively."""
        return self.diameter > self.diameter_factor * self.link_threshold


class TaskFamily(Contract):
    """A group of episodes proposed as the same repeatable task."""

    family_id: str
    descriptor: str
    """The normalized instruction shared by the family, with values masked out."""

    trace_ids: tuple[str, ...]
    medoid_trace_id: str
    """A real trace, never a synthesized one."""

    workload_mass: int
    """How many production runs this family stands for."""

    fit_trace_ids: tuple[str, ...] = ()
    held_out_trace_ids: tuple[str, ...] = ()
    proposed_by: Literal["rule", "model", "human"] = "rule"
    review_status: Literal["proposed", "accepted", "split", "merged"] = "proposed"

    coherence: FamilyCoherence | None = None
    """How wide this family is, and against what it was judged. None before mining
    recorded it, or after a correction that could not recompute it."""

    limitations: tuple[str, ...] = ()

    @property
    def over_merged(self) -> bool:
        return self.coherence is not None and self.coherence.over_merged


class TaskSet(Contract):
    """Mined families plus the selection drawn from them."""

    schema_version: int = 1
    corpus_id: str
    analysis_id: str
    families: tuple[TaskFamily, ...]
    selected: tuple[SelectedTask, ...]

    total_workload_mass: int
    workload_coverage: float
    """Fraction of production volume the selection represents."""

    missing_slots: tuple[MissingSlot, ...] = ()
    underfilled: bool = False
    """Eligibility ran out before the budget did."""

    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_families(self) -> TaskSet:
        family_ids = [f.family_id for f in self.families]
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("task set contains duplicate family ids")

        known = set(family_ids)
        for selection in self.selected:
            if selection.family_id not in known:
                raise ValueError(
                    f"selected trace {selection.trace_id} references unknown family "
                    f"{selection.family_id}"
                )

        for family in self.families:
            members = set(family.trace_ids)
            split = set(family.fit_trace_ids) | set(family.held_out_trace_ids)
            if split - members:
                raise ValueError(f"family {family.family_id} splits traces it does not contain")
            if set(family.fit_trace_ids) & set(family.held_out_trace_ids):
                raise ValueError(
                    f"family {family.family_id} has a trace on both sides of its split"
                )
            if family.medoid_trace_id not in members:
                raise ValueError(f"family {family.family_id} medoid is not one of its traces")
        return self

    def family_by_id(self) -> dict[str, TaskFamily]:
        return {f.family_id: f for f in self.families}
