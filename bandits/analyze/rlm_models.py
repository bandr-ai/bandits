"""Contracts for RLM-mined task families.

Separate from :mod:`bandits.analyze.models` because these describe a different
kind of claim. A :class:`~bandits.analyze.models.TaskFamily` is the output of a
deterministic grouping over embedding distance: re-running it on the same
analysis reproduces it byte for byte, and its evidence is a number. Everything
here is the output of a model reading raw user requests and arguing about them,
which is reproducible only in the weaker sense that the inputs, the prompt, the
seed and the budget are all recorded beside the result.

The two are kept apart so neither can be mistaken for the other, and so the
existing miner stays a usable baseline while this path is being tested. Nothing
in this module feeds grouping in ``families.py``, and nothing there is read as
ground truth here.

The organising idea is the *contract*: two traces belong to one family when the
same parameterized verifier could correctly evaluate what their users asked for.
That is a falsifiable claim about required outcomes, not a topical resemblance,
so a family here records the rules that admit and exclude members, the outcome a
verifier would have to establish, and the traces that evidence both.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from bandits.traces import Contract


class TraceView(str, Enum):
    """How much of a trajectory the miner was allowed to read.

    Three arms of one experiment, and the difference between them is the
    experiment. Every arm hides rewards, evaluator labels, and known family or
    lineage labels; what varies is how much of the episode the miner may read
    on the way to a family.

    The risk the arms are designed to separate: a model shown what an agent did
    will group episodes by how the agent behaved — the tools it reached for, the
    shape of its path, whether it succeeded — and those groups look like task
    families and are not. A family that is really "runs that ended in a handoff"
    cannot support a verifier, because its members were never asked for the same
    work. Reading only what the user asked for makes that failure impossible by
    construction; reading the whole trajectory makes it likely but might also
    supply context that disambiguates a terse request. Which of those dominates
    is measured, not assumed, so the view travels on every artifact it produced.
    """

    USER_MESSAGES = "user-messages"
    """Path U: every user-role message, in order — the request and its corrections."""

    FIRST_USER_MESSAGE = "first-user-message"
    """The opening request alone, to test whether later user turns add task
    information or agent-dependent noise."""

    FULL_TRAJECTORY = "full-trajectory"
    """Path F: the whole conversation — user and assistant messages, tool calls
    and tool results — with rewards and evaluator labels still withheld.

    The permissive arm, and the one whose result needs the most scrutiny. It
    wins only if it improves semantic coherence and verifier transfer *without*
    mainly separating traces by execution behaviour; a taxonomy of successful
    runs, refusals, and common tool sequences is behaviour mining wearing a
    task-family label.
    """

    @property
    def reads_agent_behavior(self) -> bool:
        """Whether this arm shows the miner what the agent did.

        The leakage diagnostic keys off this: a family discovered under an arm
        that saw tool calls has to be checked against tool usage before it can
        be called a task family, and one discovered without them does not.
        """
        return self is TraceView.FULL_TRAJECTORY


VIEW_PREAMBLES: dict[TraceView, str] = {
    TraceView.USER_MESSAGES: (
        "You see ONLY what the users asked for: their messages, in order. You do not "
        "see what any agent did, what tools it called, how any episode ended, or any "
        "existing label. Do not speculate about any of those."
    ),
    TraceView.FIRST_USER_MESSAGE: (
        "You see ONLY the user's opening request for each trace. You do not see later "
        "user turns, what any agent did, what tools it called, how any episode ended, "
        "or any existing label. Do not speculate about any of those."
    ),
    TraceView.FULL_TRAJECTORY: (
        "You see the FULL trajectory of each episode: the user's messages, the "
        "assistant's turns, and the tool calls and results, marked with [user], "
        "[assistant] and [tool] prefixes. You do NOT see rewards, scores, evaluator "
        "labels, or whether anything succeeded; those were withheld.\n\n"
        "Read the agent's actions only as evidence of what the USER ASKED FOR. Group "
        "by the work that was requested, never by how the agent went about it. Two "
        "episodes calling the same tools are not one family if their users wanted "
        "different things, and two episodes taking completely different paths are one "
        "family if one verifier could evaluate both. Never group by tool sequence, "
        "path length, or whether a run appears to have gone well."
    ),
}
"""What each arm may read, stated to the model in its own words.

A prompt that claims the model sees only user messages while the trajectory arm
hands it tool calls is not a small inconsistency: the instruction is the only
thing telling Path F to read actions as evidence of intent rather than as the
thing to group on, and without it the arm tests nothing the plan asked about.
"""


class UserMessageView(Contract):
    """One trace as the miner is permitted to see it.

    Deliberately thin. There is no summary, no normalization and no extracted
    keyword: the hypothesis under test is that a model reading raw requests
    finds better families than embedding geometry does, and a preprocessing step
    that decided what mattered would be testing that step instead.
    """

    trace_id: str
    messages: tuple[str, ...] = ()
    """The episode as this arm may read it, in the order the source recorded it.

    Under the user-message arms these are user-role texts alone. Under
    ``FULL_TRAJECTORY`` they also carry role-prefixed assistant turns and tool
    calls and results, which is exactly the extra information that arm exists to
    test — and exactly why an artifact never records messages without the view
    that produced them.
    """

    readable: bool = True
    """False when the source recorded no roles to read.

    A trace whose roles are unavailable is marked rather than guessed at. The
    alternative — treating a trace's declared ``task`` as if it were a user
    message when no turns were recorded — would silently mix a source-derived
    field into an arm whose whole point is that only recorded roles are read.
    """

    unreadable_reason: str = ""

    withheld_fields: tuple[str, ...] = ()
    """Outcome-bearing keys stripped from this view before the miner saw it.

    Only ever non-empty under ``FULL_TRAJECTORY``, which is the one arm that
    reads tool payloads. Recorded rather than silently dropped because it is the
    evidence that the permissive arm stayed outcome-blind: a reviewer comparing
    Path U and Path F has to be able to see what Path F was denied, and a run
    that stripped nothing at all from a scored corpus is a run to distrust.
    """

    @model_validator(mode="after")
    def unreadable_views_say_why(self) -> UserMessageView:
        if not self.readable and not self.unreadable_reason.strip():
            raise ValueError(f"unreadable view {self.trace_id} carries no reason")
        if self.readable and not self.messages:
            raise ValueError(
                f"view {self.trace_id} is readable and empty; a trace with no user "
                "messages is unreadable, not a trace that asked for nothing"
            )
        return self


class FamilyContract(Contract):
    """One proposed task family, as a claim that can be argued with.

    A name alone is not falsifiable, so a contract carries the rules that admit
    and exclude a member and the outcome a verifier would have to establish for
    every one of them. Two requests in the same domain, phrased alike, belong to
    different contracts when they need different mutations or materially
    different success checks — that is the distinction embedding distance cannot
    make and the whole reason this path exists.
    """

    contract_id: str
    name: str
    definition: str
    """What user-requested work belongs here."""

    inclusion_rules: tuple[str, ...] = ()
    exclusion_rules: tuple[str, ...] = ()
    required_outcome_shape: tuple[str, ...] = ()
    """What a verifier must establish for every member.

    The load-bearing field. Two groups sharing a topic but differing here are
    two families, because one verifier cannot correctly evaluate both.
    """

    supporting_trace_ids: tuple[str, ...] = ()
    counterexample_trace_ids: tuple[str, ...] = ()
    """Traces that look like members and are not. Kept because a boundary is
    only legible from the near misses it excludes."""

    revision: int = Field(default=1, ge=1)
    """Bumped on every material change, so an assignment can name the wording it
    was made against rather than a contract id that quietly moved."""

    @model_validator(mode="after")
    def contract_is_argued_not_named(self) -> FamilyContract:
        if not self.name.strip():
            raise ValueError(f"contract {self.contract_id} has no name")
        if not self.definition.strip():
            raise ValueError(f"contract {self.contract_id} has no definition")
        if not self.required_outcome_shape:
            # A family with no stated outcome shape is a topic, not a contract:
            # nothing about it says what a verifier would check, so nothing about
            # it can be wrong. That is exactly the failure this path exists to
            # detect in the embedding miner, and it must not be reproducible here.
            raise ValueError(
                f"contract {self.contract_id} states no required outcome shape, so it "
                "names a topic rather than a verifiable task family"
            )
        overlap = set(self.supporting_trace_ids) & set(self.counterexample_trace_ids)
        if overlap:
            raise ValueError(
                f"contract {self.contract_id} lists {sorted(overlap)} as both support "
                "and counterexample"
            )
        return self

    def fingerprint(self) -> str:
        """Digest of the semantic claim, ignoring id, revision and evidence.

        Two runs invent different contract ids and different names for the same
        idea, so recurrence across runs is measured on what the contract *says*.
        Evidence is excluded because a contract supported by different traces is
        the same claim tested against different examples.
        """
        payload = json.dumps(
            {
                "definition": self.definition.strip().lower(),
                "inclusion": sorted(r.strip().lower() for r in self.inclusion_rules),
                "exclusion": sorted(r.strip().lower() for r in self.exclusion_rules),
                "outcome": sorted(r.strip().lower() for r in self.required_outcome_shape),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ProposedContract(Contract):
    """One contract exactly as the model must return it.

    Typed rather than ``dict`` so the backend's own decoder enforces the fields.
    An untyped ``list[dict]`` accepted ``{"name": ..., "description": ...}`` as a
    valid answer: SUBMIT succeeded, and the parser then discarded every one of
    them for having no stated outcome. The model was never told, by any channel
    it could not ignore, that the outcome was required.

    ``model_config`` allows extra keys rather than forbidding them: a model that
    adds a field it was not asked for has still answered the question, and
    rejecting the whole contract over a stray key would reintroduce the failure
    this type exists to prevent.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    contract_id: str = ""
    """Blank is allowed; the miner derives one from the definition."""

    name: str
    definition: str
    required_outcome_shape: list[str]
    """What a verifier must establish. The field that makes this a family
    rather than a topic, and the reason this model is typed at all."""

    inclusion_rules: list[str] = Field(default_factory=list)
    exclusion_rules: list[str] = Field(default_factory=list)
    supporting_trace_ids: list[str] = Field(default_factory=list)
    counterexample_trace_ids: list[str] = Field(default_factory=list)


class ProposedOperation(Contract):
    """One taxonomy operation exactly as the model must return it.

    Typed for a second reason beyond field enforcement: an untyped operation let
    the model write ``KEEP`` meaning "the user wants to keep their reservation".
    Naming ``contract_ids`` in the schema makes the subject of the verb the
    taxonomy rather than the user's request.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    operation: str
    contract_ids: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class Operation(str, Enum):
    """What one discovery step did to the taxonomy.

    Enumerated rather than free text because the stop condition is defined over
    these: a sweep that creates, splits or merges nothing and revises nothing
    materially is what "converged" means, and that cannot be decided by grepping
    a rationale string.
    """

    KEEP = "KEEP"
    CREATE = "CREATE"
    REVISE = "REVISE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    MARK_AMBIGUOUS = "MARK_AMBIGUOUS"
    MARK_UNCOVERED = "MARK_UNCOVERED"


MUTATING_OPERATIONS = frozenset(
    {Operation.CREATE, Operation.SPLIT, Operation.MERGE, Operation.REVISE}
)
"""Operations that change the taxonomy rather than record a reading of it.

``MARK_AMBIGUOUS`` and ``MARK_UNCOVERED`` are deliberately outside this set.
They record that a trace could not be placed, which is a finding about coverage
and not a change to the contracts; a sweep that only marks has converged on its
taxonomy while still reporting what that taxonomy does not reach.
"""


class TaxonomyOperation(Contract):
    """One recorded change, with the evidence that motivated it."""

    operation: Operation
    contract_ids: tuple[str, ...] = ()
    """What it acted on. A CREATE names the contract it produced; a MERGE names
    the contracts it consumed followed by the one it produced."""

    trace_ids: tuple[str, ...] = ()
    rationale: str
    material: bool = True
    """Whether a REVISE changed the claim or only its wording.

    Only meaningful for REVISE. A run that reworded a definition every sweep
    would never satisfy the stop condition, and a run that quietly rewrote what
    a family admits while calling it cosmetic would satisfy it wrongly, so the
    distinction is recorded per operation rather than inferred from a diff.
    """

    @model_validator(mode="after")
    def operation_is_justified(self) -> TaxonomyOperation:
        if not self.rationale.strip():
            raise ValueError(f"{self.operation.value} was recorded with no rationale")
        return self

    @property
    def mutating(self) -> bool:
        """Whether this changed the taxonomy for stop-condition purposes."""
        if self.operation is Operation.REVISE:
            return self.material
        return self.operation in MUTATING_OPERATIONS


class ChunkResult(Contract):
    """One call over one chunk of traces, and what it did to the taxonomy."""

    index: int = Field(ge=0)
    pass_index: int = Field(default=0, ge=0)
    """Which complete corpus pass this chunk belonged to.

    Recorded per chunk because a pass is the unit the stopping rule is defined
    over, and a chunk index alone cannot say whether the corpus has been read
    through again or merely sampled from.
    """

    trace_ids: tuple[str, ...]
    operations: tuple[TaxonomyOperation, ...] = ()
    assignments: dict[str, str] = Field(default_factory=dict)
    """Provisional trace -> contract id. Provisional because the definitions
    these were made against keep changing for the rest of the loop, which is why
    a separate fresh-assignment pass exists at all."""

    ambiguous_trace_ids: tuple[str, ...] = ()
    uncovered_trace_ids: tuple[str, ...] = ()
    llm_calls: int | None = Field(default=None, ge=0)
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, ge=0)
    """What the provider charged for this chunk, as the provider reported it.

    ``None`` means nothing reported a price, which is not the same as zero: a
    monetary budget can only be enforced over chunks that actually priced
    themselves, and the run says so when it could not.
    """

    duration_seconds: float | None = Field(default=None, ge=0)
    status: Literal["success", "error"] = "success"
    error: str = ""

    raw_reply: str = ""
    """Exactly what the model returned, before any parsing.

    Kept because the parsed result cannot explain itself. A chunk that proposed
    three contracts and kept none says only that they were rejected; whether
    they lacked an outcome shape, spelled a field differently, or arrived as
    JSON text is answerable only from what was actually returned — and without
    it the answer costs another paid run to guess at.

    Stored verbatim and never trusted as data: nothing reads this to build a
    taxonomy, and every field the miner uses is parsed from the prediction
    itself. This exists to be read by a person debugging a run.
    """

    dropped_contracts: tuple[str, ...] = ()
    """Contracts this chunk proposed and the parser refused, serialized.

    The specific evidence for the most expensive failure mode: the model doing
    the work and the code discarding it. A count in a limitation says it
    happened; these say what was thrown away.
    """

    @property
    def mutated(self) -> bool:
        return any(op.mutating for op in self.operations)


class PassResult(Contract):
    """One complete read of every eligible trace, and what it changed.

    The unit the stopping rule is defined over. A pass is only complete when
    every eligible trace appeared in it — not when a sample of them did — so
    that "reviewed twice" is a checkable property of the artifact rather than a
    hoped-for consequence of running a few more chunks.
    """

    pass_index: int = Field(ge=0)
    seed: int
    """The shuffle this pass used. Recorded per pass because each is reshuffled,
    and a pass whose order cannot be reproduced cannot be rerun."""

    trace_ids: tuple[str, ...]
    """Every trace this pass read, in the order it read them."""

    chunk_indices: tuple[int, ...] = ()
    operations: tuple[TaxonomyOperation, ...] = ()
    contracts_before: tuple[str, ...] = ()
    contracts_after: tuple[str, ...] = ()
    reassigned_trace_ids: tuple[str, ...] = ()
    """Traces this pass placed differently than the previous pass had.

    The number a reviewer actually reads at the pause: it says how much the
    taxonomy moved under a second look, which no count of operations does.
    """

    complete: bool = True
    """False when a budget guard fired mid-pass, so the pass read only part of
    the corpus and must not be counted toward the schedule."""

    @model_validator(mode="after")
    def a_pass_reads_each_trace_once(self) -> PassResult:
        if self.complete and not self.trace_ids:
            # An empty *complete* pass is a contradiction. An empty incomplete
            # one is not: a pass whose first chunk failed read nothing, and that
            # attempt is still worth recording rather than vanishing.
            raise ValueError(f"pass {self.pass_index} is complete but recorded no traces")
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError(
                f"pass {self.pass_index} read the same trace twice; a pass is one "
                "read of each eligible trace"
            )
        return self

    @property
    def mutated(self) -> bool:
        return any(op.mutating for op in self.operations)

    @property
    def churn(self) -> float:
        """Fraction of what this pass read that it placed differently."""
        return len(self.reassigned_trace_ids) / max(len(self.trace_ids), 1)


class AuditFinding(Contract):
    """One adversarial challenge to a provisional contract.

    Raised by a fresh context that never saw the discovery loop's reasoning, so
    it cannot inherit the assumption that produced the family it is attacking.
    """

    contract_id: str
    recommendation: Literal["keep", "revise", "split", "merge", "uncertain"]
    least_compatible_pair: tuple[str, str] | None = None
    """The two members whose requested work differs most, so a reviewer knows
    where the family is weakest without rereading all of it."""

    strongest_outsider_trace_id: str | None = None
    """The best apparent member currently outside. A contract that cannot
    exclude it is either too narrow or wrongly bounded."""

    merge_with_contract_id: str | None = None
    """The sibling contract this one should merge with, when recommended.

    A merge is advisory: the audit records the suspected over-split boundary,
    while discovery or a reviewer decides whether and how to rewrite it.
    """

    topical_only: bool = False
    """Whether members share a subject while needing different verifiers.

    The specific failure this path exists to catch. Distinguished from a general
    split recommendation because it names *why* the split is needed.
    """

    rationale: str
    raw_reply: str = ""
    """Exactly what the auditor returned, before parsing.

    The audit decides whether a contract survives the freeze, so a verdict that
    cannot be traced to what the model actually said is a decision with no
    evidence behind it. Stored for the same reason as ``ChunkResult.raw_reply``
    and read by nobody but a person.
    """

    resolved: bool = False
    resolution: str = ""

    @model_validator(mode="after")
    def finding_is_argued_and_resolutions_say_how(self) -> AuditFinding:
        if not self.rationale.strip():
            raise ValueError(f"audit finding for {self.contract_id} carries no rationale")
        if self.recommendation == "merge" and not self.merge_with_contract_id:
            raise ValueError("a merge recommendation must name a sibling contract")
        if self.recommendation != "merge" and self.merge_with_contract_id:
            # A target on a keep or a split is a contradiction, and a reviewer
            # reading the field would act on a merge nobody recommended.
            raise ValueError(
                f"finding for {self.contract_id} names a merge target without "
                "recommending a merge"
            )
        if self.merge_with_contract_id == self.contract_id:
            raise ValueError(f"contract {self.contract_id} cannot merge with itself")
        if self.resolved and not self.resolution.strip():
            raise ValueError(
                f"audit finding for {self.contract_id} is marked resolved with no account "
                "of how; an unexplained resolution is indistinguishable from ignoring it"
            )
        return self

    @property
    def demands_action(self) -> bool:
        """Whether discovery must answer this before the taxonomy may freeze."""
        return self.recommendation in ("revise", "split", "merge")


class TaxonomyAudit(Contract):
    """One adversarial pass over a draft taxonomy. Its own artifact.

    Advisory in the same sense the family audit is: it never edits contracts.
    Discovery must explicitly resolve or preserve each finding, and the stop
    condition refuses to freeze while an actionable finding is unaddressed.
    """

    schema_version: int = 1
    draft_id: str
    findings: tuple[AuditFinding, ...] = ()
    model: str
    prompt_digest: str
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def one_finding_per_contract(self) -> TaxonomyAudit:
        seen = [f.contract_id for f in self.findings]
        if len(seen) != len(set(seen)):
            raise ValueError("audit reports the same contract twice")
        return self

    def unresolved(self) -> tuple[AuditFinding, ...]:
        """Actionable findings discovery has not answered, in a stable order."""
        return tuple(
            sorted(
                (f for f in self.findings if f.demands_action and not f.resolved),
                key=lambda f: f.contract_id,
            )
        )


DEFAULT_PASSES = 2
"""Complete corpus passes before pausing for review.

Two, then stop. The first pass builds a taxonomy from nothing, so its early
chunks were judged against definitions that did not exist yet; the second is the
first time every trace is read against a taxonomy that has already seen the
whole corpus. One pass is a draft, two is a draft that has been checked, and the
pause after them is what keeps this from becoming a loop that runs until the
model stops objecting to itself.

A pass is complete only when every eligible trace appeared in it. Two clean
chunks are not two passes, and the difference is the whole stopping rule.

Continuing after the pause is a separate, human-initiated operation —
:class:`ResumeScope` — which resumes the same workspace rather than starting
over, and can target only what is actually unresolved.
"""


class ResumeScope(str, Enum):
    """What a resumed session re-examines. Chosen by a person, never inferred.

    Rereading all 160 traces to settle eleven ambiguous ones is mostly waste,
    and the waste is not only money: every reread is another chance for the
    model to reword a definition that was already fine, which shows up as churn
    and makes the run look less stable than it was.
    """

    FULL = "full"
    """Every eligible trace again, against the taxonomy as it now stands."""

    UNRESOLVED = "unresolved"
    """Ambiguous and uncovered traces only."""

    AFFECTED = "affected"
    """Unresolved traces, plus members of contracts that changed late in the
    previous pass — the ones judged against wording that has since moved."""

    FLAGGED = "flagged"
    """Only the contracts a reviewer named."""


class Budget(Contract):
    """Hard ceilings on one mining run.

    Every limit is a stopping condition that produces an *incomplete* artifact.
    A run that hits one has not converged, and the distinction is enforced in
    the contract rather than left to a caller's discipline: a taxonomy frozen
    because the money ran out is not a taxonomy that stopped changing.
    """

    passes: int = Field(default=DEFAULT_PASSES, ge=1)
    """Complete corpus passes to run before pausing. The schedule, not a guard.

    Distinct from every other field here: the limits below are emergencies that
    produce a partial artifact, while this is the plan the run is expected to
    finish. A run that stops because it completed its passes did what it was
    asked; a run that stops on any other field did not.
    """

    max_iterations: int = Field(default=200, ge=1)
    """Emergency guard on chunk count, not the stopping rule.

    Raised well above what a two-pass schedule needs: it used to double as the
    stopping rule, and a ceiling low enough to end a run is a ceiling that ends
    it mid-pass.
    """

    max_llm_calls: int = Field(default=400, ge=1)
    max_seconds: float = Field(default=3600.0, gt=0)
    max_usd: float | None = Field(default=None, gt=0)
    """None means no monetary ceiling was set, not that the run was free."""


class StopReason(str, Enum):
    PASSES_COMPLETE = "passes_complete"
    """Every requested pass read every eligible trace. Pauses for human review.

    Deliberately not called convergence. The loop stopping is a fact about the
    schedule that was run, not evidence that the taxonomy stopped moving, and
    the previous name invited exactly that inference. What this earns is a
    checkpoint and a diff between passes for a person to read — never an
    automatic promotion to a finished taxonomy.
    """

    MAX_ITERATIONS = "max_iterations"
    MAX_LLM_CALLS = "max_llm_calls"
    MAX_SECONDS = "max_seconds"
    MAX_USD = "max_usd"
    ERROR = "error"


COMPLETE_STOP_REASONS = frozenset({StopReason.PASSES_COMPLETE})
"""The stop reasons under which the requested schedule actually finished.

A frozenset of one, written as a set so the asymmetry is stated rather than
implied by an equality check scattered through the codebase. "Complete" here
means the passes that were asked for were run to the end — nothing more. Every
other reason is an emergency guard firing, which leaves a partial artifact.
"""


class TaxonomyDraft(Contract):
    """The provisional output of one discovery loop, before freezing.

    Never a TaskSet, and never silently promoted into one. A draft records how
    it stopped, and a draft that stopped because a budget ran out carries that
    for the rest of its life.
    """

    schema_version: int = 1
    analysis_id: str
    view: TraceView
    seed: int
    contracts: tuple[FamilyContract, ...] = ()
    chunks: tuple[ChunkResult, ...] = ()
    passes: tuple[PassResult, ...] = ()
    """Each complete read of the corpus, in order.

    The auditable record behind the stopping rule: a reader can check that every
    eligible trace appears in every completed pass, rather than taking the run's
    word that it swept.
    """

    completed_passes: int = Field(default=0, ge=0)
    """Passes that read every eligible trace. Never incremented by a partial one."""

    requested_passes: int = Field(default=DEFAULT_PASSES, ge=1)
    assignments: dict[str, str] = Field(default_factory=dict)
    """The taxonomy's final placement of every trace it could place.

    Stored so a resumed session starts from the workspace the pause left, rather
    than replaying every chunk to reconstruct it.
    """

    resumed_from: str | None = None
    """The draft this session continued, when it continued one."""

    resume_scope: ResumeScope | None = None
    """What a resumed session re-examined. None on a first run."""

    ambiguous_trace_ids: tuple[str, ...] = ()
    uncovered_trace_ids: tuple[str, ...] = ()
    unreadable_trace_ids: tuple[str, ...] = ()
    stop_reason: StopReason
    budget: Budget
    model: str
    prompt_digest: str
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def contracts_are_distinct(self) -> TaxonomyDraft:
        ids = [c.contract_id for c in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("draft contains duplicate contract ids")
        return self

    @property
    def complete(self) -> bool:
        """Whether the requested passes all finished.

        Not convergence, and deliberately not named for it. This says the
        schedule ran to the end and every eligible trace was read the requested
        number of times — it says nothing about whether the taxonomy had
        stopped moving, which is a judgement the pause exists to hand a person.
        """
        return (
            self.stop_reason in COMPLETE_STOP_REASONS
            and self.completed_passes >= self.requested_passes
        )

    @property
    def awaiting_review(self) -> bool:
        """Whether this is a checkpoint a person is expected to act on."""
        return self.complete

    def pass_diff(self) -> tuple[str, ...]:
        """What changed between the last two passes, for the pause summary.

        The question a reviewer actually has at a checkpoint — did the second
        look agree with the first — which no total over the whole run answers.
        """
        if len(self.passes) < 2:
            return ()
        previous, latest = self.passes[-2], self.passes[-1]
        created = set(latest.contracts_after) - set(latest.contracts_before)
        removed = set(latest.contracts_before) - set(latest.contracts_after)
        lines = [
            f"pass {latest.pass_index}: {len(latest.operations)} operation(s), "
            f"{len(latest.reassigned_trace_ids)} trace(s) placed differently "
            f"({latest.churn:.1%} of what it read)"
        ]
        if created:
            lines.append(f"contracts created: {', '.join(sorted(created))}")
        if removed:
            lines.append(f"contracts retired: {', '.join(sorted(removed))}")
        if not latest.mutated and not previous.mutated:
            lines.append(
                "neither of the last two passes changed the taxonomy, which is "
                "evidence for stability but is not itself a convergence test"
            )
        return tuple(lines)

    def contract_by_id(self) -> dict[str, FamilyContract]:
        return {c.contract_id: c for c in self.contracts}


class FrozenTaxonomy(Contract):
    """A draft, frozen and content-addressed, ready to be assigned against.

    Freezing is the point at which contracts stop moving, so an assignment can
    name what it was made against. The taxonomy id is derived from the contract
    text, so an assignment naming a taxonomy id is naming exact wording.
    """

    schema_version: int = 1
    draft_id: str
    audit_id: str | None = None
    """The adversarial pass this was frozen after. None means none ran, which is
    recorded because an unaudited taxonomy is a weaker claim, not a clean one."""

    analysis_id: str
    view: TraceView
    contracts: tuple[FamilyContract, ...]
    complete: bool
    """Carried forward from the draft. A taxonomy frozen from an incomplete
    draft is still assignable — that is how a budget-limited run is inspected —
    but it must never read as converged."""

    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def frozen_taxonomy_has_contracts(self) -> FrozenTaxonomy:
        ids = [c.contract_id for c in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("taxonomy contains duplicate contract ids")
        if not self.contracts:
            raise ValueError("a taxonomy with no contracts cannot be assigned against")
        return self

    def fingerprints(self) -> frozenset[str]:
        """The semantic claims this taxonomy makes, for cross-run comparison."""
        return frozenset(c.fingerprint() for c in self.contracts)


class AssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    """Matched more than one contract. Preserved, never broken by a tiebreak."""

    UNCOVERED = "uncovered"
    """Matched none. A real finding about the taxonomy, not a failure to try."""

    UNREADABLE = "unreadable"
    """The source recorded no user messages to classify."""


class TraceAssignment(Contract):
    """One trace, classified against frozen contracts by a fresh context."""

    trace_id: str
    matching_contract_ids: tuple[str, ...] = ()
    primary_contract_id: str | None = None
    status: AssignmentStatus
    reason: str

    @model_validator(mode="after")
    def status_matches_the_matches(self) -> TraceAssignment:
        """Reject a result whose status disagrees with what it matched.

        A model writes these, and the three failure modes are all silent: an
        ``assigned`` row with no contract, an ``uncovered`` row that names one,
        and an ``ambiguous`` row with a single match that a downstream reader
        would treat as a confident placement. Never forcing a match is the
        point of this stage, so the contract enforces it rather than trusting
        the prompt to have asked nicely.
        """
        if not self.reason.strip():
            raise ValueError(f"assignment for {self.trace_id} carries no reason")
        matches = set(self.matching_contract_ids)
        if len(matches) != len(self.matching_contract_ids):
            raise ValueError(f"assignment for {self.trace_id} names a contract twice")

        if self.status is AssignmentStatus.ASSIGNED:
            if self.primary_contract_id is None:
                raise ValueError(f"assignment for {self.trace_id} is assigned to nothing")
            if self.primary_contract_id not in matches:
                raise ValueError(
                    f"assignment for {self.trace_id} names a primary contract it did not match"
                )
        elif self.status is AssignmentStatus.AMBIGUOUS:
            if len(matches) < 2:
                raise ValueError(
                    f"assignment for {self.trace_id} is ambiguous with fewer than two matches"
                )
            if self.primary_contract_id is not None:
                raise ValueError(
                    f"assignment for {self.trace_id} is ambiguous and still picks a primary; "
                    "ambiguity is preserved for review rather than broken by a tiebreak"
                )
        else:
            if matches or self.primary_contract_id is not None:
                raise ValueError(
                    f"assignment for {self.trace_id} is {self.status.value} and still "
                    "names a contract"
                )
        return self


class AssignmentRun(Contract):
    """Every trace classified against one frozen taxonomy, in a fresh context.

    Its own artifact, parented to the taxonomy. Discovery assignments are
    provisional because the definitions moved underneath them; these are made
    once, against wording that can no longer change, by a context that never
    watched the taxonomy being argued into existence.
    """

    schema_version: int = 1
    taxonomy_id: str
    analysis_id: str
    view: TraceView
    assignments: tuple[TraceAssignment, ...] = ()
    model: str
    prompt_digest: str
    llm_calls: int | None = Field(default=None, ge=0)
    tokens: dict[str, int] = Field(default_factory=dict)
    duration_seconds: float | None = Field(default=None, ge=0)
    raw_replies: tuple[str, ...] = ()
    """What the classifier returned for each batch, before parsing.

    One entry per batch, in order. Assignment is where a trace becomes a member
    of a family, and a membership nobody can trace back to the reply that
    produced it cannot be checked.
    """

    dropped_results: tuple[str, ...] = ()
    """Rows the parser refused, verbatim. The evidence for a trace that came
    back uncovered because of a parsing failure rather than a real gap."""

    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def one_assignment_per_trace(self) -> AssignmentRun:
        seen = [a.trace_id for a in self.assignments]
        if len(seen) != len(set(seen)):
            raise ValueError("assignment run classifies the same trace twice")
        return self

    def by_status(self, status: AssignmentStatus) -> tuple[TraceAssignment, ...]:
        return tuple(a for a in self.assignments if a.status is status)

    def members(self) -> dict[str, tuple[str, ...]]:
        """Contract id -> the traces confidently assigned to it, in a stable order.

        Only ``assigned`` rows. An ambiguous trace matched this contract and
        another, and counting it here would let a family's membership be read as
        confident when the run explicitly declined to say so.
        """
        grouped: dict[str, list[str]] = {}
        for assignment in self.assignments:
            if assignment.status is AssignmentStatus.ASSIGNED:
                assert assignment.primary_contract_id is not None
                grouped.setdefault(assignment.primary_contract_id, []).append(assignment.trace_id)
        return {cid: tuple(sorted(traces)) for cid, traces in sorted(grouped.items())}

    def co_assignment_pairs(self) -> frozenset[tuple[str, str]]:
        """Unordered trace pairs this run placed in the same family.

        The unit of cross-run comparison. Generated family names differ between
        runs and carry no information, so stability is measured on which traces
        were put together — a claim two independent runs can actually agree or
        disagree about.
        """
        pairs: set[tuple[str, str]] = set()
        for traces in self.members().values():
            for index, left in enumerate(traces):
                for right in traces[index + 1 :]:
                    pairs.add((left, right) if left < right else (right, left))
        return frozenset(pairs)
