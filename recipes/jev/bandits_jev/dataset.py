"""Turn judge verdicts into a generic decision dataset for training a decision model.

A ``DecisionExample`` is deliberately not a next-state-verifier concept: it is a
state, a question, a set of named options, and a target distribution over them.
The action-outcome row this module compiles from a ``TurnJudgeRun`` is one
producer of that shape. A future tool-routing compiler or a customer's own
uploaded dataset produces the same ``DecisionExample`` contract by a different
path, and trains on it identically -- but this version's ``DecisionSchema`` is
deliberately narrowed to ``primitive="choice"`` only; see ``DecisionExample``.

Judge votes are disagreement among repeated calls to one LLM judge, not a
calibrated or ground-truth probability (see ``TurnVerdict.judge_votes``). A
turn the judge never resolved -- unobserved, missing from the run entirely, or
every vote transport-failed or came back unparsable -- has no distribution to
report and is quarantined (or, for "never sent to the judge", excluded)
rather than folded into a target concentrated on "unclear": that would invent
a probability the judge never expressed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from bandits.analyze.models import TaskSet
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract, Trace
from bandits.verify.nextstate import TurnJudgeRun, TurnVerdict
from bandits.verify.turns import Turn, extract_turns

_PROB_TOLERANCE = 1e-6

COMPILER_VERSION = 1
"""Bumped whenever the action-outcome compiler changes what it renders into
``state`` or how it derives a target from a verdict, so two decision ids can
be trusted to mean "same compiled content", not just "same source turn"."""

ACTION_OUTCOME_QUESTION = "What does the reaction establish about this action?"
ACTION_OUTCOME_OPTIONS: dict[str, str] = {
    "success": "The reaction demonstrates progress or success.",
    "unclear": "The reaction does not establish success or failure.",
    "failure": "The reaction demonstrates that the action was wrong.",
}
_SCORE_TO_OPTION = {1: "success", 0: "unclear", -1: "failure"}


class DecisionTarget(Contract):
    kind: Literal["hard", "soft"]
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def probabilities_are_a_distribution(self) -> DecisionTarget:
        if not self.probabilities:
            raise ValueError("a target must name at least one option's probability")
        for option, p in self.probabilities.items():
            if not math.isfinite(p):
                raise ValueError(f"probability for {option!r} is not finite: {p!r}")
            if p < 0:
                raise ValueError(f"probability for {option!r} is negative: {p!r}")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > _PROB_TOLERANCE:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")
        if self.kind == "hard" and sum(1 for p in self.probabilities.values() if p > 0) != 1:
            raise ValueError("a hard target must put all mass on exactly one option")
        return self


class DecisionLineage(Contract):
    """Where one example came from. Only ``source_kind`` and ``record_id`` are
    required -- everything else is specific to a particular kind of producer
    (the action-outcome compiler fills the judge/trace/turn fields; a future
    human-labeled or synthetic producer fills none of them and is not asked
    to invent a corpus, trace or judge run it never had)."""

    source_kind: str
    """What produced this example, e.g. "action_outcome_judge_votes"."""

    record_id: str
    """This producer's own identity for the record, unique within its kind."""

    source_artifact_ids: tuple[str, ...] = ()
    """Every upstream artifact this example's content actually depends on, in
    no particular order (for the action-outcome compiler: the judge run and,
    when supplied, the task set)."""

    corpus_id: str | None = None
    trace_id: str | None = None
    turn_index: int | None = None
    action_span_id: str | None = None
    judge_run_id: str | None = None
    task_set_id: str | None = None
    source_file: str | None = None
    """For an imported row: the JSONL file it came from."""
    source_line: int | None = None
    """For an imported row: its 1-based line number in ``source_file``."""


class DecisionJudgeInfo(Contract):
    """Every inference-affecting judge setting, for reproducibility.

    Two judge runs that differ in model, prompt, temperature or vote count are
    scoring a different question even when they landed on the same trace and
    turn -- ``settings_digest`` lets a reader tell whether two decision
    examples over the same turn actually came from the same judging process.
    Stored explicitly (not a computed property) so it survives serialization:
    a payload read back from disk carries the same digest a fresh build would
    compute, without needing the reader to know the hash formula.
    """

    model: str
    prompt_digest: str
    temperature: float
    votes_requested: int
    votes_valid: int
    settings_digest: str

    @staticmethod
    def compute_settings_digest(model: str, prompt_digest: str, temperature: float, votes_requested: int) -> str:
        payload = f"{model}:{prompt_digest}:{temperature}:{votes_requested}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


DecisionSplit = Literal["train", "dev", "calibration", "test"]

def deterministic_split(key: str) -> DecisionSplit:
    """Assign a split from a stable hash of a key -- a trace id, a group id,
    or a hash of a row's own content, never a file path, line number or
    position. Roughly 70/10/10/10 train/dev/calibration/test. Shared by the
    judge compiler (keyed by trace) and the importer (keyed by group or
    content), so both producers split the same way."""
    digest = hashlib.sha256(key.encode()).digest()
    bucket = digest[0] / 256.0
    if bucket < 0.70:
        return "train"
    if bucket < 0.80:
        return "dev"
    if bucket < 0.90:
        return "calibration"
    return "test"


def deterministic_held_out_split(key: str) -> DecisionSplit:
    """Split a task set's held-out dependency groups across evaluation uses.

    Fit traces remain training-only. Held-out lineages are divided roughly
    equally into dev, calibration and test so checkpoint selection,
    temperature fitting and final evaluation use disjoint groups.
    """
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 3
    return ("dev", "calibration", "test")[bucket]


_LEGACY_SPLIT_MAP: dict[str, DecisionSplit] = {
    "within_family_fit": "train",
    "within_family_held_out": "dev",
}


class DecisionExample(Contract):
    decision_id: str
    family_id: str
    """Kept for the judge compiler and existing readers. ``group_id`` is the
    schema-v2 name for the same idea (rows that must stay in one split); new
    producers should set both to the same value, since family_id stays
    required for now rather than breaking every existing row and test."""
    group_id: str | None = None
    """Rows sharing a ``group_id`` are kept in the same split and resampled
    together by a grouped bootstrap. Optional so a producer with no natural
    grouping (an arbitrary user JSONL row) is not forced to invent one. When
    unset, ``family_id`` is used as the group for split-assignment purposes."""
    state: str
    question: str
    primitive: Literal["choice"]
    """Narrowed to "choice" for this version. "noul" and "score" are not yet
    validated (a Noul needs exactly two semantic values; a Score needs
    explicit ordinal ranks) and are not claimed until that validation exists
    -- see the module docstring."""
    options: dict[str, str]
    target: DecisionTarget
    label_source: str
    split: DecisionSplit
    """train / dev / calibration / test. The judge compiler splits by trace:
    a task set's fit -> train and held_out -> dev/calibration/test, or,
    without one, all four splits from a hash of the lineage/trace id; see
    ``_trace_split``. A row's split is fixed at compile/import time and never
    redrawn afterward."""
    lineage: DecisionLineage
    judge: DecisionJudgeInfo | None = None
    """None for a non-judge label source (human, sealed outcome, synthetic)."""
    source: str | None = None
    """Where this row's content originally came from, e.g. a dataset name or
    URL (not the file path -- that lives in ``lineage.source_file``)."""
    license: str | None = None
    """The redistribution license of the row's original content, when known."""

    @model_validator(mode="after")
    def shape_is_well_formed(self) -> DecisionExample:
        if len(self.options) < 2:
            raise ValueError("a choice example needs at least 2 options")
        if len(self.options) > 26:
            raise ValueError(f"a choice example supports at most 26 options, got {len(self.options)}")
        for option_id, description in self.options.items():
            if not option_id.strip():
                raise ValueError("an option id must not be empty")
            if not description.strip():
                raise ValueError(f"option {option_id!r} has an empty description")
        missing = set(self.options) - set(self.target.probabilities)
        if missing:
            raise ValueError(f"target is missing option(s): {sorted(missing)}")
        unknown = set(self.target.probabilities) - set(self.options)
        if unknown:
            raise ValueError(f"target names option(s) not in options: {sorted(unknown)}")
        return self

    def matches_schema(self, schema: DecisionSchema) -> bool:
        return (
            self.primitive == schema.primitive
            and self.question == schema.question
            and self.options == schema.options
        )


class RejectedDecision(Contract):
    """A row that could not become a ``DecisionExample``. Every rejection
    points at where it came from: a judge-compiler row still names its
    ``trace_id`` (and optional ``turn_index``); an imported row instead names
    ``source_file`` + ``source_line``. At least one of those two pointers is
    required -- a rejection with neither would be untraceable."""

    trace_id: str | None = None
    turn_index: int | None = None
    family_id: str | None = None
    source_file: str | None = None
    source_line: int | None = None
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def has_reason(self) -> RejectedDecision:
        if not self.reasons:
            raise ValueError("a rejected decision must explain why")
        if self.trace_id is None and self.source_file is None:
            raise ValueError("a rejected decision needs a trace_id or a source_file to be traceable")
        return self


class DecisionSchema(Contract):
    """What every example in this dataset shares in shape, for a trainer to
    read once rather than re-deriving from every row. A producer that shares
    one question/options across all rows (the judge compiler) declares this
    and every row is checked against it by
    ``DecisionDataset.every_row_matches_its_schema``. A producer whose rows
    each carry their own question/options (a user's own labeled dataset)
    leaves this unset; the check is skipped entirely rather than forcing a
    fake shared shape onto per-row data."""

    primitive: Literal["choice"]
    options: dict[str, str]
    question: str


class DecisionDatasetCounts(Contract):
    examples: int
    train: int
    dev: int
    calibration: int = 0
    test: int = 0
    quarantined: int
    votes_requested: int = 0
    votes_valid_min: int = 0
    votes_valid_max: int = 0
    """Judge-only vote fields. Optional and default to 0 for a non-judge
    producer (an imported user dataset has no votes to report)."""

    @property
    def within_family_fit(self) -> int:
        """Backward-compatible alias for the judge compiler's old name."""
        return self.train

    @property
    def within_family_held_out(self) -> int:
        return self.dev


class DecisionDataset(Contract):
    schema_version: int = 2
    producer: str
    """What compiled this dataset, e.g. "action_outcome_judge_votes" or "jsonl_import"."""

    source_artifact_ids: tuple[str, ...]
    """Every upstream artifact this dataset as a whole depends on."""

    source_judge_run_id: str | None = None
    """Convenience accessor for the common case; None for a non-judge producer."""

    source_task_set_id: str | None = None
    decision_schema: DecisionSchema | None = None
    """Set only by a producer whose rows all share one question/options (the
    judge compiler). Left unset by a per-row producer (imported user data);
    see ``DecisionSchema``."""
    examples: tuple[DecisionExample, ...]
    quarantined: tuple[RejectedDecision, ...] = ()
    counts: DecisionDatasetCounts

    @model_validator(mode="before")
    @classmethod
    def migrate_v1_split_names(cls, data: Any) -> Any:
        """A v1 payload (``schema_version: 1``, or missing it) names its
        splits ``within_family_fit``/``within_family_held_out`` on both
        ``examples[*].split`` and ``counts``. v2 renamed those to
        ``train``/``dev`` (see ``_LEGACY_SPLIT_MAP``) without a migration
        path, which made every already-saved v1 artifact fail to load under
        v2's stricter split enum. Only rewrites the old names when they are
        actually present; a v2 payload (or any non-dict input) passes
        through untouched."""
        if not isinstance(data, dict):
            return data
        if data.get("schema_version", 1) != 1:
            return data
        migrated = dict(data)
        examples = migrated.get("examples")
        if isinstance(examples, (list, tuple)):
            migrated["examples"] = [
                {**e, "split": _LEGACY_SPLIT_MAP.get(e.get("split"), e.get("split"))}
                if isinstance(e, dict)
                else e
                for e in examples
            ]
        counts = migrated.get("counts")
        if isinstance(counts, dict) and (
            "within_family_fit" in counts or "within_family_held_out" in counts
        ):
            counts = dict(counts)
            counts["train"] = counts.pop("within_family_fit", counts.get("train", 0))
            counts["dev"] = counts.pop("within_family_held_out", counts.get("dev", 0))
            migrated["counts"] = counts
        migrated["schema_version"] = 2
        return migrated

    @model_validator(mode="after")
    def counts_match_rows(self) -> DecisionDataset:
        by_split: dict[str, int] = {}
        for e in self.examples:
            by_split[e.split] = by_split.get(e.split, 0) + 1
        if self.counts.examples != len(self.examples):
            raise ValueError(
                f"counts.examples ({self.counts.examples}) does not match "
                f"len(examples) ({len(self.examples)})"
            )
        expected = (
            by_split.get("train", 0),
            by_split.get("dev", 0),
            by_split.get("calibration", 0),
            by_split.get("test", 0),
            len(self.quarantined),
        )
        actual = (
            self.counts.train,
            self.counts.dev,
            self.counts.calibration,
            self.counts.test,
            self.counts.quarantined,
        )
        if expected != actual:
            raise ValueError("counts do not match the rows they claim to summarize")
        return self

    @model_validator(mode="after")
    def rows_sharing_a_group_stay_in_one_split(self) -> DecisionDataset:
        """Only ``group_id`` is an inviolable split-grouping key. ``family_id``
        is not: the judge compiler deliberately splits a family's own traces
        across train and the three held-out splits, so enforcing this on
        family_id would reject its normal output."""
        split_by_group: dict[str, str] = {}
        for e in self.examples:
            if e.group_id is None:
                continue
            seen = split_by_group.get(e.group_id)
            if seen is None:
                split_by_group[e.group_id] = e.split
            elif seen != e.split:
                raise ValueError(
                    f"group {e.group_id!r} appears in both split {seen!r} and {e.split!r}; "
                    "rows sharing a group_id must stay in one split"
                )
        return self

    @model_validator(mode="after")
    def every_row_matches_its_schema(self) -> DecisionDataset:
        if self.decision_schema is None:
            return self
        mismatched = [e.decision_id for e in self.examples if not e.matches_schema(self.decision_schema)]
        if mismatched:
            raise ValueError(
                f"{len(mismatched)} example(s) do not match decision_schema: {mismatched[:5]}"
            )
        return self


def _vote_shares(votes: tuple[int, ...]) -> dict[str, float]:
    """Fraction of votes landing on each label, dense over -1/0/1 -- the
    same computation as ``bandits.verify.nextstate._vote_shares``, kept
    local rather than imported (that one is private to its own module).
    Used to reconstruct ``judge_votes`` for a ``TurnVerdict`` that has
    ``votes`` but no ``judge_votes`` -- an older record saved before that
    field existed, or any producer that only ever set ``votes``."""
    n = len(votes)
    return {str(label): votes.count(label) / n for label in (-1, 0, 1)}


def _judge_votes_or_reconstructed(verdict: TurnVerdict) -> dict[str, float] | None:
    """``verdict.judge_votes`` when set; otherwise reconstructed from
    ``verdict.votes`` when there are any to reconstruct from. A verdict with
    neither (no votes were ever cast) has no distribution to report and
    stays None -- the caller quarantines that case, it is not invented
    here."""
    if verdict.judge_votes is not None:
        return verdict.judge_votes
    if verdict.votes:
        return _vote_shares(verdict.votes)
    return None


def _vote_target(judge_votes: dict[str, float]) -> DecisionTarget:
    """The judge's dense vote shares, remapped from score labels ("-1"/"0"/"1")
    to option names, as a soft target -- including a majority tie, which is
    itself the vote distribution and not collapsed toward any one option."""
    probabilities = {
        _SCORE_TO_OPTION[int(label)]: share for label, share in judge_votes.items()
    }
    for option in ACTION_OUTCOME_OPTIONS:
        probabilities.setdefault(option, 0.0)
    return DecisionTarget(kind="soft", probabilities=probabilities)


def _render_state(task: str | None, previous_action: str | None, turn: Turn) -> str:
    """The same evidence the judge prompt showed it: task, previous action (when
    there is one), current action, observed reaction -- never the judge's own
    instructions or its reply. Omitting the previous action here while the
    judge saw it would train the student on labels it cannot itself justify
    from the input it is given."""
    parts = []
    if task:
        parts.append(f"Task:\n{task}")
    if previous_action:
        parts.append(f"Previous action:\n{previous_action}")
    parts.append(f"Current action:\n{turn.action}")
    parts.append(f"Observed reaction:\n{turn.next_state() or ''}")
    return "\n\n".join(parts)


class DecisionSource(Contract):
    """One trace's task and turns, as the compiler needs them -- not a raw
    ``Trace``, so a caller that only has turns and a task string (not a full
    normalized trace) can still build a dataset."""

    trace_id: str
    lineage_id: str | None = None
    """Session, ticket, or retry chain shared by related traces, when known."""
    task: str | None
    turns: tuple[Turn, ...]


def source_from_trace(trace: Trace) -> DecisionSource:
    return DecisionSource(
        trace_id=trace.trace_id,
        lineage_id=trace.lineage_id,
        task=trace.task,
        turns=extract_turns(trace),
    )


def _source_group_id(source: DecisionSource) -> str:
    """The dependency unit used for both splitting and resampling."""
    return source.lineage_id if source.lineage_id is not None else source.trace_id


def compute_decision_id(judge_run_id: str, trace_id: str, turn_index: int) -> str:
    digest = hashlib.sha256(
        f"{judge_run_id}:{trace_id}:{turn_index}:compiler={COMPILER_VERSION}".encode()
    ).hexdigest()
    return f"decision-{digest[:16]}"


def build_decision_dataset(
    sources: Sequence[DecisionSource],
    judge_run: TurnJudgeRun,
    judge_run_id: str,
    *,
    task_set: TaskSet | None = None,
    task_set_id: str | None = None,
    trace_family: Mapping[str, str] | None = None,
    minimum_valid_votes: int = 1,
) -> DecisionDataset:
    """Compile a turn-judge run's verdicts into action-outcome decision rows.

    ``sources`` gives each traced episode's task and turns (see
    ``source_from_trace``); a trace named in ``judge_run.trace_ids`` but
    absent from ``sources`` is quarantined as not found, not silently
    skipped.

    When ``task_set`` is supplied, every trace judged must resolve to exactly
    one of its families and ``task_set.corpus_id`` must match
    ``judge_run.corpus_id`` -- a task set is either the map this run's traces
    were drawn from, or it is not used, never a partial, silently-defaulting
    overlay. Traces the task set does not place are quarantined, not put in
    ``within_family_fit`` by default: that would hide a task-set/corpus
    mismatch as a smaller-than-expected dataset.

    ``minimum_valid_votes`` rejects a verdict whose vote count met the
    request but too few of those votes actually succeeded (default: any
    successful vote is enough, matching today's judge behavior of recording
    partial votes with full provenance).
    """
    if task_set is not None:
        if task_set.corpus_id != judge_run.corpus_id:
            raise ValueError(
                f"task set {task_set_id!r} was built from corpus {task_set.corpus_id!r}, "
                f"not the judge run's corpus {judge_run.corpus_id!r}"
            )
        if trace_family is None:
            trace_family = {
                trace_id: family.family_id
                for family in task_set.families
                for trace_id in family.trace_ids
            }

    verdict_by_key = judge_run.verdict_by_key()
    sources_by_id = {source.trace_id: source for source in sources}
    examples: list[DecisionExample] = []
    quarantined: list[RejectedDecision] = []
    votes_valid: list[int] = []

    for trace_id in judge_run.trace_ids:
        source = sources_by_id.get(trace_id)
        family_id = (trace_family or {}).get(trace_id)
        if task_set is not None and family_id is None:
            quarantined.append(
                RejectedDecision(
                    trace_id=trace_id,
                    reasons=(f"task set {task_set_id!r} does not place this trace in any family",),
                )
            )
            continue
        if family_id is None:
            family_id = f"corpus:{judge_run.corpus_id}"
        if source is None:
            quarantined.append(
                RejectedDecision(
                    trace_id=trace_id,
                    family_id=family_id,
                    reasons=("trace not found among the sources given to the compiler",),
                )
            )
            continue
        group_id = _source_group_id(source)
        previous_action: str | None = None
        for turn in source.turns:
            verdict = verdict_by_key.get((trace_id, turn.index))
            if verdict is None:
                if turn.observed:
                    quarantined.append(
                        RejectedDecision(
                            trace_id=trace_id,
                            turn_index=turn.index,
                            family_id=family_id,
                            reasons=(
                                "no verdict in the judge run for this observed turn "
                                "(incomplete judge artifact)",
                            ),
                        )
                    )
                previous_action = turn.action
                continue
            if not verdict.observed:
                # Excluded, not quarantined: an unobserved turn was never sent
                # to the judge at all, so there is nothing to report as failed.
                previous_action = turn.action
                continue
            judge_votes = _judge_votes_or_reconstructed(verdict)
            if verdict.score is None or judge_votes is None:
                quarantined.append(
                    RejectedDecision(
                        trace_id=trace_id,
                        turn_index=turn.index,
                        family_id=family_id,
                        reasons=(
                            f"judge produced no usable verdict: {verdict.failure or 'no votes'}",
                        ),
                    )
                )
                previous_action = turn.action
                continue
            if len(verdict.votes) < minimum_valid_votes:
                quarantined.append(
                    RejectedDecision(
                        trace_id=trace_id,
                        turn_index=turn.index,
                        family_id=family_id,
                        reasons=(
                            f"only {len(verdict.votes)} valid vote(s), needs "
                            f"{minimum_valid_votes}",
                        ),
                    )
                )
                previous_action = turn.action
                continue
            votes_valid.append(len(verdict.votes))
            settings_digest = DecisionJudgeInfo.compute_settings_digest(
                judge_run.model, judge_run.prompt_digest, judge_run.temperature, judge_run.votes
            )
            source_artifact_ids = (judge_run_id,) + ((task_set_id,) if task_set_id else ())
            examples.append(
                DecisionExample(
                    decision_id=compute_decision_id(judge_run_id, trace_id, turn.index),
                    family_id=family_id,
                    group_id=group_id,
                    state=_render_state(source.task, previous_action, turn),
                    question=ACTION_OUTCOME_QUESTION,
                    primitive="choice",
                    options=dict(ACTION_OUTCOME_OPTIONS),
                    target=_vote_target(judge_votes),
                    label_source="judge_votes",
                    split=_trace_split(task_set, family_id, trace_id, source.lineage_id),
                    lineage=DecisionLineage(
                        source_kind="action_outcome_judge_votes",
                        record_id=f"{trace_id}:{turn.index}",
                        source_artifact_ids=source_artifact_ids,
                        corpus_id=judge_run.corpus_id,
                        judge_run_id=judge_run_id,
                        task_set_id=task_set_id,
                        trace_id=trace_id,
                        turn_index=turn.index,
                        action_span_id=turn.action_span_id,
                    ),
                    source="bandits_judge",
                    judge=DecisionJudgeInfo(
                        model=judge_run.model,
                        prompt_digest=judge_run.prompt_digest,
                        temperature=judge_run.temperature,
                        votes_requested=judge_run.votes,
                        votes_valid=len(verdict.votes),
                        settings_digest=settings_digest,
                    ),
                )
            )
            previous_action = turn.action

    by_split: dict[str, int] = {}
    for e in examples:
        by_split[e.split] = by_split.get(e.split, 0) + 1
    source_artifact_ids = (judge_run_id,) + ((task_set_id,) if task_set_id else ())
    return DecisionDataset(
        producer="action_outcome_judge_votes",
        source_artifact_ids=source_artifact_ids,
        source_judge_run_id=judge_run_id,
        source_task_set_id=task_set_id,
        decision_schema=DecisionSchema(
            primitive="choice", options=dict(ACTION_OUTCOME_OPTIONS), question=ACTION_OUTCOME_QUESTION
        ),
        examples=tuple(examples),
        quarantined=tuple(quarantined),
        counts=DecisionDatasetCounts(
            examples=len(examples),
            train=by_split.get("train", 0),
            dev=by_split.get("dev", 0),
            calibration=by_split.get("calibration", 0),
            test=by_split.get("test", 0),
            quarantined=len(quarantined),
            votes_requested=judge_run.votes,
            votes_valid_min=min(votes_valid) if votes_valid else 0,
            votes_valid_max=max(votes_valid) if votes_valid else 0,
        ),
    )


def _trace_split(
    task_set: TaskSet | None,
    family_id: str,
    trace_id: str,
    lineage_id: str | None,
) -> DecisionSplit:
    """Every related lineage lands in one split.

    A trace's turns share its task, and traces in one lineage are retries or
    episodes from the same session or ticket. Splitting either unit would let
    train see information about a held-out row.

    With a task set, fit membership maps to train. Held-out lineages are
    deterministically divided across dev, calibration and test, preserving
    the task set's train boundary while keeping model selection, temperature
    fitting and final evaluation disjoint. Unplaced traces are quarantined
    by the caller before this is reached.

    Without one, the lineage id picks one of all four splits, falling back to
    the trace id when no lineage was declared. Thus related traces stay
    together while ungrouped sources retain per-trace behavior."""
    if task_set is None:
        split_key = f"lineage:{lineage_id}" if lineage_id is not None else f"trace:{trace_id}"
        return deterministic_split(split_key)
    family = task_set.family_by_id().get(family_id)
    if family is not None and trace_id in family.held_out_trace_ids:
        held_out_key = f"lineage:{lineage_id}" if lineage_id is not None else f"trace:{trace_id}"
        return deterministic_held_out_split(held_out_key)
    return "train"


def build_decision_dataset_from_corpus(
    traces: Sequence[Trace],
    judge_run: TurnJudgeRun,
    judge_run_id: str,
    *,
    task_set: TaskSet | None = None,
    task_set_id: str | None = None,
    minimum_valid_votes: int = 1,
) -> DecisionDataset:
    """Convenience wrapper: build ``DecisionSource`` rows from a corpus's
    traces, then delegate to ``build_decision_dataset``."""
    sources = [source_from_trace(trace) for trace in traces]
    return build_decision_dataset(
        sources,
        judge_run,
        judge_run_id,
        task_set=task_set,
        task_set_id=task_set_id,
        minimum_valid_votes=minimum_valid_votes,
    )


_SPLITS: tuple[DecisionSplit, ...] = ("train", "dev", "calibration", "test")

LONG_STATE_CHARS = 32_000
"""A rough flag, not a token count: at ~4 characters per token a state this
long is likely over the scorer's default 8k-token limit. The scorer's real
tokenizer count is what actually rejects a row."""


class SplitSummary(Contract):
    rows: int
    majority_label: dict[str, int]
    """Rows per target's highest-probability option. A row whose target ties
    between options is counted under "tie", not assigned to either."""
    state_chars_p50: int
    state_chars_p99: int
    long_states: int
    """Rows with a state over ``LONG_STATE_CHARS``."""


def summarize_dataset(dataset: DecisionDataset) -> dict[str, SplitSummary]:
    """Per-split size, label balance and state length -- what to check
    before training: a split dominated by one label makes "always answer
    that" look good, and long states get rejected by the scorer."""
    summary: dict[str, SplitSummary] = {}
    for split in _SPLITS:
        rows = [e for e in dataset.examples if e.split == split]
        labels: dict[str, int] = {}
        for e in rows:
            top = max(e.target.probabilities.values())
            winners = [o for o, p in e.target.probabilities.items() if p == top]
            key = winners[0] if len(winners) == 1 else "tie"
            labels[key] = labels.get(key, 0) + 1
        lengths = sorted(len(e.state) for e in rows)
        summary[split] = SplitSummary(
            rows=len(rows),
            majority_label=dict(sorted(labels.items())),
            state_chars_p50=lengths[len(lengths) // 2] if lengths else 0,
            state_chars_p99=lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))] if lengths else 0,
            long_states=sum(1 for n in lengths if n > LONG_STATE_CHARS),
        )
    return summary


def as_test_only(dataset: DecisionDataset) -> DecisionDataset:
    """The same rows, every one in ``test``: a whole source held out, so a
    model trained without it can be scored on a kind of agent it never
    saw. Its rows must never also be in a training dataset -- build that one
    from the other sources only."""
    examples = tuple(e.model_copy(update={"split": "test"}) for e in dataset.examples)
    return dataset.model_copy(
        update={
            "examples": examples,
            "counts": dataset.counts.model_copy(
                update={"train": 0, "dev": 0, "calibration": 0, "test": len(examples)}
            ),
        }
    )


def merge_decision_datasets(parts: Sequence[tuple[str, DecisionDataset]]) -> DecisionDataset:
    """One dataset from several (e.g. judge runs over different corpora).
    Every row keeps the split it was given, so merging never moves a row
    across train/test; a decision id present in two parts is refused, and a
    group that would straddle splits is rejected by the dataset's own
    validation. The shared question/options schema is kept only when every
    part declares the same one."""
    if len(parts) < 2:
        raise ValueError("merging needs at least two datasets")
    seen: dict[str, str] = {}
    seen_steps: dict[tuple[str, int], str] = {}
    for dataset_id, dataset in parts:
        for example in dataset.examples:
            if example.decision_id in seen:
                raise ValueError(
                    f"decision {example.decision_id} is in both {seen[example.decision_id]} and {dataset_id}"
                )
            seen[example.decision_id] = dataset_id
            # Two judge runs over the same traces give the same step two
            # decision ids: the merged dataset would train on it twice, and
            # its verifier cost could not be attributed between the runs.
            step = (example.lineage.trace_id, example.lineage.turn_index)
            if step[0] is not None and step[1] is not None:
                if step in seen_steps and seen_steps[step] != dataset_id:
                    raise ValueError(
                        f"trace {step[0]!r} turn {step[1]} is in both {seen_steps[step]} and {dataset_id} "
                        "(two judge runs over the same traces); merge only one run per trace"
                    )
                seen_steps[step] = dataset_id
    schemas = {d.decision_schema.model_dump_json() if d.decision_schema else None for _, d in parts}
    schema = parts[0][1].decision_schema if len(schemas) == 1 else None
    examples = tuple(e for _, d in parts for e in d.examples)
    quarantined = tuple(q for _, d in parts for q in d.quarantined)
    by_split: dict[str, int] = {}
    for e in examples:
        by_split[e.split] = by_split.get(e.split, 0) + 1
    requested = {d.counts.votes_requested for _, d in parts}
    valid_min = [d.counts.votes_valid_min for _, d in parts if d.counts.examples]
    valid_max = [d.counts.votes_valid_max for _, d in parts if d.counts.examples]
    return DecisionDataset(
        producer="merged",
        source_artifact_ids=tuple(dataset_id for dataset_id, _ in parts),
        decision_schema=schema,
        examples=examples,
        quarantined=quarantined,
        counts=DecisionDatasetCounts(
            examples=len(examples),
            train=by_split.get("train", 0),
            dev=by_split.get("dev", 0),
            calibration=by_split.get("calibration", 0),
            test=by_split.get("test", 0),
            quarantined=len(quarantined),
            votes_requested=requested.pop() if len(requested) == 1 else 0,
            votes_valid_min=min(valid_min) if valid_min else 0,
            votes_valid_max=max(valid_max) if valid_max else 0,
        ),
    )


def compute_dataset_id(dataset: DecisionDataset) -> str:
    digest = hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()
    return f"decision-dataset-{digest[:16]}"


def save_decision_dataset(dataset: DecisionDataset, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_dataset_id(dataset),
        kind="decision_dataset",
        parent_artifact_id=dataset.source_judge_run_id or dataset.source_artifact_ids[0],
        payload=dataset.model_dump_json().encode(),
        summary={
            "examples": dataset.counts.examples,
            "train": dataset.counts.train,
            "dev": dataset.counts.dev,
            "calibration": dataset.counts.calibration,
            "test": dataset.counts.test,
            "quarantined": dataset.counts.quarantined,
        },
    )


def load_decision_dataset(dataset_id: str, store: DerivedStore) -> DecisionDataset:
    return DecisionDataset.model_validate_json(store.read_payload(dataset_id))


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    lines = (json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    os.replace(temporary, path)


def write_decision_dataset(dataset: DecisionDataset, output: Path) -> tuple[Path, Path]:
    """Write fit+held-out rows and a sibling quarantine file. Both written even if empty."""
    quarantine = output.with_name(f"{output.stem}.quarantined.jsonl")
    _atomic_jsonl(output, [e.model_dump(mode="json") for e in dataset.examples])
    _atomic_jsonl(quarantine, [q.model_dump(mode="json") for q in dataset.quarantined])
    return output, quarantine
