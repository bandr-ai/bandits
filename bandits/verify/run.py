"""Run drafted verifiers over the traces they were drafted from.

A drafted check is a hypothesis. Until it has been run, an owner asked to review
it is reviewing prose. This module produces the evidence that makes the review
concrete: what each verifier said about each historical run, and — more usefully
— the runs where two verifiers for the same family disagree.

Disagreement is where human labeling pays for itself. A trace all k verifiers
agree on teaches nothing; a trace they split on is exactly one ambiguity, and
resolving it moves every verifier at once.
"""

from __future__ import annotations

import hashlib

from bandits.analyze.models import CorpusAnalysis, TaskSet
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.execute import execute_verifier
from bandits.verify.models import CheckSpec, Result, VerifierDraft, VerifierSpec
from bandits.verify.validate import (
    Agreement,
    GameabilityAssessment,
    GameabilityResult,
    Validation,
)


class TraceOutcome(Contract):
    """What one verifier said about one historical run."""

    trace_id: str
    verifier_id: str
    result: Result


class Disagreement(Contract):
    """One run the family's verifiers do not agree about."""

    trace_id: str
    scores: dict[str, float | None]
    """Aggregate score per verifier id. None is unknown, never zero."""

    kind: str
    """``split`` when they disagree on a score, ``coverage`` when some cannot score it."""


class DraftRun(Contract):
    schema_version: int = 1
    source_draft_id: str
    family_id: str
    outcomes: tuple[TraceOutcome, ...]
    disagreements: tuple[Disagreement, ...] = ()
    unscorable_trace_ids: tuple[str, ...] = ()
    """Runs no verifier could score at all. Reported, never counted as failures."""

    def scores_for(self, trace_id: str) -> dict[str, float | None]:
        return {o.verifier_id: o.result.score for o in self.outcomes if o.trace_id == trace_id}


def run_draft(draft: VerifierDraft, analysis: CorpusAnalysis, task_set: TaskSet) -> DraftRun:
    """Score every fit trace in the family with every verifier in the draft."""
    family = task_set.family_by_id().get(draft.family_id)
    if family is None:
        raise ValueError(f"unknown family id: {draft.family_id!r}")

    # Fit only. Scoring the held-out side here would spend the split before
    # calibration ever gets to use it.
    trace_ids = family.fit_trace_ids or family.trace_ids
    by_trace: dict[str, list] = {}
    for item in analysis.evidence:
        by_trace.setdefault(item.trace_id, []).append(item)

    outcomes: list[TraceOutcome] = []
    disagreements: list[Disagreement] = []
    unscorable: list[str] = []

    for trace_id in trace_ids:
        evidence = tuple(by_trace.get(trace_id, ()))
        scores: dict[str, float | None] = {}
        for spec in draft.verifiers:
            result = execute_verifier(spec, evidence)
            outcomes.append(
                TraceOutcome(trace_id=trace_id, verifier_id=spec.verifier_id, result=result)
            )
            scores[spec.verifier_id] = result.score

        known = {score for score in scores.values() if score is not None}
        if not known:
            unscorable.append(trace_id)
        elif len(known) > 1:
            disagreements.append(Disagreement(trace_id=trace_id, scores=scores, kind="split"))
        elif any(score is None for score in scores.values()):
            # Not a contradiction, but still worth a label: one check sees the
            # run and another is blind to it, and only a human can say which
            # blindness matters.
            disagreements.append(Disagreement(trace_id=trace_id, scores=scores, kind="coverage"))

    return DraftRun(
        source_draft_id=compute_run_source_id(draft),
        family_id=draft.family_id,
        outcomes=tuple(outcomes),
        disagreements=tuple(disagreements),
        unscorable_trace_ids=tuple(unscorable),
    )


def compute_run_source_id(draft: VerifierDraft) -> str:
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    return f"verifier-draft-{digest[:16]}"


def compute_run_id(run: DraftRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode()).hexdigest()
    return f"verifier-run-{digest[:16]}"


def save_draft_run(run: DraftRun, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_run_id(run),
        kind="verifier_run",
        parent_artifact_id=run.source_draft_id,
        payload=run.model_dump_json().encode(),
        summary={
            "outcomes": len(run.outcomes),
            "disagreements": len(run.disagreements),
            "unscorable": len(run.unscorable_trace_ids),
        },
    )


def load_draft_run(run_id: str, store: DerivedStore) -> DraftRun:
    return DraftRun.model_validate_json(store.read_payload(run_id))


class CheckSummary(Contract):
    """What a reviewer is shown before deciding about one check.

    Bounded on purpose. A reviewer asked to judge a check against every trace
    it ever ran on reads nothing; the counts, the runs the verifiers split on,
    and — in a later round — what validation measured are what a decision
    actually turns on.
    """

    verifier_id: str
    check_id: str
    claim: str
    passed: int = 0
    failed: int = 0
    unscorable: int = 0
    example_trace_ids: tuple[str, ...] = ()
    """For the reviewer to open. Never sent to a model: an opaque id carries no
    signal it can use."""

    disagreement_trace_ids: tuple[str, ...] = ()
    evidence_kind: str = ""
    blind_spots: tuple[str, ...] = ()
    gaming_hypotheses: tuple[str, ...] = ()

    agreements: tuple[Agreement, ...] = ()
    """Per-split label agreement, when an earlier round measured it."""

    gameability: tuple[GameabilityResult, ...] = ()
    """Every attack an earlier round ran against this verifier, resisted or not.

    Resisted attacks are shown too: "three attacks, none landed" and "no attack
    could be built" are different findings, and only the first is evidence.
    """

    assessment: GameabilityAssessment | None = None
    """What the probe covered. None when no round has probed this verifier yet."""

    def prompt_lines(self) -> tuple[str, ...]:
        """The summary as the interpreter sees it — everything but trace ids.

        A second-round reply is usually answering the validation numbers. A
        model that cannot see them cannot tell a considered "still fine" about
        a check known to be weak from a first, uninformed look.
        """
        lines = [
            f"scored: {self.passed} passed, {self.failed} failed, {self.unscorable} unscorable",
            f"evidence_kind: {self.evidence_kind}",
        ]
        if self.passed and not self.failed:
            lines.append(
                "note: this check passed every run it could score, so nothing here shows it discriminating"
            )
        if self.disagreement_trace_ids:
            lines.append(f"verifiers disagreed on {len(self.disagreement_trace_ids)} run(s)")
        if not self.agreements:
            # Said rather than left blank. A check nobody has measured and a
            # check measured as good both render as silence otherwise, and
            # silence is the more flattering of the two readings.
            lines.append("agreement: unavailable — no labeled measurement covers this verifier yet")
        for agreement in self.agreements:
            rate = "unmeasured" if agreement.agreement is None else f"{agreement.agreement:.2f}"
            lines.append(
                f"agreement {agreement.split}: {rate} over {agreement.labeled} labeled run(s)"
            )
            if agreement.scored and agreement.false_positives is not None:
                # The rate alone cannot say which way the errors went, and the
                # two are not equally costly: a false positive passes a run a
                # human failed, and that is the one that reaches training data.
                # Omitted, not printed as zero, when an older record never said.
                lines.append(
                    f"errors {agreement.split}: {agreement.false_positives} false positive(s), "
                    f"{agreement.false_negatives} false negative(s)"
                )
                caught = agreement.failure_catch_rate
                if caught is not None:
                    lines.append(
                        f"failures caught {agreement.split}: {caught:.2f} "
                        f"({agreement.caught_failures} of "
                        f"{agreement.caught_failures + agreement.false_positives})"
                    )
            if agreement.coverage is not None and agreement.unscored:
                lines.append(
                    f"coverage {agreement.split}: {agreement.coverage:.2f} "
                    f"({agreement.unscored} of {agreement.labeled} unscorable)"
                )
        if self.assessment is not None:
            state = (
                "an attack succeeded" if self.assessment.attack_succeeded else "no attack succeeded"
            )
            lines.append(
                f"gameability coverage: {self.assessment.coverage} "
                f"({self.assessment.checks_attacked} of {self.assessment.checks_total} "
                f"check(s) attackable); {state}"
            )
            if self.assessment.coverage != "complete":
                lines.append(
                    "note: checks no template could attack were never tried, so a failed "
                    "attack here is not evidence they resist one"
                )
        for attack in self.gameability:
            outcome = "passed" if attack.passed else "resisted"
            lines.append(
                f"gameability: {attack.hypothesis} ({outcome}, {attack.forged_facts} forged fact(s))"
            )
        for blind in self.blind_spots:
            lines.append(f"blind spot: {blind}")
        for gaming in self.gaming_hypotheses:
            lines.append(f"gaming hypothesis: {gaming}")
        return tuple(lines)


def build_check_summary(
    spec: VerifierSpec,
    check: CheckSpec,
    run: DraftRun,
    *,
    validation: Validation | None = None,
    examples: int = 3,
) -> CheckSummary:
    """Gather what one check did, for one reviewer, in one screen."""
    passed = failed = unscorable = 0
    example_ids: list[str] = []
    for outcome in run.outcomes:
        if outcome.verifier_id != spec.verifier_id:
            continue
        part = next((s for s in outcome.result.subscores if s.check_id == check.check_id), None)
        if part is None or part.score is None:
            unscorable += 1
        elif part.score >= 1.0:
            passed += 1
        else:
            failed += 1
        if len(example_ids) < examples:
            example_ids.append(outcome.trace_id)

    agreements: tuple[Agreement, ...] = ()
    gameability: tuple[GameabilityResult, ...] = ()
    assessment: GameabilityAssessment | None = None
    if validation is not None:
        agreements = tuple(a for a in validation.agreements if a.verifier_id == spec.verifier_id)
        gameability = tuple(g for g in validation.gameability if g.verifier_id == spec.verifier_id)
        assessment = next(
            (a for a in validation.gameability_assessments if a.verifier_id == spec.verifier_id),
            None,
        )

    return CheckSummary(
        verifier_id=spec.verifier_id,
        check_id=check.check_id,
        claim=check.claim,
        passed=passed,
        failed=failed,
        unscorable=unscorable,
        example_trace_ids=tuple(example_ids),
        disagreement_trace_ids=tuple(
            d.trace_id for d in run.disagreements if spec.verifier_id in d.scores
        ),
        evidence_kind=check.evidence_kind.value,
        blind_spots=spec.blind_spots,
        gaming_hypotheses=spec.gaming_hypotheses,
        agreements=agreements,
        gameability=gameability,
        assessment=assessment,
    )
