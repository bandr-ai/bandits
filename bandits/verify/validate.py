"""Calibrate drafted verifiers against human labels, and try to game them.

Two questions, answered separately:

Does the check agree with people who know the task? Measured out-of-fold — the
fit side is what the check was drafted from, so its agreement there is a
training-set number and is reported as such. The held-out side is the honest one.

Can the check be satisfied without doing the task? Answered by constructing
evidence that satisfies it and nothing else, then running it. A verifier with a
measured gameability result outranks one carrying a prose warning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from bandits.analyze.analysis import compute_analysis_id
from bandits.analyze.families import compute_task_set_id
from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    TaskSet,
    Visibility,
)
from bandits.labels import LabelSet, Verdict, compute_label_set_id
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.draft import compute_verifier_draft_id
from bandits.verify.execute import execute_verifier
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierSpec,
    VerifierStatus,
)

DEFAULT_SUCCESS_THRESHOLD = 0.5
"""At or above this, a verifier is taken to be claiming success."""

_MAX_COUNTEREXAMPLES = 10


class Counterexample(Contract):
    """One run where a verifier and a human disagreed."""

    trace_id: str
    verifier_score: float | None
    human_verdict: str
    rationale: str = ""

    kind: str = "false_positive"
    """``false_positive`` when the check passed a run a human failed.

    Named because the two errors are not equally dangerous. A false negative
    withholds credit for real work; a false positive rewards a run that did not
    do the task, and is the one that trains the wrong behaviour.
    """


class Agreement(Contract):
    verifier_id: str
    split: Literal["fit", "held_out"]
    """Only held_out is an honest estimate."""

    labeled: int = Field(ge=0)
    agreed: int = Field(ge=0)
    disagreed: int = Field(ge=0)
    unscored: int = Field(ge=0)
    """Labeled runs the verifier could not score. Never counted as disagreement."""

    agreement: float | None = None
    """None when nothing was both labeled and scorable — not zero."""

    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)
    """How the disagreements split, and the reason a rate alone is not enough.

    A false positive passes a run a human failed, and admits a failed trajectory
    into training data; a false negative only withholds credit for real work.
    ``0.81`` says nothing about which of those happened.

    Counted here rather than derived from ``counterexamples``, which
    ``_MAX_COUNTEREXAMPLES`` truncates: past ten disagreements the examples are
    a sample and the split is unrecoverable from them.

    Ordinary integers. Zero means measured and none occurred — an unmeasured
    split has no ``Agreement`` record at all.
    """

    counterexamples: tuple[Counterexample, ...] = ()

    @property
    def scored(self) -> int:
        return self.agreed + self.disagreed

    @property
    def coverage(self) -> float | None:
        """Share of labeled runs the verifier could score. None when none were.

        Kept apart from ``agreement``, which divides by ``scored`` and so reads
        1.00 for one agreed run beside ninety-nine unscorable ones.
        """
        return (self.scored / self.labeled) if self.labeled else None

    @property
    def failure_catch_rate(self) -> float | None:
        """Of the failures a human recorded and this could score, the share it caught.

        The number plain agreement hides: on a corpus that is mostly successes,
        a verifier that says success for everything scores well there and zero
        here. None when no scorable run was labeled a failure, and None when the
        record never said how its agreements split — a rate guessed from a
        missing field would flatter exactly the verifier this exists to catch.
        """
        caught = self.caught_failures
        if caught is None:
            return None
        failures = self.false_positives + caught
        return (caught / failures) if failures else None

    @property
    def caught_failures(self) -> int | None:
        """Scorable runs a human failed and this verifier also failed.

        Every scored run is one of four things: the two agreements and the two
        errors. Agreed runs the verifier called a failure are therefore the
        agreed total less those it called a success — which is what the false
        negatives, its wrong failure calls, cannot account for.

        None when ``successes_agreed`` was never recorded, which is how an
        artifact written before this split reads.
        """
        if self.successes_agreed is None:
            return None
        return max(0, self.agreed - self.successes_agreed)

    successes_agreed: int | None = Field(default=None, ge=0)
    """Agreed runs both a human and this verifier called a success.

    The other half of ``agreed``, kept so ``caught_failures`` is read from the
    record rather than inferred from a corpus-wide success rate that says
    nothing about this split.

    ``None``, never ``0``, when a record does not carry it. Zero would claim
    every agreed run was a caught failure, which reads as a perfect catch rate
    for the always-says-success verifier this number exists to expose.
    """

    @model_validator(mode="after")
    def counts_agree_with_the_rate(self) -> Agreement:
        """Recompute what the artifact claims rather than trusting the summary.

        A persisted measurement is read back long after the run that produced it,
        by code that cannot re-derive it. A stored rate that its own counts do not
        support would be believed, and every promotion decision downstream reads
        this number.
        """
        if self.labeled != self.agreed + self.disagreed + self.unscored:
            raise ValueError(
                f"agreement covers {self.labeled} labeled run(s) but accounts for "
                f"{self.agreed + self.disagreed + self.unscored}"
            )
        expected = (self.agreed / self.scored) if self.scored else None
        if expected is None and self.agreement is not None:
            raise ValueError("agreement rate is stated for a split with nothing scorable")
        if expected is not None and (
            self.agreement is None or abs(self.agreement - expected) > 1e-9
        ):
            raise ValueError(f"stated agreement {self.agreement} does not match {expected}")
        if self.false_positives + self.false_negatives != self.disagreed:
            raise ValueError(
                f"agreement states {self.disagreed} disagreement(s) but splits into "
                f"{self.false_positives + self.false_negatives}"
            )
        if self.successes_agreed is not None and self.successes_agreed > self.agreed:
            raise ValueError(
                f"agreement states {self.agreed} agreed run(s) but {self.successes_agreed} "
                "of them agreed on success"
            )
        if len(self.counterexamples) > self.disagreed:
            raise ValueError("more counterexamples than disagreements")
        if any(
            item.kind not in {"false_positive", "false_negative"} for item in self.counterexamples
        ):
            raise ValueError("a counterexample must be a false positive or a false negative")
        return self


class GameabilityResult(Contract):
    """Whether a constructed run satisfying a check actually passed the verifier."""

    verifier_id: str
    hypothesis: str
    constructed: dict[str, object]
    passed: bool
    """True means the attack worked: the verifier accepted a run that did not do the task."""

    scope: Literal["check", "composite"] = "check"
    """``check`` forges one check's evidence; ``composite`` forges every check's.

    Both are scored against the whole verifier, never against the check alone.
    Scoring a lone check answers a question nobody asked: a composite verifier
    fails closed when any other check loses its evidence, so an isolated forgery
    that "passes" one check leaves the real verifier returning unknown. Measuring
    it that way reported every composite verifier as gameable when none were.
    """

    checks_attacked: int = 1
    checks_total: int = 1
    """What this one attack reached, for reading a composite forgery's breadth.

    Whether the *verifier* was covered is :class:`GameabilityAssessment`'s to
    say, not this record's. A property here once answered it by comparing these
    two, and could not: it lives on a result, and the case that matters — no
    check matching any template, so no result at all — produces nothing for it
    to be read from.
    """

    forged_facts: int
    """How many facts had to be fabricated for the attack to land.

    Reported because "gameable" alone flattens a real difference: writing one
    status field is something a careless tool can do by itself, while forging a
    before/after pair requires faking the read the action was based on too. A
    check that costs more to game is a better check, and a bare pass/fail hides
    exactly that.
    """


class GameabilityAssessment(Contract):
    """What was attempted against one verifier, and what came of it.

    Two independent facts, deliberately not folded into one state. A verifier
    only half of whose checks a template could forge may still have been beaten
    by the half that was tried, and a four-valued enum would have to choose
    which of those to report.

    It exists because the per-attack records cannot say what was never
    attempted: when no check matches a template, ``probe_gameability`` returns
    nothing at all, and an empty result set read as "no attack succeeded" is
    indistinguishable from a verifier that resisted everything. This states the
    absence rather than fabricating an attack that never ran.
    """

    verifier_id: str

    coverage: Literal["none", "partial", "complete"]
    """How much of the verifier any template could attack.

    ``complete`` is not a safety claim. It says every check was attacked by the
    templates that exist, not that no other attack exists.
    """

    attack_succeeded: bool
    checks_attacked: int = Field(ge=0)
    checks_total: int = Field(ge=0)

    @model_validator(mode="after")
    def coverage_matches_the_counts(self) -> GameabilityAssessment:
        if self.checks_attacked > self.checks_total:
            raise ValueError(
                f"assessment attacked {self.checks_attacked} of {self.checks_total} check(s)"
            )
        if self.checks_attacked == 0:
            expected = "none"
        elif self.checks_attacked == self.checks_total:
            expected = "complete"
        else:
            expected = "partial"
        if self.coverage != expected:
            raise ValueError(
                f"coverage {self.coverage!r} does not match {self.checks_attacked} of "
                f"{self.checks_total} check(s) attacked"
            )
        if self.attack_succeeded and self.checks_attacked == 0:
            raise ValueError("an attack cannot have succeeded when none was attempted")
        return self


def assess_gameability(
    spec: VerifierSpec, results: Sequence[GameabilityResult]
) -> GameabilityAssessment:
    """Summarise one verifier's attack results, including their absence."""
    attacked = len({check.check_id for check in spec.checks if _attack(check) is not None})
    total = len(spec.checks)
    if attacked == 0:
        coverage: Literal["none", "partial", "complete"] = "none"
    elif attacked == total:
        coverage = "complete"
    else:
        coverage = "partial"
    return GameabilityAssessment(
        verifier_id=spec.verifier_id,
        coverage=coverage,
        attack_succeeded=any(
            result.passed for result in results if result.verifier_id == spec.verifier_id
        ),
        checks_attacked=attacked,
        checks_total=total,
    )


class Validation(Contract):
    schema_version: int = 1
    source_draft_id: str
    family_id: str
    label_set_id: str
    success_threshold: float = Field(default=DEFAULT_SUCCESS_THRESHOLD, ge=0, le=1)
    agreements: tuple[Agreement, ...] = ()
    gameability: tuple[GameabilityResult, ...] = ()

    gameability_assessments: tuple[GameabilityAssessment, ...] = ()
    """One per measured verifier, including those no template could attack.

    ``gameability`` above is the provenance — which attack, built how, and what
    it cost. This says what was covered, which an empty result set cannot.
    """

    labels_used: int = Field(default=0, ge=0)
    unclear_labels: int = Field(default=0, ge=0)

    success_labels: int = Field(default=0, ge=0)
    failure_labels: int = Field(default=0, ge=0)
    """How the family's labels split. A check measured against one verdict only
    has agreed with nothing it could have disagreed with."""

    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def measurements_are_coherent(self) -> Validation:
        seen = [(item.verifier_id, item.split) for item in self.agreements]
        if len(seen) != len(set(seen)):
            raise ValueError("validation states two agreements for one verifier and split")
        measured = {item.verifier_id for item in self.agreements}
        stray = {item.verifier_id for item in self.gameability} - measured
        if stray:
            raise ValueError(f"gameability reported for unmeasured verifier(s): {sorted(stray)}")
        assessed = [item.verifier_id for item in self.gameability_assessments]
        if len(assessed) != len(set(assessed)):
            raise ValueError("validation states two gameability assessments for one verifier")
        if assessed and (unassessed := measured - set(assessed)):
            # Partial coverage of the verifiers is worse than none: a reader
            # seeing assessments for three of four has no way to tell the fourth
            # apart from one that was assessed and came back clean. Empty stays
            # legal so validations written before assessments existed still load.
            raise ValueError(
                f"gameability assessed for some verifiers but not {sorted(unassessed)}"
            )
        stray_assessments = set(assessed) - measured
        if stray_assessments:
            raise ValueError(
                f"gameability assessed for unmeasured verifier(s): {sorted(stray_assessments)}"
            )
        for assessment in self.gameability_assessments:
            landed = any(
                item.passed
                for item in self.gameability
                if item.verifier_id == assessment.verifier_id
            )
            if assessment.attack_succeeded != landed:
                raise ValueError(
                    f"assessment for {assessment.verifier_id} disagrees with its attack results"
                )
        if self.success_labels + self.failure_labels > self.labels_used:
            raise ValueError("more adjudicated verdicts than labels used")
        return self

    def held_out(self, verifier_id: str) -> Agreement | None:
        return next(
            (a for a in self.agreements if a.verifier_id == verifier_id and a.split == "held_out"),
            None,
        )


def _claims_success(score: float | None, threshold: float) -> bool | None:
    return None if score is None else score >= threshold


def _agreement(
    *,
    spec: VerifierSpec,
    split: str,
    trace_ids: tuple[str, ...],
    evidence_by_trace: dict[str, tuple[Evidence, ...]],
    verdicts: dict[str, Verdict],
    threshold: float,
) -> Agreement:
    agreed = disagreed = unscored = 0
    false_positives = false_negatives = successes_agreed = 0
    counterexamples: list[Counterexample] = []

    for trace_id in trace_ids:
        verdict = verdicts.get(trace_id)
        if verdict is None:
            continue
        score = execute_verifier(spec, evidence_by_trace.get(trace_id, ())).score
        claim = _claims_success(score, threshold)
        if claim is None:
            unscored += 1
            continue
        if claim is (verdict is Verdict.SUCCESS):
            agreed += 1
            if claim:
                successes_agreed += 1
        else:
            disagreed += 1
            if claim:
                false_positives += 1
            else:
                false_negatives += 1
            counterexamples.append(
                Counterexample(
                    trace_id=trace_id,
                    verifier_score=score,
                    human_verdict=verdict.value,
                    kind="false_positive" if claim else "false_negative",
                )
            )

    scored = agreed + disagreed
    return Agreement(
        verifier_id=spec.verifier_id,
        split=split,
        labeled=sum(1 for trace_id in trace_ids if trace_id in verdicts),
        agreed=agreed,
        disagreed=disagreed,
        unscored=unscored,
        agreement=(agreed / scored) if scored else None,
        false_positives=false_positives,
        false_negatives=false_negatives,
        successes_agreed=successes_agreed,
        # A false positive — the check passing a run a human failed — is the one
        # that would reward the wrong behavior, so it is shown first.
        counterexamples=tuple(
            sorted(counterexamples, key=lambda c: (c.kind != "false_positive", c.trace_id))
        )[:_MAX_COUNTEREXAMPLES],
    )


def _forged(claim: str, key: str, value: object) -> Evidence:
    return Evidence(
        evidence_id=f"ev-forged-{claim}-{key}",
        claim=claim,
        value={"key": key, "field": key, "value": value, "tool": "forged"},
        visibility=Visibility.TERMINAL,
        provenance="derived",
        strength="weak",
        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        trace_id="forged",
    )


def _attack(check: CheckSpec) -> tuple[str, tuple[Evidence, ...], dict[str, object]] | None:
    """Build the least work that satisfies one check, and nothing more."""
    if check.operator is CheckOperator.EQUALS and check.claim.startswith("final_state_field:"):
        key = check.claim.partition(":")[2]
        return (
            f"Write {key!r} directly without performing the action.",
            (_forged("final_state_field", key, check.expected),),
            {key: check.expected},
        )

    if check.operator is CheckOperator.STATE_INVARIANT:
        final_key, _, initial_key = check.claim.partition(":")[2].partition("==")
        # Both sides forged to the same value: the invariant holds while nothing
        # was actually transferred.
        return (
            f"Copy {initial_key!r} into {final_key!r} without performing the action.",
            (
                _forged("initial_state_field", initial_key, 1),
                _forged("final_state_field", final_key, 1),
            ),
            {final_key: 1, initial_key: 1},
        )

    if check.operator is CheckOperator.EXIT_CODE_ZERO:
        return (
            "Run a trivially passing command instead of the required one.",
            (
                Evidence(
                    evidence_id="ev-forged-exit",
                    claim="command_exit_code",
                    value={"key": "exit_code", "value": 0, "tool": "true"},
                    visibility=Visibility.TERMINAL,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                    trace_id="forged",
                ),
            ),
            {"exit_code": 0, "tool": "true"},
        )

    if check.operator is CheckOperator.NO_SPAN_ERROR:
        return (
            "Swallow the error and report success without it reaching the trace.",
            (
                Evidence(
                    evidence_id="ev-forged-anchor",
                    claim="episode_span_count",
                    value=1,
                    visibility=Visibility.POST_HOC,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.OBSERVED_TRACE,
                    trace_id="forged",
                ),
            ),
            {"span_error": "suppressed"},
        )

    if check.operator is CheckOperator.EXACT_OUTPUT:
        return (
            "Emit the expected text without completing the underlying task.",
            (
                Evidence(
                    evidence_id="ev-forged-output",
                    claim="final_output",
                    value={"output": check.expected},
                    visibility=Visibility.TERMINAL,
                    provenance="derived",
                    strength="weak",
                    kind=EvidenceKind.AGENT_SELF_REPORT,
                    trace_id="forged",
                ),
            ),
            {"output": check.expected},
        )

    return None


def probe_gameability(
    spec: VerifierSpec, *, success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
) -> list[GameabilityResult]:
    """Construct runs that satisfy the checks without doing the task, and score them.

    Every attack is scored against the complete verifier. One forged field
    defeating a single-check verifier and one forged field defeating a
    twelve-check verifier are different findings, and only the whole spec can
    tell them apart.
    """
    attacks = [(check, _attack(check)) for check in spec.checks]
    available = [(check, attack) for check, attack in attacks if attack is not None]
    results: list[GameabilityResult] = []

    def score(evidence: tuple[Evidence, ...]) -> float | None:
        return execute_verifier(spec, evidence).score

    for _, attack in available:
        hypothesis, evidence, constructed = attack
        result = score(evidence)
        results.append(
            GameabilityResult(
                verifier_id=spec.verifier_id,
                hypothesis=hypothesis,
                constructed=constructed,
                passed=result is not None and result >= success_threshold,
                scope="check",
                checks_attacked=1,
                checks_total=len(spec.checks),
                forged_facts=len(evidence),
            )
        )

    if len(spec.checks) > 1 and available:
        # Forging one field at a time leaves the rest unknown and the verifier
        # unscorable. Forging all of them at once is the attack that actually
        # threatens a composite check, and its cost is the count of facts it took.
        merged: dict[str, Evidence] = {}
        constructed: dict[str, object] = {}
        for _, attack in available:
            _, evidence, fields = attack
            for item in evidence:
                merged.setdefault(item.evidence_id, item)
            constructed.update(fields)
        result = score(tuple(merged.values()))
        results.append(
            GameabilityResult(
                verifier_id=spec.verifier_id,
                hypothesis="Satisfy every check at once without performing the task.",
                constructed=constructed,
                passed=result is not None and result >= success_threshold,
                scope="composite",
                checks_attacked=len(available),
                checks_total=len(spec.checks),
                forged_facts=len(merged),
            )
        )
    return results


def validate_draft(
    draft: VerifierDraft,
    draft_id: str,
    task_set: TaskSet,
    analysis: CorpusAnalysis,
    label_set: LabelSet,
    label_set_id: str,
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Validation:
    """Measure each verifier against labels on both splits, then try to game it.

    Every artifact here is independently valid, so a mismatched set of them
    produces a measurement about the wrong examples rather than an error. The
    checks below are what make a measurement's subject provable, and they run
    before anything is measured.
    """
    # Identity before anything derived from it: an id that does not match its
    # own content cannot be trusted to name the artifact in later messages.
    if compute_verifier_draft_id(draft) != draft_id:
        raise ValueError(f"verifier draft content does not hash to {draft_id}")
    if compute_label_set_id(label_set) != label_set_id:
        raise ValueError(f"label set content does not hash to {label_set_id}")
    if compute_task_set_id(task_set) != draft.task_set_id:
        raise ValueError(
            f"draft was built against task set {draft.task_set_id} but the supplied "
            f"one hashes to {compute_task_set_id(task_set)}"
        )
    if compute_analysis_id(analysis) != draft.analysis_id:
        raise ValueError(
            f"draft was built from analysis {draft.analysis_id} but the supplied "
            f"one hashes to {compute_analysis_id(analysis)}"
        )
    if analysis.corpus_id != task_set.corpus_id:
        raise ValueError(
            f"analysis describes corpus {analysis.corpus_id} and the task set "
            f"{task_set.corpus_id}; they are not the same evidence"
        )
    if draft.analysis_id != task_set.analysis_id:
        raise ValueError(
            f"draft was built from analysis {draft.analysis_id} but the task set "
            f"comes from {task_set.analysis_id}; the evidence and the split do not "
            "describe the same corpus"
        )

    family = task_set.family_by_id().get(draft.family_id)
    if family is None:
        raise ValueError(f"unknown family id: {draft.family_id!r}")

    if label_set.task_set_id != draft.task_set_id:
        raise ValueError(
            f"label set {label_set_id} labels task set {label_set.task_set_id}, "
            f"not the draft's {draft.task_set_id}"
        )
    if label_set.family_id != draft.family_id:
        raise ValueError(
            f"label set {label_set_id} labels family {label_set.family_id}, "
            f"not the draft's {draft.family_id}"
        )

    by_trace: dict[str, list[Evidence]] = {}
    for item in analysis.evidence:
        by_trace.setdefault(item.trace_id, []).append(item)
    evidence_by_trace = {key: tuple(value) for key, value in by_trace.items()}

    # A label set is family-scoped by construction, so every label has to name a
    # member. Requiring only one to overlap would let a set that is mostly about
    # other traces through, and labels_used would then count labels that no
    # agreement was measured on — a summary at odds with itself.
    family_traces = set(family.fit_trace_ids) | set(family.held_out_trace_ids)
    stray = sorted({label.trace_id for label in label_set.labels} - family_traces)
    if stray:
        # Without this the agreement tally is a silent zero on both splits, which
        # reads downstream as a corpus too thin to measure rather than as the
        # wrong label set — and invites an --accept-risks override.
        raise ValueError(
            f"label set {label_set_id} labels {len(stray)} trace(s) outside family "
            f"{draft.family_id}, starting with {stray[0]!r}; a validation must be "
            "measured on the family it claims to be about"
        )

    verdicts = label_set.adjudicated()

    agreements: list[Agreement] = []
    gameability: list[GameabilityResult] = []
    assessments: list[GameabilityAssessment] = []
    for spec in draft.verifiers:
        for split, trace_ids in (
            ("fit", family.fit_trace_ids),
            ("held_out", family.held_out_trace_ids),
        ):
            agreements.append(
                _agreement(
                    spec=spec,
                    split=split,
                    trace_ids=trace_ids,
                    evidence_by_trace=evidence_by_trace,
                    verdicts=verdicts,
                    threshold=success_threshold,
                )
            )
        probed = probe_gameability(spec, success_threshold=success_threshold)
        gameability.extend(probed)
        assessments.append(assess_gameability(spec, probed))

    limitations: list[str] = []
    if not any(a.split == "held_out" and a.labeled for a in agreements):
        limitations.append(
            "no held-out trace carries a label; every agreement below is measured on "
            "the traces the checks were drafted from and is not an honest estimate"
        )
    unclear = len(label_set.labels) - len(verdicts)
    if unclear:
        limitations.append(
            f"{unclear} labeled run(s) a human could not adjudicate are excluded from "
            "every rate above rather than counted as either outcome"
        )

    in_family = [verdicts[trace_id] for trace_id in family_traces if trace_id in verdicts]
    successes = sum(1 for verdict in in_family if verdict is Verdict.SUCCESS)

    return Validation(
        source_draft_id=draft_id,
        family_id=draft.family_id,
        label_set_id=label_set_id,
        success_threshold=success_threshold,
        agreements=tuple(agreements),
        gameability=tuple(gameability),
        gameability_assessments=tuple(assessments),
        labels_used=len(in_family),
        unclear_labels=unclear,
        success_labels=successes,
        failure_labels=len(in_family) - successes,
        limitations=tuple(limitations),
    )


def calibrate(spec: VerifierSpec, validation_id: str) -> VerifierSpec:
    """Promote a verifier that has been measured. Never past calibrated."""
    return spec.replace(status=VerifierStatus.CALIBRATED, validation_artifact_id=validation_id)


def accept(spec: VerifierSpec, acceptance_id: str) -> VerifierSpec:
    """Record a human accepting a calibrated verifier.

    One outcome, where there were two. ``RISK_ACCEPTED`` existed for an owner
    promoting past an evidence blocker this stage raised on its own; promotion
    now rests on a review in which that same evidence was already shown and
    accepted, so there is no second judgement to go against. The status remains
    in ``VerifierStatus`` for artifacts written before this.
    """
    if spec.status is not VerifierStatus.CALIBRATED:
        raise ValueError("only a calibrated verifier can be accepted")
    return spec.replace(status=VerifierStatus.REVIEWED, human_acceptance_id=acceptance_id)


def compute_validation_id(validation: Validation) -> str:
    digest = hashlib.sha256(validation.model_dump_json().encode()).hexdigest()
    return f"validation-{digest[:16]}"


def save_validation(validation: Validation, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_validation_id(validation),
        kind="validation",
        parent_artifact_id=validation.source_draft_id,
        payload=validation.model_dump_json().encode(),
        summary={
            "agreements": len(validation.agreements),
            "gameable": sum(1 for g in validation.gameability if g.passed),
            "labels_used": validation.labels_used,
        },
    )


def load_validation(validation_id: str, store: DerivedStore) -> Validation:
    return Validation.model_validate_json(store.read_payload(validation_id))
