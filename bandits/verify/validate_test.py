"""Calibration and adversarial tests.

The address family is the useful case: drafting proposes two ideas of what
success looks like — the override tool reporting 'overridden', and the lookup
tool reporting 'paid' — and only labels can say which. addr-2 and addr-3 merely
looked the order up and never changed anything, so a human calls them failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.analyze import (
    analyze_corpus,
    compute_analysis_id,
    compute_task_set_id,
    mine_task_set,
)
from bandits.ingest import load_corpus
from bandits.labels import LabelSet, Verdict, compute_label_set_id, make_label
from bandits.store import DerivedStore
from bandits.verify import compute_verifier_draft_id, draft_verifiers
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.validate import (
    Agreement,
    GameabilityAssessment,
    Validation,
    _attack,
    accept,
    assess_gameability,
    calibrate,
    load_validation,
    probe_gameability,
    save_validation,
    validate_draft,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SUCCEEDED = "addr-1"
"""The only address run that actually changed anything."""


def _test_distance(left: str, right: str) -> float:
    return 0.0 if left.partition(" ")[0] == right.partition(" ")[0] else 1.0


@pytest.fixture
def address():
    analysis = analyze_corpus(load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    family = next(f for f in task_set.families if "address" in f.descriptor)
    # The real id, not a placeholder: validate_draft now proves the artifacts it
    # was handed are the ones the draft was built against.
    draft = draft_verifiers(
        task_set, compute_task_set_id(task_set), analysis, family.family_id, limit=6
    )
    return analysis, task_set, family, draft


def _labels(family, verdict_for, task_set_id: str | None = None, task_set=None) -> LabelSet:
    task_set_id = task_set_id or compute_task_set_id(task_set)
    return LabelSet(
        task_set_id=task_set_id,
        family_id=family.family_id,
        labels=tuple(
            make_label(
                trace_id=trace_id,
                family_id=family.family_id,
                verdict=verdict_for(trace_id),
                labeler="owner",
            )
            for trace_id in family.trace_ids
        ),
    )


def _validated(address, verdict_for=None):
    analysis, task_set, family, draft = address
    verdict_for = verdict_for or (lambda t: Verdict.SUCCESS if t == SUCCEEDED else Verdict.FAILURE)
    label_set = _labels(family, verdict_for, task_set=task_set)
    return validate_draft(
        draft,
        compute_verifier_draft_id(draft),
        task_set,
        analysis,
        label_set,
        compute_label_set_id(label_set),
    )


def test_labels_separate_the_right_hypothesis_from_the_wrong_one(address) -> None:
    _, _, _, draft = address
    validation = _validated(address)
    by_expected = {spec.verifier_id: spec.checks[0].expected for spec in draft.verifiers}

    fit = {
        by_expected[a.verifier_id]: a.agreement
        for a in validation.agreements
        if a.split == "fit" and a.labeled
    }

    assert fit["overridden"] == 1.0
    assert fit["paid"] == 0.0


def test_a_check_on_a_tool_that_never_reported_is_a_coverage_gap(address) -> None:
    """Naming the reporting tool turns a silent pass into a visible gap.

    While both checks read a field called ``status``, the override hypothesis
    scored addr-2 by comparing against the *lookup* tool's status and reading
    that as a failure. Each check now names the tool it needs, so a run where
    that tool never reported leaves the check unscored — which is what a human
    has to be shown, rather than a verdict assembled from another tool's field.
    """
    _, _, _, draft = address
    validation = _validated(address)
    by_expected = {spec.checks[0].expected: spec.verifier_id for spec in draft.verifiers}

    held_out = {
        a.verifier_id: a for a in validation.agreements if a.split == "held_out" and a.labeled
    }

    override = held_out[by_expected["overridden"]]
    assert override.unscored == 1
    assert override.agreement is None
    assert held_out[by_expected["paid"]].agreement == 0.0


def test_a_false_positive_is_named_and_ranked_first(address) -> None:
    """The dangerous error is the check passing a run that did nothing."""
    validation = _validated(address)

    counterexamples = [
        c for a in validation.agreements for c in a.counterexamples if a.split == "held_out"
    ]

    assert counterexamples
    worst = counterexamples[0]
    assert worst.kind == "false_positive"
    assert worst.human_verdict == "failure"
    assert worst.verifier_score == 1.0


def test_agreement_is_reported_per_split(address) -> None:
    """Fit agreement is a training-set number and must not be mixed with held-out."""
    validation = _validated(address)

    splits = {a.split for a in validation.agreements}
    assert splits == {"fit", "held_out"}


def test_a_run_the_verifier_cannot_score_is_not_a_disagreement() -> None:
    analysis = analyze_corpus(load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    family = next(f for f in task_set.families if "refund" in f.descriptor)
    draft = draft_verifiers(
        task_set, compute_task_set_id(task_set), analysis, family.family_id, limit=6
    )
    label_set = _labels(
        family,
        lambda t: Verdict.FAILURE if t in ("refund-4", "refund-9") else Verdict.SUCCESS,
        task_set=task_set,
    )

    validation = validate_draft(
        draft,
        compute_verifier_draft_id(draft),
        task_set,
        analysis,
        label_set,
        compute_label_set_id(label_set),
    )

    unscored = [a for a in validation.agreements if a.unscored]
    assert unscored, "the two failed refunds recorded no terminal state"
    assert all(a.disagreed == 0 for a in unscored)


def test_unclear_labels_are_excluded_and_reported(address) -> None:
    validation = _validated(address, verdict_for=lambda t: Verdict.UNCLEAR)

    assert validation.labels_used == 0
    assert validation.unclear_labels == 3
    assert any("could not adjudicate" in limit for limit in validation.limitations)


def test_an_unlabeled_held_out_side_is_called_out(address) -> None:
    """Otherwise a fit-only agreement rate reads as if it were an honest estimate."""
    analysis, task_set, family, draft = address
    label_set = LabelSet(
        task_set_id=compute_task_set_id(task_set),
        family_id=family.family_id,
        labels=tuple(
            make_label(
                trace_id=trace_id,
                family_id=family.family_id,
                verdict=Verdict.SUCCESS,
                labeler="owner",
            )
            for trace_id in family.fit_trace_ids
        ),
    )

    validation = validate_draft(
        draft,
        compute_verifier_draft_id(draft),
        task_set,
        analysis,
        label_set,
        compute_label_set_id(label_set),
    )

    assert any("not an honest estimate" in limit for limit in validation.limitations)


def test_forging_a_before_and_after_pair_costs_more_than_writing_one_field() -> None:
    """A bare gameable flag would hide why the invariant is the better check."""
    analysis = analyze_corpus(load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    family = next(f for f in task_set.families if "refund" in f.descriptor)
    draft = draft_verifiers(
        task_set, compute_task_set_id(task_set), analysis, family.family_id, limit=6
    )
    label_set = _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set)

    validation = validate_draft(
        draft,
        compute_verifier_draft_id(draft),
        task_set,
        analysis,
        label_set,
        compute_label_set_id(label_set),
    )
    cost = {result.hypothesis.split()[0]: result.forged_facts for result in validation.gameability}

    assert cost["Copy"] == 2, "an invariant requires faking the read it was based on"
    assert cost["Write"] == 1
    assert all(result.passed for result in validation.gameability), (
        "every deterministic replay check is gameable by construction; the honest "
        "report is how expensive each is, not whether any survives"
    )


def test_promotion_requires_measurement_then_a_human(address) -> None:
    _, _, _, draft = address
    spec = draft.verifiers[0]

    with pytest.raises(ValueError, match="only a calibrated verifier"):
        accept(spec, "acceptance-1")

    calibrated = calibrate(spec, "validation-1")
    assert calibrated.status is VerifierStatus.CALIBRATED

    reviewed = accept(calibrated, "acceptance-1")
    assert reviewed.status is VerifierStatus.REVIEWED
    assert reviewed.human_acceptance_id == "acceptance-1"


def test_a_validation_round_trips_through_the_derived_store(tmp_path, address) -> None:
    store = DerivedStore(tmp_path / ".bandits")
    validation = _validated(address)

    envelope = save_validation(validation, store)

    assert envelope.kind == "validation"
    assert load_validation(envelope.artifact_id, store) == validation


def _validate(address, label_set, label_set_id=None, draft=None, task_set=None, analysis=None):
    built_analysis, built_task_set, _family, drafted = address
    supplied_draft = draft or drafted
    return validate_draft(
        supplied_draft,
        compute_verifier_draft_id(supplied_draft),
        task_set or built_task_set,
        analysis or built_analysis,
        label_set,
        label_set_id or compute_label_set_id(label_set),
    )


def test_a_draft_id_that_does_not_match_its_content_is_refused(address) -> None:
    analysis, task_set, family, draft = address
    label_set = _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set)

    with pytest.raises(ValueError, match="verifier draft content does not hash to"):
        validate_draft(
            draft,
            "verifier-draft-0000000000000000",
            task_set,
            analysis,
            label_set,
            compute_label_set_id(label_set),
        )


def test_labels_for_another_family_are_refused_rather_than_measured(address) -> None:
    """The silent-zero path: every artifact valid, the measurement about nothing."""
    _analysis, task_set, family, _draft = address
    other = next(f for f in task_set.families if f.family_id != family.family_id)
    label_set = _labels(other, lambda t: Verdict.SUCCESS, task_set=task_set)

    with pytest.raises(ValueError, match="labels family"):
        _validate(address, label_set)


def test_labels_for_another_task_set_are_refused(address) -> None:
    _analysis, _task_set, family, _draft = address
    label_set = _labels(family, lambda t: Verdict.SUCCESS, task_set_id="ts-other")

    with pytest.raises(ValueError, match="not the draft's"):
        _validate(address, label_set)


def test_a_label_set_id_that_does_not_match_its_content_is_refused(address) -> None:
    _analysis, task_set, family, _draft = address
    label_set = _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set)

    with pytest.raises(ValueError, match="does not hash to"):
        _validate(address, label_set, label_set_id="labels-0000000000000000")


def test_a_draft_from_another_analysis_is_refused(address) -> None:
    _analysis, task_set, family, draft = address
    stray = draft.replace(analysis_id="analysis-somewhere-else")

    with pytest.raises(ValueError, match="but the supplied one hashes to"):
        _validate(
            address, _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set), draft=stray
        )


def test_a_task_set_that_is_not_the_one_the_draft_was_built_against_is_refused(address) -> None:
    """Ids can agree while the objects do not; only the content settles it."""
    analysis, task_set, family, _draft = address
    other = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        similarity=0.7,
        budget=3,
    )

    with pytest.raises(ValueError, match="but the supplied one hashes to"):
        _validate(
            address,
            _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set),
            task_set=other,
        )


def test_evidence_from_another_analysis_is_refused(address) -> None:
    """The draft's evidence has to be the analysis it was drafted from."""
    _analysis, task_set, family, _draft = address
    elsewhere = analyze_corpus(load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp"))

    with pytest.raises(ValueError, match="but the supplied one hashes to"):
        _validate(
            address,
            _labels(family, lambda t: Verdict.SUCCESS, task_set=task_set),
            analysis=elsewhere,
        )


def test_labels_naming_no_trace_in_the_family_are_refused(address) -> None:
    """Right family, right task set, but no label lands on a family trace."""
    _analysis, task_set, family, _draft = address
    label_set = LabelSet(
        task_set_id=compute_task_set_id(task_set),
        family_id=family.family_id,
        labels=(
            make_label(
                trace_id="trace-not-in-this-family",
                family_id=family.family_id,
                verdict=Verdict.SUCCESS,
                labeler="owner",
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside family"):
        _validate(address, label_set)


def test_a_label_set_mostly_about_other_traces_is_refused(address) -> None:
    """One overlapping label is not enough: labels_used would count the rest."""
    _analysis, task_set, family, _draft = address
    labels = [
        make_label(
            trace_id=family.trace_ids[0],
            family_id=family.family_id,
            verdict=Verdict.SUCCESS,
            labeler="owner",
        ),
        *(
            make_label(
                trace_id=f"foreign-{index}",
                family_id=family.family_id,
                verdict=Verdict.SUCCESS,
                labeler="owner",
            )
            for index in range(9)
        ),
    ]
    label_set = LabelSet(
        task_set_id=compute_task_set_id(task_set),
        family_id=family.family_id,
        labels=tuple(labels),
    )

    with pytest.raises(ValueError, match="outside family"):
        _validate(address, label_set)


def test_a_correctly_scoped_all_unclear_label_set_still_validates(address) -> None:
    """Unclear is a thin corpus, not a lineage error, once the traces are right."""
    _analysis, task_set, family, _draft = address

    validation = _validate(address, _labels(family, lambda t: Verdict.UNCLEAR, task_set=task_set))

    assert validation.labels_used == 0
    assert validation.unclear_labels == len(family.trace_ids)


def test_disagreements_are_counted_by_direction(address) -> None:
    """The rate alone cannot say which way an error went, and only one reaches training."""
    validation = _validated(address)

    for agreement in validation.agreements:
        assert agreement.false_positives + agreement.false_negatives == agreement.disagreed
        assert agreement.successes_agreed <= agreement.agreed


def test_the_error_split_survives_past_the_counterexample_cap() -> None:
    """Counterexamples are truncated at ten; the counts must not be read from them."""
    agreement = Agreement(
        verifier_id="v1",
        split="held_out",
        labeled=40,
        agreed=0,
        disagreed=40,
        unscored=0,
        agreement=0.0,
        false_positives=31,
        false_negatives=9,
    )

    assert agreement.false_positives == 31
    assert len(agreement.counterexamples) == 0


def test_an_error_split_that_does_not_account_for_its_disagreements_is_refused() -> None:
    with pytest.raises(ValueError, match="splits into"):
        Agreement(
            verifier_id="v1",
            split="held_out",
            labeled=4,
            agreed=2,
            disagreed=2,
            unscored=0,
            agreement=0.5,
            false_positives=1,
            false_negatives=0,
        )


def test_coverage_reads_apart_from_agreement() -> None:
    """One agreed run beside ninety-nine unscorable ones is not a verifier that works."""
    thin = Agreement(
        verifier_id="v1",
        split="held_out",
        labeled=100,
        agreed=1,
        disagreed=0,
        unscored=99,
        agreement=1.0,
        successes_agreed=1,
    )

    assert thin.agreement == 1.0
    assert thin.coverage == 0.01


def test_failure_catch_rate_exposes_a_verifier_that_only_says_success() -> None:
    """Nine of ten runs succeeded, so plain agreement flatters a check that never fails."""
    flattered = Agreement(
        verifier_id="v1",
        split="held_out",
        labeled=10,
        agreed=9,
        disagreed=1,
        unscored=0,
        agreement=0.9,
        false_positives=1,
        false_negatives=0,
        successes_agreed=9,
    )

    assert flattered.agreement == 0.9
    assert flattered.caught_failures == 0
    assert flattered.failure_catch_rate == 0.0


def _rubric_only_spec() -> VerifierSpec:
    """A verifier ``_attack`` has no template for: the softest checks there are."""
    return VerifierSpec(
        verifier_id="v-rubric",
        family_id="f1",
        task_set_id="ts1",
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=("terminal_evidence:rubric",),
        checks=(
            CheckSpec(
                check_id="c-rubric",
                claim="rubric:helpful",
                operator=CheckOperator.RUBRIC_AT_LEAST,
                expected=4,
                supporting_evidence_ids=("e1",),
                description="a judge scored this at least 4",
            ),
        ),
        unknown_when=("no judge verdict",),
        blind_spots=("a judge that always scores high",),
        gaming_hypotheses=("persuade the judge without doing the task",),
    )


def test_a_verifier_no_template_can_attack_records_that_it_was_never_tried() -> None:
    """An empty result set used to read exactly like resisting every attack."""
    spec = _rubric_only_spec()
    assert all(_attack(check) is None for check in spec.checks)
    assert probe_gameability(spec) == []

    assessment = assess_gameability(spec, ())

    assert assessment.coverage == "none"
    assert assessment.attack_succeeded is False


def test_coverage_and_outcome_are_independent_facts() -> None:
    """A half-covered verifier can still have been beaten by the half that was tried."""
    beaten = GameabilityAssessment(
        verifier_id="v1",
        coverage="partial",
        attack_succeeded=True,
        checks_attacked=1,
        checks_total=2,
    )

    assert beaten.coverage == "partial"
    assert beaten.attack_succeeded

    with pytest.raises(ValueError, match="does not match"):
        GameabilityAssessment(
            verifier_id="v1",
            coverage="complete",
            attack_succeeded=False,
            checks_attacked=1,
            checks_total=2,
        )


def test_an_attack_cannot_have_landed_when_none_was_attempted() -> None:
    with pytest.raises(ValueError, match="cannot have succeeded"):
        GameabilityAssessment(
            verifier_id="v1",
            coverage="none",
            attack_succeeded=True,
            checks_attacked=0,
            checks_total=2,
        )


def test_a_catch_rate_is_unavailable_rather_than_flattering_when_unrecorded() -> None:
    """A record that never said how its agreements split cannot state this rate.

    Defaulting the split to zero would report every agreed run as a caught
    failure — a perfect score for exactly the always-says-success verifier this
    number exists to expose.
    """
    unrecorded = Agreement(
        verifier_id="v1",
        split="held_out",
        labeled=10,
        agreed=9,
        disagreed=1,
        unscored=0,
        agreement=0.9,
        false_positives=1,
        false_negatives=0,
    )

    assert unrecorded.successes_agreed is None
    assert unrecorded.caught_failures is None
    assert unrecorded.failure_catch_rate is None
    assert unrecorded.replace(successes_agreed=9).failure_catch_rate == 0.0


def test_the_error_split_survives_a_store_round_trip(tmp_path, address) -> None:
    """Every promotion decision downstream reads these back from disk."""
    store = DerivedStore(tmp_path / ".bandits")
    validation = _validated(address).replace(
        agreements=(
            Agreement(
                verifier_id="v1",
                split="held_out",
                labeled=6,
                agreed=3,
                disagreed=3,
                unscored=0,
                agreement=0.5,
                false_positives=2,
                false_negatives=1,
                successes_agreed=1,
            ),
        ),
        gameability=(),
        gameability_assessments=(),
    )

    envelope = save_validation(validation, store)
    restored = load_validation(envelope.artifact_id, store).agreements[0]

    assert (restored.false_positives, restored.false_negatives) == (2, 1)
    assert restored.successes_agreed == 1
    assert restored.caught_failures == 2


def test_assessing_only_some_measured_verifiers_is_refused() -> None:
    """Assessing three of four leaves the fourth indistinguishable from clean."""
    agreements = tuple(
        Agreement(
            verifier_id=verifier_id,
            split="held_out",
            labeled=1,
            agreed=1,
            disagreed=0,
            unscored=0,
            agreement=1.0,
            successes_agreed=1,
        )
        for verifier_id in ("v1", "v2")
    )

    with pytest.raises(ValueError, match="assessed for some verifiers but not"):
        Validation(
            source_draft_id="d1",
            family_id="f1",
            label_set_id="l1",
            agreements=agreements,
            gameability_assessments=(
                GameabilityAssessment(
                    verifier_id="v1",
                    coverage="complete",
                    attack_succeeded=False,
                    checks_attacked=1,
                    checks_total=1,
                ),
            ),
        )


def test_an_agreement_without_a_recorded_split_still_loads() -> None:
    """An artifact written before the split fields must stay readable.

    Defaulting the two counts to zero would make the validator reject every
    stored record that had a disagreement, since zero errors cannot account for
    one. Reading as split unknown is true; reading as no errors would be both
    false and flattering to the verifier.
    """
    stored = {
        "verifier_id": "v1",
        "split": "held_out",
        "labeled": 10,
        "agreed": 7,
        "disagreed": 3,
        "unscored": 0,
        "agreement": 0.7,
        "counterexamples": [],
    }

    agreement = Agreement.model_validate(stored)

    assert agreement.false_positives is None
    assert agreement.false_negatives is None
    assert agreement.failure_catch_rate is None


def test_a_half_recorded_split_is_still_checked() -> None:
    """Nullability is for whole records, not a way past the sum check."""
    with pytest.raises(ValueError, match="splits into"):
        Agreement(
            verifier_id="v1",
            split="held_out",
            labeled=10,
            agreed=7,
            disagreed=3,
            unscored=0,
            agreement=0.7,
            false_positives=1,
            false_negatives=1,
        )
