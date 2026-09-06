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
from bandits.verify.models import VerifierStatus
from bandits.verify.validate import (
    accept,
    calibrate,
    load_validation,
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
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
        backend="first-word",
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
