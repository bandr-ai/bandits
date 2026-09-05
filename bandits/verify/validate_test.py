"""Calibration and adversarial tests.

The address family is the useful case: drafting proposes two ideas of what
success looks like — the override tool reporting 'overridden', and the lookup
tool reporting 'paid' — and only labels can say which. addr-2 and addr-3 merely
looked the order up and never changed anything, so a human calls them failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.ingest import load_corpus
from bandits.labels import LabelSet, Verdict, compute_label_set_id, make_label
from bandits.store import DerivedStore
from bandits.verify import draft_verifiers
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
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    family = next(f for f in task_set.families if "address" in f.descriptor)
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id, limit=6)
    return analysis, task_set, family, draft


def _labels(family, verdict_for, task_set_id: str = "ts-1") -> LabelSet:
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
    label_set = _labels(family, verdict_for)
    return validate_draft(
        draft,
        "draft-1",
        task_set,
        analysis.evidence,
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
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id, limit=6)
    label_set = _labels(
        family,
        lambda t: Verdict.FAILURE if t in ("refund-4", "refund-9") else Verdict.SUCCESS,
    )

    validation = validate_draft(
        draft, "draft-1", task_set, analysis.evidence, label_set, compute_label_set_id(label_set)
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
        task_set_id="ts-1",
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
        draft, "draft-1", task_set, analysis.evidence, label_set, compute_label_set_id(label_set)
    )

    assert any("not an honest estimate" in limit for limit in validation.limitations)


def test_forging_a_before_and_after_pair_costs_more_than_writing_one_field() -> None:
    """A bare gameable flag would hide why the invariant is the better check."""
    analysis = analyze_corpus(load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    family = next(f for f in task_set.families if "refund" in f.descriptor)
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id, limit=6)
    label_set = _labels(family, lambda t: Verdict.SUCCESS)

    validation = validate_draft(
        draft, "draft-1", task_set, analysis.evidence, label_set, compute_label_set_id(label_set)
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
