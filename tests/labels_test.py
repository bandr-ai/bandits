from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.analyze.models import EvidenceKind, Visibility
from bandits.labels import (
    LabelSet,
    Verdict,
    load_label_set,
    make_label,
    save_label_set,
)
from bandits.store import DerivedStore


def _label(trace_id: str, verdict: Verdict, family_id: str = "family-one"):
    return make_label(trace_id=trace_id, family_id=family_id, verdict=verdict, labeler="owner")


def test_a_label_is_evidence_and_ranks_as_a_human_label() -> None:
    evidence = _label("trace-one", Verdict.SUCCESS).as_evidence()

    assert evidence.kind is EvidenceKind.HUMAN_LABEL
    assert evidence.provenance == "human"
    assert evidence.visibility is Visibility.POST_HOC
    assert evidence.strength == "strong"


def test_a_model_label_never_masquerades_as_human_evidence() -> None:
    label = make_label(
        trace_id="trace-one",
        family_id="family-one",
        verdict=Verdict.SUCCESS,
        labeler="judge-model",
        source="model",
    )

    evidence = label.as_evidence()

    assert evidence.claim == "model_label"
    assert evidence.provenance == "model"
    assert evidence.kind is EvidenceKind.MODEL_JUDGMENT
    assert evidence.strength == "moderate"


def test_a_verdict_nobody_could_reach_is_weak_evidence() -> None:
    assert _label("trace-one", Verdict.UNCLEAR).as_evidence().strength == "weak"


def test_unclear_verdicts_are_kept_but_do_not_adjudicate() -> None:
    """Dropping them would inflate every agreement rate computed afterwards."""
    label_set = LabelSet(
        task_set_id="ts-1",
        family_id="family-one",
        labels=(
            _label("trace-one", Verdict.SUCCESS),
            _label("trace-two", Verdict.UNCLEAR),
        ),
    )

    assert len(label_set.labels) == 2
    assert set(label_set.adjudicated()) == {"trace-one"}


def test_one_trace_cannot_carry_two_verdicts() -> None:
    with pytest.raises(ValidationError, match="at most one verdict per trace"):
        LabelSet(
            task_set_id="ts-1",
            family_id="family-one",
            labels=(
                _label("trace-one", Verdict.SUCCESS),
                _label("trace-one", Verdict.FAILURE),
            ),
        )


def test_a_label_for_another_family_is_rejected() -> None:
    with pytest.raises(ValidationError, match="belongs to another family"):
        LabelSet(
            task_set_id="ts-1",
            family_id="family-one",
            labels=(_label("trace-one", Verdict.SUCCESS, family_id="family-two"),),
        )


def test_label_ids_are_stable_for_the_same_labeler_and_trace() -> None:
    assert _label("trace-one", Verdict.SUCCESS).label_id == (
        _label("trace-one", Verdict.FAILURE).label_id
    )


def test_a_label_set_round_trips_through_the_derived_store(tmp_path) -> None:
    store = DerivedStore(tmp_path / ".bandits")
    label_set = LabelSet(
        task_set_id="ts-1",
        family_id="family-one",
        labels=(_label("trace-one", Verdict.SUCCESS), _label("trace-two", Verdict.UNCLEAR)),
    )

    envelope = save_label_set(label_set, store)

    assert envelope.kind == "label_set"
    assert envelope.summary == {"labels": 2, "adjudicated": 1}
    assert load_label_set(envelope.artifact_id, store) == label_set
