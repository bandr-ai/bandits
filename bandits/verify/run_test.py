"""Running drafts over history, and finding where they disagree."""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.analyze.models import EvidenceKind
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.verify import draft_verifiers, load_draft_run, run_draft, save_draft_run
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _test_distance(left: str, right: str) -> float:
    return 0.0 if left.partition(" ")[0] == right.partition(" ")[0] else 1.0


@pytest.fixture
def context():
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    analysis = analyze_corpus(corpus)
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
    )
    family = max(task_set.families, key=lambda f: f.workload_mass)
    return analysis, task_set, family


def _spec(verifier_id: str, family_id: str, *, claim: str, expected: object) -> VerifierSpec:
    return VerifierSpec(
        verifier_id=verifier_id,
        family_id=family_id,
        task_set_id="ts-1",
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=(),
        checks=(
            CheckSpec(
                check_id=f"check-{verifier_id}",
                claim=claim,
                operator=CheckOperator.EQUALS,
                expected=expected,
                supporting_evidence_ids=(),
                description="d",
                evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
            ),
        ),
        unknown_when=(),
        blind_spots=(),
        gaming_hypotheses=(),
    )


def test_a_run_scores_every_fit_trace_with_every_verifier(context) -> None:
    analysis, task_set, family = context
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)

    run = run_draft(draft, analysis, task_set)

    scored = {o.trace_id for o in run.outcomes}
    assert scored == set(family.fit_trace_ids)
    assert len(run.outcomes) == len(scored) * len(draft.verifiers)


def test_the_held_out_side_is_not_spent_on_the_draft(context) -> None:
    """Scoring held-out traces here would consume the split before calibration."""
    analysis, task_set, family = context
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)

    run = run_draft(draft, analysis, task_set)

    assert not {o.trace_id for o in run.outcomes} & set(family.held_out_trace_ids)


def test_a_run_nothing_can_score_is_unscorable_not_failed(context) -> None:
    """The two failed refunds recorded no terminal state at all.

    Drafted here from state-field checks alone rather than from whatever
    ``draft_verifiers`` ranks highest. A draft containing a check that reads the
    episode itself — no_span_error scores every trace in the family — has no
    unscorable trace in it by construction, and this is a test about what
    ``run_draft`` does when nothing can see the evidence it needs.
    """
    analysis, task_set, family = context
    draft = VerifierDraft(
        task_set_id="ts-1",
        analysis_id="analysis-1",
        family_id=family.family_id,
        verifiers=(
            _spec(
                "verifier-status",
                family.family_id,
                claim="final_state_field:refund_order.status",
                expected="refunded",
            ),
            _spec(
                "verifier-amount",
                family.family_id,
                claim="final_state_field:refund_order.refunded_amount",
                expected=100,
            ),
        ),
    )

    run = run_draft(draft, analysis, task_set)

    assert run.unscorable_trace_ids
    for trace_id in run.unscorable_trace_ids:
        scores = run.scores_for(trace_id)
        assert set(scores.values()) == {None}, "absent evidence must never read as zero"


def test_verifiers_that_split_on_a_trace_are_surfaced(context) -> None:
    """Two checks on the same field with different expectations must disagree."""
    analysis, task_set, family = context
    draft = VerifierDraft(
        task_set_id="ts-1",
        analysis_id=task_set.analysis_id,
        family_id=family.family_id,
        verifiers=(
            _spec(
                "verifier-agree",
                family.family_id,
                claim="final_state_field:status",
                expected="refunded",
            ),
            _spec(
                "verifier-differ",
                family.family_id,
                claim="final_state_field:status",
                expected="settled",
            ),
        ),
    )

    run = run_draft(draft, analysis, task_set)
    splits = [d for d in run.disagreements if d.kind == "split"]

    assert splits, "a trace scored 1.0 by one verifier and 0.0 by another is a disagreement"
    for split in splits:
        assert set(split.scores.values()) == {1.0, 0.0}


def test_partial_coverage_is_flagged_for_labeling(context) -> None:
    """One check sees the run and another is blind to it; only a human can rank that."""
    analysis, task_set, family = context
    draft = VerifierDraft(
        task_set_id="ts-1",
        analysis_id=task_set.analysis_id,
        family_id=family.family_id,
        verifiers=(
            _spec(
                "verifier-seen",
                family.family_id,
                claim="final_state_field:status",
                expected="refunded",
            ),
            _spec(
                "verifier-blind", family.family_id, claim="final_state_field:absent", expected="x"
            ),
        ),
    )

    run = run_draft(draft, analysis, task_set)
    coverage = [d for d in run.disagreements if d.kind == "coverage"]

    assert coverage
    assert all(None in d.scores.values() for d in coverage)


def test_run_rejects_a_draft_for_a_family_the_task_set_does_not_have(context) -> None:
    analysis, task_set, _ = context
    # Internally consistent, so the draft's own validator passes; the family
    # simply is not in this task set.
    draft = VerifierDraft(
        task_set_id="ts-1",
        analysis_id=task_set.analysis_id,
        family_id="family-nope",
        verifiers=(_spec("verifier-x", "family-nope", claim="final_state_field:s", expected="x"),),
    )

    with pytest.raises(ValueError, match="unknown family"):
        run_draft(draft, analysis, task_set)


def test_a_run_round_trips_through_the_derived_store(tmp_path, context) -> None:
    analysis, task_set, family = context
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)
    store = DerivedStore(tmp_path / ".bandits")

    run = run_draft(draft, analysis, task_set)
    envelope = save_draft_run(run, store)

    assert envelope.kind == "verifier_run"
    assert load_draft_run(envelope.artifact_id, store) == run
