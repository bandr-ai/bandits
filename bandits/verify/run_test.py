"""Running drafts over history, and finding where they disagree."""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.analyze.models import EvidenceKind
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.verify import (
    Agreement,
    GameabilityAssessment,
    GameabilityResult,
    Validation,
    build_check_summary,
    draft_verifiers,
    load_draft_run,
    run_draft,
    save_draft_run,
)
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
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
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
    """The two failed refunds recorded no terminal state at all."""
    analysis, task_set, family = context
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)

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


def _summary_for(context, **kwargs):
    analysis, task_set, family = context
    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id, limit=3)
    run = run_draft(draft, analysis, task_set)
    spec = draft.verifiers[0]
    return build_check_summary(spec, spec.checks[0], run, **kwargs), spec, run


def test_a_summary_counts_what_the_check_did(context) -> None:
    summary, spec, run = _summary_for(context)
    scored = summary.passed + summary.failed + summary.unscorable
    assert scored == sum(1 for o in run.outcomes if o.verifier_id == spec.verifier_id)
    assert summary.check_id == spec.checks[0].check_id


def test_a_summary_bounds_its_examples(context) -> None:
    summary, _, _ = _summary_for(context, examples=2)
    assert len(summary.example_trace_ids) <= 2


def test_prompt_lines_never_carry_trace_ids(context) -> None:
    """Trace ids are for the reviewer to open, not for a model to read."""
    summary, _, _ = _summary_for(context)
    rendered = "\n".join(summary.prompt_lines())
    for trace_id in summary.example_trace_ids:
        assert trace_id not in rendered


def test_a_check_that_never_fails_says_so(context) -> None:
    summary, _, _ = _summary_for(context)
    if summary.passed and not summary.failed:
        assert any("discriminating" in line for line in summary.prompt_lines())


def test_validation_results_reach_the_summary_and_the_prompt(context) -> None:
    """The round-two half: the reviewer answers these numbers, so the model sees them."""
    summary, spec, run = _summary_for(context)
    validation = Validation(
        source_draft_id="draft-one",
        family_id=spec.family_id,
        label_set_id="labels-one",
        success_threshold=1.0,
        agreements=(
            Agreement(
                verifier_id=spec.verifier_id,
                split="held_out",
                labeled=10,
                agreed=5,
                disagreed=5,
                unscored=0,
                agreement=0.5,
                false_positives=3,
                false_negatives=2,
                successes_agreed=4,
            ),
            Agreement(
                verifier_id="verifier-someone-else",
                split="held_out",
                labeled=4,
                agreed=4,
                disagreed=0,
                unscored=0,
                agreement=1.0,
            ),
        ),
        gameability=(
            GameabilityResult(
                verifier_id=spec.verifier_id,
                hypothesis="Write the field directly.",
                constructed={},
                passed=True,
                forged_facts=1,
            ),
        ),
    )
    with_validation = build_check_summary(spec, spec.checks[0], run, validation=validation)

    assert len(with_validation.agreements) == 1
    assert with_validation.agreements[0].verifier_id == spec.verifier_id
    rendered = "\n".join(with_validation.prompt_lines())
    assert "agreement held_out: 0.50" in rendered
    assert "Write the field directly." in rendered
    # The rate alone cannot say which way the errors went. A reviewer answering
    # "0.50" needs to know three of those five passed a run a human failed.
    assert "errors held_out: 3 false positive(s), 2 false negative(s)" in rendered
    assert "failures caught held_out: 0.25 (1 of 4)" in rendered


def test_without_a_validation_the_summary_says_so_rather_than_going_quiet(context) -> None:
    """Unmeasured and measured-as-good must not both render as silence."""
    summary, _, _ = _summary_for(context)
    assert summary.agreements == ()
    assert summary.gameability == ()

    rendered = "\n".join(summary.prompt_lines())

    assert "agreement: unavailable" in rendered
    assert "0." not in rendered.partition("agreement:")[2].partition("\n")[0]


def test_gameability_coverage_reaches_the_reviewer(context) -> None:
    """A verifier nothing could attack must not read like one that resisted."""
    summary, spec, run = _summary_for(context)
    validation = Validation(
        source_draft_id="draft-one",
        family_id=spec.family_id,
        label_set_id="labels-one",
        agreements=(
            Agreement(
                verifier_id=spec.verifier_id,
                split="held_out",
                labeled=2,
                agreed=2,
                disagreed=0,
                unscored=0,
                agreement=1.0,
                successes_agreed=2,
            ),
        ),
        gameability_assessments=(
            GameabilityAssessment(
                verifier_id=spec.verifier_id,
                coverage="none",
                attack_succeeded=False,
                checks_attacked=0,
                checks_total=len(spec.checks),
            ),
        ),
    )

    rendered = "\n".join(
        build_check_summary(spec, spec.checks[0], run, validation=validation).prompt_lines()
    )

    assert "gameability coverage: none" in rendered
    assert "never tried" in rendered


def test_a_resisted_attack_is_shown_not_only_a_successful_one(context) -> None:
    """A resisted attack is evidence; silence about it is not."""
    summary, spec, run = _summary_for(context)
    validation = Validation(
        source_draft_id="draft-one",
        family_id=spec.family_id,
        label_set_id="labels-one",
        agreements=(
            Agreement(
                verifier_id=spec.verifier_id,
                split="held_out",
                labeled=1,
                agreed=1,
                disagreed=0,
                unscored=0,
                agreement=1.0,
                successes_agreed=1,
            ),
        ),
        gameability=(
            GameabilityResult(
                verifier_id=spec.verifier_id,
                hypothesis="Write the field directly.",
                constructed={},
                passed=False,
                forged_facts=1,
            ),
        ),
    )

    rendered = "\n".join(
        build_check_summary(spec, spec.checks[0], run, validation=validation).prompt_lines()
    )

    assert "resisted" in rendered
