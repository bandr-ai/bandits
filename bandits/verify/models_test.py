from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.analyze.models import EvidenceKind
from bandits.verify.models import (
    CheckOperator,
    CheckReview,
    CheckSpec,
    Interpretation,
    InterviewDecision,
    InterviewQuestion,
    Result,
    SubScore,
    VerifierDraft,
    VerifierInterview,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)


def _spec(**updates) -> VerifierSpec:
    values = {
        "verifier_id": "verifier-one",
        "family_id": "family-one",
        "task_set_id": "taskset-one",
        "mode": VerifierMode.REPLAY,
        "inputs": ("terminal_evidence:command_exit_code",),
        "checks": (
            CheckSpec(
                check_id="check-one",
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                supporting_evidence_ids=("ev-one",),
                description="Exit zero.",
            ),
        ),
        "unknown_when": ("exit code absent",),
        "blind_spots": ("wrong command",),
        "gaming_hypotheses": ("run true",),
    }
    values.update(updates)
    return VerifierSpec(**values)


def test_reviewed_requires_validation_and_explicit_human_acceptance() -> None:
    with pytest.raises(ValidationError, match="validation artifact"):
        _spec(status=VerifierStatus.REVIEWED)
    with pytest.raises(ValidationError, match="human acceptance"):
        _spec(status=VerifierStatus.REVIEWED, validation_artifact_id="validation-one")

    reviewed = _spec(
        status=VerifierStatus.REVIEWED,
        validation_artifact_id="validation-one",
        human_acceptance_id="acceptance-one",
    )
    assert reviewed.status is VerifierStatus.REVIEWED


def test_unknown_result_is_none_not_zero() -> None:
    result = Result(
        verifier_id="verifier-one",
        score=None,
        subscores=(SubScore(check_id="check-one", score=None),),
    )
    assert result.score is None


def test_unknown_aggregate_keeps_the_subscores_that_were_known() -> None:
    """Which checks passed while another was unscorable is the record worth having."""
    result = Result(
        verifier_id="verifier-one",
        score=None,
        subscores=(
            SubScore(check_id="check-one", score=1.0),
            SubScore(check_id="check-two", score=None),
        ),
    )

    assert result.score is None
    assert [part.score for part in result.subscores] == [1.0, None]


def test_a_known_aggregate_cannot_sit_on_an_unknown_subscore() -> None:
    """Fail closed: one unscorable check makes the whole verifier unknown."""
    with pytest.raises(ValidationError, match="cannot contain an unknown subscore"):
        Result(
            verifier_id="verifier-one",
            score=1.0,
            subscores=(
                SubScore(check_id="check-one", score=1.0),
                SubScore(check_id="check-two", score=None),
            ),
        )


def test_a_result_without_any_subscore_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one subscore"):
        Result(verifier_id="verifier-one", score=None, subscores=())


def test_replay_verifier_cannot_claim_a_live_input() -> None:
    with pytest.raises(ValidationError, match="replay verifier"):
        _spec(inputs=("live:orders_database",))


def test_a_self_report_only_verifier_cannot_be_promoted() -> None:
    """The agent asserting it finished is never sufficient on its own."""
    self_report = CheckSpec(
        check_id="check-one",
        claim="final_output",
        operator=CheckOperator.EXACT_OUTPUT,
        expected="done",
        supporting_evidence_ids=(),
        description="d",
        evidence_kind=EvidenceKind.AGENT_SELF_REPORT,
    )

    drafted = _spec(checks=(self_report,))
    assert drafted.rests_only_on_self_report

    with pytest.raises(ValidationError, match="stronger than agent self-report"):
        _spec(
            checks=(self_report,),
            status=VerifierStatus.CALIBRATED,
            validation_artifact_id="validation-1",
        )


def test_one_stronger_check_lifts_the_verifier_off_self_report() -> None:
    external = CheckSpec(
        check_id="check-two",
        claim="final_state_field:status",
        operator=CheckOperator.EQUALS,
        expected="refunded",
        supporting_evidence_ids=(),
        description="d",
        evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
    )
    self_report = external.replace(
        check_id="check-one", evidence_kind=EvidenceKind.AGENT_SELF_REPORT
    )

    spec = _spec(
        checks=(self_report, external),
        status=VerifierStatus.CALIBRATED,
        validation_artifact_id="validation-1",
    )

    assert not spec.rests_only_on_self_report
    assert spec.weakest_evidence_kind is EvidenceKind.AGENT_SELF_REPORT


def test_an_expected_question_must_name_its_check() -> None:
    with pytest.raises(ValidationError, match="must name the check"):
        InterviewQuestion(question_id="q1", verifier_id="v1", field="expected", prompt="?")


def _draft() -> VerifierDraft:
    return VerifierDraft(
        task_set_id="taskset-one",
        analysis_id="analysis-one",
        family_id="family-one",
        verifiers=(_spec(),),
    )


def _interpretation(**overrides: object) -> Interpretation:
    return Interpretation(
        **{"decision": InterviewDecision.ACCEPT, "rationale": "reads the right signal", **overrides}
    )


def test_only_a_revise_carries_a_revised_value() -> None:
    with pytest.raises(ValidationError, match="revised value or operator"):
        _interpretation(revised_expected="shipped")


def test_only_a_revise_carries_a_revised_operator() -> None:
    with pytest.raises(ValidationError, match="revised value or operator"):
        _interpretation(revised_operator=CheckOperator.EXACT_OUTPUT)


def test_only_a_combine_names_another_check() -> None:
    with pytest.raises(ValidationError, match="name another check"):
        _interpretation(combine_with="check-two")


def test_a_combine_cannot_both_resolve_and_drop_its_target() -> None:
    with pytest.raises(ValidationError, match="resolve and drop"):
        _interpretation(
            decision=InterviewDecision.COMBINE,
            combine_with="check-two",
            dropped_combine_target="check-nope",
        )


def test_a_dropped_combine_target_survives_without_a_resolved_one() -> None:
    """The decision stands; the human retargets it. Losing it would discard
    the parts of the interpretation the model got right."""
    interpretation = _interpretation(
        decision=InterviewDecision.COMBINE, dropped_combine_target="check-nope"
    )
    assert interpretation.combine_with is None
    assert interpretation.dropped_combine_target == "check-nope"


def _review(**overrides: object) -> CheckReview:
    return CheckReview(
        **{
            "review_id": "review-one",
            "verifier_id": "verifier-one",
            "check_id": "check-one",
            "reply": "fine",
            "decision": InterviewDecision.ACCEPT,
            **overrides,
        }
    )


def test_a_failed_interpretation_cannot_carry_a_result() -> None:
    with pytest.raises(ValidationError, match="failed interpretation"):
        _review(failure="timeout", interpretation=_interpretation())


def test_interview_rejects_duplicate_review_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate review ids"):
        VerifierInterview(
            source_draft_id="draft-one",
            draft=_draft(),
            reviews=(_review(), _review(check_id="check-two")),
        )


def test_interview_rejects_a_supersede_pointing_nowhere() -> None:
    with pytest.raises(ValidationError, match="superseded by unknown review"):
        VerifierInterview(
            source_draft_id="draft-one",
            draft=_draft(),
            reviews=(_review(superseded_by="review-missing"),),
        )


def test_a_review_may_be_superseded_by_a_later_one() -> None:
    interview = VerifierInterview(
        source_draft_id="draft-one",
        draft=_draft(),
        reviews=(
            _review(superseded_by="review-two"),
            _review(
                review_id="review-two", check_id="check-two", decision=InterviewDecision.COMBINE
            ),
        ),
    )
    assert interview.reviews[0].superseded_by == "review-two"


def test_an_interview_needs_no_questions() -> None:
    """The free-text flow records reviews, not question answers."""
    interview = VerifierInterview(source_draft_id="draft-one", draft=_draft())
    assert interview.questions == ()
    assert interview.reviews == ()
