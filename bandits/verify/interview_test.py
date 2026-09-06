from __future__ import annotations

import pytest

from bandits.store import DerivedStore
from bandits.verify import (
    CheckOperator,
    CheckReview,
    CheckSpec,
    Interpretation,
    InterviewDecision,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
    answer_question,
    apply_decision,
    load_interview,
    next_question,
    prior_decisions,
    save_interview,
    start_interview,
    start_review,
)


def _draft() -> VerifierDraft:
    spec = VerifierSpec(
        verifier_id="verifier-one",
        family_id="family-one",
        task_set_id="taskset-one",
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=("terminal_evidence:final_state_field:status",),
        checks=(
            CheckSpec(
                check_id="check-one",
                claim="final_state_field:status",
                operator=CheckOperator.EQUALS,
                expected="done",
                supporting_evidence_ids=("ev-one",),
                description="Status is done.",
            ),
        ),
        unknown_when=("status missing",),
        blind_spots=("status alone may be insufficient",),
        gaming_hypotheses=("set status directly",),
    )
    return VerifierDraft(
        task_set_id="taskset-one",
        analysis_id="analysis-one",
        family_id="family-one",
        verifiers=(spec,),
    )


def test_interview_is_bounded_and_asks_one_question_at_a_time() -> None:
    interview = start_interview(_draft(), "draft-one")

    assert len(interview.questions) == 3
    assert next_question(interview).field == "expected"
    interview = answer_question(interview, '"completed"')
    assert next_question(interview).field == "blind_spots"
    interview = answer_question(interview, "amount may be wrong")
    assert next_question(interview).field == "gaming_hypotheses"
    interview = answer_question(interview, "write status without refunding")

    assert interview.complete
    assert next_question(interview) is None
    assert interview.draft.verifiers[0].checks[0].expected == "completed"


def test_revision_changes_only_the_answer_target() -> None:
    original = _draft()
    interview = answer_question(start_interview(original, "draft-one"), '"completed"')
    before = original.verifiers[0]
    after = interview.draft.verifiers[0]

    assert after.checks[0].expected == "completed"
    assert after.blind_spots == before.blind_spots
    assert after.gaming_hypotheses == before.gaming_hypotheses
    assert after.status is VerifierStatus.EXECUTABLE


def test_completed_interview_refuses_extra_answers() -> None:
    interview = start_interview(_draft(), "draft-one")
    for _ in range(3):
        interview = answer_question(interview, "")

    with pytest.raises(ValueError, match="already complete"):
        answer_question(interview, "extra")


def test_interview_round_trips_with_draft_as_parent(tmp_path) -> None:
    interview = start_interview(_draft(), "draft-one")
    for _ in range(3):
        interview = answer_question(interview, "")
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_interview(interview, store)

    assert envelope.parent_artifact_id == "draft-one"
    assert load_interview(envelope.artifact_id, store) == interview


def _two_check_draft() -> VerifierDraft:
    def spec(n: str, claim: str, expected: str) -> VerifierSpec:
        return VerifierSpec(
            verifier_id=f"verifier-{n}",
            family_id="family-one",
            task_set_id="taskset-one",
            mode=VerifierMode.REPLAY,
            status=VerifierStatus.EXECUTABLE,
            inputs=(f"terminal_evidence:{claim}",),
            checks=(
                CheckSpec(
                    check_id=f"check-{n}",
                    claim=claim,
                    operator=CheckOperator.EQUALS,
                    expected=expected,
                    supporting_evidence_ids=(f"ev-{n}",),
                    description=f"{claim} is {expected}.",
                ),
            ),
            unknown_when=(f"{claim} missing",),
            blind_spots=(f"{n} blind",),
            gaming_hypotheses=(f"{n} gaming",),
        )

    return VerifierDraft(
        task_set_id="taskset-one",
        analysis_id="analysis-one",
        family_id="family-one",
        verifiers=(
            spec("one", "final_state_field:status", "done"),
            spec("two", "final_state_field:refund", "issued"),
        ),
    )


def _review(check_id: str, decision: InterviewDecision, **overrides) -> CheckReview:
    verifier = {"check-one": "verifier-one", "check-two": "verifier-two"}[check_id]
    values = {
        "review_id": f"review-{check_id}",
        "verifier_id": overrides.pop("verifier_id", verifier),
        "check_id": check_id,
        "reply": "because",
        "decision": decision,
    }
    return CheckReview(**{**values, **overrides})


def test_start_review_queues_every_check() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    assert interview.pending == (("verifier-one", "check-one"), ("verifier-two", "check-two"))
    assert not interview.complete
    assert interview.round_number == 1


def test_accept_leaves_the_spec_alone() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    after = apply_decision(interview, _review("check-one", InterviewDecision.ACCEPT))
    assert after.draft.verifiers[0] == interview.draft.verifiers[0]
    assert after.pending == (("verifier-two", "check-two"),)


def test_reject_marks_the_spec_and_keeps_it_auditable() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    after = apply_decision(
        interview, _review("check-one", InterviewDecision.REJECT, reply="wrong signal")
    )
    assert after.draft.verifiers[0].status is VerifierStatus.REJECTED
    assert after.reviews[0].reply == "wrong signal"


def test_revise_mints_a_new_id_and_clears_stale_evidence() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    before = interview.draft.verifiers[0]
    after = apply_decision(
        interview,
        _review(
            "check-one",
            InterviewDecision.REVISE,
            interpretation=Interpretation(
                decision=InterviewDecision.REVISE,
                rationale="shipped is the real terminal value",
                revised_expected="shipped",
            ),
        ),
    )
    revised = after.draft.verifiers[0]
    assert revised.verifier_id != before.verifier_id
    assert revised.checks[0].expected == "shipped"
    assert revised.checks[0].supporting_evidence_ids == ()
    assert revised.provenance == "human"


def test_extracted_hypotheses_land_on_the_spec() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    after = apply_decision(
        interview,
        _review(
            "check-one",
            InterviewDecision.ACCEPT,
            interpretation=Interpretation(
                decision=InterviewDecision.ACCEPT,
                rationale="fine",
                blind_spots=("status can lag",),
                gaming_hypotheses=("write status without the refund",),
            ),
        ),
    )
    assert "status can lag" in after.draft.verifiers[0].blind_spots
    assert "write status without the refund" in after.draft.verifiers[0].gaming_hypotheses


def _combine_review() -> CheckReview:
    return _review(
        "check-one",
        InterviewDecision.COMBINE,
        reply="fine, but combine with the refund check",
        interpretation=Interpretation(
            decision=InterviewDecision.COMBINE,
            rationale="both must hold",
            combine_with="check-two",
        ),
    )


def test_combine_folds_two_verifiers_into_one_new_identity() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    after = apply_decision(interview, _combine_review())

    assert len(after.draft.verifiers) == 1
    combined = after.draft.verifiers[0]
    assert combined.verifier_id not in {"verifier-one", "verifier-two"}
    assert combined.source_verifier_ids == ("verifier-one", "verifier-two")
    assert len(combined.checks) == 2
    assert combined.provenance == "human"


def test_combine_unions_unknown_when_and_clears_evidence() -> None:
    after = apply_decision(start_review(_two_check_draft(), "draft-one"), _combine_review())
    combined = after.draft.verifiers[0]
    assert set(combined.unknown_when) == {
        "final_state_field:status missing",
        "final_state_field:refund missing",
    }
    assert all(check.supporting_evidence_ids == () for check in combined.checks)
    assert set(combined.blind_spots) == {"one blind", "two blind"}


def test_combine_carries_weights_unchanged() -> None:
    after = apply_decision(start_review(_two_check_draft(), "draft-one"), _combine_review())
    assert [check.weight for check in after.draft.verifiers[0].checks] == [1.0, 1.0]


def test_combine_clears_the_other_check_from_the_queue() -> None:
    after = apply_decision(start_review(_two_check_draft(), "draft-one"), _combine_review())
    assert after.pending == ()
    assert after.complete


def test_combining_with_an_already_decided_check_supersedes_it() -> None:
    """A decision already made is never silently overwritten."""
    interview = start_review(_two_check_draft(), "draft-one")
    interview = apply_decision(
        interview, _review("check-two", InterviewDecision.ACCEPT, review_id="review-early")
    )
    after = apply_decision(interview, _combine_review())

    early = [r for r in after.reviews if r.review_id == "review-early"][0]
    assert early.superseded_by == "review-check-one"


def test_a_decision_for_a_check_not_pending_is_refused() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    interview = apply_decision(interview, _review("check-one", InterviewDecision.ACCEPT))
    with pytest.raises(ValueError, match="not awaiting a decision"):
        apply_decision(interview, _review("check-one", InterviewDecision.ACCEPT))


def test_a_revise_without_an_interpretation_is_refused() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    with pytest.raises(ValueError, match="needs an interpretation"):
        apply_decision(interview, _review("check-one", InterviewDecision.REVISE))


def test_prior_decisions_reads_back_what_a_round_decided() -> None:
    interview = start_review(_two_check_draft(), "draft-one")
    interview = apply_decision(
        interview, _review("check-one", InterviewDecision.ACCEPT, reply="looks right")
    )
    assert prior_decisions(interview, "check-one") == ("round 1: accept — looks right",)


def test_a_later_round_chains_to_the_one_before() -> None:
    first = start_review(_two_check_draft(), "draft-one")
    second = start_review(
        first.draft,
        "draft-one",
        validation_id="validation-xyz",
        prior_interview_id="interview-aaa",
        round_number=2,
    )
    assert second.prior_interview_id == "interview-aaa"
    assert second.validation_id == "validation-xyz"
    assert second.round_number == 2
