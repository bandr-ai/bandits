from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.verify.interview import apply_decision, start_review
from bandits.verify.models import (
    CheckOperator,
    CheckReview,
    CheckSpec,
    InterviewDecision,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.review import (
    assess_promotion,
    load_reviewed_verifier,
    review_verifier,
    save_reviewed_verifier,
)
from bandits.verify.validate import Agreement, Validation


def _inputs():
    spec = VerifierSpec(
        verifier_id="v1",
        family_id="f1",
        task_set_id="ts1",
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=("terminal_evidence:command_exit_code",),
        checks=(
            CheckSpec(
                check_id="c1",
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                supporting_evidence_ids=("e1",),
                description="exit zero",
            ),
        ),
        unknown_when=("missing exit",),
        blind_spots=("wrong command",),
        gaming_hypotheses=("run true",),
    )
    draft = VerifierDraft(task_set_id="ts1", analysis_id="a1", family_id="f1", verifiers=(spec,))
    validation = Validation(
        source_draft_id="d1",
        family_id="f1",
        label_set_id="l1",
        agreements=(
            Agreement(
                verifier_id="v1",
                split="held_out",
                labeled=1,
                agreed=1,
                disagreed=0,
                unscored=0,
                agreement=1,
            ),
        ),
        labels_used=2,
        success_labels=1,
        failure_labels=1,
    )
    return draft, validation


def _review(draft, draft_id="d1", validation_id="val1", *, decision=None, verifier_id=None):
    """A completed round, accepting every check unless told otherwise."""
    interview = start_review(draft, draft_id, validation_id=validation_id, round_number=2)
    for index, (spec_id, check_id) in enumerate(interview.pending, start=1):
        chosen = decision if decision and (verifier_id in (None, spec_id)) else None
        interview = apply_decision(
            interview,
            CheckReview(
                review_id=f"review-{index:03d}-{check_id}",
                verifier_id=spec_id,
                check_id=check_id,
                reply="reads the right field",
                decision=chosen or InterviewDecision.ACCEPT,
                authoritative=True,
            ),
        )
    return interview


def test_review_requires_artifacts_that_belong_to_one_chain() -> None:
    draft, validation = _inputs()
    interview = _review(draft)

    with pytest.raises(ValueError, match="another verifier draft"):
        review_verifier(draft, "wrong", validation, "val1", "v1", interview, "i1")

    elsewhere = _review(draft, draft_id="other")
    with pytest.raises(ValueError, match="conducted against another verifier draft"):
        review_verifier(draft, "d1", validation, "val1", "v1", elsewhere, "i1")

    stale = _review(draft, validation_id="val-earlier")
    with pytest.raises(ValueError, match="read validation val-earlier"):
        review_verifier(draft, "d1", validation, "val1", "v1", stale, "i1")


def test_a_verifier_the_validation_never_scored_cannot_promote() -> None:
    """The reviewer accepted a check no measurement in this validation covers."""
    draft, validation = _inputs()

    blockers = assess_promotion(validation.replace(agreements=()), "v1", _review(draft), "i1")

    assert {item.code for item in blockers} == {"verifier_not_measured"}


def test_a_review_that_saw_no_validation_cannot_promote() -> None:
    """Round one decides before anything is measured; that is not an informed accept."""
    draft, validation = _inputs()
    blind = start_review(draft, "d1", round_number=1)
    blind = apply_decision(
        blind,
        CheckReview(
            review_id="review-001-c1",
            verifier_id="v1",
            check_id="c1",
            reply="looks fine",
            decision=InterviewDecision.ACCEPT,
        ),
    )

    blockers = assess_promotion(validation, "v1", blind, "i1")

    assert {item.code for item in blockers} == {"review_saw_no_validation"}


def test_a_check_the_reviewer_did_not_accept_blocks_promotion() -> None:
    draft, validation = _inputs()
    rejected = _review(draft, decision=InterviewDecision.REJECT)

    blockers = assess_promotion(validation, "v1", rejected, "i1")

    assert {item.code for item in blockers} == {"checks_not_accepted"}


def test_a_superseded_accept_no_longer_stands() -> None:
    """A combination reopens an earlier decision; the accept it replaced is spent."""
    draft, validation = _inputs()
    interview = _review(draft)
    spent = interview.replace(
        reviews=tuple(
            review.replace(superseded_by=review.review_id) for review in interview.reviews
        )
    )

    blockers = assess_promotion(validation, "v1", spent, "i1")

    assert {item.code for item in blockers} == {"checks_not_accepted"}


def test_a_revised_verifier_cannot_promote_on_the_old_validation() -> None:
    """Revision mints a new id, so the accepted verifier is not the promoted one."""
    draft, validation = _inputs()
    interview = _review(draft)
    departed = interview.replace(
        draft=interview.draft.replace(
            verifiers=tuple(
                spec.replace(verifier_id="v1-revised") for spec in interview.draft.verifiers
            )
        )
    )

    blockers = assess_promotion(validation, "v1", departed, "i1")

    assert {item.code for item in blockers} == {"verifier_not_in_reviewed_draft"}


def test_promotion_refuses_without_a_confirmed_review() -> None:
    """A bare acceptance string used to be enough. It no longer reaches the call."""
    draft, validation = _inputs()
    rejected = _review(draft, decision=InterviewDecision.REJECT)

    with pytest.raises(ValueError, match="no confirmed review supports promoting"):
        review_verifier(draft, "d1", validation, "val1", "v1", rejected, "i1")


def test_a_confirmed_review_promotes_and_is_referenced() -> None:
    draft, validation = _inputs()

    reviewed = review_verifier(draft, "d1", validation, "val1", "v1", _review(draft), "i1")

    assert reviewed.spec.status is VerifierStatus.REVIEWED
    # The reasoning lives in the round, so the artifact points at it rather than
    # copying a sentence out of it.
    assert reviewed.interview_id == "i1"
    assert reviewed.human_acceptance_id == "i1"


def test_review_round_trips_as_immutable_derived_artifact(tmp_path) -> None:
    from bandits.store import DerivedStore

    draft, validation = _inputs()
    reviewed = review_verifier(draft, "d1", validation, "val1", "v1", _review(draft), "i1")
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_reviewed_verifier(reviewed, store)

    assert load_reviewed_verifier(envelope.artifact_id, store) == reviewed
    assert reviewed.spec.status is VerifierStatus.REVIEWED


def test_a_promotion_cannot_name_a_review_it_was_not_accepted_by() -> None:
    draft, validation = _inputs()
    reviewed = review_verifier(draft, "d1", validation, "val1", "v1", _review(draft), "i1")

    with pytest.raises(ValidationError, match="accepted by the review it names"):
        reviewed.replace(interview_id="some-other-round")


def test_an_unfinished_review_cannot_promote_what_it_reached() -> None:
    """A reviewer can stop at any check, and where they stopped is not where they finished."""
    draft, validation = _inputs()
    interview = _review(draft)
    abandoned = interview.replace(complete=False, pending=(("v1", "c-unreviewed"),))

    blockers = assess_promotion(validation, "v1", abandoned, "i1")

    assert {item.code for item in blockers} == {"review_unfinished"}
    with pytest.raises(ValueError, match="review_unfinished"):
        review_verifier(draft, "d1", validation, "val1", "v1", abandoned, "i1")
