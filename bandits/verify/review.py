"""Promote one calibrated verifier on the strength of a confirmed human review.

Promotion is not a formality, and it is not a second opinion either. Whether a
verifier's evidence is good enough is decided in the review round, by a person
holding the agreement counts, the error composition and the gameability
coverage. This stage proves that decision happened and was about this verifier:
it does not re-litigate it.

That is a change from what this used to do. Promotion once took a free-text
acceptance id and assessed the evidence itself, because at the time nothing put
that evidence in front of anyone — the gate was the only reader. Now that a
reviewer sees it, blocking on the same findings would ask them to inspect a
finding, accept it knowingly, and be refused over it. What is left here is the
part a review cannot attest to about itself: the exact verifier, the exact
validation, every check accepted, nothing superseded.

A revision or a combination mints a new verifier identity and clears the
validation it carried, so a changed verifier cannot promote on the review of the
shape it used to be. It goes back through validation and review, which is the
loop working rather than a refusal.
"""

from __future__ import annotations

import hashlib

from pydantic import model_validator

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract
from bandits.verify.models import (
    CheckReview,
    InterviewDecision,
    VerifierDraft,
    VerifierInterview,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.validate import Validation, accept, calibrate


class PromotionBlocker(Contract):
    """One reason the evidence does not support promoting a verifier."""

    code: str
    detail: str


def assess_promotion(
    validation: Validation,
    verifier_id: str,
    interview: VerifierInterview,
    interview_id: str,
) -> tuple[PromotionBlocker, ...]:
    """Whether a confirmed human review of this exact verifier actually exists.

    Structural only, and deliberately so. Whether the evidence is good enough is
    the reviewer's judgement, made in the interview with the agreement counts,
    the error composition and the gameability coverage in front of them. Judging
    it again here would ask that reviewer to inspect a finding, approve it
    knowingly, and then be refused over the same finding — the second gate this
    stage used to be, back when nothing showed the human anything.

    What is left is the part a human cannot attest to about their own review:
    that it was this verifier, measured by this validation, accepted in whole.
    """
    blockers: list[PromotionBlocker] = []

    if interview.validation_id is None:
        blockers.append(
            PromotionBlocker(
                code="review_saw_no_validation",
                detail=(
                    f"interview {interview_id} was opened without a validation, so the "
                    "reviewer decided before any measurement existed"
                ),
            )
        )

    if not interview.complete:
        # A reviewer reads one check at a time and can stop at any of them, so an
        # abandoned round holds real decisions about the checks it reached. Those
        # decisions were made against a draft the reviewer had not finished
        # judging, and promoting from one silently treats where they stopped as
        # where they finished.
        blockers.append(
            PromotionBlocker(
                code="review_unfinished",
                detail=(
                    f"interview {interview_id} left {len(interview.pending)} check(s) "
                    "undecided; the round has to be finished before what it accepted "
                    "can authorise anything"
                ),
            )
        )

    spec = next(
        (item for item in interview.draft.verifiers if item.verifier_id == verifier_id), None
    )
    if spec is None:
        # A revision or combination mints a new id and clears the validation it
        # carried, so an id that left the round is not a bookkeeping slip: the
        # verifier the reviewer accepted is not the one being promoted.
        blockers.append(
            PromotionBlocker(
                code="verifier_not_in_reviewed_draft",
                detail=(
                    f"{verifier_id} is not in the draft interview {interview_id} produced; "
                    "a revised or combined verifier must be validated and reviewed again"
                ),
            )
        )
        return tuple(blockers)

    if not any(item.verifier_id == verifier_id for item in validation.agreements):
        blockers.append(
            PromotionBlocker(
                code="verifier_not_measured",
                detail=(
                    f"validation carries no measurement for {verifier_id}; the reviewer "
                    "accepted a verifier this validation never scored"
                ),
            )
        )

    superseded = {review.superseded_by for review in interview.reviews if review.superseded_by}
    accepted: dict[str, CheckReview] = {}
    for review in interview.reviews:
        if review.verifier_id != verifier_id or review.review_id in superseded:
            continue
        if review.decision is InterviewDecision.ACCEPT:
            accepted[review.check_id] = review
        else:
            accepted.pop(review.check_id, None)

    unaccepted = sorted({check.check_id for check in spec.checks} - set(accepted))
    if unaccepted:
        blockers.append(
            PromotionBlocker(
                code="checks_not_accepted",
                detail=(
                    f"{len(unaccepted)} check(s) on {verifier_id} carry no standing accept "
                    f"from interview {interview_id}: {', '.join(unaccepted)}"
                ),
            )
        )
    return tuple(blockers)


class ReviewedVerifier(Contract):
    schema_version: int = 1
    source_draft_id: str
    validation_id: str
    human_acceptance_id: str
    spec: VerifierSpec

    success_threshold: float
    """The threshold this verifier was measured at, frozen at acceptance.

    Carried rather than re-defaulted downstream: a run scoring 0.6 is a failure
    to a validation run at 0.8 and a success to an exporter that assumed 0.5, and
    nothing in the artifact would have recorded the disagreement.
    """

    interview_id: str
    """The review round that accepted this verifier.

    Where the reasoning lives: the reviewer's own words, what the model made of
    them, the confirmed decision and whether they held the evidence source
    authoritative are all recorded there, against the measurements they were
    shown. Referenced rather than copied so a reader lands on the whole round
    instead of a sentence lifted out of it.
    """

    @model_validator(mode="after")
    def is_actually_reviewed(self) -> ReviewedVerifier:
        if self.spec.status is not VerifierStatus.REVIEWED:
            raise ValueError("a reviewed-verifier artifact must contain an accepted spec")
        if self.spec.validation_artifact_id != self.validation_id:
            raise ValueError("reviewed verifier and validation id disagree")
        if self.spec.human_acceptance_id != self.human_acceptance_id:
            raise ValueError("reviewed verifier and acceptance id disagree")
        if self.human_acceptance_id != self.interview_id:
            raise ValueError("a promotion must be accepted by the review it names")
        if not 0 <= self.success_threshold <= 1:
            raise ValueError("success threshold must be in [0, 1]")
        return self


def review_verifier(
    draft: VerifierDraft,
    draft_id: str,
    validation: Validation,
    validation_id: str,
    verifier_id: str,
    interview: VerifierInterview,
    interview_id: str,
) -> ReviewedVerifier:
    """Promote one measured spec on the strength of a confirmed human review.

    There is no override flag. The decision this promotion records was already
    made by a person looking at the measurements, so there is nothing here for
    them to override — and a bare acceptance string, which is what this used to
    take, asserted a review that may never have happened.
    """
    if validation.source_draft_id != draft_id:
        raise ValueError("validation was produced from another verifier draft")
    if validation.family_id != draft.family_id:
        raise ValueError("validation and verifier draft belong to different families")
    if interview.source_draft_id != draft_id:
        raise ValueError("the review was conducted against another verifier draft")
    if interview.validation_id is not None and interview.validation_id != validation_id:
        raise ValueError(
            f"the review read validation {interview.validation_id}, not {validation_id}"
        )
    spec = next((item for item in draft.verifiers if item.verifier_id == verifier_id), None)
    if spec is None:
        raise ValueError(f"unknown verifier id: {verifier_id!r}")

    blockers = assess_promotion(validation, verifier_id, interview, interview_id)
    if blockers:
        detail = "; ".join(f"{item.code}: {item.detail}" for item in blockers)
        raise ValueError(f"no confirmed review supports promoting {verifier_id!r} — {detail}")

    reviewed = accept(calibrate(spec, validation_id), interview_id)
    return ReviewedVerifier(
        source_draft_id=draft_id,
        validation_id=validation_id,
        human_acceptance_id=interview_id,
        interview_id=interview_id,
        spec=reviewed,
        success_threshold=validation.success_threshold,
    )


def compute_reviewed_verifier_id(reviewed: ReviewedVerifier) -> str:
    digest = hashlib.sha256(reviewed.model_dump_json().encode()).hexdigest()
    return f"reviewed-verifier-{digest[:16]}"


def save_reviewed_verifier(reviewed: ReviewedVerifier, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_reviewed_verifier_id(reviewed),
        kind="reviewed_verifier",
        parent_artifact_id=reviewed.validation_id,
        payload=reviewed.model_dump_json().encode(),
        summary={"verifiers": 1},
    )


def load_reviewed_verifier(reviewed_id: str, store: DerivedStore) -> ReviewedVerifier:
    return ReviewedVerifier.model_validate_json(store.read_payload(reviewed_id))
