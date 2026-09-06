"""Owner review of deterministic verifier drafts.

Two flows live here. The older one asks a fixed sequence of narrow questions
per check — ``start_interview`` / ``next_question`` / ``answer_question`` — and
is kept because recorded interviews still load and replay through it.

The newer one asks a single open question per check and reads the answer:
``start_review`` queues the checks, ``bandits.verify.interpret`` maps a
free-text reply to a proposed decision, and ``apply_decision`` records what the
owner confirmed. A fixed form cannot carry the objection that fits none of its
slots — "you are checking the wrong signal entirely" — and prose typed into one
was never read by anything downstream.
"""

from __future__ import annotations

import hashlib
import json

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.verify.draft import _spec_id
from bandits.verify.models import (
    CheckOperator,
    CheckReview,
    CheckSpec,
    Interpretation,
    InterviewAnswer,
    InterviewDecision,
    InterviewQuestion,
    VerifierDraft,
    VerifierInterview,
    VerifierSpec,
    VerifierStatus,
)

_HAS_EXPECTED_VALUE = frozenset({CheckOperator.EQUALS, CheckOperator.EXACT_OUTPUT})


def _questions(draft: VerifierDraft) -> tuple[InterviewQuestion, ...]:
    questions: list[InterviewQuestion] = []
    for spec in draft.verifiers:
        questions.extend(
            InterviewQuestion(
                question_id=f"q-{spec.verifier_id}-{check.check_id}-expected",
                verifier_id=spec.verifier_id,
                check_id=check.check_id,
                field="expected",
                current_value=check.expected,
                prompt=(
                    f"For {check.claim}, what exact value represents success? "
                    f"Press Enter to keep {check.expected!r}."
                ),
            )
            # Only checks that compare against a value have one to confirm. An
            # invariant compares two observed fields and a no-error check
            # compares nothing; asking their expected value is a question with
            # no answer, and it teaches the owner the interview is boilerplate.
            for check in spec.checks
            if check.operator in _HAS_EXPECTED_VALUE
        )
        questions.extend(
            (
                InterviewQuestion(
                    question_id=f"q-{spec.verifier_id}-blind-spot",
                    verifier_id=spec.verifier_id,
                    field="blind_spots",
                    prompt="Name one additional blind spot, or press Enter if none.",
                ),
                InterviewQuestion(
                    question_id=f"q-{spec.verifier_id}-gaming",
                    verifier_id=spec.verifier_id,
                    field="gaming_hypotheses",
                    prompt="Name one additional way this check could be gamed, or press Enter if none.",
                ),
            )
        )
    return tuple(questions)


def start_interview(draft: VerifierDraft, source_draft_id: str) -> VerifierInterview:
    questions = _questions(draft)
    return VerifierInterview(
        source_draft_id=source_draft_id,
        draft=draft,
        questions=questions,
        complete=not questions,
    )


def next_question(interview: VerifierInterview) -> InterviewQuestion | None:
    if interview.complete:
        return None
    return interview.questions[interview.next_question_index]


def _parse_expected(value: str, current: object) -> object:
    if not value.strip():
        return current
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def answer_question(interview: VerifierInterview, value: str) -> VerifierInterview:
    question = next_question(interview)
    if question is None:
        raise ValueError("interview is already complete")

    revised = []
    for spec in interview.draft.verifiers:
        if spec.verifier_id != question.verifier_id:
            revised.append(spec)
            continue
        if question.field == "expected":
            # Every check is carried through; only the one the question names is
            # revised. Rebuilding from a single check would drop the others.
            revised.append(
                spec.replace(
                    checks=tuple(
                        check.replace(expected=_parse_expected(value, check.expected))
                        if check.check_id == question.check_id
                        else check
                        for check in spec.checks
                    )
                )
            )
        else:
            additions = (value.strip(),) if value.strip() else ()
            revised.append(
                spec.replace(**{question.field: getattr(spec, question.field) + additions})
            )

    index = interview.next_question_index + 1
    return interview.replace(
        draft=interview.draft.replace(verifiers=tuple(revised)),
        answers=interview.answers
        + (InterviewAnswer(question_id=question.question_id, value=value),),
        next_question_index=index,
        complete=index == len(interview.questions),
    )


def compute_interview_id(interview: VerifierInterview) -> str:
    digest = hashlib.sha256(interview.model_dump_json().encode()).hexdigest()
    return f"verifier-interview-{digest[:16]}"


def save_interview(interview: VerifierInterview, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_interview_id(interview),
        kind="verifier_interview",
        parent_artifact_id=interview.source_draft_id,
        payload=interview.model_dump_json().encode(),
        summary={"answers": len(interview.answers), "complete": int(interview.complete)},
    )


def load_interview(interview_id: str, store: DerivedStore) -> VerifierInterview:
    return VerifierInterview.model_validate_json(store.read_payload(interview_id))


def _review_id(interview: VerifierInterview, check_id: str) -> str:
    return f"review-{len(interview.reviews) + 1:03d}-{check_id}"


def start_review(
    draft: VerifierDraft,
    source_draft_id: str,
    *,
    validation_id: str | None = None,
    prior_interview_id: str | None = None,
    round_number: int = 1,
) -> VerifierInterview:
    """Open a free-text review round over every check in the draft.

    Each round is its own artifact. ``prior_interview_id`` chains it to the
    round before so a later round can read what earlier ones decided.

    No stop condition is enforced across rounds: the loop ends when the owner
    ends it. Candidate rules exist — a round proposing nothing already decided,
    or every survivor reaching calibration — but which one matches a real review
    is unknown until one has been run, and a rule guessed now would be enforced
    before anyone had watched the loop behave. Out of scope for #14.
    """
    pending = tuple(
        (spec.verifier_id, check.check_id) for spec in draft.verifiers for check in spec.checks
    )
    return VerifierInterview(
        source_draft_id=source_draft_id,
        draft=draft,
        questions=(),
        pending=pending,
        validation_id=validation_id,
        prior_interview_id=prior_interview_id,
        round_number=round_number,
        complete=not pending,
    )


def next_check(interview: VerifierInterview) -> tuple[str, str] | None:
    """The next (verifier_id, check_id) awaiting a decision."""
    return interview.pending[0] if interview.pending else None


def find_check(
    draft: VerifierDraft, verifier_id: str, check_id: str
) -> tuple[VerifierSpec, CheckSpec]:
    for spec in draft.verifiers:
        if spec.verifier_id != verifier_id:
            continue
        for check in spec.checks:
            if check.check_id == check_id:
                return spec, check
    raise ValueError(f"no check {check_id!r} on verifier {verifier_id!r}")


def _revised_identity(spec: VerifierSpec, check: CheckSpec) -> tuple[str, str]:
    """A new verifier and check id for a materially changed check.

    Reuses the drafting hash so a revision is addressed the same way a fresh
    proposal would be: identity follows content, never the history that
    produced it.
    """
    verifier_id = _spec_id(spec.family_id, check.claim, check.expected)
    return verifier_id, f"check-{verifier_id.removeprefix('verifier-')}"


def _with_extractions(spec: VerifierSpec, interpretation: Interpretation | None) -> VerifierSpec:
    if interpretation is None:
        return spec
    blind = tuple(b for b in interpretation.blind_spots if b not in spec.blind_spots)
    gaming = tuple(g for g in interpretation.gaming_hypotheses if g not in spec.gaming_hypotheses)
    if not blind and not gaming:
        return spec
    return spec.replace(
        blind_spots=spec.blind_spots + blind,
        gaming_hypotheses=spec.gaming_hypotheses + gaming,
    )


def _apply_revision(
    spec: VerifierSpec, check: CheckSpec, interpretation: Interpretation
) -> VerifierSpec:
    """Revise a check, and give the result an identity that admits the change.

    Evidence selected for the old value is cleared rather than carried: it was
    chosen to support a claim this check no longer makes, and leaving it in
    would make a human revision look better supported than it is.
    """
    expected = (
        check.expected
        if interpretation.revised_expected is None
        else interpretation.revised_expected
    )
    operator = interpretation.revised_operator or check.operator
    revised_check = check.replace(
        expected=expected,
        operator=operator,
        supporting_evidence_ids=(),
    )
    verifier_id, check_id = _revised_identity(spec, revised_check)
    revised_check = revised_check.replace(check_id=check_id)
    return spec.replace(
        verifier_id=verifier_id,
        checks=tuple(revised_check if c.check_id == check.check_id else c for c in spec.checks),
        provenance="human",
        status=VerifierStatus.EXECUTABLE,
        validation_artifact_id=None,
        human_acceptance_id=None,
    )


def _combined_spec(left: VerifierSpec, right: VerifierSpec) -> VerifierSpec:
    """Fold two verifiers into one.

    ``unknown_when`` is the union, so the combination is unknown whenever
    either half is — the same fail-closed rule ``Result`` applies to an
    aggregate score. Weights are carried unchanged: ``Result`` already weights
    subscores, and renormalising here would quietly restate what each check was
    worth.
    """
    checks = tuple(
        check.replace(supporting_evidence_ids=()) for check in left.checks + right.checks
    )
    verifier_id = _spec_id(
        left.family_id,
        "+".join(sorted(check.claim for check in checks)),
        tuple(sorted(str(check.expected) for check in checks)),
    )
    seen: dict[str, None] = {}
    for value in left.unknown_when + right.unknown_when:
        seen.setdefault(value, None)
    return left.replace(
        verifier_id=verifier_id,
        checks=checks,
        inputs=tuple(dict.fromkeys(left.inputs + right.inputs)),
        unknown_when=tuple(seen),
        blind_spots=tuple(dict.fromkeys(left.blind_spots + right.blind_spots)),
        gaming_hypotheses=tuple(dict.fromkeys(left.gaming_hypotheses + right.gaming_hypotheses)),
        source_verifier_ids=(left.verifier_id, right.verifier_id),
        provenance="human",
        status=VerifierStatus.EXECUTABLE,
        validation_artifact_id=None,
        human_acceptance_id=None,
    )


def apply_decision(interview: VerifierInterview, review: CheckReview) -> VerifierInterview:
    """Record one confirmed decision and apply it to the draft.

    The decision is the human's. ``review.interpretation`` is what the model
    proposed, recorded whether or not it was followed.

    Four decisions, not six. An earlier draft of #14 also offered
    ``insufficient`` and ``unauthoritative``. ``insufficient`` is what
    ``reject`` already means once review runs more than once — the check returns
    in a later round with more evidence behind it, and there is no state to
    carry in between. ``unauthoritative`` had no machinery behind it: nothing
    here routes a check to another reviewer, so it would have been a state
    nobody acts on, and its real content is a rejection with a reason. Keeping
    either would also have meant widening ``VerifierStatus`` with values that
    are review outcomes rather than lifecycle positions, putting the promotion
    invariants in ``models.py::validate_lifecycle`` at risk for no gain.
    """
    if (review.verifier_id, review.check_id) not in interview.pending:
        raise ValueError(f"check {review.check_id!r} is not awaiting a decision")

    spec, check = find_check(interview.draft, review.verifier_id, review.check_id)
    interpretation = review.interpretation
    verifiers = list(interview.draft.verifiers)
    index = verifiers.index(spec)
    reviews = list(interview.reviews)
    pending = [item for item in interview.pending if item != (review.verifier_id, review.check_id)]

    if review.decision is InterviewDecision.ACCEPT:
        verifiers[index] = _with_extractions(spec, interpretation)
    elif review.decision is InterviewDecision.REJECT:
        # Marked, not dropped. #14 asks that rejected proposals stay auditable
        # with their rationale, and a verifier removed from the draft takes the
        # reason it was rejected with it — leaving a later round free to redraft
        # the same check with nothing recording that it was already refused.
        verifiers[index] = _with_extractions(spec, interpretation).replace(
            status=VerifierStatus.REJECTED
        )
    elif review.decision is InterviewDecision.REVISE:
        if interpretation is None:
            raise ValueError("a revise decision needs an interpretation to revise from")
        verifiers[index] = _apply_revision(
            _with_extractions(spec, interpretation), check, interpretation
        )
    else:
        if interpretation is None or not interpretation.combine_with:
            raise ValueError("a combine decision needs a resolved target check")
        other_spec, _ = _find_by_check(interview.draft, interpretation.combine_with)
        if other_spec.verifier_id == spec.verifier_id:
            raise ValueError("a check cannot be combined with its own verifier")
        combined = _combined_spec(_with_extractions(spec, interpretation), other_spec)
        verifiers = [
            v for v in verifiers if v.verifier_id not in {spec.verifier_id, other_spec.verifier_id}
        ]
        verifiers.insert(min(index, len(verifiers)), combined)
        # The other check leaves the queue with this decision; if it was already
        # decided, that earlier decision is superseded rather than overwritten.
        pending = [item for item in pending if item[0] != other_spec.verifier_id]
        reviews = [
            item.replace(superseded_by=review.review_id)
            if item.verifier_id == other_spec.verifier_id and not item.superseded_by
            else item
            for item in reviews
        ]

    reviews.append(review)
    return interview.replace(
        draft=interview.draft.replace(verifiers=tuple(verifiers)),
        reviews=tuple(reviews),
        pending=tuple(pending),
        complete=not pending,
    )


def _find_by_check(draft: VerifierDraft, check_id: str) -> tuple[VerifierSpec, CheckSpec]:
    for spec in draft.verifiers:
        for check in spec.checks:
            if check.check_id == check_id:
                return spec, check
    raise ValueError(f"no check {check_id!r} in this draft")


def prior_decisions(interview: VerifierInterview, check_id: str) -> tuple[str, ...]:
    """What earlier rounds decided about this check, oldest first.

    Read by the interpreter prompt: without it, a second-round "still fine"
    about a check already known to be weak is indistinguishable from a first
    look.
    """
    lines = []
    for review in interview.reviews:
        if review.check_id != check_id:
            continue
        note = " (superseded)" if review.superseded_by else ""
        lines.append(
            f"round {interview.round_number}: {review.decision.value}{note} — {review.reply}"
        )
    return tuple(lines)
