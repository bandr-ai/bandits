"""Execute deterministic replay verifier specs against recorded evidence.

Each operator decides for itself what evidence it needs and what absence means.
Absence is never failure: a check that cannot see what it requires returns
unknown, and one unknown check makes the whole verifier unknown.
"""

from __future__ import annotations

from typing import Any

from bandits.analyze.models import Evidence
from bandits.verify.models import CheckOperator, CheckSpec, Result, SubScore, VerifierSpec


class _Outcome:
    """One check's verdict: a score, the evidence behind it, and why."""

    __slots__ = ("score", "evidence_ids", "details")

    def __init__(
        self,
        score: float | None,
        evidence_ids: tuple[str, ...] = (),
        **details: Any,
    ) -> None:
        self.score = score
        self.evidence_ids = evidence_ids
        self.details = details


def _unknown(reason: str) -> _Outcome:
    return _Outcome(None, reason=reason)


def _field(evidence: tuple[Evidence, ...], claim: str, key: str) -> Evidence | None:
    """Find the state field a check names.

    Matched on the tool-qualified name first, then on the bare key, because a
    spec accepted before fields carried their tool names asks for the key alone
    and must keep executing exactly as it did when it was measured.
    """
    return next(
        (
            e
            for e in evidence
            if e.claim == claim and key in (e.value.get("field"), e.value.get("key"))
        ),
        None,
    )


def _keyed_equality(check: CheckSpec, evidence: tuple[Evidence, ...], claim: str) -> _Outcome:
    key = check.claim.partition(":")[2]
    found = _field(evidence, claim, key)
    if found is None:
        return _unknown(f"no {claim} recorded for {key!r}")
    observed = found.value.get("value")
    return _Outcome(
        1.0 if observed == check.expected else 0.0,
        (found.evidence_id,),
        key=key,
        observed=observed,
        expected=check.expected,
    )


def _state_invariant(check: CheckSpec, evidence: tuple[Evidence, ...]) -> _Outcome:
    """Compare a terminal field against the initial state it should agree with.

    Unknown, not failed, when either side went unrecorded: a relation between
    two fields needs both of them, and a trace that captured neither says
    nothing about whether the relation held.
    """
    final_key, _, initial_key = check.claim.partition(":")[2].partition("==")
    final = _field(evidence, "final_state_field", final_key)
    initial = _field(evidence, "initial_state_field", initial_key)
    if final is None or initial is None:
        missing = final_key if final is None else initial_key
        return _unknown(f"no state recorded for {missing!r}")

    final_value, initial_value = final.value.get("value"), initial.value.get("value")
    return _Outcome(
        1.0 if final_value == initial_value else 0.0,
        (final.evidence_id, initial.evidence_id),
        final={final_key: final_value},
        initial={initial_key: initial_value},
    )


def _rubric(check: CheckSpec, evidence: tuple[Evidence, ...]) -> _Outcome:
    """Threshold a judge's score.

    A verdict its own samples disagreed about is returned unknown rather than
    thresholded: forcing a coin-flip into a pass or fail would launder the
    judge's uncertainty into a number that looks decided.
    """
    found = next((e for e in evidence if e.claim == check.claim), None)
    if found is None:
        return _unknown(f"no judgement recorded for {check.claim!r}")

    score = found.value.get("score")
    if score is None:
        return _unknown("the judge returned no usable score")
    if found.value.get("agreement", 1.0) < 1.0:
        return _unknown(
            f"the judge disagreed with itself across samples {found.value.get('samples')}"
        )

    return _Outcome(
        1.0 if score >= check.expected else 0.0,
        (found.evidence_id,),
        judged_score=score,
        threshold=check.expected,
        samples=found.value.get("samples"),
    )


def _no_span_error(evidence: tuple[Evidence, ...]) -> _Outcome:
    """Pass when nothing in the episode reported an error.

    Anchored on ``episode_span_count``, which extraction always emits: without
    it we cannot tell a clean run from a trace we simply hold no evidence for,
    and absence of an error would silently read as success.
    """
    if not any(e.claim == "episode_span_count" for e in evidence):
        return _unknown("no evidence recorded for this episode at all")
    errors = tuple(e.evidence_id for e in evidence if e.claim == "span_error")
    return _Outcome(0.0 if errors else 1.0, errors, errors=len(errors))


def _evaluate(check: CheckSpec, evidence: tuple[Evidence, ...]) -> _Outcome:
    if check.operator is CheckOperator.EXIT_CODE_ZERO:
        codes = [e for e in evidence if e.claim == "command_exit_code"]
        if not codes:
            return _unknown("no command exit code recorded")
        passed = any(e.value.get("value") == 0 for e in codes)
        return _Outcome(1.0 if passed else 0.0, tuple(e.evidence_id for e in codes), matched=passed)

    if check.operator is CheckOperator.EXACT_OUTPUT:
        outputs = [e for e in evidence if e.claim == "final_output"]
        if not outputs:
            return _unknown("no final output recorded")
        passed = any(e.value.get("output") == check.expected for e in outputs)
        return _Outcome(
            1.0 if passed else 0.0, tuple(e.evidence_id for e in outputs), matched=passed
        )

    if check.operator is CheckOperator.STATE_INVARIANT:
        return _state_invariant(check, evidence)

    if check.operator is CheckOperator.NO_SPAN_ERROR:
        return _no_span_error(evidence)

    if check.operator is CheckOperator.RUBRIC_AT_LEAST:
        return _rubric(check, evidence)

    if check.operator is CheckOperator.EQUALS:
        for prefix, claim in (
            ("final_state_field:", "final_state_field"),
            ("initial_state_field:", "initial_state_field"),
            ("recorded_score:", "recorded_score"),
        ):
            if check.claim.startswith(prefix):
                return _keyed_equality(check, evidence, claim)

    return _unknown(f"no evaluator for operator {check.operator.value!r}")


def execute_verifier(spec: VerifierSpec, evidence: tuple[Evidence, ...]) -> Result:
    """Score recorded evidence; absent required evidence produces unknown, never failure."""
    subscores = tuple(
        SubScore(
            check_id=check.check_id,
            score=outcome.score,
            evidence_ids=outcome.evidence_ids,
            details=outcome.details,
        )
        for check, outcome in ((check, _evaluate(check, evidence)) for check in spec.checks)
    )

    unknown = [part.check_id for part in subscores if part.score is None]
    if unknown:
        # A composite verifier fails closed when any required check is
        # unavailable. The subscores are kept exactly as computed: the aggregate
        # is what must be conservative, not the record of how it got there.
        return Result(
            verifier_id=spec.verifier_id,
            score=None,
            subscores=subscores,
            details={"reason": "one or more required checks are unknown", "unknown": unknown},
        )

    total_weight = sum(check.weight for check in spec.checks)
    score = (
        sum(part.score * check.weight for part, check in zip(subscores, spec.checks, strict=True))
        / total_weight
        if total_weight
        else None
    )
    return Result(verifier_id=spec.verifier_id, score=score, subscores=subscores)
