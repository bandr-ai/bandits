"""Evaluate a ``Verifier`` against a final state and an effect ledger.

This is the function that turns an episode into a number. It is pure: given the
same verifier, the same final state and the same effects it returns the same
reward, forever. No sampling, no model, no clock.

**Reward rule.** Default is all-or-nothing: every assertion passes or the reward
is 0.0. Partial credit is opt-in.

Why all-or-nothing is the safe default here. bandits's output feeds a pass@k
routing gate - "can the model already do this?" - and a partial score makes that
question unanswerable: a rollout that sets the status correctly but also wipes an
unrelated table scores 0.6, and 0.6 is neither a pass nor a failure. Worse,
partial credit is a gradient toward whichever assertion is cheapest to satisfy,
which is usually the one that needs no tool call at all (STATE_UNCHANGED passes
for free if you do nothing). Partial credit is useful for *debugging* a verifier
and for dense-reward experiments where somebody has looked at the assertion mix
and decided it is safe. It is never the default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from bandits.contracts import (
    Assertion,
    AssertionKind,
    AssertionResult,
    Effect,
    JsonObject,
    JsonValue,
    VerificationResult,
    Verifier,
)

__all__ = [
    "RewardMode",
    "UnreviewedVerifierError",
    "evaluate",
    "find_row",
]

RewardMode = Literal["all_or_nothing", "partial"]

FinalState = Mapping[str, Sequence[JsonObject]]


class UnreviewedVerifierError(RuntimeError):
    """Raised when a verifier with ``reviewed_by is None`` is asked to grade.

    PLAN.md Step 11: generate the verifier once, freeze it, and have a person
    read it before it grades anything. Tests and dry runs pass
    ``allow_unreviewed=True`` explicitly, which is the point - the exception
    forces the decision to be visible in the calling code.
    """


def _key_id(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def find_row(state: FinalState, entity: str, row_key: JsonValue) -> JsonObject | None:
    """Find the row matching ``row_key`` (a ``{pk: value}`` dict).

    Matching is by string form of the key value, so an env that returns ids as
    strings and a trace that recorded them as ints still agree. Any other kind
    of type drift is a fidelity problem and belongs in stage 7, not here.
    """
    rows = state.get(entity) or []
    if not isinstance(row_key, dict):
        return None
    for row in rows:
        if all(_key_id(row.get(k)) == _key_id(v) for k, v in row_key.items()):
            return row
    return None


def _eval_state_equals(a: Assertion, state: FinalState) -> tuple[bool, JsonValue]:
    """A field must equal an expected value.

    Catches: the target field not being set, being set to the wrong value, or the
    row having been deleted out from under it.
    Misses: *how* the field got there - a direct store write satisfies this
    assertion just as well as a tool call. That is anticheat.py's job.
    """
    row = find_row(state, a.entity or "", a.row_key)
    if row is None:
        return False, None
    return row.get(a.field) == a.expected, row.get(a.field)


def _eval_state_unchanged(a: Assertion, state: FinalState) -> tuple[bool, JsonValue]:
    """A row, a set of fields, or a whole entity must be exactly as it started.

    Two forms. With ``row_key``: ``expected`` is a field->value mapping that must
    still hold on that row. Without: ``expected`` is a list of starting rows,
    every one of which must still be present with identical values.

    Catches: collateral damage - deleting or editing rows the task did not ask
    about, which is the failure mode a change-only verifier actively teaches.
    Misses: *insertions*. Semantics are deliberately subset-based, because the
    materialized environment also contains synthetic filler rows the verifier
    never saw. An agent that adds junk rows is not caught here; catch that with
    an explicit ROW_ABSENT or a row-count assertion when it matters.
    """
    if a.row_key is not None:
        row = find_row(state, a.entity or "", a.row_key)
        if row is None:
            return False, None
        expected = a.expected if isinstance(a.expected, dict) else {}
        drift = {k: row.get(k) for k, v in expected.items() if row.get(k) != v}
        return not drift, drift or None
    expected_rows = a.expected if isinstance(a.expected, list) else []
    missing: list[JsonValue] = []
    for want in expected_rows:
        if not isinstance(want, dict):
            continue
        rows = state.get(a.entity or "") or []
        if not any(all(r.get(k) == v for k, v in want.items()) for r in rows):
            missing.append(want)
    return not missing, missing or None


def _eval_row_exists(a: Assertion, state: FinalState) -> tuple[bool, JsonValue]:
    """A row must exist, and match ``expected`` on every field given.

    Catches: a create-shaped task where the agent never created anything.
    Misses: fields not present in ``expected`` - they are unconstrained.
    """
    row = find_row(state, a.entity or "", a.row_key)
    if row is None:
        return False, None
    if isinstance(a.expected, dict):
        drift = {k: row.get(k) for k, v in a.expected.items() if row.get(k) != v}
        return not drift, drift or row
    return True, row


def _eval_row_absent(a: Assertion, state: FinalState) -> tuple[bool, JsonValue]:
    """A row must not exist.

    Catches: a delete/cancel task the agent did not carry out.
    Misses: soft deletes - a row marked ``deleted=true`` but still present reads
    as a failure here even when the real system considers it gone.
    """
    row = find_row(state, a.entity or "", a.row_key)
    return row is None, row


def _eval_effect_count(a: Assertion, effects: Sequence[Effect]) -> tuple[bool, JsonValue]:
    """An external tool must have been attempted exactly N times.

    Catches: the email that was never sent, and - just as important - the email
    sent twice, which in production is a duplicate refund notice to a real
    customer.
    Misses: the *content* of the effect. One email with the wrong amount in it
    counts as one email. Constrain the payload with a separate assertion if the
    task depends on it.
    """
    n = sum(1 for e in effects if e.tool == a.tool)
    return n == a.expected, n


def evaluate(
    verifier: Verifier,
    final_state: FinalState,
    effects: Sequence[Effect] = (),
    *,
    mode: RewardMode = "all_or_nothing",
    allow_unreviewed: bool = False,
) -> VerificationResult:
    """Grade one rollout. Returns per-assertion results and a reward in [0, 1]."""
    if verifier.reviewed_by is None and not allow_unreviewed:
        raise UnreviewedVerifierError(
            f"verifier {verifier.verifier_id!r} has reviewed_by=None. A generated reward "
            f"function nobody has read is an unexamined reward function; have a person "
            f"read it and set reviewed_by, or pass allow_unreviewed=True for a dry run."
        )
    if mode not in ("all_or_nothing", "partial"):
        raise ValueError(f"unknown reward mode {mode!r}")
    if not verifier.assertions:
        raise ValueError(
            f"verifier {verifier.verifier_id!r} has no assertions; it would pass every "
            f"rollout, including the empty one"
        )

    results: list[AssertionResult] = []
    for a in verifier.assertions:
        if a.kind is AssertionKind.STATE_EQUALS:
            ok, actual = _eval_state_equals(a, final_state)
        elif a.kind is AssertionKind.STATE_UNCHANGED:
            ok, actual = _eval_state_unchanged(a, final_state)
        elif a.kind is AssertionKind.ROW_EXISTS:
            ok, actual = _eval_row_exists(a, final_state)
        elif a.kind is AssertionKind.ROW_ABSENT:
            ok, actual = _eval_row_absent(a, final_state)
        elif a.kind is AssertionKind.EFFECT_COUNT:
            ok, actual = _eval_effect_count(a, effects)
        else:  # pragma: no cover - AssertionKind is exhaustive above
            raise ValueError(f"unhandled assertion kind {a.kind!r}")
        results.append(AssertionResult(assertion=a, passed=ok, actual=actual))

    passed = all(r.passed for r in results)
    if mode == "all_or_nothing":
        reward = 1.0 if passed else 0.0
    else:
        reward = sum(1 for r in results if r.passed) / len(results)
    return VerificationResult(
        task_id=verifier.task_id,
        passed=passed,
        results=tuple(results),
        reward=reward,
    )
