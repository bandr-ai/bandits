"""Observation comparison with an explicit, documented tolerance policy.

The fidelity gate is only worth anything if the comparison behind it is
principled. A fuzzy similarity score would let a rebuilt environment that
returns *plausible* answers pass; what we need to know is whether it returns
*the same* answers. So this module does not score. It classifies every
difference it finds, one at a time, into exactly one of two buckets, and it
says which bucket and why.

Exact match required (never tolerated)
--------------------------------------
=========================  ====================================================
identifiers                ``order_id``, ``customer_id``, ``sku`` -- any
                           business key. A rebuilt environment that hands back
                           the wrong row is wrong, full stop.
statuses / enums           ``status``, ``state``, ``*_status``, ``error``,
                           ``error_kind``, ``code``. These are the fields
                           verifiers assert on.
numbers                    every int/float, including money. No epsilon: the
                           environment is deterministic arithmetic over a
                           SQLite store, so a cent of drift is a modeling bug.
booleans                   exact.
structural shape           object key sets and array lengths. An extra key the
                           real tool never returned is a divergence even when
                           its value is ``null``: it changes what an agent
                           observes.
JSON type                  ``"4200"`` is not ``4200``.
=========================  ====================================================

Tolerated differences
---------------------
=========================  ====================================================
timestamps                 fields named ``*_at``/``*_time``/``*_ts``/``*_date``
                           (and bare ``timestamp``/``date``). A replay does not
                           happen at the moment the trace was recorded, and the
                           environment is forbidden from reading a clock.
generated ids              a *closed list* of names the environment could not
                           have known: ``id``, ``uuid``, ``request_id``,
                           ``trace_id``, ``span_id``, ``call_id``,
                           ``idempotency_key``, ``transaction_id``,
                           ``receipt_id``, ``confirmation_id``,
                           ``confirmation_number``, ``token``, ``nonce``, plus
                           the ``*_uuid``/``*_token``/``*_nonce`` suffixes.
                           Deliberately **not** every ``*_id``: ``order_id`` is
                           a row handle that must match.
unordered collections      two arrays holding the same multiset of elements in
                           a different order. Same length, same contents,
                           different order -> tolerated as ``reordered``.
                           Different contents -> a real divergence.
free text                  ``message``, ``body``, ``description``, ``detail``,
                           ``details``, ``note``, ``notes``, ``text``,
                           ``summary``, ``content``, ``explanation`` -- and only
                           when both sides are strings. Prose is not state.
=========================  ====================================================

Everything else is a divergence. When in doubt the answer is "not tolerated":
a false accept ships a broken environment into training, a false reject costs
an engineer an afternoon.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bandits.contracts import CallStatus, JsonObject, JsonValue, Observation

__all__ = [
    "Divergence",
    "EXACT_FIELDS",
    "FREE_TEXT_FIELDS",
    "GENERATED_ID_FIELDS",
    "classify_field",
    "compare_observations",
    "compare_values",
]

# -- field classification --------------------------------------------------

#: Names that must match exactly no matter what other rule would match them.
#: Checked first, so a hypothetical ``status_at`` is still an exact field.
EXACT_FIELDS: frozenset[str] = frozenset(
    {"status", "state", "error", "error_kind", "code", "kind", "type"}
)

#: Names whose value the environment could not have known. A closed list on
#: purpose: tolerating every ``*_id`` would tolerate returning the wrong row.
GENERATED_ID_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "uuid",
        "guid",
        "request_id",
        "trace_id",
        "span_id",
        "call_id",
        "session_id",
        "correlation_id",
        "idempotency_key",
        "transaction_id",
        "receipt_id",
        "confirmation_id",
        "confirmation_number",
        "token",
        "nonce",
    }
)

#: Prose. Not state, and nothing a verifier is allowed to assert on.
FREE_TEXT_FIELDS: frozenset[str] = frozenset(
    {
        "message",
        "body",
        "description",
        "detail",
        "details",
        "note",
        "notes",
        "text",
        "summary",
        "content",
        "explanation",
    }
)

_TIMESTAMP_RE = re.compile(r"(^|_)(at|time|ts|timestamp|date|datetime)$")
_GENERATED_SUFFIX_RE = re.compile(r"_(uuid|guid|token|nonce)$")
_STATUS_SUFFIX_RE = re.compile(r"_(status|state|code|kind)$")


def classify_field(name: str) -> str:
    """Return one of ``exact`` / ``timestamp`` / ``generated_id`` / ``free_text``.

    Precedence is fixed and checked in this order: exact-always names, status
    suffixes, generated ids, timestamps, free text. Anything unrecognized is
    ``exact`` -- the safe default.
    """
    lowered = name.lower()
    if lowered in EXACT_FIELDS or _STATUS_SUFFIX_RE.search(lowered):
        return "exact"
    if lowered in GENERATED_ID_FIELDS or _GENERATED_SUFFIX_RE.search(lowered):
        return "generated_id"
    if _TIMESTAMP_RE.search(lowered):
        return "timestamp"
    if lowered in FREE_TEXT_FIELDS:
        return "free_text"
    return "exact"


# -- the divergence record -------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """One concrete difference between a recorded response and a replayed one.

    Never a bare bool: a human reads these to decide what to fix, so each one
    carries where it was, what was expected, what came back, why it counted the
    way it did, and whether it was tolerated.
    """

    path: str
    expected: JsonValue
    actual: JsonValue
    reason: str
    tolerated: bool = False
    field_class: str = "exact"

    def to_json(self) -> JsonObject:
        return {
            "path": self.path,
            "expected": _safe(self.expected),
            "actual": _safe(self.actual),
            "reason": self.reason,
            "tolerated": self.tolerated,
            "field_class": self.field_class,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        mark = "~" if self.tolerated else "!"
        return f"{mark} {self.path}: {self.reason} (expected={self.expected!r} actual={self.actual!r})"


def _safe(value: JsonValue) -> JsonValue:
    """Make a value safe to embed in a pydantic ``JsonObject`` example."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


# -- value comparison ------------------------------------------------------


def _canon(value: JsonValue) -> str:
    """Canonical JSON for multiset comparison of unordered collections."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


def _json_type(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


@dataclass
class _Ctx:
    out: list[Divergence] = field(default_factory=list)


def compare_values(
    expected: JsonValue,
    actual: JsonValue,
    *,
    path: str = "$",
    field_name: str = "",
) -> list[Divergence]:
    """Compare two JSON values, returning every difference found.

    Recurses through objects and arrays. ``field_name`` carries the key the
    value was found under, which is what drives the tolerance policy -- so a
    tolerated field name tolerates the whole subtree beneath it only when both
    sides are scalars of a tolerable type.
    """
    ctx = _Ctx()
    _compare(expected, actual, path, field_name, ctx)
    return ctx.out


def _compare(expected: JsonValue, actual: JsonValue, path: str, name: str, ctx: _Ctx) -> None:
    if expected == actual:
        return

    et, at = _json_type(expected), _json_type(actual)
    klass = classify_field(name) if name else "exact"

    if et != at:
        # A type change is structural. Never tolerated, even on a tolerated
        # field name: a timestamp that became null is a modeling failure.
        ctx.out.append(
            Divergence(
                path=path,
                expected=_safe(expected),
                actual=_safe(actual),
                reason=f"type changed: {et} -> {at}",
                tolerated=False,
                field_class=klass,
            )
        )
        return

    if et == "object":
        _compare_objects(expected, actual, path, ctx)
        return
    if et == "array":
        _compare_arrays(expected, actual, path, name, klass, ctx)
        return

    # Scalar, same JSON type, different value.
    tolerated, reason = _scalar_tolerance(klass, et)
    ctx.out.append(
        Divergence(
            path=path,
            expected=_safe(expected),
            actual=_safe(actual),
            reason=reason,
            tolerated=tolerated,
            field_class=klass,
        )
    )


def _scalar_tolerance(klass: str, json_type: str) -> tuple[bool, str]:
    """Decide whether a scalar difference is tolerated, and say why."""
    if klass == "timestamp" and json_type in ("string", "number"):
        return True, "timestamp differs; the environment is forbidden from reading a clock"
    if klass == "generated_id" and json_type in ("string", "number"):
        return True, "generated id the environment could not have known"
    if klass == "free_text" and json_type == "string":
        return True, "free-text field; prose is not state"
    if json_type == "boolean":
        return False, "boolean value differs"
    if json_type == "number":
        return False, "numeric value differs (no tolerance: the store is exact arithmetic)"
    return False, "value differs"


def _compare_objects(expected: JsonObject, actual: JsonObject, path: str, ctx: _Ctx) -> None:
    """Key sets are structural and exact. Then recurse into shared keys."""
    for key in sorted(set(expected) - set(actual)):
        ctx.out.append(
            Divergence(
                path=f"{path}.{key}",
                expected=_safe(expected[key]),
                actual=None,
                reason="field missing from the replayed response",
                tolerated=False,
                field_class=classify_field(key),
            )
        )
    for key in sorted(set(actual) - set(expected)):
        ctx.out.append(
            Divergence(
                path=f"{path}.{key}",
                expected=None,
                actual=_safe(actual[key]),
                reason="field present in the replayed response but never recorded",
                tolerated=False,
                field_class=classify_field(key),
            )
        )
    for key in sorted(set(expected) & set(actual)):
        _compare(expected[key], actual[key], f"{path}.{key}", key, ctx)


def _compare_arrays(
    expected: list, actual: list, path: str, name: str, klass: str, ctx: _Ctx
) -> None:
    """Length is structural. Same multiset in a different order is tolerated."""
    if len(expected) != len(actual):
        ctx.out.append(
            Divergence(
                path=path,
                expected=_safe(list(expected)),
                actual=_safe(list(actual)),
                reason=f"array length differs: {len(expected)} -> {len(actual)}",
                tolerated=False,
                field_class=klass,
            )
        )
        return
    if Counter(_canon(v) for v in expected) == Counter(_canon(v) for v in actual):
        ctx.out.append(
            Divergence(
                path=path,
                expected=_safe(list(expected)),
                actual=_safe(list(actual)),
                reason="same elements in a different order; collection treated as unordered",
                tolerated=True,
                field_class=klass,
            )
        )
        return
    for index, (e, a) in enumerate(zip(expected, actual, strict=True)):
        _compare(e, a, f"{path}[{index}]", name, ctx)


# -- observation comparison ------------------------------------------------


def compare_observations(
    expected_response: JsonValue,
    expected_status: CallStatus,
    expected_error_kind: str | None,
    actual: Observation,
    *,
    compare_response: bool = True,
) -> list[Divergence]:
    """Compare a recorded invocation against what the environment returned.

    ``status`` and ``error_kind`` are always exact: they are the coarse
    contract every verifier and every agent branches on, and an environment
    that turns a recorded ``already_refunded`` into a ``not_found`` has changed
    the dynamics even if the body looks similar.

    ``compare_response=False`` is for EXTERNAL tools, whose recorded body is a
    vendor acknowledgement the stub has no way to reproduce. See
    :mod:`bandits.fidelity.replay` for what is checked instead.
    """
    out: list[Divergence] = []
    if expected_status != actual.status:
        out.append(
            Divergence(
                path="$status",
                expected=_status_str(expected_status),
                actual=_status_str(actual.status),
                reason="call status differs; ok/error is the coarsest contract there is",
                tolerated=False,
                field_class="exact",
            )
        )
    if (expected_error_kind or None) != (actual.error_kind or None):
        out.append(
            Divergence(
                path="$error_kind",
                expected=expected_error_kind,
                actual=actual.error_kind,
                reason="error kind differs; failure modes are part of the dynamics",
                tolerated=False,
                field_class="exact",
            )
        )
    if compare_response:
        out.extend(compare_values(expected_response, actual.response, path="$"))
    return out


def _status_str(status: Any) -> str:
    return status.value if isinstance(status, CallStatus) else str(status)
