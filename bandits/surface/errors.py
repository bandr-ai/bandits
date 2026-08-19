"""Error mode collection.

"An env that can't return ``not_found`` trains an agent that has never seen
adversity" (PLAN.md, Step 7 table). So every failure the production agent ever
hit is collected here, keyed by a coarse `error_kind`, with one concrete example
response the rebuilt tool can replay verbatim.

Kind resolution order:

1. `InvocationPoint.error_kind` when ingest set it (authoritative).
2. Otherwise a string ``error``/``code``/``error_code``/``reason`` field in the
   response body, which is where every REST-ish tool in practice puts it.
3. Otherwise the literal ``"unspecified"`` -- never a guess, never dropped.

Failure mode: step 2 keys on the *value* of the error field, so a backend that
returns ``{"error": "order 7741 already refunded"}`` (message, not code) yields
one error mode per order id instead of one per failure class. Detecting that
needs clustering we deliberately do not do without an LLM; the symptom is an
absurd number of error modes for one tool, which is visible in the surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from bandits.contracts import CallStatus, ErrorMode, InvocationPoint, JsonValue

UNSPECIFIED = "unspecified"

_KIND_FIELDS = ("error_kind", "error_code", "error", "code", "reason")


def error_kind_of(invocation: InvocationPoint) -> str:
    """Best available coarse label for one failed call. See module docstring."""
    if invocation.error_kind:
        return invocation.error_kind
    response = invocation.response
    if isinstance(response, Mapping):
        for field in _KIND_FIELDS:
            value = response.get(field)
            if isinstance(value, str) and value:
                return value
    return UNSPECIFIED


def collect_error_modes(invocations: Iterable[InvocationPoint]) -> tuple[ErrorMode, ...]:
    """Group the ERROR invocations of one tool into `ErrorMode`s.

    The example response kept is the one from the earliest call (by trace id,
    then step) so re-running on the same corpus gives byte-identical output.
    Ordering is by descending frequency, then kind: the mode an environment is
    most likely to need comes first.
    """
    counts: dict[str, int] = {}
    examples: dict[str, tuple[tuple[str, int], JsonValue]] = {}

    for inv in invocations:
        if inv.status is not CallStatus.ERROR:
            continue
        kind = error_kind_of(inv)
        counts[kind] = counts.get(kind, 0) + 1
        position = (inv.trace_id, inv.step)
        if kind not in examples or position < examples[kind][0]:
            examples[kind] = (position, inv.response)

    return tuple(
        ErrorMode(
            error_kind=kind,
            occurrences=counts[kind],
            example_response=examples[kind][1],
        )
        for kind in sorted(counts, key=lambda k: (-counts[k], k))
    )
