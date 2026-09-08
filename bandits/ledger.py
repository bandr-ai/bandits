"""An append-only record of every model call the pipeline makes.

A stage records its conclusions. What produced them — the request, the reply,
what it cost, how long it took, which attempts failed first — was discarded at
each call site independently, so a verdict could be read and never checked, a
bill could not be attributed, and two runs that disagreed offered nothing to
compare.

The record lives at the transport boundary rather than in each stage, because
there is one place every provider request passes through and five places that
would otherwise each invent their own shape. A stage names what it was doing;
the transport knows everything else.

Headers are dropped rather than filtered: an allowlist that grows a new field
on a provider's next release leaks it, and nothing here needs them. That is a
narrow guarantee and worth stating narrowly — it is not the same as "no secret
can reach this file". Prompts, responses, request fields and exception messages
are all recorded verbatim, and any of them can carry a credential that some
other layer put there.

What is guaranteed: no request header, and therefore no ``Authorization``, is
written. What is not: the prompt is kept, so a corpus carrying personal data
produces a ledger carrying it too. The file belongs wherever the corpus does,
and there is no redacted export path yet.

Identity: every row carries a unique ``event_id``. A ``stage_id`` is shared by
the rows inside one stage, and a ``logical_call_id`` ties a call's retries to
the call. One row, one id, always.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENV_PATH = "BANDITS_LEDGER"
"""Where to append. Unset means record nothing, so the ledger is opt-in."""

_local = threading.local()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _context() -> dict[str, Any]:
    return getattr(_local, "context", {})


_ENV_STRICT = "BANDITS_LEDGER_STRICT"
"""Set alongside the path to make a lost record fail the run.

Best effort is right for ordinary use, where a full disk should not destroy the
work being recorded. It is wrong for a formal experiment, where a run that
quietly recorded nine tenths of itself is not a cheaper experiment but an
invalid one that still costs money.
"""


class LedgerWriteError(RuntimeError):
    """A record could not be written while the ledger was required."""


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[None]:
    """Name what the calls inside this block were for.

    Nested rather than passed down: the transport cannot know it is serving a
    family audit, and threading a label through five signatures to tell it
    would put a logging concern in every function between.
    """
    previous = _context()
    merged = {
        **previous,
        **{key: value for key, value in fields.items() if value is not None},
        "stage": name,
        # Identifies the block, not the rows inside it. Every row still gets an
        # `event_id` of its own at write time; naming this one `event_id` made
        # a dozen calls share an identity and left no way to reference one.
        "stage_id": uuid.uuid4().hex[:16],
        "parent_stage_id": previous.get("stage_id"),
    }
    _local.context = merged
    try:
        yield
    finally:
        _local.context = previous


def enabled() -> bool:
    return bool(os.environ.get(_ENV_PATH))


def strict() -> bool:
    return bool(os.environ.get(_ENV_STRICT)) and enabled()


def record(event: dict[str, Any]) -> None:
    """Append one event, stamped with an identity unique to this row.

    Best effort by default: a run that died because its logging could not write
    would trade the thing being recorded for the record of it. Under
    ``BANDITS_LEDGER_STRICT`` the trade goes the other way and the failure is
    raised, because a formal experiment missing records is not a partial
    result — it is one that cannot be checked.
    """
    path = os.environ.get(_ENV_PATH)
    if not path:
        return
    row = {
        "recorded_at": _now(),
        **_context(),
        **event,
        # Last, so the row's identity is the ledger's to assign. A caller
        # passing `event_id` would otherwise silently collide two rows, which
        # is the failure this field exists to make impossible.
        "event_id": uuid.uuid4().hex,
    }
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except OSError as exc:
        if strict():
            raise LedgerWriteError(
                f"the ledger could not be written to {path!r}: {exc}. "
                "The run is recording incompletely and BANDITS_LEDGER_STRICT is set."
            ) from exc
        # Loud even when not fatal. A silent drop is how a run finishes looking
        # complete while missing the records it was started to produce.
        print(  # noqa: T201 - deliberate: stderr must carry this past a rich console
            f"bandits: WARNING lost a ledger event, {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


@contextmanager
def model_call(*, provider: str, model: str, request: Any) -> Iterator[dict[str, Any]]:
    """Record one *logical* provider request, whether or not it succeeds.

    Logical, not physical, and the distinction is load-bearing: this wraps
    ``request_with_retry``, so a 429 followed by a success is one event here
    and two requests on the wire. The physical attempts appear as their own
    ``retry`` events carrying this call's ``logical_call_id``; the number of
    requests actually made is that count plus one, never this event alone.

    Instrumenting each ``urlopen`` instead would make the count exact and would
    put the record below the layer that knows what the call was for. It stays
    here until something needs per-attempt bodies.

    A failed call is the one most worth having: it still consumed a rate-limit
    budget and may have cost tokens, and its absence is what made a rerun that
    behaved differently impossible to explain.
    """
    slot: dict[str, Any] = {}
    started = time.monotonic()
    started_at = _now()
    status = "success"
    error: dict[str, Any] | None = None
    call_id = uuid.uuid4().hex[:16]
    previous = _context()
    # Published so the retries firing inside this block can name the call they
    # belong to without it being threaded through the transport's signature.
    _local.context = {**previous, "logical_call_id": call_id}
    try:
        yield slot
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised untouched
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        code = getattr(exc, "code", None)
        if code is not None:
            error["http_status"] = code
        raise
    finally:
        response = slot.get("response")
        # Restoring the context gets its own `finally`. Under strict mode
        # `record` raises, and a raise here would leave this call's
        # `logical_call_id` on the context for whatever ran next to inherit,
        # attributing later retries to a call that had already finished.
        try:
            record(
                {
                    "event_type": "model_call",
                    "logical_call_id": call_id,
                    "provider": provider,
                    "model": model,
                    "request": request,
                    "response": response,
                    "usage": (
                        (response or {}).get("usage") if isinstance(response, dict) else None
                    ),
                    "status": status,
                    "error": error,
                    "started_at": started_at,
                    "finished_at": _now(),
                    "duration_seconds": round(time.monotonic() - started, 4),
                }
            )
        finally:
            _local.context = previous


def record_attempt(*, attempt: int, error: Exception, delay: float) -> None:
    """One failed physical request inside a retried logical call.

    Kept separate from the call it belongs to: a call that succeeded on its
    third attempt looks identical to one that succeeded immediately, except in
    latency and in the rate-limit budget it spent getting there. The
    ``logical_call_id`` inherited from the enclosing ``model_call`` is what
    reattaches them.
    """
    event: dict[str, Any] = {
        "event_type": "retry",
        "attempt": attempt,
        "attempt_id": uuid.uuid4().hex[:16],
        "error_type": type(error).__name__,
        "message": str(error),
        "computed_delay_seconds": round(delay, 3),
    }
    code = getattr(error, "code", None)
    if code is not None:
        event["http_status"] = code
    headers = getattr(error, "headers", None)
    if headers is not None:
        # The only header worth keeping: it is the server stating a number the
        # backoff then obeys, so a delay that looks wrong can be traced to it.
        asked = headers.get("Retry-After")
        if asked:
            event["retry_after"] = str(asked)
    record(event)
