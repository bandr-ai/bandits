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

Secrets never enter. Headers are dropped rather than filtered: an allowlist
that grows a new field on a provider's next release leaks it, and nothing here
needs them. The prompt is kept, so a corpus carrying personal data produces a
ledger carrying it too — the file belongs wherever the corpus does.
"""

from __future__ import annotations

import json
import os
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


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[None]:
    """Name what the calls inside this block were for.

    Nested rather than passed down: the transport cannot know it is serving a
    family audit, and threading a label through five signatures to tell it
    would put a logging concern in every function between.
    """
    previous = _context()
    parent = previous.get("event_id")
    merged = {
        **previous,
        **{key: value for key, value in fields.items() if value is not None},
        "stage": name,
        "event_id": uuid.uuid4().hex[:16],
        "parent_event_id": parent,
    }
    _local.context = merged
    try:
        yield
    finally:
        _local.context = previous


def enabled() -> bool:
    return bool(os.environ.get(_ENV_PATH))


def record(event: dict[str, Any]) -> None:
    """Append one event. Never raises: a ledger failure must not fail a run.

    A run that dies because its logging could not write would trade the thing
    being recorded for the record of it.
    """
    path = os.environ.get(_ENV_PATH)
    if not path:
        return
    row = {"recorded_at": _now(), **_context(), **event}
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except OSError:
        return


@contextmanager
def model_call(*, provider: str, model: str, request: Any) -> Iterator[dict[str, Any]]:
    """Record one physical provider request, whether or not it succeeds.

    A failed call is the one most worth having: it still consumed a rate-limit
    budget and may have cost tokens, and its absence is what made a rerun that
    behaved differently impossible to explain. The event is written on the way
    out either way.
    """
    slot: dict[str, Any] = {}
    started = time.monotonic()
    started_at = _now()
    status = "success"
    error: dict[str, Any] | None = None
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
        record(
            {
                "event_type": "model_call",
                "provider": provider,
                "model": model,
                "request": request,
                "response": response,
                "usage": (response or {}).get("usage") if isinstance(response, dict) else None,
                "status": status,
                "error": error,
                "started_at": started_at,
                "finished_at": _now(),
                "duration_seconds": round(time.monotonic() - started, 4),
            }
        )


def record_attempt(*, attempt: int, error: Exception, delay: float) -> None:
    """One failed try inside a retried call, and what it cost to wait.

    Kept separate from the call it belongs to: a call that succeeded on its
    third attempt looks identical to one that succeeded immediately, except in
    latency and in the rate-limit budget it spent getting there.
    """
    event: dict[str, Any] = {
        "event_type": "retry",
        "attempt": attempt,
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
