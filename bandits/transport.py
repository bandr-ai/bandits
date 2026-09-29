"""Retry the model calls that a retry actually fixes.

Written against Fireworks, whose limits set the policy, and applied to every
provider: the statuses below mean the same thing everywhere. Fireworks enforces serverless limits by tokens per minute, per account and per
model, and answers a breach with HTTP 429. It asks for exponential backoff by
name, and warns that staying inside the limits still does not guarantee a
request succeeds: 503 overload is possible at any tier. Both are transient by
construction, so both are worth sleeping through.

Retrying anything else would be wrong. A 400 is a malformed request and will be
malformed again; a 401 is a bad key. Those raise on the first attempt so the
caller sees the real error rather than the same one three sleeps later.

``Retry-After``, when the response carries it, is the server saying how long it
wants; it wins over the doubling schedule.

Two error shapes arrive here: ``urllib``'s, and LiteLLM's, which carry the
provider's HTTP status as ``status_code`` and report a dropped connection as a
500 and a client timeout as a 408.
"""

from __future__ import annotations

import random
import time
import urllib.error
from collections.abc import Callable

from bandits import ledger

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
"""429 rate limit, the transient server-side failures around it, and 408, the
status LiteLLM gives a request that timed out."""

MAX_ATTEMPTS = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0


def response_header(error: BaseException, name: str) -> str | None:
    """A header from the failed response, whichever client raised it."""
    headers = getattr(error, "headers", None)
    if headers:
        # urllib's are an email Message, case-insensitive on lookup.
        value = headers.get(name)
        if value:
            return str(value)
    # LiteLLM's are a plain dict, lower-cased; its own `headers` is often None.
    raw = getattr(error, "litellm_response_headers", None) or {}
    value = {str(key).lower(): val for key, val in raw.items()}.get(name.lower())
    return str(value) if value else None


def http_status(error: BaseException) -> int | None:
    """The response's status: ``code`` on urllib's errors, ``status_code`` on
    LiteLLM's (whose ``code`` can be the same number as a string)."""
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after(error: BaseException) -> float | None:
    """Seconds the server asked us to wait, if it named a number."""
    raw = response_header(error, "Retry-After")
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except ValueError:
        # The header also permits an HTTP date. Falling back to the computed
        # schedule is better than parsing dates against a skewed local clock.
        return None
    return seconds if seconds >= 0 else None


def backoff_delay(attempt: int, error: BaseException | None = None) -> float:
    """Delay before ``attempt`` (1-based), honouring Retry-After when present.

    Jittered: without it, several calls throttled by the same minute would wake
    together and rebuild the burst that got them throttled.
    """
    if error is not None:
        asked = _retry_after(error)
        if asked is not None:
            return min(asked, MAX_DELAY)
    ceiling = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
    return random.uniform(0.0, ceiling)


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_STATUSES
    # A timeout or a dropped connection says nothing about the request itself.
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    return http_status(exc) in RETRY_STATUSES


def request_with_retry(
    send: Callable[[], object],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Call ``send``, sleeping through 429s and transient server errors.

    The last exception is re-raised once the attempts are spent, so the caller
    reports the real transport failure rather than a wrapper that hides it.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return send()
        except Exception as exc:
            if not is_retryable(exc) or attempt == max_attempts:
                raise
            last = exc
            delay = backoff_delay(attempt, exc)
            # Recorded before the sleep, so a run killed mid-backoff still says
            # what it was waiting for and how long it meant to wait.
            ledger.record_attempt(attempt=attempt, error=exc, delay=delay)
            sleep(delay)
    raise last  # unreachable: the loop either returns or raises
