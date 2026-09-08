"""The retry layer sleeps through 429s and gives up on everything else."""

from __future__ import annotations

import io
import urllib.error

import pytest

from bandits.transport import MAX_DELAY, backoff_delay, is_retryable, request_with_retry


def _http(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("u", code, "boom", headers, io.BytesIO(b""))


class _Sleeps:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


def test_a_429_is_retried_and_can_succeed() -> None:
    calls = iter([_http(429), _http(429), "ok"])

    def send() -> object:
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    sleeps = _Sleeps()
    assert request_with_retry(send, sleep=sleeps) == "ok"
    assert len(sleeps.slept) == 2


def test_a_400_is_not_retried() -> None:
    sleeps = _Sleeps()

    def send() -> object:
        raise _http(400)

    with pytest.raises(urllib.error.HTTPError):
        request_with_retry(send, sleep=sleeps)
    assert sleeps.slept == []


def test_the_last_failure_is_raised_once_attempts_are_spent() -> None:
    def send() -> object:
        raise _http(503)

    with pytest.raises(urllib.error.HTTPError) as caught:
        request_with_retry(send, max_attempts=3, sleep=_Sleeps())
    assert caught.value.code == 503


def test_retry_after_wins_over_the_computed_schedule() -> None:
    assert backoff_delay(1, _http(429, "12")) == 12.0


def test_retry_after_is_capped() -> None:
    assert backoff_delay(1, _http(429, "9999")) == MAX_DELAY


def test_an_unparseable_retry_after_falls_back_to_jitter() -> None:
    assert 0.0 <= backoff_delay(1, _http(429, "Wed, 21 Oct 2015 07:28:00 GMT")) <= 1.0


def test_timeouts_are_retryable_and_http_400_is_not() -> None:
    assert is_retryable(TimeoutError("t"))
    assert is_retryable(_http(429))
    assert not is_retryable(_http(401))
