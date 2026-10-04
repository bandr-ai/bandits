"""Admission control for every model call one mining session makes.

The old guard was checked between chunks against figures read back afterwards.
That bounds nothing inside a chunk: one RLM invocation makes root calls,
subcalls, adapter fallbacks, extraction and repair, and a twenty-trace chunk
spent 30 to 39 calls before the next check could fire. So the check moves to
the one place every call passes — the language model's ``forward`` — and runs
*before* dispatch.

What it can promise and what it cannot. Call counts and wall time are exact at
admission. Money is harder: a provider prices a call only after it returns. With
per-token prices supplied, each call reserves its worst case (prompt bytes as an
upper bound on input tokens, plus the full output allowance) before it is sent,
so the ceiling is never crossed by admitted work. Without prices, admission can
only refuse once *reported* spend reaches the ceiling, so the last admitted call
can overshoot by its own cost; that is reported, never hidden. A call whose cost
comes back unknown is not counted as free: with a ceiling and no prices to bound
it, the session stops.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

from bandits.analyze.rlm_models import StopReason


class BudgetExhausted(RuntimeError):
    """A call was refused before dispatch because a session ceiling was reached."""

    def __init__(self, reason: StopReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass
class _Ticket:
    reserved: float


class SessionBudgetGuard:
    """Shared ceilings, counted and enforced before each model call."""

    def __init__(
        self,
        *,
        max_calls: int,
        max_seconds: float,
        max_usd: float | None = None,
        usd_per_mtok_in: float | None = None,
        usd_per_mtok_out: float | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if (usd_per_mtok_in is None) != (usd_per_mtok_out is None):
            raise ValueError("give both input and output prices, or neither")
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.max_usd = max_usd
        self.price_in = usd_per_mtok_in
        self.price_out = usd_per_mtok_out
        self._clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self.calls_admitted = 0
        self.calls_settled = 0
        self.usd_reported = 0.0
        self.usd_estimated = 0.0
        """Spend computed from returned usage and the supplied prices, for calls
        the provider did not price."""

        self.unknown_cost_calls = 0
        self.reserved = 0.0
        self.refusals: list[str] = []

    @property
    def can_reserve(self) -> bool:
        return self.price_in is not None and self.price_out is not None

    def elapsed(self) -> float:
        return self._clock() - self.started

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    @property
    def spent(self) -> float:
        return self.usd_reported + self.usd_estimated

    def check(self) -> None:
        """Refuse further work at a boundary that is not a model call (a tool, a chunk)."""
        with self._lock:
            self._check_locked(reserve=0.0)

    def _check_locked(self, *, reserve: float) -> None:
        if self.calls_admitted >= self.max_calls:
            self._refuse(
                StopReason.MAX_LLM_CALLS, f"session call ceiling of {self.max_calls} reached"
            )
        if self.elapsed() >= self.max_seconds:
            self._refuse(
                StopReason.MAX_SECONDS,
                f"session wall-time ceiling of {self.max_seconds:g}s reached",
            )
        if self.max_usd is not None:
            if self.unknown_cost_calls and not self.can_reserve:
                self._refuse(
                    StopReason.MAX_USD,
                    f"{self.unknown_cost_calls} call(s) returned no cost and no prices were "
                    "supplied, so the monetary ceiling can no longer be enforced",
                )
            if self.spent + self.reserved + reserve > self.max_usd:
                self._refuse(
                    StopReason.MAX_USD,
                    f"monetary ceiling ${self.max_usd:.4f}: ${self.spent:.4f} spent, "
                    f"${self.reserved:.4f} reserved in flight, ${reserve:.4f} needed for this call",
                )

    def _refuse(self, reason: StopReason, detail: str) -> None:
        self.refusals.append(detail)
        raise BudgetExhausted(reason, detail)

    def _worst_case(self, prompt: Any, messages: Any, max_tokens: Any) -> float:
        if not self.can_reserve:
            return 0.0
        # Each token covers at least one byte, so the encoded request bounds input tokens.
        size = len(
            json.dumps({"prompt": prompt, "messages": messages}, default=str).encode("utf-8")
        )
        output = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 0
        return size / 1e6 * self.price_in + output / 1e6 * self.price_out

    def admit(self, *, prompt: Any, messages: Any, max_tokens: Any) -> _Ticket:
        with self._lock:
            reserve = self._worst_case(prompt, messages, max_tokens)
            if self.max_usd is not None and self.can_reserve and not isinstance(max_tokens, int):
                self._refuse(
                    StopReason.MAX_USD,
                    "no output token ceiling is set, so this call's cost cannot be reserved",
                )
            self._check_locked(reserve=reserve)
            self.calls_admitted += 1
            self.reserved += reserve
            return _Ticket(reserve)

    def settle(self, ticket: _Ticket, response: Any) -> None:
        """Release the reservation and count what the call actually cost."""
        with self._lock:
            self.reserved -= ticket.reserved
            self.calls_settled += 1
            if response is None:
                # Failed calls may still be billed; with a reservation, keep it.
                self.usd_estimated += ticket.reserved
                return
            cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", None) if usage is not None else None
            if isinstance(cost, (int, float)) and not (cost == 0 and tokens):
                self.usd_reported += float(cost)
                return
            # Unknown, or a zero price on a call that used tokens: never free.
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = (
                getattr(usage, "completion_tokens", None) if usage is not None else None
            )
            if (
                self.can_reserve
                and isinstance(prompt_tokens, int)
                and isinstance(completion_tokens, int)
            ):
                self.usd_estimated += (
                    prompt_tokens / 1e6 * self.price_in + completion_tokens / 1e6 * self.price_out
                )
            elif self.can_reserve:
                self.usd_estimated += ticket.reserved
            else:
                self.unknown_cost_calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "calls_admitted": self.calls_admitted,
            "calls_settled": self.calls_settled,
            "usd_reported": round(self.usd_reported, 8),
            "usd_estimated": round(self.usd_estimated, 8),
            "unknown_cost_calls": self.unknown_cost_calls,
            "elapsed_seconds": round(self.elapsed(), 3),
            "refusals": list(self.refusals),
        }
