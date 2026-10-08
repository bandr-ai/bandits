"""The hosted Jev (TypeSafe System One) as a judge: one call per decision, probabilities back.

Used for both rewards in the real-Jev arms: step judging with the same state,
question and options as the small step judge, and final-answer judging with a
full-trajectory state that fits Jev's 32k-token context.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from bandits_jev.jev_api import DEFAULT_BASE_URL, _request

FULL_STATE_CHARS = 80_000
"""About 20k tokens: leaves room in Jev's 32k context for the question and criteria."""
ACTION_CHARS = 300


def full_outcome_state(question: str, events: Sequence[dict], final: str | None, max_chars: int = FULL_STATE_CHARS) -> str:
    """Every step's action and real result, then the final answer. Results are cut evenly
    so the whole trajectory fits; nothing is dropped from the middle."""
    head = f"Research question: {question}\n"
    tail = f"Final answer: {(final or '(no answer)')[:2000]}"
    actions = [f"{e['tool']} {e['action'][:ACTION_CHARS]}" for e in events]
    room = max_chars - len(head) - len(tail) - sum(len(a) + 40 for a in actions)
    per_result = max(200, room // max(1, len(events)))
    lines = [
        f"Step {i + 1}: {action}\nResult: {e['observation'][:per_result]}"
        for i, (action, e) in enumerate(zip(actions, events, strict=True))
    ]
    return head + "\n".join(lines) + ("\n" if lines else "") + tail


class RemoteJev:
    def __init__(self, api_key: str | None = None, *, model: str = "jev-latest", base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 60, attempts: int = 4, workers: int = 32) -> None:
        self.api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("set JEV_API_KEY")
        self.model, self.base_url, self.timeout, self.attempts, self.workers = model, base_url, timeout, attempts, workers
        self.usage: list[dict] = []

    def choose(self, state: str, instructions: str, criteria: dict[str, str]) -> dict[str, float]:
        example = SimpleNamespace(decision_id="x", state=state, question=instructions, options=criteria)
        row = _request(example, api_key=self.api_key, model=self.model, base_url=self.base_url,
                       timeout=self.timeout, attempts=self.attempts)
        self.usage.append(row.get("usage") or {})
        return row["probabilities"]

    def choose_many(self, items: Sequence[tuple[str, str, dict[str, str]]]) -> list[dict[str, float] | None]:
        """Probabilities per item, or None where the call failed after retries."""
        def one(item):
            try:
                return self.choose(*item)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one, items))


def usage_tokens(usage: Sequence[dict]) -> int:
    return sum(int(u.get("input_tokens") or u.get("prompt_tokens") or 0) for u in usage)


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)
