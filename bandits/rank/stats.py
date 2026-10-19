"""Ranking statistics: pass@k, bootstrap intervals, ties and rank agreement.

Pure functions with no dependency on the emulator, so the hosted ranking and anyone checking it
use the same math. Results are per-task (passes, attempts) pairs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

ALPHA = 0.5
"""Predicted post-RL = pass@1 + ALPHA * headroom. Calibrated by backtests; 0.5 is the starting guess."""


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Probability that at least one of k draws from n attempts succeeds.

    The unbiased estimator, computed as the complement of drawing k failures.
    None when k exceeds what was actually run: reporting pass@10 from four
    attempts would be an extrapolation dressed as a measurement.
    """
    if n <= 0 or k <= 0 or k > n or c < 0:
        return None
    if c >= n or n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """A confidence interval that stays inside [0, 1] at small n."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


@dataclass(frozen=True)
class Summary:
    model: str
    tasks: int
    k: int
    pass_at_1: float
    pass_at_k: float
    headroom: float
    learnable: float
    """Share of tasks with 0 < passes < attempts: the ones GRPO can learn from."""
    predicted: float
    interval: tuple[float, float]
    """Bootstrap 95% interval over tasks for ``predicted``."""


def _point(per_task: Sequence[tuple[int, int]], k: int) -> tuple[float, float, float]:
    count = len(per_task)
    p1 = sum(c / n for c, n in per_task) / count
    pk = sum(pass_at_k(n, c, k) or 0.0 for c, n in per_task) / count
    learnable = sum(1 for c, n in per_task if 0 < c < n) / count
    return p1, pk, learnable


def summarize(
    model: str,
    per_task: Sequence[tuple[int, int]],
    *,
    alpha: float = ALPHA,
    resamples: int = 1000,
    seed: int = 0,
) -> Summary:
    """One candidate's row. k is the fewest attempts any task has, so pass@k is defined for all."""
    if not per_task or any(n <= 0 or not 0 <= c <= n for c, n in per_task):
        raise ValueError("per_task needs (passes, attempts) with 0 <= passes <= attempts > 0")
    k = min(n for _, n in per_task)
    p1, pk, learnable = _point(per_task, k)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        a, b, _ = _point([per_task[rng.randrange(len(per_task))] for _ in per_task], k)
        draws.append(a + alpha * (b - a))
    draws.sort()
    interval = (draws[int(0.025 * resamples)], draws[max(int(0.975 * resamples) - 1, 0)])
    return Summary(
        model, len(per_task), k, p1, pk, pk - p1, learnable, p1 + alpha * (pk - p1), interval
    )


def order(summaries: Sequence[Summary]) -> list[tuple[Summary, list[str]]]:
    """Best first by predicted post-RL, each with the models it cannot be separated from."""
    if len({s.k for s in summaries}) > 1:
        raise ValueError("summaries must share one k before they are compared")
    ranked = sorted(summaries, key=lambda s: s.predicted, reverse=True)
    return [
        (
            s,
            [
                o.model
                for o in ranked
                if o is not s and o.interval[0] <= s.interval[1] and s.interval[0] <= o.interval[1]
            ],
        )
        for s in ranked
    ]


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """Kendall's tau-a between two scorings of the same items; tied pairs count as 0."""
    n = len(a)
    if n != len(b) or n < 2:
        raise ValueError("need two equal-length sequences of at least 2 items")
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            product = (a[i] - a[j]) * (b[i] - b[j])
            total += (product > 0) - (product < 0)
    return total / math.comb(n, 2)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1 = smallest), so ties share their mean rank."""
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        for index in ordered[i : j + 1]:
            out[index] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation; None when either side is constant."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length sequences of at least 2 items")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else cov / math.sqrt(va * vb)
