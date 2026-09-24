"""Evaluation metrics for decision models: pure Python, no numpy, no torch.

A row is a predicted distribution over one example's options next to its
target distribution (one-hot for a hard label). Every metric here takes
those two dicts, so the same code scores an untrained run, a trained run, a
temperature-scaled run and imported third-party predictions identically.

The binary ``brier``/``reliability_bins``/``ece`` functions take
``(probability, outcome)`` pairs. Top-label calibration of a multi-option
model is exactly that shape -- (confidence in the chosen option, whether it
was right) -- so the multi-option ECE reuses them rather than re-deriving
the binning.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

NLL_PROBABILITY_FLOOR = 1e-15
"""A probability of exactly 0 on the gold option would make NLL infinite and
swamp every other row; it is floored here instead. The floor is part of the
metric definition and is recorded in every report that uses it."""


def brier(pairs: list[tuple[float, bool]]) -> float | None:
    """Mean squared error of a predicted probability against a 0/1 outcome."""
    if not pairs:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def reliability_bins(pairs: list[tuple[float, bool]], n_bins: int = 10) -> list[dict]:
    """Fixed-width probability bins: each bin's range, count, mean prediction
    and observed rate. Every bin from 0 to ``n_bins`` is reported, including
    empty ones (``n``: 0, ``mean_predicted``/``observed_rate``: None) -- an
    ECE number is easy to overread on a small slice without seeing which
    bins actually held data.
    """
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    out = []
    for i, bucket in enumerate(bins):
        out.append(
            {
                "range": [i / n_bins, (i + 1) / n_bins],
                "n": len(bucket),
                "mean_predicted": sum(p for p, _ in bucket) / len(bucket) if bucket else None,
                "observed_rate": sum(1.0 for _, y in bucket if y) / len(bucket)
                if bucket
                else None,
            }
        )
    return out


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float | None:
    """Expected calibration error: bin-size-weighted mean |predicted - observed|."""
    if not pairs:
        return None
    bins = reliability_bins(pairs, n_bins)
    total = len(pairs)
    return sum(
        b["n"] / total * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins if b["n"]
    )


def argmax_option(probabilities: Mapping[str, float]) -> str:
    """The highest-probability option, first in the dict's order on ties --
    the same rule the scorer uses to pick ``chosen_option_id``."""
    if not probabilities:
        raise ValueError("cannot take the argmax of an empty distribution")
    return max(probabilities.items(), key=lambda kv: kv[1])[0]


def is_correct(probabilities: Mapping[str, float], target: Mapping[str, float]) -> bool:
    return argmax_option(probabilities) == argmax_option(target)


def row_nll(probabilities: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Cross-entropy of the prediction against the target distribution (for a
    hard target, -log p(gold)), with ``NLL_PROBABILITY_FLOOR``."""
    return -sum(
        t * math.log(max(probabilities.get(option, 0.0), NLL_PROBABILITY_FLOOR))
        for option, t in target.items()
        if t > 0
    )


def row_brier(probabilities: Mapping[str, float], target: Mapping[str, float]) -> float:
    """Multiclass Brier: sum over options of (p - y)^2, in [0, 2]."""
    options = set(probabilities) | set(target)
    return sum((probabilities.get(o, 0.0) - target.get(o, 0.0)) ** 2 for o in options)


def macro_f1(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Unweighted mean F1 over every label that appears as a gold or a
    predicted option in ``(predicted, gold)`` pairs. Option ids are compared
    as-is, so per-row option sets (e.g. CaseHOLD's five candidate holdings)
    are scored by their ids, not their text."""
    if not pairs:
        return None
    labels = sorted({p for p, _ in pairs} | {g for _, g in pairs})
    scores = []
    for label in labels:
        tp = sum(1 for p, g in pairs if p == label and g == label)
        fp = sum(1 for p, g in pairs if p == label and g != label)
        fn = sum(1 for p, g in pairs if p != label and g == label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def total_variation(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    options = set(first) | set(second)
    return 0.5 * sum(abs(first.get(o, 0.0) - second.get(o, 0.0)) for o in options)


def grouped_bootstrap_interval(
    values: Sequence[float],
    groups: Sequence[str] | None = None,
    *,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Deterministic percentile bootstrap 95% interval of a mean.

    With ``groups``, whole groups are resampled with replacement and each
    draw's statistic is the pooled mean of the rows in the drawn groups, so
    related rows (one ``group_id``) move together and the interval is not
    narrowed by pretending they are independent. Without groups every row
    is its own group. For paired comparisons, pass per-row differences
    (candidate minus baseline on the same row).

    With every group a single row, this draws exactly the same random
    sequence as a plain per-row bootstrap, so ungrouped callers get the
    numbers they always got.
    """
    if not values:
        return None
    if groups is not None and len(groups) != len(values):
        raise ValueError(f"{len(groups)} group ids for {len(values)} values")
    if groups is None:
        clusters = [(v, 1) for v in values]
    else:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for value, group in zip(values, groups, strict=True):
            sums[group] = sums.get(group, 0.0) + value
            counts[group] = counts.get(group, 0) + 1
        clusters = [(sums[g], counts[g]) for g in sums]
    if len(clusters) == 1:
        mean = clusters[0][0] / clusters[0][1]
        return (mean, mean)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        drawn = [rng.choice(clusters) for _ in clusters]
        means.append(sum(s for s, _ in drawn) / sum(n for _, n in drawn))
    means.sort()
    return (means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))])
