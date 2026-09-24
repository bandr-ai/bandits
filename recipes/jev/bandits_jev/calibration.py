"""Temperature scaling: one scalar per trained model, fitted on the
calibration split from the raw logits a scorer run already stored.

Refitting never rescores anything -- ``OptionScore`` keeps every pass's raw
logits precisely so a temperature can be fitted and applied later without
the model. A temperature is fitted only on a run whose split is
``calibration``; fitting on dev would tune against the go/no-go set and
fitting on test would tune against the locked result.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from bandits.store import Contract, DerivedEnvelope, DerivedStore
from bandits_jev.dataset import DecisionExample
from bandits_jev.metrics import argmax_option, ece, row_brier, row_nll
from bandits_jev.scorer import DecisionScoreResult, ScorerRun, _softmax

CALIBRATION_SPLIT = "calibration"
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
_GRID_POINTS = 81
_REFINE_ITERATIONS = 60


class TemperatureCalibration(Contract):
    scorer_run_id: str
    """The calibration-split run the temperature was fitted on."""
    dataset_id: str
    model_id: str
    revision: str
    adapter_digest: str | None
    mode: str
    prompt_version: int
    template_digest: str
    temperature: float
    at_bound: bool
    """True when the fit landed on ``MIN_TEMPERATURE`` or ``MAX_TEMPERATURE``:
    the search range, not the data, decided the value."""
    rows: int
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    ece_bins: int


def tempered_probabilities(result: DecisionScoreResult, temperature: float) -> dict[str, float]:
    """The distribution this result would have reported at ``temperature``,
    rebuilt from its raw logits. At temperature 1 this reproduces the stored
    probabilities exactly (up to float rounding). A two-order result is the
    mean of both passes' tempered softmax, matching how the scorer averages."""
    first = _softmax({s.option_id: s.raw_logit_pass1 / temperature for s in result.scores})
    if result.mode != "two_order_average":
        return first
    if any(s.raw_logit_pass2 is None for s in result.scores):
        raise ValueError(f"{result.decision_id}: two-order result is missing pass-2 logits")
    second = _softmax({s.option_id: s.raw_logit_pass2 / temperature for s in result.scores})
    return {o: (first[o] + second[o]) / 2 for o in first}


def _mean_nll(
    results: list[DecisionScoreResult], targets: Mapping[str, Mapping[str, float]], temperature: float
) -> float:
    return sum(
        row_nll(tempered_probabilities(r, temperature), targets[r.decision_id]) for r in results
    ) / len(results)


def fit_temperature(
    results: list[DecisionScoreResult], targets: Mapping[str, Mapping[str, float]]
) -> float:
    """The temperature minimizing mean NLL, searched in log space over
    [MIN_TEMPERATURE, MAX_TEMPERATURE]: a fixed grid, then golden-section
    refinement between the best grid point's neighbours. No randomness and
    no optimizer state, so the same run always fits the same value."""
    if not results:
        raise ValueError("cannot fit a temperature on zero scored rows")
    lo, hi = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    step = (hi - lo) / (_GRID_POINTS - 1)
    grid = [lo + i * step for i in range(_GRID_POINTS)]
    losses = [_mean_nll(results, targets, math.exp(x)) for x in grid]
    best = min(range(_GRID_POINTS), key=lambda i: (losses[i], i))
    a, b = grid[max(best - 1, 0)], grid[min(best + 1, _GRID_POINTS - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = _mean_nll(results, targets, math.exp(c)), _mean_nll(results, targets, math.exp(d))
    for _ in range(_REFINE_ITERATIONS):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = _mean_nll(results, targets, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = _mean_nll(results, targets, math.exp(d))
    refined = (a + b) / 2
    if _mean_nll(results, targets, math.exp(refined)) > losses[best]:
        refined = grid[best]
    return math.exp(refined)


def _summary(
    results: list[DecisionScoreResult],
    targets: Mapping[str, Mapping[str, float]],
    temperature: float,
    n_bins: int,
) -> tuple[float, float, float]:
    nll = brier = 0.0
    pairs: list[tuple[float, bool]] = []
    for r in results:
        probs = tempered_probabilities(r, temperature)
        target = targets[r.decision_id]
        nll += row_nll(probs, target)
        brier += row_brier(probs, target)
        chosen = argmax_option(probs)
        pairs.append((probs[chosen], chosen == argmax_option(target)))
    n = len(results)
    return nll / n, brier / n, ece(pairs, n_bins) or 0.0


def calibrate(
    run: ScorerRun,
    scorer_run_id: str,
    examples: list[DecisionExample],
    *,
    n_bins: int = 10,
) -> TemperatureCalibration:
    """Fit one temperature on a calibration-split run. Refuses any other
    split, and refuses imported predictions (they have no logits to scale)."""
    if run.split != CALIBRATION_SPLIT:
        raise ValueError(
            f"temperature is fitted only on the {CALIBRATION_SPLIT!r} split; this run scored "
            f"{run.split!r}"
        )
    if run.predictions_source is not None:
        raise ValueError(f"cannot calibrate imported predictions ({run.predictions_source})")
    if run.dataset_id is None:
        raise ValueError("cannot calibrate a scorer run that does not record its dataset")
    targets = {e.decision_id: e.target.probabilities for e in examples}
    missing = [r.decision_id for r in run.results if r.decision_id not in targets]
    if missing:
        raise ValueError(f"{len(missing)} scored row(s) are not in the dataset: {missing[:5]}")
    results = list(run.results)
    temperature = fit_temperature(results, targets)
    nll_before, brier_before, ece_before = _summary(results, targets, 1.0, n_bins)
    nll_after, brier_after, ece_after = _summary(results, targets, temperature, n_bins)
    return TemperatureCalibration(
        scorer_run_id=scorer_run_id,
        dataset_id=run.dataset_id,
        model_id=run.model_id,
        revision=run.revision,
        adapter_digest=run.adapter_digest,
        mode=run.mode,
        prompt_version=run.prompt_version,
        template_digest=run.template_digest,
        temperature=temperature,
        at_bound=math.isclose(temperature, MIN_TEMPERATURE, rel_tol=1e-6)
        or math.isclose(temperature, MAX_TEMPERATURE, rel_tol=1e-6),
        rows=len(results),
        nll_before=nll_before,
        nll_after=nll_after,
        brier_before=brier_before,
        brier_after=brier_after,
        ece_before=ece_before,
        ece_after=ece_after,
        ece_bins=n_bins,
    )


def compute_calibration_id(calibration: TemperatureCalibration) -> str:
    digest = hashlib.sha256(calibration.model_dump_json().encode()).hexdigest()
    return f"decision-calibration-{digest[:16]}"


def save_calibration(calibration: TemperatureCalibration, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_calibration_id(calibration),
        kind="decision_calibration",
        parent_artifact_id=calibration.scorer_run_id,
        payload=calibration.model_dump_json().encode(),
        summary={"rows": calibration.rows},
    )


def load_calibration(calibration_id: str, store: DerivedStore) -> TemperatureCalibration:
    return TemperatureCalibration.model_validate_json(store.read_payload(calibration_id))
