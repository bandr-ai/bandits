"""The evaluation report: untrained vs trained vs trained + calibrated vs an
optional third-party column, on the same held-out items, rebuilt entirely
from saved artifacts.

Nothing here scores a model. Every number is recomputed from stored scorer
runs, a stored calibration and the dataset's targets, with a fixed bootstrap
seed and no timestamps, so the same artifacts always produce the same bytes.
Every aggregate is traceable: the report carries one row per evaluation item
and records each available column's chosen option and probability on the gold
option. Coverage-adjusted accuracy and paired comparisons count a rejected
item as incorrect, so rejecting hard items cannot improve a system's result.

Comparisons cover every row in the evaluation split and their 95% intervals
resample by ``group_id`` when the dataset has groups, so related rows are not
counted as independent evidence.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from bandits.store import Contract, DerivedEnvelope, DerivedStore
from bandits_jev.calibration import (
    TemperatureCalibration,
    load_calibration,
    tempered_probabilities,
)
from bandits_jev.cost import VerifierCost, load_verifier_cost
from bandits_jev.dataset import DecisionDataset, DecisionExample, load_decision_dataset
from bandits_jev.metrics import (
    NLL_PROBABILITY_FLOOR,
    argmax_option,
    ece,
    gold_option,
    grouped_bootstrap_interval,
    macro_f1,
    reliability_bins,
    row_brier,
    row_nll,
    total_variation,
)
from bandits_jev.scorer import (
    DecisionScoreResult,
    OptionScore,
    ScorerRun,
    _softmax,
    load_scorer_run,
)

REPORT_VERSION = 1
NOT_RUN = "not run"

UNTRAINED = "untrained"
UNTRAINED_TWO_ORDER = "untrained (two orders)"
TRAINED = "trained"
TRAINED_CALIBRATED = "trained + calibrated"
JEV = "jev"
VERIFIER = "verifier (reference)"
MAJORITY = "majority (train prior)"


class ReliabilityBin(Contract):
    low: float
    high: float
    n: int
    mean_confidence: float | None
    accuracy: float | None


class ColumnMetrics(Contract):
    rows: int
    coverage: float
    """Share of evaluation rows for which the column produced a prediction."""
    accuracy: float | None
    """Over scored rows whose target has a single top option. None when every
    scored row ties."""
    coverage_adjusted_accuracy: float | None
    """Correct predictions divided by every untied evaluation row; rejections
    are wrong. Tied rows are left out of both, as in ``accuracy``."""
    tied_target_rows: int
    """Scored rows whose target ties between options (e.g. a split vote). A
    tied row has no single right answer, so it is left out of accuracy,
    macro F1 and ECE and counted here; NLL and Brier still score it."""
    macro_f1: float | None
    nll: float
    brier: float
    ece: float | None
    reliability: tuple[ReliabilityBin, ...]
    order_agreement: float | None
    """Share of rows whose two option orders chose the same option. None
    unless the run scored two orders."""
    order_probability_shift: float | None
    """Mean total-variation distance between the two orders' distributions."""
    latency_mean_seconds: float | None
    """None for imported predictions that reported no latency, rather than
    a fake 0."""
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    latency_rows: int
    """Scored rows that actually reported latency."""


class RejectionSummary(Contract):
    count: int
    reasons: dict[str, int]
    """Reasons with digits collapsed to "N", so "prompt is 9123 tokens" and
    "prompt is 8411 tokens" count as one reason."""


class Column(Contract):
    name: str
    status: Literal["run", "not run"]
    scorer_run_id: str | None = None
    calibration_id: str | None = None
    temperature: float | None = None
    model_id: str | None = None
    revision: str | None = None
    adapter_digest: str | None = None
    mode: str | None = None
    predictions_source: str | None = None
    cost_usd: float | None = None
    """An imported run's total bill, as recorded at import."""
    cost_per_1k_usd: float | None = None
    """Dollars per 1,000 decisions over this column's scored rows. None when
    the price is unknown -- never 0 for "unknown"."""
    cost_basis: str | None = None
    """Where ``cost_per_1k_usd`` comes from, or why it is unknown."""
    metrics: ColumnMetrics | None = None
    rejections: RejectionSummary | None = None


class Interval(Contract):
    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


class PairedComparison(Contract):
    candidate: str
    baseline: str
    rows: int
    """Rows in the evaluation split; rejected rows are included."""
    resampling_units: int
    accuracy: Interval | None
    """candidate minus baseline over the untied rows (a rejected row counts
    as wrong); positive is better. None when every row ties."""
    nll: Interval
    """candidate minus baseline; negative is better."""
    brier: Interval
    """candidate minus baseline; negative is better."""


class RowPrediction(Contract):
    chosen: str
    gold_probability: float | None
    """None when the row's target ties (no single gold option)."""


class ReportRow(Contract):
    decision_id: str
    group: str
    gold: str | None
    """None when the target ties between options."""
    predictions: dict[str, RowPrediction]


class VersusVerifier(Contract):
    """The distillation headline, computed on one set of rows for both
    sides: the decisions the candidate scored. Rejected rows count as
    disagreement; a ratio whose inputs are missing for any compared row is
    left out (None, with the reason), never computed on a subset."""

    candidate: str
    rows: int
    """Evaluation rows, scored or not."""
    compared_rows: int
    """Rows the candidate scored: the rows both costs and latencies use."""
    agreement: float | None
    """The candidate's coverage-adjusted agreement with the verifier: correct
    over every untied evaluation row, a rejected row counting as a miss."""
    coverage: float
    cost_ratio: float | None
    """Verifier cost over candidate cost on the compared rows."""
    latency_ratio: float | None
    """Verifier median latency over candidate median latency on the
    compared rows where both were measured."""
    latency_rows: int
    notes: tuple[str, ...] = ()


class ReportSection(Contract):
    name: str
    dataset_id: str
    split: str
    items: int
    """Examples in the split, scored or not."""
    resample_by: Literal["group_id", "row"]
    columns: tuple[Column, ...]
    comparisons: tuple[PairedComparison, ...]
    rows: tuple[ReportRow, ...]
    versus_verifier: VersusVerifier | None = None


class TestRunRecord(Contract):
    scorer_run_id: str
    model_id: str
    adapter_digest: str | None
    predictions_source: str | None
    mode: str
    used_in_report: bool


class TestUsage(Contract):
    dataset_id: str
    runs: tuple[TestRunRecord, ...]
    """Every saved scorer run over this dataset's test split, used here or
    not. A test run this report does not use is a sign the locked split was
    scored more than once."""


class EvaluationReport(Contract):
    report_version: int = REPORT_VERSION
    bootstrap_draws: int
    bootstrap_seed: int
    ece_bins: int
    nll_probability_floor: float = NLL_PROBABILITY_FLOOR
    gpu_usd_per_hour: float | None = None
    """The hourly GPU price the local model columns' cost is computed from,
    as given; None when not given (those costs then read "unknown")."""
    verifier_cost_id: str | None = None
    sections: tuple[ReportSection, ...]
    test_usage: tuple[TestUsage, ...]
    """One audit for every report section evaluated on a locked test split."""


class _ColumnInput:
    def __init__(
        self,
        name: str,
        *,
        run_id: str | None = None,
        run: ScorerRun | None = None,
        calibration_id: str | None = None,
        calibration: TemperatureCalibration | None = None,
    ) -> None:
        self.name = name
        self.run_id = run_id
        self.run = run
        self.calibration_id = calibration_id
        self.calibration = calibration

    @property
    def temperature(self) -> float:
        return self.calibration.temperature if self.calibration is not None else 1.0


def _distribution(result: DecisionScoreResult, calibration: TemperatureCalibration | None) -> dict[str, float]:
    if calibration is None:
        return {s.option_id: s.probability for s in result.scores}
    return tempered_probabilities(result, calibration.temperature)


def _pass_distributions(
    result: DecisionScoreResult, temperature: float
) -> tuple[dict[str, float], dict[str, float]] | None:
    if result.mode != "two_order_average":
        return None
    first = _softmax({s.option_id: s.raw_logit_pass1 / temperature for s in result.scores})
    second = _softmax({s.option_id: s.raw_logit_pass2 / temperature for s in result.scores})
    return first, second


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty sequence."""
    index = math.ceil(q * len(sorted_values) - 1e-9) - 1
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def _rejections(run: ScorerRun) -> RejectionSummary:
    reasons: dict[str, int] = {}
    for rejection in run.rejections:
        for reason in rejection.reasons:
            key = re.sub(r"\d+", "N", reason)
            reasons[key] = reasons.get(key, 0) + 1
    return RejectionSummary(count=len(run.rejections), reasons=dict(sorted(reasons.items())))


def _column_metrics(
    column: _ColumnInput, examples: list[DecisionExample], n_bins: int
) -> ColumnMetrics | None:
    assert column.run is not None
    results = {r.decision_id: r for r in column.run.results}
    scored = [e for e in examples if e.decision_id in results]
    if not scored:
        return None
    confidence_pairs: list[tuple[float, bool]] = []
    label_pairs: list[tuple[str, str]] = []
    correct = 0
    tied_targets = 0
    nll = brier = 0.0
    agreements: list[bool] = []
    shifts: list[float] = []
    for example in scored:
        result = results[example.decision_id]
        probs = _distribution(result, column.calibration)
        target = example.target.probabilities
        chosen, gold = argmax_option(probs), gold_option(target)
        if gold is None:
            tied_targets += 1
        else:
            correct += int(chosen == gold)
            confidence_pairs.append((probs[chosen], chosen == gold))
            label_pairs.append((chosen, gold))
        nll += row_nll(probs, target)
        brier += row_brier(probs, target)
        passes = _pass_distributions(result, column.temperature)
        if passes is not None:
            agreements.append(argmax_option(passes[0]) == argmax_option(passes[1]))
            shifts.append(total_variation(*passes))
    n = len(scored)
    untied_scored = n - tied_targets
    untied_rows = sum(1 for e in examples if gold_option(e.target.probabilities) is not None)
    latencies = sorted(
        latency
        for e in scored
        if (latency := results[e.decision_id].latency_seconds) is not None
    )
    return ColumnMetrics(
        rows=n,
        coverage=n / len(examples),
        accuracy=correct / untied_scored if untied_scored else None,
        coverage_adjusted_accuracy=correct / untied_rows if untied_rows else None,
        tied_target_rows=tied_targets,
        macro_f1=macro_f1(label_pairs),
        nll=nll / n,
        brier=brier / n,
        ece=ece(confidence_pairs, n_bins),
        reliability=tuple(
            ReliabilityBin(
                low=b["range"][0],
                high=b["range"][1],
                n=b["n"],
                mean_confidence=b["mean_predicted"],
                accuracy=b["observed_rate"],
            )
            for b in reliability_bins(confidence_pairs, n_bins)
        ),
        order_agreement=sum(agreements) / len(agreements) if agreements else None,
        order_probability_shift=sum(shifts) / len(shifts) if shifts else None,
        latency_mean_seconds=sum(latencies) / len(latencies) if latencies else None,
        latency_p50_seconds=_percentile(latencies, 0.50) if latencies else None,
        latency_p95_seconds=_percentile(latencies, 0.95) if latencies else None,
        latency_rows=len(latencies),
    )


class _Pricing:
    def __init__(self, gpu_usd_per_hour: float | None, verifier_cost: VerifierCost | None) -> None:
        self.gpu_usd_per_hour = gpu_usd_per_hour
        self.verifier_cost = verifier_cost


def _cost(
    column: _ColumnInput, examples: list[DecisionExample], pricing: _Pricing
) -> tuple[float | None, str]:
    run = column.run
    assert run is not None
    ids = {e.decision_id for e in examples}
    scored = [r for r in run.results if r.decision_id in ids]
    if not scored:
        return None, "no rows scored"
    if column.name == MAJORITY:
        return None, "no model: a fixed answer"
    if column.name == VERIFIER:
        cost = pricing.verifier_cost
        if cost is None:
            return None, "unknown: no verifier cost given (jev verifier-cost)"
        covered = [cost.per_decision[r.decision_id] for r in scored if r.decision_id in cost.per_decision]
        if not covered:
            return None, "unknown: the ledger covers none of these decisions"
        per_1k = 1000 * sum(c.usd for c in covered) / len(covered)
        basis = (
            f"ledger `{cost.ledger}`, {len(covered)}/{len(scored)} decisions, "
            f"${cost.input_usd_per_mtok:g} in / ${cost.output_usd_per_mtok:g} out per Mtok"
        )
        over = sum(1 for r in scored if r.decision_id in set(cost.over_counted_decisions))
        if over:
            basis += f"; warning: {over} decision(s) have more calls than votes asked, cost may be overstated"
        return per_1k, basis
    if run.predictions_source is not None:
        if run.cost_usd is None:
            return None, "unknown: no bill recorded at import"
        return 1000 * run.cost_usd / len(run.results), f"actual bill ${run.cost_usd:g} for {len(run.results)} decisions"
    if pricing.gpu_usd_per_hour is None:
        return None, "unknown: no GPU price given (--gpu-usd-per-hour)"
    mean_seconds = sum(r.latency_seconds for r in scored) / len(scored)
    return (
        1000 * mean_seconds * pricing.gpu_usd_per_hour / 3600,
        f"GPU ${pricing.gpu_usd_per_hour:g}/h × measured latency",
    )


def _column(
    column: _ColumnInput, examples: list[DecisionExample], n_bins: int, pricing: _Pricing
) -> Column:
    if column.run is None:
        return Column(name=column.name, status=NOT_RUN)
    run = column.run
    cost_per_1k, cost_basis = _cost(column, examples, pricing)
    return Column(
        name=column.name,
        status="run",
        scorer_run_id=column.run_id,
        calibration_id=column.calibration_id,
        temperature=column.calibration.temperature if column.calibration is not None else None,
        model_id=run.model_id,
        revision=run.revision,
        adapter_digest=run.adapter_digest,
        mode=run.mode,
        predictions_source=run.predictions_source,
        cost_usd=run.cost_usd,
        cost_per_1k_usd=cost_per_1k,
        cost_basis=cost_basis,
        metrics=_column_metrics(column, examples, n_bins),
        rejections=_rejections(run),
    )


def _group_of(example: DecisionExample) -> str:
    return example.group_id if example.group_id is not None else example.decision_id


def _interval(values: list[float], groups: list[str], draws: int, seed: int) -> Interval:
    low, high = grouped_bootstrap_interval(values, groups, draws=draws, seed=seed)
    return Interval(estimate=sum(values) / len(values), low=low, high=high)


def _comparison(
    candidate: _ColumnInput,
    baseline: _ColumnInput,
    examples: list[DecisionExample],
    *,
    draws: int,
    seed: int,
) -> PairedComparison | None:
    if candidate.run is None or baseline.run is None:
        return None
    cand = {r.decision_id: r for r in candidate.run.results}
    base = {r.decision_id: r for r in baseline.run.results}
    accuracy: list[float] = []
    accuracy_groups: list[str] = []
    nll: list[float] = []
    brier: list[float] = []
    groups: list[str] = []
    for example in examples:
        target = example.target.probabilities
        c = (
            _distribution(cand[example.decision_id], candidate.calibration)
            if example.decision_id in cand
            else {}
        )
        b = (
            _distribution(base[example.decision_id], baseline.calibration)
            if example.decision_id in base
            else {}
        )
        gold = gold_option(target)
        if gold is not None:  # a tied row has no single right answer
            accuracy.append(
                float(bool(c) and argmax_option(c) == gold) - float(bool(b) and argmax_option(b) == gold)
            )
            accuracy_groups.append(_group_of(example))
        nll.append(row_nll(c, target) - row_nll(b, target))
        brier.append(row_brier(c, target) - row_brier(b, target))
        groups.append(_group_of(example))
    return PairedComparison(
        candidate=candidate.name,
        baseline=baseline.name,
        rows=len(examples),
        resampling_units=len(set(groups)),
        accuracy=_interval(accuracy, accuracy_groups, draws, seed) if accuracy else None,
        nll=_interval(nll, groups, draws, seed),
        brier=_interval(brier, groups, draws, seed),
    )


def _rows(columns: list[_ColumnInput], examples: list[DecisionExample]) -> tuple[ReportRow, ...]:
    by_column = {c.name: {r.decision_id: r for r in c.run.results} for c in columns if c.run is not None}
    rows = []
    for example in examples:
        gold = gold_option(example.target.probabilities)
        predictions = {}
        for column in columns:
            result = by_column.get(column.name, {}).get(example.decision_id)
            if result is None:
                continue
            probs = _distribution(result, column.calibration)
            predictions[column.name] = RowPrediction(
                chosen=argmax_option(probs),
                gold_probability=None if gold is None else probs.get(gold, 0.0),
            )
        rows.append(
            ReportRow(decision_id=example.decision_id, group=_group_of(example), gold=gold, predictions=predictions)
        )
    return tuple(rows)


def _section(
    name: str,
    dataset: DecisionDataset,
    dataset_id: str,
    split: str,
    columns: list[_ColumnInput],
    comparisons: list[tuple[str, str]],
    *,
    draws: int,
    seed: int,
    n_bins: int,
    pricing: _Pricing,
    prior_dataset: DecisionDataset | None = None,
) -> ReportSection:
    """``prior_dataset`` is the dataset the model trained on, when it is not
    ``dataset`` itself (external and held-out sections): the majority
    baseline answers with *its* train label mix."""
    examples = [e for e in dataset.examples if e.split == split]
    if not examples:
        raise ValueError(f"dataset {dataset_id} has no {split!r} examples")
    columns = (
        _reference_columns(dataset, dataset_id, split, examples, pricing.verifier_cost, prior_dataset=prior_dataset)
        + columns
    )
    ids = {e.decision_id for e in examples}
    for column in columns:
        if column.run is None:
            continue
        stray = [r.decision_id for r in column.run.results if r.decision_id not in ids]
        if stray:
            raise ValueError(
                f"{column.name}: {len(stray)} scored row(s) are not in {dataset_id}'s {split!r} split: {stray[:5]}"
            )
    by_name = {c.name: c for c in columns}
    built = [
        _comparison(by_name[cand], by_name[base], examples, draws=draws, seed=seed)
        for cand, base in comparisons
    ]
    built_columns = tuple(_column(c, examples, n_bins, pricing) for c in columns)
    return ReportSection(
        name=name,
        dataset_id=dataset_id,
        split=split,
        items=len(examples),
        resample_by="group_id" if any(e.group_id is not None for e in examples) else "row",
        columns=built_columns,
        comparisons=tuple(c for c in built if c is not None),
        rows=_rows(columns, examples),
        versus_verifier=_versus_verifier_numbers(columns, built_columns, examples, pricing),
    )


def _versus_verifier_numbers(
    inputs: list[_ColumnInput],
    columns: tuple[Column, ...],
    examples: list[DecisionExample],
    pricing: _Pricing,
) -> VersusVerifier | None:
    """How the trained model compares with the verifier it copies, on the
    rows the trained model scored. Only for a judge-labeled section."""
    by_input = {c.name: c for c in inputs}
    by_column = {c.name: c for c in columns}
    if VERIFIER not in by_input or by_input[VERIFIER].run is None:
        return None
    name = next(
        (n for n in (TRAINED_CALIBRATED, TRAINED) if by_input.get(n) is not None and by_input[n].run is not None),
        None,
    )
    if name is None or by_column[name].metrics is None:
        return None
    candidate = by_input[name].run
    metrics = by_column[name].metrics
    ids = {e.decision_id for e in examples}
    scored = [r for r in candidate.results if r.decision_id in ids]
    notes: list[str] = []

    cost_ratio = None
    verifier_cost = pricing.verifier_cost
    if verifier_cost is None:
        notes.append("no verifier cost given (--verifier-cost / --ledger)")
    elif pricing.gpu_usd_per_hour is None:
        notes.append("no GPU price given (--gpu-usd-per-hour)")
    else:
        uncovered = [r.decision_id for r in scored if r.decision_id not in verifier_cost.per_decision]
        unmeasured = [r.decision_id for r in scored if r.latency_seconds is None]
        if uncovered:
            notes.append(f"{len(uncovered)} compared decision(s) have no verifier cost in the ledger")
        if unmeasured:
            notes.append(f"{len(unmeasured)} compared decision(s) have no measured model latency")
        if scored and not uncovered and not unmeasured:
            verifier_usd = sum(verifier_cost.per_decision[r.decision_id].usd for r in scored)
            model_usd = sum(r.latency_seconds for r in scored) * pricing.gpu_usd_per_hour / 3600
            cost_ratio = verifier_usd / model_usd if model_usd > 0 else None

    latency_ratio = None
    pairs = [
        (verifier_cost.per_decision[r.decision_id].seconds, r.latency_seconds)
        for r in scored
        if verifier_cost is not None
        and r.decision_id in verifier_cost.per_decision
        and r.latency_seconds is not None
    ]
    if pairs:
        verifier_p50 = _percentile(sorted(v for v, _ in pairs), 0.50)
        model_p50 = _percentile(sorted(m for _, m in pairs), 0.50)
        latency_ratio = verifier_p50 / model_p50 if model_p50 > 0 else None
    return VersusVerifier(
        candidate=name,
        rows=len(examples),
        compared_rows=len(scored),
        agreement=metrics.coverage_adjusted_accuracy,
        coverage=metrics.coverage,
        cost_ratio=cost_ratio,
        latency_ratio=latency_ratio,
        latency_rows=len(pairs),
        notes=tuple(notes),
    )


def _derived_run(
    examples: list[DecisionExample],
    probabilities_of,
    *,
    dataset_id: str,
    split: str,
    source: str,
    latency_of=None,
) -> ScorerRun:
    """A run computed from the dataset itself rather than scored by a model,
    so the report can still be rebuilt from saved artifacts alone. Latency
    is unknown (None) unless ``latency_of`` supplies it for a row (the
    verifier's, from its ledger); never 0."""
    results = []
    for example in examples:
        probs = probabilities_of(example)
        scores = tuple(
            OptionScore(
                option_id=option,
                probability=p,
                raw_logit_pass1=math.log(max(p, NLL_PROBABILITY_FLOOR)),
            )
            for option, p in probs.items()
        )
        results.append(
            DecisionScoreResult(
                decision_id=example.decision_id,
                scores=scores,
                chosen_option_id=argmax_option(probs),
                candidate_token_mass=1.0,
                mode="single_order",
                prompt_digest="",
                latency_seconds=latency_of(example) if latency_of is not None else None,
            )
        )
    return ScorerRun(
        model_id=source,
        revision="",
        dtype="n/a",
        device="n/a",
        dataset_id=dataset_id,
        split=split,
        prompt_version=0,
        template_digest="",
        mode="single_order",
        max_prompt_tokens=0,
        results=tuple(results),
        rejections=(),
        predictions_source=source,
    )


def _train_prior(dataset: DecisionDataset) -> dict[str, float] | None:
    """Mean target distribution over the train split: what a model that
    ignores its input and always answers with the training label mix says."""
    train = [e for e in dataset.examples if e.split == "train"]
    if not train:
        return None
    totals: dict[str, float] = {}
    for example in train:
        for option, p in example.target.probabilities.items():
            totals[option] = totals.get(option, 0.0) + p
    return {option: total / len(train) for option, total in totals.items()}


def _reference_columns(
    dataset: DecisionDataset,
    dataset_id: str,
    split: str,
    examples: list[DecisionExample],
    verifier_cost: VerifierCost | None = None,
    *,
    prior_dataset: DecisionDataset | None = None,
) -> list[_ColumnInput]:
    """The yardsticks every trained model is read against.

    - verifier (reference): the verifier's own vote shares, present only
      when every row was labeled by the Bandits judge. It agrees with itself
      by definition; it is here so a reader sees what the student copies.
    - majority (train prior): always answer with the train split's label
      mix. A trained model that cannot beat this has learned nothing about
      its input, however good its accuracy looks on a skewed split.
    """
    columns = []
    judges = {e.judge.model for e in examples if e.judge is not None}
    if judges and all(e.judge is not None for e in examples):
        columns.append(
            _ColumnInput(
                VERIFIER,
                run=_derived_run(
                    examples,
                    lambda e: dict(e.target.probabilities),
                    dataset_id=dataset_id,
                    split=split,
                    source=f"derived:verifier_votes ({', '.join(sorted(judges))})",
                    latency_of=(
                        (lambda e: verifier_cost.per_decision[e.decision_id].seconds
                         if e.decision_id in verifier_cost.per_decision else None)
                        if verifier_cost is not None
                        else None
                    ),
                ),
            )
        )
    prior = _train_prior(prior_dataset if prior_dataset is not None else dataset)

    def prior_for(example: DecisionExample) -> dict[str, float]:
        restricted = {o: prior.get(o, 0.0) for o in example.options}
        total = sum(restricted.values())
        if total <= 0:
            return {o: 1.0 / len(example.options) for o in example.options}
        return {o: p / total for o, p in restricted.items()}

    columns.append(
        _ColumnInput(
            MAJORITY,
            run=_derived_run(
                examples, prior_for, dataset_id=dataset_id, split=split, source="derived:train_prior"
            ),
        )
        if prior is not None
        else _ColumnInput(MAJORITY)
    )
    return columns


def _check_same_items(runs: dict[str, ScorerRun]) -> tuple[str, str]:
    """Every run in a section must score the same dataset's same split."""
    keys = {(run.dataset_id, run.split) for run in runs.values()}
    if len(keys) != 1:
        detail = ", ".join(f"{name}={run.dataset_id}/{run.split}" for name, run in runs.items())
        raise ValueError(f"columns must score the same dataset and split: {detail}")
    dataset_id, split = keys.pop()
    if dataset_id is None or split is None:
        raise ValueError("every scorer run in a report must record its dataset and split")
    return dataset_id, split


def _check_evaluation_split(split: str, section: str) -> None:
    if split not in ("dev", "test"):
        raise ValueError(
            f"{section} report must evaluate a 'dev' or 'test' split, not {split!r}"
        )


def _check_models(
    untrained: ScorerRun,
    trained: ScorerRun,
    untrained_two_order: ScorerRun | None,
    calibration: TemperatureCalibration | None,
    trained_dataset_id: str,
) -> None:
    for name, run in (("untrained", untrained), ("untrained two-order", untrained_two_order)):
        if run is None:
            continue
        if run.predictions_source is not None or run.adapter_digest is not None:
            raise ValueError(f"the {name} run must be the base model with no adapter")
        if (run.model_id, run.revision) != (trained.model_id, trained.revision):
            raise ValueError(
                f"the {name} run scored {run.model_id}@{run.revision}, but the trained run's base is "
                f"{trained.model_id}@{trained.revision}"
            )
        if run.template_digest != trained.template_digest:
            raise ValueError(f"the {name} run used a different prompt template than the trained run")
    if untrained.mode != "single_order":
        raise ValueError("the untrained column must be a single-order run")
    if untrained_two_order is not None and untrained_two_order.mode != "two_order_average":
        raise ValueError("the untrained two-order column must be a two-order run")
    if trained.adapter_digest is None:
        raise ValueError("the trained run records no adapter; score it with --adapter")
    if calibration is None:
        return
    fitted = (calibration.model_id, calibration.revision, calibration.adapter_digest, calibration.mode,
              calibration.template_digest)
    scored = (trained.model_id, trained.revision, trained.adapter_digest, trained.mode, trained.template_digest)
    if fitted != scored:
        raise ValueError("the calibration was fitted for a different model, adapter, mode or prompt than the trained run")
    if calibration.dataset_id != trained_dataset_id:
        raise ValueError("the calibration was fitted on a different dataset's calibration split")


def _test_usage(store: DerivedStore, dataset_id: str, used: set[str]) -> TestUsage:
    records = []
    for envelope in sorted(store.list(kind="decision_scorer_run"), key=lambda e: e.artifact_id):
        run = load_scorer_run(envelope.artifact_id, store)
        if run.dataset_id != dataset_id or run.split != "test":
            continue
        records.append(
            TestRunRecord(
                scorer_run_id=envelope.artifact_id,
                model_id=run.model_id,
                adapter_digest=run.adapter_digest,
                predictions_source=run.predictions_source,
                mode=run.mode,
                used_in_report=envelope.artifact_id in used,
            )
        )
    return TestUsage(dataset_id=dataset_id, runs=tuple(records))


def build_report(
    store: DerivedStore,
    *,
    untrained_run_id: str,
    trained_run_id: str,
    untrained_two_order_run_id: str | None = None,
    calibration_id: str | None = None,
    jev_run_id: str | None = None,
    external: Sequence[tuple[str, str]] = (),
    draws: int = 2000,
    seed: int = 0,
    n_bins: int = 10,
    verifier_cost_id: str | None = None,
    gpu_usd_per_hour: float | None = None,
) -> EvaluationReport:
    """``external`` is a sequence of (untrained run id, trained run id) pairs
    scored on datasets the model never trained on; each becomes its own
    section, with the same temperature applied to its trained column."""
    untrained = load_scorer_run(untrained_run_id, store)
    trained = load_scorer_run(trained_run_id, store)
    two_order = load_scorer_run(untrained_two_order_run_id, store) if untrained_two_order_run_id else None
    jev = load_scorer_run(jev_run_id, store) if jev_run_id else None
    calibration = load_calibration(calibration_id, store) if calibration_id else None
    verifier_cost = load_verifier_cost(verifier_cost_id, store) if verifier_cost_id else None
    if gpu_usd_per_hour is not None and gpu_usd_per_hour < 0:
        raise ValueError("--gpu-usd-per-hour must be non-negative")
    pricing = _Pricing(gpu_usd_per_hour, verifier_cost)
    if jev is not None and jev.predictions_source is None:
        raise ValueError("the jev column must be an imported predictions run")

    main_runs = {UNTRAINED: untrained, TRAINED: trained}
    if two_order is not None:
        main_runs[UNTRAINED_TWO_ORDER] = two_order
    if jev is not None:
        main_runs[JEV] = jev
    dataset_id, split = _check_same_items(main_runs)
    _check_evaluation_split(split, "main")
    _check_models(untrained, trained, two_order, calibration, dataset_id)
    if trained.trained_on_dataset_id != dataset_id:
        raise ValueError(
            "the trained run's adapter was not trained on this dataset "
            f"({trained.trained_on_dataset_id!r} != {dataset_id!r})"
        )

    columns = [_ColumnInput(UNTRAINED, run_id=untrained_run_id, run=untrained)]
    columns.append(
        _ColumnInput(UNTRAINED_TWO_ORDER, run_id=untrained_two_order_run_id, run=two_order)
        if two_order is not None
        else _ColumnInput(UNTRAINED_TWO_ORDER)
    )
    columns.append(_ColumnInput(TRAINED, run_id=trained_run_id, run=trained))
    columns.append(
        _ColumnInput(
            TRAINED_CALIBRATED, run_id=trained_run_id, run=trained, calibration_id=calibration_id, calibration=calibration
        )
        if calibration is not None
        else _ColumnInput(TRAINED_CALIBRATED)
    )
    columns.append(_ColumnInput(JEV, run_id=jev_run_id, run=jev) if jev is not None else _ColumnInput(JEV))
    comparisons = [
        (TRAINED, UNTRAINED),
        (TRAINED_CALIBRATED, UNTRAINED),
        (TRAINED, MAJORITY),
        (TRAINED_CALIBRATED, MAJORITY),
        (TRAINED_CALIBRATED, JEV),
    ]

    main_dataset = load_decision_dataset(dataset_id, store)
    sections = [
        _section(
            "main",
            main_dataset,
            dataset_id,
            split,
            columns,
            comparisons,
            draws=draws,
            seed=seed,
            n_bins=n_bins,
            pricing=pricing,
        )
    ]
    test_usages: list[TestUsage] = []
    main_used = {untrained_run_id, trained_run_id}
    main_used |= {i for i in (untrained_two_order_run_id, jev_run_id) if i}
    if split == "test":
        test_usages.append(_test_usage(store, dataset_id, main_used))

    for ext_untrained_id, ext_trained_id in external:
        ext_untrained = load_scorer_run(ext_untrained_id, store)
        ext_trained = load_scorer_run(ext_trained_id, store)
        ext_dataset_id, ext_split = _check_same_items({UNTRAINED: ext_untrained, TRAINED: ext_trained})
        _check_evaluation_split(ext_split, "external")
        if ext_dataset_id == dataset_id:
            raise ValueError("an external set must be a different dataset than the main one")
        if (ext_trained.model_id, ext_trained.revision) != (trained.model_id, trained.revision):
            raise ValueError("the external trained run used a different base model or revision")
        if ext_trained.adapter_digest != trained.adapter_digest:
            raise ValueError("the external trained run used a different adapter than the main trained run")
        if ext_trained.trained_on_dataset_id != dataset_id:
            raise ValueError(
                "the external trained run's adapter does not record the main training dataset"
            )
        _check_models(ext_untrained, ext_trained, None, None, ext_dataset_id)
        if calibration is not None and (ext_trained.mode, ext_trained.template_digest) != (
            calibration.mode,
            calibration.template_digest,
        ):
            raise ValueError("the external trained run's mode or prompt does not match the calibration")
        ext_columns = [
            _ColumnInput(UNTRAINED, run_id=ext_untrained_id, run=ext_untrained),
            _ColumnInput(TRAINED, run_id=ext_trained_id, run=ext_trained),
            _ColumnInput(
                TRAINED_CALIBRATED,
                run_id=ext_trained_id,
                run=ext_trained,
                calibration_id=calibration_id,
                calibration=calibration,
            )
            if calibration is not None
            else _ColumnInput(TRAINED_CALIBRATED),
        ]
        sections.append(
            _section(
                f"external: {ext_dataset_id}",
                load_decision_dataset(ext_dataset_id, store),
                ext_dataset_id,
                ext_split,
                ext_columns,
                [(TRAINED, UNTRAINED), (TRAINED_CALIBRATED, UNTRAINED), (TRAINED, MAJORITY)],
                draws=draws,
                seed=seed,
                n_bins=n_bins,
                pricing=pricing,
                prior_dataset=main_dataset,
            )
        )
        if ext_split == "test":
            test_usages.append(
                _test_usage(store, ext_dataset_id, {ext_untrained_id, ext_trained_id})
            )
    return EvaluationReport(
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        ece_bins=n_bins,
        gpu_usd_per_hour=gpu_usd_per_hour,
        verifier_cost_id=verifier_cost_id,
        sections=tuple(sections),
        test_usage=tuple(test_usages),
    )


def report_json(report: EvaluationReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def compute_report_id(report: EvaluationReport) -> str:
    return f"decision-report-{hashlib.sha256(report_json(report).encode()).hexdigest()[:16]}"


def save_report(report: EvaluationReport, store: DerivedStore) -> DerivedEnvelope:
    main = report.sections[0]
    return store.write(
        compute_report_id(report),
        kind="decision_report",
        parent_artifact_id=main.dataset_id,
        payload=report_json(report).encode(),
        summary={"sections": len(report.sections), "rows": len(main.rows)},
    )


def load_report(report_id: str, store: DerivedStore) -> EvaluationReport:
    return EvaluationReport.model_validate_json(store.read_payload(report_id))


# ---------------------------------------------------------------- rendering


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _interval_text(interval: Interval | None) -> str:
    if interval is None:
        return "—"
    return f"{interval.estimate:+.4f} [{interval.low:+.4f}, {interval.high:+.4f}]"


_METRIC_ROWS: tuple[tuple[str, str, int], ...] = (
    ("rows scored", "rows", 0),
    ("coverage", "coverage", 4),
    ("accuracy (scored rows)", "accuracy", 4),
    ("coverage-adjusted accuracy", "coverage_adjusted_accuracy", 4),
    ("tied target rows (not in accuracy)", "tied_target_rows", 0),
    ("macro F1", "macro_f1", 4),
    ("NLL", "nll", 4),
    ("Brier", "brier", 4),
    ("ECE", "ece", 4),
    ("option-order agreement", "order_agreement", 4),
    ("option-order probability shift", "order_probability_shift", 4),
    ("latency mean (s)", "latency_mean_seconds", 4),
    ("latency p50 (s)", "latency_p50_seconds", 4),
    ("latency p95 (s)", "latency_p95_seconds", 4),
    ("latency rows", "latency_rows", 0),
)


def _versus_verifier(section: ReportSection) -> list[str]:
    """The distillation headline, only from numbers measured on the same
    rows (see ``VersusVerifier``); a missing ratio is left out and its
    reason printed."""
    versus = section.versus_verifier
    if versus is None:
        return []
    parts = []
    if versus.cost_ratio is not None:
        parts.append(f"{versus.cost_ratio:.1f}× cheaper per decision")
    if versus.latency_ratio is not None:
        parts.append(f"{versus.latency_ratio:.1f}× faster at the median")
    if versus.agreement is not None:
        parts.append(
            f"agrees with the verifier on {versus.agreement:.1%} of untied decisions "
            f"(rejected rows count as disagreement; coverage {versus.coverage:.1%})"
        )
    lines = [f"**{versus.candidate} vs the verifier** ({versus.compared_rows} of {versus.rows} rows scored): "
             + (", ".join(parts) if parts else "no measured comparison") + "."]
    for note in versus.notes:
        lines.append(f"- not computed: {note}")
    return lines + [""]


def _section_markdown(section: ReportSection, chart_name: str) -> list[str]:
    lines = [f"## {section.name}", ""]
    lines.append(
        f"Dataset `{section.dataset_id}`, split `{section.split}`, {section.items} items. "
        f"Intervals resample by {'group' if section.resample_by == 'group_id' else 'row'}."
    )
    lines.append("")
    if section.split == "dev":
        lines += [
            "**Selection-split warning:** checkpoints are selected by dev accuracy, so trained "
            "results on this split are optimistic and are not final test estimates.",
            "",
        ]
    names = [c.name for c in section.columns]
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for label, field, digits in _METRIC_ROWS:
        cells = []
        for column in section.columns:
            if column.metrics is None:
                cells.append(NOT_RUN if column.status == NOT_RUN else "no rows scored")
                continue
            value = getattr(column.metrics, field)
            cells.append(str(value) if digits == 0 and value is not None else _num(value, digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| rejected | "
        + " | ".join(NOT_RUN if c.rejections is None else str(c.rejections.count) for c in section.columns)
        + " |"
    )
    lines.append(
        "| temperature | " + " | ".join(_num(c.temperature) if c.temperature is not None else "—" for c in section.columns) + " |"
    )
    lines.append(
        "| cost per 1k decisions (USD) | "
        + " | ".join(
            NOT_RUN if c.status == NOT_RUN else (f"{c.cost_per_1k_usd:.4f}" if c.cost_per_1k_usd is not None else "unknown")
            for c in section.columns
        )
        + " |"
    )
    lines.append("")
    lines += _versus_verifier(section)
    if section.comparisons:
        lines += [
            "Paired differences, candidate − baseline, 95% bootstrap interval "
            "(accuracy: higher is better; NLL and Brier: lower is better):",
            "",
            "| candidate − baseline | rows | units | accuracy | NLL | Brier |",
            "|---|---|---|---|---|---|",
        ]
        for c in section.comparisons:
            lines.append(
                f"| {c.candidate} − {c.baseline} | {c.rows} | {c.resampling_units} | "
                f"{_interval_text(c.accuracy)} | {_interval_text(c.nll)} | {_interval_text(c.brier)} |"
            )
        lines.append("")
    rejected = [c for c in section.columns if c.rejections is not None and c.rejections.reasons]
    if rejected:
        lines += ["Rejection reasons:", ""]
        for column in rejected:
            for reason, count in column.rejections.reasons.items():
                lines.append(f"- {column.name}: {count} × {reason}")
        lines.append("")
    lines += [f"Reliability: ![reliability]({chart_name})", ""]
    lines += ["Sources:", ""]
    for column in section.columns:
        if column.status == NOT_RUN:
            lines.append(f"- {column.name}: {NOT_RUN}")
            continue
        if column.scorer_run_id is None:
            lines.append(
                f"- {column.name}: {column.predictions_source}, computed from the dataset itself; "
                f"cost: {column.cost_basis}"
            )
            continue
        source = f"`{column.scorer_run_id}`"
        if column.calibration_id:
            source += f" + `{column.calibration_id}`"
        model = column.predictions_source or f"{column.model_id}@{column.revision}"
        if column.adapter_digest:
            model += f" + adapter {column.adapter_digest}"
        lines.append(f"- {column.name}: {source} ({model}, {column.mode}); cost: {column.cost_basis}")
    lines.append("")
    return lines


_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2")


def reliability_svg(section: ReportSection) -> str:
    """A fixed-size reliability chart: accuracy against mean confidence per
    non-empty bin, one line per column, the diagonal for reference."""
    size, pad = 360, 40
    span = size - 2 * pad

    def xy(confidence: float, accuracy: float) -> str:
        return f"{pad + confidence * span:.2f},{size - pad - accuracy * span:.2f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size + 260}" height="{size}" '
        f'viewBox="0 0 {size + 260} {size}" font-family="sans-serif" font-size="11">',
        f'<rect x="0" y="0" width="{size + 260}" height="{size}" fill="#ffffff"/>',
        f'<rect x="{pad}" y="{pad}" width="{span}" height="{span}" fill="none" stroke="#999999"/>',
        f'<line x1="{pad}" y1="{size - pad}" x2="{size - pad}" y2="{pad}" stroke="#bbbbbb" stroke-dasharray="4 3"/>',
        f'<text x="{size / 2:.0f}" y="{size - 10}" text-anchor="middle">confidence</text>',
        f'<text x="12" y="{size / 2:.0f}" transform="rotate(-90 12 {size / 2:.0f})" text-anchor="middle">accuracy</text>',
    ]
    drawn = [c for c in section.columns if c.metrics is not None]
    for i, column in enumerate(drawn):
        color = _COLORS[i % len(_COLORS)]
        points = [
            xy(b.mean_confidence, b.accuracy)
            for b in column.metrics.reliability
            if b.n and b.mean_confidence is not None and b.accuracy is not None
        ]
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts += [f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3" fill="{color}"/>' for p in points]
        y = pad + 16 * i
        parts.append(f'<rect x="{size + 4}" y="{y - 8}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{size + 20}" y="{y + 1}">{column.name} (ECE {_num(column.metrics.ece, 3)})</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _chart_name(index: int) -> str:
    return f"reliability-{index}.svg"


def report_markdown(report: EvaluationReport, report_id: str) -> str:
    lines = [
        "# Decision model evaluation report",
        "",
        f"Report `{report_id}` (version {report.report_version}). Bootstrap: {report.bootstrap_draws} draws, "
        f"seed {report.bootstrap_seed}. ECE: {report.ece_bins} fixed-width bins over top-option confidence. "
        f"NLL floors probabilities at {report.nll_probability_floor:g}.",
        "",
        "Untrained softmax output is not calibrated; only the \"trained + calibrated\" column has a fitted "
        "temperature, and it was fitted on the calibration split only.",
        "",
    ]
    for index, section in enumerate(report.sections):
        lines += _section_markdown(section, _chart_name(index))
    if report.test_usage:
        lines += ["## Test split usage", ""]
        for usage in report.test_usage:
            runs = usage.runs
            unused = [r for r in runs if not r.used_in_report]
            lines.append(
                f"{len(runs)} saved scorer run(s) over `{usage.dataset_id}`'s test split; "
                f"{len(runs) - len(unused)} used here."
            )
            if unused:
                lines.append("")
                lines.append(
                    "**Warning:** the test split was also scored by runs this report does not use. If any "
                    "choice was made after seeing them, this test set is now a dev set; cut a new one."
                )
            lines.append("")
            for r in runs:
                who = r.predictions_source or r.model_id + (
                    f" + adapter {r.adapter_digest}" if r.adapter_digest else ""
                )
                lines.append(
                    f"- `{r.scorer_run_id}` {who}, {r.mode}"
                    f"{'' if r.used_in_report else ' (not used here)'}"
                )
            lines.append("")
    lines.append("Per-row predictions behind every number above are in `report.json` (`sections[*].rows`).")
    return "\n".join(lines) + "\n"


def write_report(report: EvaluationReport, output: Path) -> list[Path]:
    """Write report.json, report.md and one reliability chart per section.
    Pure functions of ``report``, so the same report writes the same bytes."""
    output.mkdir(parents=True, exist_ok=True)
    report_id = compute_report_id(report)
    files = {
        "report.json": report_json(report),
        "report.md": report_markdown(report, report_id),
    }
    for index, section in enumerate(report.sections):
        files[_chart_name(index)] = reliability_svg(section)
    written = []
    for name, text in files.items():
        path = output / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
