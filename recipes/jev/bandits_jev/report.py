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
from bandits_jev.dataset import DecisionDataset, DecisionExample, load_decision_dataset
from bandits_jev.metrics import (
    NLL_PROBABILITY_FLOOR,
    argmax_option,
    ece,
    grouped_bootstrap_interval,
    macro_f1,
    reliability_bins,
    row_brier,
    row_nll,
    total_variation,
)
from bandits_jev.scorer import DecisionScoreResult, ScorerRun, _softmax, load_scorer_run

REPORT_VERSION = 1
NOT_RUN = "not run"

UNTRAINED = "untrained"
UNTRAINED_TWO_ORDER = "untrained (two orders)"
TRAINED = "trained"
TRAINED_CALIBRATED = "trained + calibrated"
JEV = "jev"


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
    accuracy: float
    coverage_adjusted_accuracy: float
    """Correct predictions divided by all evaluation rows; rejections are wrong."""
    macro_f1: float
    nll: float
    brier: float
    ece: float
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
    accuracy: Interval
    """candidate minus baseline; positive is better."""
    nll: Interval
    """candidate minus baseline; negative is better."""
    brier: Interval
    """candidate minus baseline; negative is better."""


class RowPrediction(Contract):
    chosen: str
    gold_probability: float


class ReportRow(Contract):
    decision_id: str
    group: str
    gold: str
    predictions: dict[str, RowPrediction]


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
    sections: tuple[ReportSection, ...]
    test_usage: TestUsage | None


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
    nll = brier = 0.0
    agreements: list[bool] = []
    shifts: list[float] = []
    for example in scored:
        result = results[example.decision_id]
        probs = _distribution(result, column.calibration)
        target = example.target.probabilities
        chosen, gold = argmax_option(probs), argmax_option(target)
        confidence_pairs.append((probs[chosen], chosen == gold))
        label_pairs.append((chosen, gold))
        nll += row_nll(probs, target)
        brier += row_brier(probs, target)
        passes = _pass_distributions(result, column.temperature)
        if passes is not None:
            agreements.append(argmax_option(passes[0]) == argmax_option(passes[1]))
            shifts.append(total_variation(*passes))
    n = len(scored)
    latencies = sorted(results[e.decision_id].latency_seconds for e in scored)
    no_latency = column.run.predictions_source is not None and not any(latencies)
    return ColumnMetrics(
        rows=n,
        coverage=n / len(examples),
        accuracy=sum(1 for c, g in label_pairs if c == g) / n,
        coverage_adjusted_accuracy=sum(1 for c, g in label_pairs if c == g) / len(examples),
        macro_f1=macro_f1(label_pairs) or 0.0,
        nll=nll / n,
        brier=brier / n,
        ece=ece(confidence_pairs, n_bins) or 0.0,
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
        latency_mean_seconds=None if no_latency else sum(latencies) / n,
        latency_p50_seconds=None if no_latency else _percentile(latencies, 0.50),
        latency_p95_seconds=None if no_latency else _percentile(latencies, 0.95),
    )


def _column(column: _ColumnInput, examples: list[DecisionExample], n_bins: int) -> Column:
    if column.run is None:
        return Column(name=column.name, status=NOT_RUN)
    run = column.run
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
    nll: list[float] = []
    brier: list[float] = []
    groups: list[str] = []
    for example in examples:
        target = example.target.probabilities
        gold = argmax_option(target)
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
        accuracy.append(
            float(bool(c) and argmax_option(c) == gold)
            - float(bool(b) and argmax_option(b) == gold)
        )
        nll.append(row_nll(c, target) - row_nll(b, target))
        brier.append(row_brier(c, target) - row_brier(b, target))
        groups.append(_group_of(example))
    return PairedComparison(
        candidate=candidate.name,
        baseline=baseline.name,
        rows=len(examples),
        resampling_units=len(set(groups)),
        accuracy=_interval(accuracy, groups, draws, seed),
        nll=_interval(nll, groups, draws, seed),
        brier=_interval(brier, groups, draws, seed),
    )


def _rows(columns: list[_ColumnInput], examples: list[DecisionExample]) -> tuple[ReportRow, ...]:
    by_column = {c.name: {r.decision_id: r for r in c.run.results} for c in columns if c.run is not None}
    rows = []
    for example in examples:
        gold = argmax_option(example.target.probabilities)
        predictions = {}
        for column in columns:
            result = by_column.get(column.name, {}).get(example.decision_id)
            if result is None:
                continue
            probs = _distribution(result, column.calibration)
            predictions[column.name] = RowPrediction(
                chosen=argmax_option(probs), gold_probability=probs.get(gold, 0.0)
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
) -> ReportSection:
    examples = [e for e in dataset.examples if e.split == split]
    if not examples:
        raise ValueError(f"dataset {dataset_id} has no {split!r} examples")
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
    return ReportSection(
        name=name,
        dataset_id=dataset_id,
        split=split,
        items=len(examples),
        resample_by="group_id" if any(e.group_id is not None for e in examples) else "row",
        columns=tuple(_column(c, examples, n_bins) for c in columns),
        comparisons=tuple(c for c in built if c is not None),
        rows=_rows(columns, examples),
    )


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
) -> EvaluationReport:
    """``external`` is a sequence of (untrained run id, trained run id) pairs
    scored on datasets the model never trained on; each becomes its own
    section, with the same temperature applied to its trained column."""
    untrained = load_scorer_run(untrained_run_id, store)
    trained = load_scorer_run(trained_run_id, store)
    two_order = load_scorer_run(untrained_two_order_run_id, store) if untrained_two_order_run_id else None
    jev = load_scorer_run(jev_run_id, store) if jev_run_id else None
    calibration = load_calibration(calibration_id, store) if calibration_id else None
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
    comparisons = [(TRAINED, UNTRAINED), (TRAINED_CALIBRATED, UNTRAINED), (TRAINED_CALIBRATED, JEV)]

    sections = [
        _section(
            "main",
            load_decision_dataset(dataset_id, store),
            dataset_id,
            split,
            columns,
            comparisons,
            draws=draws,
            seed=seed,
            n_bins=n_bins,
        )
    ]
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
                [(TRAINED, UNTRAINED), (TRAINED_CALIBRATED, UNTRAINED)],
                draws=draws,
                seed=seed,
                n_bins=n_bins,
            )
        )

    used = {untrained_run_id, trained_run_id}
    used |= {i for i in (untrained_two_order_run_id, jev_run_id) if i}
    return EvaluationReport(
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        ece_bins=n_bins,
        sections=tuple(sections),
        test_usage=_test_usage(store, dataset_id, used) if split == "test" else None,
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


def _interval_text(interval: Interval) -> str:
    return f"{interval.estimate:+.4f} [{interval.low:+.4f}, {interval.high:+.4f}]"


_METRIC_ROWS: tuple[tuple[str, str, int], ...] = (
    ("rows scored", "rows", 0),
    ("coverage", "coverage", 4),
    ("accuracy (scored rows)", "accuracy", 4),
    ("coverage-adjusted accuracy", "coverage_adjusted_accuracy", 4),
    ("macro F1", "macro_f1", 4),
    ("NLL", "nll", 4),
    ("Brier", "brier", 4),
    ("ECE", "ece", 4),
    ("option-order agreement", "order_agreement", 4),
    ("option-order probability shift", "order_probability_shift", 4),
    ("latency mean (s)", "latency_mean_seconds", 4),
    ("latency p50 (s)", "latency_p50_seconds", 4),
    ("latency p95 (s)", "latency_p95_seconds", 4),
)


def _section_markdown(section: ReportSection, chart_name: str) -> list[str]:
    lines = [f"## {section.name}", ""]
    lines.append(
        f"Dataset `{section.dataset_id}`, split `{section.split}`, {section.items} items. "
        f"Intervals resample by {'group' if section.resample_by == 'group_id' else 'row'}."
    )
    lines.append("")
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
        "| cost (USD) | " + " | ".join(f"{c.cost_usd:.2f}" if c.cost_usd is not None else "—" for c in section.columns) + " |"
    )
    lines.append("")
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
        source = f"`{column.scorer_run_id}`"
        if column.calibration_id:
            source += f" + `{column.calibration_id}`"
        model = column.predictions_source or f"{column.model_id}@{column.revision}"
        if column.adapter_digest:
            model += f" + adapter {column.adapter_digest}"
        lines.append(f"- {column.name}: {source} ({model}, {column.mode})")
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
        parts.append(f'<text x="{size + 20}" y="{y + 1}">{column.name} (ECE {column.metrics.ece:.3f})</text>')
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
    if report.test_usage is not None:
        lines += ["## Test split usage", ""]
        runs = report.test_usage.runs
        unused = [r for r in runs if not r.used_in_report]
        lines.append(
            f"{len(runs)} saved scorer run(s) over `{report.test_usage.dataset_id}`'s test split; "
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
            who = r.predictions_source or r.model_id + (f" + adapter {r.adapter_digest}" if r.adapter_digest else "")
            lines.append(f"- `{r.scorer_run_id}` {who}, {r.mode}{'' if r.used_in_report else ' (not used here)'}")
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
