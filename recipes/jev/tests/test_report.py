from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from bandits.store import DerivedStore
from bandits_jev.calibration import calibrate, save_calibration
from bandits_jev.cli import app
from bandits_jev.report import (
    JEV,
    MAJORITY,
    NOT_RUN,
    TRAINED,
    TRAINED_CALIBRATED,
    UNTRAINED,
    UNTRAINED_TWO_ORDER,
    VERIFIER,
    build_report,
    compute_report_id,
    save_report,
    write_report,
)
from bandits_jev.scorer import import_predictions
from tests.eval_fixtures import ADAPTER, build_dataset, make_run, save


@pytest.fixture
def store(tmp_path) -> DerivedStore:
    return DerivedStore(tmp_path / "project" / ".bandits")


def _setup(store, *, group_size=1, rejected=0):
    dataset_id, dataset, truths = build_dataset(
        store, per_split={"calibration": 400, "test": 300}, group_size=group_size
    )
    test_ids = [e.decision_id for e in dataset.examples if e.split == "test"]
    reject = set(test_ids[:rejected])
    ids = {
        "dataset": dataset_id,
        # Weak, well-calibrated-ish untrained model vs a sharp, overconfident trained one.
        "untrained": save(
            make_run(dataset_id, dataset, truths, split="test", scale=0.3, noise=1.0, reject=reject), store
        ),
        "two_order": save(
            make_run(
                dataset_id, dataset, truths, split="test", scale=0.3, noise=1.0, two_order=True,
                order_bias=1.5, reject=reject,
            ),
            store,
        ),
        "trained": save(
            make_run(dataset_id, dataset, truths, split="test", scale=3.0, adapter_digest=ADAPTER, reject=reject),
            store,
        ),
    }
    cal_run = make_run(dataset_id, dataset, truths, split="calibration", scale=3.0, adapter_digest=ADAPTER)
    cal_run_id = save(cal_run, store)
    calibration = calibrate(
        cal_run, cal_run_id, [e for e in dataset.examples if e.split == "calibration"]
    )
    ids["calibration"] = save_calibration(calibration, store).artifact_id
    return ids, dataset, truths


def _columns(section):
    return {c.name: c for c in section.columns}


def test_report_has_every_column_and_jev_degrades_to_not_run(store) -> None:
    ids, _, _ = _setup(store)
    report = build_report(
        store,
        untrained_run_id=ids["untrained"],
        trained_run_id=ids["trained"],
        untrained_two_order_run_id=ids["two_order"],
        calibration_id=ids["calibration"],
        draws=300,
    )
    main = report.sections[0]
    columns = _columns(main)

    # No train split in this fixture, so the majority baseline cannot be formed;
    # rows are user-labeled, so there is no verifier column at all.
    assert list(columns) == [MAJORITY, UNTRAINED, UNTRAINED_TWO_ORDER, TRAINED, TRAINED_CALIBRATED, JEV]
    assert columns[MAJORITY].status == NOT_RUN
    assert columns[JEV].status == NOT_RUN and columns[JEV].metrics is None
    assert columns[TRAINED].metrics.accuracy > columns[UNTRAINED].metrics.accuracy
    # Temperature fixes confidence, not ranking: same accuracy, better NLL/ECE.
    assert columns[TRAINED_CALIBRATED].metrics.accuracy == columns[TRAINED].metrics.accuracy
    assert columns[TRAINED_CALIBRATED].metrics.nll < columns[TRAINED].metrics.nll
    assert columns[TRAINED_CALIBRATED].metrics.ece < columns[TRAINED].metrics.ece
    assert columns[TRAINED_CALIBRATED].temperature == pytest.approx(3.0, rel=0.25)
    # Only the two-order run can measure option-order sensitivity.
    assert columns[UNTRAINED].metrics.order_agreement is None
    assert columns[UNTRAINED_TWO_ORDER].metrics.order_agreement < 1.0
    assert columns[UNTRAINED_TWO_ORDER].metrics.order_probability_shift > 0.0
    assert columns[UNTRAINED_TWO_ORDER].metrics.latency_mean_seconds > columns[UNTRAINED].metrics.latency_mean_seconds

    comparison = next(c for c in main.comparisons if (c.candidate, c.baseline) == (TRAINED, UNTRAINED))
    assert comparison.accuracy.excludes_zero and comparison.accuracy.estimate > 0
    assert comparison.rows == 300 and comparison.resampling_units == 300
    assert not any(c.baseline == JEV for c in main.comparisons)
    assert main.resample_by == "row"


def test_every_aggregate_traces_to_per_row_predictions(store) -> None:
    ids, _, _ = _setup(store)
    report = build_report(
        store, untrained_run_id=ids["untrained"], trained_run_id=ids["trained"], draws=100
    )
    main = report.sections[0]
    trained = _columns(main)[TRAINED].metrics
    correct = sum(1 for r in main.rows if r.predictions[TRAINED].chosen == r.gold)

    assert len(main.rows) == main.items == 300
    assert correct / trained.rows == trained.accuracy


def test_report_regenerates_byte_identically(store, tmp_path) -> None:
    ids, _, _ = _setup(store, rejected=5)
    kwargs = dict(
        untrained_run_id=ids["untrained"],
        trained_run_id=ids["trained"],
        untrained_two_order_run_id=ids["two_order"],
        calibration_id=ids["calibration"],
        draws=200,
    )
    first, second = build_report(store, **kwargs), build_report(store, **kwargs)
    write_report(first, tmp_path / "one")
    write_report(second, tmp_path / "two")

    assert compute_report_id(first) == compute_report_id(second)
    for name in ("report.json", "report.md", "reliability-0.svg"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    assert save_report(first, store).artifact_id == save_report(second, store).artifact_id


def test_grouped_rows_resample_by_group(store) -> None:
    ids, _, _ = _setup(store, group_size=10)
    report = build_report(store, untrained_run_id=ids["untrained"], trained_run_id=ids["trained"], draws=300)
    main = report.sections[0]
    comparison = main.comparisons[0]

    assert main.resample_by == "group_id"
    assert comparison.rows == 300 and comparison.resampling_units == 30


def test_rejections_reduce_coverage_adjusted_metrics_and_stay_in_comparisons(store) -> None:
    ids, _, _ = _setup(store, rejected=7)
    report = build_report(store, untrained_run_id=ids["untrained"], trained_run_id=ids["trained"], draws=100)
    main = report.sections[0]
    untrained = _columns(main)[UNTRAINED]

    assert untrained.rejections.count == 7
    assert untrained.rejections.reasons == {"prompt is N tokens, over the N limit": 7}
    assert untrained.metrics.rows == 293
    assert untrained.metrics.coverage == 293 / 300
    assert untrained.metrics.coverage_adjusted_accuracy == pytest.approx(
        untrained.metrics.accuracy * untrained.metrics.coverage
    )
    assert main.comparisons[0].rows == 300


def test_jev_column_from_imported_predictions_records_the_bill(store) -> None:
    ids, dataset, truths = _setup(store)
    test = [e for e in dataset.examples if e.split == "test"]
    lines = [
        json.dumps({"decision_id": e.decision_id, "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}})
        for e in test[:-3]
    ]
    lines.append(json.dumps({"decision_id": test[-3].decision_id, "probabilities": {"a": 1.0}}))
    jev = import_predictions(
        "\n".join(lines), test, name="jev", model_id="jev", revision="2026-09-24",
        dataset_id=ids["dataset"], split="test", cost_usd=0.71,
    )
    jev_id = save(jev, store)
    report = build_report(
        store,
        untrained_run_id=ids["untrained"],
        trained_run_id=ids["trained"],
        calibration_id=ids["calibration"],
        jev_run_id=jev_id,
        draws=100,
    )
    column = _columns(report.sections[0])[JEV]

    assert column.status == "run" and column.cost_usd == 0.71
    assert column.metrics.rows == 297
    assert column.rejections.count == 3  # one malformed line, two missing: none silently dropped
    assert column.metrics.latency_mean_seconds is None  # not reported, so not shown as 0
    assert any(c.baseline == JEV for c in report.sections[0].comparisons)


def test_imported_latency_uses_only_rows_that_report_it(store) -> None:
    ids, dataset, _ = _setup(store)
    test = [e for e in dataset.examples if e.split == "test"]
    lines = []
    for index, example in enumerate(test):
        row = {
            "decision_id": example.decision_id,
            "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2},
        }
        if index < 2:
            row["latency_seconds"] = 2.0 + index * 2.0
        lines.append(json.dumps(row))
    jev = import_predictions(
        "\n".join(lines),
        test,
        name="jev",
        model_id="jev",
        revision="today",
        dataset_id=ids["dataset"],
        split="test",
    )
    report = build_report(
        store,
        untrained_run_id=ids["untrained"],
        trained_run_id=ids["trained"],
        jev_run_id=save(jev, store),
        draws=100,
    )
    metrics = _columns(report.sections[0])[JEV].metrics

    assert metrics.latency_rows == 2
    assert metrics.latency_mean_seconds == 3.0


def test_report_rejects_adapter_trained_on_a_different_dataset(store) -> None:
    ids, dataset, truths = _setup(store)
    wrong = make_run(
        ids["dataset"],
        dataset,
        truths,
        split="test",
        scale=3.0,
        adapter_digest=ADAPTER,
        trained_on_dataset_id="decision-dataset-old-all-train",
    )

    with pytest.raises(ValueError, match="adapter was not trained on this dataset"):
        build_report(
            store,
            untrained_run_id=ids["untrained"],
            trained_run_id=save(wrong, store),
        )


def test_test_usage_flags_test_runs_the_report_did_not_use(store) -> None:
    ids, dataset, truths = _setup(store)
    extra = save(
        make_run(ids["dataset"], dataset, truths, split="test", scale=2.0, adapter_digest="another-adapter"),
        store,
    )
    report = build_report(store, untrained_run_id=ids["untrained"], trained_run_id=ids["trained"], draws=100)
    assert len(report.test_usage) == 1
    usage = {r.scorer_run_id: r.used_in_report for r in report.test_usage[0].runs}

    assert usage[ids["untrained"]] and usage[ids["trained"]]
    assert usage[extra] is False
    assert usage[ids["two_order"]] is False


def test_external_set_is_its_own_section_with_the_same_temperature(store) -> None:
    ids, _, _ = _setup(store)
    ext_id, ext, ext_truths = build_dataset(store, per_split={"test": 120}, seed=9, tag="external")
    ext_untrained = save(make_run(ext_id, ext, ext_truths, split="test", scale=0.5), store)
    # The trained model got worse off-distribution: its logits are mostly noise here.
    ext_trained = save(
        make_run(
            ext_id,
            ext,
            ext_truths,
            split="test",
            scale=0.1,
            noise=3.0,
            adapter_digest=ADAPTER,
            trained_on_dataset_id=ids["dataset"],
        ),
        store,
    )
    report = build_report(
        store,
        untrained_run_id=ids["untrained"],
        trained_run_id=ids["trained"],
        calibration_id=ids["calibration"],
        external=[(ext_untrained, ext_trained)],
        draws=100,
    )
    main, external = report.sections

    assert external.name == f"external: {ext_id}"
    assert _columns(external)[TRAINED_CALIBRATED].temperature == _columns(main)[TRAINED_CALIBRATED].temperature
    assert external.comparisons[0].accuracy.estimate < 0  # the regression shows up, not hidden
    assert {usage.dataset_id for usage in report.test_usage} == {ids["dataset"], ext_id}


def test_mismatched_inputs_are_refused(store) -> None:
    ids, dataset, truths = _setup(store)
    # Calibration fitted for another adapter.
    other = make_run(ids["dataset"], dataset, truths, split="calibration", scale=3.0, adapter_digest="other")
    other_cal = save_calibration(
        calibrate(other, "x", [e for e in dataset.examples if e.split == "calibration"]), store
    ).artifact_id
    with pytest.raises(ValueError, match="different model, adapter"):
        build_report(store, untrained_run_id=ids["untrained"], trained_run_id=ids["trained"], calibration_id=other_cal)
    # Trained run without an adapter.
    with pytest.raises(ValueError, match="records no adapter"):
        build_report(store, untrained_run_id=ids["untrained"], trained_run_id=ids["untrained"])
    # Columns over different splits.
    cal_untrained = save(make_run(ids["dataset"], dataset, truths, split="calibration", scale=0.3), store)
    with pytest.raises(ValueError, match="same dataset and split"):
        build_report(store, untrained_run_id=cal_untrained, trained_run_id=ids["trained"])
    cal_trained = save(
        make_run(ids["dataset"], dataset, truths, split="calibration", scale=3.0, adapter_digest=ADAPTER),
        store,
    )
    with pytest.raises(ValueError, match="must evaluate a 'dev' or 'test' split"):
        build_report(store, untrained_run_id=cal_untrained, trained_run_id=cal_trained)


def test_external_set_must_match_main_base_model_and_revision(store) -> None:
    ids, _, _ = _setup(store)
    ext_id, ext, ext_truths = build_dataset(store, per_split={"test": 12}, seed=9, tag="external-model")
    ext_untrained_run = make_run(ext_id, ext, ext_truths, split="test", scale=0.5).model_copy(
        update={"model_id": "different-base"}
    )
    ext_trained_run = make_run(
        ext_id,
        ext,
        ext_truths,
        split="test",
        scale=0.5,
        adapter_digest=ADAPTER,
        trained_on_dataset_id=ids["dataset"],
    ).model_copy(update={"model_id": "different-base"})
    ext_untrained = save(ext_untrained_run, store)
    ext_trained = save(ext_trained_run, store)

    with pytest.raises(ValueError, match="different base model or revision"):
        build_report(
            store,
            untrained_run_id=ids["untrained"],
            trained_run_id=ids["trained"],
            external=[(ext_untrained, ext_trained)],
        )


def test_cli_calibrate_then_report(store, tmp_path) -> None:
    ids, dataset, truths = _setup(store)
    cal_run = save(
        make_run(ids["dataset"], dataset, truths, split="calibration", scale=3.0, adapter_digest=ADAPTER),
        store,
    )
    project = str(tmp_path / "project")
    runner = CliRunner()

    refused = runner.invoke(app, ["calibrate", ids["trained"], "--project", project])
    assert refused.exit_code == 1 and "calibration" in refused.output

    calibrated = runner.invoke(app, ["calibrate", cal_run, "--project", project])
    assert calibrated.exit_code == 0, calibrated.output
    calibration_id = calibrated.output.split("calibration_id:")[1].split()[0]

    args = [
        "report", "--untrained", ids["untrained"], "--trained", ids["trained"],
        "--calibration", calibration_id, "--draws", "100", "--project", project,
    ]
    first = runner.invoke(app, [*args, "--output", str(tmp_path / "a")])
    second = runner.invoke(app, [*args, "--output", str(tmp_path / "b")])
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert (tmp_path / "a" / "report.md").read_bytes() == (tmp_path / "b" / "report.md").read_bytes()
    markdown = (tmp_path / "a" / "report.md").read_text()
    assert f"- jev: {NOT_RUN}" in markdown
    assert "Test split usage" in markdown


def _judge_labeled(store, dataset, *, tie_every: int = 0):
    """Re-save a fixture dataset as if the Bandits judge had labeled it (a
    judge on every row), optionally turning every n-th test row's target
    into a 50/50 tie."""
    from bandits_jev.dataset import DecisionDataset, DecisionJudgeInfo, DecisionTarget
    from bandits_jev.importer import save_imported_dataset

    judge = DecisionJudgeInfo(
        model="judge/model", prompt_digest="p", temperature=0.0, votes_requested=2, votes_valid=2,
        settings_digest="s",
    )
    examples = []
    test_index = 0
    for e in dataset.examples:
        update = {"judge": judge}
        if e.split == "test":
            if tie_every and test_index % tie_every == 0:
                update["target"] = DecisionTarget(kind="soft", probabilities={"a": 0.5, "b": 0.5, "c": 0.0})
            test_index += 1
        examples.append(e.model_copy(update=update))
    relabeled = DecisionDataset.model_validate({**dataset.model_dump(), "examples": [x.model_dump() for x in examples]})
    return save_imported_dataset(relabeled, store, source_file="judge-labeled").artifact_id, relabeled


def test_verifier_and_majority_columns_frame_the_trained_model(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"train": 600, "calibration": 200, "test": 300})
    dataset_id, dataset = _judge_labeled(store, dataset)
    untrained = save(make_run(dataset_id, dataset, truths, split="test", scale=0.3, noise=1.0), store)
    trained = save(make_run(dataset_id, dataset, truths, split="test", scale=3.0, adapter_digest=ADAPTER), store)

    report = build_report(store, untrained_run_id=untrained, trained_run_id=trained, draws=200)
    main = report.sections[0]
    columns = _columns(main)

    assert list(columns)[:2] == [VERIFIER, MAJORITY]
    assert columns[VERIFIER].metrics.accuracy == 1.0  # agrees with itself by definition
    assert columns[VERIFIER].scorer_run_id is None and columns[VERIFIER].metrics.latency_mean_seconds is None
    majority = columns[MAJORITY].metrics
    assert len({r.predictions[MAJORITY].chosen for r in main.rows}) == 1  # always the same answer
    beats = next(c for c in main.comparisons if (c.candidate, c.baseline) == (TRAINED, MAJORITY))
    assert beats.accuracy.estimate == pytest.approx(columns[TRAINED].metrics.accuracy - majority.accuracy)
    assert beats.accuracy.excludes_zero


def test_tied_targets_are_counted_and_left_out_of_accuracy(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"train": 100, "test": 90})
    dataset_id, dataset = _judge_labeled(store, dataset, tie_every=3)
    untrained = save(make_run(dataset_id, dataset, truths, split="test", scale=0.3, noise=1.0), store)
    trained = save(make_run(dataset_id, dataset, truths, split="test", scale=3.0, adapter_digest=ADAPTER), store)

    report = build_report(store, untrained_run_id=untrained, trained_run_id=trained, draws=100)
    main = report.sections[0]
    trained_metrics = _columns(main)[TRAINED].metrics
    untied = [r for r in main.rows if r.gold is not None]

    assert trained_metrics.tied_target_rows == 30 and trained_metrics.rows == 90
    assert trained_metrics.accuracy == sum(r.predictions[TRAINED].chosen == r.gold for r in untied) / 60
    # Coverage-adjusted accuracy uses the same rule: tied rows are out of its denominator too.
    assert trained_metrics.coverage_adjusted_accuracy == trained_metrics.accuracy  # nothing rejected here
    assert all(r.predictions[TRAINED].gold_probability is None for r in main.rows if r.gold is None)
    comparison = next(c for c in main.comparisons if (c.candidate, c.baseline) == (TRAINED, UNTRAINED))
    assert comparison.rows == 90  # NLL/Brier still use every shared row


def test_costs_are_measured_or_unknown_never_zero(store) -> None:
    from bandits_jev.cost import DecisionCost, VerifierCost, save_verifier_cost

    dataset_id, dataset, truths = build_dataset(store, per_split={"train": 200, "test": 100})
    dataset_id, dataset = _judge_labeled(store, dataset)
    untrained = save(make_run(dataset_id, dataset, truths, split="test", scale=0.3, noise=1.0), store)
    trained = save(make_run(dataset_id, dataset, truths, split="test", scale=3.0, adapter_digest=ADAPTER), store)

    unpriced = _columns(build_report(store, untrained_run_id=untrained, trained_run_id=trained, draws=50).sections[0])
    assert unpriced[TRAINED].cost_per_1k_usd is None and "unknown" in unpriced[TRAINED].cost_basis
    assert unpriced[VERIFIER].cost_per_1k_usd is None and "unknown" in unpriced[VERIFIER].cost_basis
    assert unpriced[MAJORITY].cost_per_1k_usd is None

    test_ids = [e.decision_id for e in dataset.examples if e.split == "test"]
    cost = VerifierCost(
        dataset_id=dataset_id, ledger="ledger.jsonl", models=("judge/model",),
        input_usd_per_mtok=0.2, output_usd_per_mtok=0.8,
        per_decision={d: DecisionCost(calls=3, failed_calls=0, prompt_tokens=3000, completion_tokens=1500,
                                      seconds=4.0, usd=0.0018) for d in test_ids[:80]},
        uncovered_decisions=20, retries=0,
    )
    cost_id = save_verifier_cost(cost, store).artifact_id
    report = build_report(
        store, untrained_run_id=untrained, trained_run_id=trained, draws=50,
        verifier_cost_id=cost_id, gpu_usd_per_hour=3.6,
    )
    columns = _columns(report.sections[0])

    assert columns[VERIFIER].cost_per_1k_usd == pytest.approx(1.8)  # $0.0018 × 1,000, over the 80 covered
    assert "80/100 decisions" in columns[VERIFIER].cost_basis
    assert columns[VERIFIER].metrics.latency_p50_seconds == pytest.approx(4.0)  # uncovered rows not averaged as 0
    assert columns[TRAINED].cost_per_1k_usd == pytest.approx(0.01)  # 0.01 s × $3.6/h × 1,000
    assert report.gpu_usd_per_hour == 3.6 and report.verifier_cost_id == cost_id
    from bandits_jev.report import compute_report_id, report_markdown

    markdown = report_markdown(report, compute_report_id(report))
    assert "trained vs the verifier:** 180.0× cheaper per decision, 400.0× faster at the median" in markdown
