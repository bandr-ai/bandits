"""End to end on CPU: a synthetic trace corpus and judge run go in through
`jev run`, the tiny real model (hf-internal-testing/tiny-random-gpt2) trains
and scores, and a full report comes out. Requires the `train` extra."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from typer.testing import CliRunner

from bandits.store import ArtifactStore, DerivedStore
from bandits.traces import TraceCorpus
from bandits.verify.nextstate import save_turn_judge_run
from bandits.verify.turns import extract_turns
from bandits_jev.cli import app
from tests.test_dataset import _run, _trace, _verdict

_MODEL = "hf-internal-testing/tiny-random-gpt2"
runner = CliRunner()


def _corpus_and_judge_run(project, *, prefix: str = "trace", count: int = 80):
    """Traces of two judged steps each: the first step's reaction is an
    error (judged failure), the second's is a pass (judged success)."""
    traces = [_trace(f"{prefix}-{i}", task=f"fix bug {i} in {prefix}") for i in range(count)]
    corpus = TraceCorpus(source="test", traces=tuple(traces))
    corpus_id = ArtifactStore(project / ".bandits").write(corpus, source_path="test").artifact_id
    verdicts = []
    for trace in traces:
        for turn in extract_turns(trace):
            if turn.observed:
                votes = (-1,) if "not found" in (turn.next_state() or "") else (1,)
                verdicts.append(_verdict(trace.trace_id, turn.index, turn.action_span_id, votes=votes))
    judge_run = _run(traces, verdicts).model_copy(update={"corpus_id": corpus_id})
    return save_turn_judge_run(judge_run, DerivedStore(project / ".bandits")).artifact_id


def _jev_run(project, judge_run_id, output, *extra):
    return runner.invoke(
        app,
        [
            "run", judge_run_id, *extra,
            "--model", _MODEL, "--revision", "main", "--seed", "3",
            "--checkpoint-dir", str(project / "checkpoints"), "--output", str(output),
            "--effective-batch", "8", "--eval-every-steps", "0", "--lora-rank", "4", "--lora-alpha", "8",
            "--gpu-usd-per-hour", "2.0", "--draws", "100", "--eval-split", "test", "--allow-test",
            "--device", "cpu", "--dtype", "float32", "--project", str(project),
        ],
    )


def test_jev_run_goes_from_a_judge_run_to_a_report_and_reuses_finished_steps(tmp_path) -> None:
    project = tmp_path / "project"
    judge_run_id = _corpus_and_judge_run(project)

    first = _jev_run(project, judge_run_id, tmp_path / "report-1")
    assert first.exit_code == 0, first.output

    report = json.loads((tmp_path / "report-1" / "report.json").read_text())
    columns = {c["name"]: c for c in report["sections"][0]["columns"]}
    for name in ("verifier (reference)", "majority (train prior)", "untrained", "trained", "trained + calibrated"):
        assert columns[name]["status"] == "run", name
        assert columns[name]["metrics"] is not None, name
    assert columns["jev"]["status"] == "not run"
    assert columns["trained"]["cost_per_1k_usd"] is not None  # GPU price given
    assert columns["trained"]["adapter_digest"]
    assert (tmp_path / "report-1" / "report.md").exists()

    second = _jev_run(project, judge_run_id, tmp_path / "report-2")
    assert second.exit_code == 0, second.output
    assert "training: reusing" in second.output
    assert second.output.count("reusing") >= 4  # untrained, training, trained on calibration and on test
    assert (tmp_path / "report-1" / "report.json").read_bytes() == (tmp_path / "report-2" / "report.json").read_bytes()


def test_jev_run_refuses_to_score_test_without_allow_test(tmp_path) -> None:
    project = tmp_path / "project"
    judge_run_id = _corpus_and_judge_run(project)
    result = runner.invoke(
        app,
        ["run", judge_run_id, "--model", _MODEL, "--revision", "main", "--seed", "1", "--eval-split", "test",
         "--checkpoint-dir", str(tmp_path / "c"), "--output", str(tmp_path / "o"), "--project", str(project)],
    )

    assert result.exit_code == 1
    assert "--allow-test" in result.output


def test_a_held_out_source_gets_its_own_report_section(tmp_path) -> None:
    project = tmp_path / "project"
    judge_run_id = _corpus_and_judge_run(project)
    held_out_run_id = _corpus_and_judge_run(project, prefix="other-agent", count=15)

    result = _jev_run(project, judge_run_id, tmp_path / "report", "--held-out", held_out_run_id)
    assert result.exit_code == 0, result.output

    report = json.loads((tmp_path / "report" / "report.json").read_text())
    main, held_out = report["sections"]
    assert held_out["name"].startswith("external: ")
    assert held_out["items"] == 30  # 15 traces x 2 judged steps, all in test
    columns = {c["name"]: c for c in held_out["columns"]}
    assert columns["trained"]["status"] == "run" and columns["untrained"]["status"] == "run"
    assert columns["trained + calibrated"]["temperature"] is not None  # the main run's temperature


def test_a_held_out_dataset_that_is_also_the_training_dataset_is_refused(tmp_path) -> None:
    from bandits_jev.dataset import build_decision_dataset_from_corpus, save_decision_dataset
    from bandits_jev.pipeline import run_pipeline
    from bandits_jev.trainer import build_training_config

    project = tmp_path / "project"
    store = DerivedStore(project / ".bandits")
    traces = [_trace(f"t{i}") for i in range(40)]
    dataset = build_decision_dataset_from_corpus(
        traces,
        _run(traces, [_verdict(t.trace_id, x.index, x.action_span_id, votes=(1,))
                      for t in traces for x in extract_turns(t) if x.observed]),
        "judge-run-1",
    )
    dataset_id = save_decision_dataset(dataset, store).artifact_id

    def never(*_args):
        raise AssertionError("no model may load before the leak check")

    config = build_training_config(
        base_model_id=_MODEL, base_revision="main", dataset_id=dataset_id, seed=1, eval_every_steps=0
    )
    with pytest.raises(ValueError, match="must not be the training dataset"):
        run_pipeline(
            store, dataset_id, config=config, checkpoint_dir=str(tmp_path / "c"), output=tmp_path / "o",
            make_predictor=never, make_trainable=never, held_out_dataset_ids=(dataset_id,),
            allow_test=True,
        )


def test_a_dev_run_never_opens_the_test_split(tmp_path) -> None:
    project = tmp_path / "project"
    judge_run_id = _corpus_and_judge_run(project)
    result = runner.invoke(
        app,
        [
            "run", judge_run_id, "--model", _MODEL, "--revision", "main", "--seed", "3",
            "--checkpoint-dir", str(project / "checkpoints"), "--output", str(tmp_path / "dev-report"),
            "--effective-batch", "8", "--eval-every-steps", "0", "--lora-rank", "4", "--lora-alpha", "8",
            "--draws", "50", "--eval-split", "dev", "--device", "cpu", "--dtype", "float32",
            "--project", str(project),
        ],
    )
    assert result.exit_code == 0, result.output

    report = json.loads((tmp_path / "dev-report" / "report.json").read_text())
    assert report["sections"][0]["split"] == "dev"
    assert report["test_usage"] == []
    store = DerivedStore(project / ".bandits")
    from bandits_jev.scorer import load_scorer_run

    splits = {load_scorer_run(e.artifact_id, store).split for e in store.list(kind="decision_scorer_run")}
    assert "test" not in splits and "dev" in splits
    assert "optimistic" in (tmp_path / "dev-report" / "report.md").read_text()


def test_a_cpu_run_is_never_reused_for_a_gpu_request(tmp_path) -> None:
    from bandits_jev.pipeline import _find_scorer_run
    from bandits_jev.prompt import template_digest
    from bandits_jev.scorer import ScorerRun, save_scorer_run

    store = DerivedStore(tmp_path / ".bandits")
    run = ScorerRun(
        model_id=_MODEL, revision="main", dtype="float32", device="cpu", dataset_id="ds", split="test",
        prompt_version=1, template_digest=template_digest(), mode="single_order", max_prompt_tokens=8000,
        results=(), rejections=(),
    )
    run_id = save_scorer_run(run, store).artifact_id
    common = dict(
        dataset_id="ds", split="test", model_id=_MODEL, revision="main", adapter=None, trained_on=None,
        mode="single_order", max_prompt_tokens=8000,
    )

    assert _find_scorer_run(store, **common, device="cpu", dtype="float32") == run_id
    assert _find_scorer_run(store, **common, device="cuda", dtype="bfloat16") is None


def test_a_held_out_source_that_rejudged_training_traces_is_refused(tmp_path) -> None:
    from bandits_jev.dataset import (
        as_test_only,
        build_decision_dataset_from_corpus,
        save_decision_dataset,
    )
    from bandits_jev.pipeline import run_pipeline
    from bandits_jev.trainer import build_training_config

    store = DerivedStore(tmp_path / ".bandits")
    traces = [_trace(f"t{i}") for i in range(40)]
    verdicts = [_verdict(t.trace_id, x.index, x.action_span_id, votes=(1,))
                for t in traces for x in extract_turns(t) if x.observed]
    dataset = build_decision_dataset_from_corpus(traces, _run(traces, verdicts), "judge-run-1")
    dataset_id = save_decision_dataset(dataset, store).artifact_id
    # The same traces judged again: new judge run id, so every decision id is new.
    rejudged = as_test_only(build_decision_dataset_from_corpus(traces, _run(traces, verdicts), "judge-run-2"))
    rejudged_id = save_decision_dataset(rejudged, store).artifact_id
    assert not {e.decision_id for e in rejudged.examples} & {e.decision_id for e in dataset.examples}

    def never(*_args):
        raise AssertionError("no model may load before the leak check")

    config = build_training_config(base_model_id=_MODEL, base_revision="main", dataset_id=dataset_id, seed=1,
                                   eval_every_steps=0)
    with pytest.raises(ValueError, match="traces also used in training"):
        run_pipeline(store, dataset_id, config=config, checkpoint_dir=str(tmp_path / "c"), output=tmp_path / "o",
                     make_predictor=never, make_trainable=never, held_out_dataset_ids=(rejudged_id,),
                     allow_test=True)


def test_task_set_is_refused_with_several_sources(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["run", "turn-judge-a", "turn-judge-b", "--task-set", "taskset-1", "--model", _MODEL, "--revision", "main",
         "--seed", "1", "--eval-split", "dev", "--checkpoint-dir", str(tmp_path / "c"), "--output",
         str(tmp_path / "o"), "--project", str(tmp_path)],
    )

    assert result.exit_code == 1 and "task set belongs to one corpus" in result.output.replace("--task-set", "task set")
