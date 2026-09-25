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


def _corpus_and_judge_run(project):
    """80 traces of two judged steps each: the first step's reaction is an
    error (judged failure), the second's is a pass (judged success)."""
    traces = [_trace(f"trace-{i}", task=f"fix bug {i}") for i in range(80)]
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


def _jev_run(project, judge_run_id, output):
    return runner.invoke(
        app,
        [
            "run", judge_run_id,
            "--model", _MODEL, "--revision", "main", "--seed", "3",
            "--checkpoint-dir", str(project / "checkpoints"), "--output", str(output),
            "--effective-batch", "8", "--eval-every-steps", "0", "--lora-rank", "4", "--lora-alpha", "8",
            "--gpu-usd-per-hour", "2.0", "--draws", "100", "--allow-test",
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
        ["run", judge_run_id, "--model", _MODEL, "--revision", "main", "--seed", "1",
         "--checkpoint-dir", str(tmp_path / "c"), "--output", str(tmp_path / "o"), "--project", str(project)],
    )

    assert result.exit_code == 1
    assert "--allow-test" in result.output
