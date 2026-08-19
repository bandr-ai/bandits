"""Tests for the ``bandits`` command line.

Three things are load-bearing here and each has a test: artifacts round-trip
losslessly, ``--source`` is required and never guessed, and the fidelity gate
exits nonzero when it rejects. The last one is the difference between a report
and a CI gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import FidelityBundle, TaskBundle, VerifierBundle, app
from bandits.contracts import StateSchema, ToolSurface, TraceCorpus

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
TRACES = str(FIXTURES / "traces.otlp.jsonl")
TOOLS = str(FIXTURES / "tools.json")

runner = CliRunner()


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One full ``bandits run`` over the golden corpus, shared by the tests below."""
    out = tmp_path_factory.mktemp("run")
    result = runner.invoke(
        app, ["run", TRACES, "--source", "otlp", "--tools", TOOLS, "-o", str(out)]
    )
    return result, out


# ---------------------------------------------------------------- run, end to end


def test_run_writes_every_artifact(ran):
    result, out = ran
    assert result.exception is None or isinstance(result.exception, SystemExit)
    for name in ("corpus", "surface", "schema", "tasks", "verifiers", "fidelity"):
        assert (out / f"{name}.json").exists(), f"{name}.json was not written"


def test_run_exits_nonzero_because_the_golden_corpus_is_rejected(ran):
    """The gate is a CI gate. 39% must fail the build, not print a table and pass."""
    result, _ = ran
    assert result.exit_code == 1
    assert "REJECTED" in result.output


def test_run_prints_the_fidelity_table_last(ran):
    result, _ = ran
    tail = result.output[result.output.index("fidelity ·") :]
    assert "overall" in tail
    assert "REJECTED" in tail
    for tool in ("get_order", "refund_order", "send_email"):
        assert tool in tail


def test_run_surfaces_counts_issues_and_warnings(ran):
    result, _ = ran
    assert "7 traces" in result.output
    assert "23 invocation points" in result.output
    assert "7 mined, 0 skipped" in result.output
    # MiningResult.warned must not be swallowed: a silently dropped or silently
    # degraded trace is exactly what this project exists to avoid.
    assert "warning" in result.output
    assert "instruction references" in result.output
    # Verifier synthesis refuses the two negatively-labeled traces, loudly.
    assert "5 verifier(s), 2 refused" in result.output
    assert "no verifier task-ep-double-refund" in result.output
    # An entity whose structure could not be determined is labeled, not invented.
    assert "static" in result.output


def test_run_reports_the_declared_only_probing_candidate(ran):
    result, _ = ran
    assert "escalate_to_human" in result.output
    assert "declared-only" in result.output


# ---------------------------------------------------------------- round-tripping


def test_artifacts_round_trip_losslessly(ran):
    _, out = ran
    corpus = TraceCorpus.model_validate_json((out / "corpus.json").read_text())
    surface = ToolSurface.model_validate_json((out / "surface.json").read_text())
    schema = StateSchema.model_validate_json((out / "schema.json").read_text())
    tasks = TaskBundle.model_validate_json((out / "tasks.json").read_text())
    verifiers = VerifierBundle.model_validate_json((out / "verifiers.json").read_text())
    fidelity = FidelityBundle.model_validate_json((out / "fidelity.json").read_text())

    for model in (corpus, surface, schema, tasks, verifiers, fidelity):
        revived = type(model).model_validate_json(model.model_dump_json())
        assert revived == model, f"{type(model).__name__} did not survive a round trip"

    assert len(corpus.traces) == 7
    assert len(tasks.tasks) == 7
    assert len(verifiers.verifiers) == 5
    assert fidelity.overall.accepted is False
    assert len(fidelity.per_trace) == 7


def test_fidelity_json_carries_the_actual_numbers(ran):
    _, out = ran
    bundle = FidelityBundle.model_validate_json((out / "fidelity.json").read_text())
    rates = {t.tool: (t.matched, t.replayed) for t in bundle.overall.per_tool}
    assert rates["get_order"] == (7, 8)
    assert rates["send_email"] == (3, 3)
    assert sum(t.replayed for t in bundle.overall.per_tool) == 23


# ---------------------------------------------------------------- stage by stage


def test_stages_chain_through_files(tmp_path):
    corpus = tmp_path / "corpus.json"
    surface = tmp_path / "surface.json"
    schema = tmp_path / "schema.json"
    tasks = tmp_path / "tasks.json"
    verifiers = tmp_path / "verifiers.json"
    fidelity = tmp_path / "fidelity.json"

    assert runner.invoke(
        app, ["ingest", TRACES, "--source", "otlp", "--tools", TOOLS, "-o", str(corpus)]
    ).exit_code == 0
    assert runner.invoke(
        app, ["surface", str(corpus), "--tools", TOOLS, "-o", str(surface)]
    ).exit_code == 0
    assert runner.invoke(
        app, ["schema", str(corpus), str(surface), "-o", str(schema)]
    ).exit_code == 0
    assert runner.invoke(app, ["tasks", str(corpus), str(schema), "-o", str(tasks)]).exit_code == 0
    assert runner.invoke(
        app, ["verify", str(tasks), str(corpus), str(schema), "-o", str(verifiers)]
    ).exit_code == 0

    rejected = runner.invoke(
        app,
        ["fidelity", str(corpus), str(schema), str(tasks), "--surface", str(surface), "-o", str(fidelity)],
    )
    assert rejected.exit_code == 1, "the gate must exit nonzero when it rejects"
    assert "REJECTED" in rejected.output
    assert json.loads(fidelity.read_text())["overall"]["accepted"] is False


def test_fidelity_exit_code_tracks_the_gate(tmp_path):
    """Same corpus, thresholds dropped to zero: the gate accepts and exits zero.

    Proves the nonzero exit comes from the verdict and not from an unrelated
    failure in the command itself.
    """
    out = tmp_path / "art"
    runner.invoke(app, ["run", TRACES, "--source", "otlp", "--tools", TOOLS, "-o", str(out)])
    accepted = runner.invoke(
        app,
        [
            "fidelity",
            str(out / "corpus.json"),
            str(out / "schema.json"),
            str(out / "tasks.json"),
            "--surface",
            str(out / "surface.json"),
            "--threshold",
            "0.0",
            "--per-tool-floor",
            "0.0",
            "-o",
            str(tmp_path / "f.json"),
        ],
    )
    assert accepted.exit_code == 0
    assert "ACCEPTED" in accepted.output


def test_run_exits_zero_when_the_gate_is_satisfied(tmp_path):
    result = runner.invoke(
        app,
        [
            "run",
            TRACES,
            "--source",
            "otlp",
            "--tools",
            TOOLS,
            "-o",
            str(tmp_path / "o"),
            "--threshold",
            "0.0",
            "--per-tool-floor",
            "0.0",
        ],
    )
    assert result.exit_code == 0
    assert "ACCEPTED" in result.output


def test_fidelity_rebuilds_the_surface_when_none_is_given(tmp_path):
    out = tmp_path / "art"
    runner.invoke(app, ["run", TRACES, "--source", "otlp", "--tools", TOOLS, "-o", str(out)])
    result = runner.invoke(
        app,
        [
            "fidelity",
            str(out / "corpus.json"),
            str(out / "schema.json"),
            str(out / "tasks.json"),
            "-o",
            str(tmp_path / "f.json"),
        ],
    )
    assert result.exit_code == 1
    bundle = FidelityBundle.model_validate_json((tmp_path / "f.json").read_text())
    assert sum(t.replayed for t in bundle.overall.per_tool) == 23


def test_show_re_renders_a_saved_report(ran):
    _, out = ran
    result = runner.invoke(app, ["show", str(out / "fidelity.json")])
    assert result.exit_code == 1
    assert "REJECTED" in result.output
    detailed = runner.invoke(app, ["show", str(out / "fidelity.json"), "--per-trace"])
    assert "ep-refund-ok" in detailed.output


# ---------------------------------------------------------------- declared source


def test_source_is_required():
    result = runner.invoke(app, ["ingest", TRACES])
    assert result.exit_code != 0
    assert "source" in result.output.lower()


def test_an_unknown_source_is_refused_and_never_sniffed(tmp_path):
    result = runner.invoke(
        app, ["ingest", TRACES, "--source", "guess", "-o", str(tmp_path / "c.json")]
    )
    assert result.exit_code == 2
    assert "never sniffed" in result.output


def test_a_missing_artifact_is_a_clear_error(tmp_path):
    result = runner.invoke(app, ["surface", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "missing artifact" in result.output
