from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from typer.testing import CliRunner

from bandits.analyze import (
    load_analysis,
    load_task_set,
    save_task_set,
)
from bandits.cli import app
from bandits.export import direct_sft
from bandits.ingest.otlp import load_otlp
from bandits.store import DerivedStore, compute_artifact_id
from tests.tasksets import task_set_by_first_word

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI codes before substring checks: CI renders --help output
    with color (unlike a local run), which can split a plain-text match
    across color codes and fail even though the text is visibly present."""
    return _ANSI_ESCAPE.sub("", text)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "traces.otlp.jsonl"
SUPPORT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "traces.support.otlp.jsonl"
)


def _mined(tmp_path: Path) -> str:
    """Ingest and analyze the support fixture, then save a rule-built task set.

    Built by the first-word fixture rule rather than by a miner: these tests are
    about the commands downstream of grouping, and pinning them to whatever
    discovers families would break them all again on the next change to it.
    """
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analyzed = runner.invoke(app, ["analyze", corpus_id, "--project", str(tmp_path)])
    assert analyzed.exit_code == 0, analyzed.stdout
    analysis_id = analyzed.stdout.split()[1]

    store = DerivedStore(tmp_path / ".bandits")
    analysis = load_analysis(analysis_id, store)
    task_set = task_set_by_first_word(analysis, analysis_id, budget=10)
    return save_task_set(task_set, store).artifact_id


def test_ingest_prints_artifact_summary(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "artifact_id: corpus-" in plain(result.stdout)
    assert "read:     2 traces" in plain(result.stdout)


def test_ingest_unknown_source_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "nope", "--project", str(tmp_path)]
    )
    assert result.exit_code == 1


def test_ingest_records_declared_control_markers(tmp_path) -> None:
    """A benchmark's own scaffolding, declared once at ingest, must survive
    onto the stored corpus so every later RLM command reads it automatically
    rather than each one needing to know the source is tau2."""
    from bandits.store import ArtifactStore

    result = runner.invoke(
        app,
        [
            "ingest",
            str(FIXTURE),
            "--source",
            "otlp",
            "--project",
            str(tmp_path),
            "--control-marker",
            "###TRANSFER###",
            "--control-marker",
            "###STOP###",
        ],
    )
    assert result.exit_code == 0
    artifact_id = next(
        line.split("artifact_id: ", 1)[1]
        for line in plain(result.stdout).splitlines()
        if line.startswith("artifact_id:")
    )
    corpus = ArtifactStore(tmp_path / ".bandits").read(artifact_id)
    assert corpus.control_markers == ("###TRANSFER###", "###STOP###")


def test_ingest_without_control_marker_leaves_it_empty(tmp_path) -> None:
    from bandits.store import ArtifactStore

    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    artifact_id = next(
        line.split("artifact_id: ", 1)[1]
        for line in plain(result.stdout).splitlines()
        if line.startswith("artifact_id:")
    )
    corpus = ArtifactStore(tmp_path / ".bandits").read(artifact_id)
    assert corpus.control_markers == ()


def test_mine_rlm_exposes_the_per_call_token_ceiling() -> None:
    result = runner.invoke(app, ["mine-rlm", "--help"])

    assert result.exit_code == 0, result.stdout
    assert "--max-tokens" in plain(result.stdout)


def test_mine_rlm_then_materialize_rlm_taskset_through_the_real_cli(tmp_path) -> None:
    """Exercises the exact commands evaluate_bandits.py's orchestration runs as
    subprocesses: `mine-rlm --model ... --max-usd ...` then
    `materialize-rlm-taskset`. Only the model call itself is mocked — argument
    parsing, corpus loading, artifact writing, and materialization all run for
    real, so a rename or signature change to either command breaks this test
    instead of only breaking at run time months later."""
    from datetime import UTC, datetime

    from bandits.analyze import save_analysis
    from bandits.analyze.analysis import analyze_corpus
    from bandits.store import ArtifactStore, DerivedStore
    from bandits.traces import Span, SpanKind, Trace, TraceCorpus, UserTurn

    def _trace(trace_id: str, instruction: str) -> Trace:
        moment = datetime(2024, 1, 1, tzinfo=UTC)
        return Trace(
            trace_id=trace_id,
            source="chat-json",
            source_digest="0" * 64,
            task=instruction,
            user_turns=(UserTurn(text=instruction),),
            spans=(
                Span(
                    span_id=f"{trace_id}:span-0",
                    kind=SpanKind.MODEL,
                    name="model",
                    started_at=moment,
                    ended_at=moment,
                ),
            ),
        )

    corpus = TraceCorpus(
        source="chat-json",
        traces=(
            _trace("t1", "Refund order 7741"),
            _trace("t2", "Cancel order 8820"),
            _trace("t3", "Change the shipping address"),
        ),
    )
    artifact_store = ArtifactStore(tmp_path / ".bandits")
    derived_store = DerivedStore(tmp_path / ".bandits")
    artifact_store.write(corpus, source_path="synthetic")
    analysis_envelope = save_analysis(analyze_corpus(corpus), derived_store)

    def fake_predictor(*, model, view, **_settings):
        def predict(*, chunk, taxonomy, question):
            trace_ids = [row["trace_id"] for row in json.loads(chunk)]
            return SimpleNamespace(
                contracts=[
                    {
                        "contract_id": "c1",
                        "name": "Handle an order",
                        "definition": "resolve a request about an order",
                        "required_outcome_shape": ["the request is resolved"],
                    }
                ],
                operations=[],
                assignments={trace_id: "c1" for trace_id in trace_ids},
                ambiguous_trace_ids=[],
                uncovered_trace_ids=[],
            )

        return predict

    with mock.patch("bandits.cli.build_rlm_predictor", fake_predictor):
        mined = runner.invoke(
            app,
            [
                "mine-rlm",
                analysis_envelope.artifact_id,
                "--model",
                "test-model",
                "--max-usd",
                "5.0",
                "--project",
                str(tmp_path),
            ],
        )
    assert mined.exit_code == 0, mined.stdout
    run_id = next(
        line.split("draft_id:", 1)[1].strip()
        for line in plain(mined.stdout).splitlines()
        if line.startswith("draft_id:")
    )

    materialized = runner.invoke(
        app, ["materialize-rlm-taskset", run_id, "--project", str(tmp_path)]
    )
    assert materialized.exit_code == 0, materialized.stdout

    task_set_id = next(
        line.split("taskset_id:", 1)[1].strip()
        for line in plain(materialized.stdout).splitlines()
        if line.startswith("taskset_id:")
    )
    task_set = load_task_set(task_set_id, derived_store)
    assert task_set.families


def test_rlm_corpus_forwards_control_markers_from_the_stored_artifact(tmp_path) -> None:
    """Every RLM mining/audit/assignment command builds its corpus through
    this one function — a marker declared at ingest must reach the miner
    without each of those four commands having to know to ask for it."""
    from datetime import UTC, datetime

    from bandits.analyze import save_analysis
    from bandits.analyze.analysis import analyze_corpus
    from bandits.cli import _rlm_corpus
    from bandits.store import ArtifactStore, DerivedStore
    from bandits.traces import Span, SpanKind, Trace, TraceCorpus, UserTurn

    moment = datetime(2024, 1, 1, tzinfo=UTC)
    trace = Trace(
        trace_id="t1",
        source="chat-json",
        source_digest="0" * 64,
        task="please transfer me ###TRANSFER###",
        user_turns=(UserTurn(text="please transfer me ###TRANSFER###"),),
        spans=(
            Span(
                span_id="t1:span-0",
                kind=SpanKind.MODEL,
                name="model",
                started_at=moment,
                ended_at=moment,
            ),
        ),
    )
    corpus = TraceCorpus(source="chat-json", traces=(trace,), control_markers=("###TRANSFER###",))
    store = ArtifactStore(tmp_path / ".bandits")
    envelope = store.write(corpus, source_path="synthetic")

    analysis = analyze_corpus(corpus).replace(corpus_id=envelope.artifact_id)
    analysis_envelope = save_analysis(analysis, DerivedStore(tmp_path / ".bandits"))

    _, rlm_corpus, _ = _rlm_corpus(analysis_envelope.artifact_id, tmp_path, "user-messages")
    view = rlm_corpus.get_user_messages(trace.trace_id)
    assert "###TRANSFER###" not in " ".join(view.messages)
    assert "###TRANSFER###" in view.withheld_fields


def test_list_shows_ingested_artifact(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])

    result = runner.invoke(app, ["list", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "otlp" in plain(result.stdout)


def test_show_lists_traces_then_one_traces_spans(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    overview = runner.invoke(app, ["show", artifact_id, "--project", str(tmp_path)])
    assert overview.exit_code == 0
    assert "trace-1" in plain(overview.stdout)
    assert "trace-2" in plain(overview.stdout)

    detail = runner.invoke(
        app, ["show", artifact_id, "--trace", "trace-1", "--project", str(tmp_path)]
    )
    assert detail.exit_code == 0
    assert "lookup_order" in plain(detail.stdout)


CODING_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "traces.coding.otlp.jsonl"
)


def test_analyze_reports_tasks_and_never_hides_limitations(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "analysis_id: analysis-" in plain(result.stdout)
    assert "task-trace-1" in plain(result.stdout)
    assert "limitation:" in plain(result.stdout)


def test_analyze_reads_a_coding_corpus_too(tmp_path) -> None:
    runner.invoke(
        app, ["ingest", str(CODING_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    artifact_id = compute_artifact_id(load_otlp(CODING_FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "task-code-1" in plain(result.stdout)


def test_analyze_unknown_artifact_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", "corpus-nope", "--project", str(tmp_path)])
    assert result.exit_code == 1


_BLANK_ANSWERS = "\n" * 60
"""Enough blank answers to walk any draft's interview to completion."""

SUPPORT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "traces.support.otlp.jsonl"
)


def _audit_predictor(**fields):
    """Stand in for the RLM. No model, no sandbox, no credentials in CI."""

    def build(*, model, **_):
        def predict(*, members, question):
            return SimpleNamespace(**fields)

        return predict

    return build



def test_mine_stays_quiet_when_grouping_worked(tmp_path) -> None:
    assert (
        "contain one trace"
        not in runner.invoke(app, ["families", _mined(tmp_path), "--project", str(tmp_path)]).stdout
    )


def test_families_shows_one_family_in_full(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = load_task_set(task_set_id, store).families[0]

    result = runner.invoke(
        # The table truncates ids for width, so the id comes from the artifact.
        app,
        ["families", task_set_id, "--family", family.family_id, "--project", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert family.descriptor in plain(result.stdout)
    assert family.medoid_trace_id in plain(result.stdout)
    assert "held_out" in plain(result.stdout)


def test_families_rejects_an_unknown_family(tmp_path) -> None:
    task_set_id = _mined(tmp_path)

    result = runner.invoke(
        app, ["families", task_set_id, "--family", "family-nope", "--project", str(tmp_path)]
    )

    assert result.exit_code == 1


def test_build_sft_selects_traces_and_writes_three_review_buckets(tmp_path, monkeypatch) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))
    reply = (
        '{"outcome":"success","task_clarity":5,"demonstrated_success":5,'
        '"trajectory_quality":5,"self_contained":4,"recommendation":"accept",'
        '"rationale":"clear successful run","concerns":[]}'
    )
    monkeypatch.setattr(
        direct_sft,
        "model_complete",
        lambda model, prompt, temperature: reply,
    )
    output = tmp_path / "direct-dataset"

    result = runner.invoke(
        app,
        [
            "build-sft",
            artifact_id,
            "--trace",
            "trace-1",
            "--samples",
            "1",
            "--output",
            str(output),
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "reviewed:   1" in plain(result.stdout)
    assert (output / "sft.jsonl").exists()
    assert (output / "review.jsonl").exists()
    assert (output / "rejected.jsonl").exists()
    assert (output / "selection-report.json").exists()


def test_list_shows_problems_and_warns_on_mixed_derivation_versions(tmp_path) -> None:
    from datetime import UTC, datetime

    from bandits.store import ArtifactEnvelope, ArtifactStore
    from bandits.traces import TraceCorpus, WorkflowDeclaration

    store = ArtifactStore(tmp_path / ".bandits")
    for version in (1, 2):
        corpus = TraceCorpus(
            source="otlp-std", traces=(), workflow=WorkflowDeclaration(derivation_version=version)
        )
        envelope = store.write(corpus, source_path="synthetic", problem_count=version)
        assert envelope.derivation_version == version
    store.write(TraceCorpus(source="otlp", traces=()), source_path="synthetic")
    # An envelope written before these fields existed.
    old = tmp_path / ".bandits" / "artifacts" / "corpus-old"
    old.mkdir()
    (old / "envelope.json").write_text(
        ArtifactEnvelope(
            schema_version=1,
            artifact_id="corpus-old",
            created_at=datetime.now(UTC).isoformat(),
            source_path="x",
            source="langfuse",
            trace_count=0,
            span_count=0,
            issue_count=9,
        ).model_dump_json(exclude={"derivation_version", "problem_count"})
    )

    result = runner.invoke(app, ["list", "--project", str(tmp_path)])
    out = plain(result.stdout)
    assert result.exit_code == 0
    assert "problems" in out and "issues" not in out
    assert "different versions (1, 2)" in " ".join(out.split())
    row = next(line for line in out.splitlines() if "corpus-old" in line)
    assert "?" in row and " - " in row  # derivation unknown, problems not recorded


def test_list_is_quiet_when_derivation_versions_agree(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    result = runner.invoke(app, ["list", "--project", str(tmp_path)])
    assert "different versions" not in plain(result.stdout)


LANGFUSE_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures/upstream/langfuse/agno-2025-06-11.trace.json"
)


def test_ingest_prints_record_accounting_and_saves_the_report(tmp_path) -> None:
    from bandits.store import ArtifactStore

    result = runner.invoke(
        app, ["ingest", str(LANGFUSE_FIXTURE), "--source", "langfuse", "--project", str(tmp_path)]
    )
    out = plain(result.stdout)
    assert result.exit_code == 0, out
    assert re.search(r"records:  6 seen → .* · 0 dropped \(unreadable items: 0\)", out)
    artifact_id = re.search(r"artifact_id: (\S+)", out).group(1)
    report = ArtifactStore(tmp_path / ".bandits").read_report(artifact_id)
    assert report["spans_seen"] == 6 and report["dropped"] == 0


def test_other_sources_print_no_accounting(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    assert "records:" not in plain(result.stdout)


def test_accounting_that_does_not_add_up_saves_nothing(tmp_path, monkeypatch) -> None:
    import bandits.cli

    real = bandits.cli.iter_corpus

    def broken(*args, report, **kwargs):
        yield from real(*args, report=report, **kwargs)
        report.accounting_errors.append("6 span(s) seen but 5 accounted for")

    monkeypatch.setattr(bandits.cli, "iter_corpus", broken)
    result = runner.invoke(
        app, ["ingest", str(LANGFUSE_FIXTURE), "--source", "langfuse", "--project", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "record accounting does not add up" in plain(result.stdout)
    assert "nothing was saved" in plain(result.stdout)
    assert not (tmp_path / ".bandits").exists()


def _full_trajectory_project(tmp_path: Path) -> str:
    from bandits.analyze import analyze_corpus, save_analysis
    from bandits.store import ArtifactStore
    from bandits.traces import TraceCorpus
    from tests.analyze.rlm_test import _workflow_trace

    corpus = TraceCorpus(
        source="otlp", traces=(_workflow_trace("w1"), _workflow_trace("w2"), _workflow_trace("w3"))
    )
    ArtifactStore(tmp_path / ".bandits").write(corpus, source_path="synthetic")
    return save_analysis(analyze_corpus(corpus), DerivedStore(tmp_path / ".bandits")).artifact_id


def test_full_trajectory_mining_through_the_cli_selects_accounts_and_names_the_project(
    tmp_path,
):
    from tests.analyze.rlm_test import _account, _family_predict

    analysis_id = _full_trajectory_project(tmp_path)
    account_runs: list[str] = []

    def fake_account_predictor(*, catalog, **_settings):
        def predict(*, run_index, question=""):
            run_id = json.loads(run_index)["run_id"]
            account_runs.append(run_id)
            catalog.get_evidence(run_id, "clue1.first_model_prompt.json", 0, 40)
            return SimpleNamespace(account=_account(run_id))

        predict.completion = {"mode": "submit", "iterations_used": 2, "iterations_to_submit": 2}
        return predict

    def fake_family_predictor(**_settings):
        return _family_predict

    with (
        mock.patch("bandits.cli.build_rlm_predictor", fake_family_predictor),
        mock.patch("bandits.analyze.rlm_mine.build_account_predictor", fake_account_predictor),
    ):
        refused = runner.invoke(
            app,
            [
                "mine-rlm",
                analysis_id,
                "--view",
                "full-trajectory",
                "--trace-id",
                "w1",
                "--trace-id",
                "nope",
                "--project",
                str(tmp_path),
            ],
        )
        assert refused.exit_code == 1 and "nope" in plain(refused.stdout)
        assert account_runs == [], "nothing was dispatched for an invalid selection"

        mined = runner.invoke(
            app,
            [
                "mine-rlm",
                analysis_id,
                "--view",
                "full-trajectory",
                "--trace-id",
                "w3",
                "--trace-id",
                "w1",
                "--root-max-iterations",
                "8",
                "--max-subcalls",
                "4",
                "--max-llm-calls",
                "24",
                "--max-output-chars",
                "2000",
                "--subcall-workers",
                "1",
                "--provider-retries",
                "0",
                "--max-attempts",
                "1",
                "--contract-repairs",
                "0",
                "--temperature",
                "1.0",
                "--top-p",
                "0.95",
                "--model",
                "openai/test-model",
                "--project",
                str(tmp_path),
            ],
        )
    out = plain(mined.stdout)
    assert mined.exit_code == 0, out
    assert sorted(account_runs) == ["w1", "w3"]
    project = str(tmp_path.resolve())
    session_id = next(line.split()[1] for line in out.splitlines() if line.startswith("session:"))
    flat = " ".join(out.split())  # rich wraps long lines
    assert f"bandits rlm-session {session_id} --project {project} --watch" in flat
    assert "bandits rlm-families rlm-clustering-run-" in flat
    assert flat.split("review the families:")[1].split("--project ")[1].startswith(project)
    assert "account_complete 2" in flat and "source_accessed 2" in flat
    assert re.search(r"accounts:\s+rlm-account-set-", out)

    shown = runner.invoke(
        app, ["rlm-session", session_id, "--accounts", "--project", str(tmp_path)]
    )
    shown_out = plain(shown.stdout)
    assert shown.exit_code == 0, shown_out
    shown_flat = " ".join(shown_out.split())
    assert "selection: w3, w1" in shown_flat
    assert "intent (inferred)" in shown_flat and "submit" in shown_flat

    evidence = runner.invoke(
        app,
        [
            "rlm-evidence",
            analysis_id,
            "--trace-id",
            "w1",
            "--ref",
            "clue1.first_model_prompt.suffix",
            "--project",
            str(tmp_path),
        ],
    )
    assert evidence.exit_code == 0, evidence.stdout
    assert json.loads(evidence.stdout)["origin"] == "clue.first_model_prompt.suffix"

    with (
        mock.patch("bandits.cli.build_rlm_predictor", fake_family_predictor),
        mock.patch("bandits.analyze.rlm_mine.build_account_predictor", fake_account_predictor),
    ):
        mismatch = runner.invoke(
            app,
            [
                "mine-rlm",
                analysis_id,
                "--view",
                "full-trajectory",
                "--trace-id",
                "w3",
                "--trace-id",
                "w1",
                "--temperature",
                "0.0",
                "--model",
                "openai/test-model",
                "--resume",
                session_id,
                "--project",
                str(tmp_path),
            ],
        )
    assert mismatch.exit_code == 1
    assert "different generation or invocation settings" in " ".join(plain(mismatch.stdout).split())


def test_reasoning_effort_accepts_an_integer_budget_and_refuses_a_non_positive_one(tmp_path):
    from bandits.analyze.rlm_session import SessionStore
    from tests.analyze.rlm_test import _family_predict

    analysis_id = _full_trajectory_project(tmp_path)

    def fake_account_predictor(**_settings):
        def predict(*, run_index, question=""):
            raise RuntimeError("not needed")

        return predict

    base = ["mine-rlm", analysis_id, "--view", "full-trajectory", "--trace-id", "w1"]
    with (
        mock.patch("bandits.cli.build_rlm_predictor", lambda **_: _family_predict),
        mock.patch("bandits.analyze.rlm_mine.build_account_predictor", fake_account_predictor),
    ):
        refused = runner.invoke(app, [*base, "--reasoning-effort", "0", "--project", str(tmp_path)])
        assert refused.exit_code == 1 and "must be positive" in plain(refused.stdout)
        mined = runner.invoke(
            app,
            [
                *base,
                "--reasoning-effort",
                "1024",
                "--max-attempts",
                "1",
                "--project",
                str(tmp_path),
            ],
        )
    assert mined.exit_code == 0, plain(mined.stdout)
    (state,) = SessionStore(tmp_path / ".bandits").list()
    assert state.settings["reasoning_effort"] == 1024


def test_an_account_mode_run_flows_through_audit_and_task_set_materialization(tmp_path):
    """Downstream commands read an account-mode run like any other clustering run."""
    from tests.analyze.rlm_test import _account, _family_predict

    analysis_id = _full_trajectory_project(tmp_path)

    def fake_account_predictor(**_settings):
        def predict(*, run_index, question=""):
            return SimpleNamespace(account=_account(json.loads(run_index)["run_id"]))

        predict.completion = {"mode": "submit", "iterations_used": 1, "iterations_to_submit": 1}
        return predict

    with (
        mock.patch("bandits.cli.build_rlm_predictor", lambda **_: _family_predict),
        mock.patch("bandits.analyze.rlm_mine.build_account_predictor", fake_account_predictor),
    ):
        mined = runner.invoke(
            app,
            ["mine-rlm", analysis_id, "--view", "full-trajectory", "--project", str(tmp_path)],
        )
    assert mined.exit_code == 0, plain(mined.stdout)
    run_id = re.search(r"draft_id:\s+(\S+)", plain(mined.stdout)).group(1)
    session_id = re.search(r"session:\s+(\S+)", plain(mined.stdout)).group(1)

    watched = runner.invoke(app, ["rlm-session", session_id, "--watch", "--project", str(tmp_path)])
    watched_out = " ".join(plain(watched.stdout).split())
    assert watched.exit_code == 0, watched_out
    assert f"rlm-families {run_id} --project {tmp_path.resolve()}" in watched_out

    def fake_audit(*, model, view, max_tokens):
        def predict(*, contract, sibling_contracts, members, outsiders, question=""):
            return SimpleNamespace(
                recommendation="keep",
                least_compatible_pair=[],
                strongest_outsider_trace_id="",
                merge_with_contract_id="",
                topical_only=False,
                rationale="members request the same diagnosis",
            )

        return predict

    with mock.patch("bandits.cli.build_taxonomy_audit_predictor", fake_audit):
        audited = runner.invoke(app, ["audit-rlm", run_id, "--project", str(tmp_path)])
    assert audited.exit_code == 0, plain(audited.stdout)
    assert "keep" in plain(audited.stdout)

    materialized = runner.invoke(
        app, ["materialize-rlm-taskset", run_id, "--project", str(tmp_path)]
    )
    assert materialized.exit_code == 0, plain(materialized.stdout)
    task_set_id = re.search(r"taskset_id:\s+(\S+)", plain(materialized.stdout)).group(1)
    task_set = load_task_set(task_set_id, DerivedStore(tmp_path / ".bandits"))
    members = {t for family in task_set.families for t in family.trace_ids}
    assert members == {"w1", "w2", "w3"}
