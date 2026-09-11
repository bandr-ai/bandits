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
from bandits.verify import (
    apply_decision,
    draft_verifiers,
    load_interview,
    load_verifier_draft,
    save_interview,
    save_verifier_draft,
    start_review,
)
from bandits.verify.models import CheckReview, InterviewDecision
from bandits.verify.validate import Agreement, Validation, save_validation
from tests.tasksets import task_set_by_first_word

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI codes before substring checks: CI renders --help output
    with color (unlike a local run), which can split a plain-text match
    across color codes and fail even though the text is visibly present."""
    return _ANSI_ESCAPE.sub("", text)


FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.otlp.jsonl"
SUPPORT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.support.otlp.jsonl"
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
    assert "traces:      2" in plain(result.stdout)


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

    def fake_predictor(*, model, view, max_tokens):
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
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.coding.otlp.jsonl"
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
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.support.otlp.jsonl"
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


def test_draft_verifier_writes_suggested_replay_specs(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    draft_id = result.stdout.split()[1]
    draft = load_verifier_draft(draft_id, store)
    assert draft.verifiers
    assert all(spec.status.value == "executable" for spec in draft.verifiers)
    assert all(spec.mode.value == "replay" for spec in draft.verifiers)


def test_draft_verifier_rejects_unknown_family(tmp_path) -> None:
    task_set_id = _mined(tmp_path)

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            "family-nope",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1


def test_interview_verifier_completes_a_bounded_review(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )
    drafted = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
    )
    draft_id = drafted.stdout.split()[1]

    result = runner.invoke(
        app,
        ["interview-verifier", draft_id, "--project", str(tmp_path)],
        input=_BLANK_ANSWERS,
    )

    assert result.exit_code == 0, result.stdout
    assert "status:       complete" in plain(result.stdout)
    interview_id = next(
        line.split(maxsplit=1)[1]
        for line in plain(result.stdout).splitlines()
        if line.startswith("interview_id:")
    )
    interview = load_interview(interview_id, store)
    assert interview.complete
    assert len(interview.answers) == len(interview.questions)
    # Blind spots and gaming are asked of every check; an expected value only of
    # the checks that compare against one.
    assert all(
        question.check_id is not None
        for question in interview.questions
        if question.field == "expected"
    )
    assert interview.draft.verifiers[0].status.value == "executable"


def test_draft_verifier_can_run_the_interview_inline(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--interview",
            "--project",
            str(tmp_path),
        ],
        input=_BLANK_ANSWERS,
    )

    assert result.exit_code == 0, result.stdout
    assert "verifier_draft_id:" in plain(result.stdout)
    assert "interview_id:" in plain(result.stdout)
    assert "status:       complete" in plain(result.stdout)


def _drafted_family(tmp_path, descriptor_word: str) -> tuple[str, str]:
    """Ingest, analyze, mine and draft; return (draft_id, family_id)."""
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        f for f in load_task_set(task_set_id, store).families if descriptor_word in f.descriptor
    )
    drafted = runner.invoke(
        app,
        # fmt: off
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )
    assert drafted.exit_code == 0, drafted.stdout
    return drafted.stdout.split()[1], family.family_id


def test_label_then_validate_separates_the_right_check(tmp_path) -> None:
    """addr-1 changed the address; addr-2 and addr-3 only looked the order up."""
    draft_id, _ = _drafted_family(tmp_path, "address")

    labelled = runner.invoke(
        app,
        ["label", draft_id, "--labeler", "owner", "--project", str(tmp_path)],
        input="s\nchanged it\nf\nonly looked\nf\nonly looked\n",
    )
    assert labelled.exit_code == 0, labelled.stdout
    label_set_id = labelled.stdout.split("label_set_id:")[1].split()[0]

    result = runner.invoke(
        app,
        # fmt: off
        [
            "validate-verifier",
            draft_id,
            "--labels",
            label_set_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 0, result.stdout
    assert "validation_id: validation-" in plain(result.stdout)
    # One hypothesis is right and one is wrong; the run where the wrong one
    # would have rewarded doing nothing is named.
    assert "100%" in plain(result.stdout) and "0%" in plain(result.stdout)
    assert "false_positive" in plain(result.stdout)
    assert "gamed" in plain(result.stdout)


def test_labeling_can_be_quit_early(tmp_path) -> None:
    draft_id, _ = _drafted_family(tmp_path, "address")

    result = runner.invoke(
        app,
        ["label", draft_id, "--labeler", "owner", "--project", str(tmp_path)],
        input="s\nfine\nq\n",
    )

    assert result.exit_code == 0
    assert "labels:       1" in plain(result.stdout)


def test_validate_verifier_needs_an_existing_label_set(tmp_path) -> None:
    draft_id, _ = _drafted_family(tmp_path, "address")

    result = runner.invoke(
        app,
        # fmt: off
        [
            "validate-verifier",
            draft_id,
            "--labels",
            "labels-nope",
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 1


def _reviewed_refund_verifier(tmp_path: Path) -> tuple[str, str]:
    """Build through validation, then exercise explicit acceptance through the CLI."""
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    task_set = load_task_set(task_set_id, store)
    analysis = load_analysis(task_set.analysis_id, store)
    family = next(item for item in task_set.families if "refund" in item.descriptor)
    draft = draft_verifiers(task_set, task_set_id, analysis, family.family_id, limit=8)
    spec = next(
        item
        for item in draft.verifiers
        if item.checks[0].claim == "final_state_field:refund_order.status"
        and item.checks[0].expected == "refunded"
    )
    draft_id = save_verifier_draft(draft, store).artifact_id
    validation = Validation(
        source_draft_id=draft_id,
        family_id=family.family_id,
        label_set_id="labels-cli",
        agreements=(
            Agreement(
                verifier_id=spec.verifier_id,
                split="held_out",
                labeled=1,
                agreed=1,
                disagreed=0,
                unscored=0,
                agreement=1,
            ),
        ),
        labels_used=2,
        success_labels=1,
        failure_labels=1,
    )
    validation_id = save_validation(validation, store).artifact_id

    # Promotion needs a round in which a person saw these measurements and
    # accepted every check, so the export chain starts from one.
    interview = start_review(draft, draft_id, validation_id=validation_id, round_number=2)
    for index, (verifier_id, check_id) in enumerate(interview.pending, start=1):
        interview = apply_decision(
            interview,
            CheckReview(
                review_id=f"review-{index:03d}-{check_id}",
                verifier_id=verifier_id,
                check_id=check_id,
                reply="reads the refund status; keep it",
                decision=InterviewDecision.ACCEPT,
                authoritative=True,
            ),
        )
    interview_id = save_interview(interview, store).artifact_id

    result = runner.invoke(
        app,
        [
            "review-verifier",
            draft_id,
            "--validation",
            validation_id,
            "--verifier",
            spec.verifier_id,
            "--interview",
            interview_id,
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    reviewed_id = result.stdout.split("reviewed_verifier_id:")[1].split()[0]
    return task_set_id, reviewed_id


def test_eval_and_sft_export_end_to_end_write_quarantine(tmp_path) -> None:
    task_set_id, reviewed_id = _reviewed_refund_verifier(tmp_path)

    eval_output = tmp_path / "out" / "eval.jsonl"
    evaluated = runner.invoke(
        app,
        [
            "export",
            task_set_id,
            "--format",
            "eval",
            "--verifier",
            reviewed_id,
            "--output",
            str(eval_output),
            "--project",
            str(tmp_path),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    assert eval_output.exists()
    assert eval_output.with_name("eval.unresolved.jsonl").exists()
    assert '"grader"' in eval_output.read_text()

    sft_output = tmp_path / "out" / "sft.jsonl"
    trained = runner.invoke(
        app,
        [
            "export",
            task_set_id,
            "--format",
            "sft",
            "--verifier",
            reviewed_id,
            "--output",
            str(sft_output),
            "--project",
            str(tmp_path),
        ],
    )
    assert trained.exit_code == 0, trained.stdout
    assert "rows:" in plain(trained.stdout) and "unresolved:" in plain(trained.stdout)
    assert sft_output.exists()
    quarantine = sft_output.with_name("sft.unresolved.jsonl")
    assert quarantine.exists()
    assert '"messages"' in sft_output.read_text()
    assert '"reasons"' in quarantine.read_text()


def test_export_rejects_unknown_format_before_writing(tmp_path) -> None:
    output = tmp_path / "bad.jsonl"
    result = runner.invoke(
        app,
        [
            "export",
            "taskset-nope",
            "--format",
            "preference",
            "--verifier",
            "reviewed-nope",
            "--output",
            str(output),
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert not output.exists()


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
        "fireworks_completion",
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


def _review_draft(tmp_path: Path) -> str:
    """A saved two-verifier draft for the free-text review to work over."""
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    task_set = load_task_set(task_set_id, store)
    analysis = load_analysis(task_set.analysis_id, store)
    family = next(item for item in task_set.families if "refund" in item.descriptor)
    draft = draft_verifiers(task_set, task_set_id, analysis, family.family_id, limit=8)
    return save_verifier_draft(draft, store).artifact_id


def _fake_interpreter(*payloads: str):
    queue = list(payloads)

    def predict(model: str, prompt: str, temperature: float) -> str:
        item = queue.pop(0) if queue else payloads[-1]
        if isinstance(item, Exception):
            raise item
        return item

    return predict


def _decision(decision: str, **fields) -> str:
    return json.dumps({"decision": decision, "rationale": "as the reviewer said", **fields})


def _run_review(tmp_path: Path, draft_id: str, interpreter, keys: str, *extra: str):
    with mock.patch("bandits.cli._INTERPRETER", interpreter):
        return runner.invoke(
            app,
            ["interview-review", draft_id, "--project", str(tmp_path), *extra],
            input=keys,
        )


def test_a_free_text_review_accepts_a_check(tmp_path: Path) -> None:
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nit is the system of record\ny\n" * 12,
    )
    assert result.exit_code == 0, result.output
    assert "read as: accept" in plain(result.output)
    assert "interview_id:" in plain(result.output)


def test_a_review_records_the_reply_and_the_model_response(tmp_path: Path) -> None:
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("reject")),
        "wrong signal entirely\nn\nnobody owns it\ny\n" * 12,
    )
    assert result.exit_code == 0, result.output

    store = DerivedStore(tmp_path / ".bandits")
    interview_id = [
        line.split()[-1] for line in plain(result.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(interview_id, store)
    first = interview.reviews[0]
    assert first.reply == "wrong signal entirely"
    assert first.decision.value == "reject"
    assert first.authoritative is False
    assert first.authoritative_why == "nobody owns it"
    assert first.response  # the raw model reply is kept for audit
    assert first.prompt


def test_an_overruled_interpretation_takes_the_humans_decision(tmp_path: Path) -> None:
    """The model proposes; the human decides."""
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "actually no\ny\nwhy not\nn\nr\n" * 12,
    )
    assert result.exit_code == 0, result.output
    assert "overruled" in plain(result.output)

    store = DerivedStore(tmp_path / ".bandits")
    interview_id = [
        line.split()[-1] for line in plain(result.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(interview_id, store)
    assert interview.reviews[0].decision.value == "reject"
    # What the model said is still recorded, even though it was not followed.
    assert interview.reviews[0].interpretation.decision.value == "accept"


def test_a_failed_interpretation_falls_back_to_manual_entry(tmp_path: Path) -> None:
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter("this is not json"),
        "fine\ny\nowned\na\n" * 12,
    )
    assert result.exit_code == 0, result.output
    assert "could not read that reply" in plain(result.output)

    store = DerivedStore(tmp_path / ".bandits")
    interview_id = [
        line.split()[-1] for line in plain(result.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(interview_id, store)
    assert interview.reviews[0].decision.value == "accept"
    assert interview.reviews[0].failure
    assert interview.reviews[0].interpretation is None


def test_the_review_never_promotes_past_the_draft(tmp_path: Path) -> None:
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "fine\ny\nowned\ny\n" * 12,
    )
    assert "validation is still required" in plain(result.output)

    store = DerivedStore(tmp_path / ".bandits")
    interview_id = [
        line.split()[-1] for line in plain(result.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(interview_id, store)
    for spec in interview.draft.verifiers:
        assert spec.status.value in {"executable", "suggested", "rejected"}


def test_a_second_round_reads_the_decisions_of_the_first(tmp_path: Path) -> None:
    """The chain has to be loaded, not merely named.

    `--prior` once stored the id and nothing else, so a later round rebuilt from
    the original draft with no reviews: `prior_decisions` returned nothing and
    the interpreter saw a second-round reply as a first look.
    """
    draft_id = _review_draft(tmp_path)
    first = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nsystem of record\ny\n" * 12,
    )
    assert first.exit_code == 0, first.output
    first_id = [line.split()[-1] for line in plain(first.output).splitlines() if "interview_id:" in line][
        0
    ]

    second = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "still fine\ny\nsystem of record\ny\n" * 12,
        "--prior",
        first_id,
    )

    assert second.exit_code == 0, second.output
    assert "round 2" in plain(second.output)
    assert "earlier:" in plain(second.output)

    store = DerivedStore(tmp_path / ".bandits")
    second_id = [
        line.split()[-1] for line in plain(second.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(second_id, store)
    assert interview.round_number == 2
    assert interview.prior_interview_id == first_id
    assert {review.round_number for review in interview.reviews} == {1, 2}


def test_a_second_round_scores_the_verifier_the_revision_produced(tmp_path: Path) -> None:
    """The round's own draft is what the round has to execute.

    The historical run was built from the originally loaded draft, before
    `start_review` swapped in the draft the previous round left. A revision
    mints a new verifier id, so no outcome matched the spec the reviewer was
    being shown and every second-round summary read zero passed, zero failed,
    zero unscorable — a check that looks unscored rather than one never run.
    """
    draft_id = _review_draft(tmp_path)
    first = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("revise", revised_expected="shipped")),
        "change it\ny\nsystem of record\ny\n" * 12,
    )
    assert first.exit_code == 0, first.output
    first_id = [line.split()[-1] for line in plain(first.output).splitlines() if "interview_id:" in line][
        0
    ]

    store = DerivedStore(tmp_path / ".bandits")
    revised = load_interview(first_id, store)
    assert any(
        check.expected == "shipped" for s in revised.draft.verifiers for check in s.checks
    ), "round one did not revise anything, so the regression cannot be observed"

    second = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "fine now\ny\nsystem of record\ny\n" * 12,
        "--prior",
        first_id,
    )

    assert second.exit_code == 0, second.output
    scored = [line for line in plain(second.output).splitlines() if "scored:" in line]
    assert scored, second.output
    assert any("0 passed, 0 failed, 0 unscorable" not in line for line in scored), (
        f"every second-round summary scored nothing: {scored}"
    )


def test_a_review_refuses_a_prior_interview_that_does_not_exist(tmp_path: Path) -> None:
    """An unknown chain id was accepted and silently produced an unchained round."""
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nwhy\ny\n",
        "--prior",
        "interview-nope",
    )

    assert result.exit_code == 1
    assert "no interview" in plain(result.output)


def test_a_review_refuses_a_prior_interview_of_another_draft(tmp_path: Path) -> None:
    draft_id = _review_draft(tmp_path)
    first = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nwhy\ny\n" * 12,
    )
    first_id = [line.split()[-1] for line in plain(first.output).splitlines() if "interview_id:" in line][
        0
    ]

    other_draft = _review_draft(tmp_path / "other")
    result = _run_review(
        tmp_path / "other",
        other_draft,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nwhy\ny\n",
        "--prior",
        first_id,
    )

    assert result.exit_code == 1


def test_a_manual_revise_after_an_unreadable_reply_is_applied(tmp_path: Path) -> None:
    """The fallback has to be able to carry out the decision it offers.

    A manual revise once captured only the decision enum, so the revised value
    was never asked for, the decision was refused for naming nothing, and the
    check returned to the queue — re-asking a question the reviewer had no way
    to answer.
    """
    draft_id = _review_draft(tmp_path)
    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter("not json at all"),
        # reply, authoritative, why, manual decision, rationale, value, operator
        'change it\ny\nwhy\nv\nby hand\n"shipped"\n\n' * 12,
    )

    assert result.exit_code == 0, result.output
    assert "could not read that reply" in plain(result.output)

    store = DerivedStore(tmp_path / ".bandits")
    interview_id = [
        line.split()[-1] for line in plain(result.output).splitlines() if "interview_id:" in line
    ][0]
    interview = load_interview(interview_id, store)
    first = interview.reviews[0]
    assert first.decision.value == "revise"
    assert first.interpretation is not None
    assert first.interpretation.source == "human"
    assert first.interpretation.revised_expected == "shipped"
    assert any(check.expected == "shipped" for s in interview.draft.verifiers for check in s.checks)


def test_sft_export_writes_a_composition_report_beside_its_rows(tmp_path) -> None:
    task_set_id, reviewed_id = _reviewed_refund_verifier(tmp_path)
    output = tmp_path / "out" / "sft.jsonl"

    result = runner.invoke(
        app,
        [
            "export",
            task_set_id,
            "--format",
            "sft",
            "--verifier",
            reviewed_id,
            "--output",
            str(output),
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = output.with_name("sft.composition.json")
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["schema_version"] == 1
    assert payload["offered_traces"] >= payload["selected"]["rows"]
    assert "composition:" in plain(result.stdout)


def test_an_eval_export_refuses_sampling_caps_rather_than_ignoring_them(tmp_path) -> None:
    """An ignored cap would produce an eval set that looks curated and is not."""
    task_set_id, reviewed_id = _reviewed_refund_verifier(tmp_path)
    output = tmp_path / "out" / "eval.jsonl"

    result = runner.invoke(
        app,
        [
            "export",
            task_set_id,
            "--format",
            "eval",
            "--verifier",
            reviewed_id,
            "--output",
            str(output),
            "--max-rows-per-family",
            "1",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "sft only" in plain(result.stdout)
    assert not output.exists()


def test_a_cap_below_one_is_refused_before_anything_is_written(tmp_path) -> None:
    task_set_id, reviewed_id = _reviewed_refund_verifier(tmp_path)
    output = tmp_path / "out" / "sft.jsonl"

    result = runner.invoke(
        app,
        [
            "export",
            task_set_id,
            "--format",
            "sft",
            "--verifier",
            reviewed_id,
            "--output",
            str(output),
            "--max-rows-per-lineage",
            "0",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "at least 1" in plain(result.stdout)
    assert not output.exists()


def test_a_task_set_recording_no_grouping_says_so_rather_than_reading_as_normal(tmp_path) -> None:
    """An artifact from before this was recorded is the one case it cannot be recovered for."""
    store = DerivedStore(tmp_path / ".bandits")
    task_set = load_task_set(_mined(tmp_path), store)
    older = save_task_set(task_set.replace(clustering=None), store).artifact_id

    result = runner.invoke(app, ["families", older, "--project", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "records nothing about how it" in plain(result.stdout)


def test_draft_verifier_reports_candidate_behavior_and_says_when_uncalibrated(tmp_path) -> None:
    """A ranked list with nothing behind it invites the top row to be taken as the answer."""
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    family = next(
        item for item in load_task_set(task_set_id, store).families if "refund" in item.descriptor
    )

    result = runner.invoke(
        app,
        [
            "draft-verifier",
            task_set_id,
            "--family",
            family.family_id,
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "frequency-based hypothesis" in plain(result.stdout)
    draft = load_verifier_draft(result.stdout.split()[1], store)
    assert draft.candidates
    assert all(item.derivation == "frequency" for item in draft.candidates)
    assert all(item.considered for item in draft.candidates)


def _ledger_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_an_interview_turn_is_reconstructable_from_the_ledger(tmp_path, monkeypatch) -> None:
    """The chain the ledger exists for, end to end.

    What was shown to the reviewer, what they replied, what the model read it
    as, whether that reading survived, and which artifact the answer produced.
    Each piece existed somewhere already; what was missing was the order they
    happened in and whether the human agreed with the model.
    """
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right to me\ny\nthe log is the source\ny\n" * 12,
    )
    assert result.exit_code == 0, result.output

    turns = [row for row in _ledger_rows(path) if row["event_type"] == "interview_turn"]
    assert turns, "a review that decided a check has to leave a turn behind"
    turn = turns[0]

    assert turn["reply"] == "looks right to me"
    assert turn["outcome"] == "applied"
    assert turn["proposed_decision"] == "accept"
    assert turn["applied_decision"] == "accept"
    assert turn["decision_source"] == "model_accepted"
    assert turn["model_overruled"] is False

    # Everything the reviewer had in front of them, not just the prompt lines.
    shown = turn["shown"]
    assert shown["prompt_lines"]
    assert shown["check"]["check_id"]
    assert shown["check"]["claim"]
    assert "passed" in shown["scored"]
    assert "agreements" in shown and "gameability" in shown
    assert "blind_spots" in shown and "prior_decisions" in shown
    assert turn["authoritative"] is True
    assert turn["authoritative_why"] == "the log is the source"
    assert turn["answered_seconds"] >= 0
    assert turn["shown_at"]

    # Lineage, not duplication: the turn names the artifacts either side of it
    # and the store holds what is in them.
    assert turn["output_artifact_id"] != turn["input_artifact_id"]
    store = DerivedStore(tmp_path / ".bandits")
    interview = load_interview(turn["output_artifact_id"], store)
    assert any(r.review_id == turn["review_id"] for r in interview.reviews), (
        "the turn has to name a review that really exists in the artifact it produced"
    )


def test_an_overruled_reading_is_visible_as_overruled_in_the_ledger(tmp_path, monkeypatch) -> None:
    """Accepting the model and overruling it must never look the same.

    A decision the reviewer had to enter after refusing the model's reading is
    weaker evidence than one they agreed with, and a record that flattened the
    two would misrepresent every overruled check.
    """
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "actually no\ny\nwhy not\nn\nr\n" * 12,
    )
    assert result.exit_code == 0, result.output

    turn = next(row for row in _ledger_rows(path) if row["event_type"] == "interview_turn")
    assert turn["proposed_decision"] == "accept", "what the model said"
    assert turn["applied_decision"] == "reject", "what the reviewer decided instead"
    assert turn["decision_source"] == "model_overruled"
    assert turn["model_overruled"] is True


def test_an_overruled_proposal_is_kept_whole_beside_what_replaced_it(tmp_path, monkeypatch) -> None:
    """A decision and a rationale are not the proposal.

    The revised value, the operator and the combine target are what a reading
    would have done, and they are what a reviewer accepted or refused. Keeping
    only the decision leaves an overrule showing that something was rejected
    and not what.
    """
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("revise", revised_expected="shipped")),
        # reply, authoritative, why, refuse, manual reject
        "change it\ny\nwhy\nn\nr\n" * 12,
    )
    assert result.exit_code == 0, result.output

    turn = next(row for row in _ledger_rows(path) if row["event_type"] == "interview_turn")
    proposed = turn["proposed_interpretation"]
    assert proposed["decision"] == "revise"
    assert proposed["revised_expected"] == "shipped", "the payload the reviewer refused"
    # A manual reject carries no payload of its own, so nothing replaced the
    # proposal; `applied_decision` is where the reviewer's choice lands.
    assert turn["applied_interpretation"] is None
    assert turn["applied_decision"] == "reject"


def test_the_gameability_coverage_the_reviewer_saw_is_recorded(tmp_path, monkeypatch) -> None:
    """Coverage says the attacks that were never tried, which the attacks cannot.

    `_show_check_summary` prints it and the record omitted it, so a reviewer
    who was told no template could attack a check read as one shown a clean
    sheet.
    """
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter(_decision("accept")),
        "looks right\ny\nsystem of record\ny\n" * 12,
    )
    assert result.exit_code == 0, result.output

    turn = next(row for row in _ledger_rows(path) if row["event_type"] == "interview_turn")
    assert "gameability_assessment" in turn["shown"]
    assert "gaming_hypotheses" in turn["shown"]


def test_a_manual_decision_after_a_failure_is_not_called_an_override(tmp_path, monkeypatch) -> None:
    """There was no reading to overrule, so calling it one invents an opinion.

    A model that failed to parse gave no recommendation. Recording that as an
    override would report a disagreement that never happened.
    """
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter("this is not json"),
        "no idea\ny\nbecause\na\n" * 12,
    )
    assert result.exit_code == 0, result.output

    turn = next(row for row in _ledger_rows(path) if row["event_type"] == "interview_turn")
    assert turn["failure"], "the interpretation really did fail"
    assert turn["proposed_decision"] is None, "no reading existed"
    assert turn["decision_source"] == "manual_after_failure"
    assert turn["model_overruled"] is False, "nothing was overruled"


def test_a_reviewer_who_stops_still_leaves_a_record(tmp_path, monkeypatch) -> None:
    """Stopping is an interaction. A turn that vanished would shorten the review."""
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    draft_id = _review_draft(tmp_path)

    result = _run_review(
        tmp_path,
        draft_id,
        _fake_interpreter("this is not json"),
        "no idea\ny\nbecause\nq\n",
    )
    assert result.exit_code == 0, result.output

    turns = [row for row in _ledger_rows(path) if row["event_type"] == "interview_turn"]
    assert turns, "a reviewer who quit still answered a question first"
    assert turns[-1]["outcome"] == "stopped"
    assert turns[-1]["applied_decision"] is None
    assert turns[-1]["reply"] == "no idea"
