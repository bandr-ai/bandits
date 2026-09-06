from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from bandits.analyze import load_analysis, load_task_set
from bandits.analyze.embed import EmbeddingError
from bandits.cli import app
from bandits.export import direct_sft
from bandits.ingest.otlp import load_otlp
from bandits.store import DerivedStore, compute_artifact_id
from bandits.verify import draft_verifiers, load_interview, load_verifier_draft, save_verifier_draft
from bandits.verify.validate import Agreement, Validation, save_validation

runner = CliRunner()
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.otlp.jsonl"


def test_ingest_prints_artifact_summary(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "artifact_id: corpus-" in result.stdout
    assert "traces:      2" in result.stdout


def test_ingest_unknown_source_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(
        app, ["ingest", str(FIXTURE), "--source", "nope", "--project", str(tmp_path)]
    )
    assert result.exit_code == 1


def test_list_shows_ingested_artifact(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])

    result = runner.invoke(app, ["list", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "otlp" in result.stdout


def test_show_lists_traces_then_one_traces_spans(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    overview = runner.invoke(app, ["show", artifact_id, "--project", str(tmp_path)])
    assert overview.exit_code == 0
    assert "trace-1" in overview.stdout
    assert "trace-2" in overview.stdout

    detail = runner.invoke(
        app, ["show", artifact_id, "--trace", "trace-1", "--project", str(tmp_path)]
    )
    assert detail.exit_code == 0
    assert "lookup_order" in detail.stdout


CODING_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.coding.otlp.jsonl"
)


def test_analyze_reports_tasks_and_never_hides_limitations(tmp_path) -> None:
    runner.invoke(app, ["ingest", str(FIXTURE), "--source", "otlp", "--project", str(tmp_path)])
    artifact_id = compute_artifact_id(load_otlp(FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "analysis_id: analysis-" in result.stdout
    assert "task-trace-1" in result.stdout
    assert "limitation:" in result.stdout


def test_analyze_reads_a_coding_corpus_too(tmp_path) -> None:
    runner.invoke(
        app, ["ingest", str(CODING_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    artifact_id = compute_artifact_id(load_otlp(CODING_FIXTURE))

    result = runner.invoke(app, ["analyze", artifact_id, "--tasks", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "task-code-1" in result.stdout


def test_analyze_unknown_artifact_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(app, ["analyze", "corpus-nope", "--project", str(tmp_path)])
    assert result.exit_code == 1


_BLANK_ANSWERS = "\n" * 60
"""Enough blank answers to walk any draft's interview to completion."""

SUPPORT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "traces.support.otlp.jsonl"
)


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    """Keep `mine` off the network.

    Grouping now embeds every descriptor, so without this the whole suite would
    need an API key. Vectors are derived from the descriptor's own tokens, which
    gives the support fixture stable, expected families — these
    tests are about the command's plumbing, not about grouping quality.
    """

    def deterministic(model: str, texts):
        # Hashed into a fixed width rather than a per-call vocabulary: build_cache
        # embeds in batches, so a vocabulary derived from the batch would give
        # later batches a different dimensionality and make cosine_distance raise.
        # Wide enough that the fixtures' tokens do not collide, which would merge
        # descriptors that share no words.
        width = 4096
        vectors = []
        for text in texts:
            vector = [0.0] * width
            for token in text.split():
                vector[hash_token(token) % width] = 1.0
            vectors.append(vector)
        return vectors

    monkeypatch.setattr("bandits.cli.build_cache", _partial_embed(deterministic))


def hash_token(token: str) -> int:
    """Stable across processes, unlike hash(), so vectors do not shift per run."""
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def _partial_embed(embedder):
    """Bind a stub embedder into `build_cache` without changing its signature."""
    from bandits.analyze.embed import build_cache

    def patched(texts, *, model, embed=None, existing=None):
        return build_cache(texts, model=model, embed=embedder, existing=existing)

    return patched


def _mined(tmp_path: Path) -> str:
    """Ingest, analyze and mine the support fixture; return the task set id."""
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analyzed = runner.invoke(app, ["analyze", corpus_id, "--project", str(tmp_path)])
    analysis_id = analyzed.stdout.split()[1]
    mined = runner.invoke(app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)])
    assert mined.exit_code == 0, mined.stdout
    return mined.stdout.split()[1]


def test_mine_reports_coverage_and_unfilled_slots(tmp_path) -> None:
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analysis_id = runner.invoke(
        app, ["analyze", corpus_id, "--project", str(tmp_path)]
    ).stdout.split()[1]

    result = runner.invoke(app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)])

    assert result.exit_code == 0
    assert "taskset_id:  taskset-" in result.stdout
    assert "coverage:" in result.stdout
    assert "missing slot" in result.stdout


def test_mine_embeds_and_records_the_cache_it_grouped_with(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    task_set = load_task_set(task_set_id, DerivedStore(tmp_path / ".bandits"))

    embeddings = DerivedStore(tmp_path / ".bandits").list(kind="embeddings")
    assert len(embeddings) == 1, "grouping should have saved exactly one cache"
    assert embeddings[0].parent_artifact_id == task_set.corpus_id
    assert {f.proposed_by for f in task_set.families} == {"model"}


def test_mining_twice_reuses_the_saved_cache(tmp_path) -> None:
    """A second run must not pay for the same vectors again."""
    _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    before = store.list(kind="embeddings")[0].artifact_id

    def refuse(model, texts):
        raise AssertionError(f"re-embedded {len(texts)} descriptor(s) already cached")

    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analysis_id = runner.invoke(
        app, ["analyze", corpus_id, "--project", str(tmp_path)]
    ).stdout.split()[1]

    with mock.patch("bandits.cli.build_cache", _partial_embed(refuse)):
        again = runner.invoke(
            app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)]
        )

    assert again.exit_code == 0, again.stdout
    assert store.list(kind="embeddings")[0].artifact_id == before


def test_mine_fails_loudly_when_embedding_fails(tmp_path) -> None:
    """No silent fallback: a task set nobody knows to distrust is the bug itself."""
    runner.invoke(
        app, ["ingest", str(SUPPORT_FIXTURE), "--source", "otlp", "--project", str(tmp_path)]
    )
    corpus_id = compute_artifact_id(load_otlp(SUPPORT_FIXTURE))
    analysis_id = runner.invoke(
        app, ["analyze", corpus_id, "--project", str(tmp_path)]
    ).stdout.split()[1]

    def unreachable(model, texts):
        raise EmbeddingError("FIREWORKS_API_KEY is not set")

    with mock.patch("bandits.cli.build_cache", _partial_embed(unreachable)):
        result = runner.invoke(
            app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)]
        )

    assert result.exit_code == 1
    assert "FIREWORKS_API_KEY" in result.stdout
    assert not DerivedStore(tmp_path / ".bandits").list(kind="taskset")


def test_mine_warns_when_grouping_found_no_structure(tmp_path) -> None:
    """An inert grouping stage still reports high coverage; the warning is the tell."""
    # The support fixture repeats four instructions across its traces, and exact
    # duplicates collapse before clustering. Singletons need distinct text.
    corpus = tmp_path / "varied.otlp.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "trace_id": f"t-{i}",
                    "span_id": f"t-{i}-s0",
                    "parent_span_id": None,
                    "name": "gpt-5",
                    "start_time": "2026-03-01T00:00:00Z",
                    "end_time": "2026-03-01T00:00:30Z",
                    "attributes": {
                        "gen_ai.operation.name": "chat",
                        "task": "unrelated request "
                        + "alpha bravo charlie delta echo foxtrot".split()[i],
                        "gen_ai.completion": "done",
                    },
                }
            )
            for i in range(6)
        )
    )
    runner.invoke(app, ["ingest", str(corpus), "--source", "otlp", "--project", str(tmp_path)])
    corpus_id = compute_artifact_id(load_otlp(corpus))
    analysis_id = runner.invoke(
        app, ["analyze", corpus_id, "--project", str(tmp_path)]
    ).stdout.split()[1]

    # Every descriptor mutually distant, so each lands in its own family.
    def orthogonal(model, texts):
        return [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(len(texts))]

    with mock.patch("bandits.cli.build_cache", _partial_embed(orthogonal)):
        result = runner.invoke(
            app, ["mine", analysis_id, "--budget", "10", "--project", str(tmp_path)]
        )

    assert result.exit_code == 0
    assert "contain one trace" in result.stdout


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
    assert family.descriptor in result.stdout
    assert family.medoid_trace_id in result.stdout
    assert "held_out" in result.stdout


def test_families_rejects_an_unknown_family(tmp_path) -> None:
    task_set_id = _mined(tmp_path)

    result = runner.invoke(
        app, ["families", task_set_id, "--family", "family-nope", "--project", str(tmp_path)]
    )

    assert result.exit_code == 1


def test_merge_families_writes_a_new_task_set(tmp_path) -> None:
    task_set_id = _mined(tmp_path)
    store = DerivedStore(tmp_path / ".bandits")
    families = load_task_set(task_set_id, store).families

    result = runner.invoke(
        app,
        # fmt: off
        [
            "merge-families",
            task_set_id,
            families[0].family_id,
            families[1].family_id,
            "--project",
            str(tmp_path),
        ],
        # fmt: on
    )

    assert result.exit_code == 0
    assert "coherence unknown" in result.stdout
    merged_id = result.stdout.split()[1]
    assert merged_id != task_set_id
    assert load_task_set(task_set_id, store).families == families, "the original is untouched"
    assert len(load_task_set(merged_id, store).families) == len(families) - 1


def test_mine_unknown_analysis_exits_nonzero(tmp_path) -> None:
    result = runner.invoke(app, ["mine", "analysis-nope", "--project", str(tmp_path)])
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
    assert "status:       complete" in result.stdout
    interview_id = next(
        line.split(maxsplit=1)[1]
        for line in result.stdout.splitlines()
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
    assert "verifier_draft_id:" in result.stdout
    assert "interview_id:" in result.stdout
    assert "status:       complete" in result.stdout


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
    assert "validation_id: validation-" in result.stdout
    # One hypothesis is right and one is wrong; the run where the wrong one
    # would have rewarded doing nothing is named.
    assert "100%" in result.stdout and "0%" in result.stdout
    assert "false_positive" in result.stdout
    assert "gamed" in result.stdout


def test_labeling_can_be_quit_early(tmp_path) -> None:
    draft_id, _ = _drafted_family(tmp_path, "address")

    result = runner.invoke(
        app,
        ["label", draft_id, "--labeler", "owner", "--project", str(tmp_path)],
        input="s\nfine\nq\n",
    )

    assert result.exit_code == 0
    assert "labels:       1" in result.stdout


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
        if item.checks[0].claim == "final_state_field:status"
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
    result = runner.invoke(
        app,
        [
            "review-verifier",
            draft_id,
            "--validation",
            validation_id,
            "--verifier",
            spec.verifier_id,
            "--acceptance-id",
            "owner-ticket-cli",
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
    assert "rows:" in trained.stdout and "unresolved:" in trained.stdout
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
    assert "reviewed:   1" in result.stdout
    assert (output / "sft.jsonl").exists()
    assert (output / "review.jsonl").exists()
    assert (output / "rejected.jsonl").exists()
    assert (output / "selection-report.json").exists()
