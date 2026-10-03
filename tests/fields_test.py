"""Finding and reading any field by its path, absent told apart from null."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bandits.cli import app
from bandits.fields import ABSENT, fields, resolve, values
from bandits.ingest import load_corpus
from bandits.store import ArtifactStore


def _corpus(tmp_path: Path):
    observation = {
        "id": "o1",
        "type": "GENERATION",
        "name": "chat",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": [{"role": "user", "content": "hi"}],
        "output": "hello",
        "usageDetails": {"input": 5, "output": 2},
        "tags": ["a", "b"],
        "custom_null": None,
        "metadata": {"reasoning": "because", "nested": {"service.name": "svc"}},
    }
    path = tmp_path / "lf.json"
    path.write_text(json.dumps({"id": "t1", "observations": [observation]}))
    return load_corpus(path, "langfuse")


def test_fields_lists_paths_types_counts_and_examples(tmp_path: Path) -> None:
    rows = {row["path"]: row for row in fields(_corpus(tmp_path))}
    assert rows["misc.usageDetails.input"]["types"] == {"number": 1}
    assert rows["misc.usageDetails.input"]["examples"] == ["5"]
    assert rows["misc.tags[]"]["types"] == {"string": 2}
    assert rows["misc.custom_null"]["types"] == {"null": 1}
    assert rows["metadata.reasoning"]["group"] == "metadata"
    assert rows["metadata.nested.service.name"]["examples"] == ["svc"]
    assert rows["gen_ai.usage.input_tokens"]["group"] == "normalized"
    assert rows["trace.trace_id"]["of"] == 1
    assert all(row["count"] <= row["of"] for row in rows.values())


def test_resolve_takes_the_longest_key_and_tells_absent_from_null() -> None:
    view = {"gen_ai.usage.input_tokens": 3, "misc": {"a": None, "list": [{"x": 1}]}}
    assert resolve(view, "gen_ai.usage.input_tokens") == 3
    assert resolve(view, "misc.a") is None
    assert resolve(view, "misc.b") is ABSENT
    assert resolve(view, "misc.list[0].x") == 1
    assert resolve(view, "misc.list[5].x") is ABSENT
    assert resolve({"m": '{"k": {"service.name": "s"}}'}, "m.k.service.name") == "s"


def test_values_carry_ids_and_mark_absent(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    rows = list(values(corpus, "misc.custom_null"))
    held = [row for row in rows if row["present"]]
    assert held and held[0]["value"] is None and held[0]["step_id"]
    missing = list(values(corpus, "misc.not_there"))
    assert missing and not any(row["present"] for row in missing)
    assert list(values(corpus, "misc.not_there", present_only=True)) == []
    (trace_row,) = values(corpus, "trace.trace_id")
    assert trace_row["step_id"] is None and trace_row["value"] == corpus.traces[0].trace_id


def test_cli_prints_json_for_agents(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    artifact = ArtifactStore(tmp_path / ".bandits").write(corpus, source_path="x").artifact_id
    runner = CliRunner()
    listed = runner.invoke(app, ["fields", artifact, "--json", "--project", str(tmp_path)])
    assert listed.exit_code == 0, listed.output
    assert any(row["path"] == "metadata.reasoning" for row in json.loads(listed.stdout))
    got = runner.invoke(
        app,
        [
            "get",
            artifact,
            "--field",
            "metadata.reasoning",
            "--present",
            "--json",
            "--project",
            str(tmp_path),
        ],
    )
    assert got.exit_code == 0, got.output
    (line,) = [json.loads(line) for line in got.stdout.splitlines()]
    assert line["value"] == "because" and line["present"]
    missing = runner.invoke(app, ["fields", "corpus-nope", "--project", str(tmp_path)])
    assert missing.exit_code == 1


def test_metadata_stored_as_the_resource_still_reads_under_its_own_name(tmp_path: Path) -> None:
    observation = {
        "id": "o1",
        "type": "GENERATION",
        "name": "chat",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": "hi",
        "output": "hello",
        "metadata": {"resourceAttributes": {"service.name": "svc"}, "kept": 1},
    }
    path = tmp_path / "lf.json"
    path.write_text(json.dumps({"id": "t1", "observations": [observation]}))
    corpus = load_corpus(path, "langfuse")
    (step,) = corpus.traces[0].spans
    assert "metadata.resourceAttributes" not in step.attributes  # stored once, as the resource
    (row,) = values(corpus, "metadata.resourceAttributes.service.name", present_only=True)
    assert row["value"] == "svc"
    (row,) = values(corpus, "resource.service.name", present_only=True)
    assert row["value"] == "svc"


def test_values_stored_once_still_read_under_their_own_names(tmp_path: Path) -> None:
    base = {"startTime": "2026-01-01T00:00:00Z", "endTime": "2026-01-01T00:00:01Z"}
    observations = [
        {
            **base,
            "id": "o1",
            "type": "GENERATION",
            "name": "chat",
            "output": "hello",
            "input": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        },
        {
            **base,
            "id": "o2",
            "type": "TOOL",
            "name": "lookup",
            "input": {"order_id": "A-1"},
            "output": {"status": "shipped"},
        },
    ]
    path = tmp_path / "lf.json"
    path.write_text(json.dumps({"id": "t1", "userId": "u-1", "observations": observations}))
    corpus = load_corpus(path, "langfuse")
    spans = {span.name: span for span in corpus.traces[0].spans}
    tool, model = spans["lookup"], spans["chat"]
    # Held once, in the step's own fields; the copies are gone.
    assert "input.value" not in tool.attributes and "output.value" not in tool.attributes
    assert tool.attributes["bandits.stored_as"] == {
        "input.value": "arguments",
        "output.value": "output",
    }
    assert "input.value" not in model.attributes
    (row,) = (
        values(corpus, "input.value", present_only=True, trace_ids=[])
        if False
        else [r for r in values(corpus, "input.value", present_only=True) if r["step"] == "lookup"]
    )
    assert row["value"] == {"order_id": "A-1"}
    (row,) = [r for r in values(corpus, "input.value", present_only=True) if r["step"] == "chat"]
    assert row["value"] == observations[0]["input"]
    # The trace's own record is on the trace, not copied onto steps.
    assert corpus.traces[0].source_record["userId"] == "u-1"
    (row,) = values(corpus, "trace.record.userId")
    assert row["value"] == "u-1"


def test_a_prompt_the_messages_cannot_rebuild_is_kept(tmp_path: Path) -> None:
    observation = {
        "id": "o1",
        "type": "GENERATION",
        "name": "chat",
        "output": "hello",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": [{"role": "user", "content": "hi", "name": "alex"}],
    }
    path = tmp_path / "lf.json"
    path.write_text(json.dumps({"id": "t1", "observations": [observation]}))
    (step,) = load_corpus(path, "langfuse").traces[0].spans
    assert "input.value" in step.attributes


def test_fields_filters_by_kind_and_pages(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    rows = fields(corpus, kind="tool")
    assert rows == []  # the only step is a model call
    assert all(not r["path"].startswith("trace.") for r in fields(corpus, kind="model"))
    artifact = ArtifactStore(tmp_path / ".bandits").write(corpus, source_path="x").artifact_id
    runner = CliRunner()
    out = runner.invoke(
        app,
        [
            "fields",
            artifact,
            "--json",
            "--prefix",
            "misc.",
            "--limit",
            "2",
            "--project",
            str(tmp_path),
        ],
    )
    assert out.exit_code == 0, out.output
    rows = json.loads(out.stdout)
    assert len(rows) == 2 and all(r["path"].startswith("misc.") for r in rows)
    assert "--offset 2 for more" in out.stderr
