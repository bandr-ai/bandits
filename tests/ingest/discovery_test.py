"""Discovery against the real loader: what it proposes is what the load resolves."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.discovery import discover, discover_requests
from bandits.traces import WorkflowDeclaration
from tests.cli_test import plain
from tests.ingest.otlp_standard_test import _request, _span, _write

QUESTION = "Can I change the delivery address after checkout?"


def _old(trace: str) -> list[dict]:
    return [
        _span(
            "run",
            "handle",
            {
                "langfuse.observation.type": "SPAN",
                "input.value": json.dumps({"query": QUESTION}),
                "output.value": json.dumps({"answer": "Yes, until it ships."}),
            },
            parent="gone",
            trace=trace,
        ),
        _span(
            "m",
            "call",
            {
                "gen_ai.operation.name": "chat",
                "input.value": QUESTION,
                "output.value": "Yes, until it ships.",
            },
            parent="run",
            at=1,
            trace=trace,
        ),
    ]


def _new(trace: str) -> list[dict]:
    return [
        _span(
            "run",
            "handle",
            {
                "langfuse.observation.type": "SPAN",
                "input.value": json.dumps({"payload": {"query": QUESTION}}),
                "output.value": json.dumps({"answer": "Yes, until it ships."}),
            },
            parent="gone-a",
            trace=trace,
        ),
        _span(
            "graph",
            "graph",
            {
                "langfuse.observation.type": "CHAIN",
                "input.value": json.dumps({"question": QUESTION}),
            },
            parent="gone-b",
            trace=trace,
        ),
        _span(
            "m",
            "call",
            {
                "gen_ai.operation.name": "chat",
                "input.value": QUESTION,
                "output.value": "Yes, until it ships.",
            },
            parent="graph",
            at=1,
            trace=trace,
        ),
    ]


def _load(path, found):
    return load_corpus(
        path,
        "otlp-std",
        workflow=WorkflowDeclaration(
            task_fields=found.task_fields, delivered_field=found.delivered_field
        ),
    )


def test_mixed_shapes_resolve_every_task_and_answer(tmp_path) -> None:
    traces = [_old("1" * 32), _new("2" * 32), _old("3" * 32), _new("4" * 32)]
    path = _write(tmp_path / "mixed.jsonl", *(_request(t) for t in traces))
    found = discover(discover_requests(path, "otlp-std"))
    assert found.task_fields == ("input.query", "input.payload.query")
    assert found.delivered_field == "output.answer"
    corpus = _load(path, found)
    assert [t.request.task_status for t in corpus.traces] == ["declared"] * 4
    assert {t.request.delivered for t in corpus.traces} == {"Yes, until it ships."}


def test_two_identities_holding_the_same_question_agree(tmp_path) -> None:
    path = _write(tmp_path / "new.jsonl", *(_request(_new(c * 32)) for c in "56"))
    found = discover(discover_requests(path, "otlp-std"))
    assert found.task_fields == ("input.payload.query", "input.question")
    assert found.delivered_field == "output.answer"
    assert [o.describe() for o in found.task_options] == [
        "input.payload.query → handle(SPAN) 2/2",
        "input.question → graph(CHAIN) 2/2",
    ]
    corpus = _load(path, found)
    for trace in corpus.traces:
        request = trace.request
        assert (request.task_status, request.task) == ("declared", QUESTION)
        assert request.invocation_basis.startswith("agreement: 2 outermost candidates")
        assert {c.path for c in request.task_candidates} == {
            "input.payload.query",
            "input.question",
        }
        assert request.delivered == "Yes, until it ships."


def test_different_questions_are_kept_unresolved_with_every_value(tmp_path) -> None:
    other = "What is the return window?"
    traces = []
    for c in "78":
        spans = _new(c * 32)
        for span in spans:
            for item in span["attributes"]:
                if item["key"] == "input.value" and "question" in item["value"]["stringValue"]:
                    item["value"]["stringValue"] = json.dumps({"question": other})
        traces.append(spans)
    path = _write(tmp_path / "conflict.jsonl", *(_request(t) for t in traces))
    found = discover(discover_requests(path, "otlp-std"))
    corpus = _load(path, found)
    for trace in corpus.traces:
        request = trace.request
        assert (request.task_status, request.task, request.source_span_id) == (
            "conflict",
            None,
            None,
        )
        assert {c.value for c in request.task_candidates} == {QUESTION, other}
        assert request.invocation_basis.startswith("conflict:")


def test_native_discovery_reads_the_same_candidates(tmp_path) -> None:
    def observation(oid, kind, parent, **io):
        return {
            "id": oid,
            "type": kind,
            "name": oid,
            "parentObservationId": parent,
            "startTime": "2026-01-01T00:00:00Z",
            "endTime": "2026-01-01T00:00:01Z",
            **io,
        }

    trace = {
        "id": "t",
        "observations": [
            observation(
                "run", "SPAN", None, input={"payload": {"query": QUESTION}}, output={"answer": "ok"}
            ),
            observation("m", "GENERATION", "run", input=QUESTION, output="ok"),
        ],
    }
    path = tmp_path / "lf.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    found = discover(discover_requests(path, "langfuse"))
    assert (found.task_fields, found.delivered_field) == (("input.payload.query",), "output.answer")


def test_cli_builds_the_corpus_once_and_prints_the_options(tmp_path, monkeypatch) -> None:
    import bandits.cli

    calls = []
    real = bandits.cli.iter_corpus
    monkeypatch.setattr(
        bandits.cli, "iter_corpus", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    path = _write(tmp_path / "new.jsonl", *(_request(_new(c * 32)) for c in "56"))
    result = CliRunner().invoke(app, ["ingest", str(path), "--source", "otlp-std", "--dry-run"])
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert result.exit_code == 0, out
    assert len(calls) == 1
    assert "task:     input.payload.query, input.question (found" in out
    assert "problems: none" in out


def test_temporary_files_go_in_the_project_not_the_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/upstream/langfuse/agno-2025-06-11.trace.json"
    )
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    monkeypatch.chdir(readonly)
    try:
        for args in (
            ["ingest", str(fixture), "--source", "langfuse"],
            ["mapping", "propose", str(fixture), "--source", "langfuse", "--name", "m"],
        ):
            result = CliRunner().invoke(app, [*args, "--project", str(tmp_path / "project")])
            assert result.exit_code == 0, (result.output, result.exception)
        assert list(readonly.iterdir()) == []
        assert [p.name for p in (tmp_path / "project").iterdir()] == [".bandits"]

        result = CliRunner().invoke(
            app, ["ingest", str(fixture), "--source", "langfuse", "--project", str(readonly)]
        )
        assert result.exit_code == 1, result.output
        assert "cannot write to" in plain(result.output)
    finally:
        readonly.chmod(0o755)
