"""Discovery against the real loader: what it proposes is what the load resolves."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.discovery import discover, discover_requests
from bandits.traces import WorkflowDeclaration
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


def test_two_identities_with_different_fields_leave_the_task_unresolved(tmp_path) -> None:
    path = _write(tmp_path / "new.jsonl", *(_request(_new(c * 32)) for c in "56"))
    found = discover(discover_requests(path, "otlp-std"))
    assert found.task_fields == () and found.delivered_field is None
    assert [o.describe() for o in found.task_options] == [
        "input.payload.query → handle(SPAN) 2/2",
        "input.question → graph(CHAIN) 2/2",
    ]
    corpus = _load(path, found)
    assert {t.request.task_status for t in corpus.traces} == {"unresolved"}


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
    assert "task:     not chosen: input.payload.query → handle(SPAN) 2/2" in out
    assert "--task-field input.payload.query (selects handle(SPAN); 2/2)" in out
    assert "--task-field input.question (selects graph(CHAIN); 2/2)" in out
