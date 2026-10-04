"""Native records are reached through a pointer into the redacted source archive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import load_corpus
from bandits.ingest.otlp_standard import _Decoded, _pruned_context
from bandits.store import ArtifactStore
from tests.ingest.otlp_standard_test import _request, _span, _write

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "upstream"
SECRET = "reach me at someone@mailhost.test"  # redacted by the default ruleset


def _observation(oid: str, kind: str = "GENERATION", **fields) -> dict:
    return {
        "id": oid,
        "type": kind,
        "name": oid,
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": SECRET,
        "output": "ok",
        **fields,
    }


def _trace(tid: str) -> dict:
    root = _observation(f"{tid}-root", "SPAN", children=[_observation(f"{tid}-nested")])
    return {"id": tid, "observations": [root, _observation(f"{tid}-flat")]}


def _run(rid: str, **fields) -> dict:
    return {
        "id": rid,
        "run_type": "llm",
        "name": rid,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "inputs": {"text": SECRET},
        "outputs": {"text": "ok"},
        **fields,
    }


def _phoenix(sid: str) -> dict:
    return {
        "name": sid,
        "context": {"trace_id": "t" * 8, "span_id": sid},
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {"openinference.span.kind": "LLM", "input.value": SECRET},
    }


LAYOUTS = {
    "langfuse-jsonl": ("langfuse", "a.jsonl", "\n\n".join(json.dumps(_trace(t)) for t in "ab")),
    "langfuse-array": ("langfuse", "a.json", json.dumps([_trace("a"), _trace("b")], indent=1)),
    "langfuse-document": ("langfuse", "a.json", json.dumps(_trace("a"), indent=1)),
    # One object per line in a .json file: archived by the whole-file path.
    "langfuse-json-lines": ("langfuse", "a.json", "\n".join(json.dumps(_trace(t)) for t in "ab")),
    "langsmith-runs": (
        "langsmith",
        "r.json",
        json.dumps({"runs": [_run("r1", child_runs=[_run("r2")]), _run("r3")]}),
    ),
    "langsmith-jsonl": ("langsmith", "r.jsonl", "\n".join(json.dumps(_run(r)) for r in "xy")),
    "phoenix-spans": ("phoenix", "p.json", json.dumps({"spans": [_phoenix("s1"), _phoenix("s2")]})),
    "phoenix-line-array": ("phoenix", "p.jsonl", json.dumps([_phoenix("s1"), _phoenix("s2")])),
    # A bare CR is JSON whitespace inside a record, not a line break.
    "langsmith-jsonl-cr": (
        "langsmith",
        "r.jsonl",
        "\n".join(json.dumps(_run(r)).replace(", ", ",\r ", 1) for r in "xy"),
    ),
    "langfuse-jsonl-crlf": (
        "langfuse",
        "a.jsonl",
        "\r\n".join(json.dumps(_trace(t)) for t in "ab"),
    ),
}


def _every_span_resolves(tmp_path: Path, path: Path, source: str) -> int:
    corpus = load_corpus(path, source)
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path)).artifact_id
    spans = [s for t in corpus.traces for s in t.spans] + [
        n for t in corpus.traces for n in t.workflow_nodes
    ]
    for span in spans:
        pointer = json.loads(span.attributes["bandits.source.record"])
        record = store.read_native_record(artifact, pointer)
        context = record.get("context") or {}
        assert pointer["observation_id"] in {
            str(record.get("id")),
            str(record.get("run_id")),
            str(context.get("span_id")),
        }
        assert "someone@mailhost.test" not in json.dumps(record)
    return len(spans)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_every_span_points_at_its_archived_record(tmp_path, layout) -> None:
    source, name, text = LAYOUTS[layout]
    path = tmp_path / name
    path.write_bytes(text.encode())
    assert _every_span_resolves(tmp_path, path, source) >= 2


@pytest.mark.parametrize(
    ("fixture", "source"),
    [
        ("langfuse/agno-2025-06-11.trace.json", "langfuse"),
        ("phoenix/sdk-server-getspans.json", "phoenix"),
    ],
)
def test_upstream_fixtures_resolve(tmp_path, fixture, source) -> None:
    assert _every_span_resolves(tmp_path, FIXTURES / fixture, source) >= 1


def test_the_copied_native_record_is_gone(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(_trace("a")))
    span = load_corpus(path, "langfuse").traces[0].spans[0]
    assert "bandits.native.record" not in span.attributes
    assert json.loads(span.attributes["bandits.source.record"])["line"] == 1


def _decoded(resource: dict, span_attributes: dict) -> _Decoded:
    context = {
        "resource": resource,
        "span_attributes": span_attributes,
        "scope": {},
        "links": [],
        "events": [],
        "status": None,
    }
    return _Decoded(source_context=context)


def test_only_shadowed_or_rewritten_declarations_are_kept() -> None:
    declared = {"same": 1, "shadow": "span", "zero": 0, "messages": "[]", "plain": "x"}
    decoded = _decoded({"shadow": "resource", "zero": False}, declared)
    final = {**declared, "messages": [{"role": "user", "parts": []}]}
    pruned = _pruned_context(decoded, final)
    assert pruned["span_attributes"] == {"shadow": "span", "zero": 0, "messages": "[]"}
    assert _pruned_context(decoded)["span_attributes"] == {"shadow": "span", "zero": 0}
    # Never mutated: the episode attributes share this dict.
    assert decoded.source_context["span_attributes"] == declared


def test_normalized_messages_keep_their_declared_bytes(tmp_path) -> None:
    attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.input.messages": "[]",
        "input.value": "hello",
        "output.value": "hi",
    }
    path = _write(tmp_path / "x.jsonl", _request([_span("m", "call", attributes)]))
    span = load_corpus(path, "otlp-std").traces[0].spans[0]
    kept = span.attributes["bandits.otlp.source_context"]["span_attributes"]
    assert kept == {"gen_ai.input.messages": "[]"}
    assert span.attributes["gen_ai.input.messages"][0]["parts"][0]["content"] == "hello"


def test_a_one_line_array_past_the_line_limit_streams(tmp_path, monkeypatch) -> None:
    # A compact multi-GB array is one line; reading it as a line would hold
    # the whole file. A small limit stands in for the real one.
    import bandits.ingest.native as native
    import bandits.store as store

    monkeypatch.setattr(native, "_LINE_LIMIT", 64)
    monkeypatch.setattr(store, "_LINE_LIMIT", 64)
    path = tmp_path / "a.json"
    path.write_text(json.dumps([_trace("a"), _trace("b")]))
    assert _every_span_resolves(tmp_path, path, "langfuse") >= 2


def test_streamed_array_redaction_counts_file_lines_and_archives_the_rest_verbatim(
    tmp_path,
) -> None:
    path = tmp_path / "a.json"
    text = json.dumps([_trace("a"), _trace("b")], indent=1)
    path.write_text(text)
    corpus = load_corpus(path, "langfuse")
    expected = {
        f"{path}:{number}"
        for number, line in enumerate(text.splitlines(), start=1)
        if "someone@mailhost.test" in line
    }
    found = {i.location for i in corpus.issues if i.kind == "redaction"}
    assert found == expected
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path)).artifact_id
    archived = (store._dir(artifact) / "source" / "000000.json").read_text()
    # Only the redacted values differ: same layout, same line count.
    assert archived.splitlines()[0] == "["
    assert len(archived.splitlines()) == len(text.splitlines())
    assert "someone@mailhost.test" not in archived


def test_show_issues_counts_redactions_instead_of_listing_each(tmp_path) -> None:
    from bandits.inspect import issue_rows as _issue_rows

    path = tmp_path / "a.jsonl"
    path.write_text("\n".join(json.dumps(_trace(t)) for t in "abc"))
    corpus = load_corpus(path, "langfuse")
    hidden = [issue for issue in corpus.issues if issue.kind == "redaction"]
    others = [issue for issue in corpus.issues if issue.kind != "redaction"]
    assert len(hidden) > 3
    rows = _issue_rows(corpus.issues, all_redactions=False)
    assert rows[0] == (
        "redaction",
        "3 location(s) in a.jsonl",
        f"{hidden[0].detail} ({len(hidden)} value(s); --all-redactions lists each)",
    )
    assert rows[1:] == [(i.kind, i.location or "", i.detail) for i in others]
    assert len(_issue_rows(corpus.issues, all_redactions=True)) == len(corpus.issues)
