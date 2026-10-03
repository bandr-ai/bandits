"""Streaming storage must preserve the materialized contract and health gate."""

import json
from pathlib import Path

import pytest

from bandits.ingest import iter_corpus, load_corpus
from bandits.ingest.health import Health, check, collect, finish
from bandits.ingest.report import IngestReport
from bandits.redact import DEFAULT_RULESET
from bandits.store import ArtifactStore, StreamingWrite, compute_artifact_id
from bandits.traces import TraceCorpus, WorkflowDeclaration
from tests.cli_test import plain

FIXTURES = Path(__file__).parents[1] / "fixtures" / "upstream"


@pytest.mark.parametrize(
    "source,fixture",
    [
        ("langfuse", "langfuse/agno-2025-06-11.trace.json"),
        ("phoenix", "phoenix/sdk-server-getspans.json"),
        ("otlp-std", "interlingua/openinference.otlp.json"),
        ("otlp-std", "interlingua/openllmetry.otlp.json"),
        ("otlp-std", "interlingua/braintrust.json"),
        ("otlp-std", "interlingua/litellm.json"),
        ("otlp-std", "interlingua/vercel.json"),
        ("otlp-std", "interlingua/langchain.json"),
        ("otlp-std", "interlingua/openllmetry-legacy.json"),
    ],
)
def test_stream_preserves_bytes_ids_health_and_accounting(tmp_path, source, fixture):
    path = FIXTURES / fixture
    workflow = WorkflowDeclaration()
    expected_report = IngestReport()
    expected = load_corpus(path, source, workflow=workflow, report=expected_report)
    expected = expected.replace(control_markers=("marker α",))
    report = IngestReport()
    health = Health()
    store = ArtifactStore(tmp_path / ".bandits")
    streamed = []
    with StreamingWrite(store, source) as writer:
        for item in iter_corpus(path, source, workflow=workflow, report=report):
            if isinstance(item, TraceCorpus):
                footer = item.replace(control_markers=expected.control_markers)
                writer.finish(footer)
            else:
                streamed.append(item)
                collect(health, item, source, workflow=True)
                writer.add(item)
        assert tuple(streamed) == expected.traces
        assert [(i.kind, i.detail) for i in footer.issues] == [
            (i.kind, i.detail) for i in expected.issues
        ]
        expected = footer.replace(traces=tuple(streamed))
        assert (
            writer.directory / "corpus.json"
        ).read_bytes() == expected.model_dump_json().encode()
        assert writer.artifact_id == compute_artifact_id(expected)
        assert finish(health, footer) == check(expected, source)
        actual_counts, expected_counts = report.as_dict(), expected_report.as_dict()
        actual_counts.pop("evidence_seconds")
        expected_counts.pop("evidence_seconds")
        assert actual_counts == expected_counts
        envelope = writer.commit(source_path=str(path), problem_count=0, report=report.as_dict())
    assert store.read(envelope.artifact_id) == expected
    assert envelope.trace_count == len(expected.traces)
    assert envelope.span_count == sum(len(t.spans) for t in expected.traces)
    assert not list(tmp_path.glob(".bandits-ingest-*"))


def test_abandoned_stream_publishes_nothing_and_cleans_up(tmp_path):
    store = ArtifactStore(tmp_path / ".bandits")
    with pytest.raises(ValueError), StreamingWrite(store, "langfuse"):
        raise ValueError("health failed")
    assert not (tmp_path / ".bandits").exists()
    assert not list(tmp_path.glob(".bandits-ingest-*"))


def test_stream_commit_rejects_changed_source_without_publishing(tmp_path):
    original = FIXTURES / "phoenix/sdk-server-getspans.json"
    source = tmp_path / "export.json"
    source.write_bytes(original.read_bytes())
    store = ArtifactStore(tmp_path / ".bandits")
    with StreamingWrite(store, "phoenix") as writer:
        for item in iter_corpus(source, "phoenix", workflow=WorkflowDeclaration()):
            if isinstance(item, TraceCorpus):
                writer.finish(item)
            else:
                writer.add(item)
        source.write_text("{}")
        with pytest.raises(ValueError, match="source file changed"):
            writer.commit(source_path=str(source), problem_count=0, report=None)
    assert store.list() == []
    assert not list(tmp_path.glob(".bandits-ingest-*"))


def test_stream_deduplicates_materialized_artifact(tmp_path):
    source = FIXTURES / "phoenix/sdk-server-getspans.json"
    corpus = load_corpus(source, "phoenix", workflow=WorkflowDeclaration())
    store = ArtifactStore(tmp_path / ".bandits")
    existing = store.write(corpus, source_path=str(source))
    with StreamingWrite(store, corpus.source) as writer:
        for trace in corpus.traces:
            writer.add(trace)
        writer.finish(corpus.replace(traces=()))
        assert writer.commit(source_path=str(source), problem_count=0, report=None) == existing
    assert len(store.list()) == 1


def _stream(store, corpus, source_path, report):
    with StreamingWrite(store, corpus.source) as writer:
        for trace in corpus.traces:
            writer.add(trace)
        writer.finish(corpus.replace(traces=()))
        return writer.commit(source_path=source_path, problem_count=0, report=report)


def test_stream_fills_in_a_missing_archive_and_report(tmp_path):
    source = FIXTURES / "phoenix/sdk-server-getspans.json"
    corpus = load_corpus(source, "phoenix", workflow=WorkflowDeclaration())
    store = ArtifactStore(tmp_path / ".bandits")
    # A library save under a logical name leaves neither archive nor report.
    first = store.write(corpus, source_path="logical-name")
    artifact = store._dir(first.artifact_id)
    assert not (artifact / "source-manifest.json").exists()
    assert _stream(store, corpus, str(source), {"spans_seen": 2}) == first
    assert store.read_report(first.artifact_id) == {"spans_seen": 2}
    for span in (s for t in corpus.traces for s in t.spans):
        pointer = json.loads(span.attributes["bandits.source.record"])
        assert store.read_native_record(first.artifact_id, pointer)
    assert not list(tmp_path.glob(".bandits-ingest-*"))


def test_stream_never_archives_a_changed_source_into_an_existing_artifact(tmp_path):
    source = tmp_path / "export.json"
    source.write_bytes((FIXTURES / "phoenix/sdk-server-getspans.json").read_bytes())
    corpus = load_corpus(source, "phoenix", workflow=WorkflowDeclaration())
    store = ArtifactStore(tmp_path / ".bandits")
    first = store.write(corpus, source_path="logical-name")
    source.write_text("{}")
    with pytest.raises(ValueError, match="source file changed"):
        _stream(store, corpus, str(source), None)
    assert not (store._dir(first.artifact_id) / "source-manifest.json").exists()
    assert not (store._dir(first.artifact_id) / "source").exists()


def _otlp_record(trace_id, span_id, parent=None, *, model=False, text="hello"):
    attrs = {
        "openinference.span.kind": "LLM" if model else "CHAIN",
        "input.value": json.dumps([{"role": "user", "content": text}])
        if model
        else json.dumps({"query": text}),
        "output.value": json.dumps([{"role": "assistant", "content": "finished"}])
        if model
        else json.dumps({"answer": "finished"}),
    }
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": "model" if model else "run",
        "startTimeUnixNano": "1767225600000000000",
        "endTimeUnixNano": "1767225601000000000",
        "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in attrs.items()],
    }
    if parent:
        span["parentSpanId"] = parent
    return {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}


def _native_record(source, trace_id, span_id, parent=None, *, model=False, text="hello"):
    if source == "langsmith":
        return {
            "trace_id": trace_id,
            "id": span_id,
            "parent_run_id": parent,
            "name": "model" if model else "run",
            "run_type": "llm" if model else "chain",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            "inputs": {"messages": [{"role": "user", "content": text}]}
            if model
            else {"query": text},
            "outputs": {"messages": [{"role": "assistant", "content": "finished"}]}
            if model
            else {"answer": "finished"},
        }
    return {
        "context": {"trace_id": trace_id, "span_id": span_id},
        "parent_id": parent,
        "name": "model" if model else "run",
        "span_kind": "LLM" if model else "CHAIN",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {
            "input.value": json.dumps([{"role": "user", "content": text}])
            if model
            else json.dumps({"query": text}),
            "output.value": json.dumps([{"role": "assistant", "content": "finished"}])
            if model
            else json.dumps({"answer": "finished"}),
        },
    }


def _materialized_native_reference(path, source, workflow):
    """Independent pre-streaming normalization through the existing OTLP loader."""
    import hashlib

    from bandits.ingest.native import NativeConversion
    from bandits.ingest.otlp_standard import load_otlp_standard
    from bandits.ingest.report import aggregate_issues

    conversion = NativeConversion(path, source, DEFAULT_RULESET)
    report = IngestReport()
    traces, issues = [], []
    for chunk in conversion.chunks():
        corpus = load_otlp_standard(
            chunk, workflow=workflow, report=report, defer_aggregate_issues=True
        )
        traces.extend(corpus.traces)
        issues.extend(corpus.issues)
    total = conversion.finish(report)
    issues = conversion.issues + issues + aggregate_issues(total, str(path), workflow=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return TraceCorpus(
        source=source,
        traces=tuple(t.replace(source=source, source_digest=digest) for t in traces),
        issues=tuple(issues),
        workflow=workflow,
        redaction_ruleset=DEFAULT_RULESET.name,
    ), total


@pytest.mark.parametrize("source", ["otlp-std", "langsmith", "phoenix"])
def test_interleaved_children_before_parents_preserve_legacy_bytes(tmp_path, source):
    records = []
    factory = (
        _otlp_record if source == "otlp-std" else lambda *a, **k: _native_record(source, *a, **k)
    )
    # Lexical trace order intentionally differs from first appearance.
    for trace in ["f" * 32, "a" * 32]:
        records.append(factory(trace, "2" * 16, "1" * 16, model=True))
    for trace in ["a" * 32, "f" * 32]:
        records.append(factory(trace, "1" * 16))
    path = tmp_path / "interleaved.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    workflow = WorkflowDeclaration(task_fields=("input.query",), delivered_field="output.answer")
    if source == "otlp-std":
        expected_report = IngestReport()
        expected = load_corpus(path, source, workflow=workflow, report=expected_report)
    else:
        expected, expected_report = _materialized_native_reference(path, source, workflow)
    report = IngestReport()
    traces = []
    for item in iter_corpus(path, source, workflow=workflow, report=report):
        if isinstance(item, TraceCorpus):
            actual = item.replace(traces=tuple(traces))
        else:
            traces.append(item)
    assert actual.model_dump_json() == expected.model_dump_json()
    assert all(t.task == "hello" and t.request.delivered == "finished" for t in traces)
    actual_counts, expected_counts = report.as_dict(), expected_report.as_dict()
    actual_counts.pop("evidence_seconds")
    expected_counts.pop("evidence_seconds")
    assert actual_counts == expected_counts


def test_otlp_directory_duplicates_malformed_and_redactions_preserve_bytes(tmp_path):
    from bandits.ingest.otlp_standard import load_otlp_standard

    export = tmp_path / "export"
    export.mkdir()
    child = _otlp_record(
        "f" * 32, "2" * 16, "1" * 16, model=True, text="first@sample.invalid\nsecond@sample.invalid"
    )
    child["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"].append(
        {"key": "note", "value": {"stringValue": "one@sample.invalid\ntwo@sample.invalid"}}
    )
    other = _otlp_record("a" * 32, "3" * 16, model=True)
    (export / "one.jsonl").write_text(
        "\n" + json.dumps(child) + "\ninvalid json\n" + json.dumps(other) + "\n"
    )
    parent = _otlp_record("f" * 32, "1" * 16, text="first@sample.invalid\nsecond@sample.invalid")
    malformed = {"resourceSpans": [{"scopeSpans": [{"spans": [None]}]}]}
    (export / "two.jsonl").write_text(
        json.dumps(parent) + "\n" + json.dumps(child) + "\n" + json.dumps(malformed) + "\n"
    )
    workflow = WorkflowDeclaration(task_fields=("input.query",), delivered_field="output.answer")
    reference_report = IngestReport()
    expected = load_otlp_standard(export, workflow=workflow, report=reference_report)
    report = IngestReport()
    traces = []
    for item in iter_corpus(export, "otlp-std", workflow=workflow, report=report):
        if isinstance(item, TraceCorpus):
            actual = item.replace(traces=tuple(traces))
        else:
            traces.append(item)
    assert actual.model_dump_json() == expected.model_dump_json()
    assert any(i.kind == "redaction" for i in actual.issues)
    assert any(i.detail.startswith("redacted decoded") for i in actual.issues)
    assert report.buckets["duplicate"] == 1
    assert report.buckets["malformed_span"] == 1
    assert report.unreadable_items["malformed_json"] == 1
    assert not report.accounting_errors


@pytest.mark.parametrize("source", ["otlp-std", "langsmith", "phoenix"])
def test_cli_discovery_and_archive_never_read_entire_jsonl(tmp_path, monkeypatch, source):
    from typer.testing import CliRunner

    from bandits.cli import app

    factory = (
        _otlp_record if source == "otlp-std" else lambda *a, **k: _native_record(source, *a, **k)
    )
    path = tmp_path / "export.jsonl"
    path.write_text(
        "\n"
        + json.dumps(factory("a" * 32, "2" * 16, "1" * 16, model=True))
        + "\n"
        + json.dumps(factory("a" * 32, "1" * 16))
        + "\n"
    )
    original = Path.read_bytes

    def guarded(file):
        if file.suffix == ".jsonl":
            raise AssertionError("the JSONL export was read into one byte buffer")
        return original(file)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = CliRunner().invoke(
        app, ["ingest", str(path), "--source", source, "--project", str(tmp_path / "project")]
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    assert "problems: none" in plain(result.stdout)
    store = ArtifactStore(tmp_path / "project" / ".bandits")
    (envelope,) = store.list()
    corpus = store.read(envelope.artifact_id)
    assert corpus.traces[0].task == "hello"
    assert store.read_report(envelope.artifact_id)["spans_seen"] == 2
    if source != "otlp-std":
        pointer = corpus.traces[0].spans[0].attributes["bandits.source.record"]
        assert json.loads(pointer)["line"] == 2
        assert store.read_native_record(envelope.artifact_id, pointer)["name"] == "model"


@pytest.mark.parametrize("content", ["\ninvalid\n", "\n{}\n", "\n[]\n", "\n42\n"])
def test_single_record_and_invalid_first_line_locations_match_legacy(tmp_path, content):
    path = tmp_path / "input.jsonl"
    path.write_text(content)
    expected = load_corpus(path, "otlp-std")
    traces = []
    for item in iter_corpus(path, "otlp-std"):
        if isinstance(item, TraceCorpus):
            actual = item.replace(traces=tuple(traces))
        else:
            traces.append(item)
    assert actual.model_dump_json() == expected.model_dump_json()


def test_source_changed_between_disk_passes_is_rejected_and_spool_removed(tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    original_text = json.dumps(_otlp_record("a" * 32, "2" * 16, model=True)) + "\n"
    path.write_text(original_text)
    original_open = Path.open
    reads = 0

    def changed(file, mode="r", *args, **kwargs):
        nonlocal reads
        if file == path and mode == "rb":
            reads += 1
            if reads == 2:
                with original_open(file, "w") as output:
                    output.write(original_text.replace("hello", "changed"))
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(ValueError, match="source file changed during ingest"):
        list(iter_corpus(path, "otlp-std", scratch_dir=tmp_path))
    assert not list(tmp_path.glob(".bandits-read-*"))


@pytest.mark.parametrize("source", ["langsmith", "phoenix"])
def test_native_jsonl_array_records_are_streamed_and_pointers_resolve(tmp_path, source):
    from typer.testing import CliRunner

    from bandits.cli import app

    path = tmp_path / "arrays.jsonl"
    child = _native_record(source, "a" * 32, "2" * 16, "1" * 16, model=True)
    root = _native_record(source, "a" * 32, "1" * 16)
    path.write_text(json.dumps([child]) + "\n" + json.dumps([root]) + "\n")
    result = CliRunner().invoke(
        app, ["ingest", str(path), "--source", source, "--project", str(tmp_path)]
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    store = ArtifactStore(tmp_path / ".bandits")
    (envelope,) = store.list()
    trace = store.read(envelope.artifact_id).traces[0]
    assert trace.task == "hello"
    pointer = trace.spans[0].attributes["bandits.source.record"]
    assert json.loads(pointer)["line"] == 1
    assert json.loads(pointer)["index"] == 0
    assert store.read_native_record(envelope.artifact_id, pointer) == child
