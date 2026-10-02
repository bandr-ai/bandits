"""Streaming storage must preserve the materialized contract and health gate."""

from pathlib import Path

import pytest

from bandits.ingest import iter_corpus, load_corpus
from bandits.ingest.health import Health, check, collect, finish
from bandits.ingest.report import IngestReport
from bandits.store import ArtifactStore, StreamingWrite, compute_artifact_id
from bandits.traces import TraceCorpus, WorkflowDeclaration

FIXTURES = Path(__file__).parents[1] / "fixtures" / "upstream"


@pytest.mark.parametrize(
    "source,fixture",
    [
        ("langfuse", "langfuse/agno-2025-06-11.trace.json"),
        ("phoenix", "phoenix/sdk-server-getspans.json"),
        ("otlp-std", "interlingua/openinference.otlp.json"),
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
