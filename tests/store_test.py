from __future__ import annotations

import time

import pytest

from bandits.store import ArtifactConflict, ArtifactStore, DerivedStore, compute_artifact_id
from bandits.traces import TraceCorpus


def _corpus(source: str = "otlp") -> TraceCorpus:
    return TraceCorpus(source=source, traces=(), issues=())


def test_write_then_read_round_trips(tmp_path) -> None:
    store = ArtifactStore(tmp_path / ".bandits")
    corpus = _corpus()

    envelope = store.write(corpus, source_path="traces.jsonl")

    assert envelope.artifact_id == compute_artifact_id(corpus)
    assert store.read(envelope.artifact_id) == corpus


def test_rewriting_identical_corpus_is_a_noop(tmp_path) -> None:
    store = ArtifactStore(tmp_path / ".bandits")
    corpus = _corpus()

    first = store.write(corpus, source_path="a.jsonl")
    second = store.write(corpus, source_path="a.jsonl")

    assert first == second


def test_conflicting_content_at_the_same_id_raises(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path / ".bandits")
    monkeypatch.setattr("bandits.store.compute_artifact_id", lambda corpus: "corpus-forced")

    store.write(_corpus(source="otlp"), source_path="a.jsonl")
    with pytest.raises(ArtifactConflict):
        store.write(_corpus(source="chat-json"), source_path="b.jsonl")


def test_list_orders_newest_first(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path / ".bandits")

    monkeypatch.setattr("bandits.store.compute_artifact_id", lambda corpus: "corpus-first")
    store.write(_corpus(source="otlp"), source_path="a.jsonl")
    time.sleep(0.01)
    monkeypatch.setattr("bandits.store.compute_artifact_id", lambda corpus: "corpus-second")
    store.write(_corpus(source="chat-json"), source_path="b.jsonl")

    envelopes = store.list()
    assert [e.artifact_id for e in envelopes] == ["corpus-second", "corpus-first"]


def test_list_on_empty_project_is_empty(tmp_path) -> None:
    store = ArtifactStore(tmp_path / ".bandits")
    assert store.list() == []


def test_derived_artifact_records_its_parent(tmp_path) -> None:
    store = DerivedStore(tmp_path / ".bandits")

    envelope = store.write(
        "analysis-1",
        kind="analysis",
        parent_artifact_id="corpus-abc",
        payload=b'{"tasks": []}',
        summary={"tasks": 0},
    )

    assert envelope.parent_artifact_id == "corpus-abc"
    assert store.read_payload("analysis-1") == b'{"tasks": []}'


def test_rewriting_a_derived_artifact_with_different_content_raises(tmp_path) -> None:
    store = DerivedStore(tmp_path / ".bandits")
    store.write("analysis-1", kind="analysis", parent_artifact_id="corpus-abc", payload=b"{}")

    with pytest.raises(ArtifactConflict):
        store.write(
            "analysis-1", kind="analysis", parent_artifact_id="corpus-abc", payload=b'{"a": 1}'
        )


def test_derived_artifacts_are_not_listed_as_corpora(tmp_path) -> None:
    """A derived artifact must never be mistaken for source evidence."""
    project = tmp_path / ".bandits"
    ArtifactStore(project).write(_corpus(), source_path="a.jsonl")
    DerivedStore(project).write(
        "analysis-1", kind="analysis", parent_artifact_id="corpus-abc", payload=b"{}"
    )

    assert [e.artifact_id for e in DerivedStore(project).list(kind="analysis")] == ["analysis-1"]
    assert all(e.artifact_id.startswith("corpus-") for e in ArtifactStore(project).list())
