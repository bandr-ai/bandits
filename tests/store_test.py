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
    monkeypatch.setattr("bandits.store._id_of", lambda data: "corpus-forced")

    store.write(_corpus(source="otlp"), source_path="a.jsonl")
    with pytest.raises(ArtifactConflict):
        store.write(_corpus(source="chat-json"), source_path="b.jsonl")


def test_list_orders_newest_first(tmp_path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path / ".bandits")

    monkeypatch.setattr("bandits.store._id_of", lambda data: "corpus-first")
    store.write(_corpus(source="otlp"), source_path="a.jsonl")
    time.sleep(0.01)
    monkeypatch.setattr("bandits.store._id_of", lambda data: "corpus-second")
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


def test_envelope_records_counts_and_code_version(tmp_path) -> None:
    from bandits.traces import TraceIssue, WorkflowDeclaration

    corpus = TraceCorpus(
        source="otlp-std",
        traces=(),
        issues=(
            TraceIssue(kind="redaction", detail="x"),
            TraceIssue(kind="malformed_span", detail="y"),
        ),
        workflow=WorkflowDeclaration(),
    )
    envelope = ArtifactStore(tmp_path / ".bandits").write(
        corpus, source_path="a.jsonl", problem_count=1
    )
    assert envelope.schema_version == 2
    assert (envelope.issue_count, envelope.problem_count, envelope.redaction_count) == (2, 1, 1)
    assert envelope.derivation_version == WorkflowDeclaration().derivation_version
    assert envelope.bandits_version
    # Run from this checkout, so the commit is this repo's.
    assert envelope.git_commit is not None and len(envelope.git_commit) == 40


def test_envelope_derivation_is_none_without_a_workflow(tmp_path) -> None:
    envelope = ArtifactStore(tmp_path / ".bandits").write(_corpus(), source_path="a.jsonl")
    assert envelope.derivation_version is None and envelope.problem_count is None


def test_an_old_envelope_still_validates() -> None:
    from bandits.store import ArtifactEnvelope

    old = ArtifactEnvelope.model_validate_json(
        '{"schema_version":1,"artifact_id":"corpus-x","created_at":"t","source_path":"p",'
        '"source":"otlp","trace_count":0,"span_count":0,"issue_count":3}'
    )
    assert old.derivation_version is None and old.git_commit is None


def test_code_version_ignores_another_projects_checkout(tmp_path, monkeypatch) -> None:
    import subprocess

    import bandits
    from bandits import store

    package = tmp_path / "project" / ".venv" / "bandits"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (tmp_path / "project" / "pyproject.toml").write_text('[project]\nname = "other"\n')
    subprocess.run(["git", "init", "-q", str(tmp_path / "project")], check=True)
    monkeypatch.setattr(bandits, "__file__", str(package / "__init__.py"))
    store.code_version.cache_clear()
    try:
        assert store.code_version() == (bandits.__version__, None, None)
    finally:
        store.code_version.cache_clear()


def test_code_version_reads_bandits_own_checkout(tmp_path, monkeypatch) -> None:
    import subprocess

    import bandits
    from bandits import store

    repo = tmp_path / "checkout"
    package = repo / "bandits"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (repo / "pyproject.toml").write_text('[project]\nname = "bandits"\n')
    (repo / "notes.md").write_text("")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    monkeypatch.setattr(bandits, "__file__", str(package / "__init__.py"))
    try:
        store.code_version.cache_clear()
        _, commit, dirty = store.code_version()
        assert commit is not None and dirty is False
        (repo / "notes.md").write_text("elsewhere")  # outside the package: not dirty
        store.code_version.cache_clear()
        assert store.code_version()[2] is False
        (package / "__init__.py").write_text("# changed")
        store.code_version.cache_clear()
        assert store.code_version()[2] is True
    finally:
        store.code_version.cache_clear()


def test_write_serializes_the_corpus_once(tmp_path, monkeypatch) -> None:
    calls = []
    real = TraceCorpus.model_dump_json

    def counted(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TraceCorpus, "model_dump_json", counted)
    envelope = ArtifactStore(tmp_path / ".bandits").write(_corpus(), source_path="a.jsonl")
    assert len(calls) == 1
    assert envelope.artifact_id == compute_artifact_id(_corpus())
