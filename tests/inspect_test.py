"""The inspect page: written by ingest, rebuilt by ``bandits inspect``, self-contained."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from bandits.cli import app
from bandits.inspect import TraceSample
from bandits.store import ArtifactStore
from tests.cli_test import plain

FIXTURE = Path(__file__).parent / "fixtures/upstream/langfuse/agno-2025-06-11.trace.json"


def _data(page: Path) -> dict:
    html = page.read_text(encoding="utf-8")
    assert "http://" not in html.split("<script", 1)[0] and "src=" not in html
    found = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert found, "the page carries its data inline"
    return json.loads(found.group(1).replace("<\\/", "</"))


def test_ingest_writes_the_page_beside_the_corpus(tmp_path: Path) -> None:
    project = tmp_path / "project"
    result = CliRunner().invoke(app, ["ingest", str(FIXTURE), "--project", str(project)])
    assert result.exit_code == 0, result.output
    (envelope,) = ArtifactStore(project / ".bandits").list()
    page = project / ".bandits" / "artifacts" / envelope.artifact_id / "inspect.html"
    assert f"inspect:  {page}" in plain(result.output)
    data = _data(page)
    assert data["envelope"]["artifact_id"] == envelope.artifact_id
    assert data["report"]["spans_seen"] == 6 and data["problems"] == 0
    (trace,) = data["traces"]
    steps = {step["id"]: step for step in trace["steps"]}
    assert {"model", "invocation"} <= {step["type"] for step in steps.values()}
    # Every step but the run hangs off another step in the page: the tree is whole.
    roots = [s for s in steps.values() if s["parent"] not in steps]
    assert [s["type"] for s in roots] == ["invocation"]
    assert data["shapes"][0]["in_page"]


def test_dry_run_writes_no_page_and_inspect_rebuilds_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(FIXTURE), "--dry-run", "--project", str(project)])
    assert result.exit_code == 0 and "inspect:" not in plain(result.output)
    runner.invoke(app, ["ingest", str(FIXTURE), "--project", str(project)])
    (envelope,) = ArtifactStore(project / ".bandits").list()
    page = project / ".bandits" / "artifacts" / envelope.artifact_id / "inspect.html"
    page.unlink()
    result = runner.invoke(app, ["inspect", envelope.artifact_id, "--project", str(project)])
    assert result.exit_code == 0, result.output
    assert _data(page)["traces"]


def test_long_payloads_are_cut_and_script_text_cannot_end_the_data(tmp_path: Path) -> None:
    from bandits.inspect import _clip, render

    assert len(_clip("x" * 10_000)) < 4_100
    assert _clip({"docs": ["y" * 50] * 1000}).startswith('{"docs"')
    html = render({"envelope": {"artifact_id": "a"}, "text": "</script><b>"})
    assert html.count("</script>") == 2  # the page's own two scripts only


def test_sample_keeps_the_first_traces_and_one_per_new_layout() -> None:
    from bandits.ingest import load_corpus

    (trace,) = load_corpus(FIXTURE, "langfuse").traces
    sample = TraceSample(limit=1)
    sample.add(trace)
    other = trace.replace(trace_id="other")
    sample.add(other)  # same layout, past the limit: not kept
    assert list(sample.kept) == [trace.trace_id] and sample.seen == 2
    wanted = TraceSample(limit=1, wanted=["wanted"])
    wanted.add(trace)
    wanted.add(trace.replace(trace_id="wanted"))
    assert "wanted" in wanted.kept


def test_raw_vs_parsed_finds_every_value_and_flags_a_dropped_one(tmp_path: Path) -> None:
    from bandits.ingest import load_corpus
    from bandits.inspect import check_fidelity, missing_values

    corpus = load_corpus(FIXTURE, "langfuse")
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(FIXTURE)).artifact_id
    fidelity = check_fidelity(store, artifact, corpus.traces, "langfuse")
    stored = sum(len(t.spans) + len(t.workflow_nodes) for t in corpus.traces)
    assert fidelity["steps"] == stored and fidelity["values"] > 50
    assert fidelity["exact"] and fidelity["carried"] > 0
    assert fidelity["carried_wrong"] == 0, fidelity["mismatches"]

    raw = {"id": "o1", "usage": {"input": 313}, "name": "chat", "children": [{"x": "skip"}]}
    parsed = {"name": "chat", "attributes": {"id": "o1", "kept": '{"nested": "x"}'}}
    assert missing_values(raw, parsed) == ["usage.input"]
    parsed["attributes"]["bandits.unmapped"] = json.dumps({"usage": {"input": 313}})
    assert missing_values(raw, parsed) == []


def test_every_shape_is_kept_with_its_outline_and_question_field(tmp_path: Path) -> None:
    from bandits.ingest import load_corpus
    from bandits.ingest.report import IngestReport
    from bandits.traces import WorkflowDeclaration

    report = IngestReport()
    load_corpus(
        FIXTURE,
        "langfuse",
        workflow=WorkflowDeclaration(task_fields=("input.message",)),
        report=report,
    )
    (shape,) = report.as_dict()["shapes"]
    assert shape["task_paths"] == {"input.message": 1}
    assert shape["outline"][0].startswith("invocation · ") or " · " in shape["outline"][0]
    assert any("model · " in line for line in shape["outline"])


def test_the_exact_check_catches_a_carried_field_that_changed(tmp_path: Path) -> None:
    from bandits.ingest import load_corpus
    from bandits.inspect import check_fidelity

    corpus = load_corpus(FIXTURE, "langfuse")
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(FIXTURE)).artifact_id
    (trace,) = corpus.traces
    step = next(s for s in trace.spans if s.attributes.get("bandits.unmapped"))
    unmapped = step.attributes["bandits.unmapped"]
    unmapped = json.loads(unmapped) if isinstance(unmapped, str) else dict(unmapped)
    key = next(iter(unmapped))
    unmapped[key] = "changed"
    changed = step.replace(attributes={**step.attributes, "bandits.unmapped": unmapped})
    tampered = trace.replace(
        spans=tuple(changed if s.span_id == step.span_id else s for s in trace.spans)
    )
    fidelity = check_fidelity(store, artifact, [tampered], "langfuse")
    assert fidelity["carried_wrong"] == 1
    assert fidelity["mismatches"][0]["field"] == key
