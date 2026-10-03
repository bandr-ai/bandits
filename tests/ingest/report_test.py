"""Record accounting, skipped native observations and absent parents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import load_corpus
from bandits.ingest.health import check
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.ingest.report import KEPT, IngestReport
from bandits.store import ArtifactStore
from bandits.traces import WorkflowDeclaration
from tests.ingest.otlp_standard_test import _request, _span, _write
from tests.ingest.otlp_workflow_test import _chain, _gen, _workflow

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "upstream"
WORKFLOW = WorkflowDeclaration(task_fields=("input.query",))


def _observation(oid: str, kind: str = "SPAN", *, parent: str | None = None, **fields) -> dict:
    return {
        "id": oid,
        "type": kind,
        "name": oid,
        "parentObservationId": parent,
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": {"query": "Where is my parcel?"},
        "output": "It ships today.",
        **fields,
    }


def _langfuse(tmp_path: Path, *traces: dict, name: str = "lf.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(t) + "\n" for t in traces))
    return path


def _sums(report: IngestReport) -> None:
    assert sum(report.buckets.values()) == report.spans_seen
    assert not report.accounting_errors


@pytest.mark.parametrize(
    ("fixture", "source"),
    [
        *(
            (f"interlingua/{p.name}", "otlp-std")
            for p in sorted((FIXTURES / "interlingua").glob("*.json"))
        ),
        ("langfuse/agno-2025-06-11.trace.json", "langfuse"),
        ("phoenix/sdk-server-getspans.json", "phoenix"),
    ],
)
@pytest.mark.parametrize("workflow", [None, WORKFLOW])
@pytest.mark.parametrize("pipeline_steps", [True, False])
def test_every_span_lands_in_one_bucket(fixture, source, workflow, pipeline_steps) -> None:
    report = IngestReport()
    load_corpus(
        FIXTURES / fixture, source, workflow=workflow, pipeline_steps=pipeline_steps, report=report
    )
    assert report.spans_seen > 0
    _sums(report)


def test_otlp_counts_match_raw_spans(tmp_path) -> None:
    spans = _workflow()
    report = IngestReport()
    load_otlp_standard(
        _write(tmp_path / "wf.jsonl", _request(spans)), workflow=WORKFLOW, report=report
    )
    assert report.spans_seen == len(spans)
    assert report.buckets["invocation"] == 1 and report.buckets["container"] == 1
    assert report.buckets["model"] == 3
    assert report.dropped == 0
    _sums(report)


def test_a_running_observation_is_counted_and_reported(tmp_path) -> None:
    running = _observation("m1", "GENERATION", parent="inv", endTime=None)
    child = _observation("t1", "TOOL", parent="m1")
    report = IngestReport()
    path = _langfuse(tmp_path, {"id": "t", "observations": [_observation("inv"), running, child]})
    corpus = load_corpus(path, "langfuse", workflow=WORKFLOW, report=report)
    assert report.spans_seen == 3
    assert report.buckets["unconvertible"] == 1
    _sums(report)
    (issue,) = [i for i in corpus.issues if i.kind == "unconvertible_observation"]
    assert "end time missing (still running?)" in issue.detail and "m1" in issue.detail
    assert issue.location == str(path)
    assert not [i for i in corpus.issues if i.kind == "unrepresented_record"]
    # Data loss is a warning, not a notice.
    assert any("unconvertible_observation" in w for w in check(corpus, "langfuse").warnings)


def test_skipped_observations_and_their_children_are_counted(tmp_path) -> None:
    nameless = {**_observation("x"), "id": None, "children": [_observation("lost")]}
    observations = [_observation("inv"), _observation("inv"), "not-an-object", nameless]
    report = IngestReport()
    path = _langfuse(tmp_path, {"id": "t", "observations": observations})
    corpus = load_corpus(path, "langfuse", workflow=WORKFLOW, report=report)
    assert report.spans_seen == 5
    assert report.buckets["duplicate_native"] == 1
    assert report.buckets["unconvertible"] == 3  # not an object, empty id, inside it
    _sums(report)
    assert sum(i.kind == "unconvertible_observation" for i in corpus.issues) == 1


def test_langsmith_child_runs_that_cannot_convert_are_counted(tmp_path) -> None:
    def run(rid: str, **fields) -> dict:
        return {
            "id": rid,
            "run_type": "chain",
            "name": rid,
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            **fields,
        }

    model = run("m", run_type="llm", inputs={"text": "hi"}, outputs={"text": "hello"})
    root = run("root", child_runs=[model, model, 7, run("open", end_time=None)])
    path = tmp_path / "ls.json"
    path.write_text(json.dumps(root))
    report = IngestReport()
    load_corpus(path, "langsmith", report=report)
    assert report.buckets["duplicate_native"] == 1
    assert report.buckets["unconvertible"] == 2
    _sums(report)


def test_non_object_resource_entries_are_unreadable_items(tmp_path) -> None:
    request = _request([_span("m", "call", {"gen_ai.operation.name": "chat"})])
    request["resourceSpans"].append("garbage")
    request["resourceSpans"][0]["scopeSpans"].append(3)
    report = IngestReport()
    corpus = load_otlp_standard(
        _write(tmp_path / "x.jsonl", request, {"no": "spans"}), report=report
    )
    assert report.unreadable_items == {"malformed_record": 3}
    assert sum(i.kind == "malformed_record" for i in corpus.issues) == 3
    _sums(report)


def test_workflow_steps_without_pipeline_steps_are_nodes_not_unrepresented(tmp_path) -> None:
    spans = _workflow() + [_chain("lookup", "lookup", parent="graph", at=5, output={"x": 1})]
    report = IngestReport()
    corpus = load_otlp_standard(
        _write(tmp_path / "wf.jsonl", _request(spans)),
        workflow=WORKFLOW,
        pipeline_steps=False,
        report=report,
    )
    assert "lookup" in {n.span_id for n in corpus.traces[0].workflow_nodes}
    assert not [i for i in corpus.issues if i.kind == "unrepresented_span"]
    assert report.buckets["pipeline_step"] == 0
    _sums(report)


def test_a_root_step_with_nothing_beneath_is_reported(tmp_path) -> None:
    spans = [
        _span("root", "idle", {"openinference.span.kind": "CHAIN"}),
        _span("other", "agent", {"openinference.span.kind": "AGENT"}, trace="1" * 32),
        _span("m", "call", {"gen_ai.operation.name": "chat"}, parent="other", trace="1" * 32),
    ]
    report = IngestReport()
    corpus = load_otlp_standard(_write(tmp_path / "x.jsonl", _request(spans)), report=report)
    assert report.buckets["empty_trace"] == 1  # the idle trace has nothing else
    assert report.buckets["step_with_calls"] == 1
    assert any(i.kind == "unrepresented_span" and "CHAIN" in i.detail for i in corpus.issues)
    _sums(report)


def test_summary_issues_are_issued_once_across_native_chunks(tmp_path) -> None:
    # 400 spans fill one chunk, so the second trace is read in another.
    def trace(tid: str, calls: int) -> dict:
        return {
            "id": tid,
            "observations": [_observation(f"{tid}-inv")]
            + [_observation(f"{tid}{i}", "GENERATION", parent=f"{tid}-inv") for i in range(calls)],
        }

    first, second, third = trace("a", 399), trace("b", 1), trace("a", 1)
    third["observations"][0]["id"] = "a-late-inv"
    third["observations"][1].update(id="a-late", parentObservationId="a-late-inv")
    path = _langfuse(tmp_path, first, second, third)
    report = IngestReport()
    corpus = load_corpus(path, "langfuse", workflow=WorkflowDeclaration(), report=report)
    (unresolved,) = [i for i in corpus.issues if i.kind == "task_unresolved"]
    assert unresolved.detail.startswith("3 workflow trace(s)")
    assert unresolved.location == str(path)
    assert report.split_trace_ids == 1
    assert sum(t.trace_id == corpus.traces[0].trace_id for t in corpus.traces) == 2
    assert any(i.kind == "trace_split_across_chunks" for i in corpus.issues)
    _sums(report)


def _three_top_steps() -> list[dict]:
    return [
        _span(
            "app",
            "app",
            {
                "langfuse.observation.type": "SPAN",
                "input.value": json.dumps({"payload": {"query": "q"}}),
            },
            parent="gone-1",
        ),
        _chain("graph", "graph", parent="gone-2", at=0, input={"question": "q"}),
        _gen("m_in", "STEP", "q", "draft", parent="graph", at=1),
        _gen("m_final", "FINAL", "q", "final answer", parent="gone-3", at=2),
    ]


def test_absent_parents_are_one_aggregated_notice(tmp_path) -> None:
    report = IngestReport()
    corpus = load_otlp_standard(
        _write(tmp_path / "wf.jsonl", _request(_three_top_steps())),
        workflow=WORKFLOW,
        report=report,
    )
    (issue,) = [i for i in corpus.issues if i.kind == "parent_not_exported"]
    assert issue.detail.startswith(
        "1 trace(s) have top-level steps whose parent is not in the export (max 3 per trace)"
    )
    assert "app(SPAN)→gone-1" in issue.detail and "m_final" not in issue.detail
    assert "FINAL(GENERATION)→gone-3" in issue.detail
    assert (report.traces_with_absent_parents, report.max_top_steps) == (1, 3)
    # A notice: the only warning is the ambiguity it causes.
    warnings = check(corpus, "otlp-std").warnings
    assert not any("parent_not_exported" in w for w in warnings)
    assert sum("ambiguous_invocation" in w for w in warnings) == 1


def test_a_container_is_looked_through_for_absent_parents(tmp_path) -> None:
    spans = _workflow()  # inv sits under a container that is itself the root
    report = IngestReport()
    load_otlp_standard(
        _write(tmp_path / "a.jsonl", _request(spans)), workflow=WORKFLOW, report=report
    )
    assert report.traces_with_absent_parents == 0
    spans[0]["parentSpanId"] = "never-exported"  # the container's own parent is absent
    report = IngestReport()
    load_otlp_standard(
        _write(tmp_path / "b.jsonl", _request(spans)), workflow=WORKFLOW, report=report
    )
    assert (report.traces_with_absent_parents, report.max_top_steps) == (1, 1)


def test_report_is_saved_beside_the_corpus_and_never_replaced(tmp_path) -> None:
    corpus = load_otlp_standard(
        _write(tmp_path / "wf.jsonl", _request(_workflow())), workflow=WORKFLOW
    )
    store = ArtifactStore(tmp_path / ".bandits")
    first = store.write(corpus, source_path="wf.jsonl", report={"spans_seen": 1})
    again = store.write(corpus, source_path="wf.jsonl", report={"spans_seen": 2})
    assert first.artifact_id == again.artifact_id
    assert store.read_report(first.artifact_id) == {"spans_seen": 1}


def test_buckets_are_kept_or_dropped() -> None:
    from bandits.ingest.report import BUCKETS

    assert KEPT < set(BUCKETS)


def test_evidence_links_are_counted_per_kind_with_the_most_in_one_trace(tmp_path) -> None:
    traces = [_workflow(), _workflow()]
    traces[1].append(_gen("m_more", "MORE", "q", "x" * 50, parent="graph", at=6))
    for i, spans in enumerate(traces):
        for span in spans:
            span["traceId"] = str(i + 1) * 32
    report = IngestReport()
    corpus = load_otlp_standard(
        _write(tmp_path / "wf.jsonl", *(_request(t) for t in traces)),
        workflow=WORKFLOW,
        report=report,
    )
    expected: dict[str, int] = {}
    for trace in corpus.traces:
        for link in trace.evidence:
            expected[link.kind] = expected.get(link.kind, 0) + 1
    assert dict(report.evidence_links) == expected
    assert report.evidence_max_per_trace["enclosing_result"] == max(
        sum(link.kind == "enclosing_result" for link in t.evidence) for t in corpus.traces
    )
    assert report.evidence_seconds >= 0
    assert report.evidence_line().startswith(f"{sum(expected.values())} links: ")
    assert report.as_dict()["evidence_links"] == expected
