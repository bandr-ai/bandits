"""Trace shapes: order- and repeat-independent structure signatures."""

from __future__ import annotations

from bandits.ingest.discovery import discover_requests
from bandits.ingest.otlp_standard import load_otlp_standard, shape_id
from bandits.ingest.report import IngestReport
from bandits.traces import WorkflowDeclaration
from tests.ingest.otlp_standard_test import _request, _span, _write
from tests.ingest.otlp_workflow_test import _workflow


class _D:
    def __init__(self, span_id, parent, name, role="step", label="kind=CHAIN"):
        self.span_id, self.parent_id, self.name, self.role, self.label = (
            span_id,
            parent,
            name,
            role,
            label,
        )


def _shape(*spans: _D) -> str:
    return shape_id({s.span_id: s for s in spans})


def test_repeated_children_collapse() -> None:
    one = _shape(_D("r", None, "run"), _D("a", "r", "call", "model"))
    three = _shape(
        _D("r", None, "run"),
        *(_D(f"a{i}", "r", "call", "model") for i in range(3)),
    )
    assert one == three


def test_sibling_order_does_not_matter() -> None:
    first = _shape(_D("r", None, "run"), _D("a", "r", "search"), _D("b", "r", "rank"))
    second = _shape(_D("b", "r", "rank"), _D("r", None, "run"), _D("a", "r", "search"))
    assert first == second


def test_different_nesting_is_a_different_shape() -> None:
    flat = _shape(_D("r", None, "run"), _D("a", "r", "search"), _D("b", "r", "rank"))
    nested = _shape(_D("r", None, "run"), _D("a", "r", "search"), _D("b", "a", "rank"))
    assert flat != nested
    renamed = _shape(_D("r", None, "run"), _D("a", "r", "search"), _D("b", "r", "rerank"))
    assert flat != renamed


def test_forest_roots_share_one_virtual_root_without_their_absent_parents() -> None:
    forest = _shape(_D("a", "gone-1", "graph"), _D("b", "gone-2", "app"))
    other_parents = _shape(_D("b", "elsewhere", "app"), _D("a", None, "graph"))
    assert forest == other_parents
    assert forest != _shape(_D("a", None, "graph"), _D("b", "a", "app"))


def test_loader_reports_shapes_and_discovery_agrees(tmp_path) -> None:
    traces = [_workflow(), _workflow()]
    for i, spans in enumerate(traces):
        for span in spans:
            span["traceId"] = str(i + 1) * 32
    traces[1].append(
        _span("extra", "late", {"openinference.span.kind": "CHAIN"}, parent="graph", trace="2" * 32)
    )
    path = _write(tmp_path / "wf.jsonl", *(_request(t) for t in traces))
    report = IngestReport()
    load_otlp_standard(
        path, workflow=WorkflowDeclaration(task_fields=("input.query",)), report=report
    )
    assert len(report.shapes) == 2
    assert {s.task_status["declared"] for s in report.shapes.values()} == {1}
    assert {s.model_calls for s in report.shapes.values()} == {3}
    assert {t.shape_id for t in discover_requests(path, "otlp-std").traces} == set(report.shapes)
    saved = report.as_dict()
    assert saved["shape_count"] == 2 and len(saved["shapes"]) == 2
    assert report.shape_lines()[0].split()[1] == "50%"


def test_only_the_top_five_shapes_are_printed() -> None:
    report = IngestReport()
    for i in range(7):
        report.add_shape(f"shape{i}", f"t{i}", 1, None)
    lines = report.shape_lines()
    assert len(lines) == 6 and lines[-1] == "+2 more shapes"
