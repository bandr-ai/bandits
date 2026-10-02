"""What one ingest saw and where each record went, kept beside the corpus.

Not part of ``corpus.json``: these are facts about the read (how many spans
the reader saw, which were dropped and why, how traces are shaped), not about
the episodes, and putting them in the corpus would change its id for no
analytic gain. The CLI prints the report and the store saves it as
``report.json`` outside the artifact id.

Every span the reader saw lands in exactly one bucket. A sum that does not
hold is a bandits bug, reported in ``accounting_errors`` and treated as fatal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bandits.traces import TraceIssue

BUCKETS = (
    # Dropped before or while decoding.
    "unconvertible",
    "duplicate_native",
    "malformed_span",
    "duplicate",
    "cyclic",
    "excluded",
    "empty_trace",
    # Kept, as an action, the request, or structure.
    "model",
    "tool",
    "invocation",
    "pipeline_step",
    "container",
    "node",
    "step_with_calls",
    "covered",
    # Not in the corpus.
    "root_step",
    "unrepresented",
)
"""In the order a span is assigned: the first that matches wins."""

KEPT = frozenset(
    {
        "model",
        "tool",
        "invocation",
        "pipeline_step",
        "container",
        "node",
        "step_with_calls",
        "covered",
    }
)
"""Buckets whose records the corpus represents (or, for a converter container,
the source archive keeps). Everything else is dropped and counted."""

EXAMPLES = 3
STEPS_PER_EXAMPLE = 5
SHAPES_KEPT = 50
"""Shapes saved in ``report.json``; the total count is always kept."""


@dataclass
class ShapeStats:
    traces: int = 0
    example_trace_id: str = ""
    model_calls: int = 0
    task_status: Counter[str] = field(default_factory=Counter)


@dataclass
class IngestReport:
    spans_seen: int = 0
    """Spans the decoder read (``otlp-std``), or observations the native
    converters read, nested ones included."""

    buckets: Counter[str] = field(default_factory=Counter)
    unreadable_items: Counter[str] = field(default_factory=Counter)
    """Whole input items skipped before any span in them could be counted:
    a line that is not JSON, a record with no spans list, a native record of
    an unsupported shape. Separate from the span sum."""

    unconvertible_examples: list[str] = field(default_factory=list)
    accounting_errors: list[str] = field(default_factory=list)

    split_trace_ids: int = 0
    """Trace ids that came out of more than one native chunk, so the corpus
    holds several traces with that id."""

    traces_with_absent_parents: int = 0
    max_top_steps: int = 0
    absent_parent_examples: list[str] = field(default_factory=list)

    shapes: dict[str, ShapeStats] = field(default_factory=dict)
    """Trace structure (``otlp_standard.shape_id``) and what the traces of each
    shape yielded."""

    unmapped_shapes: int = 0
    """Traces whose shape the applied mapping was not confirmed on."""

    # Counted per chunk, issued once per ingest (see ``aggregate_issues``).
    unparsed: Counter[str] = field(default_factory=Counter)
    unrepresented: Counter[str] = field(default_factory=Counter)
    containers: Counter[str] = field(default_factory=Counter)
    excluded: Counter[str] = field(default_factory=Counter)
    task_status: Counter[str] = field(default_factory=Counter)
    excluded_by_mapping: Counter[str] = field(default_factory=Counter)
    override_not_applicable: Counter[str] = field(default_factory=Counter)

    def add_shape(
        self, shape: str, trace_id: str, model_calls: int, task_status: str | None
    ) -> None:
        stats = self.shapes.setdefault(shape, ShapeStats(example_trace_id=trace_id))
        stats.traces += 1
        stats.model_calls += model_calls
        if task_status is not None:
            stats.task_status[task_status] += 1

    def ranked_shapes(self) -> list[tuple[str, ShapeStats]]:
        return sorted(self.shapes.items(), key=lambda item: (-item[1].traces, item[0]))

    def merge(self, other: IngestReport) -> None:
        for shape, theirs in other.shapes.items():
            mine = self.shapes.setdefault(
                shape, ShapeStats(example_trace_id=theirs.example_trace_id)
            )
            mine.traces += theirs.traces
            mine.model_calls += theirs.model_calls
            mine.task_status.update(theirs.task_status)
        self.spans_seen += other.spans_seen
        for name in (
            "buckets",
            "unreadable_items",
            "unparsed",
            "unrepresented",
            "containers",
            "excluded",
            "task_status",
            "excluded_by_mapping",
            "override_not_applicable",
        ):
            getattr(self, name).update(getattr(other, name))
        self.unmapped_shapes += other.unmapped_shapes
        self.unconvertible_examples.extend(
            other.unconvertible_examples[: EXAMPLES - len(self.unconvertible_examples)]
        )
        self.accounting_errors.extend(other.accounting_errors)
        self.split_trace_ids += other.split_trace_ids
        self.traces_with_absent_parents += other.traces_with_absent_parents
        self.max_top_steps = max(self.max_top_steps, other.max_top_steps)
        self.absent_parent_examples.extend(
            other.absent_parent_examples[: EXAMPLES - len(self.absent_parent_examples)]
        )

    @property
    def dropped(self) -> int:
        return sum(n for bucket, n in self.buckets.items() if bucket not in KEPT)

    def summary(self) -> str:
        """``records: 53 seen → 13 model · 23 node · 0 dropped (unreadable items: 0)``."""
        kept = [f"{self.buckets[b]} {b}" for b in BUCKETS if b in KEPT and self.buckets[b]]
        lost = [f"{self.buckets[b]} {b}" for b in BUCKETS if b not in KEPT and self.buckets[b]]
        dropped = f"{self.dropped} dropped" + (f" ({', '.join(lost)})" if lost else "")
        unreadable = sum(self.unreadable_items.values())
        return (
            f"{self.spans_seen} seen → {' · '.join([*kept, dropped])} "
            f"(unreadable items: {unreadable})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "spans_seen": self.spans_seen,
            "buckets": dict(self.buckets),
            "dropped": self.dropped,
            "unreadable_items": dict(self.unreadable_items),
            "unconvertible_examples": self.unconvertible_examples,
            "accounting_errors": self.accounting_errors,
            "split_trace_ids": self.split_trace_ids,
            "traces_with_absent_parents": self.traces_with_absent_parents,
            "max_top_steps": self.max_top_steps,
            "absent_parent_examples": self.absent_parent_examples,
            "unmapped_shapes": self.unmapped_shapes,
            "shape_count": len(self.shapes),
            "shapes": [
                {
                    "shape_id": shape,
                    "traces": stats.traces,
                    "example_trace_id": stats.example_trace_id,
                    "model_calls": stats.model_calls,
                    "task_status": dict(stats.task_status),
                }
                for shape, stats in self.ranked_shapes()[:SHAPES_KEPT]
            ],
        }

    def shape_lines(self, top: int = 5) -> list[str]:
        """The most common shapes, one line each, for printing."""
        total = sum(stats.traces for stats in self.shapes.values())
        ranked = self.ranked_shapes()
        lines = [
            f"{shape}  {stats.traces / total:>4.0%}  {stats.traces} trace(s), "
            f"{stats.model_calls} model calls"
            + (
                ", tasks " + " ".join(f"{k} {v}" for k, v in sorted(stats.task_status.items()))
                if stats.task_status
                else ""
            )
            + f"; e.g. trace {stats.example_trace_id}"
            for shape, stats in ranked[:top]
        ]
        if len(ranked) > top:
            lines.append(f"+{len(ranked) - top} more shapes")
        return lines


def aggregate_issues(
    report: IngestReport, location: str, *, workflow: bool, mapping_name: str | None = None
) -> list[TraceIssue]:
    """One issue per kind and label for the whole ingest, never one per chunk."""
    issues = [
        TraceIssue(
            kind="unparsed_value",
            detail=f"{count} value(s) under {key} open as JSON but do not parse, as an "
            "exporter that cuts long values off leaves them; none was read as message text",
            location=location,
        )
        for key, count in sorted(report.unparsed.items())
    ]
    issues += [
        TraceIssue(
            kind="unrepresented_span",
            detail=f"{count} span(s) with {label} carry no model or tool call and are not "
            "in the corpus",
            location=location,
        )
        for label, count in sorted(report.unrepresented.items())
    ]
    issues += [
        TraceIssue(
            kind="source_container",
            detail=f"{count} converter container span(s) with {label} are kept in the "
            "source archive, not as application actions",
            location=location,
        )
        for label, count in sorted(report.containers.items())
    ]
    issues += [
        TraceIssue(
            kind="excluded_evaluator",
            detail=f"{count} evaluator span(s) with {label} excluded from the action corpus",
            location=location,
        )
        for label, count in sorted(report.excluded.items())
    ]
    if workflow:
        issues += [
            TraceIssue(
                kind=f"task_{status}",
                detail=f"{report.task_status[status]} workflow trace(s) have their task "
                f"{status}; see each trace's request.task_reason",
                location=location,
            )
            for status in ("unresolved", "conflict")
            if report.task_status[status]
        ]
        if report.traces_with_absent_parents:
            issues.append(
                TraceIssue(
                    kind="parent_not_exported",
                    detail=f"{report.traces_with_absent_parents} trace(s) have top-level steps "
                    "whose parent is not in the export (max "
                    f"{report.max_top_steps} per trace); e.g. "
                    + "; ".join(report.absent_parent_examples),
                    location=location,
                )
            )
    if report.excluded_by_mapping:
        issues.append(
            TraceIssue(
                kind="excluded_by_mapping",
                detail=f"mapping {mapping_name} excluded "
                + ", ".join(
                    f"{count} span(s) of {key}"
                    for key, count in sorted(report.excluded_by_mapping.items())
                )
                + " (with everything beneath them); they stay in the source archive",
                location=location,
            )
        )
    issues += [
        TraceIssue(
            kind="mapping_override_not_applicable",
            detail=f"mapping {mapping_name} marks {key} as 'tool', but {count} such span(s) "
            "have model or tool calls beneath them, which represent them; left as recorded",
            location=location,
        )
        for key, count in sorted(report.override_not_applicable.items())
    ]
    if report.unmapped_shapes:
        issues.append(
            TraceIssue(
                kind="shape_not_in_mapping",
                detail=f"{report.unmapped_shapes} trace(s) have shapes mapping {mapping_name} "
                "wasn't confirmed on; they were ingested with its rules. Review with "
                "`bandits mapping propose ... --force`",
                location=location,
            )
        )
    if report.unconvertible_examples:
        lost = report.buckets["unconvertible"] + report.buckets["duplicate_native"]
        issues.append(
            TraceIssue(
                kind="unconvertible_observation",
                detail=f"{lost} observation(s) could not become spans and are not in the "
                "corpus; e.g. " + "; ".join(report.unconvertible_examples),
                location=location,
            )
        )
    if report.split_trace_ids:
        issues.append(
            TraceIssue(
                kind="trace_split_across_chunks",
                detail=f"{report.split_trace_ids} trace id(s) appear in records read in "
                "different batches, so the corpus holds more than one trace with that id; "
                "keep each trace's observations in one record",
                location=location,
            )
        )
    return issues
