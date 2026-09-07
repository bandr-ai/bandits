"""Portable evaluation cases gated by an explicitly reviewed verifier."""

from __future__ import annotations

import hashlib

from bandits.analyze.models import CorpusAnalysis, TaskFamily, TaskSet, Visibility
from bandits.export.models import (
    EvalCase,
    ExportBundle,
    ExportKind,
    ExportManifest,
    Partition,
    RejectedTrace,
)
from bandits.store import compute_artifact_id
from bandits.traces import TraceCorpus
from bandits.verify.review import ReviewedVerifier


def prompt_rejection_reasons(trace_id: str, analysis: CorpusAnalysis) -> list[str]:
    task = next((item for item in analysis.tasks if item.trace_id == trace_id), None)
    if task is None:
        return ["analysis contains no task candidate for this trace"]
    reasons: list[str] = []
    if not task.instruction:
        reasons.append("source declared no instruction")
    evidence = analysis.evidence_by_id()
    for evidence_id in task.prompt_evidence_ids:
        item = evidence.get(evidence_id)
        if item is None:
            reasons.append(f"prompt references missing evidence {evidence_id!r}")
        elif item.visibility is not Visibility.AT_START:
            reasons.append("task prompt contains evidence unavailable at start")
    return reasons


def partition_trace_ids(family: TaskFamily, partition: Partition) -> tuple[str, ...]:
    """The traces one export may draw from, and nothing else.

    Traces outside the partition are not rejected candidates; they were never
    candidates. Quarantining them would bury the reasons that describe a real
    problem under one line per trace saying only that it was on the other side
    of a split.
    """
    if partition is Partition.FIT:
        return family.fit_trace_ids
    if partition is Partition.HELD_OUT:
        return family.held_out_trace_ids
    return family.trace_ids


def _partition_warnings(family: TaskFamily, partition: Partition) -> tuple[str, ...]:
    if partition is Partition.ALL:
        return (
            "drawn from the whole family: these traces overlap whatever the other "
            "export format drew from either side of the split",
        )
    if not partition_trace_ids(family, partition):
        return (
            f"family {family.family_id} has no {partition.value} traces; "
            "nothing was eligible before any gate ran",
        )
    return ()


def _check_lineage(
    corpus: TraceCorpus,
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    reviewed: ReviewedVerifier,
) -> None:
    if task_set.corpus_id != analysis.corpus_id:
        raise ValueError("task set and analysis refer to different corpora")
    if compute_artifact_id(corpus) != task_set.corpus_id:
        raise ValueError("corpus does not match the task set's corpus id")
    if reviewed.spec.task_set_id != task_set_id:
        raise ValueError("reviewed verifier belongs to another task set")
    if reviewed.spec.family_id not in task_set.family_by_id():
        raise ValueError("reviewed verifier family is not present in the task set")
    if corpus.source != analysis.source:
        raise ValueError("corpus and analysis source disagree")


def build_eval_export(
    corpus: TraceCorpus,
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    reviewed: ReviewedVerifier,
    reviewed_id: str,
    *,
    partition: Partition = Partition.HELD_OUT,
) -> ExportBundle:
    """Export evaluation cases from the side of the split SFT may not train on."""
    _check_lineage(corpus, task_set, task_set_id, analysis, reviewed)
    spec = reviewed.spec
    family = task_set.family_by_id()[spec.family_id]
    traces = {trace.trace_id: trace for trace in corpus.traces}
    rows: list[EvalCase] = []
    rejected: list[RejectedTrace] = []

    for trace_id in partition_trace_ids(family, partition):
        reasons = prompt_rejection_reasons(trace_id, analysis)
        trace = traces.get(trace_id)
        if trace is None:
            reasons.append("corpus contains no matching trace")
        if reasons:
            rejected.append(
                RejectedTrace(trace_id=trace_id, family_id=family.family_id, reasons=tuple(reasons))
            )
            continue
        digest = hashlib.sha256(f"{trace_id}\0{spec.verifier_id}".encode()).hexdigest()[:16]
        rows.append(
            EvalCase(
                case_id=f"eval-{digest}",
                instruction=trace.task,  # type: ignore[union-attr,arg-type]
                system_prompt=trace.system_prompt,  # type: ignore[union-attr]
                tools=(
                    tuple(tool.model_dump(mode="json") for tool in trace.tools_available)  # type: ignore[union-attr]
                    if trace.tools_available is not None  # type: ignore[union-attr]
                    else None
                ),
                grader=spec.model_dump(mode="json"),
                corpus_id=task_set.corpus_id,
                task_set_id=task_set_id,
                family_id=family.family_id,
                trace_id=trace_id,
                verifier_id=spec.verifier_id,
                validation_id=reviewed.validation_id,
            )
        )

    return ExportBundle(
        manifest=ExportManifest(
            format=ExportKind.EVAL,
            corpus_id=task_set.corpus_id,
            task_set_id=task_set_id,
            reviewed_verifier_id=reviewed_id,
            verifier_id=spec.verifier_id,
            validation_id=reviewed.validation_id,
            human_acceptance_id=reviewed.human_acceptance_id,
            verifier_status=spec.status.value,
            interview_id=reviewed.interview_id,
            partition=partition,
            partition_trace_count=len(partition_trace_ids(family, partition)),
            success_threshold=reviewed.success_threshold,
            rows=len(rows),
            unresolved=len(rejected),
            warnings=_partition_warnings(family, partition),
        ),
        rows=tuple(rows),
        unresolved=tuple(rejected),
    )
