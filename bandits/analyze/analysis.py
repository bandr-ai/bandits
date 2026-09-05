"""Assemble one corpus into one analysis artifact."""

from __future__ import annotations

import hashlib

from bandits.analyze.models import CorpusAnalysis, Evidence, TaskCandidate
from bandits.analyze.outcomes import extract_outcome_evidence
from bandits.analyze.tasks import extract_task
from bandits.store import DerivedEnvelope, DerivedStore, compute_artifact_id
from bandits.traces import TraceCorpus


def compute_analysis_id(analysis: CorpusAnalysis) -> str:
    digest = hashlib.sha256(analysis.model_dump_json().encode("utf-8")).hexdigest()
    return f"analysis-{digest[:16]}"


def analyze_corpus(corpus: TraceCorpus) -> CorpusAnalysis:
    """Extract task candidates and outcome evidence from every trace in a corpus."""
    tasks: list[TaskCandidate] = []
    evidence: list[Evidence] = []

    for trace in corpus.traces:
        task, task_evidence = extract_task(trace)
        outcome_evidence = extract_outcome_evidence(trace)
        tasks.append(
            task.replace(outcome_evidence_ids=tuple(e.evidence_id for e in outcome_evidence))
        )
        evidence.extend(task_evidence)
        evidence.extend(outcome_evidence)

    limitations: list[str] = []
    if corpus.issues:
        limitations.append(
            f"{len(corpus.issues)} source record(s) could not be normalized; "
            "the corpus is not a complete view of the export"
        )
    tasks_without_outcome = sum(1 for t in tasks if not t.outcome_evidence_ids)
    if tasks_without_outcome:
        limitations.append(
            f"{tasks_without_outcome} task(s) have no recorded outcome evidence at all; "
            "their success cannot be judged from the trace alone"
        )

    truncated = {e.trace_id for e in evidence if e.claim == "truncated_outcome_fields"}
    if truncated:
        limitations.append(
            f"{len(truncated)} trace(s) recorded a terminal result too large or too deeply "
            "nested to read in full; a field absent from a truncated read is unknown, "
            "not missing"
        )
    unstructured = {e.trace_id for e in evidence if e.claim == "unstructured_final_result"}
    if unstructured:
        limitations.append(
            f"{len(unstructured)} trace(s) recorded a terminal result holding no comparable "
            "field; whatever those results mean is domain knowledge this extractor does "
            "not have"
        )

    return CorpusAnalysis(
        corpus_id=compute_artifact_id(corpus),
        source=corpus.source,
        tasks=tuple(tasks),
        evidence=tuple(evidence),
        limitations=tuple(limitations),
    )


def save_analysis(analysis: CorpusAnalysis, store: DerivedStore) -> DerivedEnvelope:
    """Persist an analysis, keyed on its own content and linked to its corpus."""
    return store.write(
        compute_analysis_id(analysis),
        kind="analysis",
        parent_artifact_id=analysis.corpus_id,
        payload=analysis.model_dump_json().encode("utf-8"),
        summary={
            "tasks": len(analysis.tasks),
            "evidence": len(analysis.evidence),
            "limitations": len(analysis.limitations),
        },
    )


def load_analysis(analysis_id: str, store: DerivedStore) -> CorpusAnalysis:
    return CorpusAnalysis.model_validate_json(store.read_payload(analysis_id))
