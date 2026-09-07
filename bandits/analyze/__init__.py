"""Deterministic analysis of a trace corpus: task candidates and outcome evidence.

Nothing here is inferred by a model. Every value is read directly off spans that
are already in the corpus, and anything the source did not record is named as a
limitation rather than filled in. Model-assisted analysis is a later, explicitly
labelled addition on top of this layer, never a replacement for it.
"""

from __future__ import annotations

from bandits.analyze.analysis import (
    analyze_corpus,
    compute_analysis_id,
    load_analysis,
    save_analysis,
)
from bandits.analyze.audit import (
    DEFAULT_MODEL as AUDIT_MODEL,
)
from bandits.analyze.audit import (
    AuditError,
    audit_family,
    audit_task_set,
    build_predictor,
    compute_audit_run_id,
    load_audit_run,
    save_audit_run,
)
from bandits.analyze.embed import (
    EmbeddingCache,
    EmbeddingError,
    build_cache,
    cosine_distance,
    embedding_distance,
    load_cache,
    save_cache,
)
from bandits.analyze.families import (
    DEFAULT_BUDGET,
    DEFAULT_HELD_OUT,
    DEFAULT_NEIGHBORS,
    compute_task_set_id,
    fingerprint,
    load_task_set,
    merge_families,
    mine_task_set,
    normalize_instruction,
    save_task_set,
    split_family,
)
from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    FamilyAudit,
    FamilyAuditRun,
    LeakageError,
    MissingSlot,
    SelectedTask,
    SkippedAudit,
    SlotKind,
    TaskCandidate,
    TaskFamily,
    TaskSet,
    Visibility,
    build_task_candidate,
)
from bandits.analyze.outcomes import extract_outcome_evidence
from bandits.analyze.tasks import extract_task

__all__ = [
    "AUDIT_MODEL",
    "DEFAULT_BUDGET",
    "DEFAULT_HELD_OUT",
    "DEFAULT_NEIGHBORS",
    "AuditError",
    "CorpusAnalysis",
    "EmbeddingCache",
    "EmbeddingError",
    "Evidence",
    "EvidenceKind",
    "FamilyAudit",
    "FamilyAuditRun",
    "LeakageError",
    "MissingSlot",
    "SelectedTask",
    "SkippedAudit",
    "SlotKind",
    "TaskCandidate",
    "TaskFamily",
    "TaskSet",
    "Visibility",
    "analyze_corpus",
    "audit_family",
    "audit_task_set",
    "build_cache",
    "build_predictor",
    "build_task_candidate",
    "compute_analysis_id",
    "compute_audit_run_id",
    "compute_task_set_id",
    "cosine_distance",
    "embedding_distance",
    "extract_outcome_evidence",
    "extract_task",
    "fingerprint",
    "load_analysis",
    "load_audit_run",
    "load_cache",
    "load_task_set",
    "merge_families",
    "mine_task_set",
    "normalize_instruction",
    "save_analysis",
    "save_audit_run",
    "save_cache",
    "save_task_set",
    "split_family",
]
