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
from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    LeakageError,
    MissingSlot,
    SelectedTask,
    SlotKind,
    TaskCandidate,
    TaskFamily,
    TaskSet,
    Visibility,
    build_task_candidate,
)
from bandits.analyze.outcomes import extract_outcome_evidence
from bandits.analyze.rlm_audit import (
    ClusteringAuditError,
    audit_clustering,
)
from bandits.analyze.rlm_audit import (
    compute_audit_id as compute_rlm_audit_id,
)
from bandits.analyze.rlm_audit import (
    load_audit as load_rlm_audit,
)
from bandits.analyze.rlm_audit import (
    save_audit as save_rlm_audit,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus, build_view
from bandits.analyze.rlm_mine import (
    MiningError,
    compute_run_id,
    load_clustering_run,
    mine_taxonomy,
    save_clustering_run,
)
from bandits.analyze.rlm_models import (
    AuditFinding,
    Budget,
    ChunkResult,
    FamilyContract,
    Operation,
    RLMClusteringAudit,
    RLMClusteringRun,
    StopReason,
    TaxonomyOperation,
    TraceView,
    UserMessageView,
)
from bandits.analyze.rlm_taskset import MaterializationError, materialize_task_set
from bandits.analyze.tasks import extract_task
from bandits.analyze.tasksets import (
    DEFAULT_HELD_OUT,
    compute_task_set_id,
    load_task_set,
    save_task_set,
)
from bandits.analyze.text import normalize_instruction

__all__ = [
    "DEFAULT_HELD_OUT",
    "AuditFinding",
    "Budget",
    "ChunkResult",
    "CorpusAnalysis",
    "DEFAULT_HELD_OUT",
    "Evidence",
    "EvidenceKind",
    "FamilyContract",
    "LeakageError",
    "MaterializationError",
    "MiningError",
    "MissingSlot",
    "Operation",
    "ReadOnlyCorpus",
    "SelectedTask",
    "SlotKind",
    "StopReason",
    "TaskCandidate",
    "TaskFamily",
    "TaskSet",
    "RLMClusteringAudit",
    "ClusteringAuditError",
    "RLMClusteringRun",
    "TaxonomyOperation",
    "TraceView",
    "UserMessageView",
    "Visibility",
    "analyze_corpus",
    "audit_clustering",
    "build_task_candidate",
    "build_view",
    "compute_analysis_id",
    "compute_run_id",
    "compute_rlm_audit_id",
    "compute_task_set_id",
    "extract_outcome_evidence",
    "extract_task",
    "load_analysis",
    "load_clustering_run",
    "load_rlm_audit",
    "load_task_set",
    "materialize_task_set",
    "mine_taxonomy",
    "normalize_instruction",
    "save_analysis",
    "save_clustering_run",
    "save_rlm_audit",
    "save_task_set",
]
