"""Deterministic task sets for tests downstream of grouping.

Test infrastructure, deliberately outside the shipped package. The split rule
below is tuned to how the fixture corpora order their episodes, which is a
reasonable thing for a test helper to do and a misleading thing to find sitting
in ``bandits.analyze`` beside code that makes claims about real corpora.

Verifier drafting, running, validation and export all need a ``TaskSet`` to
work on, and none of them care how the grouping was produced. Building one here
by an obvious rule keeps those suites independent of whatever discovers families
this month: they broke as a set when the embedding miner was removed, and would
break again on the next change to mining.

Not a miner. Grouping is by first word of the instruction, which is not a claim
about anything — it is a rule that puts known fixtures in known families so a
test can assert about a family it can predict.
"""

from __future__ import annotations

import hashlib

from bandits.analyze.models import (
    ClusteringProvenance,
    CorpusAnalysis,
    SelectedTask,
    SlotKind,
    TaskFamily,
    TaskSet,
)
from bandits.analyze.text import normalize_instruction


def _stable_fraction(*parts: str) -> float:
    """A deterministic value in [0, 1) for one key, so splits are reproducible."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def task_set_by_first_word(
    analysis: CorpusAnalysis,
    analysis_id: str,
    *,
    budget: int = 10,
    held_out: float = 0.3,
) -> TaskSet:
    """Group an analysis by the first word of each instruction.

    ``budget`` caps how many families are kept, largest first, so a test can ask
    for one family without constructing a single-instruction corpus. Whole
    lineage groups move to the held-out side, so nothing here can put one
    lineage on both sides of a boundary a downstream test relies on.
    """
    groups: dict[str, list[str]] = {}
    descriptors: dict[str, str] = {}
    lineages: dict[str, str] = {}
    for task in analysis.tasks:
        if not task.instruction:
            continue
        normalized = normalize_instruction(task.instruction)
        key = normalized.partition(" ")[0]
        groups.setdefault(key, []).append(task.trace_id)
        descriptors.setdefault(key, normalized)
        lineages[task.trace_id] = task.lineage_id or f"trace:{task.trace_id}"

    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:budget]

    families: list[TaskFamily] = []
    selected: list[SelectedTask] = []
    for key, trace_ids in ranked:
        ordered = tuple(sorted(trace_ids))
        by_lineage: dict[str, list[str]] = {}
        for trace_id in ordered:
            by_lineage.setdefault(lineages[trace_id], []).append(trace_id)

        held: list[str] = []
        fit: list[str] = []
        target = round(len(ordered) * held_out)
        family_id = f"family-{key}"
        # Every third group, counted from the second. Fixture corpora order
        # their episodes by outcome — the successes of a skewed family are all
        # at one end, the single interesting run of a small family at the other
        # — so any rule that takes a contiguous run from either end strips one
        # kind of episode entirely from the fit side, and drafting reads only
        # the fit side. Sampling across the order leaves both kinds on both.
        lineage_keys = sorted(by_lineage)
        for index, lineage in enumerate(lineage_keys):
            group = by_lineage[lineage]
            holdable = index % 3 == 1 and len(lineage_keys) > 1
            if holdable and len(held) + len(group) <= target:
                held.extend(group)
            else:
                fit.extend(group)

        families.append(
            TaskFamily(
                family_id=family_id,
                descriptor=descriptors[key],
                trace_ids=ordered,
                medoid_trace_id=ordered[0],
                workload_mass=len(ordered),
                fit_trace_ids=tuple(sorted(fit)),
                held_out_trace_ids=tuple(sorted(held)),
                proposed_by="rule",
                coherence=None,
                limitations=("grouped by first word for a test; not a mined family",),
            )
        )
        selected.append(
            SelectedTask(trace_id=ordered[0], family_id=family_id, slot=SlotKind.MEDOID)
        )

    total = sum(len(ids) for ids in groups.values())
    covered = sum(family.workload_mass for family in families)
    return TaskSet(
        corpus_id=analysis.corpus_id,
        analysis_id=analysis_id,
        families=tuple(families),
        selected=tuple(selected),
        clustering=ClusteringProvenance(
            backend="first-word",
            similarity=0.0,
            neighbors=1,
            embedding_model=None,
            embedding_cache_id=None,
        ),
        total_workload_mass=total,
        workload_coverage=(covered / total) if total else 0.0,
        missing_slots=(),
        limitations=("built by a test fixture rule, not by mining",),
    )
