"""Turn a clustering run into a TaskSet the rest of the pipeline can use.

The honesty problem this solves. ``TaskSet`` was designed around embedding
geometry: a family carries a ``medoid_trace_id``, a ``FamilyCoherence`` diameter,
and a ``ClusteringProvenance`` naming a similarity threshold. None of those exist
here — nothing measured a distance, so there is no centre and no diameter — and
filling them in with plausible numbers would put fabricated geometry into an
artifact whose whole claim is that it did not use any.

So they are left absent. ``coherence`` stays ``None``, the provenance names the
RLM backend with the model and view rather than a threshold, and the medoid is
the lexically first member: a real trace, chosen by a rule that is obviously not
a centrality claim. Every family says in its limitations that it was not measured.

Ambiguous, uncovered and unreadable traces are never materialized. They are
findings awaiting review, and a task set that quietly absorbed them would be
claiming a coverage the run explicitly refused to claim.

The fit/held-out split moves whole lineage groups, never individual traces. No
distance is involved: a lineage the analysis declared is one group, a trace
without one is its own group, and two traces of the same normalized request are
held together so a verifier is never measured against a rerun of what it was
drafted from.
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
from bandits.analyze.rlm_models import RLMClusteringRun, TraceView
from bandits.analyze.text import normalize_instruction


def backend_for(view: TraceView) -> str:
    """What produced this grouping, naming the arm that produced it.

    Derived from the view rather than fixed, because the arm *is* the experiment:
    a Path F task set labelled ``rlm-user-messages`` would make the two arms
    indistinguishable in the artifact store, and every comparison drawn from
    them afterwards would be reading the wrong provenance.
    """
    return f"rlm-{view.value}"


class MaterializationError(RuntimeError):
    """The clustering run cannot honestly become a task set."""


def _stable_fraction(*parts: str) -> float:
    """A deterministic value in [0, 1) for one key, so splits are reproducible."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _split_groups(analysis: CorpusAnalysis, trace_ids: tuple[str, ...]) -> dict[str, list[str]]:
    """The indivisible groups inside one family.

    Two traces share a group when the analysis declares the same lineage, or
    when their requests normalize identically. The second rule matters as much
    as the first: a corpus that never recorded lineage still repeats requests,
    and splitting a repeated request across the boundary measures a verifier
    against the run it was drafted from.
    """
    by_trace = {task.trace_id: task for task in analysis.tasks}
    groups: dict[str, list[str]] = {}
    for trace_id in trace_ids:
        task = by_trace.get(trace_id)
        lineage = task.lineage_id if task else None
        if lineage:
            key = f"lineage:{lineage}"
        elif task and task.instruction:
            key = f"request:{normalize_instruction(task.instruction)}"
        else:
            # No lineage and no instruction to match on. Its own group, never
            # merged with another, because nothing here says it is a rerun.
            key = f"trace:{trace_id}"
        groups.setdefault(key, []).append(trace_id)
    return groups


def _split_family(
    analysis: CorpusAnalysis,
    *,
    family_id: str,
    trace_ids: tuple[str, ...],
    held_out: float,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fit ids, held-out ids, and any limitations the split itself produced.

    Groups are ordered by a hash of the analysis, the family and the group key,
    so the split is reproducible from the artifact without recording a seed.
    Whole groups move: the target fraction is approached and never met exactly,
    because meeting it exactly means cutting a group in half.
    """
    groups = _split_groups(analysis, trace_ids)
    if held_out <= 0.0 or len(groups) < 2:
        limitation = ()
        if held_out > 0.0:
            limitation = (
                f"family {family_id} holds one independent group of episodes, so no "
                "held-out side could be taken and a verifier drafted here cannot be "
                "validated against traces it did not see",
            )
        return trace_ids, (), limitation

    ordered = sorted(
        groups.items(),
        key=lambda item: (_stable_fraction(analysis.corpus_id, family_id, item[0]), item[0]),
    )
    target = len(trace_ids) * held_out
    held: list[str] = []
    fit: list[str] = []
    for _, members in ordered:
        # The last group always lands in fit: a family whose every group went
        # held-out has nothing left to draft a verifier from.
        if len(held) + len(members) <= target and len(fit) + len(members) < len(trace_ids):
            held.extend(members)
        else:
            fit.extend(members)
    if not held:
        return (
            tuple(sorted(fit)),
            (),
            (
                f"family {family_id} has no group small enough to hold out at the requested "
                "fraction; every episode is on the fit side",
            ),
        )
    return tuple(sorted(fit)), tuple(sorted(held)), ()


def materialize_task_set(
    run: RLMClusteringRun,
    analysis: CorpusAnalysis,
    *,
    held_out: float = 0.0,
) -> TaskSet:
    """Build a TaskSet from the traces the miner placed.

    The run's own final assignments are authoritative. Nothing reclassifies
    here, and a trace the miner left ambiguous, uncovered or unreadable stays
    out of every family rather than being placed by this code.
    """
    if not 0.0 <= held_out < 1.0:
        raise ValueError("held_out must be a fraction below 1.0")
    # Compared on content, not on a passed-in id: an analysis loaded from the
    # wrong artifact would otherwise materialize families whose members were
    # never in it, and the mismatch would only surface downstream as missing
    # traces.
    from bandits.analyze.analysis import compute_analysis_id

    if compute_analysis_id(analysis) != run.analysis_id:
        raise MaterializationError(
            f"this clustering run was made against analysis {run.analysis_id!r}, "
            "which is not the analysis given"
        )

    live = {contract.contract_id for contract in run.contracts}
    unknown = sorted({cid for cid in run.assignments.values() if cid not in live})
    if unknown:
        raise MaterializationError(
            f"the run assigns traces to contract(s) it does not define: {unknown}"
        )

    unresolved = (
        set(run.ambiguous_trace_ids) | set(run.uncovered_trace_ids) | set(run.unreadable_trace_ids)
    )
    members: dict[str, list[str]] = {}
    for trace_id, contract_id in run.assignments.items():
        if trace_id in unresolved:
            continue
        members.setdefault(contract_id, []).append(trace_id)
    if not members:
        raise MaterializationError("no trace was placed, so there is no family to materialize")

    contracts = {contract.contract_id: contract for contract in run.contracts}
    families: list[TaskFamily] = []
    selected: list[SelectedTask] = []
    split_limitations: list[str] = []
    for contract_id in sorted(members):
        trace_ids = tuple(sorted(members[contract_id]))
        contract = contracts[contract_id]
        fit, held, limits = _split_family(
            analysis, family_id=contract_id, trace_ids=trace_ids, held_out=held_out
        )
        split_limitations.extend(limits)
        families.append(
            TaskFamily(
                family_id=contract_id,
                # The definition, not the name: a descriptor is what downstream
                # code reads to understand what the family is, and a short name
                # would lose the claim the contract actually makes.
                descriptor=contract.definition,
                trace_ids=trace_ids,
                # Lexically first, not central. Nothing measured distance here,
                # so there is no medoid to compute; this is a real member chosen
                # by a rule that cannot be mistaken for a centrality claim.
                medoid_trace_id=trace_ids[0],
                workload_mass=len(trace_ids),
                fit_trace_ids=fit,
                held_out_trace_ids=held,
                proposed_by="model",
                coherence=None,
                limitations=(
                    f"mined by an RLM from the {run.view.value} view; no embedding "
                    "distance was computed, so this family has no measured coherence and "
                    "its medoid is the lexically first member rather than a central one",
                )
                + (
                    (
                        "the miner could see what the agent did, so this family may group "
                        "episodes by execution behaviour rather than by requested work",
                    )
                    if run.view.reads_agent_behavior
                    else ()
                ),
            )
        )
        selected.append(
            SelectedTask(trace_id=trace_ids[0], family_id=contract_id, slot=SlotKind.MEDOID)
        )

    total_mass = sum(f.workload_mass for f in families)
    # Every trace the miner could read, not just the ones it placed. Dividing
    # placed traces by placed traces would report 100% for a run that left most
    # of the corpus unplaced — the more traces the run failed to reach, the
    # better it would score.
    classified = len({task.trace_id for task in analysis.tasks} - set(run.unreadable_trace_ids))

    limitations = [
        "families here were proposed by a language model, not by a reproducible "
        "distance computation; two runs of the miner may disagree",
        f"mined from the {run.view.value} view",
        "membership is the miner's own final placement; no independent pass "
        "reclassified these traces, so a family and its members were decided in "
        "the same context",
        *run.limitations,
        *split_limitations,
    ]
    for label, ids in (
        ("ambiguous", run.ambiguous_trace_ids),
        ("uncovered", run.uncovered_trace_ids),
        ("unreadable", run.unreadable_trace_ids),
    ):
        if ids:
            limitations.append(
                f"{len(ids)} trace(s) were {label} in the clustering run and are "
                "excluded from every family here until a reviewer resolves them"
            )
    if not run.complete:
        limitations.append(
            "the clustering run behind this task set never finished its requested passes"
        )
    if run.view.reads_agent_behavior:
        limitations.append(
            "mined from full trajectories, so the miner could see what the agent did; "
            "these families must be checked against tool usage, trajectory length and "
            "outcome before being read as task families rather than behaviour groups"
        )

    return TaskSet(
        corpus_id=analysis.corpus_id,
        analysis_id=run.analysis_id,
        families=tuple(families),
        selected=tuple(selected),
        clustering=ClusteringProvenance(
            backend=backend_for(run.view),
            # Not thresholds. There is no similarity here, and these are the
            # neutral values the contract requires; what actually produced the
            # grouping is the model and view recorded beside them.
            similarity=0.0,
            neighbors=1,
            embedding_model=None,
            embedding_cache_id=None,
        ),
        total_workload_mass=classified,
        workload_coverage=(total_mass / classified) if classified else 0.0,
        missing_slots=(),
        limitations=tuple(dict.fromkeys(limitations)),
    )
