"""Turn a reviewed assignment run into a TaskSet the rest of the pipeline can use.

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
claiming a coverage the assignment run explicitly refused to claim.
"""

from __future__ import annotations

from bandits.analyze.models import (
    ClusteringProvenance,
    SelectedTask,
    SlotKind,
    TaskFamily,
    TaskSet,
)
from bandits.analyze.rlm_models import (
    AssignmentRun,
    AssignmentStatus,
    FrozenTaxonomy,
    TraceView,
)


def backend_for(view: TraceView) -> str:
    """What produced this grouping, naming the arm that produced it.

    Derived from the view rather than fixed, because the arm *is* the experiment:
    a Path F task set labelled ``rlm-user-messages`` would make the two arms
    indistinguishable in the artifact store, and every comparison drawn from
    them afterwards would be reading the wrong provenance.
    """
    return f"rlm-{view.value}"


class MaterializationError(RuntimeError):
    """The assignment run cannot honestly become a task set."""


def materialize_task_set(
    run: AssignmentRun,
    taxonomy: FrozenTaxonomy,
    *,
    corpus_id: str,
    held_out: float = 0.0,
) -> TaskSet:
    """Build a TaskSet from confidently assigned traces only.

    ``held_out`` splits each family for downstream verifier measurement. The
    split is taken over the sorted member list rather than sampled, because
    nothing here has a seed to record and an unrecorded random split cannot be
    reproduced from the artifact.
    """
    if run.taxonomy_id != _taxonomy_matches(run, taxonomy):
        raise MaterializationError(
            "this assignment run was made against a different taxonomy than the one given"
        )
    if not 0.0 <= held_out < 1.0:
        raise ValueError("held_out must be a fraction below 1.0")

    contracts = {c.contract_id: c for c in taxonomy.contracts}
    members = run.members()
    if not members:
        raise MaterializationError(
            "no trace was confidently assigned, so there is no family to materialize"
        )

    families: list[TaskFamily] = []
    selected: list[SelectedTask] = []
    for contract_id, trace_ids in members.items():
        contract = contracts.get(contract_id)
        if contract is None:  # pragma: no cover - assignment validates against the taxonomy
            continue
        cut = int(len(trace_ids) * held_out)
        held = trace_ids[:cut]
        fit = trace_ids[cut:]
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
    # Every trace the assignment run classified, not just the ones it placed.
    # Coverage is the fraction of the workload these families actually stand
    # for, and dividing placed traces by placed traces would report 100% for a
    # run that left most of the corpus unplaced — the more traces the taxonomy
    # failed to reach, the better it would score.
    classified = sum(1 for a in run.assignments if a.status is not AssignmentStatus.UNREADABLE)
    unresolved = {
        status: len(run.by_status(status))
        for status in (
            AssignmentStatus.AMBIGUOUS,
            AssignmentStatus.UNCOVERED,
            AssignmentStatus.UNREADABLE,
        )
    }

    limitations = [
        "families here were proposed by a language model, not by a reproducible "
        "distance computation; two runs of the miner may disagree",
        f"mined from the {run.view.value} view",
        *taxonomy.limitations,
    ]
    for status, count in unresolved.items():
        if count:
            limitations.append(
                f"{count} trace(s) were {status.value} in the assignment run and are "
                "excluded from every family here until a reviewer resolves them"
            )
    if not taxonomy.complete:
        limitations.append(
            "the taxonomy behind this task set was frozen from a draft that never converged"
        )
    if run.view.reads_agent_behavior:
        limitations.append(
            "mined from full trajectories, so the miner could see what the agent did; "
            "these families must be checked against tool usage, trajectory length and "
            "outcome before being read as task families rather than behaviour groups"
        )

    return TaskSet(
        corpus_id=corpus_id,
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


def _taxonomy_matches(run: AssignmentRun, taxonomy: FrozenTaxonomy) -> str:
    """The run's taxonomy id if the taxonomy given is really the one it used.

    Compared on contract text rather than on the id alone, so a taxonomy loaded
    from the wrong artifact cannot pass by carrying a matching id.
    """
    from bandits.analyze.rlm_audit import compute_taxonomy_id

    return compute_taxonomy_id(taxonomy) if compute_taxonomy_id(taxonomy) == run.taxonomy_id else ""
