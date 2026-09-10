"""Materializing a clustering run into a TaskSet, without embeddings."""

from __future__ import annotations

import pytest

from bandits.analyze.analysis import compute_analysis_id
from bandits.analyze.models import CorpusAnalysis, TaskCandidate
from bandits.analyze.rlm_models import (
    Budget,
    FamilyContract,
    RLMClusteringRun,
    StopReason,
    TraceView,
)
from bandits.analyze.rlm_taskset import MaterializationError, materialize_task_set
from bandits.analyze.tasksets import load_task_set, save_task_set
from bandits.store import DerivedStore


def _contract(contract_id: str, definition: str = "refund an eligible order") -> FamilyContract:
    return FamilyContract(
        contract_id=contract_id,
        name="Refund an order",
        definition=definition,
        required_outcome_shape=("the order is refunded",),
    )


def _analysis(*traces: tuple[str, str | None, str | None]) -> CorpusAnalysis:
    """One analysis over ``(trace_id, lineage_id, instruction)`` triples."""
    return CorpusAnalysis(
        corpus_id="corpus-1",
        source="test",
        tasks=tuple(
            TaskCandidate(
                task_id=f"task-{trace_id}",
                trace_id=trace_id,
                lineage_id=lineage_id,
                instruction=instruction,
            )
            for trace_id, lineage_id, instruction in traces
        ),
        evidence=(),
    )


def _run(analysis: CorpusAnalysis, assignments: dict[str, str], **overrides) -> RLMClusteringRun:
    contract_ids = overrides.pop("contract_ids", None) or sorted(set(assignments.values()))
    base = dict(
        analysis_id=compute_analysis_id(analysis),
        view=TraceView.USER_MESSAGES,
        seed=42,
        contracts=tuple(_contract(cid) for cid in contract_ids),
        assignments=assignments,
        stop_reason=StopReason.PASSES_COMPLETE,
        completed_passes=2,
        requested_passes=2,
        budget=Budget(),
        model="test-model",
        prompt_digest="digest",
    )
    return RLMClusteringRun(**{**base, **overrides})


def test_every_assignment_becomes_family_membership() -> None:
    analysis = _analysis(("t1", None, "refund"), ("t2", None, "cancel"), ("t3", None, "refund me"))
    run = _run(analysis, {"t1": "c1", "t2": "c2", "t3": "c1"})

    task_set = materialize_task_set(run, analysis)

    families = {f.family_id: f for f in task_set.families}
    assert set(families) == {"c1", "c2"}
    assert families["c1"].trace_ids == ("t1", "t3")
    assert families["c2"].trace_ids == ("t2",)


def test_the_contract_definition_becomes_the_descriptor() -> None:
    """Downstream code reads the descriptor to learn what a family is; a short
    name would lose the claim the contract actually makes."""
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"})
    run = run.model_copy(
        update={"contracts": (_contract("c1", definition="refund an order shipped late"),)}
    )

    task_set = materialize_task_set(run, analysis)

    assert task_set.families[0].descriptor == "refund an order shipped late"


def test_an_assignment_to_an_undefined_contract_is_refused() -> None:
    """A placement naming a contract the run never defined is a broken run, not
    a family to invent."""
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"}, contract_ids=["c1"])
    run = run.model_copy(update={"assignments": {"t1": "c1", "t2": "ghost"}})

    with pytest.raises(MaterializationError, match="does not define"):
        materialize_task_set(run, analysis)


def test_an_analysis_the_run_was_not_made_against_is_refused() -> None:
    analysis = _analysis(("t1", None, "refund"))
    other = _analysis(("t9", None, "something else entirely"))
    run = _run(analysis, {"t1": "c1"})

    with pytest.raises(MaterializationError, match="not the analysis given"):
        materialize_task_set(run, other)


def test_unresolved_traces_are_excluded_and_reported() -> None:
    analysis = _analysis(
        ("t1", None, "refund"),
        ("t2", None, "cancel"),
        ("t3", None, "unclear"),
        ("t4", None, "garbled"),
    )
    run = _run(
        analysis,
        {"t1": "c1", "t2": "c1", "t3": "c1"},
        ambiguous_trace_ids=("t3",),
        uncovered_trace_ids=("t4",),
    )

    task_set = materialize_task_set(run, analysis)

    assert task_set.families[0].trace_ids == ("t1", "t2"), "an ambiguous trace is not a member"
    limitations = " ".join(task_set.limitations)
    assert "1 trace(s) were ambiguous" in limitations
    assert "1 trace(s) were uncovered" in limitations


def test_coverage_is_measured_against_every_readable_trace() -> None:
    """Dividing placed traces by placed traces would report full coverage for a
    run that reached almost none of the corpus."""
    analysis = _analysis(*[(f"t{i}", None, f"request {i}") for i in range(1, 11)])
    run = _run(
        analysis, {"t1": "c1", "t2": "c1"}, uncovered_trace_ids=tuple(f"t{i}" for i in range(3, 11))
    )

    task_set = materialize_task_set(run, analysis)

    assert task_set.total_workload_mass == 10
    assert task_set.workload_coverage == pytest.approx(0.2)


def test_an_unreadable_trace_leaves_the_coverage_denominator() -> None:
    """It was never classifiable, so counting it as uncovered workload would
    charge the run for a trace no grouping could have reached."""
    analysis = _analysis(("t1", None, "refund"), ("t2", None, "cancel"), ("t3", None, None))
    run = _run(analysis, {"t1": "c1", "t2": "c1"}, unreadable_trace_ids=("t3",))

    task_set = materialize_task_set(run, analysis)

    assert task_set.total_workload_mass == 2
    assert task_set.workload_coverage == pytest.approx(1.0)


def test_a_lineage_never_appears_on_both_sides_of_the_split() -> None:
    analysis = _analysis(
        ("t1", "L1", "refund"),
        ("t2", "L1", "refund"),
        ("t3", "L2", "refund again"),
        ("t4", "L3", "refund once more"),
        ("t5", "L4", "and again"),
    )
    run = _run(analysis, {f"t{i}": "c1" for i in range(1, 6)})

    family = materialize_task_set(run, analysis, held_out=0.4).families[0]

    fit, held = set(family.fit_trace_ids), set(family.held_out_trace_ids)
    assert fit & held == set(), "no trace is on both sides"
    assert fit | held == set(family.trace_ids), "every member is on one side"
    assert not ({"t1", "t2"} & fit and {"t1", "t2"} & held), "lineage L1 was split"


def test_exact_duplicate_requests_never_cross_the_split() -> None:
    """A corpus with no lineage metadata still repeats requests, and measuring a
    verifier against a rerun of what it was drafted from proves nothing."""
    analysis = _analysis(
        ("t1", None, "Refund my order"),
        ("t2", None, "refund my order"),
        ("t3", None, "Cancel the flight"),
        ("t4", None, "Change my seat"),
        ("t5", None, "Add a bag"),
    )
    run = _run(analysis, {f"t{i}": "c1" for i in range(1, 6)})

    family = materialize_task_set(run, analysis, held_out=0.4).families[0]

    fit, held = set(family.fit_trace_ids), set(family.held_out_trace_ids)
    assert not ({"t1", "t2"} & fit and {"t1", "t2"} & held), "one request landed on both sides"


def test_a_family_of_one_group_says_it_cannot_be_validated() -> None:
    analysis = _analysis(("t1", "L1", "refund"), ("t2", "L1", "refund"))
    run = _run(analysis, {"t1": "c1", "t2": "c1"})

    task_set = materialize_task_set(run, analysis, held_out=0.5)

    family = task_set.families[0]
    assert family.held_out_trace_ids == ()
    assert family.fit_trace_ids == ("t1", "t2")
    assert any("one independent group" in limit for limit in task_set.limitations)


def test_the_split_is_reproducible() -> None:
    analysis = _analysis(*[(f"t{i}", f"L{i}", f"request {i}") for i in range(1, 9)])
    run = _run(analysis, {f"t{i}": "c1" for i in range(1, 9)})

    first = materialize_task_set(run, analysis, held_out=0.3).families[0]
    second = materialize_task_set(run, analysis, held_out=0.3).families[0]

    assert first.held_out_trace_ids == second.held_out_trace_ids


def test_the_representative_is_a_real_member_and_says_it_is_not_central() -> None:
    analysis = _analysis(("t2", None, "refund"), ("t1", None, "refund also"))
    run = _run(analysis, {"t1": "c1", "t2": "c1"})

    family = materialize_task_set(run, analysis).families[0]

    assert family.medoid_trace_id in family.trace_ids
    assert family.medoid_trace_id == "t1", "lexically first, deterministically"
    assert any("lexically first member" in limit for limit in family.limitations)
    assert family.coherence is None, "nothing measured a distance, so there is no diameter"


def test_an_incomplete_run_carries_that_into_the_task_set() -> None:
    analysis = _analysis(("t1", None, "refund"))
    run = _run(
        analysis,
        {"t1": "c1"},
        stop_reason=StopReason.MAX_LLM_CALLS,
        completed_passes=1,
        limitations=("the budget ran out mid-pass",),
    )

    task_set = materialize_task_set(run, analysis)

    limitations = " ".join(task_set.limitations)
    assert "never finished its requested passes" in limitations
    assert "the budget ran out mid-pass" in limitations


def test_the_task_set_says_membership_was_never_independently_checked() -> None:
    """The miner named the family and placed its members in one context. A
    reader must not mistake that for a cold classification."""
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"})

    task_set = materialize_task_set(run, analysis)

    assert any("in the same context" in limit for limit in task_set.limitations)


def test_a_run_that_placed_nothing_is_refused() -> None:
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {}, contract_ids=["c1"])

    with pytest.raises(MaterializationError, match="no trace was placed"):
        materialize_task_set(run, analysis)


def test_the_task_set_round_trips_through_the_store(tmp_path) -> None:
    analysis = _analysis(("t1", None, "refund"), ("t2", "L2", "cancel"))
    run = _run(analysis, {"t1": "c1", "t2": "c2"})
    store = DerivedStore(tmp_path)

    task_set = materialize_task_set(run, analysis)
    envelope = save_task_set(task_set, store)

    assert load_task_set(envelope.artifact_id, store) == task_set


def test_the_provenance_names_the_arm_rather_than_a_threshold() -> None:
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"}, view=TraceView.FULL_TRAJECTORY)

    task_set = materialize_task_set(run, analysis)

    assert task_set.clustering.backend == "rlm-full-trajectory"
    assert task_set.clustering.embedding_model is None
    assert any("behaviour groups" in limit for limit in task_set.limitations)


def test_one_request_in_two_lineages_stays_on_one_side() -> None:
    """The two grouping rules compose; the second is not a fallback for the first.

    Two runs of the same request from different sessions arrive as different
    lineages. Reading the request rule only when lineage is absent leaves them
    free to land on opposite sides, which is the leak the rule exists to close.
    """
    analysis = _analysis(
        ("t1", "L1", "Refund order 123"),
        ("t2", "L2", "refund order 123"),
        ("t3", "L3", "Cancel the flight"),
        ("t4", "L4", "Change my seat"),
        ("t5", "L5", "Add a bag"),
        ("t6", "L6", "Upgrade the cabin"),
    )
    run = _run(analysis, {f"t{i}": "c1" for i in range(1, 7)})

    family = materialize_task_set(run, analysis, held_out=0.4).families[0]

    fit, held = set(family.fit_trace_ids), set(family.held_out_trace_ids)
    assert not ({"t1", "t2"} & fit and {"t1", "t2"} & held), (
        "one request in two lineages was split across the boundary"
    )


def test_a_lineage_chain_and_a_shared_request_merge_into_one_group() -> None:
    """Transitivity: t1-t2 by lineage, t2-t3 by request, so all three move together."""
    analysis = _analysis(
        ("t1", "L1", "Refund order 123"),
        ("t2", "L1", "Cancel the flight"),
        ("t3", "L2", "cancel the flight"),
        ("t4", "L3", "Change my seat"),
        ("t5", "L4", "Add a bag"),
        ("t6", "L5", "Upgrade the cabin"),
    )
    run = _run(analysis, {f"t{i}": "c1" for i in range(1, 7)})

    family = materialize_task_set(run, analysis, held_out=0.5).families[0]

    fit, held = set(family.fit_trace_ids), set(family.held_out_trace_ids)
    chain = {"t1", "t2", "t3"}
    assert not (chain & fit and chain & held), "a transitively joined component was split"


def test_a_trace_the_analysis_never_had_is_refused() -> None:
    """Without this it becomes a real family member: it has no task to read, so
    the split treats it as its own group and every count downstream includes it."""
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"}, contract_ids=["c1"])
    run = run.model_copy(update={"assignments": {"t1": "c1", "hallucinated": "c1"}})

    with pytest.raises(MaterializationError, match="does not contain"):
        materialize_task_set(run, analysis)


def test_an_unresolved_trace_the_analysis_never_had_is_refused() -> None:
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"}, ambiguous_trace_ids=("ghost",))

    with pytest.raises(MaterializationError, match="does not contain"):
        materialize_task_set(run, analysis)


def test_the_task_set_records_the_model_and_run_that_produced_it() -> None:
    """"Which model produced this" is the first question asked of two task sets
    that disagree, and prose in limitations cannot be compared across artifacts."""
    analysis = _analysis(("t1", None, "refund"))
    run = _run(analysis, {"t1": "c1"})

    task_set = materialize_task_set(run, analysis, run_id="rlm-clustering-run-abc")

    assert task_set.clustering.model == "test-model"
    assert task_set.clustering.source_run_id == "rlm-clustering-run-abc"
