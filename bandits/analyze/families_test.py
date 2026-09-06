"""Mining tests.

The support fixture is built to have real structure: a dominant refund family
containing a three-attempt retry chain under one session, a smaller cancel
family, an address family whose single long episode is also the only user of two
rare tools, and two traces whose outcome nothing in the corpus records.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bandits.analyze import (
    SlotKind,
    analyze_corpus,
    compute_analysis_id,
    compute_task_set_id,
    load_task_set,
    merge_families,
    mine_task_set,
    normalize_instruction,
    save_task_set,
    split_family,
)
from bandits.analyze.families import _cluster, _medoid, _TraceFeatures
from bandits.analyze.models import (
    CorpusAnalysis,
    FamilyCoherence,
    TaskCandidate,
    TaskFamily,
    TaskSet,
)
from bandits.ingest import load_corpus
from bandits.store import DerivedStore

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
RETRY_CHAIN = {"refund-1", "refund-2", "refund-3"}
"""All three share session sess-r1 and must never straddle a split."""


def _test_distance(left: str, right: str) -> float:
    """Fixture-specific distance for mining mechanics; production uses embeddings."""
    return 0.0 if left.partition(" ")[0] == right.partition(" ")[0] else 1.0


@pytest.fixture
def analysis():
    return analyze_corpus(load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp"))


@pytest.fixture
def task_set(analysis):
    return mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )


def _family(task_set: TaskSet, word: str) -> TaskFamily:
    return next(f for f in task_set.families if word in f.descriptor)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Refund order 7741", "refund order <order_id>"),
        ("Refund order 8820", "refund order <order_id>"),
        ("Ticket #A-9928 is stuck", "ticket <ticket_id> is stuck"),
        ("Email alice@example.com", "email <email>"),
        ("Fetch https://example.com/x", "fetch <url>"),
        ("Check 3f2504e0-4f89-11d3-9a0c-0305e82c3301", "check <uuid>"),
    ],
)
def test_normalization_masks_the_values_that_vary(instruction: str, expected: str) -> None:
    assert normalize_instruction(instruction) == expected


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Handle the HTTP 404", "handle the http 404"),
        ("Upgrade to Python 3.12", "upgrade to python 3_12"),
        ("Retry 3 times", "retry 3 times"),
        ("Upgrade to v2", "upgrade to v2"),
    ],
)
def test_normalization_keeps_values_that_change_the_task(instruction: str, expected: str) -> None:
    """A version, a status code and a count are the task, not identifiers in it."""
    assert normalize_instruction(instruction) == expected


def test_distinct_status_codes_do_not_share_a_descriptor() -> None:
    assert normalize_instruction("Handle the 404") != normalize_instruction("Handle the 500")


def test_repeated_tasks_collapse_into_one_family(task_set: TaskSet) -> None:
    refunds = _family(task_set, "refund")

    assert refunds.workload_mass == 12
    assert refunds.descriptor == "refund order <order_id>"


def test_distinct_tasks_are_not_merged(task_set: TaskSet) -> None:
    """Grouping is conservative: a refund and a cancellation are not one task."""
    descriptors = {f.descriptor for f in task_set.families}

    assert "refund order <order_id>" in descriptors
    assert "cancel order <order_id>" in descriptors


def test_medoid_is_a_real_trace_from_the_family(task_set: TaskSet) -> None:
    for family in task_set.families:
        assert family.medoid_trace_id in family.trace_ids


def _feature(trace_id: str, instruction: str) -> _TraceFeatures:
    return _TraceFeatures(
        trace_id=trace_id,
        lineage_id=None,
        instruction=instruction,
        tools=frozenset(),
        span_count=1,
        has_failure=False,
        measurable=True,
    )


def test_mutual_knn_avoids_a_plain_single_link_chain() -> None:
    features = [
        _feature("a", "alpha beta"),
        _feature("b", "alpha beta gamma"),
        _feature("c", "beta gamma"),
    ]

    clusters = _cluster(features, similarity=0.6, neighbors=1, distance=_test_distance)

    assert sorted(sorted(f.trace_id for f in cluster) for cluster in clusters) == [
        ["a", "b"],
        ["c"],
    ]


def test_medoid_work_scales_with_distinct_descriptors() -> None:
    members = [*(_feature(f"a-{i:03}", "alpha beta") for i in range(100))]
    members.extend(_feature(f"b-{i:03}", "alpha gamma") for i in range(50))
    calls = 0

    def counted(left: str, right: str) -> float:
        nonlocal calls
        calls += 1
        return _test_distance(left, right)

    assert _medoid(members, counted) == "a-000"
    assert calls == 4


def test_workload_mass_sums_to_the_groupable_corpus(task_set: TaskSet) -> None:
    assert sum(f.workload_mass for f in task_set.families) == task_set.total_workload_mass


def test_a_retry_chain_never_straddles_the_split(task_set: TaskSet) -> None:
    refunds = _family(task_set, "refund")

    in_fit = RETRY_CHAIN & set(refunds.fit_trace_ids)
    in_held = RETRY_CHAIN & set(refunds.held_out_trace_ids)

    assert not (in_fit and in_held), "one session's attempts landed on both sides"
    assert in_fit or in_held


def test_the_split_is_reproducible(analysis) -> None:
    first = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=10,
        distance=_test_distance,
        similarity=0.7,
    )
    second = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=10,
        distance=_test_distance,
        similarity=0.7,
    )

    assert first == second
    assert compute_task_set_id(first) == compute_task_set_id(second)


def test_grouping_ignores_everything_a_live_request_would_not_know(analysis) -> None:
    """Two refunds group together despite one failing and one succeeding."""
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=10,
        distance=_test_distance,
        similarity=0.7,
    )
    refunds = set(_family(task_set, "refund").trace_ids)

    assert {"refund-4", "refund-9"} <= refunds, "failed runs belong to the family too"
    assert "refund-1" in refunds


def test_selection_leads_with_the_largest_workload(task_set: TaskSet) -> None:
    medoids = [s for s in task_set.selected if s.slot is SlotKind.MEDOID]

    assert medoids[0].family_id == _family(task_set, "refund").family_id


def test_tail_slots_capture_failures_and_unmeasurable_runs(task_set: TaskSet) -> None:
    """A set of only cluster centres has no failure signal in it."""
    by_slot = {s.slot: s.trace_id for s in task_set.selected}

    assert by_slot[SlotKind.KNOWN_FAILURE].startswith("refund-")
    assert by_slot[SlotKind.UNMEASURABLE].startswith("vague-")


def test_unfillable_slots_are_named_with_a_reason(task_set: TaskSet) -> None:
    reasons = {slot.slot: slot.reason for slot in task_set.missing_slots}

    # addr-1 is the only long episode and the only rare-tool user, and it was
    # already taken as its family's medoid.
    assert "already selected" in reasons["long_episode"]
    assert "already selected" in reasons["rare_tool"]


def test_missing_slots_name_no_domain_specific_category(task_set: TaskSet) -> None:
    """Every reserved slot is structural, so the same set is reachable in any domain."""
    assert {slot.slot for slot in task_set.missing_slots} <= {s.value for s in SlotKind}


def test_semantic_categories_are_declared_a_limitation_not_a_missing_slot(
    task_set: TaskSet,
) -> None:
    """The core says it cannot select for meaning, without guessing at what meaning."""
    assert any("domain extension to select for" in limit for limit in task_set.limitations)


def test_coverage_reports_what_a_small_budget_leaves_out(analysis) -> None:
    small = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=1,
        distance=_test_distance,
        similarity=0.7,
    )
    full = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=10,
        distance=_test_distance,
        similarity=0.7,
    )

    assert small.workload_coverage < full.workload_coverage
    assert full.workload_coverage == 1.0
    assert len(small.selected) == 1


def test_a_budget_larger_than_the_corpus_is_reported_as_underfilled(analysis) -> None:
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=40,
        distance=_test_distance,
        similarity=0.7,
    )

    assert task_set.underfilled
    assert len(task_set.selected) == task_set.total_workload_mass


def test_the_last_slot_is_never_taken_by_the_tail_reserve(analysis) -> None:
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=1,
        distance=_test_distance,
        similarity=0.7,
    )

    assert [s.slot for s in task_set.selected] == [SlotKind.MEDOID]


def test_traces_without_an_instruction_count_against_coverage() -> None:
    corpus = load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp")
    analysis = analyze_corpus(corpus)

    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        budget=5,
        distance=_test_distance,
        similarity=0.7,
    )

    assert task_set.total_workload_mass == len(analysis.tasks)


def test_merging_records_a_human_correction(task_set: TaskSet) -> None:
    refunds, cancels = _family(task_set, "refund"), _family(task_set, "cancel")

    merged = merge_families(task_set, (refunds.family_id, cancels.family_id))
    survivor = merged.family_by_id()[refunds.family_id]

    assert len(merged.families) == len(task_set.families) - 1
    assert survivor.workload_mass == refunds.workload_mass + cancels.workload_mass
    assert survivor.review_status == "merged"
    assert survivor.proposed_by == "human"


def test_a_correction_does_not_change_which_traces_were_selected(task_set: TaskSet) -> None:
    refunds, cancels = _family(task_set, "refund"), _family(task_set, "cancel")

    merged = merge_families(task_set, (refunds.family_id, cancels.family_id))

    assert [s.trace_id for s in merged.selected] == [s.trace_id for s in task_set.selected]
    assert set(merged.family_by_id()) >= {s.family_id for s in merged.selected}


def test_merge_then_split_restores_the_original_grouping(task_set, analysis) -> None:
    refunds, cancels = _family(task_set, "refund"), _family(task_set, "cancel")

    merged = merge_families(task_set, (refunds.family_id, cancels.family_id))
    restored = split_family(merged, refunds.family_id, analysis)

    by_id = restored.family_by_id()
    assert by_id[refunds.family_id].trace_ids == refunds.trace_ids
    assert by_id[cancels.family_id].trace_ids == cancels.trace_ids
    assert by_id[refunds.family_id].held_out_trace_ids == refunds.held_out_trace_ids


def test_splitting_preserves_which_side_each_trace_was_on(task_set, analysis) -> None:
    """A reviewer's correction must not move a trace across the split."""
    refunds, cancels = _family(task_set, "refund"), _family(task_set, "cancel")
    merged = merge_families(task_set, (refunds.family_id, cancels.family_id))

    restored = split_family(merged, refunds.family_id, analysis)

    held_before = set(refunds.held_out_trace_ids) | set(cancels.held_out_trace_ids)
    held_after = {t for f in restored.families for t in f.held_out_trace_ids}
    assert held_before <= held_after


@pytest.mark.parametrize(
    ("family_ids", "message"),
    [
        (("family-nope", "family-also-nope"), "unknown family"),
        ((), "at least two"),
    ],
)
def test_merge_rejects_an_unusable_request(task_set, family_ids, message) -> None:
    with pytest.raises(ValueError, match=message):
        merge_families(task_set, family_ids)


def test_splitting_a_single_instruction_family_is_refused(task_set, analysis) -> None:
    cancels = _family(task_set, "cancel")

    with pytest.raises(ValueError, match="nothing to split"):
        split_family(task_set, cancels.family_id, analysis)


def test_task_set_rejects_a_trace_on_both_sides_of_its_split(task_set: TaskSet) -> None:
    refunds = _family(task_set, "refund")
    broken = refunds.model_copy(update={"fit_trace_ids": refunds.trace_ids})

    payload = task_set.model_copy(update={"families": (broken,), "selected": ()}).model_dump()

    with pytest.raises(ValidationError, match="both sides"):
        TaskSet.model_validate(payload)


def test_task_set_rejects_a_medoid_from_outside_its_family(task_set: TaskSet) -> None:
    refunds = _family(task_set, "refund")
    broken = refunds.model_copy(update={"medoid_trace_id": "not-a-member"})

    payload = task_set.model_copy(update={"families": (broken,), "selected": ()}).model_dump()

    with pytest.raises(ValidationError, match="medoid is not one of its traces"):
        TaskSet.model_validate(payload)


def test_task_set_rejects_a_selection_pointing_at_no_family(task_set: TaskSet) -> None:
    payload = task_set.model_copy(update={"families": ()}).model_dump()

    with pytest.raises(ValidationError, match="unknown family"):
        TaskSet.model_validate(payload)


def test_task_set_round_trips_through_the_derived_store(tmp_path, task_set: TaskSet) -> None:
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_task_set(task_set, store)

    assert envelope.kind == "taskset"
    assert envelope.parent_artifact_id == task_set.analysis_id
    assert load_task_set(envelope.artifact_id, store) == task_set


_CHAIN = ("descriptor alpha", "descriptor bravo", "descriptor charlie", "descriptor delta")
"""Four descriptors placed on a line by _chain_distance, 0.25 apart in order."""


def _chain_analysis(count: int) -> CorpusAnalysis:
    return CorpusAnalysis(
        corpus_id="corpus-chain",
        source="otlp",
        evidence=(),
        tasks=tuple(
            TaskCandidate(task_id=f"task-{i}", trace_id=f"c{i}", instruction=text)
            for i, text in enumerate(_CHAIN[:count])
        ),
    )


def _chain_distance(left: str, right: str) -> float:
    """Distance along a line: adjacent descriptors 0.25 apart, the ends 0.75."""
    position = {normalize_instruction(text): i * 0.25 for i, text in enumerate(_CHAIN)}
    return abs(position[left] - position[right])


def test_a_transitively_chained_family_is_reported_as_over_merged() -> None:
    """Every edge legal at 0.25, the component 0.75 wide: the span no link explains."""
    task_set = mine_task_set(
        _chain_analysis(4), "analysis-x", budget=10, similarity=0.7, distance=_chain_distance
    )

    family = next(f for f in task_set.families if f.workload_mass == 4)
    assert family.over_merged
    assert family.coherence.diameter == pytest.approx(0.75)
    assert family.coherence.widest_pair == (
        normalize_instruction(_CHAIN[0]),
        normalize_instruction(_CHAIN[3]),
    )


def test_the_flag_names_the_pair_that_spans_the_family() -> None:
    task_set = mine_task_set(
        _chain_analysis(4), "analysis-x", budget=10, similarity=0.7, distance=_chain_distance
    )

    limitation = next(
        limit for f in task_set.families for limit in f.limitations if "widest pair" in limit
    )
    assert "descriptor alpha" in limitation
    assert "descriptor delta" in limitation


def test_a_family_no_wider_than_one_legal_link_is_not_flagged() -> None:
    task_set = mine_task_set(
        _chain_analysis(2), "analysis-x", budget=10, similarity=0.7, distance=_chain_distance
    )

    family = next(f for f in task_set.families if f.workload_mass == 2)
    assert not family.over_merged
    assert family.coherence.diameter == pytest.approx(0.25)


def test_flagging_changes_no_grouping(task_set: TaskSet, analysis) -> None:
    """Advisory only: a different factor must move no trace between families."""
    loose = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        similarity=0.7,
        budget=10,
        diameter_factor=99.0,
    )

    assert [f.trace_ids for f in loose.families] == [f.trace_ids for f in task_set.families]
    assert not any(f.over_merged for f in loose.families)


def test_invalid_clustering_parameters_are_refused(analysis) -> None:
    """An out-of-range similarity inverts the threshold and flags every family."""
    analysis_id = compute_analysis_id(analysis)

    with pytest.raises(ValueError, match="similarity must be between 0 and 1"):
        mine_task_set(analysis, analysis_id, distance=_test_distance, similarity=1.5)
    with pytest.raises(ValueError, match="diameter_factor must be positive"):
        mine_task_set(
            analysis,
            analysis_id,
            distance=_test_distance,
            similarity=0.7,
            diameter_factor=0.0,
        )


def test_a_merge_says_its_coherence_was_not_recomputed(task_set: TaskSet) -> None:
    """A merged family is wider than either input; a stale figure would understate it."""
    ids = tuple(f.family_id for f in task_set.families[:2])

    merged = merge_families(task_set, ids)

    survivor = next(f for f in merged.families if f.family_id in ids)
    assert survivor.coherence is None
    assert not survivor.over_merged
    assert any("not recomputed" in limit for limit in survivor.limitations)


def test_a_merge_drops_stale_coherence_limitations(task_set: TaskSet) -> None:
    """A widest pair measured before a merge says nothing about the merged family."""
    first, second, *rest = task_set.families
    stale = "widest pair inside this family is 0.75 apart, over the old threshold"
    flagged = first.replace(
        coherence=FamilyCoherence(
            diameter=0.75,
            link_threshold=0.4,
            diameter_factor=1.25,
            widest_pair=("old-left", "old-right"),
        ),
        limitations=(*first.limitations, stale),
    )
    assert flagged.over_merged
    source = task_set.replace(families=(flagged, second, *rest))

    merged = merge_families(source, (first.family_id, second.family_id))
    survivor = merged.family_by_id()[first.family_id]

    assert stale not in survivor.limitations
    assert not any(limit.startswith("widest pair") for limit in survivor.limitations)
    assert any("coherence was not recomputed" in limit for limit in survivor.limitations)


def test_a_split_drops_the_parents_over_merge_finding(analysis) -> None:
    """The finding was about the parent's span, which no subfamily still has."""
    task_set = mine_task_set(
        _chain_analysis(4), "analysis-x", budget=10, similarity=0.7, distance=_chain_distance
    )
    flagged = next(f for f in task_set.families if f.over_merged)

    split = split_family(task_set, flagged.family_id, _chain_analysis(4))

    replacements = [f for f in split.families if f.review_status == "split"]
    assert replacements
    assert all(f.coherence is None for f in replacements)
    assert not any(limit.startswith("widest pair") for f in replacements for limit in f.limitations)
