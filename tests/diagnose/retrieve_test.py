"""Retrieval, exercised on the leakage boundaries it exists to hold."""

from __future__ import annotations

from bandits.diagnose.models import (
    ActionCall,
    ExpectedEffect,
    GroundingObservation,
    GroundingTransition,
    Partition,
    Scenario,
    ScenarioKind,
    ScenarioState,
    SealedSuccessContract,
    StateField,
    SuccessShape,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
    WorldOrigin,
)
from bandits.diagnose.retrieve import (
    RetrievalQuery,
    build_index,
    coverage_by_tool,
    query_for,
    retrieve,
    support_level,
)


def _transition(
    tid,
    trace,
    tool="cancel_reservation",
    shape=SuccessShape.MUTATION,
    error=False,
    role="tool",
    context="cancel my reservation",
    state=(),
):
    return GroundingTransition(
        transition_id=tid,
        trace_id=trace,
        lineage_id=trace,
        family_id="family-451ae91f975c",
        turn_index=0,
        task_context=context,
        success_shape=shape,
        state_before=ScenarioState(
            fields=tuple(
                StateField(path=p, value=1, origin=WorldOrigin.RECORDED, revealed_by_span_id="s")
                for p in state
            )
        ),
        action_span_id=f"span-{tid}",
        action_calls=(ActionCall(tool=tool, arguments={"reservation_id": "A"}),) if tool else (),
        observations=(GroundingObservation(role=role, content={"status": "ok"}, error=error),),
    )


def _query(**kw) -> RetrievalQuery:
    base = dict(
        family_id="family-451ae91f975c",
        shape=SuccessShape.MUTATION,
        task_context="cancel my reservation",
        tools=("cancel_reservation",),
    )
    return RetrievalQuery(**{**base, **kw})


def _catalog():
    return ToolEffectCatalog(
        catalog_id="c",
        toolset_digest="d",
        entries=(
            ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="a"),
            ToolEffectEntry(
                tool="get_reservation_details", effect=ToolEffect.READ, reviewed_by="a"
            ),
        ),
    )


# --- the leakage boundaries ---------------------------------------------


def test_index_holds_fit_only() -> None:
    """Grounding on held-out reports memorisation as generalisation."""
    transitions = (_transition("t1", "airline-10"), _transition("t2", "airline-11"))
    index = build_index(transitions, fit_trace_ids=("airline-10",))
    assert [t.trace_id for t in index] == ["airline-10"]


def test_index_never_holds_sealed_traces() -> None:
    """Sealed is a different exclusion from held-out, with a different reason."""
    transitions = (_transition("t1", "airline-10"),)
    index = build_index(
        transitions, fit_trace_ids=("airline-10",), sealed_trace_ids=("airline-10",)
    )
    assert index == ()


def test_index_excludes_unobserved_transitions() -> None:
    unobserved = GroundingTransition(
        transition_id="t9",
        trace_id="airline-10",
        family_id="f",
        turn_index=0,
        action_span_id="s",
        observed=False,
    )
    assert build_index((unobserved,), fit_trace_ids=("airline-10",)) == ()


def test_query_excludes_its_own_lineage_at_query_time() -> None:
    """Build-time filtering is not enough; one wrong caller is all it takes."""
    index = (_transition("t1", "airline-10"), _transition("t2", "airline-99"))
    found = retrieve(_query(excluded_trace_ids=("airline-10",)), index)
    assert [e.transition.trace_id for e in found] == ["airline-99"]


def test_a_sealed_scenario_retrieves_from_fit_but_never_from_sealed() -> None:
    """The exclusion runs one way.

    A sealed scenario still needs grounding to be runnable at all; what must
    never happen is any scenario reading a *sealed transition*, which would
    spend the one clean measurement the split exists to hold.
    """
    index = (_transition("t1", "airline-10"), _transition("t2", "airline-77"))
    found = retrieve(_query(partition=Partition.SEALED), index, sealed_trace_ids=("airline-77",))
    assert [e.transition.trace_id for e in found] == ["airline-10"]


def test_evidence_never_crosses_a_family_boundary() -> None:
    """A shared index spans families; another family is another world's entities."""
    other = _transition("t1", "airline-50").replace(family_id="family-other")
    assert retrieve(_query(), (other,)) == ()


def test_the_tool_world_is_not_grounded_on_a_user_turn() -> None:
    """A turn where the tool was called and a person replied says nothing
    about what the tool returns."""
    user_only = _transition("t1", "airline-10", role="user")
    assert retrieve(_query(response_role="tool_world"), (user_only,)) == ()
    assert retrieve(_query(response_role="user_policy"), (user_only,))


def test_user_policy_evidence_must_match_the_scenario_shape() -> None:
    mutation_user = _transition(
        "t1", "airline-10", role="user", shape=SuccessShape.MUTATION
    )
    assert retrieve(
        _query(response_role="user_policy", shape=SuccessShape.REFUSAL),
        (mutation_user,),
    ) == ()


def test_support_requires_a_tool_answer_not_two_name_matches() -> None:
    """Two actions naming the tool whose reactions were user turns are not
    evidence that the tool does anything."""
    named_only = tuple(_transition(f"t{i}", f"airline-{i}", role="user") for i in range(3))
    found = retrieve(_query(response_role="user_policy"), named_only)
    assert support_level(found, role="tool_world") in ("low", "none")


# --- compatibility filtering happens before ranking ---------------------


def test_a_refusal_scenario_may_read_mutation_evidence() -> None:
    """Tool semantics are not agent policy.

    What ``cancel_reservation`` does when called is one fact; whether the agent
    should have called it is another. Blocking mutation evidence here made the
    tool world abstain on an improper call, so the environment refused on the
    agent's behalf and the verifier never saw the mistake — which is exactly
    what D27 forbids.
    """
    index = (
        _transition("t1", "airline-10", tool="cancel_reservation", shape=SuccessShape.MUTATION),
    )
    found = retrieve(_query(shape=SuccessShape.REFUSAL, tools=("cancel_reservation",)), index)
    assert found
    assert found[0].transition.success_shape is SuccessShape.MUTATION


def test_an_unbound_transition_is_still_usable_for_tool_semantics() -> None:
    """Shape gates the user policy, whose behaviour is the task's.

    Tool semantics are the system's, and an unbound source task does not change
    what ``cancel_reservation`` returns.
    """
    index = (_transition("t1", "airline-10", shape=None),)
    assert retrieve(_query(response_role="tool_world"), index)


def test_reads_do_not_stand_in_for_writes() -> None:
    index = (
        _transition(
            "t1", "airline-10", tool="get_reservation_details", shape=SuccessShape.MUTATION
        ),
    )
    found = retrieve(_query(tools=("cancel_reservation",)), index, catalog=_catalog())
    assert found == ()


def test_an_unreviewed_toolset_is_not_filtered_by_guesswork() -> None:
    """With no catalog the effect filter does not fire, rather than inferring."""
    index = (_transition("t1", "airline-10", tool="mystery_tool"),)
    assert retrieve(_query(tools=("cancel_reservation",)), index, catalog=None)


# --- ranking and contrast -----------------------------------------------


def test_exact_tool_outranks_a_merely_similar_one() -> None:
    index = (
        _transition("t1", "airline-10", tool="book_reservation"),
        _transition("t2", "airline-11", tool="cancel_reservation"),
    )
    found = retrieve(_query(), index)
    assert found[0].transition.action_calls[0].tool == "cancel_reservation"
    assert "exact_tool" in found[0].reasons


def test_shared_state_raises_the_score() -> None:
    plain = _transition("t1", "airline-10")
    shared = _transition("t2", "airline-11", state=("reservations.A.status",))
    found = retrieve(_query(state_keys=("reservations.A.status",)), (plain, shared))
    assert found[0].transition.transition_id == "t2"
    assert "shared_state" in found[0].reasons


def test_an_error_case_is_kept_even_when_outranked() -> None:
    """A simulator shown only successes learns the action always succeeds."""
    index = tuple(_transition(f"t{i}", f"airline-{i}") for i in range(6)) + (
        _transition("terr", "airline-99", error=True),
    )
    found = retrieve(_query(), index, limit=3)
    assert any("error_case" in e.reasons for e in found)


def test_reserved_slots_do_not_overwrite_each_other() -> None:
    """An earlier version wrote both reserved categories to the same slot."""
    index = tuple(_transition(f"t{i}", f"airline-{i}") for i in range(6)) + (
        _transition("terr", "airline-98", error=True),
    )
    found = retrieve(_query(), index, limit=4)
    assert any("error_case" in e.reasons for e in found)
    assert len(found) == 4
    assert len({e.transition.transition_id for e in found}) == 4


def test_results_are_deterministic() -> None:
    index = tuple(_transition(f"t{i}", f"airline-{i}") for i in range(5))
    assert [e.transition.transition_id for e in retrieve(_query(), index)] == [
        e.transition.transition_id for e in retrieve(_query(), index)
    ]


# --- support ------------------------------------------------------------


def test_support_is_conservative_where_evidence_is_thin() -> None:
    """cancel_reservation never failed once in 26 calls; generosity invents refusals."""
    assert support_level(()) == "none"
    index = (_transition("t1", "airline-10"),)
    one = retrieve(_query(), index)
    assert support_level(one) == "medium"
    two = retrieve(_query(), index + (_transition("t2", "airline-11"),))
    assert support_level(two) == "high"


def test_coverage_reports_which_tools_have_no_error_evidence() -> None:
    index = (
        _transition("t1", "airline-10", tool="cancel_reservation"),
        _transition("t2", "airline-11", tool="get_user_details", error=True),
    )
    table = coverage_by_tool(index)
    assert table["cancel_reservation"] == {"total": 1, "errors": 0}
    assert table["get_user_details"] == {"total": 1, "errors": 1}


# --- query construction -------------------------------------------------


def test_query_carries_the_scenario_exclusions() -> None:
    scenario = Scenario(
        scenario_id="s1",
        kind=ScenarioKind.TASK_START,
        task="cancel my flight",
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="7",
            shape=SuccessShape.MUTATION,
            required_effects=(ExpectedEffect(effect_id="e1", tool="cancel_reservation"),),
        ),
        source_trace_id="airline-10",
        source_task_id="7",
        family_id="family-451ae91f975c",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("airline-10", "airline-11"),
    )
    query = query_for(scenario, state=ScenarioState(), tools=("cancel_reservation",))
    assert query.excluded_trace_ids == ("airline-10", "airline-11")
    assert query.shape is SuccessShape.MUTATION
