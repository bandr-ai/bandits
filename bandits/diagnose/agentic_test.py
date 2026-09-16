"""Boundary tests for the agentic AWM's grounding tools, before any paid call.

Per review: written and passing before the real-model smoke runner exists,
using fake tool-call sequences -- the same order I39 itself was actually
verified in (tests first, real-artifact rescore only after). Every claim here
is about what code enforces, not what a model is instructed to do: an AWM
instructed not to enumerate the hidden world is not the same guarantee as an
AWM whose tool literally cannot return anything outside its allowed entities.
"""

from __future__ import annotations

import pytest

from bandits.diagnose.agentic import (
    AWMExecutionTrace,
    AWMRuntimeContext,
    AWMToolCall,
    BudgetExceeded,
    build_agentic_tool_world_predictor,
    build_grounding_tools,
    make_inspect_grounding_history,
    make_inspect_tool_contract,
    make_read_world_state,
    make_search_transitions,
    step_agentic_tool_world,
    with_budget,
)
from bandits.diagnose.models import (
    ActionCall,
    GroundingObservation,
    GroundingTransition,
    Partition,
    ScenarioState,
    StateField,
    SupportLevel,
    WorldOrigin,
)
from bandits.diagnose.world import (
    ProposedCallOutcome,
    ProposedTransition,
    StateDelta,
    validate_transition,
)


def _transition(tid, tool, arguments, observation, trace="airline-1", reasons_hint=None):
    return GroundingTransition(
        transition_id=tid,
        trace_id=trace,
        family_id="f",
        turn_index=0,
        action_span_id=f"span-{tid}",
        action_calls=(ActionCall(tool=tool, arguments=arguments),),
        observations=(GroundingObservation(role="tool", content=observation),),
    )


def _context(**overrides):
    base = dict(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
        ),
        current_state=ScenarioState(),
        history_text="",
        offered_tool_schemas=({"name": "get_user_details", "parameters": {}},),
        fit_index=(),
        family_id="f",
        task_context="",
        excluded_trace_ids=(),
        prior_grounding_calls=(),
    )
    base.update(overrides)
    return AWMRuntimeContext(**base)


# --- inspect_tool_contract -------------------------------------------------


def test_inspect_tool_contract_only_offered_tools() -> None:
    context = _context(offered_tool_schemas=({"name": "get_user_details", "parameters": {}},))
    audit: list[AWMToolCall] = []
    tool = make_inspect_tool_contract(context, audit=audit)

    offered = tool("get_user_details")
    assert offered["offered"] is True
    assert "schema" in offered

    not_offered = tool("cancel_reservation")
    assert not_offered["offered"] is False
    assert "schema" not in not_offered


def test_inspect_tool_contract_never_exposes_verifier_fields() -> None:
    context = _context(offered_tool_schemas=({"name": "get_user_details", "parameters": {}},))
    audit: list[AWMToolCall] = []
    tool = make_inspect_tool_contract(context, audit=audit)
    result = tool("get_user_details")
    assert "success_contract" not in result
    assert "verifier" not in result
    assert "expected_result" not in result


# --- read_world_state -------------------------------------------------


def test_read_world_state_restricted_to_candidate_named_entities() -> None:
    """The candidate's action named raj_sanchez_7340. Asking about a
    different, unrelated user must be refused, not answered from whatever
    else happens to be in state -- there is nothing else in state here, but
    the boundary must hold even if there were."""
    context = _context(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
        ),
    )
    audit: list[AWMToolCall] = []
    tool = make_read_world_state(context, audit=audit)

    disallowed = tool("user", "emma_kim_9957")
    assert disallowed["found"] is False
    assert disallowed["completeness"] == "unknown"
    assert disallowed["fields"] == {}
    assert audit[-1].error is not None


def test_read_world_state_does_not_grant_access_merely_from_state_presence() -> None:
    """The fixed over-broad-access bug: an entity having a record somewhere
    in state (raj_sanchez_7340, from some unrelated earlier context) is NOT
    itself a relationship to the candidate's own named entity (Q1). Only a
    read of Q1 that discloses a user_id field pointing at Raj would create
    that link -- his mere presence in the state object must not."""
    state = ScenarioState(
        fields=(
            StateField(
                path="get_user_details.raj_sanchez_7340.name",
                value="Raj Sanchez",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    context = _context(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_reservation_details", arguments={"reservation_id": "Q1"}),
        ),
        current_state=state,
    )
    audit: list[AWMToolCall] = []
    tool = make_read_world_state(context, audit=audit)

    result = tool("user", "raj_sanchez_7340")
    assert result["found"] is False
    assert result["completeness"] == "unknown"
    assert audit[-1].error is not None


def test_read_world_state_allows_entity_linked_by_an_actual_relationship() -> None:
    """The real, narrower relationship-traversal case: Q1's own state
    discloses user_id=raj_sanchez_7340 -- reading Raj is now allowed, because
    an actual field on the already-read reservation points at him, not
    because his data happens to sit somewhere else in the state object."""
    state = ScenarioState(
        fields=(
            StateField(
                path="get_reservation_details.Q1.user_id",
                value="raj_sanchez_7340",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
            StateField(
                path="get_user_details.raj_sanchez_7340.name",
                value="Raj Sanchez",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    context = _context(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_reservation_details", arguments={"reservation_id": "Q1"}),
        ),
        current_state=state,
    )
    audit: list[AWMToolCall] = []
    tool = make_read_world_state(context, audit=audit)

    # Q1 itself is named by the candidate's call, so reading it is always
    # allowed -- this is what makes the user_id relationship "already read."
    reservation = tool("reservation", "Q1")
    assert reservation["found"] is True

    user = tool("user", "raj_sanchez_7340")
    assert user["found"] is True
    assert user["fields"]["name"] == "Raj Sanchez"


def test_read_world_state_distinguishes_not_found_from_unknown() -> None:
    """found=False for an allowed entity with no state means 'never
    reconstructed' (completeness=unknown) -- never 'confirmed absent'."""
    context = _context(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
        ),
        current_state=ScenarioState(),
    )
    audit: list[AWMToolCall] = []
    tool = make_read_world_state(context, audit=audit)

    result = tool("user", "raj_sanchez_7340")
    assert result["found"] is False
    assert result["completeness"] == "unknown"


def test_read_world_state_records_state_paths_read() -> None:
    state = ScenarioState(
        fields=(
            StateField(
                path="get_user_details.raj_sanchez_7340.name",
                value="Raj Sanchez",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    context = _context(current_state=state)
    audit: list[AWMToolCall] = []
    tool = make_read_world_state(context, audit=audit)
    tool("user", "raj_sanchez_7340")

    assert audit[-1].state_paths_read == ("get_user_details.raj_sanchez_7340.name",)


# --- search_transitions -------------------------------------------------


def test_search_transitions_always_fit_partition() -> None:
    """The tool's signature has no partition argument at all -- it cannot be
    set from anything the model supplies. This test asserts the query built
    internally is always Partition.FIT regardless of arguments."""
    from bandits.diagnose import agentic as agentic_module

    captured_queries = []
    original_retrieve = agentic_module.retrieve

    def spy_retrieve(query, index, *, limit):
        captured_queries.append(query)
        return original_retrieve(query, index, limit=limit)

    agentic_module.retrieve = spy_retrieve
    try:
        context = _context(fit_index=())
        audit: list[AWMToolCall] = []
        tool = make_search_transitions(context, audit=audit)
        tool(tool="get_user_details", query="anything")
        assert all(q.partition is Partition.FIT for q in captured_queries)
    finally:
        agentic_module.retrieve = original_retrieve


def test_search_transitions_excludes_configured_trace_ids() -> None:
    fit_index = (
        _transition("t1", "get_user_details", {"user_id": "x"}, {"name": "X"}, trace="airline-10"),
        _transition("t2", "get_user_details", {"user_id": "y"}, {"name": "Y"}, trace="airline-99"),
    )
    context = _context(
        fit_index=fit_index,
        family_id="f",
        excluded_trace_ids=("airline-10",),
    )
    audit: list[AWMToolCall] = []
    tool = make_search_transitions(context, audit=audit)
    result = tool(tool="get_user_details", query="")

    trace_ids = {r["transition_id"] for r in result["results"]}
    assert "t1" not in trace_ids


def test_search_transitions_caps_result_count() -> None:
    from bandits.diagnose import agentic as agentic_module

    fit_index = tuple(
        _transition(f"t{i}", "get_user_details", {"user_id": f"u{i}"}, {"name": f"U{i}"})
        for i in range(20)
    )
    context = _context(fit_index=fit_index)
    audit: list[AWMToolCall] = []
    tool = make_search_transitions(context, audit=audit)
    result = tool(tool="get_user_details", query="", limit=1000)

    assert len(result["results"]) <= agentic_module.MAX_SEARCH_RESULTS


def test_search_transitions_result_carries_evidence_ids_in_audit() -> None:
    fit_index = (_transition("t1", "get_user_details", {"user_id": "x"}, {"name": "X"}),)
    context = _context(fit_index=fit_index)
    audit: list[AWMToolCall] = []
    tool = make_search_transitions(context, audit=audit)
    tool(tool="get_user_details", query="")

    assert "t1" in audit[-1].evidence_ids


def test_search_transitions_entity_id_is_a_hard_filter_not_just_ranking_text() -> None:
    """The I39-in-a-new-location bug: entity_id used to only feed ranking
    text, so a same-tool-different-entity result silently counted as support
    for the requested entity. Now every result is labeled entity_matched, and
    only matched results land in exact_entity_evidence_ids."""
    fit_index = (
        _transition("t1", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}),
        _transition("t2", "get_user_details", {"user_id": "anya_garcia_5901"}, {"name": "Anya Garcia"}),
    )
    context = _context(fit_index=fit_index)
    audit: list[AWMToolCall] = []
    tool = make_search_transitions(context, audit=audit)
    result = tool(tool="get_user_details", query="", entity_id="raj_sanchez_7340", limit=5)

    by_id = {r["transition_id"]: r for r in result["results"]}
    assert by_id["t1"]["entity_matched"] is True
    assert by_id["t2"]["entity_matched"] is False
    assert audit[-1].exact_entity_evidence_ids == ("t1",)
    # Both still returned as behavioral evidence -- the fix restricts what
    # counts as a value claim, not what the AWM is allowed to see at all.
    assert set(audit[-1].evidence_ids) == {"t1", "t2"}


def test_search_transitions_no_entity_id_means_no_exact_matches() -> None:
    """A pure shape/behavior search (no entity_id given) never claims any
    exact-entity evidence, however many results come back."""
    fit_index = (_transition("t1", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}),)
    context = _context(fit_index=fit_index)
    audit: list[AWMToolCall] = []
    tool = make_search_transitions(context, audit=audit)
    tool(tool="get_user_details", query="")

    assert audit[-1].exact_entity_evidence_ids == ()


# --- inspect_grounding_history -------------------------------------------------


def test_inspect_grounding_history_sees_only_this_episodes_prior_calls() -> None:
    prior = (
        AWMToolCall(
            index=0,
            tool="read_world_state",
            arguments={"entity_kind": "user", "entity_id": "raj_sanchez_7340"},
            result_summary="found=True",
            entity=("user", "raj_sanchez_7340"),
        ),
    )
    context = _context(prior_grounding_calls=prior)
    audit: list[AWMToolCall] = []
    tool = make_inspect_grounding_history(context, audit=audit)

    result = tool("user", "raj_sanchez_7340")
    assert len(result["prior_grounding_calls"]) == 1

    result_other = tool("user", "emma_kim_9957")
    assert len(result_other["prior_grounding_calls"]) == 0


def test_inspect_grounding_history_entity_kind_matters_not_just_id() -> None:
    """The fixed entity_kind-ignored bug: a "reservation" and a "user" that
    happen to share an id-shaped string must not cross-match."""
    prior = (
        AWMToolCall(
            index=0,
            tool="read_world_state",
            arguments={"entity_kind": "reservation", "entity_id": "SHARED_ID"},
            result_summary="found=True",
            entity=("reservation", "SHARED_ID"),
        ),
    )
    context = _context(prior_grounding_calls=prior)
    audit: list[AWMToolCall] = []
    tool = make_inspect_grounding_history(context, audit=audit)

    matching = tool("reservation", "SHARED_ID")
    assert len(matching["prior_grounding_calls"]) == 1

    mismatched_kind = tool("user", "SHARED_ID")
    assert len(mismatched_kind["prior_grounding_calls"]) == 0


def test_inspect_grounding_history_never_exposes_verifier_or_future_transitions() -> None:
    context = _context(prior_grounding_calls=())
    audit: list[AWMToolCall] = []
    tool = make_inspect_grounding_history(context, audit=audit)
    result = tool("user", "raj_sanchez_7340")
    assert set(result.keys()) == {"prior_grounding_calls"}


# --- budget -------------------------------------------------


def test_budget_exceeded_raises_after_max_calls() -> None:
    audit: list[AWMToolCall] = []
    rejections: list[int] = []

    def noop(**_):
        audit.append(AWMToolCall(index=len(audit), tool="noop"))
        return {}

    wrapped = with_budget(noop, audit=audit, max_calls=2, rejections=rejections)
    wrapped()
    wrapped()
    with pytest.raises(BudgetExceeded):
        wrapped()
    assert rejections == [1]


def test_build_grounding_tools_shares_one_audit_across_all_four() -> None:
    context = _context()
    tools, audit, _rejections = build_grounding_tools(context, max_calls=10)
    assert len(tools) == 4

    tools[0]("get_user_details")  # inspect_tool_contract
    tools[2](tool="get_user_details", query="")  # search_transitions
    assert len(audit) == 2
    assert audit[0].tool == "inspect_tool_contract"
    assert audit[1].tool == "search_transitions"


def test_build_grounding_tools_budget_shared_across_tool_types() -> None:
    """The budget must be per-episode, not per-tool: 3 calls to
    inspect_tool_contract plus 1 more to any tool must exhaust a budget of 3
    total, not reset per tool."""
    context = _context()
    tools, audit, _rejections = build_grounding_tools(context, max_calls=3)
    inspect_contract = tools[0]
    read_state = tools[1]

    inspect_contract("get_user_details")
    inspect_contract("get_user_details")
    inspect_contract("get_user_details")
    with pytest.raises(BudgetExceeded):
        read_state("user", "raj_sanchez_7340")


def test_budget_exhaustion_is_detectable_from_audit_length_alone() -> None:
    """len(audit) >= max_calls is a sound *exhaustion* signal (audit never
    exceeds max_calls, so this check reliably fires once the budget is used
    up) -- but exhaustion is not the same as rejection. See
    test_rejection_counter_distinguishes_actual_overrun_from_clean_finish."""
    context = _context()
    tools, audit, _rejections = build_grounding_tools(context, max_calls=2)
    inspect_contract = tools[0]

    inspect_contract("get_user_details")
    assert len(audit) == 1
    inspect_contract("get_user_details")
    assert len(audit) == 2

    # Simulate dspy.ReAct.forward's own behavior: it catches the exception
    # from the 3rd call and continues rather than propagating it.
    try:
        inspect_contract("get_user_details")
    except BudgetExceeded:
        pass

    assert len(audit) == 2  # never exceeds max_calls
    assert len(audit) >= 2  # exhaustion is detectable purely from this


def test_rejection_counter_distinguishes_actual_overrun_from_clean_finish() -> None:
    """The fixed inaccurate-exhaustion-signal bug: len(audit) == max_calls
    alone cannot distinguish "the model used exactly its budget and then
    cleanly finished" from "the model tried to go over and was refused."
    Only the rejections counter tells them apart."""
    context = _context()

    # Case 1: model uses exactly its budget, no overrun attempted.
    tools, audit, rejections = build_grounding_tools(context, max_calls=2)
    tools[0]("get_user_details")
    tools[0]("get_user_details")
    assert len(audit) == 2  # exhausted_budget would read True
    assert rejections == []  # but no rejection ever happened

    # Case 2: model actually tries to exceed budget.
    tools2, audit2, rejections2 = build_grounding_tools(context, max_calls=2)
    tools2[0]("get_user_details")
    tools2[0]("get_user_details")
    with pytest.raises(BudgetExceeded):
        tools2[0]("get_user_details")
    assert len(audit2) == 2  # same audit length as case 1
    assert rejections2 == [1]  # but this time a rejection is recorded


# --- inspect_grounding_history sees the live episode, not just prior steps -


def test_inspect_grounding_history_sees_calls_made_earlier_in_the_same_live_episode() -> None:
    """The fix for the disconnected-history bug: inspect_grounding_history
    must see a call made two tool-invocations earlier *within this same
    episode*, not only context.prior_grounding_calls (which is fixed before
    the episode starts and, before multi-step integration exists, is always
    empty)."""
    context = _context(prior_grounding_calls=())
    tools, audit, _rejections = build_grounding_tools(context, max_calls=10)
    inspect_contract, read_state, _search, inspect_history = tools

    read_state("user", "raj_sanchez_7340")
    assert len(audit) == 1

    result = inspect_history("user", "raj_sanchez_7340")
    matches = result["prior_grounding_calls"]
    assert len(matches) == 1
    assert matches[0]["tool"] == "read_world_state"


def test_inspect_grounding_history_still_sees_prior_step_calls_when_seeded() -> None:
    """The context.prior_grounding_calls source is not removed by the fix --
    a future multi-step caller that seeds it must still have those calls
    visible, merged with whatever this episode's own live audit adds."""
    earlier_step = AWMToolCall(
        index=0,
        tool="read_world_state",
        arguments={"entity_kind": "user", "entity_id": "raj_sanchez_7340"},
        result_summary="found=True",
        entity=("user", "raj_sanchez_7340"),
    )
    context = _context(prior_grounding_calls=(earlier_step,))
    tools, audit, _rejections = build_grounding_tools(context, max_calls=10)
    _inspect_contract, _read_state, _search, inspect_history = tools

    result = inspect_history("user", "raj_sanchez_7340")
    assert len(result["prior_grounding_calls"]) == 1
    assert len(audit) == 1  # only inspect_grounding_history's own call, not the seeded one


# --- AWMExecutionTrace -------------------------------------------------


def test_execution_trace_deduplicates_evidence_ids_across_calls() -> None:
    trace = AWMExecutionTrace(
        grounding_calls=(
            AWMToolCall(index=0, tool="search_transitions", evidence_ids=("t1", "t2")),
            AWMToolCall(index=1, tool="search_transitions", evidence_ids=("t2", "t3")),
        )
    )
    assert trace.all_evidence_ids == ("t1", "t2", "t3")


def test_execution_trace_deduplicates_state_paths_across_calls() -> None:
    trace = AWMExecutionTrace(
        grounding_calls=(
            AWMToolCall(index=0, tool="read_world_state", state_paths_read=("a.b", "a.c")),
            AWMToolCall(index=1, tool="read_world_state", state_paths_read=("a.c", "a.d")),
        )
    )
    assert trace.all_state_paths_read == ("a.b", "a.c", "a.d")


def test_execution_trace_deduplicates_exact_entity_evidence_ids_across_calls() -> None:
    trace = AWMExecutionTrace(
        grounding_calls=(
            AWMToolCall(index=0, tool="search_transitions", exact_entity_evidence_ids=("t1",)),
            AWMToolCall(index=1, tool="search_transitions", exact_entity_evidence_ids=("t1", "t2")),
        )
    )
    assert trace.all_exact_entity_evidence_ids == ("t1", "t2")


def test_execution_trace_empty_when_no_calls_made() -> None:
    trace = AWMExecutionTrace()
    assert trace.all_evidence_ids == ()
    assert trace.all_exact_entity_evidence_ids == ()
    assert trace.all_state_paths_read == ()
    assert trace.entities_grounded() == set()


def test_entities_grounded_requires_actual_hits_not_just_any_call() -> None:
    """A read_world_state call that found nothing (state_paths_read empty) or
    a search that matched no entity (exact_entity_evidence_ids empty) must
    not register as grounded -- only a call that actually produced something
    for its entity counts."""
    trace = AWMExecutionTrace(
        grounding_calls=(
            AWMToolCall(
                index=0,
                tool="read_world_state",
                entity=("user", "nobody"),
                state_paths_read=(),  # found nothing
            ),
            AWMToolCall(
                index=1,
                tool="search_transitions",
                entity=("user", "also_nobody"),
                exact_entity_evidence_ids=(),  # matched nothing
            ),
        )
    )
    assert trace.entities_grounded() == set()


# --- step_agentic_tool_world: integration with fake predictors -------------


def test_budget_exhaustion_becomes_abstention_not_a_crash() -> None:
    def fake_predict(*, instruction, context):
        return None, AWMExecutionTrace(exhausted_budget=True)

    proposal, trace = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.abstain is True
    assert trace.exhausted_budget is True


def test_malformed_output_is_output_invalid_not_abstain() -> None:
    def fake_predict(*, instruction, context):
        return {"not": "a valid transition shape", "call_outcomes": "garbage"}, AWMExecutionTrace()

    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.output_invalid is True
    assert proposal.abstain is False


def test_support_capped_when_nothing_was_grounded() -> None:
    """The model claims 'high' support but grounded nothing at all -- the
    claim must not survive uncapped."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(index=0, tool="read_world_state", arguments={}, result_summary="found"),
            )
        )
        return (
            ProposedTransition(
                observation={"status": "ok"},
                support=SupportLevel.HIGH,
                evidence_ids=(),
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.support is SupportLevel.NONE


def test_support_survives_when_the_candidates_own_target_entity_is_state_backed() -> None:
    """A call that grounded the CANDIDATE'S OWN target entity entirely through
    read_world_state (state_values_read populated with the exact claimed
    value, zero search calls) must not be capped down just because no search
    happened -- Experiment A's cancellation is exactly this shape.
    _context()'s default candidate call names user:raj_sanchez_7340, so the
    grounding call below matches it."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    arguments={"entity_kind": "user", "entity_id": "raj_sanchez_7340"},
                    result_summary="found=True, 14 field(s)",
                    state_paths_read=("get_user_details.raj_sanchez_7340.name",),
                    state_values_read={"name": "Raj Sanchez"},
                    entity=("user", "raj_sanchez_7340"),
                ),
            )
        )
        return (
            ProposedTransition(observation={"name": "Raj Sanchez"}, support=SupportLevel.HIGH),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.support is SupportLevel.HIGH
    assert proposal.observation == {"name": "Raj Sanchez"}  # exact match: not nulled


def test_support_not_laundered_from_a_different_entitys_grounding() -> None:
    """The central fix this round closes: grounding some OTHER entity (Anya)
    this episode must not license HIGH support for the candidate's actual
    target entity (Raj) just because "something was grounded somewhere."
    This is the exact failure mode the review flagged as still open."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    arguments={"entity_kind": "user", "entity_id": "anya_garcia_5901"},
                    result_summary="found=True, 14 field(s)",
                    state_paths_read=("get_user_details.anya_garcia_5901.name",),
                    entity=("user", "anya_garcia_5901"),
                ),
            )
        )
        return (
            ProposedTransition(observation={"name": "Raj Sanchez"}, support=SupportLevel.HIGH),
            trace,
        )

    # _context()'s default candidate call targets user:raj_sanchez_7340, not
    # anya_garcia_5901 -- the grounding above concerns a different entity.
    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.support is SupportLevel.NONE


def test_support_high_for_entity_matched_search_with_the_exact_claimed_value() -> None:
    """Search evidence that actually matched the candidate's own target
    entity (exact_entity_values carrying the specific claimed field) grounds
    it as an exact fact, not merely a behavioral analogy."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="search_transitions",
                    evidence_ids=("t1",),
                    exact_entity_evidence_ids=("t1",),
                    exact_entity_values={"name": "Raj Sanchez"},
                    entity=("user", "raj_sanchez_7340"),
                ),
            )
        )
        return (
            ProposedTransition(observation={"name": "Raj Sanchez"}, support=SupportLevel.HIGH),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.support is SupportLevel.HIGH
    assert proposal.observation == {"name": "Raj Sanchez"}


def test_support_low_for_search_that_never_entity_matched() -> None:
    """Search results *about the right tool* but that never actually matched
    the candidate's target entity (entity_matched: false for all of them)
    can still license a shape/behavior claim (BEHAVIORAL_ANALOGY -> LOW), but
    must never promote a specific claimed value to HIGH/EXACT -- the
    I39-in-search bug, at the capping layer. BEHAVIORAL_ANALOGY is a
    downgrade, not a rejection: the field survives (unlike UNSUPPORTED,
    which nulls it), but support is capped to LOW."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="search_transitions",
                    arguments={"tool": "get_user_details"},
                    evidence_ids=("t1", "t2"),
                    exact_entity_evidence_ids=(),  # neither result actually matched
                    entity=("user", "raj_sanchez_7340"),
                ),
            )
        )
        return (
            ProposedTransition(observation={"name": "Raj Sanchez"}, support=SupportLevel.HIGH),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_context())
    assert proposal.support is SupportLevel.LOW
    assert proposal.observation == {"name": "Raj Sanchez"}  # not nulled -- analogy, not invention


def test_final_proposal_still_passes_external_validation() -> None:
    """The agentic AWM's output is not exempt from the same validate_transition
    gate everything else goes through -- it is not a second, looser path."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="search_transitions",
                    evidence_ids=("t1", "t2"),
                    exact_entity_evidence_ids=("t1",),
                    entity=("user", "raj_sanchez_7340"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"name": "Raj Sanchez"},
                support=SupportLevel.HIGH,
                evidence_ids=("t1",),
            ),
            trace,
        )

    context = _context()
    proposal, trace = step_agentic_tool_world(fake_predict, context=context)
    outcome = validate_transition(
        proposal,
        calls=context.candidate_calls,
        state=context.current_state,
        step_index=0,
        allowed_evidence_ids=trace.all_evidence_ids,
    )
    assert outcome.accepted is True


def test_validator_rejects_evidence_the_awm_never_actually_searched_for() -> None:
    """Citing an id no search_transitions call in this episode ever returned
    must be rejected by the same validator every other path uses -- the
    agentic AWM cannot cite evidence it never had access to."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="search_transitions",
                    evidence_ids=("t1", "t2"),
                    exact_entity_evidence_ids=("t1",),
                    entity=("user", "raj_sanchez_7340"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"name": "Raj Sanchez"},
                support=SupportLevel.HIGH,
                evidence_ids=("t999-never-searched",),
            ),
            trace,
        )

    context = _context()
    proposal, trace = step_agentic_tool_world(fake_predict, context=context)
    outcome = validate_transition(
        proposal,
        calls=context.candidate_calls,
        state=context.current_state,
        step_index=0,
        allowed_evidence_ids=trace.all_evidence_ids,
    )
    assert outcome.accepted is False


# --- claim-level attribution: the exact adversarial attack the entity-level
# fix missed -- read one real field, invent several more, and check that
# only the invented ones are caught. --------------------------------------


def _reservation_context(**overrides):
    base = dict(
        candidate_calls=(
            ActionCall(call_id="a", tool="cancel_reservation", arguments={"reservation_id": "Q69X3R"}),
        ),
        current_state=ScenarioState(),
        history_text="",
        offered_tool_schemas=({"name": "cancel_reservation", "parameters": {}},),
        fit_index=(),
        family_id="f",
        task_context="",
        excluded_trace_ids=(),
        prior_grounding_calls=(),
    )
    base.update(overrides)
    return AWMRuntimeContext(**base)


def test_reading_one_field_does_not_license_inventing_others() -> None:
    """The exact attack: AWM reads reservation.Q69X3R.status=confirmed, then
    claims a fabricated refund amount, payment mutation, and flight change.
    The old entity-level check ("was Q69X3R grounded at all") would have
    scored this HIGH in full. Claim-level attribution must reject the
    proposal outright, because the fabricated claims are state-delta/event
    claims -- exactly the kind that get committed to the ledger."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    arguments={"entity_kind": "reservation", "entity_id": "Q69X3R"},
                    result_summary="found=True, 1 field(s)",
                    state_paths_read=("get_reservation_details.Q69X3R.status",),
                    state_values_read={"status": "confirmed"},
                    entity=("reservation", "Q69X3R"),
                ),
                AWMToolCall(
                    index=1,
                    tool="search_transitions",
                    evidence_ids=("t1",),
                    exact_entity_evidence_ids=("t1",),
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"status": "cancelled"},
                # Fabricated: nothing about a refund, a payment, or a flight
                # change was ever read or matched by search for this entity.
                state_delta=(
                    StateDelta(path="get_reservation_details.Q69X3R.status", old_value="confirmed", new_value="cancelled"),
                    StateDelta(path="get_reservation_details.Q69X3R.refund_amount", old_value=None, new_value=430.0),
                ),
                events=({"type": "RefundIssued", "amount": 430.0},),
                support=SupportLevel.HIGH,
                evidence_ids=("t1",),
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    assert proposal.abstain is True
    assert "refund_amount" in proposal.abstain_reason
    # No mutations/events survive -- the whole proposal is rejected, not
    # partially committed with only the real field kept.
    assert proposal.state_delta == ()
    assert proposal.events == ()


def test_legitimate_cancellation_mutation_survives_with_behavioral_evidence() -> None:
    """The correction's key point: a mutation to an already-known field
    (status: confirmed -> cancelled) is DERIVABLE_FROM_STATE, not UNSUPPORTED
    -- simple string-diffing against state_paths_read would have wrongly
    rejected this, since "cancelled" never appeared verbatim in pre-state.
    Legitimate because the path was read (pre-mutation value known) AND the
    tool has behavioral evidence (search_transitions found something for
    cancel_reservation)."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q69X3R.status",),
                    # Full canonical "<tool>.<entity_id>.<field>" path -- the
                    # same convention StateDelta.path/observation fields use
                    # (world.py's own instructed contract), keyed under the
                    # READ tool since that is what actually produced the
                    # pre-mutation value.
                    state_values_read={"get_reservation_details.Q69X3R.status": "confirmed"},
                    entity=("reservation", "Q69X3R"),
                ),
                AWMToolCall(
                    index=1,
                    tool="search_transitions",
                    arguments={"tool": "cancel_reservation"},
                    evidence_ids=("t1",),
                    exact_entity_evidence_ids=("t1",),
                    # Tool-shape evidence: cancel_reservation has actually
                    # been observed to produce status="cancelled" at this
                    # bare sub-path, for some entity -- this is what licenses
                    # the specific new value, not merely "the tool has some
                    # evidence."
                    observed_tool_values={"status": ("cancelled",)},
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"status": "cancelled"},
                # The candidate call is cancel_reservation, so the mutation's
                # own canonical path is under cancel_reservation -- but the
                # KNOWN pre-mutation value was read via a different tool
                # (get_reservation_details). This is realistic: the fact is
                # about the entity, tool-namespaced by whichever tool read it.
                state_delta=(
                    StateDelta(path="get_reservation_details.Q69X3R.status", old_value="confirmed", new_value="cancelled"),
                ),
                support=SupportLevel.HIGH,
                evidence_ids=("t1",),
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    assert proposal.abstain is False
    assert len(proposal.state_delta) == 1
    assert proposal.state_delta[0].new_value == "cancelled"
    assert proposal.support is SupportLevel.MEDIUM  # DERIVABLE_FROM_STATE, not EXACT


def test_mutation_to_a_known_field_without_tool_behavioral_evidence_is_rejected() -> None:
    """Same known-field mutation as above, but with zero search_transitions
    evidence for this tool -- there is no evidence this tool's write
    semantics are even real, so the mutation claim is UNSUPPORTED despite
    the path being known, and the whole proposal is rejected."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q69X3R.status",),
                    state_values_read={"status": "confirmed"},
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"status": "cancelled"},
                state_delta=(
                    StateDelta(path="get_reservation_details.Q69X3R.status", old_value="confirmed", new_value="cancelled"),
                ),
                support=SupportLevel.HIGH,
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    assert proposal.abstain is True


def test_invented_observation_field_is_nulled_not_rejected_wholesale() -> None:
    """A fabricated field on an ordinary (read-only) observation -- no
    state_delta, no events -- is nulled rather than causing a full
    rejection: it is not about to be committed to the ledger the way a
    mutation is, so a partial/unknown answer is the honest response, not an
    abstention."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q69X3R.status",),
                    state_values_read={"status": "confirmed"},
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                # customer_notes was never read or searched for -- invented.
                observation={"status": "confirmed", "customer_notes": "prefers aisle seat"},
                support=SupportLevel.HIGH,
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    assert proposal.abstain is False
    assert proposal.observation["status"] == "confirmed"
    assert proposal.observation["customer_notes"] is None


def test_batch_claims_stay_separate_under_claim_level_attribution() -> None:
    """Call A's real grounding must never license call B's fabricated
    mutation, and vice versa -- extends the earlier entity-level batch
    isolation test to the claim level."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q1.status",),
                    state_values_read={"status": "confirmed"},
                    entity=("reservation", "Q1"),
                ),
                AWMToolCall(
                    index=1,
                    tool="search_transitions",
                    evidence_ids=("t1",),
                    exact_entity_evidence_ids=("t1",),
                    entity=("reservation", "Q1"),
                ),
            )
        )
        return (
            ProposedTransition(
                call_outcomes=(
                    ProposedCallOutcome(
                        call_id="a",
                        observation={"status": "cancelled"},
                        state_delta=(
                            StateDelta(path="get_reservation_details.Q1.status", old_value="confirmed", new_value="cancelled"),
                        ),
                        evidence_ids=("t1",),
                    ),
                    ProposedCallOutcome(
                        call_id="b",
                        # Q2 was never read or searched for at all -- fabricated batch entry.
                        observation={"status": "cancelled"},
                        state_delta=(
                            StateDelta(path="get_reservation_details.Q2.status", old_value="confirmed", new_value="cancelled"),
                        ),
                    ),
                ),
                support=SupportLevel.HIGH,
                evidence_ids=("t1",),
            ),
            trace,
        )

    context = _reservation_context(
        candidate_calls=(
            ActionCall(call_id="a", tool="cancel_reservation", arguments={"reservation_id": "Q1"}),
            ActionCall(call_id="b", tool="cancel_reservation", arguments={"reservation_id": "Q2"}),
        ),
    )
    proposal, _ = step_agentic_tool_world(fake_predict, context=context)
    # Call B's fabricated mutation must reject the whole proposal -- there is
    # no partial-commit path for a batch with any unsupported mutation claim.
    assert proposal.abstain is True
    assert "Q2" in proposal.abstain_reason


# --- P0: list-nested nulling and call-scoped nulling -----------------------


def test_unsupported_field_inside_a_list_is_nulled() -> None:
    """P0. ``_flatten_paths`` emits list-nested paths ("flights[0].destination"),
    so a fabricated field inside a list is correctly classified UNSUPPORTED and
    dropped from the surviving-claims support computation -- but a dict-only
    nulling walk left its value sitting in the response. Worst case: the
    fabrication survives AND removing its claim raises the proposal's support.
    """

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q69X3R.flights[0].origin",),
                    state_values_read={"flights[0].origin": "PHL"},
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={
                    "flights": [{"origin": "PHL", "destination": "Mars"}],
                },
                support=SupportLevel.HIGH,
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    flight = proposal.observation["flights"][0]
    # The read field survives; the fabricated sibling is an explicit unknown.
    assert flight["origin"] == "PHL"
    assert flight["destination"] is None
    # List shape is preserved -- length and element positions are untouched.
    assert len(proposal.observation["flights"]) == 1


def test_scalar_directly_inside_a_list_is_nulled_in_place() -> None:
    """A fabricated scalar list element is nulled in place rather than
    dropped, so the list's ".length" claim stays true and sibling indices
    keep their meaning."""

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.Q69X3R.seats[0]",),
                    state_values_read={"seats[0]": "12A"},
                    entity=("reservation", "Q69X3R"),
                ),
            )
        )
        return (
            ProposedTransition(
                observation={"seats": ["12A", "99Z"]},
                support=SupportLevel.HIGH,
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=_reservation_context())
    assert proposal.observation["seats"] == ["12A", None]


def test_batch_nulling_is_scoped_to_the_offending_call() -> None:
    """P0. Nulling keyed by path alone let call A's unsupported "status" null
    call B's exactly-grounded "status". Nulling must be keyed by
    (call_id, path)."""
    from bandits.diagnose.world import ProposedCallOutcome

    context = _reservation_context(
        candidate_calls=(
            ActionCall(call_id="a", tool="get_reservation_details", arguments={"reservation_id": "AAA111"}),
            ActionCall(call_id="b", tool="get_reservation_details", arguments={"reservation_id": "BBB222"}),
        ),
    )

    def fake_predict(*, instruction, context):
        trace = AWMExecutionTrace(
            grounding_calls=(
                # Only call B's entity was ever read.
                AWMToolCall(
                    index=0,
                    tool="read_world_state",
                    state_paths_read=("get_reservation_details.BBB222.status",),
                    state_values_read={"status": "confirmed"},
                    entity=("reservation", "BBB222"),
                ),
            )
        )
        return (
            ProposedTransition(
                call_outcomes=(
                    # Call A: nothing grounded this entity at all -> fabricated.
                    ProposedCallOutcome(call_id="a", observation={"status": "cancelled"}),
                    # Call B: exactly what was read.
                    ProposedCallOutcome(call_id="b", observation={"status": "confirmed"}),
                ),
                support=SupportLevel.HIGH,
            ),
            trace,
        )

    proposal, _ = step_agentic_tool_world(fake_predict, context=context)
    by_id = {o.call_id: o for o in proposal.call_outcomes}
    assert by_id["a"].observation["status"] is None
    # The grounded call keeps its value -- it shares a path name, not a claim.
    assert by_id["b"].observation["status"] == "confirmed"


def test_react_can_build_its_fallback_signature() -> None:
    """dspy.ReAct rebuilds a fallback signature from the original signature's
    fields. agentic.py uses `from __future__ import annotations`, so those
    fields carry annotation *strings*, and make_signature rejects a ForwardRef
    outright ("Field types must be types"). dspy.Predict tolerates it, which
    is why nothing else in the suite caught this -- the smoke runner failed on
    all three transitions before reaching a single model call.

    Drives the real builder and the real predict() path, stopping at the
    network boundary: anything raised before that (ValueError from
    make_signature) is the regression; a provider/auth error means signature
    construction succeeded.
    """
    predict = build_agentic_tool_world_predictor(model="test-model", api_key="dummy-key")
    try:
        predict(instruction="probe", context=_reservation_context())
    except ValueError as exc:  # pragma: no cover -- the regression itself
        if "Field types must be types" in str(exc):
            raise AssertionError(f"ReAct could not build its fallback signature: {exc}") from exc
    except Exception:  # noqa: BLE001 -- reaching the provider is the success case
        pass
