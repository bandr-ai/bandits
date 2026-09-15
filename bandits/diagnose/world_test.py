"""The world model, exercised through an injected predictor. No credential."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.diagnose.models import (
    ActionCall,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    ScenarioState,
    StateField,
    SupportLevel,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
    WorldOrigin,
)
from bandits.diagnose.retrieve import RetrievedExample
from bandits.diagnose.world import (
    ProposedCallOutcome,
    ProposedTransition,
    ProposedUserTurn,
    StateDelta,
    commit,
    render_evidence,
    render_profile,
    render_state,
    step_tool_world,
    step_user_policy,
    validate_transition,
)


def _example(tid="t1", tool="cancel_reservation", reasons=("exact_tool",)):
    return RetrievedExample(
        transition=GroundingTransition(
            transition_id=tid,
            trace_id="airline-10",
            family_id="f",
            turn_index=0,
            action_span_id="s1",
            action_calls=(ActionCall(tool=tool, arguments={"reservation_id": "ABC"}),),
            observations=(GroundingObservation(role="tool", content={"status": "cancelled"}),),
        ),
        score=1.0,
        reasons=reasons,
    )


def _supported(n=2):
    return tuple(_example(f"t{i}") for i in range(n))


def _state(*pairs, origin=WorldOrigin.RECORDED):
    return ScenarioState(
        fields=tuple(
            StateField(
                path=p,
                value=v,
                origin=origin,
                **(
                    {"revealed_by_span_id": "s"}
                    if origin is WorldOrigin.RECORDED
                    else {"revealed_at_step": 0}
                ),
            )
            for p, v in pairs
        )
    )


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


def _predictor(payload):
    def predict(**_):
        return payload

    return predict


# --- abstention ---------------------------------------------------------


def test_no_evidence_abstains_without_calling_the_model() -> None:
    """The whole point of D27: an unsupported counterfactual is declined."""
    calls = []

    def predict(**_):
        calls.append(1)
        return ProposedTransition(observation={"status": "cancelled"})

    result = step_tool_world(
        predict,
        calls=(ActionCall(tool="cancel_reservation"),),
        content=None,
        state=ScenarioState(),
        history="",
        examples=(),
    )
    assert result.abstain
    assert result.support is SupportLevel.NONE
    assert not calls  # no spend on a question with no evidence behind it


def test_a_malformed_answer_abstains_rather_than_being_salvaged() -> None:
    result = step_tool_world(
        _predictor("not json at all"),
        calls=(ActionCall(tool="cancel_reservation"),),
        content=None,
        state=ScenarioState(),
        history="",
        examples=_supported(),
    )
    assert result.abstain
    assert "contract" in result.abstain_reason


def test_an_abstaining_proposal_cannot_also_propose_changes() -> None:
    with pytest.raises(ValidationError, match="cannot also propose changes"):
        ProposedTransition(abstain=True, state_delta=(StateDelta(path="a.b", new_value=1),))


def test_abstention_is_rejected_by_the_validator_not_committed() -> None:
    outcome = validate_transition(
        ProposedTransition(abstain=True),
        calls=(ActionCall(tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert outcome.committed == ()


# --- support is taken from retrieval, not from the model ----------------


def test_model_claimed_support_is_capped_by_what_retrieval_found() -> None:
    """A model rating its own grounding rates it generously."""
    result = step_tool_world(
        _predictor(ProposedTransition(observation={"ok": 1}, support=SupportLevel.HIGH)),
        calls=(ActionCall(tool="cancel_reservation"),),
        content=None,
        state=ScenarioState(),
        history="",
        examples=(_example("t1"),),  # one exact match == medium
    )
    assert result.support is SupportLevel.MEDIUM


def test_a_lower_self_reported_support_is_kept() -> None:
    result = step_tool_world(
        _predictor(ProposedTransition(observation={"ok": 1}, support=SupportLevel.LOW)),
        calls=(ActionCall(tool="cancel_reservation"),),
        content=None,
        state=ScenarioState(),
        history="",
        examples=_supported(),  # would allow high
    )
    assert result.support is SupportLevel.LOW


def test_support_below_the_floor_is_refused_by_the_validator() -> None:
    outcome = validate_transition(
        ProposedTransition(
            observation={"ok": 1},
            support=SupportLevel.NONE,
            state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
        min_support=SupportLevel.LOW,
    )
    assert not outcome.accepted
    assert any("support" in reason for reason in outcome.rejections)


# --- the validator owns the ledger --------------------------------------


def test_a_change_with_no_evidence_behind_it_is_refused() -> None:
    """However confident the model sounded, this is an invention."""
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=(),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert any("no supporting evidence" in r for r in outcome.rejections)


def test_a_change_cannot_cite_evidence_that_was_not_retrieved() -> None:
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("invented",),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
        catalog=_catalog(),
        allowed_evidence_ids=("t1", "t2"),
    )
    assert not outcome.accepted
    assert any("not retrieved" in reason for reason in outcome.rejections)


def test_a_write_may_only_touch_reviewed_paths() -> None:
    catalog = ToolEffectCatalog(
        catalog_id="c",
        toolset_digest="d",
        entries=(
            ToolEffectEntry(
                tool="cancel_reservation",
                effect=ToolEffect.WRITE,
                reviewed_by="reviewer",
                mutates_paths=("reservations.*.status",),
            ),
        ),
    )
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(StateDelta(path="payments.ABC.amount", new_value=0),),
            support=SupportLevel.HIGH,
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
        catalog=catalog,
        allowed_evidence_ids=("t1",),
    )
    assert not outcome.accepted
    assert any("reviewed mutation path" in reason for reason in outcome.rejections)


def test_a_call_about_one_entity_cannot_mutate_another() -> None:
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(StateDelta(path="cancel_reservation.XYZ.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert any("names no entity" in r for r in outcome.rejections)


def test_a_stale_old_value_means_the_model_reasoned_from_another_world() -> None:
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(
                StateDelta(
                    path="cancel_reservation.ABC.status",
                    old_value="confirmed",
                    new_value="cancelled",
                ),
            ),
            support=SupportLevel.HIGH,
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=_state(("cancel_reservation.ABC.status", "cancelled")),
        step_index=0,
    )
    assert not outcome.accepted
    assert any("not 'confirmed'" in r for r in outcome.rejections)


def test_a_read_only_call_cannot_change_state() -> None:
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(
                StateDelta(path="get_reservation_details.ABC.status", new_value="cancelled"),
            ),
            support=SupportLevel.HIGH,
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(tool="get_reservation_details", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=0,
        catalog=_catalog(),
    )
    assert not outcome.accepted
    assert any("read-only" in r for r in outcome.rejections)


def test_an_accepted_change_is_committed_as_simulated() -> None:
    outcome = validate_transition(
        ProposedTransition(
            state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("t1", "t2"),
        ),
        calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),),
        state=ScenarioState(),
        step_index=3,
        catalog=_catalog(),
    )
    assert outcome.accepted
    field = outcome.committed[0]
    assert field.origin is WorldOrigin.SIMULATED
    assert field.revealed_at_step == 3
    assert field.evidence_ids == ("t1", "t2")


def test_batched_calls_have_independent_execution_and_commit_outcomes() -> None:
    calls = (
        ActionCall(call_id="a", tool="cancel_reservation", arguments={"reservation_id": "ABC"}),
        ActionCall(call_id="b", tool="cancel_reservation", arguments={"reservation_id": "XYZ"}),
    )
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(
                    call_id="a",
                    observation={"status": "cancelled"},
                    state_delta=(
                        StateDelta(
                            path="cancel_reservation.ABC.status", new_value="cancelled"
                        ),
                    ),
                    evidence_ids=("t1",),
                ),
                ProposedCallOutcome(
                    call_id="b",
                    observation={"error": "already cancelled"},
                    error=True,
                    evidence_ids=("t2",),
                ),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=calls,
        state=ScenarioState(),
        step_index=0,
        catalog=_catalog(),
        allowed_evidence_ids=("t1", "t2"),
    )
    assert outcome.accepted
    assert outcome.executed_call_ids == ("a", "b")
    assert outcome.committed_call_ids == ("a",)


def test_declared_tool_output_schema_is_enforced() -> None:
    outcome = validate_transition(
        ProposedTransition(
            observation={"status": 42},
            support=SupportLevel.HIGH,
            evidence_ids=("t1",),
        ),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        allowed_evidence_ids=("t1",),
        tool_schemas=(
            {
                "name": "cancel_reservation",
                "output_schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}},
                },
            },
        ),
    )
    assert not outcome.accepted
    assert outcome.output_schema_validation == {"a": "invalid"}
    assert any("output.status" in reason for reason in outcome.rejections)


def test_batched_events_are_attributed_to_their_call() -> None:
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(
                    call_id="a",
                    observation={"ok": True},
                    events=({"type": "reservation_cancelled"},),
                    evidence_ids=("t1",),
                ),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        allowed_evidence_ids=("t1",),
    )
    assert outcome.events == ({"type": "reservation_cancelled", "_call_id": "a"},)


def test_spoofed_event_call_id_is_rejected() -> None:
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(
                    call_id="a",
                    events=({"type": "reservation_cancelled", "_call_id": "b"},),
                    evidence_ids=("t1",),
                ),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        allowed_evidence_ids=("t1",),
    )
    assert not outcome.accepted
    assert any("spoofed" in reason for reason in outcome.rejections)


def test_missing_output_schema_is_reported_as_unavailable() -> None:
    outcome = validate_transition(
        ProposedTransition(observation={"status": "ok"}, support=SupportLevel.HIGH),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        tool_schemas=({"name": "cancel_reservation", "parameters": {"type": "object"}},),
    )
    assert outcome.accepted
    assert outcome.output_schema_validation == {"a": "unavailable"}


def test_openai_wrapped_output_schema_is_validated() -> None:
    outcome = validate_transition(
        ProposedTransition(observation={"status": "ok"}, support=SupportLevel.HIGH),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        tool_schemas=(
            {
                "type": "function",
                "function": {
                    "name": "cancel_reservation",
                    "output_schema": {
                        "type": "object",
                        "required": ["status"],
                        "properties": {"status": {"type": "string"}},
                    },
                },
            },
        ),
    )
    assert outcome.accepted
    assert outcome.output_schema_validation == {"a": "validated"}


def test_an_invalid_declared_output_schema_rejects_the_transition() -> None:
    outcome = validate_transition(
        ProposedTransition(observation={"status": "ok"}, support=SupportLevel.HIGH),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
        tool_schemas=(
            {
                "name": "cancel_reservation",
                "output_schema": {"type": "definitely-not-a-json-schema-type"},
            },
        ),
    )
    assert not outcome.accepted
    assert any("declared output schema is invalid" in reason for reason in outcome.rejections)


def test_a_false_boolean_output_schema_rejects_every_output() -> None:
    outcome = validate_transition(
        ProposedTransition(observation=None, support=SupportLevel.HIGH),
        calls=(ActionCall(call_id="a", tool="never_returns"),),
        state=ScenarioState(),
        step_index=0,
        tool_schemas=({"name": "never_returns", "output_schema": False},),
    )
    assert not outcome.accepted
    assert any("False schema" in reason for reason in outcome.rejections)


def test_an_event_without_retrieved_evidence_is_rejected() -> None:
    outcome = validate_transition(
        ProposedTransition(
            events=({"type": "reservation_cancelled"},),
            support=SupportLevel.HIGH,
        ),
        calls=(ActionCall(call_id="a", tool="cancel_reservation"),),
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert any("no supporting evidence" in reason for reason in outcome.rejections)


def test_one_call_cannot_borrow_another_calls_evidence() -> None:
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(
                    call_id="a",
                    observation={"status": "cancelled"},
                    events=({"type": "reservation_cancelled"},),
                ),
                ProposedCallOutcome(
                    call_id="b",
                    observation={"status": "cancelled"},
                    evidence_ids=("t1",),
                ),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=(
            ActionCall(call_id="a", tool="cancel_reservation"),
            ActionCall(call_id="b", tool="cancel_reservation"),
        ),
        state=ScenarioState(),
        step_index=0,
        allowed_evidence_ids=("t1",),
    )
    assert not outcome.accepted
    assert any(
        "call a" in reason and "no supporting evidence" in reason
        for reason in outcome.rejections
    )


def test_commit_preserves_the_origin_of_untouched_fields() -> None:
    before = _state(("a.ABC.x", 1), ("b.ABC.y", 2))
    after = commit(
        before,
        (StateField(path="a.ABC.x", value=9, origin=WorldOrigin.SIMULATED, revealed_at_step=1),),
    )
    assert after.get("a.ABC.x").origin is WorldOrigin.SIMULATED
    assert after.get("b.ABC.y").origin is WorldOrigin.RECORDED
    assert after.simulated_paths == ("a.ABC.x",)


# --- the two roles stay separate ----------------------------------------


def test_the_user_policy_can_never_return_a_state_change() -> None:
    """Structural, not a rule the prompt has to remember."""
    assert not hasattr(ProposedUserTurn(), "state_delta")
    assert not hasattr(ProposedUserTurn(), "events")


def test_the_profile_reaches_the_user_policy_prompt() -> None:
    seen = {}

    def predict(**kwargs):
        seen.update(kwargs)
        return ProposedUserTurn(user_message="My id is anya_garcia_5901.")

    step_user_policy(
        predict,
        profile=HiddenUserProfile(
            known_info="user id is anya_garcia_5901", unknown_info="whether insurance applies"
        ),
        history="",
        message="Could I have your user id?",
    )
    assert "anya_garcia_5901" in seen["profile"]
    assert "whether insurance applies" in seen["profile"]


def test_disclosed_facts_are_returned_for_the_disclosure_gate() -> None:
    turn = step_user_policy(
        _predictor(
            ProposedUserTurn(user_message="It is 3RK2T9.", disclosed_facts=("reservation_id",))
        ),
        profile=HiddenUserProfile(known_info="reservation 3RK2T9"),
        history="",
        message="What is your reservation number?",
    )
    assert turn.disclosed_facts == ("reservation_id",)


def test_a_malformed_user_answer_abstains() -> None:
    turn = step_user_policy(
        _predictor(object()),
        profile=HiddenUserProfile(),
        history="",
        message="hello",
    )
    assert turn.abstain


# --- rendering ----------------------------------------------------------


def test_state_rendering_marks_simulated_facts() -> None:
    state = ScenarioState(
        fields=(
            StateField(
                path="a.ABC.x", value=1, origin=WorldOrigin.RECORDED, revealed_by_span_id="s"
            ),
            StateField(path="b.ABC.y", value=2, origin=WorldOrigin.SIMULATED, revealed_at_step=1),
        )
    )
    rendered = render_state(state)
    assert "[simulated]" in rendered
    assert rendered.count("[simulated]") == 1


def test_empty_state_says_so_rather_than_rendering_nothing() -> None:
    assert "nothing is known" in render_state(ScenarioState())


def test_evidence_rendering_is_where_clipping_belongs() -> None:
    """The index stays lossless; the prompt is what has a budget."""
    rendered = render_evidence(_supported(20), budget=300)
    assert "further examples omitted" in rendered


def test_no_evidence_renders_an_explicit_statement() -> None:
    assert "no relevant recorded transition" in render_evidence(())


def test_profile_rendering_omits_empty_sections() -> None:
    assert render_profile(HiddenUserProfile()) == ""
    assert "Persona" in render_profile(HiddenUserProfile(persona="impatient"))


def test_a_batch_missing_an_outcome_is_refused() -> None:
    """D70. Silence is not an encoding for "this call did not run".

    The validator checked outcomes against submitted calls but never the
    reverse, so an AWM could drop a call it did not want to answer and the
    step was accepted as though only the answered call had been submitted.
    The dropped call may be the policy-forbidden one.
    """
    calls = (
        ActionCall(call_id="a", tool="get_reservation_details", arguments={"reservation_id": "ABC"}),
        ActionCall(call_id="b", tool="cancel_reservation", arguments={"reservation_id": "XYZ"}),
    )
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(call_id="a", observation={"status": "confirmed"}),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=calls,
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert any("no proposed outcome" in reason for reason in outcome.rejections)
    assert "b" not in outcome.executed_call_ids


def test_an_unanswerable_call_is_stated_not_omitted() -> None:
    """D70's other half: the encoding that replaces silence already works."""
    calls = (
        ActionCall(call_id="a", tool="get_reservation_details", arguments={"reservation_id": "ABC"}),
        ActionCall(call_id="b", tool="cancel_reservation", arguments={"reservation_id": "XYZ"}),
    )
    outcome = validate_transition(
        ProposedTransition(
            call_outcomes=(
                ProposedCallOutcome(call_id="a", observation={"status": "confirmed"}),
                ProposedCallOutcome(call_id="b", executed=False, error=True, observation={}),
            ),
            support=SupportLevel.HIGH,
        ),
        calls=calls,
        state=ScenarioState(),
        step_index=0,
    )
    assert outcome.accepted
    assert outcome.executed_call_ids == ("a",)


def test_execution_is_never_inferred_from_the_submitted_calls() -> None:
    """D70. The sharper defect: with no outcomes at all, the fallback branch
    marked every submitted call executed from the call list alone. That is the
    validator manufacturing the claim D53 exists to make the AWM own.
    """
    calls = (
        ActionCall(call_id="a", tool="cancel_reservation", arguments={"reservation_id": "ABC"}),
        ActionCall(call_id="b", tool="cancel_reservation", arguments={"reservation_id": "XYZ"}),
    )
    outcome = validate_transition(
        ProposedTransition(support=SupportLevel.HIGH),
        calls=calls,
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert outcome.executed_call_ids == ()


def test_a_single_call_claiming_nothing_at_all_is_refused() -> None:
    """D67 keeps the aggregate single-call form, so a lone call may state its
    result as the proposal's own observation/delta. D70 draws the line at
    silence: a proposal claiming nothing cannot report the call as executed.
    """
    outcome = validate_transition(
        ProposedTransition(support=SupportLevel.HIGH),
        calls=(
            ActionCall(call_id="a", tool="cancel_reservation", arguments={"reservation_id": "ABC"}),
        ),
        state=ScenarioState(),
        step_index=0,
    )
    assert not outcome.accepted
    assert outcome.executed_call_ids == ()
