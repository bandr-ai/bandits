"""The rollout loop, driven by injected candidates and predictors."""

from __future__ import annotations

from bandits.emulate.models import (
    ActionCall,
    CommunicationRequirement,
    ExpectedEffect,
    ForbiddenEffect,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    Partition,
    ResultStatus,
    Scenario,
    ScenarioKind,
    SealedSuccessContract,
    SuccessShape,
    SupportLevel,
    TerminationReason,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
)
from bandits.emulate.rollout import Budget, CandidateAction, reset, run_rollout
from bandits.emulate.world import ProposedTransition, ProposedUserTurn, StateDelta


def _index(n=3, tool="cancel_reservation"):
    return tuple(
        GroundingTransition(
            transition_id=f"t{i}",
            trace_id=f"airline-{90 + i}",
            family_id="family-451ae91f975c",
            turn_index=0,
            task_context="cancel my reservation",
            success_shape=SuccessShape.MUTATION,
            action_span_id=f"s{i}",
            action_calls=(ActionCall(tool=tool, arguments={"reservation_id": "ABC"}),),
            observations=(GroundingObservation(role="tool", content={"status": "cancelled"}),),
        )
        for i in range(n)
    )


def _scenario(**updates) -> Scenario:
    base = dict(
        scenario_id="scenario-1",
        kind=ScenarioKind.TASK_START,
        task="cancel reservation ABC",
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="7",
            shape=SuccessShape.MUTATION,
            required_effects=(
                ExpectedEffect(
                    effect_id="e1",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC"},
                ),
            ),
        ),
        source_trace_id="airline-10",
        source_task_id="7",
        family_id="family-451ae91f975c",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("airline-10",),
    )
    return Scenario(**{**base, **updates})


def _catalog():
    return ToolEffectCatalog(
        catalog_id="c",
        toolset_digest="d",
        entries=(
            ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="a"),
        ),
    )


def _cancelling_candidate(then_done=True):
    calls = {"n": 0}

    def candidate(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            return CandidateAction(
                calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),)
            )
        return CandidateAction(done=then_done, content="All set.")

    return candidate


def _tool_world(proposal=None):
    def predict(**_):
        return proposal or ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("t0",),
        )

    return predict


def _user_policy(turn=None):
    def predict(**_):
        return turn or ProposedUserTurn(user_message="Thanks.")

    return predict


def _run(scenario=None, candidate=None, tool_world=None, user_policy=None, **kw):
    return run_rollout(
        scenario or _scenario(),
        candidate or _cancelling_candidate(),
        index=kw.pop("index", _index()),
        tool_world=tool_world or _tool_world(),
        user_policy=user_policy or _user_policy(),
        binding_id="binding-1",
        candidate_id="candidate-1",
        catalog=kw.pop("catalog", _catalog()),
        **kw,
    )


# --- reset --------------------------------------------------------------


def test_reset_hands_the_candidate_only_its_view() -> None:
    view, state, history = reset(_scenario())
    assert "success_contract" not in view.model_dump()
    assert "hidden_user" not in view.model_dump()
    assert state.fields == ()
    assert history == []


def test_branches_do_not_share_state() -> None:
    """Repeated sampling needs independent attempts, or pass@k is not pass@k."""
    scenario = _scenario()
    first = _run(scenario, seed=1)
    second = _run(scenario, seed=2)
    assert scenario.initial_state.fields == ()
    assert first.final_state.simulated_paths == second.final_state.simulated_paths


# --- the happy path -----------------------------------------------------


def test_a_committed_required_effect_passes() -> None:
    result = _run()
    assert result.terminated_by is TerminationReason.CANDIDATE_COMPLETED
    assert result.operational_result.status is ResultStatus.PASS
    assert result.overall is ResultStatus.PASS


def test_a_simulated_verdict_says_it_is_simulation_conditioned() -> None:
    assert _run().simulation_conditioned


def test_committed_state_is_marked_simulated() -> None:
    result = _run()
    assert "cancel_reservation.ABC.status" in result.final_state.simulated_paths


def test_rollout_preserves_validator_owned_schema_status_and_events() -> None:
    scenario = _scenario(
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {"type": "object"},
                "output_schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(
                ActionCall(
                    call_id="call-a",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC"},
                ),
            )
        )

    proposal = ProposedTransition(
        observation={"status": "cancelled"},
        events=({"type": "reservation_cancelled"},),
        state_delta=(StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),),
        support=SupportLevel.HIGH,
        evidence_ids=("t0",),
        terminal=True,
    )
    result = _run(scenario, candidate=candidate, tool_world=_tool_world(proposal))
    assert result.steps[0].output_schema_validation == {"call-a": "validated"}
    assert result.steps[0].events == (
        {"type": "reservation_cancelled", "_call_id": "call-a"},
    )


def test_rollout_preserves_invalid_output_status_on_a_rejected_step() -> None:
    scenario = _scenario(
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {"type": "object", "properties": {}},
                "output_schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}},
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(
                ActionCall(
                    call_id="call-a",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC"},
                ),
            )
        )

    proposal = ProposedTransition(
        observation={"status": 42},
        support=SupportLevel.HIGH,
        terminal=True,
    )
    result = _run(scenario, candidate=candidate, tool_world=_tool_world(proposal))
    assert result.steps[0].output_schema_validation == {"call-a": "invalid"}
    assert result.terminated_by is TerminationReason.INVALID_TRANSITION


def test_versions_are_pinned_on_the_result() -> None:
    result = _run(
        versions={
            "tool_awm": "awm-v1",
            "user_policy": "user-v1",
            "retrieval_index": "idx-v1",
            "scenario_set": "set-v1",
        }
    )
    assert result.tool_awm_version == "awm-v1"
    assert result.user_policy_version == "user-v1"


# --- abstention leaves the denominator ----------------------------------


def test_an_unsupported_action_abstains_and_claims_nothing() -> None:
    """Charging the candidate for the simulator's ignorance would be a lie."""
    result = _run(index=())
    assert result.terminated_by is TerminationReason.AWM_ABSTAINED
    assert result.overall is ResultStatus.UNKNOWN
    assert result.steps[0].abstained


def test_calling_an_unoffered_tool_is_a_candidate_failure_not_awm_abstention() -> None:
    calls = []

    def candidate(**_):
        return CandidateAction(calls=(ActionCall(tool="invented_tool"),))

    def world(**_):
        calls.append(1)
        return ProposedTransition()

    result = _run(candidate=candidate, tool_world=world)
    assert result.terminated_by is TerminationReason.UNAVAILABLE_TOOL
    assert result.overall is ResultStatus.FAIL
    assert calls == []


def test_malformed_tool_arguments_are_candidate_failure_before_the_awm() -> None:
    scenario = _scenario(
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {
                    "type": "object",
                    "required": ["reservation_id"],
                    "properties": {"reservation_id": {"type": "string"}},
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": 42}),)
        )

    result = _run(scenario, candidate=candidate)
    assert result.terminated_by is TerminationReason.INVALID_CANDIDATE_ACTION
    assert result.overall is ResultStatus.FAIL


def test_a_rejected_transition_ends_the_rollout_as_unknown() -> None:
    """Continuing from uncommitted state would reason from a world that never was."""
    bad = ProposedTransition(
        observation={"status": "cancelled"},
        state_delta=(StateDelta(path="cancel_reservation.OTHER.status", new_value="cancelled"),),
        support=SupportLevel.HIGH,
        evidence_ids=("t0",),
    )
    result = _run(tool_world=_tool_world(bad))
    assert result.terminated_by is TerminationReason.INVALID_TRANSITION
    assert result.overall is ResultStatus.UNKNOWN
    assert result.steps[0].validator_rejections


def test_an_unbindable_contract_is_unknown_not_excluded() -> None:
    scenario = _scenario(
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="9",
            shape=SuccessShape.MUTATION,
            required_effects=(),
            unbindable_reason="source declares no checkable outcome",
        )
    )
    assert _run(scenario).overall is ResultStatus.UNKNOWN


# --- budget exhaustion is a real failure --------------------------------


def test_a_loop_is_a_candidate_failure_not_an_abstention() -> None:
    def stuck(**_):
        return CandidateAction(
            calls=(ActionCall(tool="cancel_reservation", arguments={"reservation_id": "ABC"}),)
        )

    result = _run(candidate=stuck, budget=Budget(max_steps=20, max_repeats=3))
    assert result.terminated_by is TerminationReason.ACTION_LOOP
    assert not result.terminated_by.invalidates_rollout
    assert result.overall is ResultStatus.FAIL


def test_the_step_limit_stops_a_candidate_that_never_finishes() -> None:
    calls = {"n": 0}

    def wandering(**_):
        calls["n"] += 1
        return CandidateAction(
            calls=(
                ActionCall(
                    tool="cancel_reservation", arguments={"reservation_id": f"ABC{calls['n']}"}
                ),
            )
        )

    def observing_world(**_):
        # Observes without committing, so nothing the validator can reject —
        # the candidate simply never finishes.
        return ProposedTransition(
            observation={"status": "confirmed"}, support=SupportLevel.HIGH, evidence_ids=("t0",)
        )

    result = _run(candidate=wandering, tool_world=observing_world, budget=Budget(max_steps=4))
    assert result.terminated_by is TerminationReason.STEP_LIMIT
    assert len(result.steps) == 4
    assert result.overall is ResultStatus.FAIL


def test_a_budget_failure_after_simulated_steps_still_discloses_simulation() -> None:
    """The failure population is where the simulator is most likely at fault.

    A step-limit loss reached through committed simulated transitions rests on
    the AWM as much as any pass does. Reporting it as unconditioned hid the
    disclosure on exactly the rollouts that most needed it.
    """
    calls = {"n": 0}

    def wandering(**_):
        calls["n"] += 1
        return CandidateAction(
            calls=(
                ActionCall(
                    tool="cancel_reservation", arguments={"reservation_id": "ABC"}
                ),
            ),
            content=f"attempt {calls['n']}",
        )

    def committing_world(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(
                StateDelta(path="cancel_reservation.ABC.status", new_value="cancelled"),
            ),
            support=SupportLevel.HIGH,
            evidence_ids=("t0",),
        )

    result = _run(
        candidate=wandering,
        tool_world=committing_world,
        budget=Budget(max_steps=3, max_repeats=99),
    )
    assert result.terminated_by is TerminationReason.STEP_LIMIT
    assert result.overall is ResultStatus.FAIL
    assert result.simulation_conditioned


# --- the user policy ----------------------------------------------------


def test_speech_is_answered_by_the_user_policy_not_the_tool_world() -> None:
    def talker(**_):
        return CandidateAction(content="Could I have your reservation number?")

    seen = {"tool": 0, "user": 0}

    def tool_world(**_):
        seen["tool"] += 1
        return ProposedTransition(observation={})

    def user_policy(**_):
        seen["user"] += 1
        return ProposedUserTurn(user_message="It is ABC.", terminal=True)

    result = _run(candidate=talker, tool_world=tool_world, user_policy=user_policy)
    assert seen == {"tool": 0, "user": 1}
    assert result.steps[0].responder == "user_policy"


def test_the_authentic_future_is_never_replayed() -> None:
    """The recorded reply answered a different action; it is not a reply to this one."""
    scenario = _scenario(
        hidden_user=HiddenUserProfile(known_info="reservation ABC"),
    )

    def talker(**_):
        return CandidateAction(content="What can I help with?")

    generated = []

    def user_policy(**kwargs):
        generated.append(kwargs["message"])
        return ProposedUserTurn(user_message="I need to cancel.", terminal=True)

    _run(scenario, candidate=talker, user_policy=user_policy)
    assert generated == ["What can I help with?"]


def test_disclosed_facts_reach_the_step_for_the_gate() -> None:
    def talker(**_):
        return CandidateAction(content="Your reservation number?")

    result = _run(
        candidate=talker,
        user_policy=_user_policy(
            ProposedUserTurn(
                user_message="ABC.", disclosed_facts=("reservation_id",), terminal=True
            )
        ),
    )
    assert result.steps[0].disclosed_facts == ("reservation_id",)


def test_a_user_abstention_ends_the_rollout_unknown() -> None:
    def talker(**_):
        return CandidateAction(content="hello")

    result = _run(
        candidate=talker,
        user_policy=_user_policy(ProposedUserTurn(abstain=True, abstain_reason="out of character")),
    )
    assert result.terminated_by is TerminationReason.AWM_ABSTAINED
    assert result.overall is ResultStatus.UNKNOWN


# --- refusal shapes -----------------------------------------------------


def test_a_refusal_passes_when_the_candidate_declines() -> None:
    scenario = _scenario(
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="0",
            shape=SuccessShape.REFUSAL,
            forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
            communication=(
                CommunicationRequirement(
                    requirement_id="r1",
                    assertion="agent refuses",
                    kind="communicate_info",
                    expected_values=("not eligible",),
                ),
            ),
        )
    )

    def refuser(**_):
        return CandidateAction(done=True, content="That booking is not eligible for a refund.")

    result = _run(scenario, candidate=refuser)
    assert result.operational_result.status is ResultStatus.PASS
    assert result.communication_result.status is ResultStatus.PASS
    assert result.overall is ResultStatus.PASS


def test_a_refusal_fails_when_the_candidate_cancels_anyway() -> None:
    scenario = _scenario(
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="0",
            shape=SuccessShape.REFUSAL,
            forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
        )
    )
    result = _run(scenario)
    assert result.operational_result.status is ResultStatus.FAIL
    assert result.overall is ResultStatus.FAIL


def test_the_environment_executes_a_call_the_agent_should_not_have_made() -> None:
    """D27: refusing on the agent's behalf would hide the mistake from the verifier."""
    scenario = _scenario(
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="0",
            shape=SuccessShape.REFUSAL,
            forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
        )
    )
    result = _run(scenario)
    # The tool executed; the verifier, not the simulator, called it a failure.
    assert result.steps[0].observation == {"status": "cancelled"}
    assert not result.steps[0].abstained
    assert result.overall is ResultStatus.FAIL


def test_the_closing_message_is_not_lost_when_the_candidate_finishes() -> None:
    """A refusal is stated in the same breath as finishing.

    Breaking on ``done`` before recording the step dropped the content, and
    every communication requirement became unmeetable by a candidate that had
    actually met it.
    """
    scenario = _scenario(
        success_contract=SealedSuccessContract(
            contract_id="c",
            source_task_id="0",
            shape=SuccessShape.REFUSAL,
            forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
            communication=(
                CommunicationRequirement(
                    requirement_id="r1",
                    assertion="agent refuses",
                    kind="communicate_info",
                    expected_values=("not eligible",),
                ),
            ),
        )
    )

    def refuser(**_):
        return CandidateAction(done=True, content="That booking is not eligible.")

    result = _run(scenario, candidate=refuser)
    assert any(step.action_content for step in result.steps)
    assert result.communication_result.status is ResultStatus.PASS


def test_an_argument_outside_a_declared_enum_is_candidate_failure() -> None:
    """D72. Input validation was a hand-rolled required/primitive-type check, so
    a well-typed but contract-violating argument reached the AWM and asked it to
    invent behavior for a call the tool would have rejected.
    """
    scenario = _scenario(
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {
                    "type": "object",
                    "required": ["reservation_id", "cabin"],
                    "properties": {
                        "reservation_id": {"type": "string"},
                        "cabin": {"type": "string", "enum": ["economy", "business"]},
                    },
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(
                ActionCall(
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC", "cabin": "first"},
                ),
            )
        )

    result = _run(scenario, candidate=candidate)
    assert result.terminated_by is TerminationReason.INVALID_CANDIDATE_ACTION
    assert result.overall is ResultStatus.FAIL


def test_a_nested_argument_constraint_is_enforced() -> None:
    """The old checker stopped at the top level, so nested objects and array
    items were unvalidated however the contract declared them.
    """
    scenario = _scenario(
        offered_tools=(
            {
                "name": "book_reservation",
                "parameters": {
                    "type": "object",
                    "required": ["passengers"],
                    "properties": {
                        "passengers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            },
                        }
                    },
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(
                ActionCall(
                    tool="book_reservation",
                    arguments={"passengers": [{"name": 7}]},
                ),
            )
        )

    result = _run(scenario, candidate=candidate)
    assert result.terminated_by is TerminationReason.INVALID_CANDIDATE_ACTION
    assert result.overall is ResultStatus.FAIL


def test_a_valid_argument_under_a_rich_schema_still_reaches_the_world() -> None:
    """The gate must not reject calls the contract allows."""
    scenario = _scenario(
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {
                    "type": "object",
                    "required": ["reservation_id", "cabin"],
                    "properties": {
                        "reservation_id": {"type": "string"},
                        "cabin": {"type": "string", "enum": ["economy", "business"]},
                    },
                },
            },
        )
    )

    def candidate(**_):
        return CandidateAction(
            calls=(
                ActionCall(
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC", "cabin": "economy"},
                ),
            )
        )

    result = _run(scenario, candidate=candidate)
    assert result.terminated_by is not TerminationReason.INVALID_CANDIDATE_ACTION


# --- a raw reply is parsed, never blanked --------------------------------


def test_a_json_reply_is_recorded_as_the_call_it_named() -> None:
    """A candidate that answers in JSON used to be replaced by an empty
    action, so a real tool call became silence: no call reached the AWM, the
    step was scored as a plain message, and nothing recorded that the
    candidate had acted at all."""
    replies = iter(
        [
            '{"calls": [{"tool": "cancel_reservation", "arguments": {"reservation_id": "ABC"}}]}',
            CandidateAction(done=True),
        ]
    )

    def candidate(**_):
        return next(replies)

    result = _run(candidate=candidate)

    assert [call.tool for step in result.steps for call in step.calls] == ["cancel_reservation"]


def test_a_raw_text_reply_is_kept_as_the_message_it_was() -> None:
    """The communication half is scored from what the candidate said. An
    erased reply makes every communication requirement unmeetable."""

    def candidate(**_):
        return "I have cancelled the reservation for you."

    result = _run(candidate=candidate)

    assert any(
        step.action_content == "I have cancelled the reservation for you."
        for step in result.steps
    )
