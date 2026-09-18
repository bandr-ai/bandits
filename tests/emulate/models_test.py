"""The emulate contracts, exercised where they are meant to refuse.

Each test names the failure it prevents rather than the field it sets: an
invariant nobody can state the harm of is an invariant nobody will keep.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bandits.analyze.models import EvidenceKind
from bandits.emulate.models import (
    ActionCall,
    CandidateView,
    ClaimOrigin,
    CommunicationRequirement,
    ComponentResult,
    ExpectedEffect,
    ForbiddenEffect,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    MatchPolicy,
    Partition,
    PrefixStep,
    ResultStatus,
    RolloutResult,
    RolloutStep,
    Scenario,
    ScenarioKind,
    ScenarioState,
    SealedSuccessContract,
    StateField,
    SuccessShape,
    TerminationReason,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
    VerifierBinding,
    VerifierInputClaim,
    WorldOrigin,
)


def _contract(**updates) -> SealedSuccessContract:
    base = dict(
        contract_id="contract-1",
        source_task_id="7",
        shape=SuccessShape.MUTATION,
        required_effects=(
            ExpectedEffect(
                effect_id="e1",
                tool="cancel_reservation",
                arguments={"reservation_id": "ABC123"},
            ),
        ),
    )
    return SealedSuccessContract(**{**base, **updates})


def _scenario(**updates) -> Scenario:
    base = dict(
        scenario_id="scenario-1",
        kind=ScenarioKind.TASK_START,
        task="cancel my flight",
        success_contract=_contract(),
        source_trace_id="airline-10",
        source_task_id="7",
        family_id="family-451ae91f975c",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("airline-10",),
    )
    return Scenario(**{**base, **updates})


def _component(status: ResultStatus) -> ComponentResult:
    return ComponentResult(status=status)


def _rollout(**updates) -> RolloutResult:
    base = dict(
        rollout_id="rollout-1",
        scenario_id="scenario-1",
        binding_id="binding-1",
        candidate_id="candidate-1",
        seed=0,
        terminated_by=TerminationReason.CANDIDATE_COMPLETED,
        operational_result=_component(ResultStatus.PASS),
        process_result=_component(ResultStatus.NOT_APPLICABLE),
        communication_result=_component(ResultStatus.PASS),
        overall=ResultStatus.PASS,
    )
    return RolloutResult(**{**base, **updates})


# --- the candidate/environment boundary ---------------------------------


def test_candidate_view_carries_no_private_field() -> None:
    """A leak here hands the candidate the answer and inflates every score."""
    scenario = _scenario(
        hidden_user=HiddenUserProfile(known_info="user id is anya_garcia_5901"),
    )
    view = scenario.candidate_view()

    assert isinstance(view, CandidateView)
    shown = set(view.model_dump())
    assert "hidden_user" not in shown
    assert "success_contract" not in shown
    assert "retrieval_excluded_trace_ids" not in shown
    assert "initial_state" not in shown
    # The whole point: what a candidate sees is a strict subset of the scenario.
    assert shown < set(scenario.model_dump())


def test_scenario_must_exclude_its_own_trace_from_retrieval() -> None:
    """Otherwise the AWM copies the real next observation and fidelity is a lie."""
    with pytest.raises(ValidationError, match="exclude its own trace"):
        _scenario(retrieval_excluded_trace_ids=())


def test_initial_state_cannot_contain_simulated_fields() -> None:
    """A scenario starts from what the prefix revealed; anything else is invention."""
    invented = ScenarioState(
        fields=(
            StateField(
                path="reservations.ABC123.status",
                value="cancelled",
                origin=WorldOrigin.SIMULATED,
                revealed_at_step=0,
            ),
        )
    )
    with pytest.raises(ValidationError, match="not recorded"):
        _scenario(initial_state=invented)


def test_prefixed_scenario_records_where_it_cut() -> None:
    step = PrefixStep(role="user", content="hi")
    with pytest.raises(ValidationError, match="must record where it cut"):
        _scenario(kind=ScenarioKind.MIDDLE_PREFIX, prefix=(step,), cut_span_id=None)


def test_task_start_shows_no_history() -> None:
    with pytest.raises(ValidationError, match="shows no authentic history"):
        _scenario(prefix=(PrefixStep(role="user", content="hi"),))


# --- success shapes -----------------------------------------------------


def test_refusal_contract_cannot_require_a_mutation() -> None:
    """A refusal succeeds by *not* acting; a required effect contradicts it."""
    with pytest.raises(ValidationError, match="cannot require a state effect"):
        _contract(
            shape=SuccessShape.REFUSAL,
            forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
        )


def test_refusal_contract_must_name_what_was_forbidden() -> None:
    with pytest.raises(ValidationError, match="must name what was forbidden"):
        _contract(shape=SuccessShape.REFUSAL, required_effects=())


def test_informational_contract_claims_no_operational_success() -> None:
    """The five read-only tasks in the target family must not report a pass."""
    with pytest.raises(ValidationError, match="makes it a mutation"):
        _contract(
            shape=SuccessShape.INFORMATIONAL,
            communication=(CommunicationRequirement(requirement_id="c1", assertion="tell them"),),
        )


def test_informational_contract_needs_something_to_score() -> None:
    with pytest.raises(ValidationError, match="nothing would be scored"):
        _contract(shape=SuccessShape.INFORMATIONAL, required_effects=(), communication=())


def test_compound_contract_requires_more_than_one_effect() -> None:
    with pytest.raises(ValidationError, match="more than one effect"):
        _contract(shape=SuccessShape.COMPOUND)


def test_only_mutating_shapes_claim_operational_success() -> None:
    informational = _contract(
        shape=SuccessShape.INFORMATIONAL,
        required_effects=(),
        communication=(CommunicationRequirement(requirement_id="c1", assertion="report status"),),
    )
    assert not informational.claims_operational_success
    assert _contract().claims_operational_success


def test_unbindable_contract_skips_shape_validation() -> None:
    """An unreviewable source contract is verifier-unknown, not forced into a tag."""
    contract = _contract(
        shape=SuccessShape.REFUSAL,
        required_effects=(),
        forbidden_effects=(),
        unbindable_reason="source declares no checkable outcome",
    )
    assert contract.unbindable_reason


# --- reads vs writes ----------------------------------------------------


def test_unreviewed_tool_is_unknown_not_guessed_from_its_name() -> None:
    """``get_*`` looks like metadata and is actually a guess; tau2 declares none."""
    catalog = ToolEffectCatalog(
        catalog_id="catalog-1",
        toolset_digest="digest",
        entries=(
            ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="alex"),
        ),
    )
    assert catalog.is_write("cancel_reservation")
    assert catalog.effect_of("get_reservation_details") is ToolEffect.UNKNOWN
    assert not catalog.is_write("get_reservation_details")


def test_read_tool_cannot_declare_mutated_paths() -> None:
    with pytest.raises(ValidationError, match="cannot declare mutated paths"):
        ToolEffectEntry(
            tool="get_user_details",
            effect=ToolEffect.READ,
            reviewed_by="alex",
            mutates_paths=("users.x.name",),
        )


def test_classifying_a_tool_requires_a_named_reviewer() -> None:
    with pytest.raises(ValidationError, match="named reviewer"):
        ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="  ")


def test_catalog_refuses_to_classify_one_tool_twice() -> None:
    with pytest.raises(ValidationError, match="one tool twice"):
        ToolEffectCatalog(
            catalog_id="catalog-1",
            toolset_digest="digest",
            entries=(
                ToolEffectEntry(
                    tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="a"
                ),
                ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.READ, reviewed_by="a"),
            ),
        )


# --- match policy -------------------------------------------------------


def test_multiplicity_is_the_default() -> None:
    """Task 7 needs two distinct cancellations; cancelling one twice is not both."""
    assert MatchPolicy().multiplicity_required
    assert MatchPolicy().mode == "multiset"


def test_ordering_is_only_meaningful_when_declared() -> None:
    with pytest.raises(ValidationError, match="only meaningful in partial_order"):
        MatchPolicy(mode="multiset", ordered_pairs=((0, 1),))
    with pytest.raises(ValidationError, match="must declare at least one ordered pair"):
        MatchPolicy(mode="partial_order")


def test_informational_binding_has_no_effects_to_order() -> None:
    with pytest.raises(ValidationError, match="no effects to order"):
        VerifierBinding(
            binding_id="binding-1",
            scenario_id="scenario-1",
            contract_id="contract-1",
            shape=SuccessShape.INFORMATIONAL,
            template_id="template-informational-v1",
            match_policy=MatchPolicy(mode="partial_order", ordered_pairs=((0, 1),)),
        )


# --- communication ------------------------------------------------------


def test_communicate_info_must_name_the_facts_to_convey() -> None:
    with pytest.raises(ValidationError, match="must name the facts"):
        CommunicationRequirement(
            requirement_id="c1", assertion="tell the refund amount", kind="communicate_info"
        )


def test_nl_assertions_default_to_model_judgment() -> None:
    """Most tau2 communication rests here, so its authority must not read higher."""
    requirement = CommunicationRequirement(requirement_id="c1", assertion="agent should refuse")
    assert requirement.authority is EvidenceKind.MODEL_JUDGMENT


# --- state origin -------------------------------------------------------


def test_unknown_state_is_absent_not_false() -> None:
    state = ScenarioState(
        fields=(
            StateField(
                path="reservations.ABC123.status",
                value="confirmed",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="span-3",
            ),
        )
    )
    assert state.known("reservations.ABC123.status")
    assert state.get("reservations.ABC123.cabin") is None
    assert not state.known("reservations.ABC123.cabin")


def test_state_field_origin_matches_its_provenance() -> None:
    with pytest.raises(ValidationError, match="cannot have been committed by a rollout step"):
        StateField(path="p", value=1, origin=WorldOrigin.RECORDED, revealed_at_step=2)
    with pytest.raises(ValidationError, match="cannot cite an authentic span"):
        StateField(path="p", value=1, origin=WorldOrigin.SIMULATED, revealed_by_span_id="span-1")


def test_simulated_paths_are_what_condition_a_verdict() -> None:
    state = ScenarioState(
        fields=(
            StateField(path="a", value=1, origin=WorldOrigin.RECORDED, revealed_by_span_id="s1"),
            StateField(path="b", value=2, origin=WorldOrigin.SIMULATED, revealed_at_step=1),
        )
    )
    assert state.simulated_paths == ("b",)


def test_world_origin_is_separate_from_authority() -> None:
    """Origin and trust are independent axes; collapsing them loses both."""
    claim = VerifierInputClaim(
        claim="final_state_field",
        value={"field": "reservations.ABC123.status", "value": "cancelled"},
        origin=ClaimOrigin(
            world=WorldOrigin.SIMULATED,
            authority=EvidenceKind.TERMINAL_STATE_CHECK,
        ),
    )
    assert claim.origin.world is WorldOrigin.SIMULATED
    assert claim.origin.authority is EvidenceKind.TERMINAL_STATE_CHECK


# --- grounding transitions ----------------------------------------------


def test_unobserved_transition_carries_no_observation() -> None:
    """Silence is not approval, and cannot teach a next-observation predictor."""
    with pytest.raises(ValidationError, match="cannot carry an observation"):
        GroundingTransition(
            transition_id="t1",
            trace_id="airline-10",
            family_id="f",
            turn_index=4,
            action_span_id="span-9",
            observed=False,
            observations=(GroundingObservation(role="tool", content={"status": "ok"}),),
        )


def test_observed_transition_must_carry_what_reacted() -> None:
    with pytest.raises(ValidationError, match="must carry what reacted"):
        GroundingTransition(
            transition_id="t1",
            trace_id="airline-10",
            family_id="f",
            turn_index=4,
            action_span_id="span-9",
            observed=True,
        )


def test_grounding_transition_keeps_structure_and_records_stripped_markers() -> None:
    transition = GroundingTransition(
        transition_id="t1",
        trace_id="airline-10",
        family_id="f",
        turn_index=4,
        action_span_id="span-9",
        action_calls=(
            ActionCall(
                call_id="c1",
                tool="get_reservation_details",
                arguments={"reservation_id": "3RK2T9"},
            ),
        ),
        observations=(
            GroundingObservation(
                role="tool",
                tool_name="get_reservation_details",
                tool_call_id="c1",
                content={"reservation_id": "3RK2T9", "cabin": "basic_economy"},
                span_id="span-10",
            ),
        ),
        stripped_markers=("###TRANSFER###",),
    )
    # Structured, not rendered: a fidelity diff needs the fields, not a string.
    assert transition.observations[0].content["cabin"] == "basic_economy"
    assert transition.stripped_markers == ("###TRANSFER###",)
    assert transition.reaction_role == "tool"
    assert transition.action_tool == "get_reservation_details"


def test_a_batched_action_keeps_every_call_and_its_pairing() -> None:
    """54 tau2 actions carry several calls; one carries ten."""
    transition = GroundingTransition(
        transition_id="t1",
        trace_id="airline-10",
        family_id="f",
        turn_index=0,
        action_span_id="s1",
        action_span_ids=("s1", "s2"),
        action_calls=(
            ActionCall(
                call_id="c1", tool="get_reservation_details", arguments={"reservation_id": "A"}
            ),
            ActionCall(
                call_id="c2", tool="get_reservation_details", arguments={"reservation_id": "B"}
            ),
        ),
        observations=(
            GroundingObservation(role="tool", tool_call_id="c1", content={"status": "confirmed"}),
            GroundingObservation(role="tool", tool_call_id="c2", content={"status": "cancelled"}),
        ),
    )
    assert len(transition.action_calls) == 2
    # A batch has no single tool, and saying otherwise would pick one arbitrarily.
    assert transition.action_tool is None
    paired = {o.tool_call_id: o.content["status"] for o in transition.observations}
    assert paired == {"c1": "confirmed", "c2": "cancelled"}


def test_a_mixed_reaction_reports_itself_as_mixed() -> None:
    transition = GroundingTransition(
        transition_id="t1",
        trace_id="a",
        family_id="f",
        turn_index=0,
        action_span_id="s1",
        observations=(
            GroundingObservation(role="tool", content={"ok": True}),
            GroundingObservation(role="user", content="thanks"),
        ),
    )
    assert transition.reaction_role == "mixed"


# --- rollout verdicts ---------------------------------------------------


def test_unknown_required_component_makes_the_verdict_unknown() -> None:
    with pytest.raises(ValidationError, match="never a pass"):
        _rollout(communication_result=_component(ResultStatus.UNKNOWN), overall=ResultStatus.PASS)


def test_failed_component_fails_the_whole_verdict() -> None:
    with pytest.raises(ValidationError, match="must fail the overall verdict"):
        _rollout(operational_result=_component(ResultStatus.FAIL), overall=ResultStatus.PASS)


def test_informational_rollout_passes_on_communication_alone() -> None:
    """Operational is not-applicable, not unknown, so it does not block a pass."""
    rollout = _rollout(
        operational_result=_component(ResultStatus.NOT_APPLICABLE),
        communication_result=_component(ResultStatus.PASS),
        overall=ResultStatus.PASS,
    )
    assert rollout.overall is ResultStatus.PASS


def test_a_pass_needs_something_that_actually_passed() -> None:
    with pytest.raises(ValidationError, match="at least one component that passed"):
        _rollout(
            operational_result=_component(ResultStatus.NOT_APPLICABLE),
            communication_result=_component(ResultStatus.NOT_APPLICABLE),
            overall=ResultStatus.PASS,
        )


def test_abstention_leaves_the_denominator() -> None:
    """Charging the candidate for the simulator's ignorance would be a lie."""
    assert TerminationReason.AWM_ABSTAINED.invalidates_rollout
    assert TerminationReason.UNSUPPORTED_ACTION.invalidates_rollout
    assert not TerminationReason.CANDIDATE_COMPLETED.invalidates_rollout

    with pytest.raises(ValidationError, match="left the denominator"):
        _rollout(terminated_by=TerminationReason.AWM_ABSTAINED, overall=ResultStatus.PASS)


def test_step_limit_is_a_real_failure_not_an_abstention() -> None:
    """The candidate ran out of budget; the environment answered every time."""
    assert not TerminationReason.STEP_LIMIT.invalidates_rollout
    rollout = _rollout(
        terminated_by=TerminationReason.STEP_LIMIT,
        operational_result=_component(ResultStatus.FAIL),
        communication_result=_component(ResultStatus.NOT_APPLICABLE),
        overall=ResultStatus.FAIL,
    )
    assert rollout.overall is ResultStatus.FAIL


def test_abstained_step_commits_nothing() -> None:
    with pytest.raises(ValidationError, match="cannot commit state"):
        RolloutStep(index=0, abstained=True, committed_delta={"a": 1})


def test_not_applicable_component_reports_no_unmet_requirements() -> None:
    with pytest.raises(ValidationError, match="cannot report unmet requirements"):
        ComponentResult(status=ResultStatus.NOT_APPLICABLE, unmet=("e1",))
