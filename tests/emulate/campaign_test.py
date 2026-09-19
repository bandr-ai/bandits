"""The campaign refuses invalid simulators before spending candidate calls."""

import pytest

from bandits.emulate.campaign import (
    PUBLISHABLE_FIDELITY_METRICS,
    CampaignBlocked,
    CampaignResult,
    CampaignSpec,
    load_campaign,
    run_campaign,
    save_campaign,
)
from bandits.emulate.fidelity import FidelityReport, TransitionFidelity
from bandits.emulate.models import (
    ActionCall,
    ExpectedEffect,
    GroundingObservation,
    GroundingTransition,
    Partition,
    Scenario,
    ScenarioKind,
    SealedSuccessContract,
    SuccessShape,
    SupportLevel,
    SupportPolicy,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
    VerifierBinding,
)
from bandits.emulate.rollout import CandidateAction
from bandits.emulate.world import ProposedTransition, StateDelta
from bandits.store import DerivedStore


def _spec() -> CampaignSpec:
    return CampaignSpec(
        campaign_id="campaign",
        scenario_set_version="scenarios",
        retrieval_index_version="index",
        tool_awm_version="awm",
        user_policy_version="user",
        fidelity_report_id="fidelity",
        support_policy=SupportPolicy(
            policy_id="support",
            tool_minimum=SupportLevel.MEDIUM,
            reviewed_by="reviewer",
            calibration_report_id="fidelity",
        ),
        mode="development",
    )


def _binding(scenario: Scenario) -> VerifierBinding:
    return VerifierBinding(
        binding_id="binding-1",
        scenario_id=scenario.scenario_id,
        contract_id=scenario.success_contract.contract_id,
        shape=scenario.success_contract.shape,
        template_id="family-checks",
        required_reward_basis=scenario.success_contract.reward_basis,
    )


def test_failed_fidelity_blocks_before_any_candidate_runs() -> None:
    calls = []

    def factory(_seed):
        calls.append(1)
        raise AssertionError("candidate must not be constructed")

    fidelity = FidelityReport(
        awm_version="awm",
        user_policy_version="user",
        retrieval_index_version="index",
        transitions=(
            TransitionFidelity(
                transition_id="bad", trace_id="bad", status_correct=False
            ),
        )
    )
    with pytest.raises(CampaignBlocked, match="fidelity gate"):
        run_campaign(
            _spec(),
            scenarios=(),
            index=(),
            candidates={"candidate": factory},
            bindings={},
            tool_world=lambda **_: None,
            user_policy=lambda **_: None,
            fidelity=fidelity,
            fidelity_thresholds={"status_accuracy": 1.0},
        )
    assert calls == []


def test_a_publishable_campaign_must_bound_invention_and_disclosure() -> None:
    """The two metrics the environment's whole claim rests on.

    An unbounded invention rate means the simulator may be rescuing or
    condemning the candidate on the tool's behalf; an unbounded disclosure rate
    means the user policy may be handing over the scenario. Either one changes
    the difficulty the capability number is supposed to be measuring, so a
    publishable policy that never names a bar for them is not publishable.
    """
    assert "invention_rate" in PUBLISHABLE_FIDELITY_METRICS
    assert "premature_disclosure_rate" in PUBLISHABLE_FIDELITY_METRICS

    spec = _spec().replace(mode="publishable", reference_id="ref", control_id="ctl")
    with pytest.raises(CampaignBlocked, match="omits required metrics") as raised:
        run_campaign(
            spec,
            scenarios=(),
            index=(),
            candidates={},
            bindings={},
            tool_world=lambda **_: None,
            user_policy=lambda **_: None,
            fidelity=FidelityReport(split="held_out"),
            fidelity_thresholds={
                "status_accuracy": 1.0,
                "field_accuracy": 1.0,
                "supported_coverage": 1.0,
                "wrong_abstention_rate": 0.0,
                "validation_rejection_rate": 0.0,
            },
        )
    assert "invention_rate" in str(raised.value)
    assert "premature_disclosure_rate" in str(raised.value)


def test_campaign_artifact_round_trips_without_dropping_failures(tmp_path) -> None:
    result = CampaignResult(
        campaign_id="campaign",
        fidelity_report_id="fidelity",
        support_policy_id="support",
        reports=(),
        rollouts=(),
    )
    store = DerivedStore(tmp_path / ".bandits")
    envelope = save_campaign(result, parent_artifact_id="corpus", store=store)
    assert envelope.kind == "emulation-campaign"
    assert load_campaign(envelope.artifact_id, store) == result


def test_fake_campaign_persists_output_validation_and_event_provenance(tmp_path) -> None:
    scenario = Scenario(
        scenario_id="scenario-1",
        kind=ScenarioKind.TASK_START,
        task="cancel reservation ABC",
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {
                    "type": "object",
                    "required": ["reservation_id"],
                    "properties": {"reservation_id": {"type": "string"}},
                },
                "output_schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"const": "cancelled"}},
                },
            },
        ),
        success_contract=SealedSuccessContract(
            contract_id="contract-1",
            source_task_id="7",
            shape=SuccessShape.MUTATION,
            required_effects=(
                ExpectedEffect(
                    effect_id="cancelled",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC"},
                    event_type="reservation_cancelled",
                    state_path="cancel_reservation.ABC.status",
                    expected_value="cancelled",
                ),
            ),
        ),
        source_trace_id="source-trace",
        source_task_id="7",
        family_id="family-1",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("source-trace",),
    )
    index = tuple(
        GroundingTransition(
            transition_id=f"evidence-{number}",
            trace_id=f"fit-trace-{number}",
            family_id="family-1",
            turn_index=0,
            task_context="cancel reservation ABC",
            success_shape=SuccessShape.MUTATION,
            action_span_id=f"span-{number}",
            action_calls=(
                ActionCall(
                    call_id=f"recorded-{number}",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "ABC"},
                ),
            ),
            observations=(
                GroundingObservation(role="tool", content={"status": "cancelled"}),
            ),
        )
        for number in range(3)
    )

    def candidate_factory(_seed):
        calls = {"count": 0}

        def candidate(**_):
            calls["count"] += 1
            if calls["count"] == 1:
                return CandidateAction(
                    calls=(
                        ActionCall(
                            call_id="call-a",
                            tool="cancel_reservation",
                            arguments={"reservation_id": "ABC"},
                        ),
                    )
                )
            return CandidateAction(done=True)

        return candidate

    def tool_world(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(
                StateDelta(
                    path="cancel_reservation.ABC.status",
                    new_value="cancelled",
                ),
            ),
            events=({"type": "reservation_cancelled"},),
            support=SupportLevel.HIGH,
            evidence_ids=("evidence-0",),
        )

    catalog = ToolEffectCatalog(
        catalog_id="catalog",
        toolset_digest="tools",
        entries=(
            ToolEffectEntry(
                tool="cancel_reservation",
                effect=ToolEffect.WRITE,
                reviewed_by="reviewer",
                mutates_paths=("cancel_reservation.*.status",),
            ),
        ),
    )
    fidelity = FidelityReport(
        awm_version="awm",
        user_policy_version="user",
        retrieval_index_version="index",
        transitions=(
            TransitionFidelity(
                transition_id="fidelity-1",
                trace_id="held-out-1",
                status_correct=True,
            ),
        )
    )
    result = run_campaign(
        _spec(),
        scenarios=(scenario,),
        index=index,
        candidates={"candidate": candidate_factory},
        bindings={"scenario-1": _binding(scenario)},
        tool_world=tool_world,
        user_policy=lambda **_: None,
        fidelity=fidelity,
        fidelity_thresholds={"status_accuracy": 1.0},
        catalog=catalog,
    )
    store = DerivedStore(tmp_path / ".bandits")
    envelope = save_campaign(result, parent_artifact_id="corpus", store=store)
    restored = load_campaign(envelope.artifact_id, store)

    assert len(restored.rollouts) == 1
    assert restored.rollouts[0].steps[0].output_schema_validation == {
        "call-a": "validated"
    }
    assert restored.rollouts[0].steps[0].events == (
        {"type": "reservation_cancelled", "_call_id": "call-a"},
    )
