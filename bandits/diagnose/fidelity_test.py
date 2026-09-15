"""Adversarial tests for simulator-fidelity measurement."""

from bandits.diagnose.fidelity import (
    FidelityReport,
    TransitionFidelity,
    compare_observation,
    gate,
    multi_step_drift,
    score_disclosure,
    score_transition_fidelity,
)
from bandits.diagnose.models import (
    ActionCall,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    ScenarioState,
    SupportLevel,
)
from bandits.diagnose.retrieve import RetrievedExample
from bandits.diagnose.world import ProposedTransition, ProposedUserTurn, StateDelta


def test_list_contents_are_compared_not_only_their_lengths() -> None:
    rows = compare_observation(
        {"flights": [{"number": "BA2", "status": "cancelled"}]},
        {"flights": [{"number": "BA1", "status": "confirmed"}]},
    )
    assert {row.path for row in rows} >= {"flights[0].number", "flights[0].status"}
    assert any(not row.correct for row in rows)


def test_wrong_abstentions_are_measured_separately() -> None:
    report = FidelityReport(
        transitions=(
            TransitionFidelity(
                transition_id="supported-but-declined",
                trace_id="trace-1",
                abstained=True,
                abstain_correct=False,
            ),
            TransitionFidelity(
                transition_id="unsupported-and-declined",
                trace_id="trace-2",
                abstained=True,
                abstain_correct=True,
            ),
        )
    )
    assert report.wrong_abstention_rate == 0.5
    assert report.correct_abstention_rate == 0.5


def test_supported_coverage_counts_predictions_and_wrong_abstentions() -> None:
    report = FidelityReport(
        transitions=(
            TransitionFidelity(transition_id="predicted", trace_id="t1"),
            TransitionFidelity(
                transition_id="declined", trace_id="t2", abstained=True, abstain_correct=False
            ),
            TransitionFidelity(
                transition_id="correct-decline",
                trace_id="t3",
                abstained=True,
                abstain_correct=True,
            ),
        )
    )
    assert report.supported_coverage == 0.5


def _transition(tid: str, *, before: ScenarioState = ScenarioState(), value="cancelled"):
    return GroundingTransition(
        transition_id=tid,
        trace_id=f"trace-{tid}",
        family_id="family",
        turn_index=0,
        action_span_id=f"span-{tid}",
        action_calls=(
            ActionCall(
                call_id=f"call-{tid}",
                tool="cancel_reservation",
                arguments={"reservation_id": "ABC"},
            ),
        ),
        observations=(
            GroundingObservation(
                role="tool", tool_name="cancel_reservation", content={"status": value}
            ),
        ),
        state_before=before,
        inferred_state_delta={"reservations.ABC.status": value},
    )


def _example(transition: GroundingTransition) -> RetrievedExample:
    return RetrievedExample(transition=transition, score=1.0, reasons=("exact_tool",))


def test_delta_fidelity_compares_values_not_only_paths() -> None:
    transition = _transition("one")

    def predictor(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(
                StateDelta(path="reservations.ABC.status", new_value="confirmed"),
            ),
            support=SupportLevel.HIGH,
            evidence_ids=("support",),
        )

    result = score_transition_fidelity(
        transition,
        predictor,
        examples=(_example(_transition("support")),),
    )
    assert result.delta_correct is False


def test_multi_step_drift_feeds_predicted_state_into_the_next_step() -> None:
    transitions = (_transition("one", value="cancelled"), _transition("two", value="refunded"))
    seen_states: list[ScenarioState] = []
    values = iter(("cancelled", "refunded"))

    def predictor(**_):
        value = next(values)
        return ProposedTransition(
            observation={"status": value},
            state_delta=(StateDelta(path="reservations.ABC.status", new_value=value),),
            support=SupportLevel.HIGH,
            evidence_ids=("support",),
        )

    def retrieve_for(_transition, state):
        seen_states.append(state)
        return (_example(_transition_for_support),)

    _transition_for_support = _transition("support")
    drift = multi_step_drift(transitions, predictor, retrieve_for=retrieve_for)
    assert drift == (1.0, 1.0)
    assert seen_states[1].get("reservations.ABC.status").value == "cancelled"


def test_gate_rejects_low_supported_coverage_and_wrong_abstention() -> None:
    report = FidelityReport(
        transitions=(
            TransitionFidelity(transition_id="made", trace_id="t1"),
            TransitionFidelity(
                transition_id="declined", trace_id="t2", abstained=True, abstain_correct=False
            ),
        )
    )
    passed, failures = gate(
        report,
        thresholds={"supported_coverage": 0.75, "wrong_abstention_rate": 0.1},
    )
    assert not passed
    assert any("supported_coverage" in failure for failure in failures)
    assert any("wrong_abstention_rate" in failure for failure in failures)


def test_disclosure_uses_structured_fact_ids_when_available() -> None:
    profile = HiddenUserProfile(
        known_facts={"reservation_id": "ABC"},
        unknown_fact_ids=("insurance_status",),
    )
    outcome = score_disclosure(
        ProposedUserTurn(
            user_message="ABC; insurance is active",
            disclosed_facts=("reservation_id", "insurance_status"),
        ),
        transition_id="turn",
        profile=profile,
        history_text="",
        agent_message="Hello",
    )
    assert outcome.premature == ("reservation_id",)
    assert outcome.invented == ("insurance_status",)


def test_a_proposal_the_runtime_would_reject_does_not_score_as_correct() -> None:
    """D71/I32. The gate called ``step_tool_world`` and scored the raw proposal,
    so every rule added under D51/D63/D64/D67 was invisible to the measurement
    that decides whether the simulator may be used. A prompt could pass the
    fidelity gate on transitions rollouts would throw away.

    Here the prediction matches the recording exactly, but cites evidence that
    was never retrieved — an invention the runtime refuses.
    """
    transition = _transition("one", value="cancelled")

    def predictor(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(StateDelta(path="reservations.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("never-retrieved",),
        )

    result = score_transition_fidelity(
        transition,
        predictor,
        examples=(_example(_transition("support")),),
    )
    assert result.validator_rejected
    assert result.validator_rejections
    assert result.status_correct is not True


def test_a_valid_proposal_is_not_marked_rejected() -> None:
    """The gate must not report runtime rejection for transitions that pass."""
    transition = _transition("one", value="cancelled")

    def predictor(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(StateDelta(path="reservations.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("support",),
        )

    result = score_transition_fidelity(
        transition,
        predictor,
        examples=(_example(_transition("support")),),
    )
    assert not result.validator_rejected
    assert result.status_correct is True


def test_the_validation_rejection_rate_is_a_reported_gate_output() -> None:
    """D71. Rejection rate sits beside accuracy and coverage, not inside them."""
    report = FidelityReport(
        transitions=(
            TransitionFidelity(transition_id="ok", trace_id="t1"),
            TransitionFidelity(
                transition_id="rejected",
                trace_id="t2",
                validator_rejected=True,
                validator_rejections=("evidence ids were not retrieved for this step",),
            ),
        )
    )
    assert report.validation_rejection_rate == 0.5


def test_the_gate_can_block_on_runtime_rejection_rate() -> None:
    """D71. The rate is gateable, not merely reported."""
    report = FidelityReport(
        transitions=(
            TransitionFidelity(
                transition_id="rejected", trace_id="t1", validator_rejected=True
            ),
        )
    )
    passed, failures = gate(report, thresholds={"validation_rejection_rate": 0.1})
    assert not passed
    assert any("validation_rejection_rate" in failure for failure in failures)


def test_drift_does_not_carry_an_unvalidated_prediction_forward() -> None:
    """D71. Drift commits each prediction into the state the next step reasons
    from, so an invalid transition there contaminates every later horizon --
    the same defect as one-step scoring, compounded.
    """
    transitions = (_transition("one", value="cancelled"), _transition("two", value="refunded"))

    def predictor(**_):
        return ProposedTransition(
            observation={"status": "cancelled"},
            state_delta=(StateDelta(path="reservations.ABC.status", new_value="cancelled"),),
            support=SupportLevel.HIGH,
            evidence_ids=("never-retrieved",),
        )

    support = _transition("support")
    drift = multi_step_drift(
        transitions, predictor, retrieve_for=lambda _t, _s: (_example(support),)
    )
    assert drift[0] == 0.0
