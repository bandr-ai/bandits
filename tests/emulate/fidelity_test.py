"""Adversarial tests for simulator-fidelity measurement."""

from bandits.emulate.fidelity import (
    FidelityReport,
    TransitionFidelity,
    compare_observation,
    gate,
    multi_step_drift,
    score_disclosure,
    score_transition_fidelity,
)
from bandits.emulate.models import (
    ActionCall,
    DeltaGroundTruthStatus,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    ScenarioState,
    SupportLevel,
)
from bandits.emulate.retrieve import RetrievedExample
from bandits.emulate.world import (
    ProposedCallOutcome,
    ProposedTransition,
    ProposedUserTurn,
    StateDelta,
)


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
        delta_ground_truth_status=DeltaGroundTruthStatus.MEASURED,
    )


def _example(transition: GroundingTransition) -> RetrievedExample:
    return RetrievedExample(transition=transition, score=1.0, reasons=("exact_tool",))


def _batched_transition(tid: str, *, call_ids: tuple[str, ...]) -> GroundingTransition:
    """A transition whose action batched len(call_ids) calls to the same tool,
    each with its own recorded tool_call_id-correlated observation."""
    return GroundingTransition(
        transition_id=tid,
        trace_id=f"trace-{tid}",
        family_id="family",
        turn_index=0,
        action_span_id=f"span-{tid}",
        action_calls=tuple(
            ActionCall(call_id=cid, tool="cancel_reservation", arguments={"reservation_id": cid})
            for cid in call_ids
        ),
        observations=tuple(
            GroundingObservation(
                role="tool",
                tool_name="cancel_reservation",
                tool_call_id=cid,
                content={"reservation_id": cid, "status": "cancelled"},
            )
            for cid in call_ids
        ),
        state_before=ScenarioState(),
    )


def _batched_predictor_response(
    outcomes: tuple[ProposedCallOutcome, ...]
) -> ProposedTransition:
    return ProposedTransition(call_outcomes=outcomes, support=SupportLevel.HIGH)


def test_single_call_field_accuracy_compares_the_committed_observation() -> None:
    """The regression this whole correlation path exists for: a model that
    correctly used the batch call_outcomes form for a single call must still
    be scored against what it actually said, not against the (correctly
    empty) top-level proposal.observation."""
    transition = _batched_transition("single", call_ids=("call-1",))

    def predictor(**_):
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-1",
                    observation={"reservation_id": "call-1", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
            )
        )

    result = score_transition_fidelity(
        transition, predictor, examples=(_example(transition.replace(transition_id="support")),)
    )
    assert result.field_accuracy == 1.0
    assert result.unmatched_call_observations == ()


def test_reordered_batch_still_correlates_by_call_id() -> None:
    """Calls answered out of order must still pair with the right recorded
    observation -- correlation is by call_id, never by position."""
    transition = _batched_transition("reordered", call_ids=("call-A", "call-B"))

    def predictor(**_):
        # Outcomes deliberately returned in reverse order.
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-B",
                    observation={"reservation_id": "call-B", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
                ProposedCallOutcome(
                    call_id="call-A",
                    observation={"reservation_id": "call-A", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
            )
        )

    result = score_transition_fidelity(
        transition, predictor, examples=(_example(transition.replace(transition_id="support")),)
    )
    assert result.field_accuracy == 1.0
    assert result.unmatched_call_observations == ()


def test_partial_batch_scores_matched_calls_and_flags_the_rest() -> None:
    """A batch where only some calls got an outcome: the validator rejects
    this outright (D70 -- every submitted call must be answered), so fidelity
    must see a validator rejection, not a partial/guessed score."""
    transition = _batched_transition("partial", call_ids=("call-A", "call-B"))

    def predictor(**_):
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-A",
                    observation={"reservation_id": "call-A", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
            )
        )

    result = score_transition_fidelity(
        transition, predictor, examples=(_example(transition.replace(transition_id="support")),)
    )
    assert result.validator_rejected is True
    assert result.fields == ()


def test_batch_status_correctness_is_per_call_not_aggregate() -> None:
    """A batch where call-A actually errored and call-B did not: a prediction
    that swaps which call errored must be scored wrong, even though "an error
    happened somewhere in the batch" is true on both sides. Aggregating
    error_predicted/error_recorded across the whole batch (the old behaviour)
    could not detect this -- both sides say "yes, one call errored" and the
    comparison passes despite attributing the failure to the wrong call.
    """
    transition = GroundingTransition(
        transition_id="swapped-error",
        trace_id="trace-swapped-error",
        family_id="family",
        turn_index=0,
        action_span_id="span-swapped-error",
        action_calls=(
            ActionCall(call_id="call-A", tool="cancel_reservation", arguments={"reservation_id": "A"}),
            ActionCall(call_id="call-B", tool="cancel_reservation", arguments={"reservation_id": "B"}),
        ),
        observations=(
            GroundingObservation(
                role="tool", tool_name="cancel_reservation", tool_call_id="call-A",
                content={"reservation_id": "A", "status": "error"}, error=True,
            ),
            GroundingObservation(
                role="tool", tool_name="cancel_reservation", tool_call_id="call-B",
                content={"reservation_id": "B", "status": "cancelled"},
            ),
        ),
        state_before=ScenarioState(),
    )

    def predictor(**_):
        # Swapped: predicts B errored and A succeeded -- the reverse of what
        # actually happened.
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-A",
                    observation={"reservation_id": "A", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
                ProposedCallOutcome(
                    call_id="call-B",
                    observation={"reservation_id": "B", "status": "error"},
                    evidence_ids=("support",),
                ),
            )
        )

    result = score_transition_fidelity(
        transition, predictor, examples=(_example(transition.replace(transition_id="support")),)
    )
    assert result.unmatched_call_observations == ()
    assert result.status_correct is False


def test_recorded_error_flag_counts_even_without_an_error_shaped_payload() -> None:
    """A recorded observation can be flagged error=True while its payload
    looks perfectly fine (no "error" key, no "error"-prefixed string) -- the
    source system's own error signal, not the payload shape, is authoritative.
    A prediction that (correctly, per the actual recorded outcome) also
    predicts an error must be scored right; one that predicts success must be
    scored wrong, in both cases driven by the .error flag, not just content.
    """
    transition = GroundingTransition(
        transition_id="silent-error",
        trace_id="trace-silent-error",
        family_id="family",
        turn_index=0,
        action_span_id="span-silent-error",
        action_calls=(
            ActionCall(call_id="call-1", tool="cancel_reservation", arguments={"reservation_id": "X"}),
        ),
        observations=(
            GroundingObservation(
                role="tool",
                tool_name="cancel_reservation",
                tool_call_id="call-1",
                # error=True but the payload itself doesn't look like an
                # error at all -- no "error" key, no "error"-prefixed string.
                content={"reservation_id": "X", "status": "pending"},
                error=True,
            ),
        ),
        state_before=ScenarioState(),
    )

    def predicts_success(**_):
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-1",
                    observation={"reservation_id": "X", "status": "pending"},
                    evidence_ids=("support",),
                ),
            )
        )

    result = score_transition_fidelity(
        transition, predicts_success, examples=(_example(transition.replace(transition_id="support")),)
    )
    # The predicted payload doesn't look like an error and the model never
    # flagged one -- but the recorded call DID error (per its .error flag),
    # so this must be scored wrong, not right.
    assert result.status_correct is False


def test_unmatched_call_id_is_flagged_not_silently_dropped() -> None:
    """An outcome citing a call_id the validator never even sees matched to a
    recorded observation (mismatched ids on both sides) must show up as
    unmatched, never silently vanish from field accuracy."""
    transition = _batched_transition("unmatched", call_ids=("call-A", "call-B"))

    def predictor(**_):
        # Outcome call_ids match the submitted calls (so the validator
        # accepts), but the recorded side's tool_call_id for one of them is
        # deliberately absent from this transition's own observations by
        # construction below.
        return _batched_predictor_response(
            (
                ProposedCallOutcome(
                    call_id="call-A",
                    observation={"reservation_id": "call-A", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
                ProposedCallOutcome(
                    call_id="call-B",
                    observation={"reservation_id": "call-B", "status": "cancelled"},
                    evidence_ids=("support",),
                ),
            )
        )

    # Rebuild the transition with call-B's recorded observation missing its
    # tool_call_id, so the correlator cannot pair it despite the validator
    # accepting the proposal (both submitted calls got an outcome).
    broken = transition.replace(
        observations=(
            transition.observations[0],
            transition.observations[1].replace(tool_call_id=None),
        )
    )

    result = score_transition_fidelity(
        broken, predictor, examples=(_example(transition.replace(transition_id="support")),)
    )
    assert result.validator_rejected is False
    assert "call-B" in result.unmatched_call_observations
    assert "call-A" not in result.unmatched_call_observations
    # Incomplete correlation makes the WHOLE transition unscorable, not a
    # partial score over call-A alone: reporting call-A's fields as if they
    # were the transition's field_accuracy would hide that call-B was never
    # scored at all.
    assert result.fields == ()
    assert result.status_correct is None


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
