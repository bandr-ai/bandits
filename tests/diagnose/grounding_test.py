"""Grounding assessment: I39's fix, on the boundaries it exists to hold.

Kept narrow on purpose: only single-call get_user_details/get_reservation_details
reads are classified. Writes, batches, and non-entity tools report UNAVAILABLE
rather than a guessed verdict -- the false positive this must never reproduce
is an entity id merely *mentioned* in state (e.g. a reservation's user_id
field) being read as that user's own record being known.
"""

from __future__ import annotations

from bandits.diagnose.grounding import assess_grounding
from bandits.diagnose.models import (
    ActionCall,
    GroundingKind,
    GroundingObservation,
    GroundingTransition,
    ScenarioState,
    StateField,
    WorldOrigin,
)
from bandits.diagnose.retrieve import RetrievedExample


def _transition(tid, calls, observations, trace="airline-1"):
    return GroundingTransition(
        transition_id=tid,
        trace_id=trace,
        family_id="f",
        turn_index=0,
        action_span_id=f"span-{tid}",
        action_calls=calls,
        observations=observations,
    )


def _single(tid, tool, arguments, observation, call_id=None, trace="airline-1"):
    return _transition(
        tid,
        (ActionCall(call_id=call_id, tool=tool, arguments=arguments),),
        (GroundingObservation(role="tool", content=observation, tool_call_id=call_id),),
        trace=trace,
    )


def _example(tid, tool, arguments, observation, call_id="c1", reasons=("exact_tool",)):
    return RetrievedExample(
        transition=_single(tid, tool, arguments, observation, call_id=call_id),
        score=1.0,
        reasons=reasons,
    )


def test_different_entity_evidence_is_behavioral_analogy_not_identifiable() -> None:
    """The I39 case: get_user_details for a stranger, only other users retrieved."""
    target = _single(
        "target", "get_user_details", {"user_id": "emma_kim_9957"}, {"name": "Emma Kim"}
    )
    examples = (
        _example(
            "e1",
            "get_user_details",
            {"user_id": "anya_garcia_5901"},
            {"name": "Anya Garcia", "email": "anya@example.com"},
        ),
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=examples)
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is False
    assert call.behavior_supported is True
    assert call.kind is GroundingKind.BEHAVIORAL_ANALOGY
    assert call.missing_fact_paths == ("name",)
    assert assessment.values_identifiable is False


def test_entity_id_merely_mentioned_in_state_is_not_identifiable() -> None:
    """A reservation naming its owner does not disclose the owner's record."""
    target = _single(
        "target",
        "get_user_details",
        {"user_id": "raj_sanchez_7340"},
        {"name": "Raj Sanchez", "email": "raj@example.com"},
    )
    state = ScenarioState(
        fields=(
            StateField(
                path="get_reservation_details.Q69X3R.user_id",
                value="raj_sanchez_7340",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    assessment = assess_grounding(target, state_before=state, examples=())
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is False
    assert call.kind in (GroundingKind.BEHAVIORAL_ANALOGY, GroundingKind.UNSUPPORTED)


def test_same_entity_already_in_state_is_derivable() -> None:
    """The record was already read earlier this trace, under the same tool."""
    target = _single(
        "target", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}
    )
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
    assessment = assess_grounding(target, state_before=state, examples=())
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is True
    # Reads a record already in state; does not derive one -- EXACT_ENTITY_FACT,
    # not DERIVABLE_FROM_STATE (reserved for mutations).
    assert call.kind is GroundingKind.EXACT_ENTITY_FACT


def test_same_entity_known_from_a_different_tool_is_still_identifiable() -> None:
    """Entity identity is tool-independent: cancel_reservation(Q69X3R) after a
    prior get_reservation_details(Q69X3R) must resolve to the same entity.
    Tool-namespaced keys (the first draft's bug) would miss this."""
    target = _single(
        "target",
        "get_reservation_details",
        {"reservation_id": "Q69X3R"},
        {"status": "confirmed"},
    )
    state = ScenarioState(
        fields=(
            StateField(
                path="get_reservation_details.Q69X3R.status",
                value="confirmed",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    assessment = assess_grounding(target, state_before=state, examples=())
    assert assessment.single_call.kind is GroundingKind.EXACT_ENTITY_FACT


def test_same_entity_in_retrieval_is_exact_entity_fact() -> None:
    target = _single(
        "target", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}
    )
    examples = (
        _example(
            "e1", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}
        ),
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=examples)
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is True
    assert call.kind is GroundingKind.EXACT_ENTITY_FACT
    assert call.exact_fact_evidence_ids == ("e1",)


def test_no_evidence_at_all_is_unsupported() -> None:
    target = _single(
        "target", "get_user_details", {"user_id": "emma_kim_9957"}, {"name": "Emma Kim"}
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=())
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is False
    assert call.behavior_supported is False
    assert call.kind is GroundingKind.UNSUPPORTED


def test_write_call_is_unavailable_not_guessed() -> None:
    """cancel_reservation is a mutation, not record materialization -- its
    recorded result is a new derived status, not facts pre-existing in
    state. This classifier must not claim it as identifiable or unsupported;
    it must say it was not assessed."""
    target = _single(
        "target",
        "cancel_reservation",
        {"reservation_id": "Q69X3R"},
        {"status": "cancelled"},
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=())
    assert assessment.single_call.kind is GroundingKind.UNAVAILABLE
    assert assessment.values_identifiable is None


def test_non_entity_tool_is_unavailable() -> None:
    target = _single("target", "transfer_to_human_agents", {}, {"transferred": True})
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=())
    assert assessment.single_call.kind is GroundingKind.UNAVAILABLE
    assert assessment.values_identifiable is None


def test_batch_with_one_identifiable_and_one_unidentifiable_call_is_not_pooled() -> None:
    """The batch bug from the first draft: known Raj facts must never leak
    into Emma's call, and each call's own verdict must stay separate."""
    target = _transition(
        "target",
        (
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
            ActionCall(call_id="b", tool="get_user_details", arguments={"user_id": "emma_kim_9957"}),
        ),
        (
            GroundingObservation(role="tool", content={"name": "Raj Sanchez"}, tool_call_id="a"),
            GroundingObservation(role="tool", content={"name": "Emma Kim"}, tool_call_id="b"),
        ),
    )
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
    assessment = assess_grounding(target, state_before=state, examples=())

    by_call = {c.call_id: c for c in assessment.calls}
    assert by_call["a"].values_identifiable is True
    assert by_call["b"].values_identifiable is False
    # Mixed batch: transition-level verdict must not be guessed either way.
    assert assessment.values_identifiable is None


def test_batch_where_every_call_is_unidentifiable_is_still_not_calibrated() -> None:
    """The sharper batch bug: two calls that individually agree (both False)
    must not let the aggregate collapse to False -- proposal-wide abstain is
    an all-or-nothing flag with no correct label for a batch, agreeing or not."""
    target = _transition(
        "target",
        (
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "emma_kim_9957"}),
            ActionCall(call_id="b", tool="get_user_details", arguments={"user_id": "anya_garcia_5901"}),
        ),
        (
            GroundingObservation(role="tool", content={"name": "Emma Kim"}, tool_call_id="a"),
            GroundingObservation(role="tool", content={"name": "Anya Garcia"}, tool_call_id="b"),
        ),
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=())

    by_call = {c.call_id: c for c in assessment.calls}
    assert by_call["a"].values_identifiable is False
    assert by_call["b"].values_identifiable is False
    assert assessment.values_identifiable is None


def test_batch_evidence_does_not_cross_tools() -> None:
    """A batch call for get_reservation_details must not borrow behavioral
    support from a same-batch get_user_details call just because both
    cleared retrieval's generic "exact_tool" reason."""
    target = _transition(
        "target",
        (
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
            ActionCall(call_id="b", tool="get_reservation_details", arguments={"reservation_id": "ZZZZZZ"}),
        ),
        (
            GroundingObservation(role="tool", content={"name": "Raj Sanchez"}, tool_call_id="a"),
            GroundingObservation(role="tool", content={"status": "confirmed"}, tool_call_id="b"),
        ),
    )
    # Retrieval only ever found get_user_details evidence -- never anything
    # about get_reservation_details.
    examples = (
        _example(
            "e1", "get_user_details", {"user_id": "raj_sanchez_7340"}, {"name": "Raj Sanchez"}
        ),
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=examples)
    by_call = {c.call_id: c for c in assessment.calls}

    assert by_call["a"].behavior_supported is True
    assert by_call["b"].behavior_supported is False
    assert by_call["b"].kind is GroundingKind.UNSUPPORTED


def test_conflicting_retrieved_value_does_not_count_as_identifiable() -> None:
    """A retrieved trace's snapshot of this entity can be stale -- state_before
    is this rollout's own history and wins. A field that conflicts between
    the two is neither trustworthy nor identifiable, not silently overwritten."""
    target = _single(
        "target",
        "get_reservation_details",
        {"reservation_id": "Q69X3R"},
        {"status": "cancelled"},
    )
    state = ScenarioState(
        fields=(
            StateField(
                path="get_reservation_details.Q69X3R.status",
                value="cancelled",
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id="s",
            ),
        )
    )
    examples = (
        _example(
            "e1",
            "get_reservation_details",
            {"reservation_id": "Q69X3R"},
            {"status": "confirmed"},
        ),
    )
    assessment = assess_grounding(target, state_before=state, examples=examples)
    call = assessment.single_call

    assert call is not None
    # state_before wins outright here (it already had the required field),
    # so this stays identifiable -- but from state, not from the conflicting
    # retrieved value.
    assert call.values_identifiable is True
    assert call.identifiable_fact_paths == ("status",)


def test_disagreeing_retrieved_examples_are_not_identifiable() -> None:
    """Two retrieved snapshots of the same entity disagree on `status` --
    neither state_before nor a single retrieved value resolves it, so the
    field must be dropped rather than letting whichever example was seen
    last silently win."""
    target = _single(
        "target",
        "get_reservation_details",
        {"reservation_id": "Q69X3R"},
        {"status": "cancelled"},
    )
    examples = (
        _example(
            "e1",
            "get_reservation_details",
            {"reservation_id": "Q69X3R"},
            {"status": "confirmed"},
        ),
        _example(
            "e2",
            "get_reservation_details",
            {"reservation_id": "Q69X3R"},
            {"status": "cancelled"},
        ),
    )
    assessment = assess_grounding(target, state_before=ScenarioState(), examples=examples)
    call = assessment.single_call

    assert call is not None
    assert call.values_identifiable is False
    assert call.missing_fact_paths == ("status",)


def test_reordered_observations_are_correlated_by_tool_call_id_not_position() -> None:
    """The result for call "b" is listed first; correlation must still pair
    each observation with its own call, not with whichever call comes first."""
    target = _transition(
        "target",
        (
            ActionCall(call_id="a", tool="get_user_details", arguments={"user_id": "raj_sanchez_7340"}),
            ActionCall(call_id="b", tool="get_user_details", arguments={"user_id": "emma_kim_9957"}),
        ),
        (
            GroundingObservation(role="tool", content={"name": "Emma Kim"}, tool_call_id="b"),
            GroundingObservation(role="tool", content={"name": "Raj Sanchez"}, tool_call_id="a"),
        ),
    )
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
    assessment = assess_grounding(target, state_before=state, examples=())
    by_call = {c.call_id: c for c in assessment.calls}

    assert by_call["a"].required_fact_paths == ("name",)
    assert by_call["a"].values_identifiable is True
    assert by_call["b"].values_identifiable is False
