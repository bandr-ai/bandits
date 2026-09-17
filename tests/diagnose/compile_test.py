"""Compilation, exercised on trace shapes the real corpus actually produces.

Fixtures mirror tau2: a system policy, user turns placed by ``after_span_id``,
model spans whose arguments carry ``tool_calls``, and tool results arriving as
JSON text. A fixture that simplified any of those would test a corpus we do not
have.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from bandits.analyze.models import TaskFamily
from bandits.diagnose.compile import (
    build_prefix,
    compile_scenarios,
    cut_points,
    extract_transitions,
    reconstruct_state,
    strip_markers,
)
from bandits.diagnose.models import (
    DeltaGroundTruthStatus,
    ExpectedEffect,
    HiddenUserProfile,
    Partition,
    ScenarioKind,
    SealedSuccessContract,
    SuccessShape,
    WorldOrigin,
)
from bandits.traces import Span, SpanKind, SpanStatus, ToolSchema, Trace, UserTurn

_T0 = datetime(2024, 5, 15, 15, 0, tzinfo=UTC)


def _model(span_id: str, tool: str | None = None, arguments: dict | None = None, text: str = ""):
    """A model span the way the chat-JSON adapter actually writes one.

    Verified against ``corpus-ee3b33086ef177d7``: a tool call is a MODEL span
    whose ``name`` is the tool and whose ``arguments`` are the call's arguments
    flat, with no output; speech is ``name="assistant"`` with the text as
    output. An earlier version of this fixture invented a nested ``tool_calls``
    payload, and every assertion passed while the module produced zero
    tool-typed transitions on the real corpus.
    """
    return Span(
        span_id=span_id,
        kind=SpanKind.MODEL,
        name=tool or "assistant",
        started_at=_T0,
        ended_at=_T0,
        arguments=dict(arguments or {}),
        output=None if tool else text,
    )


def _model_nested(span_id: str, tool: str, arguments: dict | None = None, text: str = ""):
    """The wire-format shape other adapters produce: the call nested inside."""
    return Span(
        span_id=span_id,
        kind=SpanKind.MODEL,
        name="claude-sonnet-4-5",
        started_at=_T0,
        ended_at=_T0,
        arguments={
            "content": text,
            "tool_calls": [
                {
                    "id": f"call-{span_id}",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments or {})},
                }
            ],
        },
        output=text,
    )


def _tool(span_id: str, name: str, output, error: bool = False):
    return Span(
        span_id=span_id,
        kind=SpanKind.TOOL,
        name=name,
        started_at=_T0,
        ended_at=_T0,
        status=SpanStatus.ERROR if error else SpanStatus.OK,
        output=output if isinstance(output, str) else json.dumps(output),
    )


def _trace(**updates) -> Trace:
    base = dict(
        trace_id="airline-10",
        source="chat-json",
        source_digest="d" * 64,
        task="cancel my reservation 3RK2T9",
        lineage_id="airline-7",
        system_prompt="# Airline Agent Policy\nYou must obtain explicit confirmation.",
        tools_available=(
            ToolSchema(
                name="cancel_reservation",
                parameters={"type": "object", "properties": {"reservation_id": {"type": "string"}}},
            ),
        ),
        user_turns=(
            UserTurn(text="Hi, I'm Anya Garcia.", after_span_id=None),
            UserTurn(text="My id is anya_garcia_5901.###TRANSFER###", after_span_id="s1"),
        ),
        spans=(
            _model("s1", text="Hi! How can I help?"),
            _model("s2", tool="get_reservation_details", arguments={"reservation_id": "3RK2T9"}),
            _tool(
                "s3",
                "get_reservation_details",
                {"reservation_id": "3RK2T9", "status": "confirmed", "cabin": "basic_economy"},
            ),
            _model("s4", tool="cancel_reservation", arguments={"reservation_id": "3RK2T9"}),
            _tool("s5", "cancel_reservation", {"reservation_id": "3RK2T9", "status": "cancelled"}),
            _model("s6", text="Your reservation is cancelled."),
        ),
    )
    return Trace(**{**base, **updates})


def _family(**updates) -> TaskFamily:
    base = dict(
        family_id="family-451ae91f975c",
        descriptor="cancellation and refund",
        trace_ids=("airline-10", "airline-11"),
        medoid_trace_id="airline-10",
        workload_mass=2,
        fit_trace_ids=("airline-10",),
        held_out_trace_ids=("airline-11",),
    )
    return TaskFamily(**{**base, **updates})


def _contract() -> SealedSuccessContract:
    return SealedSuccessContract(
        contract_id="contract-7",
        source_task_id="7",
        shape=SuccessShape.MUTATION,
        required_effects=(
            ExpectedEffect(
                effect_id="e1",
                tool="cancel_reservation",
                arguments={"reservation_id": "3RK2T9"},
            ),
        ),
    )


# --- markers ------------------------------------------------------------


def test_markers_are_stripped_and_recorded_not_dropped() -> None:
    """tau2 appends ###TRANSFER### to most airline user turns, not only escalations."""
    text, found = strip_markers("I need help.###TRANSFER###", ("###TRANSFER###",))
    assert text == "I need help."
    assert found == ("###TRANSFER###",)


def test_absent_marker_records_nothing() -> None:
    text, found = strip_markers("I need help.", ("###TRANSFER###",))
    assert text == "I need help."
    assert found == ()


def test_user_prefix_text_has_markers_removed() -> None:
    prefix = build_prefix(_trace(), "s3", markers=("###TRANSFER###",))
    users = [step for step in prefix if step.role == "user"]
    assert any("anya_garcia_5901" in str(step.content) for step in users)
    assert not any("###TRANSFER###" in str(step.content) for step in users)


# --- prefix boundaries --------------------------------------------------


def test_prefix_stops_at_the_cut_and_shows_no_future_action() -> None:
    """The entire basis of a prefix rollout: no future historical action is visible."""
    prefix = build_prefix(_trace(), "s3")
    span_ids = [step.span_id for step in prefix if step.span_id]
    assert "s3" in span_ids
    assert "s4" not in span_ids  # the decisive cancel the candidate must now choose
    assert "s5" not in span_ids


def test_prefix_opens_with_the_system_policy_and_first_user_turn() -> None:
    prefix = build_prefix(_trace(), "s3")
    assert prefix[0].role == "system"
    assert "Airline Agent Policy" in str(prefix[0].content)
    assert prefix[1].role == "user"


def test_prefix_places_user_turns_after_the_span_they_followed() -> None:
    prefix = build_prefix(_trace(), "s3")
    roles = [(step.role, step.span_id) for step in prefix]
    assert ("user", "s1") in roles


def test_assistant_step_carries_structured_tool_call() -> None:
    prefix = build_prefix(_trace(), "s3")
    call = next(s for s in prefix if s.role == "assistant" and s.tool_name)
    assert call.tool_name == "get_reservation_details"
    assert call.arguments == {"reservation_id": "3RK2T9"}


def test_both_recorded_call_shapes_are_read() -> None:
    """Reading only the nested shape produced zero tool transitions on tau2."""
    flat = _trace(
        spans=(
            _model("s1", tool="cancel_reservation", arguments={"reservation_id": "A"}),
            _tool("s2", "cancel_reservation", {"status": "cancelled"}),
        )
    )
    nested = _trace(
        spans=(
            _model_nested("s1", "cancel_reservation", {"reservation_id": "A"}),
            _tool("s2", "cancel_reservation", {"status": "cancelled"}),
        )
    )
    for trace in (flat, nested):
        transition = extract_transitions(trace, family_id="f")[0]
        assert transition.action_calls[0].tool == "cancel_reservation"
        assert transition.action_calls[0].arguments == {"reservation_id": "A"}


def test_speech_is_not_read_as_a_tool_named_assistant() -> None:
    trace = _trace(spans=(_model("s1", text="Hi! How can I help?"),))
    assert extract_transitions(trace, family_id="f")[0].action_calls == ()


def test_consecutive_tool_calls_are_one_batched_action() -> None:
    """The adapter splits a multi-call assistant message into several spans."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "A"}),
            _model("s2", tool="get_reservation_details", arguments={"reservation_id": "B"}),
            _tool("s3", "get_reservation_details", {"status": "confirmed"}),
            _tool("s4", "get_reservation_details", {"status": "cancelled"}),
        )
    )
    transitions = extract_transitions(trace, family_id="f")
    assert len(transitions) == 1
    assert len(transitions[0].action_calls) == 2
    assert len(transitions[0].observations) == 2
    assert transitions[0].action_span_ids == ("s1", "s2")


# --- state reconstruction -----------------------------------------------


def test_state_is_namespaced_by_the_entity_the_call_named() -> None:
    """Two reservations both reporting `status` must not collapse onto one path."""
    state = reconstruct_state(_trace(), "s3")
    paths = {f.path for f in state.fields}
    assert "get_reservation_details.3RK2T9.status" in paths
    assert state.get("get_reservation_details.3RK2T9.status").value == "confirmed"


def test_reconstructed_state_is_entirely_recorded() -> None:
    state = reconstruct_state(_trace(), "s5")
    assert state.fields
    assert all(f.origin is WorldOrigin.RECORDED for f in state.fields)
    assert all(f.revealed_by_span_id for f in state.fields)
    assert state.simulated_paths == ()


def test_unrevealed_path_is_unknown_not_absent() -> None:
    state = reconstruct_state(_trace(), "s3")
    assert not state.known("get_reservation_details.3RK2T9.refund_amount")
    assert state.get("get_reservation_details.3RK2T9.refund_amount") is None


def test_state_stops_at_the_cut_point() -> None:
    """A middle-prefix scenario must not know the outcome of the action it asks for."""
    early = reconstruct_state(_trace(), "s3")
    late = reconstruct_state(_trace(), "s5")
    assert not early.known("cancel_reservation.3RK2T9.status")
    assert late.get("cancel_reservation.3RK2T9.status").value == "cancelled"


def test_errored_result_contributes_no_state() -> None:
    """An error says what did not happen; reading it as state would invent it."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_user_details", arguments={"user_id": "emma_kim"}),
            _tool("s2", "get_user_details", "Error: User emma_kim not found", error=True),
        )
    )
    assert reconstruct_state(trace, "s2").fields == ()


# --- inferred_state_delta -----------------------------------------------


def test_cross_tool_delta_is_unavailable_not_a_silent_empty_dict() -> None:
    """The real-corpus case: cancel_reservation's fields are tool-prefixed
    differently from the get_reservation_details fields that populated
    state_before, so no reported field can be aligned even though the
    cancellation obviously changed status. UNAVAILABLE must be reported
    explicitly rather than an indistinguishable empty {}."""
    transitions = extract_transitions(_trace(), family_id="family-451ae91f975c")
    cancel = next(t for t in transitions if t.action_tool == "cancel_reservation")
    assert cancel.inferred_state_delta == {}
    assert cancel.delta_ground_truth_status is DeltaGroundTruthStatus.UNAVAILABLE
    assert "cancel_reservation.3RK2T9.status" in cancel.unmatched_post_paths


def test_aligned_path_produces_a_measured_delta() -> None:
    """When a later call reports a result under the SAME path family an
    earlier call already populated, the value can genuinely be compared."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "3RK2T9"}),
            _tool(
                "s2",
                "get_reservation_details",
                {"reservation_id": "3RK2T9", "status": "confirmed"},
            ),
            _model("s3", tool="get_reservation_details", arguments={"reservation_id": "3RK2T9"}),
            _tool(
                "s4",
                "get_reservation_details",
                {"reservation_id": "3RK2T9", "status": "cancelled"},
            ),
        )
    )
    transitions = extract_transitions(trace, family_id="family-451ae91f975c")
    second_lookup = transitions[1]
    assert second_lookup.delta_ground_truth_status is DeltaGroundTruthStatus.MEASURED
    assert second_lookup.inferred_state_delta == {
        "get_reservation_details.3RK2T9.status": "cancelled"
    }
    assert second_lookup.unmatched_post_paths == ()


def test_partial_alignment_is_conservatively_unavailable_not_measured() -> None:
    """A repeated call that reports one aligned field (genuinely changed) AND
    one brand-new field the earlier call never reported: the whole delta must
    be UNAVAILABLE, not a partial MEASURED delta that silently omits the
    unaligned field and overstates completeness."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "3RK2T9"}),
            _tool(
                "s2",
                "get_reservation_details",
                {"reservation_id": "3RK2T9", "status": "confirmed"},
            ),
            _model("s3", tool="get_reservation_details", arguments={"reservation_id": "3RK2T9"}),
            _tool(
                "s4",
                "get_reservation_details",
                {
                    "reservation_id": "3RK2T9",
                    "status": "cancelled",
                    "refund_amount": 430,
                },
            ),
        )
    )
    transitions = extract_transitions(trace, family_id="family-451ae91f975c")
    second_lookup = transitions[1]
    assert second_lookup.delta_ground_truth_status is DeltaGroundTruthStatus.UNAVAILABLE
    assert second_lookup.inferred_state_delta == {}
    assert "get_reservation_details.3RK2T9.refund_amount" in second_lookup.unmatched_post_paths


def test_uncorrelatable_reaction_is_unavailable_not_not_applicable() -> None:
    """A batch calling the same tool twice with no tool_call_id on either
    side: real mutation evidence exists in both reactions, but neither can be
    attributed to a specific call, so correlation fails for both. This must
    read as UNAVAILABLE (evidence existed, couldn't be used) -- returning
    NOT_APPLICABLE here would be indistinguishable from a transition that
    reported nothing comparable at all."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "A"}),
            _model("s2", tool="get_reservation_details", arguments={"reservation_id": "B"}),
            _tool("s3", "get_reservation_details", {"status": "confirmed"}),
            _tool("s4", "get_reservation_details", {"status": "cancelled"}),
        )
    )
    transitions = extract_transitions(trace, family_id="f")
    assert transitions[0].delta_ground_truth_status is DeltaGroundTruthStatus.UNAVAILABLE
    assert transitions[0].inferred_state_delta == {}


def test_no_reported_fields_is_not_applicable() -> None:
    """A read whose only prior result has nothing comparable carries no
    ground-truth claim at all -- distinct from a genuinely-verified no-op."""
    trace = _trace(
        spans=(
            _model("s1", tool="transfer_to_human_agents", arguments={}),
            _tool("s2", "transfer_to_human_agents", {}),
        )
    )
    transitions = extract_transitions(trace, family_id="family-451ae91f975c")
    assert transitions[0].delta_ground_truth_status is DeltaGroundTruthStatus.NOT_APPLICABLE
    assert transitions[0].inferred_state_delta == {}
    assert transitions[0].unmatched_post_paths == ()


# --- transitions --------------------------------------------------------


def test_transitions_keep_structured_observations() -> None:
    """A fidelity diff needs fields; PR #44's Turn would have a clipped string."""
    transitions = extract_transitions(_trace(), family_id="family-451ae91f975c")
    cancel = next(t for t in transitions if t.action_tool == "cancel_reservation")
    assert cancel.observations[0].content["status"] == "cancelled"
    assert cancel.action_calls[0].arguments == {"reservation_id": "3RK2T9"}
    assert cancel.reaction_role == "tool"
    assert cancel.observed


def test_final_action_with_no_reaction_is_unobserved() -> None:
    """Silence is not approval, and cannot teach a next-observation predictor."""
    transitions = extract_transitions(_trace(), family_id="family-451ae91f975c")
    last = transitions[-1]
    assert not last.observed
    assert last.observations == ()
    assert last.reaction_role == "none"


def test_user_reply_is_a_transition_with_the_user_role() -> None:
    transitions = extract_transitions(
        _trace(), family_id="family-451ae91f975c", markers=("###TRANSFER###",)
    )
    reply = next(t for t in transitions if t.reaction_role == "user")
    text = str(reply.observations[0].content)
    assert "anya_garcia_5901" in text
    assert "###TRANSFER###" not in text
    assert reply.stripped_markers == ("###TRANSFER###",)


def test_transition_history_is_not_clipped() -> None:
    """Clipping is a rendering concern; a truncated index teaches truncated output."""
    long_text = "x" * 5000
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "A"}),
            _tool("s2", "get_reservation_details", {"note": long_text}),
            _model("s3", tool="cancel_reservation", arguments={"reservation_id": "A"}),
            _tool("s4", "cancel_reservation", {"status": "cancelled"}),
        )
    )
    transitions = extract_transitions(trace, family_id="f")
    cancel = next(t for t in transitions if t.action_tool == "cancel_reservation")
    rendered = json.dumps([s.model_dump(mode="json") for s in cancel.history_before])
    assert long_text in rendered


def test_transition_state_before_excludes_its_own_outcome() -> None:
    """The example must not contain the answer it is evidence for."""
    transitions = extract_transitions(_trace(), family_id="f")
    cancel = next(t for t in transitions if t.action_tool == "cancel_reservation")
    assert not cancel.state_before.known("cancel_reservation.3RK2T9.status")
    assert cancel.state_before.known("get_reservation_details.3RK2T9.status")


def test_transitions_carry_lineage_for_retrieval_filtering() -> None:
    transitions = extract_transitions(_trace(), family_id="family-451ae91f975c")
    assert all(t.lineage_id == "airline-7" for t in transitions)
    assert all(t.trace_id == "airline-10" for t in transitions)


def test_transition_ids_are_deterministic() -> None:
    first = extract_transitions(_trace(), family_id="f")
    second = extract_transitions(_trace(), family_id="f")
    assert [t.transition_id for t in first] == [t.transition_id for t in second]


# --- cut points and scenarios -------------------------------------------


def test_cut_points_deduplicate_repeated_lookups() -> None:
    """Task 43 calls get_reservation_details six times; that is one decision shape."""
    trace = _trace(
        spans=(
            _model("s1", tool="get_reservation_details", arguments={"reservation_id": "A"}),
            _tool("s2", "get_reservation_details", {"status": "confirmed"}),
            _model("s3", tool="get_reservation_details", arguments={"reservation_id": "B"}),
            _tool("s4", "get_reservation_details", {"status": "confirmed"}),
            _model("s5", tool="cancel_reservation", arguments={"reservation_id": "A"}),
            _tool("s6", "cancel_reservation", {"status": "cancelled"}),
        )
    )
    assert cut_points(trace) == ("s2", "s6")


def test_error_and_success_of_one_tool_are_different_cut_points() -> None:
    trace = _trace(
        spans=(
            _model("s1", tool="get_user_details", arguments={"user_id": "x"}),
            _tool("s2", "get_user_details", "Error: not found", error=True),
            _model("s3", tool="get_user_details", arguments={"user_id": "y"}),
            _tool("s4", "get_user_details", {"name": "Anya"}),
        )
    )
    assert cut_points(trace) == ("s2", "s4")


def test_compile_produces_start_middle_and_end_scenarios() -> None:
    scenarios = compile_scenarios(_trace(), family=_family(), contract=_contract())
    kinds = [s.kind for s in scenarios]
    assert ScenarioKind.TASK_START in kinds
    assert ScenarioKind.END_PREFIX in kinds
    assert all(s.family_id == "family-451ae91f975c" for s in scenarios)


def test_every_scenario_excludes_its_own_lineage_from_retrieval() -> None:
    """A retry of the same request carries the same answer."""
    scenarios = compile_scenarios(
        _trace(),
        family=_family(),
        contract=_contract(),
        lineage_group=("airline-10", "airline-11", "airline-12"),
    )
    for scenario in scenarios:
        assert "airline-10" in scenario.retrieval_excluded_trace_ids
        assert "airline-12" in scenario.retrieval_excluded_trace_ids


def test_task_start_scenario_shows_no_history_and_no_state() -> None:
    scenarios = compile_scenarios(_trace(), family=_family(), contract=_contract())
    start = next(s for s in scenarios if s.kind is ScenarioKind.TASK_START)
    assert start.prefix == ()
    assert start.initial_state.fields == ()
    assert start.candidate_view().prefix == ()


def test_sealed_traces_are_partitioned_apart_from_held_out() -> None:
    """Sealed and held-out are different exclusions with different reasons."""
    scenarios = compile_scenarios(
        _trace(),
        family=_family(),
        contract=_contract(),
        sealed_trace_ids=("airline-10",),
    )
    assert all(s.partition is Partition.SEALED for s in scenarios)


def test_held_out_trace_compiles_as_held_out() -> None:
    trace = _trace(trace_id="airline-11")
    scenarios = compile_scenarios(trace, family=_family(), contract=_contract())
    assert all(s.partition is Partition.HELD_OUT for s in scenarios)


def test_hidden_user_never_reaches_the_candidate_view() -> None:
    """Only what the profile adds is private.

    A fact the real user already said out loud is in the authentic prefix and
    belongs there — the candidate is meant to have heard it. The leak this
    guards is the environment's *private* copy: the goal, the persona, and the
    facts the user is holding back.
    """
    withheld = "the passenger was added after booking"
    scenarios = compile_scenarios(
        _trace(),
        family=_family(),
        contract=_contract(),
        hidden_user=HiddenUserProfile(
            reason_for_call="cancel and get a full refund",
            known_info=withheld,
            unknown_info="whether insurance was purchased",
            task_instructions="do not accept a partial refund",
        ),
    )
    for scenario in scenarios:
        assert scenario.hidden_user.known_info == withheld
        rendered = json.dumps(scenario.candidate_view().model_dump(mode="json"))
        assert withheld not in rendered
        assert "do not accept a partial refund" not in rendered
        assert "whether insurance was purchased" not in rendered


def test_sealed_contract_never_reaches_the_candidate_view() -> None:
    scenarios = compile_scenarios(_trace(), family=_family(), contract=_contract())
    for scenario in scenarios:
        rendered = json.dumps(scenario.candidate_view().model_dump(mode="json"))
        assert "contract-7" not in rendered


def test_middle_prefix_scenarios_are_capped() -> None:
    trace = _trace(
        spans=tuple(
            span
            for index in range(8)
            for span in (
                _model(f"m{index}", tool=f"tool_{index}", arguments={"id": str(index)}),
                _tool(f"r{index}", f"tool_{index}", {"status": "ok"}),
            )
        )
    )
    scenarios = compile_scenarios(trace, family=_family(), contract=_contract(), max_middle=2)
    middles = [s for s in scenarios if s.kind is ScenarioKind.MIDDLE_PREFIX]
    assert len(middles) == 2


def test_scenario_ids_are_deterministic() -> None:
    first = compile_scenarios(_trace(), family=_family(), contract=_contract())
    second = compile_scenarios(_trace(), family=_family(), contract=_contract())
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]


def test_compiled_scenario_state_passes_the_recorded_only_invariant() -> None:
    """compile_scenarios must never produce a scenario models.py would reject."""
    scenarios = compile_scenarios(_trace(), family=_family(), contract=_contract())
    for scenario in scenarios:
        assert all(f.origin is WorldOrigin.RECORDED for f in scenario.initial_state.fields)
