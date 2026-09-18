"""Rollout scoring, and the join to the existing verifier plane.

The load-bearing test in this file is
``test_rollout_claims_render_the_evidence_a_verifier_reads``: it renders a
rollout's claims as ``Evidence`` rows and pins the claim names and value
shapes. If those drift from ``analyze/outcomes.py``, the capability number
stops being comparable to a historical one.

The evidence-based ``execute_verifier`` this file once drove is gone (its
Loop A lifecycle was removed on main; see ``docs/next-state-verifier.md``).
Task success is now owned by ``SealedSuccessContract`` here, and a separate
failure guard applies the RLM verifier's accepted ``FamilyCheck`` predicates
to simulated turns. Those checks are deterministic compiled code, so they
travel to a rollout; the turn judge's verdicts do not, because they are keyed
to recorded trace turns.
"""

from __future__ import annotations

from bandits.analyze.models import EvidenceKind, Visibility
from bandits.diagnose.models import (
    ActionCall,
    ClaimOrigin,
    CommunicationRequirement,
    ComponentResult,
    ExpectedEffect,
    ForbiddenEffect,
    MatchPolicy,
    ProcessExpectation,
    ResultStatus,
    RolloutStep,
    ScenarioState,
    SealedSuccessContract,
    StateField,
    SuccessShape,
    ToolEffect,
    ToolEffectCatalog,
    ToolEffectEntry,
    VerifierInputClaim,
    WorldOrigin,
)
from bandits.diagnose.verify import (
    claims_to_evidence,
    compose_overall,
    is_simulation_conditioned,
    match_forbidden_effects,
    match_required_effects,
    rollout_claims,
    score_communication,
    score_operational,
    score_process,
)


def _step(index, tool=None, arguments=None, delta=None, events=(), content=None, calls=None, **kw):
    if calls is None:
        calls = (ActionCall(tool=tool, arguments=arguments or {}),) if tool else ()
    return RolloutStep(
        index=index,
        calls=tuple(calls),
        action_content=content,
        committed_delta=delta or {},
        events=tuple(events),
        **kw,
    )


def _state(*pairs, origin=WorldOrigin.SIMULATED):
    return ScenarioState(
        fields=tuple(
            StateField(
                path=path,
                value=value,
                origin=origin,
                **(
                    {"revealed_by_span_id": "s1"}
                    if origin is WorldOrigin.RECORDED
                    else {"revealed_at_step": 0}
                ),
            )
            for path, value in pairs
        )
    )


def _catalog():
    return ToolEffectCatalog(
        catalog_id="catalog-1",
        toolset_digest="d",
        entries=(
            ToolEffectEntry(tool="cancel_reservation", effect=ToolEffect.WRITE, reviewed_by="alex"),
            ToolEffectEntry(
                tool="get_reservation_details", effect=ToolEffect.READ, reviewed_by="alex"
            ),
        ),
    )


# --- the join to the existing verifier plane ----------------------------


def test_rollout_claims_render_the_evidence_a_verifier_reads() -> None:
    """The join: one claim vocabulary, so history and rollout numbers compare.

    Pins the claim name and the value as they reach a verifier. The scorer
    downstream has changed once already; what must not drift is the shape
    ``analyze/outcomes.py`` produces for the same fact.
    """
    final = _state(("cancel_reservation.3RK2T9.status", "cancelled"))
    claims = rollout_claims(final_state=final, initial_state=ScenarioState(), steps=())
    evidence = claims_to_evidence(claims, trace_id="rollout-1")

    row = next(
        item
        for item in evidence
        if item.claim == "final_state_field"
        and item.value["field"] == "cancel_reservation.3RK2T9.status"
    )
    assert row.value["value"] == "cancelled"
    # The tool travels beside the field, so a value can be attributed to the
    # call that produced it rather than to whichever tool shares the prefix.
    assert row.value["tool"] == "cancel_reservation"
    # Never "observed": nothing here was read off a real span.
    assert row.provenance == "derived"
    assert row.visibility is Visibility.TERMINAL
    assert row.trace_id == "rollout-1"


def test_a_wrong_terminal_value_travels_as_the_value_it_was() -> None:
    """A failing rollout must reach the verifier as the wrong value, not as a
    missing one -- a scorer cannot tell "wrong" from "unobserved" otherwise."""
    final = _state(("cancel_reservation.3RK2T9.status", "confirmed"))
    evidence = claims_to_evidence(
        rollout_claims(final_state=final, initial_state=ScenarioState(), steps=()),
        trace_id="rollout-1",
    )

    row = next(
        item
        for item in evidence
        if item.claim == "final_state_field"
        and item.value["field"] == "cancel_reservation.3RK2T9.status"
    )
    assert row.value["value"] == "confirmed"


def test_a_path_the_rollout_never_observed_emits_no_claim() -> None:
    """Absence stays absence. The rollout emits nothing for an unobserved
    path, which is what lets a scorer report unknown rather than failure --
    the distinction the whole abstention design rests on.
    """
    evidence = claims_to_evidence(
        rollout_claims(final_state=ScenarioState(), initial_state=ScenarioState(), steps=()),
        trace_id="rollout-1",
    )

    assert not [
        item
        for item in evidence
        if item.claim == "final_state_field"
        and item.value["field"] == "cancel_reservation.X.status"
    ]


def test_claims_are_never_recorded_as_observed() -> None:
    """A simulated observation must not be indistinguishable from a real one."""
    final = _state(("a.b.status", "cancelled"))
    evidence = claims_to_evidence(
        rollout_claims(final_state=final, initial_state=ScenarioState(), steps=()),
        trace_id="rollout-1",
    )
    assert all(item.provenance == "derived" for item in evidence)


def test_episode_span_count_is_always_emitted() -> None:
    """NO_SPAN_ERROR anchors on it; without it absence would read as success."""
    claims = rollout_claims(
        final_state=ScenarioState(), initial_state=ScenarioState(), steps=(_step(0),)
    )
    assert any(claim.claim == "episode_span_count" for claim in claims)


def test_rejected_transition_emits_a_span_error_claim() -> None:
    claims = rollout_claims(
        final_state=ScenarioState(),
        initial_state=ScenarioState(),
        steps=(_step(0, tool="cancel_reservation", validator_rejections=("bad delta",)),),
    )
    assert any(claim.claim == "span_error" for claim in claims)


# --- world origin -------------------------------------------------------


def test_origin_travels_per_claim_not_per_campaign() -> None:
    """An end-prefix rollout resting on recorded state is not conditioned."""
    recorded = _state(("a.b.status", "confirmed"), origin=WorldOrigin.RECORDED)
    claims = rollout_claims(final_state=recorded, initial_state=recorded, steps=())
    state_claims = [c for c in claims if c.claim != "episode_span_count"]
    assert not is_simulation_conditioned(state_claims)

    simulated = _state(("a.b.status", "cancelled"))
    assert is_simulation_conditioned(
        rollout_claims(final_state=simulated, initial_state=ScenarioState(), steps=())
    )


def test_authority_is_independent_of_origin() -> None:
    claim = VerifierInputClaim(
        claim="final_state_field",
        value={"field": "a.b", "value": 1},
        origin=ClaimOrigin(
            world=WorldOrigin.SIMULATED, authority=EvidenceKind.TERMINAL_STATE_CHECK
        ),
    )
    assert claim.origin.world is WorldOrigin.SIMULATED
    assert claim.origin.authority is EvidenceKind.TERMINAL_STATE_CHECK


# --- required effects ---------------------------------------------------


def test_attempting_a_write_is_not_committing_one() -> None:
    """An abstained or rejected action changed nothing; rewarding it rewards the ask."""
    required = (ExpectedEffect(effect_id="e1", tool="cancel_reservation"),)
    attempted = (_step(0, tool="cancel_reservation", abstained=True),)
    _, unmet = match_required_effects(required, attempted, MatchPolicy())
    assert unmet == ("e1",)

    committed = (_step(0, tool="cancel_reservation", delta={"status": "cancelled"}),)
    met, unmet = match_required_effects(required, committed, MatchPolicy())
    assert met == ("e1",) and unmet == ()


def test_two_required_cancellations_need_two_distinct_events() -> None:
    """Task 7's shape: cancelling one reservation twice satisfies one requirement."""
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
        ),
        ExpectedEffect(
            effect_id="e2", tool="cancel_reservation", arguments={"reservation_id": "B"}
        ),
    )
    wrong = (
        _step(0, tool="cancel_reservation", arguments={"reservation_id": "A"}, delta={"s": 1}),
        _step(1, tool="cancel_reservation", arguments={"reservation_id": "A"}, delta={"s": 1}),
    )
    met, unmet = match_required_effects(required, wrong, MatchPolicy())
    assert met == ("e1",) and unmet == ("e2",)

    right = (
        _step(0, tool="cancel_reservation", arguments={"reservation_id": "A"}, delta={"s": 1}),
        _step(1, tool="cancel_reservation", arguments={"reservation_id": "B"}, delta={"s": 1}),
    )
    met, unmet = match_required_effects(required, right, MatchPolicy())
    assert set(met) == {"e1", "e2"} and unmet == ()


def test_unnamed_arguments_are_free_under_subset_matching() -> None:
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
        ),
    )
    steps = (
        _step(
            0,
            tool="cancel_reservation",
            arguments={"reservation_id": "A", "reason": "customer request"},
            delta={"s": 1},
        ),
    )
    met, _ = match_required_effects(required, steps, MatchPolicy(argument_match="subset"))
    assert met == ("e1",)
    _, unmet = match_required_effects(required, steps, MatchPolicy(argument_match="exact"))
    assert unmet == ("e1",)


def test_order_is_ignored_unless_the_contract_declares_it() -> None:
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
        ),
        ExpectedEffect(effect_id="e2", tool="book_reservation", arguments={"user_id": "u"}),
    )
    reversed_steps = (
        _step(0, tool="book_reservation", arguments={"user_id": "u"}, delta={"s": 1}),
        _step(1, tool="cancel_reservation", arguments={"reservation_id": "A"}, delta={"s": 1}),
    )
    _, unmet = match_required_effects(required, reversed_steps, MatchPolicy())
    assert unmet == ()

    ordered = MatchPolicy(mode="partial_order", ordered_pairs=((0, 1),))
    _, unmet = match_required_effects(required, reversed_steps, ordered)
    assert unmet == ("e2",)


# --- forbidden effects --------------------------------------------------


def test_forbidden_write_that_never_happened_holds() -> None:
    forbidden = (ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),)
    held, violated, unknown = match_forbidden_effects(
        forbidden, (_step(0, tool="get_user_details"),), ScenarioState()
    )
    assert held == ("f1",) and violated == () and unknown == ()


def test_forbidden_write_that_happened_is_violated() -> None:
    forbidden = (ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),)
    steps = (_step(0, tool="cancel_reservation", delta={"status": "cancelled"}),)
    held, violated, _ = match_forbidden_effects(forbidden, steps, ScenarioState())
    assert violated == ("f1",) and held == ()


def test_a_forbidden_execution_that_committed_nothing_is_still_a_violation() -> None:
    """Checking only committed effects let the call vanish entirely."""
    steps = (_step(0, tool="cancel_reservation"),)  # reached the tool, no delta
    forbidden = (ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),)
    _, violated, _ = match_forbidden_effects(forbidden, steps, ScenarioState())
    assert violated == ("f1",)


def test_an_observed_baseline_makes_a_refusal_checkable() -> None:
    forbidden = (
        ForbiddenEffect(
            effect_id="f1", state_path="cancel_reservation.ABC.status", must_remain="confirmed"
        ),
    )
    unchanged = _state(("cancel_reservation.ABC.status", "confirmed"), origin=WorldOrigin.RECORDED)
    held, _, _ = match_forbidden_effects(forbidden, (), unchanged)
    assert held == ("f1",)

    changed = _state(("cancel_reservation.ABC.status", "cancelled"))
    _, violated, _ = match_forbidden_effects(forbidden, (), changed)
    assert violated == ("f1",)


def test_an_unobserved_baseline_a_call_touched_is_unknown_not_held() -> None:
    """Nobody looked, so nothing establishes the path was left alone."""
    forbidden = (
        ForbiddenEffect(
            effect_id="f1", state_path="cancel_reservation.ABC.status", must_remain="confirmed"
        ),
    )
    touched = (_step(0, tool="cancel_reservation", arguments={"reservation_id": "ABC"}),)
    held, violated, unknown = match_forbidden_effects(forbidden, touched, ScenarioState())
    assert unknown == ("f1",) and held == () and violated == ()


def test_an_untouched_entity_holds_without_an_observed_baseline() -> None:
    forbidden = (
        ForbiddenEffect(
            effect_id="f1", state_path="cancel_reservation.ZZZ.status", must_remain="confirmed"
        ),
    )
    elsewhere = (_step(0, tool="get_user_details", arguments={"user_id": "u"}),)
    held, _, unknown = match_forbidden_effects(forbidden, elsewhere, ScenarioState())
    assert held == ("f1",) and unknown == ()


# --- component scoring --------------------------------------------------


def _contract(**updates) -> SealedSuccessContract:
    base = dict(
        contract_id="c",
        source_task_id="7",
        shape=SuccessShape.MUTATION,
        required_effects=(
            ExpectedEffect(
                effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
            ),
        ),
    )
    return SealedSuccessContract(**{**base, **updates})


def test_informational_task_reports_not_applicable_not_unknown() -> None:
    """Nothing was owed, so nothing was measured — and it must not block a pass."""
    contract = _contract(
        shape=SuccessShape.INFORMATIONAL,
        required_effects=(),
        communication=(CommunicationRequirement(requirement_id="c1", assertion="report status"),),
    )
    result = score_operational(contract, (), ScenarioState())
    assert result.status is ResultStatus.NOT_APPLICABLE


def test_refusal_passes_when_nothing_forbidden_happened() -> None:
    contract = _contract(
        shape=SuccessShape.REFUSAL,
        required_effects=(),
        forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
    )
    result = score_operational(contract, (_step(0, tool="get_user_details"),), ScenarioState())
    assert result.status is ResultStatus.PASS


def test_refusal_fails_when_the_forbidden_write_happened() -> None:
    contract = _contract(
        shape=SuccessShape.REFUSAL,
        required_effects=(),
        forbidden_effects=(ForbiddenEffect(effect_id="f1", tool="cancel_reservation"),),
    )
    steps = (_step(0, tool="cancel_reservation", delta={"status": "cancelled"}),)
    assert score_operational(contract, steps, ScenarioState()).status is ResultStatus.FAIL


def test_unreviewed_write_tool_makes_the_operational_half_unknown() -> None:
    """A required effect on an unclassified tool cannot establish a world change."""
    contract = _contract(
        required_effects=(ExpectedEffect(effect_id="e1", tool="mystery_tool"),),
    )
    steps = (_step(0, tool="mystery_tool", delta={"x": 1}),)
    result = score_operational(contract, steps, ScenarioState(), catalog=_catalog())
    assert result.status is ResultStatus.UNKNOWN


def test_reads_never_carry_operational_success() -> None:
    """The five read-only tasks: making the lookup is not accomplishing the task."""
    contract = _contract(
        required_effects=(ExpectedEffect(effect_id="e1", tool="get_reservation_details"),),
    )
    steps = (_step(0, tool="get_reservation_details", delta={"status": "confirmed"}),)
    result = score_operational(contract, steps, ScenarioState(), catalog=_catalog())
    assert result.status is ResultStatus.UNKNOWN


def test_process_result_carries_no_success_claim() -> None:
    expectations = (ProcessExpectation(expectation_id="p1", tool="get_reservation_details"),)
    made = score_process(expectations, (_step(0, tool="get_reservation_details"),))
    assert made.status is ResultStatus.PASS
    assert made.authority is EvidenceKind.OBSERVED_TRACE
    assert score_process((), ()).status is ResultStatus.NOT_APPLICABLE


def test_communicate_info_is_checked_literally() -> None:
    requirement = CommunicationRequirement(
        requirement_id="c1",
        assertion="state the refund",
        kind="communicate_info",
        expected_values=("$57",),
    )
    said = (_step(0, content="Your refund of $57 will arrive in 5 days."),)
    assert score_communication((requirement,), said).status is ResultStatus.PASS
    silent = (_step(0, content="All set."),)
    assert score_communication((requirement,), silent).status is ResultStatus.FAIL


def test_nl_assertion_without_a_judge_is_unknown_not_waved_through() -> None:
    requirement = CommunicationRequirement(requirement_id="c1", assertion="agent should refuse")
    result = score_communication((requirement,), (_step(0, content="I cannot do that."),))
    assert result.status is ResultStatus.UNKNOWN


def test_nl_assertion_with_a_judge_is_scored_as_model_judgment() -> None:
    requirement = CommunicationRequirement(requirement_id="c1", assertion="agent should refuse")
    result = score_communication(
        (requirement,),
        (_step(0, content="I cannot do that."),),
        judge=lambda **_: True,
    )
    assert result.status is ResultStatus.PASS
    assert result.authority is EvidenceKind.MODEL_JUDGMENT


# --- composition --------------------------------------------------------


def test_overall_fails_closed_on_unknown() -> None:
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.PASS),
            ComponentResult(status=ResultStatus.UNKNOWN),
        )
        is ResultStatus.UNKNOWN
    )


def test_overall_fails_on_any_failed_component() -> None:
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.FAIL),
            ComponentResult(status=ResultStatus.PASS),
        )
        is ResultStatus.FAIL
    )


def test_components_are_never_averaged() -> None:
    """One pass and one fail is a fail, not a half."""
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.PASS),
            ComponentResult(status=ResultStatus.FAIL),
        )
        is ResultStatus.FAIL
    )


def test_a_declared_db_basis_cannot_be_silently_dropped() -> None:
    """reward_basis says DB carries this task; producing no DB claim is unknown."""
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.NOT_APPLICABLE),
            ComponentResult(status=ResultStatus.PASS),
            reward_basis=("DB", "COMMUNICATE"),
        )
        is ResultStatus.UNKNOWN
    )


def test_a_declared_communication_basis_cannot_be_silently_dropped() -> None:
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.PASS),
            ComponentResult(status=ResultStatus.NOT_APPLICABLE),
            reward_basis=("DB", "COMMUNICATE"),
        )
        is ResultStatus.UNKNOWN
    )


def test_nothing_scored_is_unknown_not_a_pass() -> None:
    assert (
        compose_overall(
            ComponentResult(status=ResultStatus.NOT_APPLICABLE),
            ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        )
        is ResultStatus.UNKNOWN
    )


def test_a_batched_action_commits_every_call_it_made() -> None:
    """Reading only the first call reports the rest as unmet."""
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
        ),
        ExpectedEffect(
            effect_id="e2", tool="cancel_reservation", arguments={"reservation_id": "B"}
        ),
    )
    batched = (
        _step(
            0,
            calls=(
                ActionCall(tool="cancel_reservation", arguments={"reservation_id": "A"}),
                ActionCall(tool="cancel_reservation", arguments={"reservation_id": "B"}),
            ),
            delta={"s": 1},
        ),
    )
    met, unmet = match_required_effects(required, batched, MatchPolicy())
    assert set(met) == {"e1", "e2"} and unmet == ()


def test_a_batch_has_no_single_action_tool() -> None:
    step = _step(
        0,
        calls=(
            ActionCall(tool="cancel_reservation", arguments={"reservation_id": "A"}),
            ActionCall(tool="get_user_details", arguments={"user_id": "u"}),
        ),
    )
    assert step.action_tool is None
    assert step.action_arguments == {}


# --- effects must actually show in the ledger ---------------------------


def test_the_right_call_with_the_wrong_outcome_is_not_a_met_effect() -> None:
    """Matching a tool name says the call was made, not that the world changed."""
    required = (
        ExpectedEffect(
            effect_id="e1",
            tool="cancel_reservation",
            arguments={"reservation_id": "A"},
            state_path="cancel_reservation.A.status",
            expected_value="cancelled",
        ),
    )
    # A delta was committed, but not the one the contract named.
    steps = (
        _step(0, tool="cancel_reservation", arguments={"reservation_id": "A"}, delta={"note": 1}),
    )
    _, unmet = match_required_effects(
        required,
        steps,
        MatchPolicy(),
        final_state=_state(("cancel_reservation.A.status", "confirmed")),
    )
    assert unmet == ("e1",)

    met, unmet = match_required_effects(
        required,
        steps,
        MatchPolicy(),
        final_state=_state(("cancel_reservation.A.status", "cancelled")),
    )
    assert met == ("e1",) and unmet == ()


def test_a_required_event_type_must_appear_in_the_ledger() -> None:
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", event_type="reservation_cancelled"
        ),
    )
    without = (_step(0, tool="cancel_reservation", delta={"x": 1}),)
    _, unmet = match_required_effects(required, without, MatchPolicy())
    assert unmet == ("e1",)

    with_event = (
        _step(
            0,
            tool="cancel_reservation",
            delta={"x": 1},
            events=({"type": "reservation_cancelled"},),
        ),
    )
    met, _ = match_required_effects(required, with_event, MatchPolicy())
    assert met == ("e1",)


def test_a_required_event_must_come_from_the_call_that_matched_arguments() -> None:
    required = (
        ExpectedEffect(
            effect_id="e1",
            tool="cancel_reservation",
            arguments={"reservation_id": "A"},
            event_type="reservation_cancelled",
        ),
    )
    steps = (
        RolloutStep(
            index=0,
            calls=(
                ActionCall(
                    call_id="call-a",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "A"},
                ),
                ActionCall(
                    call_id="call-b",
                    tool="cancel_reservation",
                    arguments={"reservation_id": "B"},
                ),
            ),
            committed_delta={"reservation": "cancelled"},
            committed_call_ids=("call-a", "call-b"),
            events=({"type": "reservation_cancelled", "_call_id": "call-b"},),
        ),
    )
    _, unmet = match_required_effects(required, steps, MatchPolicy())
    assert unmet == ("e1",)


def test_a_legacy_unidentified_call_cannot_consume_an_attributed_event() -> None:
    required = (
        ExpectedEffect(
            effect_id="e1",
            tool="cancel_reservation",
            arguments={"reservation_id": "A"},
            event_type="reservation_cancelled",
        ),
    )
    steps = (
        RolloutStep(
            index=0,
            calls=(
                ActionCall(
                    tool="cancel_reservation",
                    arguments={"reservation_id": "A"},
                ),
            ),
            committed_delta={"reservation": "cancelled"},
            events=({"type": "reservation_cancelled", "_call_id": "some-other-call"},),
        ),
    )
    _, unmet = match_required_effects(required, steps, MatchPolicy())
    assert unmet == ("e1",)


def test_a_partial_batch_satisfies_only_the_calls_that_committed() -> None:
    """cancel(A) succeeding and cancel(B) failing is not two cancellations."""
    required = (
        ExpectedEffect(
            effect_id="e1", tool="cancel_reservation", arguments={"reservation_id": "A"}
        ),
        ExpectedEffect(
            effect_id="e2", tool="cancel_reservation", arguments={"reservation_id": "B"}
        ),
    )
    partial = (
        RolloutStep(
            index=0,
            calls=(
                ActionCall(
                    call_id="c1", tool="cancel_reservation", arguments={"reservation_id": "A"}
                ),
                ActionCall(
                    call_id="c2", tool="cancel_reservation", arguments={"reservation_id": "B"}
                ),
            ),
            committed_delta={"cancel_reservation.A.status": "cancelled"},
            committed_call_ids=("c1",),
            executed_call_ids=("c1", "c2"),
        ),
    )
    met, unmet = match_required_effects(required, partial, MatchPolicy())
    assert met == ("e1",) and unmet == ("e2",)
