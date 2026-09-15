"""Score a rollout with the verifier semantics history was scored by.

``execute_verifier`` reads ``Evidence`` built by ``analyze/outcomes.py`` from
recorded spans. A rollout has no spans — it has a state ledger, an event log and
a simulated transcript — so nothing joined the two planes, and the capability
number the whole exercise exists to produce was unreachable.

This module is the join, and it deliberately does not re-implement scoring. The
verifier's operator semantics — absence is unknown rather than failure, one
unknown check makes the whole verifier unknown, a bare key two tools both
answered refuses rather than guessing — are the reviewed, tested, human-promoted
part of this system. A second executor would re-derive them and drift, and a
historical score and a rollout score would stop being comparable, which is the
only property that makes the rollout number mean anything.

So a rollout is rendered into the same claims the analyzer emits, and the same
``execute_verifier`` runs. What travels alongside each claim is its
``WorldOrigin``: the verdict is *simulation-conditioned* exactly when some claim
behind it came from the simulator, and that is decided per claim rather than
per campaign.

The sealed contract is checked separately from the deterministic verifier,
because its three components have different authorities and must never be
averaged:

    operational   required/forbidden effects against the committed event ledger
    process       reads the agent was expected to make; carries no success claim
    communication what had to be said; usually a rubric, so MODEL_JUDGMENT
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from bandits.analyze.models import Evidence, EvidenceKind, Visibility, evidence_id
from bandits.diagnose.models import (
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
    SuccessShape,
    ToolEffect,
    ToolEffectCatalog,
    VerifierInputClaim,
    WorldOrigin,
)


def _origin_of(state: ScenarioState, path: str) -> WorldOrigin:
    field = state.get(path)
    return field.origin if field else WorldOrigin.SIMULATED


def rollout_claims(
    *,
    final_state: ScenarioState,
    initial_state: ScenarioState,
    steps: Sequence[RolloutStep],
) -> tuple[VerifierInputClaim, ...]:
    """A rollout rendered as the claims the analyzer emits from a real episode.

    Claim names and value shapes match ``analyze/outcomes.py`` exactly, so the
    existing operators execute unchanged. ``field`` is the tool-qualified name
    the analyzer uses, because a check accepted against that naming must keep
    resolving the same way here — anything else would silently turn a reviewed
    check into an unknown.
    """
    claims: list[VerifierInputClaim] = []

    for field in final_state.fields:
        claims.append(
            VerifierInputClaim(
                claim="final_state_field",
                value={
                    "key": field.path,
                    "field": field.path,
                    "value": field.value,
                    "tool": field.path.split(".", 1)[0],
                },
                origin=ClaimOrigin(
                    world=field.origin,
                    authority=EvidenceKind.TERMINAL_STATE_CHECK,
                    step_index=field.revealed_at_step,
                ),
                detail=field.path,
            )
        )

    for field in initial_state.fields:
        claims.append(
            VerifierInputClaim(
                claim="initial_state_field",
                value={
                    "key": field.path,
                    "field": field.path,
                    "value": field.value,
                    "tool": field.path.split(".", 1)[0],
                },
                origin=ClaimOrigin(world=field.origin, authority=EvidenceKind.OBSERVED_TRACE),
                detail=field.path,
            )
        )

    # ``NO_SPAN_ERROR`` anchors on this: without it the operator cannot tell a
    # clean run from one we hold no evidence for, and absence reads as success.
    claims.append(
        VerifierInputClaim(
            claim="episode_span_count",
            value={"value": len(steps)},
            origin=ClaimOrigin(world=WorldOrigin.SIMULATED),
        )
    )

    for step in steps:
        if step.validator_rejections or (
            isinstance(step.observation, dict) and step.observation.get("status") == "error"
        ):
            claims.append(
                VerifierInputClaim(
                    claim="span_error",
                    value={
                        "tool": step.calls[0].tool if step.calls else None,
                        "step": step.index,
                    },
                    origin=ClaimOrigin(world=step.world, step_index=step.index),
                )
            )
    return tuple(claims)


def claims_to_evidence(
    claims: Iterable[VerifierInputClaim], *, trace_id: str
) -> tuple[Evidence, ...]:
    """Claims as ``Evidence``, so ``execute_verifier`` runs unmodified.

    ``provenance`` is ``derived`` on every row, never ``observed``: nothing here
    was read off a real span, and a simulated observation must never be
    indistinguishable from a recorded one. Origin is preserved separately by the
    caller — ``Evidence`` has nowhere to carry it, which is precisely why
    ``VerifierInputClaim`` exists and why this conversion is the last step
    rather than the representation.
    """
    return tuple(
        Evidence(
            evidence_id=evidence_id(
                trace_id=trace_id, claim=claim.claim, detail=claim.detail or str(index)
            ),
            claim=claim.claim,
            value=claim.value,
            visibility=Visibility.TERMINAL,
            provenance="derived",
            strength="moderate",
            kind=claim.origin.authority,
            trace_id=trace_id,
        )
        for index, claim in enumerate(claims)
    )


def _arguments_agree(expected: dict[str, Any], actual: dict[str, Any], policy: MatchPolicy) -> bool:
    if policy.argument_match == "exact":
        return expected == actual
    # ``subset``/``reviewed``: every argument the contract names must agree, and
    # arguments it does not name are free. A contract that named none would
    # otherwise match any call to the right tool.
    return all(actual.get(key) == value for key, value in expected.items())


def _committed_effects(
    steps: Sequence[RolloutStep],
) -> list[tuple[str, dict[str, Any], int, str | None]]:
    """Writes the rollout actually committed, per call, in order.

    Attempted calls do not count: an action the validator rejected or the
    environment abstained on changed nothing, and scoring it as success rewards
    the request rather than the result.

    Per *call*, not per step. A batch is not all-or-nothing — ``cancel(A)``
    succeeding while ``cancel(B)`` fails in one action must not satisfy two
    required cancellations — so a call counts only when
    ``committed_call_ids`` names it. A step that recorded no per-call identity
    at all falls back to its whole batch, which is what older rollouts mean.
    """
    committed = []
    for step in steps:
        if step.abstained or not step.calls:
            continue
        if not (step.committed_delta or step.events):
            continue
        named = set(step.committed_call_ids)
        for call in step.calls:
            identity = call.call_id
            if named and (identity is None or identity not in named):
                continue
            committed.append((call.tool, dict(call.arguments), step.index, identity))
    return committed


def _executed_calls(steps: Sequence[RolloutStep]) -> list[tuple[str, dict[str, Any], int]]:
    """Calls that reached the tool at all, committed or not.

    A forbidden execution that emitted no delta still happened. Checking only
    committed effects let it disappear, which is precisely the case a refusal
    task exists to catch.
    """
    executed = []
    for step in steps:
        if step.abstained or not step.calls:
            continue
        named = set(step.executed_call_ids)
        for call in step.calls:
            if named and (call.call_id is None or call.call_id not in named):
                continue
            executed.append((call.tool, dict(call.arguments), step.index))
    return executed


def _effect_evidenced(
    effect: ExpectedEffect,
    steps: Sequence[RolloutStep],
    final_state: ScenarioState,
    *,
    matched_call_id: str | None,
) -> bool:
    """Whether the *effect* the contract named actually shows in the ledger.

    Matching a tool name and arguments says the right call was made. It does
    not say the world changed: an earlier version accepted any non-empty delta
    on the step as proof that the required change occurred, so a call that
    returned an unrelated field satisfied the requirement. Where the contract
    names an event type or a terminal state value, that is what must be found.
    """
    if effect.event_type is not None:
        events = [event for step in steps for event in step.events]
        attributed = any("_call_id" in event for event in events)
        if not any(
            event.get("type") == effect.event_type
            and (
                not attributed
                or (
                    matched_call_id is not None
                    and event.get("_call_id") == matched_call_id
                )
            )
            for event in events
        ):
            return False
    if effect.state_path is not None:
        field = final_state.get(effect.state_path)
        if field is None:
            return False
        if effect.expected_value is not None and field.value != effect.expected_value:
            return False
    return True


def match_required_effects(
    required: Sequence[ExpectedEffect],
    steps: Sequence[RolloutStep],
    policy: MatchPolicy,
    *,
    final_state: ScenarioState | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Which required effects were met, and which were not.

    Multiset by default, and each committed effect is consumed at most once —
    so two required cancellations of different reservations need two distinct
    committed events, and cancelling one reservation twice satisfies one
    requirement rather than both.
    """
    available = _committed_effects(steps)
    consumed: set[int] = set()
    met: list[str] = []
    unmet: list[str] = []
    positions: dict[str, int] = {}

    for effect in required:
        found = None
        for position, (tool, arguments, _step_index, _call_id) in enumerate(available):
            if position in consumed:
                continue
            if tool != effect.tool:
                continue
            if not _arguments_agree(effect.arguments, arguments, policy):
                continue
            found = position
            break
        if found is None:
            unmet.append(effect.effect_id)
            continue
        matched_call_id = available[found][3]
        if not _effect_evidenced(
            effect,
            steps,
            final_state or ScenarioState(),
            matched_call_id=matched_call_id,
        ):
            # The right call was made and the named change is not in the ledger.
            unmet.append(effect.effect_id)
            continue
        if policy.multiplicity_required:
            consumed.add(found)
        met.append(effect.effect_id)
        positions[effect.effect_id] = available[found][2]

    if policy.mode == "partial_order" and not unmet:
        for before, after in policy.ordered_pairs:
            ids = [effect.effect_id for effect in required]
            if before < len(ids) and after < len(ids):
                if positions.get(ids[before], -1) > positions.get(ids[after], -1):
                    unmet.append(ids[after])
    return tuple(met), tuple(dict.fromkeys(unmet))


def match_forbidden_effects(
    forbidden: Sequence[ForbiddenEffect],
    steps: Sequence[RolloutStep],
    final_state: ScenarioState,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Forbidden effects that held, that were violated, and that cannot be judged.

    Execution counts, not only commitment: a forbidden call that reached the
    tool and emitted no delta still happened, and a refusal task exists to
    catch exactly that.

    A ``must_remain`` path missing from the final state is **unknown**, not
    held. Nobody observed it, so nothing establishes it was left alone — unless
    the ledger independently shows no call touched that entity, which is
    checked first.
    """
    held: list[str] = []
    violated: list[str] = []
    unknown: list[str] = []
    executed = _executed_calls(steps)

    for effect in forbidden:
        breach = False
        if effect.tool:
            breach = any(
                tool == effect.tool
                and all(arguments.get(k) == v for k, v in effect.arguments.items())
                for tool, arguments, _ in executed
            )
        if not breach and effect.event_type:
            breach = any(
                event.get("type") == effect.event_type for step in steps for event in step.events
            )
        if breach:
            violated.append(effect.effect_id)
            continue

        if effect.state_path and effect.must_remain is not None:
            field = final_state.get(effect.state_path)
            if field is not None:
                (violated if field.value != effect.must_remain else held).append(effect.effect_id)
                continue
            # "Did anything in this rollout plausibly reach that path?" A path
            # is ``tool.entity.field``, so a call only touches it when it named
            # that entity. Matching on the tool alone treated an unrelated
            # lookup as a reason to doubt an untouched reservation.
            segments = effect.state_path.split(".")
            entity = segments[1] if len(segments) > 1 else None
            touched = entity is not None and any(
                any(str(value) == entity for value in arguments.values())
                for _tool, arguments, _ in executed
            )
            if touched:
                unknown.append(effect.effect_id)
            else:
                # Nothing in the ledger reached that entity at all.
                held.append(effect.effect_id)
            continue
        held.append(effect.effect_id)
    return tuple(held), tuple(violated), tuple(unknown)


def score_operational(
    contract: SealedSuccessContract,
    steps: Sequence[RolloutStep],
    final_state: ScenarioState,
    *,
    catalog: ToolEffectCatalog | None = None,
) -> ComponentResult:
    """The operational half: did the required world changes happen, and no forbidden ones.

    An informational contract returns ``NOT_APPLICABLE``, not ``UNKNOWN``:
    nothing was owed, so nothing was measured, and reporting unknown would say
    a measurement was attempted and failed.
    """
    if not contract.claims_operational_success:
        return ComponentResult(
            status=ResultStatus.NOT_APPLICABLE,
            detail="this task requires no state change; reads carry no success claim",
        )

    held, violated, unjudgeable = match_forbidden_effects(
        contract.forbidden_effects, steps, final_state
    )
    if violated:
        return ComponentResult(
            status=ResultStatus.FAIL,
            detail="a forbidden effect occurred",
            authority=EvidenceKind.TERMINAL_STATE_CHECK,
            checked=held,
            unmet=violated,
            simulated_inputs=len(final_state.simulated_paths),
        )
    if unjudgeable:
        return ComponentResult(
            status=ResultStatus.UNKNOWN,
            detail="a forbidden state assertion names a path nothing observed",
            checked=held,
            unmet=unjudgeable,
        )

    if contract.shape is SuccessShape.REFUSAL:
        return ComponentResult(
            status=ResultStatus.PASS,
            detail="no forbidden effect occurred",
            authority=EvidenceKind.TERMINAL_STATE_CHECK,
            checked=held,
            simulated_inputs=len(final_state.simulated_paths),
        )

    if catalog is not None:
        unreviewed = [
            effect.effect_id
            for effect in contract.required_effects
            if catalog.effect_of(effect.tool) is not ToolEffect.WRITE
        ]
        if unreviewed:
            # A required effect on a tool nobody classified as a write cannot
            # establish that the world changed. Unknown, never a pass.
            return ComponentResult(
                status=ResultStatus.UNKNOWN,
                detail="a required effect names a tool not reviewed as a write",
                unmet=tuple(unreviewed),
            )

    met, unmet = match_required_effects(
        contract.required_effects, steps, contract.match_policy, final_state=final_state
    )
    return ComponentResult(
        status=ResultStatus.FAIL if unmet else ResultStatus.PASS,
        detail=f"{len(met)}/{len(contract.required_effects)} required effects committed",
        authority=EvidenceKind.TERMINAL_STATE_CHECK,
        checked=met,
        unmet=unmet,
        simulated_inputs=len(final_state.simulated_paths),
    )


def score_process(
    expectations: Sequence[ProcessExpectation], steps: Sequence[RolloutStep]
) -> ComponentResult:
    """Whether the agent gathered what it was expected to gather.

    Never a success claim. A read-only task whose every lookup was made has not
    thereby accomplished anything — five tasks in the target family are exactly
    this, and counting the lookup as the outcome reports a confident pass for
    doing nothing.
    """
    if not expectations:
        return ComponentResult(status=ResultStatus.NOT_APPLICABLE, detail="no reads were expected")
    attempted = {
        (call.tool, tuple(sorted(call.arguments.items()))) for step in steps for call in step.calls
    }
    tools_called = {call.tool for step in steps for call in step.calls}
    met, unmet = [], []
    for expectation in expectations:
        if expectation.arguments:
            key = (expectation.tool, tuple(sorted(expectation.arguments.items())))
            (met if key in attempted else unmet).append(expectation.expectation_id)
        else:
            (met if expectation.tool in tools_called else unmet).append(expectation.expectation_id)
    return ComponentResult(
        status=ResultStatus.PASS if not unmet else ResultStatus.FAIL,
        detail=f"{len(met)}/{len(expectations)} expected reads made",
        authority=EvidenceKind.OBSERVED_TRACE,
        checked=tuple(met),
        unmet=tuple(unmet),
    )


def score_communication(
    requirements: Sequence[CommunicationRequirement],
    steps: Sequence[RolloutStep],
    *,
    judge: Any = None,
) -> ComponentResult:
    """What had to be said. Literal facts are checkable; assertions need a judge.

    ``communicate_info`` is populated on 6 of 50 tau2 tasks, so most of this
    resolves to a natural-language assertion. Without a judge those are
    ``UNKNOWN`` — which fails the overall verdict closed — rather than being
    waved through, because an unscored requirement is not a met one.
    """
    if not requirements:
        return ComponentResult(
            status=ResultStatus.NOT_APPLICABLE, detail="nothing had to be communicated"
        )

    spoken = "\n".join(str(step.action_content) for step in steps if step.action_content).lower()
    met, unmet, unscored = [], [], []

    for requirement in requirements:
        if requirement.kind == "communicate_info":
            missing = [v for v in requirement.expected_values if v.lower() not in spoken]
            (unmet if missing else met).append(requirement.requirement_id)
        elif judge is None:
            unscored.append(requirement.requirement_id)
        else:
            verdict = judge(assertion=requirement.assertion, transcript=spoken)
            (met if verdict else unmet).append(requirement.requirement_id)

    if unmet:
        return ComponentResult(
            status=ResultStatus.FAIL,
            detail="a required statement was not made",
            authority=EvidenceKind.MODEL_JUDGMENT if judge else EvidenceKind.OBSERVED_TRACE,
            checked=tuple(met),
            unmet=tuple(unmet),
        )
    if unscored:
        return ComponentResult(
            status=ResultStatus.UNKNOWN,
            detail="no judge was supplied for a natural-language assertion",
            checked=tuple(met),
            unmet=tuple(unscored),
        )
    return ComponentResult(
        status=ResultStatus.PASS,
        detail=f"{len(met)} communication requirements met",
        authority=EvidenceKind.MODEL_JUDGMENT if judge else EvidenceKind.OBSERVED_TRACE,
        checked=tuple(met),
    )


def compose_overall(
    operational: ComponentResult,
    communication: ComponentResult,
    *,
    reward_basis: Sequence[str] = (),
) -> ResultStatus:
    """Fail closed, never average.

    Mirrors ``execute_verifier``: any failing required component fails the
    whole verdict, any unknown one makes it unknown, and a pass requires that
    something was actually scored. ``reward_basis`` names the components the
    source itself considered required, so a half it declared cannot be quietly
    dropped by a binding that failed to score it.
    """
    required = [operational, communication]
    if "DB" in reward_basis and operational.status is ResultStatus.NOT_APPLICABLE:
        # The source says the database half carries this task, so a binding
        # that produced no operational claim has not scored it.
        return ResultStatus.UNKNOWN
    if any(component.status is ResultStatus.FAIL for component in required):
        return ResultStatus.FAIL
    if any(component.status is ResultStatus.UNKNOWN for component in required):
        return ResultStatus.UNKNOWN
    if all(component.status is ResultStatus.NOT_APPLICABLE for component in required):
        return ResultStatus.UNKNOWN
    return ResultStatus.PASS


def is_simulation_conditioned(claims: Iterable[VerifierInputClaim]) -> bool:
    """Whether any claim behind a verdict came from the simulator.

    Per verdict, not per campaign: an end-prefix rollout the candidate finished
    without needing a simulated transition rests entirely on recorded state, and
    saying otherwise would understate the one result that is strongest.
    """
    return any(claim.origin.world is WorldOrigin.SIMULATED for claim in claims)
