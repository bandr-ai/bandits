"""Run one candidate against one scenario, and persist what happened.

The loop is small on purpose. Everything that decides whether a number is
trustworthy lives in the pieces it calls: retrieval holds the leakage boundary,
the validator owns the ledger, the verifier scores the outcome, and this module
only sequences them and enforces the budget.

    reset(scenario) -> the candidate's view
    step(action)    -> tool world or user policy, validated, committed

Two rules here are load-bearing and both concern what a rollout may claim.

**Authentic future is never replayed.** Once the candidate diverges, the
recorded next user message was a reply to a different agent action and is not a
valid reply to this one. Replaying it would score the candidate against a
conversation it did not have.

**An abstention is not a failure.** The environment declining to answer says
nothing about the candidate, so such a rollout leaves the denominator entirely
rather than being counted as a loss. Budget exhaustion is the opposite: the
environment answered every time and the candidate did not finish, which is a
real failure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from bandits.diagnose.models import (
    ActionCall,
    CandidateView,
    ComponentResult,
    GroundingTransition,
    ResultStatus,
    RolloutResult,
    RolloutStep,
    Scenario,
    ScenarioState,
    SupportLevel,
    TerminationReason,
    ToolEffectCatalog,
    WorldOrigin,
)
from bandits.diagnose.retrieve import query_for, retrieve
from bandits.diagnose.verify import (
    compose_overall,
    is_simulation_conditioned,
    rollout_claims,
    score_communication,
    score_operational,
    score_process,
)
from bandits.diagnose.world import (
    ToolWorldPredictor,
    UserPolicyPredictor,
    commit,
    step_tool_world,
    step_user_policy,
    validate_instance,
    validate_transition,
)
from bandits.traces import Contract


class CandidateAction(Contract):
    """What a candidate decided to do this step."""

    calls: tuple[ActionCall, ...] = ()
    content: Any = None
    done: bool = False
    """The candidate declaring itself finished."""


class Candidate(Protocol):
    """The one call a candidate makes, so tests need no model."""

    def __call__(self, *, view: CandidateView, history: Sequence[dict[str, Any]]) -> Any: ...


class Budget(Contract):
    """What a rollout may spend before it is stopped.

    ``max_steps`` defaults to what the recorded episodes actually ran under —
    tau-bench's own ``max_steps: 200`` — so a different ceiling is a deliberate
    divergence rather than an arbitrary number.
    """

    max_steps: int = 200
    max_repeats: int = 3
    """Identical consecutive actions before the rollout is called a loop."""

    min_support: SupportLevel = SupportLevel.LOW


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


def _history_text(history: Sequence[dict[str, Any]], limit: int = 4000) -> str:
    rendered = "\n".join(f"{row.get('role')}: {row.get('content')}" for row in history)
    return rendered[-limit:]


def _action_key(action: CandidateAction) -> str:
    return "|".join(f"{call.tool}:{sorted(call.arguments.items())}" for call in action.calls)


def _offered_tool_names(view: CandidateView) -> frozenset[str]:
    names = set()
    for schema in view.offered_tools:
        name = schema.get("name") or (schema.get("function") or {}).get("name")
        if name:
            names.add(str(name))
    return frozenset(names)


def _tool_schema(view: CandidateView, name: str) -> dict[str, Any] | None:
    for schema in view.offered_tools:
        function = schema.get("function") if isinstance(schema.get("function"), dict) else schema
        if function.get("name") == name:
            parameters = function.get("parameters")
            return parameters if isinstance(parameters, dict) else {}
    return None


def _argument_errors(view: CandidateView, action: CandidateAction) -> tuple[str, ...]:
    """Validate each call's arguments against the schema the source declared.

    D72. This was a required-fields check plus a six-entry primitive type map,
    so enums, nested objects, array items, ``oneOf`` and ``additionalProperties``
    went unenforced: a call the real tool contract rejects reached the AWM and
    asked it to invent that behavior. Inputs now use the same ``jsonschema``
    machinery as declared outputs (D63).

    A tool offered without a schema is not validated here -- that is the
    ``unavailable`` case, and it is not reported as a candidate error.
    """
    errors: list[str] = []
    for call in action.calls:
        schema = _tool_schema(view, call.tool)
        if schema is None:
            continue
        error = validate_instance(call.arguments, schema, label="arguments")
        if error:
            errors.append(f"{call.tool} {error}")
    return tuple(errors)


def reset(scenario: Scenario) -> tuple[CandidateView, ScenarioState, list[dict[str, Any]]]:
    """Open a rollout branch: the candidate's view, the ledger, and the history.

    The state is the scenario's own, which came from the authentic prefix and
    nothing else. Branches never share it — a rollout that mutated a scenario
    would poison every later attempt at the same task, and the repeated sampling
    pass@k needs would silently stop being independent.
    """
    history = [
        {"role": step.role, "content": step.content, "tool": step.tool_name}
        for step in scenario.prefix
    ]
    return scenario.candidate_view(), scenario.initial_state, history


def run_rollout(
    scenario: Scenario,
    candidate: Candidate,
    *,
    index: Sequence[GroundingTransition],
    tool_world: ToolWorldPredictor,
    user_policy: UserPolicyPredictor,
    binding_id: str,
    candidate_id: str,
    seed: int = 0,
    budget: Budget = Budget(),
    catalog: ToolEffectCatalog | None = None,
    judge: Callable[..., bool] | None = None,
    versions: dict[str, str] | None = None,
) -> RolloutResult:
    """One complete attempt, scored and persisted whatever the outcome.

    Failures, abstentions and verifier-unknowns are all returned as results.
    A rollout that vanished because it went wrong is a rollout missing from the
    denominator, and a campaign that only records its successes reports the
    wrong number by construction.
    """
    view, state, history = reset(scenario)
    steps: list[RolloutStep] = []
    reason = TerminationReason.STEP_LIMIT
    repeats = 0
    previous_key = ""

    while len(steps) < budget.max_steps:
        raw = candidate(view=view, history=history)
        action = raw if isinstance(raw, CandidateAction) else CandidateAction()

        if action.done:
            if action.content:
                # The closing message is where a refusal or a summary is
                # actually stated. Dropping it because the candidate finished
                # in the same breath would make every communication
                # requirement unmeetable by a candidate that did it right.
                steps.append(
                    RolloutStep(index=len(steps), action_content=action.content, responder="none")
                )
            reason = TerminationReason.CANDIDATE_COMPLETED
            break

        key = _action_key(action)
        if key and key == previous_key:
            repeats += 1
            if repeats >= budget.max_repeats:
                # The environment answered every time; the candidate is stuck.
                steps.append(RolloutStep(index=len(steps), calls=action.calls))
                reason = TerminationReason.ACTION_LOOP
                break
        else:
            repeats = 0
        previous_key = key

        if action.calls:
            offered = _offered_tool_names(view)
            unavailable = tuple(call.tool for call in action.calls if call.tool not in offered)
            if unavailable:
                steps.append(
                    RolloutStep(
                        index=len(steps),
                        calls=action.calls,
                        action_content=action.content,
                        responder="none",
                        validator_rejections=(
                            f"candidate called unavailable tools: {', '.join(unavailable)}",
                        ),
                    )
                )
                reason = TerminationReason.UNAVAILABLE_TOOL
                break
            argument_errors = _argument_errors(view, action)
            if argument_errors:
                steps.append(
                    RolloutStep(
                        index=len(steps),
                        calls=action.calls,
                        action_content=action.content,
                        responder="none",
                        validator_rejections=argument_errors,
                    )
                )
                reason = TerminationReason.INVALID_CANDIDATE_ACTION
                break
            step, state, stop = _tool_step(
                scenario, action, state, history, index, tool_world, len(steps), budget, catalog
            )
        else:
            step, stop = _user_step(
                scenario, action, history, index, user_policy, len(steps), state
            )

        steps.append(step)
        history.append({"role": "assistant", "content": action.content, "tool": step.action_tool})
        if step.observation is not None:
            history.append(
                {
                    "role": "user" if step.responder == "user_policy" else "tool",
                    "content": step.observation,
                }
            )
        if stop is not None:
            reason = stop
            break
        if step.terminal:
            reason = TerminationReason.TERMINAL_EVENT
            break

    return _score(
        scenario,
        steps,
        state,
        binding_id=binding_id,
        candidate_id=candidate_id,
        seed=seed,
        terminated_by=reason,
        catalog=catalog,
        judge=judge,
        versions=versions or {},
    )


def _tool_step(
    scenario: Scenario,
    action: CandidateAction,
    state: ScenarioState,
    history: Sequence[dict[str, Any]],
    index: Sequence[GroundingTransition],
    tool_world: ToolWorldPredictor,
    step_index: int,
    budget: Budget,
    catalog: ToolEffectCatalog | None,
) -> tuple[RolloutStep, ScenarioState, TerminationReason | None]:
    query = query_for(
        scenario,
        state=state,
        tools=tuple(call.tool for call in action.calls),
        arguments_text=" ".join(str(call.arguments) for call in action.calls),
        history_text=_history_text(history),
    )
    examples = retrieve(query, index, catalog=catalog)
    proposal = step_tool_world(
        tool_world,
        calls=action.calls,
        content=action.content,
        state=state,
        history=_history_text(history),
        examples=examples,
    )

    if proposal.abstain:
        return (
            RolloutStep(
                index=step_index,
                calls=action.calls,
                action_content=action.content,
                responder="tool_world",
                abstained=True,
                support=proposal.support,
                retrieved_transition_ids=tuple(e.transition.transition_id for e in examples),
                validator_rejections=(proposal.abstain_reason,),
            ),
            state,
            TerminationReason.AWM_ABSTAINED,
        )

    outcome = validate_transition(
        proposal,
        calls=action.calls,
        state=state,
        step_index=step_index,
        catalog=catalog,
        min_support=budget.min_support,
        allowed_evidence_ids=tuple(e.transition.transition_id for e in examples),
        tool_schemas=scenario.offered_tools,
    )
    if not outcome.accepted:
        # One rejection is not fatal by itself, but a transition that cannot be
        # validated cannot be committed, and continuing from uncommitted state
        # would make every later step reason from a world that never existed.
        return (
            RolloutStep(
                index=step_index,
                calls=action.calls,
                action_content=action.content,
                responder="tool_world",
                abstained=True,
                support=proposal.support,
                validator_rejections=outcome.rejections,
                output_schema_validation=outcome.output_schema_validation,
                retrieved_transition_ids=tuple(e.transition.transition_id for e in examples),
            ),
            state,
            TerminationReason.INVALID_TRANSITION,
        )

    return (
        RolloutStep(
            index=step_index,
            calls=action.calls,
            action_content=action.content,
            responder="tool_world",
            observation=(
                {outcome.call_id: outcome.observation for outcome in proposal.call_outcomes}
                if proposal.call_outcomes
                else proposal.observation
            ),
            committed_delta={f.path: f.value for f in outcome.committed},
            committed_call_ids=outcome.committed_call_ids,
            executed_call_ids=outcome.executed_call_ids,
            events=outcome.events,
            output_schema_validation=outcome.output_schema_validation,
            terminal=proposal.terminal,
            world=WorldOrigin.SIMULATED,
            support=proposal.support,
            retrieved_transition_ids=tuple(e.transition.transition_id for e in examples),
        ),
        commit(state, outcome.committed),
        None,
    )


def _user_step(
    scenario: Scenario,
    action: CandidateAction,
    history: Sequence[dict[str, Any]],
    index: Sequence[GroundingTransition],
    user_policy: UserPolicyPredictor,
    step_index: int,
    state: ScenarioState,
) -> tuple[RolloutStep, TerminationReason | None]:
    """The user's reply, generated — never replayed from the trace.

    The recorded next user message answered a different agent action. Once the
    candidate diverges it is not a valid reply to what this candidate said, and
    using it would score the candidate against a conversation it never had.
    """
    query = query_for(
        scenario, state=state, history_text=_history_text(history), response_role="user_policy"
    )
    examples = retrieve(query, index)
    turn = step_user_policy(
        user_policy,
        profile=scenario.hidden_user,
        history=_history_text(history),
        message=str(action.content or ""),
        examples=examples,
    )

    if turn.abstain:
        return (
            RolloutStep(
                index=step_index,
                action_content=action.content,
                responder="user_policy",
                abstained=True,
                validator_rejections=(turn.abstain_reason,),
            ),
            TerminationReason.AWM_ABSTAINED,
        )

    return (
        RolloutStep(
            index=step_index,
            action_content=action.content,
            responder="user_policy",
            observation=turn.user_message,
            terminal=turn.terminal,
            world=WorldOrigin.SIMULATED,
            support=turn.support,
            disclosed_facts=turn.disclosed_facts,
        ),
        TerminationReason.USER_TERMINATED if turn.terminal else None,
    )


def _score(
    scenario: Scenario,
    steps: Sequence[RolloutStep],
    state: ScenarioState,
    *,
    binding_id: str,
    candidate_id: str,
    seed: int,
    terminated_by: TerminationReason,
    catalog: ToolEffectCatalog | None,
    judge: Callable[..., bool] | None,
    versions: dict[str, str],
) -> RolloutResult:
    contract = scenario.success_contract
    claims = rollout_claims(final_state=state, initial_state=scenario.initial_state, steps=steps)

    if terminated_by.is_candidate_failure:
        failed = ComponentResult(
            status=ResultStatus.FAIL,
            detail=f"candidate failed by {terminated_by.value}",
        )
        not_applicable = ComponentResult(status=ResultStatus.NOT_APPLICABLE)
        return RolloutResult(
            rollout_id=f"rollout-{_digest(scenario.scenario_id, candidate_id, str(seed))}",
            scenario_id=scenario.scenario_id,
            binding_id=binding_id,
            candidate_id=candidate_id,
            seed=seed,
            steps=tuple(steps),
            final_state=state,
            terminated_by=terminated_by,
            operational_result=failed,
            process_result=not_applicable,
            communication_result=not_applicable,
            overall=ResultStatus.FAIL,
            simulation_conditioned=False,
            **_versions(versions),
        )

    if terminated_by.invalidates_rollout or contract.unbindable_reason:
        # Nothing this rollout did can be read as evidence about the candidate.
        unknown = ComponentResult(
            status=ResultStatus.UNKNOWN,
            detail=contract.unbindable_reason or f"terminated by {terminated_by.value}",
        )
        return RolloutResult(
            rollout_id=f"rollout-{_digest(scenario.scenario_id, candidate_id, str(seed))}",
            scenario_id=scenario.scenario_id,
            binding_id=binding_id,
            candidate_id=candidate_id,
            seed=seed,
            steps=tuple(steps),
            final_state=state,
            terminated_by=terminated_by,
            operational_result=unknown,
            process_result=unknown,
            communication_result=unknown,
            overall=ResultStatus.UNKNOWN,
            simulation_conditioned=is_simulation_conditioned(claims),
            **_versions(versions),
        )

    operational = score_operational(contract, steps, state, catalog=catalog)
    process = score_process(contract.process_expectations, steps)
    communication = score_communication(contract.communication, steps, judge=judge)
    overall = compose_overall(operational, communication, reward_basis=contract.reward_basis)

    return RolloutResult(
        rollout_id=f"rollout-{_digest(scenario.scenario_id, candidate_id, str(seed))}",
        scenario_id=scenario.scenario_id,
        binding_id=binding_id,
        candidate_id=candidate_id,
        seed=seed,
        steps=tuple(steps),
        final_state=state,
        terminated_by=terminated_by,
        operational_result=operational,
        process_result=process,
        communication_result=communication,
        overall=overall,
        simulation_conditioned=is_simulation_conditioned(claims),
        **_versions(versions),
    )


def _versions(versions: dict[str, str]) -> dict[str, str]:
    """Every full-rollout number is conditioned on all four of these."""
    return {
        "tool_awm_version": versions.get("tool_awm", ""),
        "user_policy_version": versions.get("user_policy", ""),
        "retrieval_index_version": versions.get("retrieval_index", ""),
        "scenario_set_version": versions.get("scenario_set", ""),
    }
