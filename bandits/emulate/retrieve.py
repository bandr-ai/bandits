"""Fit-only, lineage-filtered retrieval of real transitions for the AWM.

Retrieval supplies enterprise facts; the world model decides how they apply.
This module is deliberately not a transition engine: it never returns "the
answer", only evidence, and the AWM normally produces the response even when an
exact-looking example exists, so it can substitute entities, account for
preceding state changes, and keep cross-turn consistency.

Compatibility filtering happens **before** similarity ranking, not after. A pure
refusal scenario must not receive a booking mutation as an analogous success
merely because both appeared in the same mined family — ranking a wrong example
highly is how an over-merged family teaches the simulator that booking is a
reasonable continuation for a task whose whole point was declining.

The exclusions are asserted at query time as well as index build time. A shared
index is exactly the thing that quietly stops being partitioned: one caller
passing the wrong filter once is enough, and the cost is a fidelity number that
measured copying rather than prediction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from bandits.emulate.models import (
    GroundingTransition,
    Partition,
    Scenario,
    ScenarioState,
    SuccessShape,
    ToolEffect,
    ToolEffectCatalog,
)
from bandits.traces import Contract

_WORD = re.compile(r"[a-z0-9_]+")


class LeakageError(ValueError):
    """A retrieval would have returned evidence the scenario must not see.

    Raised rather than filtered silently. A query that reaches for its own
    lineage is a bug in the caller, and quietly returning fewer results would
    hide it until the fidelity number was already published.
    """


ResponseRole = Literal["tool_world", "user_policy"]


class RetrievalQuery(Contract):
    """What the environment is asking about, not merely which tool was called.

    An action alone is a poor query: the same `cancel_reservation` call means
    different things after an eligibility check than before one, and similar
    language with different mutations must not be treated as interchangeable.
    """

    family_id: str
    shape: SuccessShape
    response_role: ResponseRole = "tool_world"
    """Which policy is asking.

    The two need different evidence and must not share a result set. A tool
    world needs transitions whose *observation* was a tool result for the same
    call; a user policy needs turns where a person replied, and there the
    hidden goal and the success shape do matter. Mixing them meant the tool
    world could be grounded on a user turn that never touched the tool.
    """

    task_context: str = ""
    history_text: str = ""
    state_keys: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    arguments_text: str = ""
    prior_errors: tuple[str, ...] = ()
    excluded_trace_ids: tuple[str, ...] = ()
    partition: Partition = Partition.FIT


class RetrievedExample(Contract):
    transition: GroundingTransition
    score: float
    reasons: tuple[str, ...] = ()
    """Why this was returned: exact tool, shared entity, error case, contrast."""


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    """Jaccard, with a floor at zero rather than a division error.

    Lexical on purpose: no embedding backend is configured anywhere in this
    pipeline, and a plausible similarity number from one that was never run
    would be fabricated geometry — the same objection the RLM miner records
    about family coherence.
    """
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_index(
    transitions: Sequence[GroundingTransition],
    *,
    fit_trace_ids: Sequence[str],
    sealed_trace_ids: Sequence[str] = (),
) -> tuple[GroundingTransition, ...]:
    """The retrievable set: fit-side, observed, never sealed.

    Three exclusions with three different reasons, and naming only one is how
    the others quietly stop holding:

    * **held-out** is what measures a verifier, so grounding on it would report
      memorisation as generalisation;
    * **sealed** is the partition kept out of the corpus entirely, and touching
      it here would spend the one clean fidelity measurement available;
    * **unobserved** transitions carry no observation at all, so they can teach
      a next-observation predictor nothing.
    """
    fit = set(fit_trace_ids)
    sealed = set(sealed_trace_ids)
    return tuple(
        transition
        for transition in transitions
        if transition.trace_id in fit and transition.trace_id not in sealed and transition.observed
    )


def query_for(
    scenario: Scenario,
    *,
    state: ScenarioState,
    tools: Sequence[str] = (),
    arguments_text: str = "",
    history_text: str = "",
    prior_errors: Sequence[str] = (),
    response_role: ResponseRole = "tool_world",
) -> RetrievalQuery:
    """Build a query from a scenario and the live rollout, exclusions included."""
    return RetrievalQuery(
        family_id=scenario.family_id,
        shape=scenario.success_contract.shape,
        response_role=response_role,
        task_context=scenario.task,
        history_text=history_text,
        state_keys=tuple(field.path for field in state.fields),
        tools=tuple(tools),
        arguments_text=arguments_text,
        prior_errors=tuple(prior_errors),
        excluded_trace_ids=scenario.retrieval_excluded_trace_ids,
        partition=scenario.partition,
    )


def _shape_compatible(candidate: SuccessShape | None, asked: SuccessShape) -> bool:
    """User-policy evidence is compatible only for the same bound shape.

    This predicate is used only by user-policy retrieval. Tool-world retrieval
    deliberately does not apply it: what a tool does is independent of whether
    the candidate should have called it. An unbound ``None`` shape is never
    compatible.
    """
    return candidate is asked


def _has_role(transition: GroundingTransition, role: ResponseRole) -> bool:
    """Whether this transition actually answered the kind of question being asked."""
    wanted = "user" if role == "user_policy" else "tool"
    return any(observation.role == wanted for observation in transition.observations)


def _tool_answered(transition: GroundingTransition, tools: set[str]) -> bool:
    """Whether a *tool observation* for one of these tools was recorded.

    An action naming the tool is not the same fact as the tool answering. A
    transition where the agent called ``cancel_reservation`` and the next thing
    recorded was a user message says nothing about what the tool returns, and
    counting it as support is how two name matches become "high" support for a
    transition nobody ever observed.
    """
    if not tools:
        return False
    named = {
        observation.tool_name
        for observation in transition.observations
        if observation.role == "tool"
    }
    if named & tools:
        return True
    # Older transitions may not carry the tool name on the observation; fall
    # back to the call only when a tool did in fact answer.
    return bool({call.tool for call in transition.action_calls} & tools) and any(
        observation.role == "tool" for observation in transition.observations
    )


def _effect_compatible(
    transition: GroundingTransition, tools: Sequence[str], catalog: ToolEffectCatalog | None
) -> bool:
    """Reads may not stand in for writes, or writes for reads.

    Effect class comes from the reviewed catalog. With no catalog the filter
    does not fire — an unreviewed toolset cannot be filtered honestly, and
    guessing from a name is the thing D26 refuses.
    """
    if catalog is None or not tools:
        return True
    asked = {catalog.effect_of(tool) for tool in tools}
    if ToolEffect.UNKNOWN in asked:
        return True
    got = {catalog.effect_of(call.tool) for call in transition.action_calls}
    return not got or bool(asked & got)


def retrieve(
    query: RetrievalQuery,
    index: Sequence[GroundingTransition],
    *,
    limit: int = 8,
    catalog: ToolEffectCatalog | None = None,
    include_contrast: bool = True,
    sealed_trace_ids: Sequence[str] = (),
) -> tuple[RetrievedExample, ...]:
    """Relevant real transitions: compatible first, then ranked, then diversified.

    The result mixes exact, analogous and contrasting examples rather than
    returning the single nearest neighbour. An AWM shown only successes of one
    action learns that the action always succeeds; the contrasting case — the
    same call answered differently, or answered with an error — is what carries
    the information that the outcome depends on state.
    """
    excluded = set(query.excluded_trace_ids) | set(sealed_trace_ids)
    asked_tools = set(query.tools)
    wants_user = query.response_role == "user_policy"
    query_tokens = _tokens(
        " ".join((query.task_context, query.history_text, query.arguments_text))
    ) | set(query.state_keys)

    scored: list[RetrievedExample] = []
    for transition in index:
        if transition.trace_id in excluded:
            continue
        if not transition.observed:
            continue
        if transition.family_id != query.family_id:
            # A shared index spans families. Evidence from another family is
            # about another world's entities and another contract's rules.
            continue
        if not _has_role(transition, query.response_role):
            continue
        if wants_user and not _shape_compatible(transition.success_shape, query.shape):
            # Shape matters to the user policy, whose behaviour is the task's,
            # and not to tool semantics, which are the system's. See D40.
            continue
        if not wants_user and not _effect_compatible(transition, query.tools, catalog):
            continue

        reasons: list[str] = []
        score = 0.0
        transition_tools = {call.tool for call in transition.action_calls}

        if asked_tools and _tool_answered(transition, asked_tools):
            score += 1.0
            reasons.append("exact_tool")
        elif asked_tools and transition_tools & asked_tools:
            score += 0.4
            reasons.append("tool_called_not_answered")
        elif asked_tools and transition_tools:
            score += 0.1
            reasons.append("other_tool")

        text = " ".join(
            (
                transition.task_context,
                " ".join(str(call.arguments) for call in transition.action_calls),
            )
        )
        lexical = _overlap(query_tokens, _tokens(text))
        score += lexical
        if lexical > 0.2:
            reasons.append("similar_context")

        shared = set(
            transition.state_before.fields and [f.path for f in transition.state_before.fields]
        ) & set(query.state_keys)
        if shared:
            score += 0.3 * min(1.0, len(shared) / 5)
            reasons.append("shared_state")

        if any(observation.error for observation in transition.observations):
            reasons.append("error_case")
            if query.prior_errors:
                score += 0.4

        if any(observation.role == "user" for observation in transition.observations):
            reasons.append("user_reaction")

        if score > 0:
            scored.append(
                RetrievedExample(
                    transition=transition, score=round(score, 4), reasons=tuple(reasons)
                )
            )

    scored.sort(key=lambda example: (-example.score, example.transition.transition_id))
    if not include_contrast:
        return tuple(scored[:limit])
    return _diversify(scored, limit)


def _diversify(examples: Sequence[RetrievedExample], limit: int) -> tuple[RetrievedExample, ...]:
    """Keep the best, then make room for a differently-shaped example.

    Without this the top-k of a happy-path corpus is k near-identical successes
    of the same call, and an AWM shown only those learns the action always
    succeeds. Reserving a slot for an error case is the cheapest guard against
    a simulator that has never seen anything fail.

    Replacement walks the tail from the end and never reuses a slot: an earlier
    version overwrote the same final position twice, silently dropping the
    first category it had just made room for. The reserved categories are
    ordered rather than a set, because set iteration order would make which
    category survived depend on hash seeding.
    """
    chosen = list(examples[:limit])
    if not chosen:
        return ()

    present = {reason for example in chosen for reason in example.reasons}
    slot = len(chosen) - 1
    for wanted in ("error_case", "user_reaction"):
        if wanted in present or slot < 0:
            continue
        replacement = next(
            (
                example
                for example in examples
                if wanted in example.reasons and example not in chosen
            ),
            None,
        )
        if replacement is None:
            continue
        chosen[slot] = replacement
        present |= set(replacement.reasons)
        slot -= 1
    return tuple(chosen)


def support_level(
    examples: Sequence[RetrievedExample],
    *,
    exact_needed: int = 2,
    role: ResponseRole = "tool_world",
) -> str:
    """How well-evidenced a transition is, from what retrieval actually found.

    ``exact_tool`` is only awarded where a *tool observation* for the asked
    tool was recorded — an action naming the tool whose next recorded event was
    a user message is not evidence about what the tool returns. An earlier
    version counted two such name matches as ``high``, which is how a
    transition nobody ever observed came to be treated as well grounded.

    Deliberately conservative beyond that. Given how thin this corpus is off
    the happy path — 2.1% of results are errors, and ``cancel_reservation``
    never failed once in 26 calls — a generous estimate is how an invented
    refusal gets committed as though it were grounded.
    """
    if role == "user_policy":
        answered = [e for e in examples if "user_reaction" in e.reasons]
    else:
        answered = [e for e in examples if "exact_tool" in e.reasons]
    if len(answered) >= exact_needed:
        return "high"
    if answered:
        return "medium"
    if examples:
        return "low"
    return "none"


def coverage_by_tool(
    index: Sequence[GroundingTransition],
) -> dict[str, dict[str, int]]:
    """How much evidence exists per tool, and how much of it is error evidence.

    Reported before any fidelity number is trusted. A fidelity score computed
    only over happy-path transitions says nothing about the transitions a weak
    candidate will actually produce, and this is the table that shows which
    tools have no off-happy-path evidence at all.
    """
    table: dict[str, dict[str, int]] = {}
    for transition in index:
        for call in transition.action_calls:
            row = table.setdefault(call.tool, {"total": 0, "errors": 0})
            row["total"] += 1
            matching = [
                observation
                for observation in transition.observations
                if observation.role == "tool"
                and observation.tool_call_id == call.call_id
            ]
            if not matching and len(transition.action_calls) == 1:
                matching = [
                    observation
                    for observation in transition.observations
                    if observation.role == "tool"
                    and (
                        observation.tool_name == call.tool
                        or observation.tool_name is None
                    )
                ]
            if any(observation.error for observation in matching):
                row["errors"] += 1
    return table
