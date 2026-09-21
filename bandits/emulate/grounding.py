"""Is a call's exact result identifiable, or only its tool's behavior?

I39: ``support_level()`` counts a same-tool retrieval hit as grounding
regardless of which entity it concerned. Two of ten held-out lookups in the
first real fidelity run asked for a specific stranger's private record
(``get_user_details("emma_kim_9957")``) with only *other* users' records
retrieved -- shape evidence, never a license to state Emma's exact fields.
The model correctly declined; the scorer read it as wrong.

Two independent dimensions, not one categorical verdict (Q21):

    values_identifiable  -- are the exact required field values knowable?
    behavior_supported   -- is the tool's shape/status/error behavior evidenced?

A same-tool-different-entity example proves the second without the first.

Deliberately narrow (per review of the first draft): grounding is assessed
per *call*, correlated by ``tool_call_id``, using a tool-independent entity
identity (``reservation:Q69X3R``, not ``get_reservation_details.Q69X3R`` --
the latter silently breaks identity across tools, reproducing I38/D77's
cross-tool state-path bug). Only the two single-call record-materialization
reads this was built to fix are classified; everything else -- writes,
batches, search/calculate/transfer/booking -- reports ``UNAVAILABLE`` rather
than guessing at semantics this module was not reviewed for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bandits.emulate.models import (
    ActionCall,
    CallGroundingAssessment,
    EntityRef,
    GroundingAssessment,
    GroundingKind,
    GroundingTransition,
    ScenarioState,
)
from bandits.emulate.retrieve import RetrievedExample

# Reviewed tau2-specific mapping, restricted to single-call record-materialization
# reads: the call's entire result is that one entity's record. Writes
# (cancel_reservation, update_*), creation (book_reservation), and
# non-entity tools (search, calculate, transfer_to_human_agents) do not fit
# this shape and are deliberately excluded -- see module docstring.
ENTITY_TOOLS: Mapping[str, tuple[str, str]] = {
    # tool -> (argument name, entity kind)
    "get_user_details": ("user_id", "user"),
    "get_reservation_details": ("reservation_id", "reservation"),
}


def _target_entity(tool: str, arguments: dict[str, Any]) -> EntityRef | None:
    mapping = ENTITY_TOOLS.get(tool)
    if mapping is None:
        return None
    arg_name, kind = mapping
    value = arguments.get(arg_name)
    if not isinstance(value, str) or not value:
        return None
    return EntityRef(kind=kind, id=value)


def _flatten_paths(payload: Any, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                found.update(_flatten_paths(value, path))
            else:
                found[path] = value
    elif isinstance(payload, list):
        found[f"{prefix}.length" if prefix else "length"] = len(payload)
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(value, (dict, list)):
                found.update(_flatten_paths(value, path))
            else:
                found[path] = value
    elif prefix:
        found[prefix] = payload
    return found


def _entity_facts_in_state(entity: EntityRef, state: ScenarioState) -> dict[str, Any]:
    """Facts about this entity already known, independent of which tool put them there.

    State paths are still tool-namespaced (``get_user_details.<id>.*``) by
    how ``compile.py`` writes them; this checks every tool whose entity
    argument matches, not just the one the call under assessment used --
    the entity's identity does not depend on how it was last looked up.
    """
    facts: dict[str, Any] = {}
    for tool, (_, kind) in ENTITY_TOOLS.items():
        if kind != entity.kind:
            continue
        prefix = f"{tool}.{entity.id}."
        for field in state.fields:
            if field.path.startswith(prefix):
                facts[field.path[len(prefix) :]] = field.value
    return facts


def _entity_facts_in_examples(
    entity: EntityRef, tool: str, examples: Sequence[RetrievedExample]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Facts about this exact entity found in retrieved evidence for this tool,
    and which transitions supplied them. Correlated by tool_call_id, never by
    pooling every observation on the retrieved transition.

    Two retrieved examples about the same entity that disagree on a field are
    two different snapshots in time, not corroborating evidence -- a
    disagreeing path is dropped rather than letting whichever example was
    seen last silently win.
    """
    arg_name, _ = ENTITY_TOOLS[tool]
    facts: dict[str, Any] = {}
    conflicting: set[str] = set()
    evidence_ids: list[str] = []
    for example in examples:
        for call in example.transition.action_calls:
            if call.tool != tool or call.arguments.get(arg_name) != entity.id:
                continue
            observation = next(
                (
                    o
                    for o in example.transition.observations
                    if o.role == "tool" and o.tool_call_id == call.call_id
                ),
                None,
            )
            if observation is None and len(example.transition.action_calls) == 1:
                # Single-call transitions in this corpus often carry no
                # call_id on either side -- fall back to positional pairing
                # rather than dropping evidence tool_call_id was never
                # populated for.
                observation = next(
                    (o for o in example.transition.observations if o.role == "tool"), None
                )
            if observation is None:
                continue
            found = _flatten_paths(observation.content)
            for path, value in found.items():
                if path in facts and facts[path] != value:
                    conflicting.add(path)
                else:
                    facts[path] = value
            if found:
                evidence_ids.append(example.transition.transition_id)
    for path in conflicting:
        facts.pop(path, None)
    return facts, tuple(dict.fromkeys(evidence_ids))


def _assess_call(
    call: ActionCall,
    recorded_observation: Any,
    *,
    state_before: ScenarioState,
    examples: Sequence[RetrievedExample],
) -> CallGroundingAssessment:
    entity = _target_entity(call.tool, call.arguments)
    # "exact_tool" on a RetrievedExample says retrieval matched *some* call in
    # the query -- for a batch that can be a different call's tool. Filter to
    # examples whose own transition actually called this call's tool, so
    # evidence for tool A in a batch can never license tool B's behavior.
    behavioral_evidence_ids = tuple(
        dict.fromkeys(
            example.transition.transition_id
            for example in examples
            if "exact_tool" in example.reasons
            and any(c.tool == call.tool for c in example.transition.action_calls)
        )
    )
    behavior_supported = bool(behavioral_evidence_ids)

    if entity is None or recorded_observation is None:
        return CallGroundingAssessment(
            call_id=call.call_id,
            tool=call.tool,
            behavioral_evidence_ids=behavioral_evidence_ids,
            behavior_supported=behavior_supported,
            kind=GroundingKind.UNAVAILABLE,
        )

    required_paths = tuple(_flatten_paths(recorded_observation).keys())
    known_from_state = _entity_facts_in_state(entity, state_before)
    known_from_examples, exact_fact_evidence_ids = _entity_facts_in_examples(
        entity, call.tool, examples
    )
    # state_before wins outright on conflict: it is this rollout's own
    # history, and a retrieved trace is a different episode's entity snapshot
    # that may predate a mutation (e.g. a retrieved "status: confirmed"
    # against a state_before that already recorded "status: cancelled"). A
    # path present in state_before is trusted from state regardless of what
    # retrieval says. Retrieval-only disagreement (no state_before backing
    # for that path at all -- checked in _entity_facts_in_examples, which
    # already drops a path where two retrieved examples disagree) is the only
    # case that can make a path unknown.
    known = {**known_from_examples, **known_from_state}

    identifiable = tuple(path for path in required_paths if path in known)
    missing = tuple(path for path in required_paths if path not in known)
    values_identifiable = bool(required_paths) and not missing

    if values_identifiable:
        # Always EXACT_ENTITY_FACT here, whether the source was state or
        # retrieval: a get_user_details/get_reservation_details read reports
        # a record, it does not derive one. DERIVABLE_FROM_STATE is reserved
        # for an operation (e.g. cancellation) whose post-state is a
        # transformation of known pre-state plus supported behavior -- this
        # classifier does not cover writes, so it never emits that kind.
        kind = GroundingKind.EXACT_ENTITY_FACT
    elif behavior_supported:
        kind = GroundingKind.BEHAVIORAL_ANALOGY
    else:
        kind = GroundingKind.UNSUPPORTED

    return CallGroundingAssessment(
        call_id=call.call_id,
        tool=call.tool,
        target_entities=(entity,),
        required_fact_paths=required_paths,
        identifiable_fact_paths=identifiable,
        missing_fact_paths=missing,
        exact_fact_evidence_ids=exact_fact_evidence_ids,
        behavioral_evidence_ids=behavioral_evidence_ids,
        values_identifiable=values_identifiable,
        behavior_supported=behavior_supported,
        kind=kind,
    )


def assess_grounding(
    transition: GroundingTransition,
    *,
    state_before: ScenarioState,
    examples: Sequence[RetrievedExample],
) -> GroundingAssessment:
    """Per-call grounding for this transition's action.

    Each call is correlated to its own recorded observation by
    ``tool_call_id`` where populated; a single-call transition falls back to
    positional pairing since this corpus does not always carry call ids on
    single-call spans. A batch is never unioned into one shared fact pool --
    a known entity in call A must not license an answer about call B.
    """
    tool_observations = [o for o in transition.observations if o.role == "tool"]
    assessments = []
    for call in transition.action_calls:
        recorded = None
        if call.call_id is not None:
            recorded = next(
                (o.content for o in tool_observations if o.tool_call_id == call.call_id), None
            )
        if recorded is None and len(transition.action_calls) == 1 and len(tool_observations) == 1:
            recorded = tool_observations[0].content
        assessments.append(
            _assess_call(call, recorded, state_before=state_before, examples=examples)
        )
    return GroundingAssessment(calls=tuple(assessments))
