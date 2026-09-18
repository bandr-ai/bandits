"""An agentic AWM: the tool world gathers its own grounding through read-only tools.

The fixed-RAG tool world (``world.py``) is handed one retrieval result and one
state snapshot and must answer from that alone. This module asks a different
question: can an AWM that actively inspects state, searches recorded
transitions, and checks tool contracts produce a *better-grounded* transition
than one -- and can it be prevented from gaining any authority a bug in
fixed-RAG couldn't already have?

The central design constraint, everywhere in this file: the AWM's tools are
read-only and boundary-checked in code, never by trusting the model to stay
inside them. Every tool:

    - can see only what this call/entity/partition is allowed to see;
    - is wrapped so every invocation is captured structurally (never by
      parsing DSPy's free-text ``trajectory``), for audit and for enforcing a
      call budget;
    - returns "not found" and "not reconstructed" as distinct answers, so the
      AWM cannot read silence as evidence of absence (I39's exact mistake,
      one level up: a tool that can't tell "no" from "unknown" teaches the
      model to guess).

The AWM still cannot write state. ``ProposedTransition`` goes through the same
``validate_transition`` gate everything else does (world.py), including the
same ``allowed_evidence_ids`` check -- for the agentic case that check must be
built from every transition ``search_transitions`` actually returned across
the whole episode, not one fixed retrieval, or the model could cite evidence
it never had a chance to see.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, ForwardRef, Protocol

from bandits.diagnose.models import (
    ActionCall,
    GroundingKind,
    GroundingTransition,
    Partition,
    PrefixStep,
    ScenarioState,
    SuccessShape,
    SupportLevel,
)
from bandits.diagnose.retrieve import RetrievalQuery, RetrievedExample, retrieve
from bandits.traces import Contract

MAX_GROUNDING_CALLS_DEFAULT = 6
MAX_SEARCH_RESULTS = 5


class AWMToolCall(Contract):
    """One internal grounding call the AWM made, captured structurally.

    Never reconstructed from DSPy's ``trajectory`` text: that format is an
    implementation detail of dspy.ReAct and not contractually stable, and a
    text-based reconstruction could not carry ``evidence_ids``/
    ``state_paths_read`` as typed fields the auditor can compare against
    ``validate_transition``'s ``allowed_evidence_ids``.
    """

    index: int
    tool: str
    arguments: dict[str, Any] = {}
    result_summary: str = ""
    """Short, loggable description of what the tool returned -- not the full
    payload, which may be large; the full payload lives in ``AWMRuntimeContext``
    accessible state for the auditor to re-derive if needed."""
    evidence_ids: tuple[str, ...] = ()
    """Every transition id a search_transitions call returned, matched or not
    -- behavioral/shape evidence regardless of entity. Never on its own a
    license for a specific entity's field values; see exact_entity_evidence_ids."""
    exact_entity_evidence_ids: tuple[str, ...] = ()
    """Subset of evidence_ids whose retrieved transition's own call actually
    named the entity_id this search asked about -- I39's exact distinction
    (exact_entity_fact vs. behavioral_analogy), enforced here instead of only
    at scoring time, so a same-tool-different-entity result can never silently
    stand in for a value claim about the entity the candidate call named."""
    exact_entity_values: dict[str, Any] = {}
    """Full dotted field path -> value, flattened (via _flatten_paths) from
    every entity_matched=true search result's recorded observation -- same
    representation state_values_read and every claim's own path use, so a
    nested field can actually be matched by path, not merely a leaf name that
    collides across unrelated siblings (billing.address.city vs
    shipping.address.city). A path where two matched examples disagreed is
    dropped rather than keeping whichever was seen last."""
    observed_tool_values: dict[str, tuple[Any, ...]] = {}
    """Full dotted field path -> every distinct value this exact search's
    results (matched or not) actually showed at that path, for THIS call's
    tool specifically. Used to check a claimed mutation's new value against
    what this tool has actually been observed to produce at this path --
    "origin was PHL, tool has evidence, so claim origin=Mars" must not pass
    just because the tool has *some* evidence; the claimed value itself has
    to have been observed, at this path, for this tool."""
    state_paths_read: tuple[str, ...] = ()
    state_values_read: dict[str, Any] = {}
    """Full sub-path (everything after ``{tool}.{entity_id}.``) -> value, for
    every field a read_world_state call actually returned -- the same
    representation _flatten_paths produces for a proposal's claimed fields,
    so a nested field (billing.address.city) is compared by its full path,
    never collapsed to a leaf name that collides with an unrelated sibling
    (shipping.address.city). Without values recorded at all (not just paths),
    "read status=confirmed, then claim status=cancelled" and "read
    status=confirmed, then claim status=confirmed" were indistinguishable."""
    entity: tuple[str, str] | None = None
    """(kind, id) this call concerned, when applicable (read_world_state,
    entity-scoped search_transitions) -- lets support-capping attribute
    grounding to the specific entity a candidate call named, instead of
    treating "something was grounded somewhere this episode" as support for
    every call."""
    error: str | None = None
    """Set when the tool call itself failed (bad arguments, disallowed
    access) -- distinct from a tool returning "not found," which is a valid
    result, not an error."""


class AWMExecutionTrace(Contract):
    """Everything the AWM did en route to one proposed transition."""

    grounding_calls: tuple[AWMToolCall, ...] = ()
    exhausted_budget: bool = False
    """True if the AWM used every call in its budget -- i.e. reached exactly
    max_calls. This alone does NOT mean a call was rejected: a model that
    made exactly max_calls successful calls and then cleanly called finish
    also reaches this count. See budget_rejection_attempted for the
    distinguishing signal."""
    budget_rejection_attempted: bool = False
    """True only if a tool call was actually refused for being over budget
    (with_budget's BudgetExceeded branch was reached at least once this
    episode) -- set by the caller from a counter with_budget itself
    increments, since dspy.ReAct.forward swallows the raised exception into
    an observation string and never lets it reach this frame directly."""
    raw_prediction: dict[str, Any] | None = None
    """The ReAct Prediction as returned, before projection onto the contract's
    declared fields -- including `trajectory` (the per-iteration thought /
    tool_name / tool_args / observation log) and `reasoning`. Kept so a parse
    failure is diagnosable after the fact: without it, a run that produced no
    structured output left nothing to inspect but a warning line, and the
    stage that actually failed (tool selection vs final extraction) could
    not be told apart from a repetition loop. Never read by scoring."""
    lm_history: tuple[dict[str, Any], ...] = ()
    """Per-LM-call records for this episode (stage, response text, token
    usage, finish reason), captured from dspy's own history. The signal that
    distinguishes an empty `text` with a long `reasoning_content` (a
    repetition loop) from a genuinely truncated near-complete answer."""
    trajectory: dict[str, Any] = {}
    """The controller's own per-iteration log (thought / tool_name /
    tool_args / observation). The audit trail remains the authority on what
    was actually called; this records what the model *said* it wanted,
    including selections that named no valid tool."""
    controller_error: str | None = None
    """Set when tool selection or final extraction failed. A selection
    failure does not end the episode -- extraction still runs on the
    grounding already gathered -- so this is how a run that finished on a
    degraded path is told apart from a clean one."""
    rejected_unsupported_claim_paths: tuple[str, ...] = ()
    """Set by step_agentic_tool_world when it abstained a proposal because a
    committed mutation/event claim was insufficiently grounded
    (assess_proposal_claims) -- distinct from the AWM's own epistemic
    abstention. A caller must not score this the same way as "the AWM said
    it didn't know": this is always a correct outcome by construction, since
    it only fires when a specific fabricated claim was actually caught.
    Empty for every other abstention reason."""

    @property
    def all_evidence_ids(self) -> tuple[str, ...]:
        """Every transition id any grounding call actually surfaced --
        behavioral/shape evidence, matched entity or not. Never on its own a
        license for a specific entity's field values; see
        all_exact_entity_evidence_ids.

        The only valid ``allowed_evidence_ids`` set for this episode's
        ``validate_transition`` call: citing an id no ``search_transitions``
        call ever returned is citing evidence the AWM never had.
        """
        seen: dict[str, None] = {}
        for call in self.grounding_calls:
            for evidence_id in call.evidence_ids:
                seen[evidence_id] = None
        return tuple(seen)

    @property
    def all_exact_entity_evidence_ids(self) -> tuple[str, ...]:
        """Subset of all_evidence_ids whose retrieved transition actually
        named the entity that specific search asked about (I39's
        distinction, enforced at the tool boundary -- see
        AWMToolCall.exact_entity_evidence_ids)."""
        seen: dict[str, None] = {}
        for call in self.grounding_calls:
            for evidence_id in call.exact_entity_evidence_ids:
                seen[evidence_id] = None
        return tuple(seen)

    @property
    def all_state_paths_read(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for call in self.grounding_calls:
            for path in call.state_paths_read:
                seen[path] = None
        return tuple(seen)

    def entities_grounded(self) -> set[tuple[str, str]]:
        """Every (kind, id) this episode actually grounded -- via a
        read_world_state call that found fields, or a search_transitions
        call with at least one exact-entity match. The set support-capping
        must check a candidate call's own target entity against, instead of
        asking only "was anything grounded anywhere this episode."
        """
        grounded: set[tuple[str, str]] = set()
        for call in self.grounding_calls:
            if call.entity is None:
                continue
            if call.tool == "read_world_state" and call.state_paths_read:
                grounded.add(call.entity)
            elif call.tool == "search_transitions" and call.exact_entity_evidence_ids:
                grounded.add(call.entity)
        return grounded


class AWMRuntimeContext(Contract):
    """Everything one agentic transition call is allowed to touch.

    Constructed once per candidate action by the caller (the smoke runner, or
    later the rollout loop) -- never by the AWM itself. Every tool below reads
    only from this object and the arguments the model supplies; nothing here
    is fetched fresh from a shared index mid-call, which is what would let a
    later-added tool accidentally see a different partition than the one the
    episode was constructed against.
    """

    candidate_calls: tuple[ActionCall, ...]
    action_content: Any = None
    current_state: ScenarioState = ScenarioState()
    history_text: str = ""
    history_before: tuple[PrefixStep, ...] = ()
    """The same prefix `history_text` renders, kept structured.

    `history_text` flattens every replayed tool observation into prose, so the
    AWM can read a recorded payload that the claim auditor has no typed view
    of -- the authority asymmetry. Keeping the steps lets read_world_state
    surface those observations as first-class, provenance-tagged state, so a
    fact the model can see is a fact the auditor can recognize."""
    offered_tool_schemas: tuple[dict[str, Any], ...] = ()
    fit_index: tuple[GroundingTransition, ...] = ()
    """Always fit-partition only (Partition.FIT) -- constructed by the caller
    from build_index(), never re-derived by a tool from a broader corpus."""
    family_id: str = ""
    task_context: str = ""
    excluded_trace_ids: tuple[str, ...] = ()
    """This transition's own trace plus every trace sharing its lineage_id --
    the same exclusion excluded_trace_ids_for() computes in run_awm_fidelity.py.
    search_transitions applies this on every call; there is no way for the
    AWM to opt out of it."""
    prior_grounding_calls: tuple[AWMToolCall, ...] = ()
    """Read-only history of this episode's own earlier tool calls, for
    inspect_grounding_history. Never includes another rollout's calls,
    another scenario's, or anything from a future step."""


class AllowedEntity(Contract):
    kind: str
    id: str


def _entities_from_calls(calls: Sequence[ActionCall]) -> tuple[AllowedEntity, ...]:
    """Entities the candidate's own calls named -- the only ones read_world_state
    may resolve without an explicit relationship through already-known state.

    A narrower, local restatement of grounding.py's ENTITY_TOOLS mapping.
    Importing that module directly would couple this file's access-control
    boundary to grounding.py's read-only-lookup-tool scope, which may change
    independently; this tiny local table only needs "which argument names can
    a candidate call plausibly point at an entity," not grounding.py's full
    per-call fidelity semantics.
    """
    entity_args = {
        "user_id": "user",
        "reservation_id": "reservation",
        "flight_number": "flight",
        "payment_id": "payment",
    }
    found: list[AllowedEntity] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        for arg_name, kind in entity_args.items():
            value = call.arguments.get(arg_name)
            if isinstance(value, str) and value and (kind, value) not in seen:
                seen.add((kind, value))
                found.append(AllowedEntity(kind=kind, id=value))
    return tuple(found)


_KIND_BY_STATE_TOOL = {
    "get_user_details": "user",
    "get_reservation_details": "reservation",
}


class GroundingToolError(ValueError):
    """A tool call asked for something outside its access boundary.

    Raised, not silently narrowed -- a tool that quietly returned "not found"
    for a disallowed entity would look identical to a genuinely absent
    record, hiding the boundary violation from anyone reading the audit.
    """


def make_inspect_tool_contract(
    context: AWMRuntimeContext, *, audit: list[AWMToolCall]
) -> Callable[[str], dict[str, Any]]:
    """Only candidate-offered tools; no success contract, no verifier, no
    hidden expected result -- ``offered_tool_schemas`` never carries those."""
    by_name = {
        schema.get("name") or schema.get("function", {}).get("name"): schema
        for schema in context.offered_tool_schemas
    }

    def inspect_tool_contract(tool_name: str) -> dict[str, Any]:
        index = len(audit)
        schema = by_name.get(tool_name)
        offered = schema is not None
        if not offered:
            audit.append(
                AWMToolCall(
                    index=index,
                    tool="inspect_tool_contract",
                    arguments={"tool_name": tool_name},
                    result_summary="not offered",
                )
            )
            return {"name": tool_name, "offered": False}
        audit.append(
            AWMToolCall(
                index=index,
                tool="inspect_tool_contract",
                arguments={"tool_name": tool_name},
                result_summary="offered",
            )
        )
        return {"name": tool_name, "offered": True, "schema": schema}

    return inspect_tool_contract


def _linked_entities(
    already_read: set[tuple[str, str]],
    state: ScenarioState,
    audit: Sequence[AWMToolCall] = (),
) -> set[tuple[str, str]]:
    """Entities reachable by an actual relationship from an entity already
    read this episode -- e.g. a reservation's own ``user_id`` field, once the
    reservation itself was read via read_world_state. This is genuine
    traversal, not "anything already sitting in state": an entity the
    candidate never named and that read_world_state never actually reached
    must not become readable just because it happens to have a record in the
    same state object (a different entity's data being present is not a
    relationship to it).
    """
    linked: set[tuple[str, str]] = set()
    relationship_args = {"user_id": "user", "reservation_id": "reservation"}
    for kind, entity_id in already_read:
        tool = next((t for t, k in _KIND_BY_STATE_TOOL.items() if k == kind), None)
        if tool is None:
            continue
        prefix = f"{tool}.{entity_id}."
        for field in state.fields:
            if not field.path.startswith(prefix):
                continue
            leaf = field.path[len(prefix) :]
            linked_kind = relationship_args.get(leaf)
            if linked_kind and isinstance(field.value, str) and field.value:
                linked.add((linked_kind, field.value))
        for call in audit:
            if call.tool != "read_world_state" or call.entity != (kind, entity_id):
                continue
            for path, value in call.state_values_read.items():
                linked_kind = relationship_args.get(path.rsplit(".", 1)[-1])
                if linked_kind and isinstance(value, str) and value:
                    linked.add((linked_kind, value))
    return linked


_ENTITY_ID_KEYS = {"user": "user_id", "reservation": "reservation_id"}
"""Which payload key identifies the entity a recorded tool observation is
about. A prior observation is attributed by the id its OWN payload carries,
not by the call's arguments -- the corpus records `arguments=None` on replayed
tool steps, and one episode's history routinely holds several reservations
(five, in the pinned family's cancel case). Attributing by position or by the
conversation's subject would let one reservation's payment amount license a
claim about another."""


class _PriorField(Contract):
    """One field from one recorded tool observation replayed before this action."""

    value: Any = None
    span_id: str | None = None
    tool: str = ""
    conflicting_values: tuple[Any, ...] = ()
    """Other values earlier observations gave this same path. Non-empty means
    the history disagrees with itself, and the field is reported as a conflict
    rather than silently resolved to the latest -- a silently-picked value is
    indistinguishable from a verified one downstream."""


def _prior_observation_fields(
    context: AWMRuntimeContext, entity_kind: str, entity_id: str
) -> dict[str, _PriorField]:
    """Fields recorded by tool observations replayed BEFORE the current action,
    scoped to one entity.

    These are part of the reconstructed state, not a separate knowledge
    source: the environment observed them in this very episode. Surfacing
    them through read_world_state is what closes the authority asymmetry --
    otherwise the AWM can read a fact in the rendered history that the claim
    auditor cannot recognize, so using it looks like fabrication and ignoring
    it forces an abstention on a recoverable answer.

    Only ``history_before`` is read, so nothing at or after the current action
    can leak in. Scenario/lineage isolation is inherited: history_before
    belongs to this transition's own prefix, which is exactly the material the
    candidate itself was shown.
    """
    id_key = _ENTITY_ID_KEYS.get(entity_kind)
    if not id_key:
        return {}
    found: dict[str, _PriorField] = {}
    for step in context.history_before:
        if step.role != "tool" or step.error or not isinstance(step.content, dict):
            continue
        if step.content.get(id_key) != entity_id:
            continue
        for path, value in _flatten_paths(step.content).items():
            existing = found.get(path)
            if existing is None:
                found[path] = _PriorField(
                    value=value, span_id=step.span_id, tool=step.tool_name or ""
                )
            elif existing.value != value:
                # Later observation disagrees with an earlier one. Keep the
                # newer value but record the disagreement: a field the history
                # contradicts itself about is not a verified fact.
                found[path] = _PriorField(
                    value=value,
                    span_id=step.span_id,
                    tool=step.tool_name or "",
                    conflicting_values=(*existing.conflicting_values, existing.value),
                )
    return found


def make_read_world_state(
    context: AWMRuntimeContext, *, audit: list[AWMToolCall]
) -> Callable[[str, str], dict[str, Any]]:
    """Restricted to entities the candidate's own calls named, plus entities
    reachable by an actual relationship field from an entity already read
    this episode -- never an arbitrary id the model invents to go fishing in
    the hidden world, and never merely "present somewhere in state" (the
    fixed bug: state can hold other, unrelated entities' data, and their mere
    presence is not a relationship to the entity being requested).

    ``found``/``completeness`` are reported as two separate axes so "no field
    at this path" (found=false) can never be misread as "field confirmed
    absent" versus "field never reconstructed" (I39's found-vs-unknown
    mistake, generalized): this corpus is a partial reconstruction, and a
    tool that collapses those into one boolean teaches the AWM to treat
    silence as a negative fact.
    """
    named = {(e.kind, e.id) for e in _entities_from_calls(context.candidate_calls)}

    def read_world_state(entity_kind: str, entity_id: str) -> dict[str, Any]:
        index = len(audit)
        already_read = {
            (c.arguments.get("entity_kind"), c.arguments.get("entity_id"))
            for c in audit
            if c.tool == "read_world_state" and not c.error
        }
        allowed = named | already_read | _linked_entities(
            already_read, context.current_state, audit
        )
        if (entity_kind, entity_id) not in allowed:
            audit.append(
                AWMToolCall(
                    index=index,
                    tool="read_world_state",
                    arguments={"entity_kind": entity_kind, "entity_id": entity_id},
                    error="entity not reachable from the candidate's own calls or known state",
                )
            )
            return {
                "found": False,
                "completeness": "unknown",
                "fields": {},
                "reason": "entity not accessible from this call's context",
            }

        tool_prefix = {"user": "get_user_details", "reservation": "get_reservation_details"}.get(
            entity_kind
        )
        fields: dict[str, Any] = {}
        # Full canonical path ("<tool>.<entity_id>.<field>") -> value, kept
        # separately from `fields` (which strips the prefix for the tool's
        # own JSON response to the model, where the redundant prefix on every
        # key would be noise). This is the audit-internal representation, and
        # it must match the SAME convention StateDelta.path/observation
        # fields actually use -- world.py's own instruction defines
        # StateDelta.path as exactly "<tool>.<entity_id>.<field>". Using the
        # stripped sub-path here (the earlier bug) meant no real mutation
        # claim from an instructed model could ever match against it.
        canonical_values: dict[str, Any] = {}
        state_paths_read: list[str] = []
        if tool_prefix:
            prefix = f"{tool_prefix}.{entity_id}."
            for field in context.current_state.fields:
                if field.path.startswith(prefix):
                    fields[field.path[len(prefix) :]] = field.value
                    canonical_values[field.path] = field.value
                    state_paths_read.append(field.path)

        # Committed scenario state is authoritative; a prior recorded
        # observation fills only what state does not already carry. State is
        # this rollout's own committed view, while an observation is a
        # snapshot from earlier in the episode that a later mutation may
        # already have superseded -- the same precedence read_world_state
        # already takes over search_transitions results.
        provenance: dict[str, str] = dict.fromkeys(fields, "scenario_state")
        conflicts: dict[str, list[Any]] = {}
        if tool_prefix:
            prior = _prior_observation_fields(context, entity_kind, entity_id)
            for path, prior_field in prior.items():
                if path in fields:
                    continue
                if prior_field.conflicting_values:
                    # The history disagrees with itself about this path.
                    # Reported as an explicit conflict rather than resolved to
                    # the newest value: a silently-picked value is
                    # indistinguishable downstream from a verified one.
                    conflicts[path] = [*prior_field.conflicting_values, prior_field.value]
                    continue
                fields[path] = prior_field.value
                provenance[path] = "prior_recorded_observation"
                canonical = f"{tool_prefix}.{entity_id}.{path}"
                canonical_values[canonical] = prior_field.value
                state_paths_read.append(canonical)

        found = bool(fields)
        audit.append(
            AWMToolCall(
                index=index,
                tool="read_world_state",
                arguments={"entity_kind": entity_kind, "entity_id": entity_id},
                result_summary=(
                    f"found={found}, {len(fields)} field(s)"
                    + (f", {len(conflicts)} conflicting" if conflicts else "")
                ),
                state_paths_read=tuple(state_paths_read),
                state_values_read=canonical_values,
                entity=(entity_kind, entity_id),
            )
        )
        return {
            "found": found,
            # complete/partial only ever describes a hit: an entity that
            # produced zero fields was never reconstructed at all, which is
            # "unknown," not "confirmed empty."
            "completeness": "partial" if found else "unknown",
            "fields": fields,
            # Per-field origin, so "the environment committed this" is never
            # confused with "the environment observed this earlier in the
            # episode." completeness stays "partial" regardless: prior
            # observations filling gaps does not make the reconstruction
            # complete, and upgrading it would teach the AWM to read silence
            # as a confirmed absence.
            "provenance": provenance,
            # Paths whose replayed observations disagree with each other.
            # Deliberately not merged into `fields`: the AWM is told the
            # history is inconsistent here rather than handed one arbitrary
            # side of the disagreement as if it were established.
            "conflicts": {path: list(values) for path, values in conflicts.items()},
        }

    return read_world_state


_SEARCH_ENTITY_ARGS = {
    "get_user_details": "user_id",
    "get_reservation_details": "reservation_id",
    "cancel_reservation": "reservation_id",
    "update_reservation_flights": "reservation_id",
    "update_reservation_passengers": "reservation_id",
    "update_reservation_baggages": "reservation_id",
    "book_reservation": "user_id",
}
"""Extends beyond the two read-only ENTITY_TOOLS in grounding.py: entity
matching for search_transitions/_matches_entity must also work for write
tools (cancel_reservation etc.) -- a search for the exact reservation a
mutation targets is a legitimate exact-entity query, not only a lookup."""


def _matches_entity(example: RetrievedExample, tool: str, entity_id: str) -> bool:
    """True only if the retrieved transition's own call named this exact
    entity -- never lexical/ranking similarity. Mirrors grounding.py's
    ENTITY_TOOLS distinction (exact_entity_fact vs. behavioral_analogy)."""
    arg_name = _SEARCH_ENTITY_ARGS.get(tool)
    if arg_name is None:
        return False
    return any(
        call.tool == tool and call.arguments.get(arg_name) == entity_id
        for call in example.transition.action_calls
    )


def _example_entity_id(example: RetrievedExample, tool: str) -> str | None:
    """The entity id the retrieved transition's own call for this tool
    actually used, for canonicalizing exact_entity_values to the same full
    "<tool>.<entity_id>.<field>" path convention every other value source
    uses."""
    arg_name = _SEARCH_ENTITY_ARGS.get(tool)
    if arg_name is None:
        return None
    for call in example.transition.action_calls:
        if call.tool == tool:
            value = call.arguments.get(arg_name)
            if isinstance(value, str) and value:
                return value
    return None


def make_search_transitions(
    context: AWMRuntimeContext, *, audit: list[AWMToolCall]
) -> Callable[..., dict[str, Any]]:
    """Always fit-partition, always family-filtered, always lineage-excluded --
    the AWM chooses *what* to search for, never *which partition* to search.
    ``RetrievalQuery.partition`` defaults to ``Partition.FIT`` and is never set
    from an argument this tool exposes.

    ``entity_id``, when given, used to be ranking text only -- every result
    still counted as support regardless of which entity it actually concerned
    (I39's exact bug, reborn here). Results are still returned unfiltered (the
    AWM may legitimately want same-tool-different-entity examples to learn
    shape/behavior), but each result is now labeled ``entity_matched``, and
    only the matched subset is reported as ``exact_entity_evidence_ids`` --
    the only evidence support-capping may treat as a specific-value claim.
    """

    def search_transitions(
        tool: str,
        query: str = "",
        entity_id: str = "",
        include_errors: bool = False,
        limit: int = MAX_SEARCH_RESULTS,
    ) -> dict[str, Any]:
        index = len(audit)
        capped_limit = min(max(limit, 1), MAX_SEARCH_RESULTS)
        retrieval_query = RetrievalQuery(
            family_id=context.family_id,
            shape=SuccessShape.INFORMATIONAL,
            response_role="tool_world",
            task_context=context.task_context or query,
            history_text=context.history_text,
            tools=(tool,) if tool else (),
            arguments_text=f"{query} {entity_id}".strip(),
            excluded_trace_ids=context.excluded_trace_ids,
            partition=Partition.FIT,
        )
        found = retrieve(retrieval_query, context.fit_index, limit=capped_limit)
        if include_errors:
            error_examples = [e for e in found if "error_case" in e.reasons]
            if not error_examples:
                # Look specifically for error evidence even if the default
                # ranking didn't surface any within the cap -- the AWM asked
                # for it explicitly.
                broader = retrieve(retrieval_query, context.fit_index, limit=MAX_SEARCH_RESULTS * 4)
                # Slice the combined result, not just the appended tail: with
                # `found` already at the cap, slicing only the generator let
                # the call return up to twice MAX_SEARCH_RESULTS.
                found = (
                    tuple(found)
                    + tuple(e for e in broader if "error_case" in e.reasons and e not in found)
                )[:capped_limit]

        results = []
        exact_entity_evidence_ids: list[str] = []
        # exact_entity_values is keyed by full canonical path
        # ("<tool>.<entity_id>.<field>") -- the SAME convention
        # StateDelta.path/observation fields and state_values_read use
        # (world.py's own instructed contract), built from the matched
        # example's own entity_id, since these values genuinely concern that
        # specific entity.
        #
        # observed_tool_values is entity-agnostic tool-shape evidence
        # (does this tool ever show THIS bare field at THIS value, for ANY
        # entity) and is deliberately keyed by the bare sub-path instead --
        # canonicalizing it to one specific entity's id would make it useless
        # for checking a claim about a *different* (or not-yet-grounded)
        # entity's mutation, which is exactly the case it exists to cover.
        exact_entity_values: dict[str, Any] = {}
        conflicting_exact_entity_paths: set[str] = set()
        observed_tool_values: dict[str, set[Any]] = {}
        for example in found:
            matched = bool(entity_id) and _matches_entity(example, tool, entity_id)
            rendered = _render_example(example, tool=tool)
            rendered["entity_matched"] = matched
            results.append(rendered)
            example_entity_id = _example_entity_id(example, tool)
            rendered_observations = rendered.get("observations", ())
            for observation in rendered_observations:
                if not isinstance(observation, dict):
                    continue
                for path, value in _flatten_paths(observation).items():
                    observed_tool_values.setdefault(path, set()).add(_hashable(value))
            if matched:
                exact_entity_evidence_ids.append(example.transition.transition_id)
                if example_entity_id:
                    canonical_prefix = f"{tool}.{example_entity_id}."
                    for observation in rendered_observations:
                        if not isinstance(observation, dict):
                            continue
                        for sub_path, value in _flatten_paths(observation).items():
                            path = f"{canonical_prefix}{sub_path}"
                            if path in exact_entity_values and exact_entity_values[path] != value:
                                conflicting_exact_entity_paths.add(path)
                            else:
                                exact_entity_values[path] = value
        for path in conflicting_exact_entity_paths:
            exact_entity_values.pop(path, None)

        evidence_ids = tuple(example.transition.transition_id for example in found)
        audit.append(
            AWMToolCall(
                index=index,
                tool="search_transitions",
                arguments={
                    "tool": tool,
                    "query": query,
                    "entity_id": entity_id,
                    "include_errors": include_errors,
                    "limit": limit,
                },
                result_summary=f"{len(results)} result(s), {len(exact_entity_evidence_ids)} entity-matched",
                evidence_ids=evidence_ids,
                exact_entity_values=exact_entity_values,
                exact_entity_evidence_ids=tuple(exact_entity_evidence_ids),
                observed_tool_values={
                    path: tuple(values) for path, values in observed_tool_values.items()
                },
                entity=(
                    ("user" if _SEARCH_ENTITY_ARGS.get(tool) == "user_id" else "reservation"),
                    entity_id,
                )
                if entity_id and tool in _SEARCH_ENTITY_ARGS
                else None,
            )
        )
        return {"results": results}

    return search_transitions


def _hashable(value: Any) -> Any:
    """Leaf values from _flatten_paths are always scalars, but guard against
    an unhashable edge case rather than crash on a malformed retrieved
    payload."""
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _flatten_paths(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Same shape as fidelity.py's ``_flatten``/grounding.py's own copy --
    kept local rather than imported across module boundaries, since it is a
    tiny pure utility and the leading underscore in both existing copies
    already signals "not a cross-module import surface." """
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


def _render_example(example: RetrievedExample, *, tool: str | None = None) -> dict[str, Any]:
    transition = example.transition
    calls = [call for call in transition.action_calls if tool is None or call.tool == tool]
    call_ids = {call.call_id for call in calls if call.call_id}
    observations = [
        observation
        for observation in transition.observations
        if observation.role == "tool" and observation.tool_call_id in call_ids
    ]
    if not observations and len(calls) == 1:
        same_tool = [
            observation
            for observation in transition.observations
            if observation.role == "tool"
            and (observation.tool_name == calls[0].tool or observation.tool_name is None)
        ]
        if len(same_tool) == 1:
            observations = same_tool
    rendered_observations = [observation.content for observation in observations]
    return {
        "transition_id": transition.transition_id,
        "tool": tool or transition.action_tool,
        "arguments": [dict(call.arguments) for call in calls],
        "observation": rendered_observations[0] if len(rendered_observations) == 1 else None,
        "observations": rendered_observations,
        "error": any(observation.error for observation in observations),
        "reasons": list(example.reasons),
    }


def make_inspect_grounding_history(
    context: AWMRuntimeContext, *, audit: list[AWMToolCall]
) -> Callable[[str, str], dict[str, Any]]:
    """This episode's own prior *grounding* calls about an entity -- never the
    verifier, never a future recorded transition, never another rollout's
    steps, and (the name matters) never the candidate's own tool-call
    outcomes, committed observations, state changes, or events. Those belong
    to a real world/rollout ledger this module does not have access to; a
    tool named merely "inspect_ledger" implied it did, which it never did.
    This lets the AWM avoid re-reading/re-searching the same entity it
    already grounded earlier in this same episode -- nothing more.

    Two sources, both scoped to "this episode, before now," never anything
    else:

        - ``audit`` itself -- the same live, shared list every other tool in
          this call appends to. Reading it (excluding the entry this very
          call is about to add) is what lets this tool see a call made two
          turns earlier *within this same ReAct loop*; reading only
          ``context.prior_grounding_calls`` (fixed at context construction,
          before the loop started) can never see that -- it would only ever
          be empty for a single-step call.
        - ``context.prior_grounding_calls`` -- calls from earlier *steps* of
          a multi-step episode, seeded by the caller once multi-step rollout
          integration exists (Phase 10). Empty today; kept for that reason
          rather than being reused to fake the live-audit case.

    Matched by (entity_kind, entity_id) equality against each call's own
    ``entity`` field -- not the id alone, which could otherwise cross-match a
    reservation and a user that happen to share an id-shaped string.
    """

    def inspect_grounding_history(entity_kind: str, entity_id: str) -> dict[str, Any]:
        index = len(audit)
        candidates = tuple(context.prior_grounding_calls) + tuple(audit)
        matches = [
            {
                "tool": call.tool,
                "arguments": call.arguments,
                "result_summary": call.result_summary,
            }
            for call in candidates
            if call.entity == (entity_kind, entity_id)
        ]
        audit.append(
            AWMToolCall(
                index=index,
                tool="inspect_grounding_history",
                arguments={"entity_kind": entity_kind, "entity_id": entity_id},
                result_summary=f"{len(matches)} prior grounding call(s)",
                entity=(entity_kind, entity_id),
            )
        )
        return {"prior_grounding_calls": matches}

    return inspect_grounding_history


class BudgetExceeded(GroundingToolError):
    """The AWM used its entire grounding-call budget without calling finish."""


def with_budget(
    tool: Callable[..., Any],
    *,
    audit: list[AWMToolCall],
    max_calls: int,
    rejections: list[int],
    on_call: Callable[[AWMToolCall], None] | None = None,
) -> Callable[..., Any]:
    """Wraps a tool so exceeding the shared per-episode call budget refuses
    the call, rather than letting the AWM keep spending calls (and tokens)
    indefinitely. dspy.ReAct's own ``max_iters`` is a second, coarser backstop
    already covering finish/non-finish iterations together; this one is exact
    about grounding-tool calls specifically.

    ``rejections`` is a shared mutable counter (a one-element list, appended
    to rather than incremented in place so every wrapped tool shares the same
    object): reaching ``len(audit) == max_calls`` alone conflates "the model
    used exactly its budget and then cleanly finished" with "the model tried
    to go over and was refused" -- only the second is a real rejection event,
    and it is the only one worth reporting as ``budget_rejection_attempted``.

    Raises ``BudgetExceeded`` -- correct for a caller invoking the tool
    directly (every boundary test in agentic_test.py does exactly this) -- but
    dspy.ReAct.forward wraps every tool call in a bare `except Exception` and
    turns the exception into an observation string instead of letting it
    propagate. build_agentic_tool_world_predictor reads exhaustion/rejection
    back from ``audit``/``rejections`` after the run, not from catching this.
    """

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if len(audit) >= max_calls:
            rejections.append(1)
            raise BudgetExceeded(f"grounding-call budget of {max_calls} exhausted")
        result = tool(*args, **kwargs)
        # Fired after the tool has appended its own AWMToolCall, so the
        # observer sees the recorded call, not a reconstruction. Purely a
        # progress hook: an observer that raises must never turn a successful
        # grounding call into a failed one, so it is called defensively.
        if on_call is not None and audit:
            try:
                on_call(audit[-1])
            except Exception:  # noqa: BLE001 -- a progress hook cannot fail the run
                pass
        return result

    wrapped.__name__ = getattr(tool, "__name__", "tool")
    wrapped.__doc__ = getattr(tool, "__doc__", None)
    return wrapped


def build_grounding_tools(
    context: AWMRuntimeContext,
    *,
    max_calls: int = MAX_GROUNDING_CALLS_DEFAULT,
    on_call: Callable[[AWMToolCall], None] | None = None,
) -> tuple[list[Callable[..., Any]], list[AWMToolCall], list[int]]:
    """The four read-only tools, each budget-wrapped and sharing one audit log.

    Returns (tools, audit, rejections) -- ``rejections`` is the shared counter
    with_budget appends to; its non-emptiness is the actual
    ``budget_rejection_attempted`` signal, distinct from ``len(audit) ==
    max_calls`` alone.
    """
    audit: list[AWMToolCall] = []
    rejections: list[int] = []
    tools = [
        make_inspect_tool_contract(context, audit=audit),
        make_read_world_state(context, audit=audit),
        make_search_transitions(context, audit=audit),
        make_inspect_grounding_history(context, audit=audit),
    ]
    budgeted = [
        with_budget(
            tool, audit=audit, max_calls=max_calls, rejections=rejections, on_call=on_call
        )
        for tool in tools
    ]
    for original, wrapped in zip(tools, budgeted, strict=True):
        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
    return budgeted, audit, rejections


class AgenticToolWorldPredictor(Protocol):
    """One agentic transition call: build tools scoped to this context, run
    the AWM's grounding loop, return the raw signature output plus the audit."""

    def __call__(
        self, *, instruction: str, context: AWMRuntimeContext
    ) -> tuple[Any, AWMExecutionTrace]: ...


AGENTIC_TOOL_WORLD_INSTRUCTION = """\
You are the ENVIRONMENT, not the agent. You return what a tool call produces.

You have read-only tools to gather grounding before answering. Use them:
- inspect_tool_contract(tool_name) to see a tool's schema before answering it.
- read_world_state(entity_kind, entity_id) to check what is actually known
  about an entity the ACTION named (or a related entity reached from one
  already read this episode). This can return found=false -- that means the
  record was never reconstructed, NOT that it does not exist. Do not invent
  values for an entity read_world_state could not find.
- search_transitions(tool, query, entity_id, include_errors, limit) to find
  real recorded examples of how this tool behaves. When entity_id is given,
  each result is labeled entity_matched: true only if that result concerns
  the SAME entity, false if it only shows the tool's general shape for a
  DIFFERENT entity. A false match is never grounds to state that entity's
  own specific values.
- inspect_grounding_history(entity_kind, entity_id) to see this episode's own
  earlier grounding calls about an entity, so you don't repeat one.

Rules:
- Ground every field in what your tools actually returned, for the SPECIFIC
  entity ACTION concerns. A search result with entity_matched: false tells
  you the tool's shape, never a specific value for a different entity. Cite
  only evidence_ids your own search_transitions calls returned.
- If read_world_state cannot find the entity ACTION asks about, and search_transitions
  finds no evidence naming that exact entity, you do not have enough to answer --
  set abstain=true and support="none". A plausible invention is worse than no answer.
- Substitute entities from what read_world_state told you. An example from
  search_transitions tells you the SHAPE of the answer, not its values for a
  different entity.
- A read reports; it never changes state. Return an empty state_delta for one.
- Call finish only once you have gathered what you need.

OUTPUT SHAPE -- this is a strict contract, not a description to paraphrase.

`call_outcomes` is a list with exactly one entry per call id shown in ACTION,
using that exact call id. Every entry has this shape, with no other keys:
    {"call_id": "<exactly the id from ACTION>", "executed": true,
     "observation": <the tool's result for this call, any JSON value>,
     "error": false, "state_delta": [...], "events": [...],
     "evidence_ids": ["<ids from your own search_transitions results>"]}
Do not put business fields (like "reservation_id" or "status") directly on a
call_outcome -- they belong inside its "observation" or "state_delta".

If ACTION named exactly one call and you are not using call_outcomes for it,
you may instead answer with the top-level `observation`/`state_delta`/`events`
fields directly -- but never fill in both call_outcomes and the top-level
fields for the same action."""


def render_agentic_action(context: AWMRuntimeContext) -> str:
    from bandits.diagnose.world import render_action

    return render_action(context.candidate_calls, context.action_content)


def render_agentic_state_summary(context: AWMRuntimeContext) -> str:
    """A brief summary, not the full ledger: the point of the agentic AWM is
    that it fetches specifics through read_world_state itself rather than
    being handed everything up front -- a full render here would make the
    tools redundant."""
    entities = _entities_from_calls(context.candidate_calls)
    if not entities:
        return "(no entities named by the current action)"
    return "known entities this action concerns: " + ", ".join(
        f"{e.kind}:{e.id}" for e in entities
    )


def _summarize_lm_entry(
    entry: dict[str, Any], stage: str, max_tokens: int | None = None
) -> dict[str, Any]:
    """One dspy history record, reduced to what diagnosing a protocol failure
    needs. Deliberately not the whole record: `messages` carries the full
    rendered prompt (the corpus history, every tool result so far) and would
    dwarf the result file while duplicating what the audit trail already has.

    `text_len` vs `reasoning_len` is the discriminating pair. A repetition
    loop shows empty text beside a very long reasoning_content; a genuinely
    truncated near-complete answer shows substantial text. Both merely look
    like "unparseable response" from outside.
    """
    usage = entry.get("usage") or {}
    outputs = entry.get("outputs") or []
    text = ""
    reasoning = ""
    if outputs:
        first = outputs[0]
        if isinstance(first, dict):
            text = first.get("text") or ""
            reasoning = first.get("reasoning_content") or ""
        elif isinstance(first, str):
            text = first
    return {
        "stage": stage,
        "text_len": len(text),
        "reasoning_len": len(reasoning),
        "text_head": text[:400],
        "reasoning_tail": reasoning[-600:],
        "max_tokens": max_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cost": entry.get("cost"),
    }


def _recent_lm_history(*stage_lms: Any) -> tuple[dict[str, Any], ...]:
    """Per-stage LM records for the episode just run.

    Each ReAct stage holds its own LM copy (they carry different token
    budgets), so each keeps its own history list; reading only the base LM
    would come back empty. Best-effort throughout: diagnostics must never
    fail a run that otherwise succeeded.
    """
    records: list[dict[str, Any]] = []
    for stage, lm in stage_lms:
        try:
            # The configured ceiling comes from the LM's own kwargs: dspy's
            # history records per-call kwargs, which do not carry it, so
            # reading it from the entry reports null for every record and
            # hides exactly which stage hit its cap.
            budget = (getattr(lm, "kwargs", {}) or {}).get("max_tokens")
            for entry in getattr(lm, "history", []) or []:
                records.append(_summarize_lm_entry(entry, stage, budget))
        except Exception:  # noqa: BLE001 -- diagnostics never fail the run
            continue
    return tuple(records)


def _render_trajectory(trajectory: dict[str, Any]) -> str:
    """The loop's own log, rendered for the model.

    A plain readable transcript rather than a typed field: the trajectory is
    heterogeneous (thoughts, tool names, argument dicts, arbitrary tool
    payloads), and giving it a schema would force every tool's return shape
    into one contract for no benefit to the model reading it.
    """
    if not trajectory:
        return "(nothing yet)"
    lines = []
    for key, value in trajectory.items():
        rendered = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _coerce_tool_args(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict, whatever the model emitted.

    Models return this field as a dict, as a JSON string, or as nothing.
    A non-dict is treated as no arguments rather than raising: the call then
    fails on its own missing-argument error, which goes back to the model as
    an observation it can act on, instead of ending the episode.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_select_signature(transition_signature: type, tools: Sequence[Callable[..., Any]]) -> type:
    """The tool-selection stage: pick one tool, or finish.

    Its instructions name the tools explicitly, including `finish`, since
    this stage's entire job is choosing among them -- and failing to emit
    `finish` in the expected shape is exactly what lost episodes here.
    """
    import dspy

    catalogue = "\n".join(
        f"- {tool.__name__}: {(tool.__doc__ or '').strip().splitlines()[0] if tool.__doc__ else ''}"
        for tool in tools
    )

    class _Select(dspy.Signature):
        state_summary: str = dspy.InputField()
        history: str = dspy.InputField()
        action: str = dspy.InputField()
        trajectory: str = dspy.InputField(desc="what you have already done this episode")
        next_thought: str = dspy.OutputField(desc="one or two sentences, not an essay")
        next_tool_name: str = dspy.OutputField(desc="exactly one tool name, or 'finish'")
        next_tool_args: dict = dspy.OutputField(desc="arguments for that tool; {} for finish")

    _Select.__doc__ = (
        f"{transition_signature.__doc__}\n\n"
        "You are gathering grounding before answering. Choose ONE tool to call next:\n"
        f"{catalogue}\n- finish: stop gathering and produce the answer.\n\n"
        "Keep next_thought to one or two sentences. When you have enough "
        "grounding -- or when further calls would add nothing -- set "
        "next_tool_name to 'finish' and next_tool_args to {}. Do not explain "
        "at length before finishing; the answer itself is produced separately."
    )
    return _Select


def _build_extract_signature(transition_signature: type) -> type:
    """The extraction stage: the full transition contract, given the
    trajectory. Built from the transition signature's own output fields so
    the contract stays defined in exactly one place."""
    import dspy

    fields = {
        "state_summary": (str, dspy.InputField()),
        "history": (str, dspy.InputField()),
        "action": (str, dspy.InputField()),
        "trajectory": (str, dspy.InputField(desc="the grounding you gathered")),
    }
    for name, field in transition_signature.output_fields.items():
        fields[name] = (field.annotation, dspy.OutputField(desc=field.json_schema_extra.get("desc", "")))
    return dspy.Signature(fields, transition_signature.__doc__)


def build_agentic_tool_world_predictor(
    *,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 6000,
    select_max_tokens: int = 800,
    extract_max_tokens: int = 3000,
    max_grounding_calls: int = MAX_GROUNDING_CALLS_DEFAULT,
    max_iters: int = 8,
    max_select_retries: int = 1,
    on_grounding_call: Callable[[AWMToolCall], None] | None = None,
) -> AgenticToolWorldPredictor:
    """A dspy.ReAct-driven tool world: same output contract as the fixed-RAG
    predictor (world.py's build_tool_world_predictor), but the model gathers
    its own grounding through the four read-only tools instead of being
    handed one fixed retrieval result."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        from bandits.diagnose.world import WorldModelError

        raise WorldModelError(
            "the grounded world model needs the 'diagnose' extra: uv sync --extra diagnose"
        ) from exc

    from bandits.diagnose.world import ProposedCallOutcome, StateDelta, WorldModelError
    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
    )

    class _AgenticTransition(dspy.Signature):
        state_summary: str = dspy.InputField(desc="which entities the current action concerns")
        history: str = dspy.InputField(desc="the conversation so far")
        action: str = dspy.InputField(desc="the call the agent just made, with its call_id")
        observation: dict = dspy.OutputField(
            desc="only for the single-call aggregate form; leave {} when using call_outcomes"
        )
        call_outcomes: list[ProposedCallOutcome] = dspy.OutputField(
            desc="exactly one entry per call_id shown in ACTION, using that exact call_id; "
            "leave empty [] only when answering via the aggregate observation/state_delta form"
        )
        state_delta: list[StateDelta] = dspy.OutputField(
            desc="top-level aggregate form only; leave [] when using call_outcomes"
        )
        events: list[dict] = dspy.OutputField()
        terminal: bool = dspy.OutputField()
        support: str = dspy.OutputField(desc="high, medium, low, or none")
        evidence_ids: list[str] = dspy.OutputField()
        abstain: bool = dspy.OutputField()
        abstain_reason: str = dspy.OutputField()

    # This module uses `from __future__ import annotations`, so the field
    # annotations above are stored as strings. dspy.Predict tolerates that,
    # but dspy.ReAct rebuilds a fallback signature from `signature.*_fields`
    # and hands each stored annotation back to make_signature(), which
    # rejects a ForwardRef outright ("Field types must be types"). Resolving
    # them once here, against this function's locals (where
    # ProposedCallOutcome/StateDelta are imported), makes the fields carry
    # real types before ReAct ever copies them.
    for _field_name, _field in _AgenticTransition.model_fields.items():
        if isinstance(_field.annotation, ForwardRef):
            _field.annotation = eval(  # noqa: S307 -- our own annotation strings
                _field.annotation.__forward_arg__,
                {**globals(), "ProposedCallOutcome": ProposedCallOutcome, "StateDelta": StateDelta},
            )
    _AgenticTransition.model_rebuild(force=True)

    def predict(*, instruction: str, context: AWMRuntimeContext) -> tuple[Any, AWMExecutionTrace]:
        tools, audit, rejections = build_grounding_tools(
            context, max_calls=max_grounding_calls, on_call=on_grounding_call
        )
        _AgenticTransition.__doc__ = instruction

        # A small controlled loop rather than stock dspy.ReAct. Three
        # behaviours ReAct does not offer, each one observed losing a paid
        # episode on this model:
        #
        #  - ReAct.forward catches ValueError around tool selection, but the
        #    adapter raises AdapterParseError, which propagates and kills the
        #    episode outright. Here an unparseable selection ends the
        #    gathering loop and still runs extraction, so grounding already
        #    paid for is not thrown away at the last step.
        #  - the selection and extraction stages are separately budgeted and
        #    separately labelled, so which stage hit its ceiling is readable
        #    afterward instead of inferred.
        #  - tool errors (BudgetExceeded included) become observations fed
        #    back to the model, matching ReAct, but the audit trail stays the
        #    authority on what was actually called.
        select_lm = language_model.copy(max_tokens=select_max_tokens)
        extract_lm = language_model.copy(max_tokens=extract_max_tokens)
        select_signature = _build_select_signature(_AgenticTransition, tools)
        extract_signature = _build_extract_signature(_AgenticTransition)
        select = dspy.Predict(select_signature)
        extract = dspy.Predict(extract_signature)
        by_name = {tool.__name__: tool for tool in tools}

        input_args = {
            "state_summary": render_agentic_state_summary(context),
            "history": context.history_text,
            "action": render_agentic_action(context),
        }
        trajectory: dict[str, Any] = {}
        controller_error: str | None = None
        consecutive_failures = 0

        for idx in range(max_iters):
            try:
                with dspy.context(lm=select_lm):
                    step = select(trajectory=_render_trajectory(trajectory), **input_args)
            except Exception as exc:  # noqa: BLE001 -- see comment above
                # Selection failed (unparseable, truncated, context
                # exceeded). The episode is NOT abandoned: whatever grounding
                # already succeeded still feeds extraction below.
                #
                # Retried once, then given up on. A reasoning model that
                # deliberates past the selection ceiling fails the same way
                # every attempt -- the prompt and trajectory are identical and
                # the temperature is 0 -- so further attempts buy nothing and
                # each costs a full-ceiling call. Observed: three consecutive
                # 800-token failures before the loop ended, all reasoning
                # toward the same finish it never emitted.
                consecutive_failures += 1
                controller_error = f"{type(exc).__name__}: {exc}"
                if consecutive_failures > max_select_retries:
                    break
                continue

            consecutive_failures = 0
            tool_name = (getattr(step, "next_tool_name", "") or "").strip()
            arguments = _coerce_tool_args(getattr(step, "next_tool_args", None))
            trajectory[f"thought_{idx}"] = getattr(step, "next_thought", "")
            trajectory[f"tool_name_{idx}"] = tool_name
            trajectory[f"tool_args_{idx}"] = arguments

            if tool_name == "finish" or not tool_name:
                break
            if tool_name not in by_name:
                trajectory[f"observation_{idx}"] = (
                    f"Unknown tool {tool_name!r}. Available: "
                    f"{', '.join(sorted(by_name))}, finish."
                )
                continue
            try:
                trajectory[f"observation_{idx}"] = by_name[tool_name](**arguments)
            except Exception as exc:  # noqa: BLE001 -- tool errors inform the model
                trajectory[f"observation_{idx}"] = f"Execution error in {tool_name}: {exc}"

        # Extraction always runs, even after a failed selection: grounding
        # calls are the expensive part and they have already been made.
        try:
            with dspy.context(lm=extract_lm):
                raw = extract(trajectory=_render_trajectory(trajectory), **input_args)
        except Exception as exc:  # noqa: BLE001 -- reported, never raised
            raw = None
            controller_error = (
                f"{controller_error}; extraction: {type(exc).__name__}: {exc}"
                if controller_error
                else f"extraction: {type(exc).__name__}: {exc}"
            )

        declared = set(_AgenticTransition.output_fields)
        raw_prediction = raw.toDict() if hasattr(raw, "toDict") else None
        if raw_prediction is not None:
            raw = {k: v for k, v in raw_prediction.items() if k in declared}
        trace = AWMExecutionTrace(
            grounding_calls=tuple(audit),
            exhausted_budget=len(audit) >= max_grounding_calls,
            budget_rejection_attempted=bool(rejections),
            raw_prediction=raw_prediction,
            lm_history=_recent_lm_history(("select", select_lm), ("extract", extract_lm)),
            trajectory=dict(trajectory),
            controller_error=controller_error,
        )
        return raw, trace

    return predict


class ClaimGrounding(Contract):
    """What backs one specific claimed field, delta, or event.

    The unit I39 (and the earlier entity-level fix) both missed: an entity
    being "grounded" at all does not mean every field predicted about it was.
    Reading ``reservation.Q69X3R.status`` and then also claiming a refund
    amount, a payment mutation, and a flight change is one grounded entity
    and three fabricated claims -- entity-level attribution alone reports
    this as fully supported. Each claim here is judged on its own evidence.
    """

    call_id: str | None = None
    path: str
    predicted_value: Any = None
    kind: GroundingKind = GroundingKind.UNSUPPORTED
    """
    EXACT_ENTITY_FACT: this exact (path, value) was actually read via
        read_world_state, or matched an exact-entity search result carrying
        the same field. The strongest claim: the value itself is verified,
        not merely the path name.
    DERIVABLE_FROM_STATE: the *path* was read for this entity (so its
        pre-mutation value is known) but the claimed *value* differs -- a
        legitimate mutation claim (e.g. status: confirmed -> cancelled) --
        AND this exact tool has behavioral evidence (a search_transitions
        result for the same tool, matched or not: shape evidence is enough
        to license *that* a field like this changes, never a specific new
        value the shape evidence didn't itself show; see note below).
    BEHAVIORAL_ANALOGY: no pre-state anchor for this path at all, but the
        tool has some behavioral evidence -- shape only, never a value
        claim. Downgrades a proposal, never licenses a mutation/event.
    UNSUPPORTED: no pre-state anchor and no behavioral evidence. A wholly
        invented field (refund_amount when nothing about payments was ever
        read) -- this is the exact attack the entity-level check missed.
    """
    evidence_ids: tuple[str, ...] = ()


def assess_proposal_claims(
    proposal: Any,
    *,
    context: AWMRuntimeContext,
    trace: AWMExecutionTrace,
) -> tuple[ClaimGrounding, ...]:
    """Classify every claimed observation field, state-delta, and event
    against what this episode's grounding calls actually established for the
    specific call it belongs to -- never pooled across the whole proposal or
    the whole entity.

    Batches stay separate throughout: each ``ProposedCallOutcome`` (or the
    single aggregate form) is matched to its own ``ActionCall``, whose own
    target entity's own ``state_values_read``/``exact_entity_evidence_ids``
    are what its claims are checked against -- call A's evidence can never
    license call B's claims.
    """
    call_by_id = {c.call_id: c for c in context.candidate_calls if c.call_id}
    single_call = context.candidate_calls[0] if len(context.candidate_calls) == 1 else None

    # Per-entity: exact values actually established (full dotted path ->
    # value), from either source -- a read_world_state hit, or an
    # entity_matched=true search_transitions result's own recorded fields.
    # read_world_state wins on conflict (this rollout's own history over a
    # possibly-stale retrieved snapshot, same principle as grounding.py's
    # I39 fix). Full paths throughout: a leaf-only key ("city") would let
    # billing.address.city and shipping.address.city license each other.
    values_by_entity: dict[tuple[str, str], dict[str, Any]] = {}
    exact_search_evidence_ids_by_entity: dict[tuple[str, str], tuple[str, ...]] = {}
    for call in trace.grounding_calls:
        if call.tool == "search_transitions" and call.entity is not None:
            if call.exact_entity_values:
                values_by_entity.setdefault(call.entity, {})
                for path, value in call.exact_entity_values.items():
                    values_by_entity[call.entity].setdefault(path, value)
            exact_search_evidence_ids_by_entity.setdefault(call.entity, ())
            if call.exact_entity_evidence_ids:
                exact_search_evidence_ids_by_entity[call.entity] = (
                    exact_search_evidence_ids_by_entity[call.entity] + call.exact_entity_evidence_ids
                )
    for call in trace.grounding_calls:
        if call.tool == "read_world_state" and call.entity is not None and call.state_values_read:
            # Second pass, after search results are seeded: read_world_state
            # always overwrites, giving it priority on any conflict.
            values_by_entity.setdefault(call.entity, {}).update(call.state_values_read)

    # Per tool (not pooled across tools -- fixes the bug where searching
    # get_user_details licensed a cancel_reservation mutation): every path ->
    # set of values search_transitions results for THAT tool actually showed,
    # matched or not. Licenses a mutation's specific new value, never merely
    # "the tool has some evidence somewhere."
    observed_values_by_tool: dict[str, dict[str, tuple[Any, ...]]] = {}
    evidence_ids_by_tool: dict[str, tuple[str, ...]] = {}
    for call in trace.grounding_calls:
        if call.tool != "search_transitions":
            continue
        queried_tool = call.arguments.get("tool")
        if not isinstance(queried_tool, str) or not queried_tool:
            continue
        if call.observed_tool_values:
            bucket = observed_values_by_tool.setdefault(queried_tool, {})
            for path, values in call.observed_tool_values.items():
                bucket[path] = tuple(dict.fromkeys(bucket.get(path, ()) + values))
        if call.evidence_ids:
            evidence_ids_by_tool[queried_tool] = evidence_ids_by_tool.get(queried_tool, ()) + call.evidence_ids

    def _target_entity(call: ActionCall | None) -> tuple[str, str] | None:
        if call is None:
            return None
        entities = _entities_from_calls((call,))
        return (entities[0].kind, entities[0].id) if entities else None

    def _tool_behavior_supported(tool: str) -> bool:
        return bool(evidence_ids_by_tool.get(tool))

    def _sub_path(entity: tuple[str, str] | None, tool: str, path: str) -> str:
        """Strips a "<some_tool>.<entity_id>." canonical prefix, to compare
        against observed_tool_values -- entity-agnostic tool-shape evidence,
        which is deliberately keyed by bare sub-path (see search_transitions'
        own comment: canonicalizing it to one entity's id would make it
        useless for a not-yet-grounded or different entity's mutation).

        Strips whichever tool the PATH is actually namespaced under, not the
        claim's own candidate-call tool: a mutation's state_delta targets a
        field whose pre-state value was namespaced by the tool that first
        reported it (e.g. get_reservation_details.Q69X3R.status), which is
        frequently a different tool from the one performing the mutation
        (cancel_reservation) -- this matches the real corpus's own read-side
        convention (get_user_details.<id>.<field>), confirmed against the
        pinned tau2 artifact.
        """
        if entity is None:
            return path
        suffix = f".{entity[1]}."
        marker = path.find(suffix)
        if marker == -1:
            return path
        return path[marker + len(suffix) :]

    def _tool_has_observed_this_value(tool: str, sub_path: str, value: Any) -> bool:
        """Whether search_transitions, queried for THIS exact tool, ever
        showed this exact sub-path taking this exact value -- what licenses a
        mutation's specific new value, distinct from merely proving the
        tool has some behavioral evidence at all."""
        observed = observed_values_by_tool.get(tool, {}).get(sub_path, ())
        return _hashable(value) in observed

    def _classify(call_id: str | None, tool: str, path: str, value: Any) -> ClaimGrounding:
        action_call = call_by_id.get(call_id) or single_call
        entity = _target_entity(action_call)
        known = values_by_entity.get(entity, {}) if entity else {}

        exact_ids = exact_search_evidence_ids_by_entity.get(entity, ()) if entity else ()
        tool_ids = evidence_ids_by_tool.get(tool, ())

        # An observation field's own path is never canonically prefixed
        # (a tool's actual JSON response has no "<tool>.<entity_id>." on its
        # keys -- that prefix only exists on ScenarioState/StateDelta paths).
        # `known` is keyed by full canonical path, so an observation claim
        # ("status") must also be checked against `known`'s bare sub-path
        # form, or every observation field would always miss even when the
        # exact fact was read.
        matched_known_path = path if path in known else None
        if matched_known_path is None and entity is not None:
            matched_known_path = next(
                (k for k in known if _sub_path(entity, tool, k) == path), None
            )
        known_value = known.get(matched_known_path) if matched_known_path else None

        if matched_known_path is not None and known_value == value:
            return ClaimGrounding(
                call_id=call_id, path=path, predicted_value=value,
                kind=GroundingKind.EXACT_ENTITY_FACT, evidence_ids=exact_ids,
            )
        if matched_known_path is not None and known_value != value:
            # The path itself is a known pre-state fact -- this is a mutation
            # claim, not an invention out of nothing. It is only DERIVABLE if
            # search_transitions, queried for THIS exact tool, has actually
            # shown this exact new value at this exact sub-path -- not merely
            # that the tool has *some* unrelated evidence. Without this, a
            # known origin="PHL" plus any cancel_reservation evidence would
            # license claiming origin="Mars".
            if _tool_has_observed_this_value(tool, _sub_path(entity, tool, matched_known_path), value):
                return ClaimGrounding(
                    call_id=call_id, path=path, predicted_value=value,
                    kind=GroundingKind.DERIVABLE_FROM_STATE, evidence_ids=tool_ids,
                )
            return ClaimGrounding(
                call_id=call_id, path=path, predicted_value=value, kind=GroundingKind.UNSUPPORTED,
            )
        # No pre-state anchor for this path at all -- behavioral evidence for
        # THIS tool specifically proves shape, never licenses a brand-new
        # field/value.
        if _tool_behavior_supported(tool):
            return ClaimGrounding(
                call_id=call_id, path=path, predicted_value=value, kind=GroundingKind.BEHAVIORAL_ANALOGY,
            )
        return ClaimGrounding(
            call_id=call_id, path=path, predicted_value=value, kind=GroundingKind.UNSUPPORTED,
        )

    def _classify_all(call_id: str | None, tool: str, payload: Any, deltas: Sequence[Any], events: Sequence[dict[str, Any]]) -> list[ClaimGrounding]:
        found: list[ClaimGrounding] = []
        for path, value in _flatten_paths(payload).items():
            found.append(_classify(call_id, tool, path, value))
        for delta in deltas:
            found.append(_classify(call_id, tool, delta.path, delta.new_value))
        for event_index, event in enumerate(events):
            for field_name, field_value in _flatten_paths(event).items():
                # Events have no ledger "path" of their own to check against
                # known pre-state (an event is a new fact, e.g. "RefundIssued
                # occurred", not a mutation of an existing field) -- so an
                # event field can only ever be EXACT_ENTITY_FACT (if it
                # happens to echo a known value), BEHAVIORAL_ANALOGY (tool
                # evidence exists), or UNSUPPORTED. It is never
                # DERIVABLE_FROM_STATE, since there is no pre-state value to
                # derive it from.
                event_path = f"events[{event_index}].{field_name}"
                semantic = _classify(call_id, tool, field_name, field_value)
                kind = semantic.kind
                if kind is GroundingKind.BEHAVIORAL_ANALOGY and _tool_has_observed_this_value(
                    tool, field_name, field_value
                ):
                    kind = GroundingKind.DERIVABLE_FROM_STATE
                claim = ClaimGrounding(
                    call_id=call_id,
                    path=event_path,
                    predicted_value=field_value,
                    kind=kind,
                    evidence_ids=semantic.evidence_ids,
                )
                found.append(claim)
        return found

    claims: list[ClaimGrounding] = []
    if proposal.call_outcomes:
        for outcome in proposal.call_outcomes:
            action_call = call_by_id.get(outcome.call_id)
            tool = action_call.tool if action_call else ""
            claims.extend(
                _classify_all(outcome.call_id, tool, outcome.observation, outcome.state_delta, outcome.events)
            )
    else:
        call_id = single_call.call_id if single_call else None
        tool = single_call.tool if single_call else ""
        claims.extend(_classify_all(call_id, tool, proposal.observation, proposal.state_delta, proposal.events))

    return tuple(claims)


_KIND_ORDER = [
    GroundingKind.UNSUPPORTED,
    GroundingKind.BEHAVIORAL_ANALOGY,
    GroundingKind.DERIVABLE_FROM_STATE,
    GroundingKind.EXACT_ENTITY_FACT,
]
_KIND_TO_SUPPORT = {
    GroundingKind.UNSUPPORTED: SupportLevel.NONE,
    GroundingKind.BEHAVIORAL_ANALOGY: SupportLevel.LOW,
    GroundingKind.DERIVABLE_FROM_STATE: SupportLevel.MEDIUM,
    GroundingKind.EXACT_ENTITY_FACT: SupportLevel.HIGH,
}


def step_agentic_tool_world(
    predict: AgenticToolWorldPredictor,
    *,
    context: AWMRuntimeContext,
    instruction: str = AGENTIC_TOOL_WORLD_INSTRUCTION,
) -> tuple[Any, AWMExecutionTrace]:
    """Runs the agentic AWM and caps its self-reported support the same way
    step_tool_world does for fixed RAG -- except here "what was found" is
    read from the audit trail's own search_transitions results, since the
    agentic AWM chooses how much to search rather than being handed one
    retrieval result up front. A model asked to rate its own grounding rates
    it generously; the number that gates committing has to come from what the
    tools actually returned, not from the model's own claim.
    """
    from bandits.diagnose.world import ProposedTransition, _coerce, _CoerceFailure

    raw, trace = predict(instruction=instruction, context=context)
    if trace.exhausted_budget and raw is None:
        return (
            ProposedTransition(
                abstain=True,
                abstain_reason="grounding-call budget exhausted before the AWM reached finish",
                support=SupportLevel.NONE,
            ),
            trace,
        )

    coerced = _coerce(raw, ProposedTransition)
    if isinstance(coerced, _CoerceFailure):
        return (
            ProposedTransition(
                output_invalid=True,
                output_invalid_errors=coerced.errors,
                support=SupportLevel.NONE,
            ),
            trace,
        )
    if coerced is None:
        return (
            ProposedTransition(
                abstain=True,
                abstain_reason="the world model returned nothing that fit the transition contract",
                support=SupportLevel.NONE,
            ),
            trace,
        )
    proposal = coerced

    # Claim-level, not entity-level. Entity-level grounding ("was Q69X3R read
    # at all") let one discovered field (status) license every other claimed
    # field (refund amount, payment mutation, flight changes) as if the whole
    # entity's record had been verified -- the exact attack this replaces.
    # Every observation field and every state-delta/event's new value is
    # classified on its own evidence via assess_proposal_claims, and:
    #
    #   - a claim behind a committed mutation or event is UNSUPPORTED ->
    #     the whole proposal is rejected (abstained), because state_delta and
    #     events are exactly what the validator commits to the ledger next --
    #     an unsupported one must never reach that gate at all, "capped
    #     support" is not a strong enough guard for something about to be
    #     written down as having happened;
    #   - a claim behind an ordinary observation field is UNSUPPORTED ->
    #     that field alone is nulled (explicit unknown) rather than passed
    #     through with its fabricated value, since a read is not a commitment
    #     the same way a mutation is;
    #   - the proposal's support is capped to the WEAKEST surviving claim
    #     across all calls, never rounded up by any one strong claim.
    claims = assess_proposal_claims(proposal, context=context, trace=trace)
    # Keyed by (call_id, path), never path alone. A batch where call A
    # mutates "status" and call B merely *observes* "status" must not let A's
    # delta make B's observation look like a mutation (or vice versa) -- the
    # two are different claims about different calls that happen to share a
    # field name.
    single_call_id = (
        context.candidate_calls[0].call_id if len(context.candidate_calls) == 1 else None
    )
    mutation_keys = {
        (outcome.call_id, d.path)
        for outcome in proposal.call_outcomes
        for d in outcome.state_delta
    } | {(single_call_id, d.path) for d in proposal.state_delta}

    # A mutation/event claim is rejected on anything weaker than
    # DERIVABLE_FROM_STATE -- BEHAVIORAL_ANALOGY included. Shape evidence
    # proves the tool does *some* kind of write; it never licenses a
    # specific brand-new field or value (refund_amount, a payment mutation,
    # a flight change with no pre-state anchor at all). Only
    # UNSUPPORTED/BEHAVIORAL_ANALOGY are ever downgrades for an ordinary
    # observation field -- for a committed mutation/event, both are a reject.
    #
    # Event claims are identified by their own synthetic path
    # ("events[N].field", from assess_proposal_claims) rather than a coarse
    # "any events exist" flag -- each event field is classified on its own
    # evidence, so an event-only fabrication (no accompanying state_delta) is
    # caught by its own claim, not merely by piggybacking on an unrelated
    # weak delta claim.
    weak_mutation_claims = [
        c
        for c in claims
        if c.kind in (GroundingKind.UNSUPPORTED, GroundingKind.BEHAVIORAL_ANALOGY)
        and ((c.call_id, c.path) in mutation_keys or c.path.startswith("events["))
    ]
    if weak_mutation_claims:
        offending = tuple(sorted({c.path for c in weak_mutation_claims}))
        return (
            ProposedTransition(
                abstain=True,
                abstain_reason=(
                    f"proposed state change(s)/event(s) touching insufficiently grounded "
                    f"field(s) [{', '.join(offending)}] -- rejected rather than committed"
                ),
                support=SupportLevel.NONE,
            ),
            trace.replace(rejected_unsupported_claim_paths=offending),
        )

    unsupported_observation_keys = {
        (c.call_id, c.path)
        for c in claims
        if c.kind is GroundingKind.UNSUPPORTED and (c.call_id, c.path) not in mutation_keys
    }
    if unsupported_observation_keys:
        proposal = _null_unsupported_observation_fields(
            proposal, unsupported_observation_keys, single_call_id=single_call_id
        )

    # Support is computed over the SURVIVING claims only -- a nulled
    # observation field is no longer something the proposal asserts, so it
    # must not still drag the weakest-claim computation down to NONE and get
    # the partial answer rejected by validate_transition's support floor
    # anyway. That was the bug: nulling looked like it produced an accepted
    # partial result, but support was still computed against the pre-nulling
    # claim list, so the proposal was silently rejected downstream regardless.
    surviving_claims = [c for c in claims if (c.call_id, c.path) not in unsupported_observation_keys]
    if surviving_claims:
        weakest_kind = min(surviving_claims, key=lambda c: _KIND_ORDER.index(c.kind)).kind
        found_support = _KIND_TO_SUPPORT[weakest_kind]
    else:
        # No claims survive at all (e.g. a pure error/terminal response with
        # no observation fields or deltas, or every observation field was
        # nulled) -- fall back to the episode's coarser behavioral signal.
        found_support = SupportLevel.MEDIUM if trace.all_evidence_ids else SupportLevel.NONE

    order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, SupportLevel.HIGH]
    capped = min(proposal.support, found_support, key=order.index)
    return proposal.replace(support=capped), trace


def _null_unsupported_observation_fields(
    proposal: Any,
    unsupported_keys: set[tuple[str | None, str]],
    *,
    single_call_id: str | None = None,
) -> Any:
    """Replace a fabricated observation field's value with None (explicit
    unknown) rather than let it pass through unflagged. Only observation
    fields are touched here -- state_delta/events with any unsupported claim
    are handled by the caller as a full rejection, never partially nulled,
    since a partially-nulled *mutation* is not a meaningful thing to commit.

    Walks lists as well as dicts, emitting exactly the paths
    ``_flatten_paths`` produces ("flights[0].destination"). A dict-only walk
    silently skipped every list-nested field: the fabricated value stayed in
    the response while its claim was still dropped from the surviving-claims
    support computation -- so the fabrication survived *and* raised the
    proposal's support. Scalars nested in lists are nulled in place rather
    than dropped, keeping list length and element positions intact (the
    ".length" claim stays true, and sibling indices keep their meaning).

    Keyed by ``(call_id, path)``: in a batch, call A's unsupported "status"
    must not null call B's exact "status".
    """
    from bandits.diagnose.world import ProposedCallOutcome

    def _null_payload(payload: Any, call_id: str | None, prefix: str = "") -> Any:
        if isinstance(payload, dict):
            result = {}
            for key, value in payload.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if (call_id, path) in unsupported_keys:
                    result[key] = None
                else:
                    result[key] = _null_payload(value, call_id, path)
            return result
        if isinstance(payload, list):
            items = []
            for index, value in enumerate(payload):
                path = f"{prefix}[{index}]" if prefix else f"[{index}]"
                if (call_id, path) in unsupported_keys:
                    items.append(None)
                else:
                    items.append(_null_payload(value, call_id, path))
            return items
        return payload

    if proposal.call_outcomes:
        new_outcomes = tuple(
            outcome.replace(observation=_null_payload(outcome.observation, outcome.call_id))
            if isinstance(outcome, ProposedCallOutcome)
            else outcome
            for outcome in proposal.call_outcomes
        )
        return proposal.replace(call_outcomes=new_outcomes)
    return proposal.replace(observation=_null_payload(proposal.observation, single_call_id))
