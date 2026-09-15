"""The grounded world model: two roles, a strict contract, and abstention.

The environment holds two policies and they must not be one. A tool-world model
answers a call with an observation and a state delta; a user policy answers an
assistant message with the next thing the customer says. Merging them produces a
simulator that can invent a helpful customer to rescue a failing candidate,
which is the failure mode that made tau2's own hidden-state errors invisible.

    candidate --- tool call ------> tool world  --> observation + delta
              \\-- message --------> user policy --> next user message

Neither ever does the other's job: the tool world never speaks as the user, and
the user policy never mutates enterprise state.

Nothing here trains weights. A frozen capable model is called through an
injected predictor, grounded by retrieved real transitions, instructed by a
versioned prompt, and checked by a deterministic validator before anything is
committed. The validator — not the model — owns the ledger, because a simulator
that can write state unchecked can write itself a pass, and an external verifier
reading that state would then accept it.

Model access follows the idiom the rest of this repository uses: a ``Protocol``
the tests inject, with the real ``dspy``/LM import living only inside
``build_*_predictor``. CI needs no credential and no sandbox.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from bandits.diagnose.models import (
    ActionCall,
    HiddenUserProfile,
    ScenarioState,
    StateField,
    SupportLevel,
    ToolEffect,
    ToolEffectCatalog,
    WorldOrigin,
)
from bandits.diagnose.retrieve import RetrievedExample, support_level
from bandits.traces import Contract

PROMPT_VERSION = 1

TOOL_WORLD_INSTRUCTION = """\
You are the ENVIRONMENT, not the agent. You return what a tool call produces.

Rules:
- Return the tool's next observation and the state changes it causes. Never
  decide what the agent should do, never speak as the user, never give advice.
- Ground every field in the retrieved real transitions. They are recorded
  behaviour of this exact system.
- Substitute entities from the CURRENT STATE. An example about reservation ABC
  tells you the SHAPE of the answer, not its values.
- Respect what the state already says. If a reservation is already cancelled,
  the tool says so; do not replay an example that assumed it was confirmed.
- A read reports; it never changes state. Return an empty state_delta for one.
- Enforce the tool's own preconditions, not the agent's policy. If the tool
  would execute a call the agent should not have made, execute it. Judging the
  agent is the verifier's job, and refusing on its behalf hides the mistake.
- If the retrieved evidence does not determine the outcome, ABSTAIN. Say so by
  setting `abstain: true` and `support: "none"`. A plausible invention is
  worse than no answer.

OUTPUT SHAPE -- this is a strict contract, not a description to paraphrase.

`call_outcomes` is a list with exactly one entry per call id shown in ACTION,
using that exact call id. Every entry has this shape, with no other keys:
    {"call_id": "<exactly the id from ACTION>", "executed": true,
     "observation": <the tool's result for this call, any JSON value>,
     "error": false, "state_delta": [...], "events": [...],
     "evidence_ids": ["<ids from EVIDENCE this outcome is grounded in>"]}
Do not put business fields (like "reservation_id" or "status") directly on a
call_outcome -- they belong inside its "observation" or "state_delta".

`state_delta` (both at the top level and inside a call_outcome) is a list of
path operations, one per changed field, each shaped exactly like:
    {"path": "<tool>.<entity_id>.<field>", "old_value": <value before, or null
     if unknown>, "new_value": <value after>}
Never put a whole business object as one state_delta entry. Only include a
field here if the action actually changed it -- an unchanged field is not a
delta, even if it also appears in "observation".

If ACTION named exactly one call and you are not using call_outcomes for it,
you may instead answer with the top-level `observation`/`state_delta`/`events`
fields directly (the aggregate form) -- but never fill in both call_outcomes
and the top-level fields for the same action.

Return only the required structured fields."""

USER_POLICY_INSTRUCTION = """\
You are the CUSTOMER, not the agent and not the system.

Rules:
- Follow the scenario instructions exactly. Generate one message at a time.
- Never invent information the scenario did not give you. Anything not in your
  known information is unknown or unavailable to you.
- Disclose a fact only when the agent has appropriately asked for it. Do not
  volunteer identifiers, reservation numbers, or preferences unprompted.
- Stay in persona. Your goal does not change because the agent struggled.
- You never call tools and never change any system state.
- If you cannot answer in character from what the scenario gave you, ABSTAIN.

Return only the required structured fields."""


class WorldModelError(RuntimeError):
    """The world model could not be built or returned nothing usable."""


class ToolWorldPredictor(Protocol):
    """The one call the tool world makes, so tests need no model."""

    def __call__(
        self, *, instruction: str, state: str, history: str, action: str, evidence: str
    ) -> Any: ...


class UserPolicyPredictor(Protocol):
    def __call__(
        self, *, instruction: str, profile: str, history: str, message: str, evidence: str
    ) -> Any: ...


def prompt_digest(instruction: str, model: str) -> str:
    """Pins wording, model and version onto every artifact they produced."""
    payload = json.dumps(
        {"instruction": instruction, "model": model, "version": PROMPT_VERSION}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class StateDelta(Contract):
    """One proposed change to the ledger, with the value it claims to replace."""

    path: str
    old_value: Any = None
    new_value: Any = None


class ProposedCallOutcome(Contract):
    """The world-model result for one call in an action batch."""

    call_id: str
    executed: bool = True
    observation: Any = None
    error: bool = False
    state_delta: tuple[StateDelta, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def an_unexecuted_call_has_no_effect(self) -> ProposedCallOutcome:
        if not self.executed and (self.state_delta or self.events):
            raise ValueError("an unexecuted call cannot change state or emit events")
        return self


class ProposedTransition(Contract):
    """What the tool world returns, before anything is committed.

    Proposed, not applied. The name is the contract: the model does not own the
    ledger, and every field here is a claim the validator may reject.
    """

    observation: Any = None
    call_outcomes: tuple[ProposedCallOutcome, ...] = ()
    state_delta: tuple[StateDelta, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    terminal: bool = False
    support: SupportLevel = SupportLevel.NONE
    evidence_ids: tuple[str, ...] = ()
    abstain: bool = False
    abstain_reason: str = ""
    output_invalid: bool = False
    """The model attempted an answer that did not fit this contract.

    Distinct from ``abstain``: abstention is the model saying it lacks
    evidence, this is the model saying something else entirely that _coerce
    could not parse into ProposedCallOutcome/StateDelta shapes. Conflating the
    two hides a parser/prompt defect behind a metric that looks epistemic.
    """

    output_invalid_errors: tuple[str, ...] = ()
    """Raw validation errors from the failed parse, for diagnosis."""

    @model_validator(mode="after")
    def abstention_proposes_nothing(self) -> ProposedTransition:
        if self.abstain and (self.call_outcomes or self.state_delta or self.events):
            raise ValueError("an abstaining transition cannot also propose changes")
        return self


class ProposedUserTurn(Contract):
    """What the user policy returns. It can never carry a state change."""

    user_message: str = ""
    disclosed_facts: tuple[str, ...] = ()
    """Which known facts this turn revealed. The disclosure gate reads it."""

    goal_status: str = "continuing"
    terminal: bool = False
    support: SupportLevel = SupportLevel.NONE
    abstain: bool = False
    abstain_reason: str = ""


class ValidationOutcome(Contract):
    """The validator's verdict on a proposed transition."""

    accepted: bool
    rejections: tuple[str, ...] = ()
    committed: tuple[StateField, ...] = ()
    executed_call_ids: tuple[str, ...] = ()
    committed_call_ids: tuple[str, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    output_schema_validation: dict[
        str, Literal["validated", "unavailable", "invalid"]
    ] = Field(default_factory=dict)

    committed_observations: dict[str, Any] = Field(default_factory=dict)
    """The accepted per-call observation, keyed by call_id.

    The one canonical mapping from a submitted call to what the environment
    said it returned -- batch (call_outcomes) and aggregate (single-call
    top-level observation) forms both normalize into this same shape here, so
    a caller (fidelity scoring, a rollout) never has to re-derive which model
    field corresponds to which call. A call with no call_id (an unbatched
    single call the source never assigned one to) is keyed by its tool name
    instead, since it is unambiguous only when exactly one such call exists.
    """


def _tool_contract(
    tool_schemas: Sequence[dict[str, Any]], tool: str
) -> dict[str, Any] | None:
    for declared in tool_schemas:
        body = (
            declared.get("function")
            if isinstance(declared.get("function"), dict)
            else declared
        )
        if body.get("name") == tool:
            return body
    return None


def _output_schema(declared: dict[str, Any] | None) -> dict[str, Any] | bool | None:
    if declared is None:
        return None
    schema = declared.get("output_schema")
    if not isinstance(schema, (dict, bool)):
        schema = declared.get("outputSchema")
    return schema if isinstance(schema, (dict, bool)) else None


def validate_instance(
    instance: Any, schema: dict[str, Any] | bool, *, label: str = "output"
) -> str | None:
    """Return a standards-compliant JSON Schema error, or None when valid.

    D72. Candidate arguments and declared tool results are held to the same
    standard and the same library; only the reported label differs. A schema
    the source declared but that is not itself valid JSON Schema is an error
    about the contract, never a silent pass.
    """
    try:
        from jsonschema import SchemaError, ValidationError
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise WorldModelError(
            "schema validation needs the 'diagnose' extra: uv sync --extra diagnose"
        ) from exc

    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema).validate(instance)
    except SchemaError as exc:
        return f"declared {label} schema is invalid: {exc.message}"
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path)
        location = f"{label}.{path}" if path else label
        return f"{location}: {exc.message}"
    return None


def _validate_output(instance: Any, schema: dict[str, Any] | bool) -> str | None:
    return validate_instance(instance, schema, label="output")


def render_evidence(examples: Sequence[RetrievedExample], *, budget: int = 6000) -> str:
    """Retrieved transitions as prompt text, clipped here and only here.

    This is the one place clipping is correct. The index stores structured
    payloads whole precisely so that a fidelity diff can compare fields; what a
    prompt can afford is a rendering concern, and deciding it at extraction time
    would have put truncated JSON into the evidence itself.
    """
    lines: list[str] = []
    spent = 0
    for index, example in enumerate(examples, start=1):
        transition = example.transition
        calls = ", ".join(
            f"{call.tool}({json.dumps(call.arguments, sort_keys=True)})"
            for call in transition.action_calls
        )
        observed = json.dumps(
            [observation.content for observation in transition.observations],
            default=str,
            sort_keys=True,
        )
        block = (
            f"[{index}] id={transition.transition_id} "
            f"why={'/'.join(example.reasons) or 'similar'}\n"
            f"    action: {calls or '(spoke, no call)'}\n"
            f"    observed: {observed[:800]}\n"
        )
        if spent + len(block) > budget:
            lines.append(f"... {len(examples) - index + 1} further examples omitted\n")
            break
        lines.append(block)
        spent += len(block)
    return "".join(lines) or "(no relevant recorded transition was found)"


def render_state(state: ScenarioState, *, limit: int = 80) -> str:
    """The ledger as text, marking which facts are already simulated.

    The origin is shown to the model deliberately: a value the simulator itself
    produced two steps ago is weaker ground than one a real trace revealed, and
    a model told which is which can prefer the recorded fact when they conflict.
    """
    if not state.fields:
        return "(nothing is known about the world yet)"
    rows = []
    for field in state.fields[:limit]:
        mark = "" if field.origin is WorldOrigin.RECORDED else "  [simulated]"
        rows.append(f"  {field.path} = {json.dumps(field.value, default=str)}{mark}")
    if len(state.fields) > limit:
        rows.append(f"  ... {len(state.fields) - limit} further known fields")
    return "\n".join(rows)


def render_action(calls: Sequence[ActionCall], content: Any = None) -> str:
    """Each call rendered with its call_id, when it has one.

    A batched call_outcome must cite the exact call_id it answers (D70/the
    validator's own unanswered-call check), so that id has to be visible here
    -- a predictor asked to echo an id it was never shown can only guess one.
    """
    parts = [
        f"call_id={call.call_id} tool={call.tool} "
        f"arguments={json.dumps(call.arguments, sort_keys=True, default=str)}"
        if call.call_id is not None
        else f"{call.tool}({json.dumps(call.arguments, sort_keys=True, default=str)})"
        for call in calls
    ]
    if content:
        parts.append(f'(said: "{str(content)[:400]}")')
    return "\n".join(parts) or "(the agent spoke without calling a tool)"


def render_profile(profile: HiddenUserProfile) -> str:
    """The customer's private brief. Never rendered into a candidate prompt."""
    return "\n".join(
        part
        for part in (
            f"Why you are calling: {profile.reason_for_call}" if profile.reason_for_call else "",
            f"What you know: {profile.known_info}" if profile.known_info else "",
            f"What you do NOT know: {profile.unknown_info}" if profile.unknown_info else "",
            f"How to behave: {profile.task_instructions}" if profile.task_instructions else "",
            f"Persona: {profile.persona}" if profile.persona else "",
            (
                f"Normalized known facts: {json.dumps(profile.known_facts, sort_keys=True)}"
                if profile.known_facts
                else ""
            ),
            (
                f"Unavailable fact ids: {', '.join(profile.unknown_fact_ids)}"
                if profile.unknown_fact_ids
                else ""
            ),
        )
        if part
    )


def validate_transition(
    proposal: ProposedTransition,
    *,
    calls: Sequence[ActionCall],
    state: ScenarioState,
    step_index: int,
    catalog: ToolEffectCatalog | None = None,
    min_support: SupportLevel = SupportLevel.LOW,
    allowed_evidence_ids: Sequence[str] | None = None,
    tool_schemas: Sequence[dict[str, Any]] = (),
) -> ValidationOutcome:
    """Check a proposal before it reaches the ledger.

    The AWM does not own state. Each rule here closes a way a simulator could
    write itself a pass that a correct verifier would then accept.
    """
    rejections: list[str] = []

    if proposal.output_invalid:
        return ValidationOutcome(
            accepted=False,
            rejections=("the environment's output did not fit the transition contract",)
            + proposal.output_invalid_errors,
        )
    if proposal.abstain:
        return ValidationOutcome(accepted=False, rejections=("the environment abstained",))

    call_by_id = {call.call_id: call for call in calls if call.call_id}
    normalized = proposal
    executed_call_ids: tuple[str, ...] = ()
    committed_call_ids: tuple[str, ...] = ()
    canonical_events: tuple[dict[str, Any], ...] = ()
    committed_observations: dict[str, Any] = {}
    schema_validation: dict[str, str] = {}
    if proposal.call_outcomes:
        outcome_ids = [outcome.call_id for outcome in proposal.call_outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            rejections.append("a call has more than one proposed outcome")
        unknown_ids = set(outcome_ids) - set(call_by_id)
        if unknown_ids:
            rejections.append(f"outcomes name unknown call ids: {sorted(unknown_ids)!r}")
        unanswered = set(call_by_id) - set(outcome_ids)
        if unanswered:
            # D70. Every submitted call is answered, including with
            # executed=False. A call the AWM simply omits is indistinguishable
            # from one it decided not to run, and the omitted call may be the
            # forbidden one the verifier is looking for.
            rejections.append(
                f"submitted calls have no proposed outcome: {sorted(unanswered)!r}"
            )
        if len(calls) > 1 and any(call.call_id is None for call in calls):
            rejections.append("batched calls require call ids for independent outcomes")
        if proposal.state_delta or proposal.events:
            rejections.append("batched call outcomes cannot also use aggregate effects")

        executed_call_ids = tuple(
            outcome.call_id for outcome in proposal.call_outcomes if outcome.executed
        )
        committed_call_ids = tuple(
            outcome.call_id
            for outcome in proposal.call_outcomes
            if outcome.executed and (outcome.state_delta or outcome.events)
        )
        canonical_events_list: list[dict[str, Any]] = []
        for call_outcome in proposal.call_outcomes:
            if (
                call_outcome.state_delta or call_outcome.events
            ) and not call_outcome.evidence_ids:
                rejections.append(
                    f"call {call_outcome.call_id} proposed an effect with no supporting evidence"
                )
            for event in call_outcome.events:
                supplied = event.get("_call_id")
                if supplied is not None and supplied != call_outcome.call_id:
                    rejections.append(
                        f"event for call {call_outcome.call_id} spoofed _call_id {supplied}"
                    )
                    continue
                canonical_events_list.append({**event, "_call_id": call_outcome.call_id})
        canonical_events = tuple(canonical_events_list)
        normalized = proposal.replace(
            observation={outcome.call_id: outcome.observation for outcome in proposal.call_outcomes},
            state_delta=tuple(
                delta for outcome in proposal.call_outcomes for delta in outcome.state_delta
            ),
            events=canonical_events,
            evidence_ids=tuple(
                dict.fromkeys(
                    evidence_id
                    for outcome in proposal.call_outcomes
                    for evidence_id in outcome.evidence_ids
                )
            ),
        )
        committed_observations = {
            outcome.call_id: outcome.observation for outcome in proposal.call_outcomes
        }
        for call_outcome in proposal.call_outcomes:
            call = call_by_id.get(call_outcome.call_id)
            if call is None:
                continue
            named_by_call = {
                str(value)
                for value in call.arguments.values()
                if isinstance(value, (str, int))
            }
            for delta in call_outcome.state_delta:
                if named_by_call and not any(token in delta.path for token in named_by_call):
                    rejections.append(
                        f"{delta.path} names no entity call {call_outcome.call_id} referred to"
                    )
            schema = _output_schema(_tool_contract(tool_schemas, call.tool))
            schema_validation[call_outcome.call_id] = (
                "validated" if schema is not None else "unavailable"
            )
            if schema is not None:
                error = _validate_output(call_outcome.observation, schema)
                if error:
                    schema_validation[call_outcome.call_id] = "invalid"
                    rejections.append(f"{call.tool} {error}")
    elif len(calls) > 1 and (proposal.state_delta or proposal.events):
        rejections.append("a batched action with effects requires per-call outcomes")
    elif calls:
        # D67 keeps the aggregate single-call form: one call may state its
        # result as the proposal's own observation/delta/events. D70 draws the
        # line at silence -- a proposal that claims nothing at all cannot have
        # its execution inferred from the fact that a call was submitted.
        if not (proposal.observation or proposal.state_delta or proposal.events):
            rejections.append(
                f"submitted calls have no proposed outcome: "
                f"{sorted(call.call_id for call in calls if call.call_id)!r}"
            )
        else:
            executed_call_ids = tuple(call.call_id for call in calls if call.call_id)
            if proposal.state_delta or proposal.events:
                committed_call_ids = executed_call_ids
        if len(calls) > 1:
            # The aggregate top-level observation answers exactly one call.
            # This branch is reached by a genuine batch (multiple calls, no
            # call_outcomes, no aggregate effects either) that never said
            # which call the aggregate observation belongs to -- attributing
            # it to calls[0] would be a guess, and D70's own "every call must
            # be answered" rule already implies the others got no answer at
            # all. Reject explicitly rather than mis-key.
            rejections.append(
                "a batched action with no call_outcomes cannot use the aggregate "
                "single-call observation form -- it does not say which call it answers"
            )
        identity = calls[0].call_id
        # The aggregate form's single top-level observation answers the one
        # submitted call (guaranteed unique here: the multi-call case above
        # just rejected). Key by its call_id when it has one, else its tool.
        committed_observations = {(identity or calls[0].tool): proposal.observation}
        canonical_event_list: list[dict[str, Any]] = []
        for event in proposal.events:
            supplied = event.get("_call_id")
            if identity and supplied is not None and supplied != identity:
                rejections.append(f"event for call {identity} spoofed _call_id {supplied}")
                continue
            canonical_event_list.append(
                {**event, **({"_call_id": identity} if identity else {})}
            )
        canonical_events = tuple(canonical_event_list)
        schema = _output_schema(_tool_contract(tool_schemas, calls[0].tool))
        schema_validation[identity or "__single__"] = (
            "validated" if schema is not None else "unavailable"
        )
        if schema is not None:
            error = _validate_output(proposal.observation, schema)
            if error:
                schema_validation[identity or "__single__"] = "invalid"
                rejections.append(f"{calls[0].tool} {error}")

    order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, SupportLevel.HIGH]
    if order.index(proposal.support) < order.index(min_support):
        rejections.append(
            f"support {proposal.support.value!r} is below the required {min_support.value!r}"
        )

    if (normalized.state_delta or normalized.events) and not normalized.evidence_ids:
        # A committed change/event with nothing behind it is an invention,
        # however confident the model sounded about it.
        rejections.append("an effect was proposed with no supporting evidence")

    if allowed_evidence_ids is not None:
        invented = set(normalized.evidence_ids) - set(allowed_evidence_ids)
        if invented:
            rejections.append(
                f"evidence ids were not retrieved for this step: {sorted(invented)!r}"
            )

    if catalog is not None and normalized.state_delta:
        read_only = [call.tool for call in calls if catalog.effect_of(call.tool) is ToolEffect.READ]
        if read_only and len(read_only) == len(calls):
            rejections.append(f"read-only tools {read_only} cannot change state")

        reviewed_paths = tuple(
            pattern
            for call in calls
            if (entry := catalog.entry_for(call.tool)) is not None
            for pattern in entry.mutates_paths
        )
        if reviewed_paths:
            for delta in normalized.state_delta:
                if not any(fnmatchcase(delta.path, pattern) for pattern in reviewed_paths):
                    rejections.append(
                        f"{delta.path} matches no reviewed mutation path for the called tools"
                    )

    named = {
        str(value)
        for call in calls
        for value in call.arguments.values()
        if isinstance(value, (str, int))
    }
    for delta in normalized.state_delta:
        known = state.get(delta.path)
        if known is not None and delta.old_value is not None and known.value != delta.old_value:
            # The model is reasoning from a world that is not this one.
            rejections.append(
                f"{delta.path} is {known.value!r} in the ledger, not {delta.old_value!r}"
            )
        if named and not any(token in delta.path for token in named):
            # A call about reservation A must not quietly mutate reservation B.
            rejections.append(f"{delta.path} names no entity this action referred to")

    if rejections:
        return ValidationOutcome(
            accepted=False,
            rejections=tuple(rejections),
            output_schema_validation=schema_validation,
        )

    committed = tuple(
        StateField(
            path=delta.path,
            value=delta.new_value,
            origin=WorldOrigin.SIMULATED,
            revealed_at_step=step_index,
            evidence_ids=normalized.evidence_ids,
        )
        for delta in normalized.state_delta
    )
    return ValidationOutcome(
        accepted=True,
        committed=committed,
        executed_call_ids=executed_call_ids,
        committed_call_ids=committed_call_ids,
        events=canonical_events,
        output_schema_validation=schema_validation,
        committed_observations=committed_observations,
    )


def commit(state: ScenarioState, committed: Sequence[StateField]) -> ScenarioState:
    """Apply accepted fields, preserving the origin of everything else."""
    fields = {field.path: field for field in state.fields}
    for field in committed:
        fields[field.path] = field
    return ScenarioState(fields=tuple(fields.values()))


class _CoerceFailure:
    """Payload existed but did not fit the contract -- distinct from nothing at all.

    Carries the raw validation errors so a caller can report *why* parsing
    failed instead of collapsing every failure into one unexplained abstention.
    """

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors


def _coerce(payload: Any, model: type[Contract]) -> Any:
    """Read a predictor's answer into the contract.

    Returns the parsed model on success, ``None`` when there was truly nothing
    to parse (no payload, or a string that isn't even JSON), or a
    ``_CoerceFailure`` carrying the validation errors when a real payload
    existed but did not fit the contract's shape. Callers must not treat the
    latter as an abstention: the model attempted an answer, and salvaging a
    partial structured response would let it commit fields it never actually
    asserted, but silently discarding the attempt as "no answer" hides a
    prompt/parser defect behind a metric that looks epistemic.
    """
    if isinstance(payload, model):
        return payload
    if not payload:
        # Nothing was returned at all -- this alone is "no answer," never a
        # malformed one.
        return None

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError) as exc:
            # A non-empty string that isn't even valid JSON is an attempted
            # answer that failed to parse -- distinct from no answer at all.
            return _CoerceFailure((f"response was not valid JSON: {exc}",))
    elif hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif hasattr(payload, "toDict"):
        # dspy.Prediction (returned by dspy.Predict) is not a pydantic model
        # and has no model_dump -- without this branch every real prediction
        # fell straight through to `return None` below, silently misreported
        # as "the model returned nothing" (which step_tool_world reads as an
        # abstention) regardless of what the model actually said.
        payload = payload.toDict()

    if not isinstance(payload, dict):
        # Parsed to something real (a JSON array, a bare scalar, ...) but not
        # a shape that could ever fit the contract -- attempted, not absent.
        return _CoerceFailure((f"response parsed to {type(payload).__name__}, expected an object",))
    try:
        return model.model_validate(payload)
    except (TypeError, ValueError) as exc:
        detail = getattr(exc, "errors", None)
        if callable(detail):
            errors = tuple(
                f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
                for err in exc.errors()
            )
        else:
            errors = (str(exc),)
        return _CoerceFailure(errors)


def step_tool_world(
    predict: ToolWorldPredictor,
    *,
    calls: Sequence[ActionCall],
    content: Any,
    state: ScenarioState,
    history: str,
    examples: Sequence[RetrievedExample],
    instruction: str = TOOL_WORLD_INSTRUCTION,
) -> ProposedTransition:
    """One tool-world transition proposal, or an abstention.

    Support is taken from what retrieval actually found, not from what the model
    claims: a model asked to rate its own grounding rates it generously, and the
    number that gates committing has to come from outside it. The model's own
    figure is kept only where it is *lower*.
    """
    found = support_level(examples)
    if found == "none":
        return ProposedTransition(
            abstain=True,
            abstain_reason="no relevant recorded transition supports this action",
            support=SupportLevel.NONE,
        )

    raw = predict(
        instruction=instruction,
        state=render_state(state),
        history=history,
        action=render_action(calls, content),
        evidence=render_evidence(examples),
    )
    coerced = _coerce(raw, ProposedTransition)
    if isinstance(coerced, _CoerceFailure):
        # The model attempted an answer; it just didn't fit the contract.
        # This is a parse/prompt failure, not the model declining to answer,
        # and must not be scored as an abstention.
        return ProposedTransition(
            output_invalid=True,
            output_invalid_errors=coerced.errors,
            support=SupportLevel.NONE,
        )
    if coerced is None:
        return ProposedTransition(
            abstain=True,
            abstain_reason="the world model returned nothing that fit the transition contract",
            support=SupportLevel.NONE,
        )
    proposal = coerced

    retrieved = SupportLevel(found)
    order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, SupportLevel.HIGH]
    capped = min(proposal.support, retrieved, key=order.index)
    return proposal.replace(support=capped)


def step_user_policy(
    predict: UserPolicyPredictor,
    *,
    profile: HiddenUserProfile,
    history: str,
    message: str,
    examples: Sequence[RetrievedExample] = (),
    instruction: str = USER_POLICY_INSTRUCTION,
) -> ProposedUserTurn:
    """One simulated user reply, or an abstention.

    The profile reaches this prompt and never the candidate's. The returned
    ``disclosed_facts`` is what the D23 disclosure gate measures: a simulated
    user that volunteers what the real persona withheld has made the task
    easier, and pass@k would rise for a reason unrelated to the candidate.
    """
    raw = predict(
        instruction=instruction,
        profile=render_profile(profile),
        history=history,
        message=message,
        evidence=render_evidence(examples) if examples else "(no example turns retrieved)",
    )
    coerced = _coerce(raw, ProposedUserTurn)
    if not isinstance(coerced, ProposedUserTurn):
        # ProposedUserTurn carries no output_invalid distinction yet (D23's
        # user-policy fidelity work is separate scope); a failed parse -- gate
        # or contract mismatch -- still reads as abstention here rather than
        # crashing on a _CoerceFailure, but see world.py's ProposedTransition
        # for the taxonomy this should eventually get too.
        return ProposedUserTurn(
            abstain=True,
            abstain_reason="the user policy returned nothing that fit the contract",
        )
    return coerced


def build_tool_world_predictor(
    *, model: str, api_key: str | None = None, max_tokens: int = 4000
) -> ToolWorldPredictor:
    """A structured tool-world call, importing the model stack only when used."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise WorldModelError(
            "the grounded world model needs the 'diagnose' extra: uv sync --extra diagnose"
        ) from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        # The environment is not being asked to be creative. Run-to-run
        # variation is measured across seeds, never sampled per transition.
        temperature=0.0,
        max_tokens=max_tokens,
    )

    class _Transition(dspy.Signature):
        state: str = dspy.InputField(desc="what is currently known about the world")
        history: str = dspy.InputField(desc="the conversation so far")
        action: str = dspy.InputField(desc="the call the agent just made, with its call_id")
        evidence: str = dspy.InputField(desc="real recorded transitions from this system")
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

    def predict(*, instruction: str, state: str, history: str, action: str, evidence: str) -> Any:
        _Transition.__doc__ = instruction
        with dspy.context(lm=language_model):
            return dspy.Predict(_Transition)(
                state=state, history=history, action=action, evidence=evidence
            )

    return predict


def build_user_policy_predictor(
    *, model: str, api_key: str | None = None, max_tokens: int = 1500
) -> UserPolicyPredictor:
    """A structured user-policy call.

    The base instruction is derived from the guidelines the tau2 simulator
    itself ran under, recovered from the export's own ``info.user_info``. The
    reference user turns this is measured against were produced under exactly
    those rules, so starting from a freshly written prompt would optimize
    toward a target the reference behaviour never had.
    """
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise WorldModelError(
            "the grounded world model needs the 'diagnose' extra: uv sync --extra diagnose"
        ) from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
    )

    class _UserTurn(dspy.Signature):
        profile: str = dspy.InputField(desc="your private scenario brief")
        history: str = dspy.InputField(desc="the conversation so far")
        message: str = dspy.InputField(desc="what the agent just said to you")
        evidence: str = dspy.InputField(desc="example recorded user turns")
        user_message: str = dspy.OutputField()
        disclosed_facts: list[str] = dspy.OutputField()
        goal_status: str = dspy.OutputField()
        terminal: bool = dspy.OutputField()
        support: str = dspy.OutputField()
        abstain: bool = dspy.OutputField()
        abstain_reason: str = dspy.OutputField()

    def predict(
        *, instruction: str, profile: str, history: str, message: str, evidence: str
    ) -> Any:
        _UserTurn.__doc__ = instruction
        with dspy.context(lm=language_model):
            return dspy.Predict(_UserTurn)(
                profile=profile, history=history, message=message, evidence=evidence
            )

    return predict
