"""Does the simulator predict what actually happened?

Fidelity is measured by replaying *recorded* transitions: hide the observation
the trace holds, ask the world model what it would be, and compare. No candidate
is involved, and that separation is the point — a candidate scoring well on a
low-fidelity simulator has demonstrated nothing, and the two numbers must never
share an artifact (D3).

Two halves, with different authorities, never averaged:

    tool world                      user policy
    status and error correctness    premature disclosure
    per-field accuracy              disclosure after being asked
    state-delta accuracy            persona and goal consistency
    invariant violations            abstention on unsupported
    abstention calibration

The user half is not secondary. In the target family 189 of 371 observed
transitions are user replies, so the user policy governs the majority of the
environment, and a simulated user that volunteers what the real persona withheld
raises pass@k for a reason unrelated to the candidate.

Text similarity is never a fidelity measure. tau2 tool results are JSON, so the
tool half is a deterministic field diff and needs no judge; a judge is reserved
for natural-language user turns, where no structured diff exists.

An abstention is scored as *correctly* or *wrongly* declined — never as a wrong
prediction. Declining where no evidence exists is the behaviour the design asks
for, and counting it as an error would train the optimizer to invent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import Field

from bandits.diagnose.models import (
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    ScenarioState,
    StateField,
    SupportLevel,
    ToolEffectCatalog,
    WorldOrigin,
)
from bandits.diagnose.retrieve import RetrievedExample
from bandits.diagnose.retrieve import support_level as retrieved_support_level
from bandits.diagnose.world import (
    ProposedTransition,
    ProposedUserTurn,
    ToolWorldPredictor,
    UserPolicyPredictor,
    commit,
    step_tool_world,
    step_user_policy,
    validate_transition,
)
from bandits.traces import Contract


class FieldComparison(Contract):
    """One predicted field against the one that was recorded."""

    path: str
    predicted: Any = None
    recorded: Any = None
    present: bool = True
    """False when the prediction omitted a field the real result carried."""

    @property
    def correct(self) -> bool:
        return self.present and self.predicted == self.recorded


class TransitionFidelity(Contract):
    """How one recorded transition was predicted.

    ``abstained`` is kept separate from every accuracy figure. A declined
    prediction is not a wrong one, and folding the two together would make the
    honest behaviour look like the failure it exists to prevent.
    """

    transition_id: str
    trace_id: str
    abstained: bool = False
    abstain_correct: bool | None = None
    """True when abstaining was right — nothing supported a prediction here."""

    fields: tuple[FieldComparison, ...] = ()
    status_correct: bool | None = None
    """Whether success-versus-error was predicted correctly. The coarsest
    question, and the one a rollout's shape depends on most."""

    error_predicted: bool | None = None
    error_recorded: bool | None = None
    delta_correct: bool | None = None
    invariant_violations: tuple[str, ...] = ()
    support: SupportLevel = SupportLevel.NONE
    validator_rejected: bool = False
    """Whether the rollout runtime would have refused this prediction.

    D71. A gate that scores proposals the runtime discards measures a simulator
    that never runs. Kept beside accuracy rather than inside it: a rejected
    prediction is a fidelity failure of a different kind from a wrong one.
    """

    validator_rejections: tuple[str, ...] = ()

    @property
    def field_accuracy(self) -> float | None:
        """Share of recorded fields predicted exactly. None when nothing to compare."""
        if not self.fields:
            return None
        return sum(1 for field in self.fields if field.correct) / len(self.fields)


class DisclosureOutcome(Contract):
    """What a simulated user revealed, against what it was allowed to reveal.

    The measurement D23 exists for. A user policy that hands over a reservation
    id the agent never asked for has made the task easier, and every pass@k
    computed against it is inflated by an amount nothing else would reveal.
    """

    transition_id: str
    disclosed: tuple[str, ...] = ()
    premature: tuple[str, ...] = ()
    """Facts revealed without the agent having asked for them."""

    withheld_correctly: tuple[str, ...] = ()
    invented: tuple[str, ...] = ()
    """Facts stated that the profile said the user does not know.

    The worst failure available to a user policy: it does not merely leak the
    scenario, it fabricates it.
    """

    in_persona: bool | None = None
    goal_consistent: bool | None = None
    abstained: bool = False


class FidelityReport(Contract):
    """Aggregate fidelity for one AWM version.

    Parented to the world-model version and never to a candidate. A report that
    carried both would let a capability claim borrow credibility from a
    fidelity measurement that was about something else.
    """

    schema_version: int = 1
    awm_version: str = ""
    user_policy_version: str = ""
    retrieval_index_version: str = ""
    split: str = "held_out"
    """Which partition this was measured on. ``sealed`` is opened once."""

    transitions: tuple[TransitionFidelity, ...] = ()
    disclosures: tuple[DisclosureOutcome, ...] = ()
    drift: tuple[float, ...] = ()
    """Field accuracy at each step of a multi-step replay."""

    by_tool: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def considered(self) -> int:
        return len(self.transitions)

    @property
    def validation_rejection_rate(self) -> float | None:
        """Share of attempted predictions the runtime would refuse (D71)."""
        attempted = [row for row in self.transitions if not row.abstained]
        if not attempted:
            return None
        return sum(1 for row in attempted if row.validator_rejected) / len(attempted)

    @property
    def attempted(self) -> int:
        """Transitions where a prediction was actually made."""
        return sum(1 for row in self.transitions if not row.abstained)

    @property
    def abstention_rate(self) -> float | None:
        if not self.transitions:
            return None
        return 1 - self.attempted / self.considered

    @property
    def wrong_abstention_rate(self) -> float | None:
        """Share of abstentions that declined a supported transition."""
        abstained = [row for row in self.transitions if row.abstained]
        if not abstained:
            return None
        return sum(row.abstain_correct is False for row in abstained) / len(abstained)

    @property
    def correct_abstention_rate(self) -> float | None:
        """Share of abstentions that correctly refused an unsupported transition."""
        abstained = [row for row in self.transitions if row.abstained]
        if not abstained:
            return None
        return sum(row.abstain_correct is True for row in abstained) / len(abstained)

    @property
    def supported_coverage(self) -> float | None:
        """Share of supported transitions on which the AWM made a prediction.

        Correct abstentions are outside the supported population. Predictions and
        wrong abstentions are inside it, so declining a hard supported case lowers
        coverage instead of improving conditional accuracy for free.
        """
        supported = [
            row for row in self.transitions if not row.abstained or row.abstain_correct is False
        ]
        if not supported:
            return None
        return sum(not row.abstained for row in supported) / len(supported)

    @property
    def status_accuracy(self) -> float | None:
        scored = [row for row in self.transitions if row.status_correct is not None]
        if not scored:
            return None
        return sum(1 for row in scored if row.status_correct) / len(scored)

    @property
    def field_accuracy(self) -> float | None:
        scored = [row.field_accuracy for row in self.transitions if row.field_accuracy is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    @property
    def delta_accuracy(self) -> float | None:
        scored = [row for row in self.transitions if row.delta_correct is not None]
        if not scored:
            return None
        return sum(1 for row in scored if row.delta_correct) / len(scored)

    @property
    def premature_disclosure_rate(self) -> float | None:
        """The D23 headline. Above zero means pass@k is inflated."""
        if not self.disclosures:
            return None
        return sum(1 for row in self.disclosures if row.premature) / len(self.disclosures)

    @property
    def invention_rate(self) -> float | None:
        if not self.disclosures:
            return None
        return sum(1 for row in self.disclosures if row.invented) / len(self.disclosures)


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Comparable leaves of a structured payload, addressed by path."""
    found: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                found.update(_flatten(value, path))
            else:
                found[path] = value
    elif isinstance(payload, list):
        found[f"{prefix}.length" if prefix else "length"] = len(payload)
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(value, (dict, list)):
                found.update(_flatten(value, path))
            else:
                found[path] = value
    elif prefix:
        found[prefix] = payload
    return found


def compare_observation(predicted: Any, recorded: Any) -> tuple[FieldComparison, ...]:
    """Field-by-field diff, driven by what was *recorded*.

    The recorded result sets the field list, so a prediction cannot improve its
    score by omitting fields it was unsure about, and every real field it
    failed to produce is counted as missing rather than ignored.
    """
    real = _flatten(recorded)
    guessed = _flatten(predicted)
    return tuple(
        FieldComparison(
            path=path,
            predicted=guessed.get(path),
            recorded=value,
            present=path in guessed,
        )
        for path, value in sorted(real.items())
    )


def _looks_like_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        status = str(payload.get("status", "")).lower()
        if status == "error" or "error" in payload:
            return True
    return isinstance(payload, str) and payload.strip().lower().startswith("error")


def _recorded_observation(transition: GroundingTransition) -> GroundingObservation | None:
    return next(
        (item for item in transition.observations if item.role == "tool"),
        transition.observations[0] if transition.observations else None,
    )


def score_transition_fidelity(
    transition: GroundingTransition,
    predict: ToolWorldPredictor,
    *,
    examples: Sequence[RetrievedExample],
    history_text: str = "",
    invariants: Sequence[Callable[[ProposedTransition], str | None]] = (),
    minimum_support: SupportLevel = SupportLevel.LOW,
    catalog: ToolEffectCatalog | None = None,
    tool_schemas: Sequence[dict[str, Any]] = (),
) -> TransitionFidelity:
    """Predict one recorded transition and compare against what happened.

    The transition's own observation is never shown — only the history before
    it, the state before it, and retrieved evidence that excludes its lineage.
    """
    recorded = _recorded_observation(transition)
    proposal = step_tool_world(
        predict,
        calls=transition.action_calls,
        content=transition.action_content,
        state=transition.state_before,
        history=history_text,
        examples=examples,
    )

    if proposal.abstain:
        order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, SupportLevel.HIGH]
        found = SupportLevel(retrieved_support_level(examples))
        return TransitionFidelity(
            transition_id=transition.transition_id,
            trace_id=transition.trace_id,
            abstained=True,
            abstain_correct=order.index(found) < order.index(minimum_support),
            support=proposal.support,
        )

    # D71. Score what the runtime would accept. Without this the gate can pass
    # a prompt on transitions every rollout would discard -- the fidelity number
    # would describe a simulator that does not exist.
    validation = validate_transition(
        proposal,
        calls=transition.action_calls,
        state=transition.state_before,
        step_index=transition.turn_index,
        catalog=catalog,
        min_support=minimum_support,
        allowed_evidence_ids=tuple(
            example.transition.transition_id for example in examples
        ),
        tool_schemas=tool_schemas,
    )
    if not validation.accepted:
        # No accuracy fields: a refused prediction has no standing to be
        # called correct, and filling them in would let a rejected transition
        # contribute to the accuracy the gate reads.
        return TransitionFidelity(
            transition_id=transition.transition_id,
            trace_id=transition.trace_id,
            support=proposal.support,
            validator_rejected=True,
            validator_rejections=validation.rejections,
        )

    recorded_payload = recorded.content if recorded else None
    error_recorded = bool(recorded and recorded.error) or _looks_like_error(recorded_payload)
    error_predicted = _looks_like_error(proposal.observation)

    violations = tuple(message for check in invariants if (message := check(proposal)) is not None)
    predicted_delta = {delta.path: delta.new_value for delta in proposal.state_delta}
    recorded_delta = dict(transition.inferred_state_delta)

    return TransitionFidelity(
        transition_id=transition.transition_id,
        trace_id=transition.trace_id,
        fields=compare_observation(proposal.observation, recorded_payload),
        status_correct=error_predicted == error_recorded,
        error_predicted=error_predicted,
        error_recorded=error_recorded,
        delta_correct=predicted_delta == recorded_delta if recorded_delta else None,
        invariant_violations=violations,
        support=proposal.support,
    )


def _asked_for(history_text: str, message: str, fact: str) -> bool:
    """Whether the agent plausibly asked for this fact.

    Lexical and deliberately generous: the measurement it feeds is *premature*
    disclosure, so erring toward "was asked" under-reports rather than
    manufacturing a violation that did not happen.
    """
    haystack = f"{history_text}\n{message}".lower()
    return any(token in haystack for token in fact.lower().split() if len(token) > 3)


def score_disclosure(
    turn: ProposedUserTurn,
    *,
    transition_id: str,
    profile: HiddenUserProfile,
    history_text: str,
    agent_message: str,
) -> DisclosureOutcome:
    """What the simulated user revealed, and whether it was entitled to.

    Three outcomes matter and they are not the same failure: disclosing early
    makes the task easier, withholding correctly is the behaviour being asked
    for, and stating something from ``unknown_info`` is fabrication — the user
    policy inventing the scenario rather than playing it.
    """
    if turn.abstain:
        return DisclosureOutcome(transition_id=transition_id, abstained=True)

    spoken = turn.user_message.lower()
    if profile.known_facts or profile.unknown_fact_ids:
        invented = tuple(
            fact for fact in turn.disclosed_facts if fact in set(profile.unknown_fact_ids)
        )
        premature = tuple(
            fact
            for fact in turn.disclosed_facts
            if fact in profile.known_facts
            and not _asked_for(
                history_text,
                agent_message,
                f"{fact} {profile.known_facts[fact]}",
            )
        )
        withheld = tuple(
            fact for fact in profile.known_facts if fact not in set(turn.disclosed_facts)
        )
    else:
        # Legacy exports retain prose only. Kept as an explicitly weaker
        # fallback until re-ingestion supplies normalized fact ids.
        unknown_tokens = [
            token for token in profile.unknown_info.lower().split() if len(token) > 3
        ]
        invented = tuple(token for token in unknown_tokens if token in spoken)
        premature = tuple(
            fact
            for fact in turn.disclosed_facts
            if not _asked_for(history_text, agent_message, fact)
        )
        withheld = tuple(
            token
            for token in profile.known_info.lower().split()
            if len(token) > 3 and token not in spoken
        )

    return DisclosureOutcome(
        transition_id=transition_id,
        disclosed=turn.disclosed_facts,
        premature=premature,
        withheld_correctly=withheld,
        invented=invented,
    )


def score_user_fidelity(
    transition: GroundingTransition,
    predict: UserPolicyPredictor,
    *,
    profile: HiddenUserProfile,
    examples: Sequence[RetrievedExample] = (),
    history_text: str = "",
) -> DisclosureOutcome:
    """One recorded user turn, predicted and checked for disclosure discipline."""
    turn = step_user_policy(
        predict,
        profile=profile,
        history=history_text,
        message=str(transition.action_content or ""),
        examples=examples,
    )
    return score_disclosure(
        turn,
        transition_id=transition.transition_id,
        profile=profile,
        history_text=history_text,
        agent_message=str(transition.action_content or ""),
    )


def multi_step_drift(
    transitions: Sequence[GroundingTransition],
    predict: ToolWorldPredictor,
    *,
    retrieve_for: Callable[[GroundingTransition, ScenarioState], Sequence[RetrievedExample]],
    horizon: int = 5,
) -> tuple[float, ...]:
    """Field accuracy at each step of a teacher-action replay.

    The actions are the *recorded* ones, so the candidate is held fixed and only
    the environment's own error accumulates. One-step accuracy hides this
    entirely: a simulator that is 90% right per step is right about half the
    time after five, and a rollout is many more steps than five.
    """
    accuracies: list[float] = []
    state = transitions[0].state_before if transitions else ScenarioState()

    for step_index, transition in enumerate(transitions[:horizon]):
        examples_here = retrieve_for(transition, state)
        proposal = step_tool_world(
            predict,
            calls=transition.action_calls,
            content=transition.action_content,
            state=state,
            history="",
            examples=examples_here,
        )
        if proposal.abstain:
            accuracies.append(0.0)
            continue

        # D71. A prediction the runtime would refuse must not be committed:
        # the next step would then reason from a world the rollout would never
        # have produced, and the drift curve would describe that world instead.
        if not validate_transition(
            proposal,
            calls=transition.action_calls,
            state=state,
            step_index=step_index,
            allowed_evidence_ids=tuple(
                example.transition.transition_id for example in examples_here
            ),
        ).accepted:
            accuracies.append(0.0)
            continue

        state = commit(
            state,
            tuple(
                StateField(
                    path=delta.path,
                    value=delta.new_value,
                    origin=WorldOrigin.SIMULATED,
                    revealed_at_step=step_index,
                    evidence_ids=proposal.evidence_ids,
                )
                for delta in proposal.state_delta
            ),
        )
        expected = {field.path: field.value for field in transition.state_before.fields}
        expected.update(transition.inferred_state_delta)
        predicted = {field.path: field.value for field in state.fields}
        paths = set(expected) | set(predicted)
        accuracy = (
            sum(expected.get(path) == predicted.get(path) for path in paths) / len(paths)
            if paths
            else 1.0
        )
        accuracies.append(accuracy)
    return tuple(accuracies)


def build_report(
    transitions: Sequence[TransitionFidelity],
    *,
    disclosures: Sequence[DisclosureOutcome] = (),
    drift: Sequence[float] = (),
    awm_version: str = "",
    user_policy_version: str = "",
    retrieval_index_version: str = "",
    split: str = "held_out",
    tool_of: Callable[[str], str] | None = None,
) -> FidelityReport:
    """Assemble the report, stratified by tool where the caller can say which.

    Per-tool stratification is not decoration. With zero error evidence for
    every tool in the target family, an aggregate accuracy can be carried
    entirely by the one tool that happens to be well covered, and the table is
    what prevents that reading.
    """
    by_tool: dict[str, dict[str, float]] = {}
    if tool_of is not None:
        grouped: dict[str, list[TransitionFidelity]] = {}
        for row in transitions:
            grouped.setdefault(tool_of(row.transition_id), []).append(row)
        for tool, rows in grouped.items():
            scored = [r.field_accuracy for r in rows if r.field_accuracy is not None]
            statuses = [r for r in rows if r.status_correct is not None]
            by_tool[tool] = {
                "considered": float(len(rows)),
                "abstained": float(sum(1 for r in rows if r.abstained)),
                "field_accuracy": sum(scored) / len(scored) if scored else 0.0,
                "status_accuracy": (
                    sum(1 for r in statuses if r.status_correct) / len(statuses)
                    if statuses
                    else 0.0
                ),
            }

    return FidelityReport(
        awm_version=awm_version,
        user_policy_version=user_policy_version,
        retrieval_index_version=retrieval_index_version,
        split=split,
        transitions=tuple(transitions),
        disclosures=tuple(disclosures),
        drift=tuple(drift),
        by_tool=by_tool,
    )


def gate(report: FidelityReport, *, thresholds: dict[str, float]) -> tuple[bool, tuple[str, ...]]:
    """Whether this AWM version may be used for capability measurement.

    Both halves must pass. A tool world that predicts well while its user policy
    leaks the scenario produces rollouts whose difficulty was changed by the
    environment, and the capability number would be measuring that change.

    Thresholds are supplied rather than defaulted: no measurement exists yet to
    justify a particular bar, and inventing one here would make an unreviewed
    number look like a reviewed one.
    """
    failures: list[str] = []

    def check(name: str, value: float | None, minimum: float | None, higher_is_better=True) -> None:
        if minimum is None:
            return
        if value is None:
            failures.append(f"{name} was not measured")
        elif higher_is_better and value < minimum:
            failures.append(f"{name} {value:.3f} is below {minimum:.3f}")
        elif not higher_is_better and value > minimum:
            failures.append(f"{name} {value:.3f} is above {minimum:.3f}")

    check("status_accuracy", report.status_accuracy, thresholds.get("status_accuracy"))
    check("field_accuracy", report.field_accuracy, thresholds.get("field_accuracy"))
    check("delta_accuracy", report.delta_accuracy, thresholds.get("delta_accuracy"))
    check(
        "supported_coverage", report.supported_coverage, thresholds.get("supported_coverage")
    )
    check(
        "wrong_abstention_rate",
        report.wrong_abstention_rate,
        thresholds.get("wrong_abstention_rate"),
        higher_is_better=False,
    )
    check(
        "premature_disclosure_rate",
        report.premature_disclosure_rate,
        thresholds.get("premature_disclosure_rate"),
        higher_is_better=False,
    )
    check(
        "invention_rate",
        report.invention_rate,
        thresholds.get("invention_rate"),
        higher_is_better=False,
    )
    check(
        "validation_rejection_rate",
        report.validation_rejection_rate,
        thresholds.get("validation_rejection_rate"),
        higher_is_better=False,
    )

    violations = sum(len(row.invariant_violations) for row in report.transitions)
    if violations and thresholds.get("invariant_violations", 0) < violations:
        failures.append(f"{violations} invariant violations")

    return (not failures, tuple(failures))
