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

from bandits.diagnose.grounding import assess_grounding
from bandits.diagnose.models import (
    DeltaGroundTruthStatus,
    GroundingAssessment,
    GroundingObservation,
    GroundingTransition,
    HiddenUserProfile,
    ScenarioState,
    StateField,
    SupportLevel,
    ToolEffectCatalog,
    WorldOrigin,
)
from bandits.diagnose.retrieve import RetrievedExample, support_level
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

    grounding: GroundingAssessment | None = None
    """What was and wasn't identifiable (I39), on every row -- abstained,
    output-invalid, rejected, or accepted. Not only an abstention-correctness
    input: it also answers whether an accepted success was state-grounded,
    and whether a non-abstaining answer was fabricated on an unidentifiable
    case. None only if scoring never reached grounding assessment at all."""

    output_invalid: bool = False
    """The model attempted an answer that did not parse into the transition
    contract -- a prompt/parser failure, not the model declining to answer.
    Mutually exclusive with ``abstained``: conflating the two would let a
    structured-output defect masquerade as a legitimate abstention rate."""

    output_invalid_errors: tuple[str, ...] = ()

    fields: tuple[FieldComparison, ...] = ()
    status_correct: bool | None = None
    """Whether success-versus-error was predicted correctly. The coarsest
    question, and the one a rollout's shape depends on most."""

    error_predicted: bool | None = None
    error_recorded: bool | None = None
    delta_correct: bool | None = None
    delta_ground_truth_status: DeltaGroundTruthStatus = DeltaGroundTruthStatus.NOT_APPLICABLE
    """Whether delta_correct is even a meaningful comparison for this row.
    None/absent-looking delta_correct with status MEASURED means a genuinely
    verified no-op was compared; UNAVAILABLE means no comparison was possible
    at all, which delta_correct=None alone cannot distinguish on its own."""

    invariant_violations: tuple[str, ...] = ()
    support: SupportLevel = SupportLevel.NONE
    validator_rejected: bool = False
    """Whether the rollout runtime would have refused this prediction.

    D71. A gate that scores proposals the runtime discards measures a simulator
    that never runs. Kept beside accuracy rather than inside it: a rejected
    prediction is a fidelity failure of a different kind from a wrong one.
    """

    validator_rejections: tuple[str, ...] = ()

    unmatched_call_observations: tuple[str, ...] = ()
    """Call ids (or tool names) whose recorded observation could not be
    correlated with a predicted one -- e.g. a batch where the model's
    committed_observations and the recorded tool_call_ids never overlap.
    Distinct from a validator rejection: the proposal was accepted, but part
    of it cannot be scored for field accuracy because there is nothing to
    compare it against.

    A non-empty tuple here means `fields`/`status_correct`/`error_predicted`/
    `error_recorded` are ALL absent (empty/None), not a partial score over
    whichever calls did match: scoring only the matched subset and reporting
    it as this transition's field_accuracy would let one perfectly-correlated
    call in a two-call batch report 1.0 while the other call goes entirely
    unscored and invisible. An accepted-but-incompletely-correlated
    prediction is unscorable as a whole, and this tuple says why.
    """

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
    def model_output_invalid_rate(self) -> float | None:
        """Share of considered transitions where the model answered but the
        answer did not fit the transition contract (a parser/prompt failure).

        Kept separate from abstention_rate and, among accepted-vs-attempted
        accuracy figures, from every accuracy number: an invalid-output row
        carries no fields to score and must not silently inflate or deflate
        either abstention or accuracy the way folding it into "abstained"
        used to.
        """
        if not self.transitions:
            return None
        return sum(1 for row in self.transitions if row.output_invalid) / self.considered

    @property
    def validation_rejection_rate(self) -> float | None:
        """Share of attempted predictions the runtime would refuse (D71).

        "Attempted" here means the model neither abstained nor produced an
        output-invalid response -- both of those are excluded from
        ``attempted`` and scored by their own rates instead.
        """
        attempted = [row for row in self.transitions if not row.abstained and not row.output_invalid]
        if not attempted:
            return None
        return sum(1 for row in attempted if row.validator_rejected) / len(attempted)

    @property
    def attempted(self) -> int:
        """Transitions where a prediction was actually made and parsed.

        Excludes both abstentions (the model declined) and output-invalid
        rows (the model tried but its answer didn't fit the contract) -- the
        latter is a distinct failure mode with its own rate, not evidence the
        model attempted nothing.
        """
        return sum(1 for row in self.transitions if not row.abstained and not row.output_invalid)

    @property
    def abstention_rate(self) -> float | None:
        if not self.transitions:
            return None
        return sum(1 for row in self.transitions if row.abstained) / self.considered

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
    def delta_ground_truth_coverage(self) -> float | None:
        """Share of considered transitions where a real delta comparison was
        even possible (status MEASURED).

        Low coverage means delta_accuracy is being computed over a small,
        possibly unrepresentative slice -- most commonly because state paths
        are tool-prefixed and a mutating tool's paths never align with the
        read tool's paths that populated state_before (status UNAVAILABLE).
        Report this beside delta_accuracy, never delta_accuracy alone.
        """
        if not self.transitions:
            return None
        measured = sum(
            1
            for row in self.transitions
            if row.delta_ground_truth_status is DeltaGroundTruthStatus.MEASURED
        )
        return measured / self.considered

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


def _correlate_observations(
    transition: GroundingTransition, committed_observations: dict[str, Any]
) -> tuple[dict[str, tuple[Any, GroundingObservation]], tuple[str, ...]]:
    """Pair each recorded tool observation with the prediction that answers it.

    committed_observations is keyed by call_id (or tool name, only when the
    sole submitted call has no call_id -- see validate_transition). Recorded
    GroundingObservations carry tool_call_id, which in real tau2 data is
    frequently None even for a single unbatched call: the source simply never
    assigned one. Matching is therefore, in order:

    1. tool_call_id == a committed call_id (both sides have real ids);
    2. exactly one action_call and exactly one recorded tool observation and
       exactly one committed observation, all unlabeled -- the unambiguous
       single-call case, paired positionally;
    3. otherwise unmatched -- never guessed at with a bare [0].

    Returns (matched: {call_key: (predicted_content, recorded_observation)}, ...)
    -- pairs keyed by a stable id, unmatched_call_ids). The recorded side is
    the FULL GroundingObservation, not just its content: an error recorded
    via the ``error`` flag with a payload that doesn't itself look like an
    error (no "error" key, no "error"-prefixed string) must not be invisible
    to the caller just because only .content was kept.
    """
    tool_observations = [obs for obs in transition.observations if obs.role == "tool"]
    calls = transition.action_calls

    pairs: dict[str, tuple[Any, GroundingObservation]] = {}
    unmatched: list[str] = []

    if len(calls) == 1 and len(tool_observations) == 1 and len(committed_observations) == 1:
        # The unambiguous single-call case: one call submitted, one tool
        # reaction recorded, one committed prediction -- pair them even when
        # neither side carries a call_id, since there is no other call this
        # observation could answer.
        (only_key, only_predicted) = next(iter(committed_observations.items()))
        pairs[only_key] = (only_predicted, tool_observations[0])
        return pairs, ()

    for call in calls:
        key = call.call_id or call.tool
        if key not in committed_observations:
            unmatched.append(key)
            continue
        recorded = next(
            (obs for obs in tool_observations if obs.tool_call_id == call.call_id),
            None,
        )
        if recorded is None:
            unmatched.append(key)
            continue
        pairs[key] = (committed_observations[key], recorded)

    return pairs, tuple(unmatched)


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

    # Computed unconditionally, not only on the abstain path: whether the
    # AWM fabricated an unavailable fact on a non-abstaining answer, whether
    # an output-invalid response was actually facing an identifiable case, and
    # whether an accepted success was state-grounded are all questions this
    # answers too -- an auditor that only runs when the model already declined
    # cannot see any of them.
    grounding = assess_grounding(transition, state_before=transition.state_before, examples=examples)

    if proposal.output_invalid:
        # The model attempted an answer; it did not fit the contract. This is
        # a structural failure, not an epistemic one -- it must not touch
        # abstention or accuracy metrics. It also never reaches
        # validate_transition (score_transition_fidelity returns here first),
        # so validator_rejected stays False: that flag means "the runtime
        # validator specifically refused this," which did not happen here,
        # there was nothing parseable to hand it. model_output_invalid_rate
        # is the metric for this failure mode, not validation_rejection_rate
        # -- both FidelityReport properties already treat these rows as
        # mutually exclusive; setting validator_rejected=True here would
        # contradict that split even though the rate computation itself
        # already excludes output_invalid rows.
        return TransitionFidelity(
            transition_id=transition.transition_id,
            trace_id=transition.trace_id,
            output_invalid=True,
            output_invalid_errors=proposal.output_invalid_errors,
            support=proposal.support,
            grounding=grounding,
        )

    if proposal.abstain:
        # I39, narrowly: aggregate behavioral support (SupportLevel) is not
        # licence to require an exact-value answer from a single-call
        # get_user_details/get_reservation_details read -- a same-tool-
        # different-entity example can clear `minimum_support` while the
        # specific entity's fields remain genuinely unknown. Everything else
        # (writes, batches, non-entity tools) is outside what grounding.py
        # was reviewed to classify; values_identifiable is None there and the
        # prior SupportLevel-based rule is kept unchanged, not reinterpreted.
        identifiable = grounding.values_identifiable
        if identifiable is None:
            order = [SupportLevel.NONE, SupportLevel.LOW, SupportLevel.MEDIUM, SupportLevel.HIGH]
            found = SupportLevel(support_level(examples))
            abstain_correct = order.index(found) < order.index(minimum_support)
        else:
            abstain_correct = not identifiable
        return TransitionFidelity(
            transition_id=transition.transition_id,
            trace_id=transition.trace_id,
            abstained=True,
            abstain_correct=abstain_correct,
            grounding=grounding,
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
            grounding=grounding,
        )

    # Fidelity must compare the same canonical observation the runtime
    # validator accepted -- not proposal.observation directly, which is left
    # empty ({}) by design whenever the model correctly used the batch
    # call_outcomes form. validation.committed_observations is that
    # canonical, call-correlated mapping; _correlate_observations pairs each
    # entry against the recorded GroundingObservation it actually answers.
    pairs, unmatched = _correlate_observations(transition, validation.committed_observations)

    def _recorded_error(recorded_obs: GroundingObservation) -> bool:
        # The recorded ``error`` flag is authoritative provenance (D-whatever
        # marked the span itself as a runtime error) and must be checked
        # alongside the payload-shape heuristic: a call whose error flag is
        # True but whose payload happens not to look like an error (no
        # "error" key, no "error"-prefixed string) must still read as an
        # error here, not silently as a success.
        return bool(recorded_obs.error) or _looks_like_error(recorded_obs.content)

    fields: list[FieldComparison] = []
    for predicted_payload, recorded_obs in pairs.values():
        fields.extend(compare_observation(predicted_payload, recorded_obs.content))

    # Per-call, not "did any call in the batch error": aggregating status
    # across calls means a batch where call A errored and B didn't, matched
    # against a prediction where B errored and A didn't, would both read as
    # "an error happened somewhere" and score status_correct=True despite
    # the error being attributed to the wrong call entirely.
    per_call_status_correct = [
        _looks_like_error(predicted_payload) == _recorded_error(recorded_obs)
        for predicted_payload, recorded_obs in pairs.values()
    ]
    error_recorded = bool(recorded and recorded.error) or any(
        _recorded_error(recorded_obs) for _, recorded_obs in pairs.values()
    )
    error_predicted = any(_looks_like_error(predicted_payload) for predicted_payload, _ in pairs.values())

    violations = tuple(message for check in invariants if (message := check(proposal)) is not None)
    # validation.committed, not proposal.state_delta: the latter is the
    # top-level aggregate field, left empty by design whenever the model used
    # the batch call_outcomes form (its deltas live inside each outcome
    # instead). validation.committed is the validator's own canonical,
    # already-flattened result -- the same one score_transition_fidelity must
    # use for observations, for the same reason.
    predicted_delta = {field.path: field.value for field in validation.committed}
    recorded_delta = dict(transition.inferred_state_delta)
    # delta_correct is only meaningful when the recorded delta was actually
    # measured (before/after paths aligned). UNAVAILABLE means "we don't know
    # if anything changed," not "nothing changed" -- scoring against it would
    # silently launder an alignment gap into a fidelity number.
    delta_measured = transition.delta_ground_truth_status is DeltaGroundTruthStatus.MEASURED

    # Incomplete correlation must not report a confident number computed only
    # over whichever calls happened to match. A batch where call A correlated
    # perfectly and call B did not must not read as field_accuracy=1.0 --
    # that hides an entire unscored call behind a number that looks complete.
    # unmatched_call_observations already names which calls are missing; the
    # accuracy fields themselves fall back to "not scorable" rather than
    # scoring a strict subset and calling it whole.
    fully_correlated = not unmatched

    return TransitionFidelity(
        transition_id=transition.transition_id,
        trace_id=transition.trace_id,
        fields=tuple(fields) if fully_correlated else (),
        status_correct=(all(per_call_status_correct) if pairs else None) if fully_correlated else None,
        error_predicted=(error_predicted if pairs else None) if fully_correlated else None,
        error_recorded=(error_recorded if pairs else None) if fully_correlated else None,
        delta_correct=(predicted_delta == recorded_delta) if delta_measured else None,
        delta_ground_truth_status=transition.delta_ground_truth_status,
        invariant_violations=violations,
        support=proposal.support,
        unmatched_call_observations=unmatched,
        grounding=grounding,
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
        # Only MEASURED deltas are trustworthy ground truth for this step's
        # change. UNAVAILABLE's inferred_state_delta is {} the same shape as
        # a genuine no-op, so it is never merged into `expected` here --
        # doing so would assert "nothing changed" for paths ground truth
        # simply could not verify.
        if transition.delta_ground_truth_status is DeltaGroundTruthStatus.MEASURED:
            expected.update(transition.inferred_state_delta)

        # Score only paths `expected` actually has an opinion on -- state
        # carries every path the simulator has accumulated across ALL prior
        # steps (state.fields is cumulative), and scoring the union with
        # `expected` penalised the predictor for any self-consistent path
        # `expected` never asserted anything about, including this step's own
        # UNAVAILABLE-delta prediction. A path `expected` is silent on is
        # unmeasurable at this step, not evidence the prediction was wrong.
        predicted = {field.path: field.value for field in state.fields}
        paths = set(expected)
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
                "output_invalid": float(sum(1 for r in rows if r.output_invalid)),
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
