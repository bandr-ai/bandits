"""Compile existing Bandits artifacts into scenarios and grounding transitions.

This module extracts nothing new from a source export. Ingest already produced
canonical spans, analysis already produced tasks and outcome evidence, mining
already produced families and a lineage-safe split, and review already produced
verifiers. What is missing between those artifacts and an environment is only
two things, and they are all this module makes:

* **cut points** — the places in a real episode a candidate could be asked to
  take over from, with the authentic history before them and nothing after;
* **lossless transitions** — each recorded action paired with the observation
  that actually followed it, kept structured so a fidelity check can diff
  fields rather than strings.

The span-boundary rule is shared with PR #44's ``Turn`` and is correct there:
a model span opens an action, the tool results and user messages before the
next model span are its reaction, and a trailing action with no reaction is
unobserved. What differs is the payload — see ``GroundingTransition``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from bandits.analyze.models import TaskFamily
from bandits.diagnose.models import (
    ActionCall,
    GroundingObservation,
    GroundingTransition,
    Partition,
    PrefixStep,
    Scenario,
    ScenarioKind,
    ScenarioState,
    SealedSuccessContract,
    StateField,
    ToolEffectCatalog,
    WorldOrigin,
)
from bandits.traces import Span, SpanKind, SpanStatus, Trace

MAX_STATE_DEPTH = 4
"""How far into a tool result to walk when reading state.

Matches ``analyze/outcomes.py``: deeper than this is a document, not state.
"""

MAX_STATE_FIELDS = 64

SPEAKER_SPAN_NAMES = frozenset({"assistant", "model", "llm"})
"""MODEL span names that denote speech rather than a tool call.

The chat-JSON adapter puts the tool's own name on the span when the assistant
called one, and a generic speaker name when it only talked. Without this set a
model named ``assistant`` would be read as a tool called ``assistant``.
"""


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def strip_markers(text: str, markers: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Remove declared control markers, recording which ones were there.

    Filtering marker-carrying content out of the index entirely would discard
    most of tau2's user turns: ``###TRANSFER###`` is appended to a user turn's
    own text on most airline episodes, not only the ones that escalate. An AWM
    shown it learns to emit it; an AWM never shown those turns loses the
    majority of its user evidence. Stripping and recording keeps both.
    """
    found: list[str] = []
    for marker in markers:
        if marker and marker in text:
            found.append(marker)
            text = text.replace(marker, "")
    return text.strip(), tuple(found)


def _json_value(value: Any) -> Any:
    """Parse a tool result that arrived as JSON text, else return it as-is."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except (TypeError, ValueError):
                return value
    return value


def _state_paths(payload: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Comparable leaves of a structured result, addressed by path.

    Arrays contribute their length only. Position in a list is not stable
    between runs of one task, so a fact about the third element is a fact about
    whichever item happened to land third.
    """
    found: dict[str, Any] = {}
    if depth >= MAX_STATE_DEPTH or len(found) >= MAX_STATE_FIELDS:
        return found
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                found[path] = value
            elif isinstance(value, list):
                found[f"{path}.length"] = len(value)
            elif isinstance(value, dict):
                found.update(_state_paths(value, path, depth + 1))
            if len(found) >= MAX_STATE_FIELDS:
                break
    return found


def _entity_prefix(tool: str, arguments: dict[str, Any]) -> str:
    """Namespace a result's fields by the entity the call named.

    Without this, two reservations both reporting ``status`` collapse onto one
    path and the second silently overwrites the first. The id the call carried
    is the only thing that separates them, and it is available on every call.
    """
    for key in ("reservation_id", "user_id", "flight_number", "id"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return f"{tool}.{value}"
    return tool


def _render_action(span: Span) -> tuple[str | None, dict[str, Any], Any]:
    """A model span as (tool, arguments, content).

    Two shapes reach here and both are real. The chat-JSON adapter records a
    tool call as a MODEL span whose ``name`` *is* the tool and whose
    ``arguments`` are the call's arguments directly; an assistant that only
    spoke gets ``name="assistant"`` and empty arguments. Other adapters nest the
    call under ``tool_calls`` in the span's arguments, the way the wire format
    does. Reading only the nested shape would have silently produced zero
    tool-typed transitions on the tau2 corpus — every action would have looked
    like plain speech — so both are handled and neither is assumed.
    """
    arguments = dict(span.arguments or {})
    calls = arguments.get("tool_calls")
    if isinstance(calls, list) and calls:
        first = calls[0]
        if isinstance(first, dict):
            function = first.get("function") if isinstance(first.get("function"), dict) else first
            raw = function.get("arguments") if isinstance(function, dict) else None
            parsed = _json_value(raw) if raw is not None else {}
            return (
                (function or {}).get("name"),
                parsed if isinstance(parsed, dict) else {},
                arguments.get("content"),
            )
    if span.name and span.name not in SPEAKER_SPAN_NAMES:
        return span.name, arguments, span.output
    return None, arguments, span.output


def _prefix_step(span: Span, markers: Sequence[str]) -> PrefixStep:
    if span.kind is SpanKind.MODEL:
        tool, arguments, content = _render_action(span)
        return PrefixStep(
            role="assistant",
            content=content,
            tool_name=tool,
            arguments=arguments or None,
            span_id=span.span_id,
        )
    output = span.output
    if isinstance(output, str):
        output, _ = strip_markers(output, markers)
    return PrefixStep(
        role="tool",
        content=output,
        tool_name=span.name,
        span_id=span.span_id,
        error=span.status is SpanStatus.ERROR,
    )


def _user_steps(
    trace: Trace, after_span_id: str | None, markers: Sequence[str]
) -> tuple[list[PrefixStep], tuple[str, ...]]:
    """User turns at one position, cleaned, plus the markers that were removed.

    The markers are returned rather than discarded because stripping twice finds
    nothing the second time: a caller that re-stripped already-clean text would
    record an empty provenance, and D29's guarantee — that a removal is always
    visible in the artifact — would hold in the code and not in the data.
    """
    steps: list[PrefixStep] = []
    stripped: list[str] = []
    for turn in trace.user_turns:
        if turn.after_span_id == after_span_id:
            text, found = strip_markers(turn.text, markers)
            stripped.extend(found)
            steps.append(PrefixStep(role="user", content=text, span_id=after_span_id))
    return steps, tuple(dict.fromkeys(stripped))


def build_prefix(
    trace: Trace, through_span_id: str | None, *, markers: Sequence[str] = ()
) -> tuple[PrefixStep, ...]:
    """Authentic history up to and including one span, and nothing after it.

    The candidate must never see a future historical action: that is the whole
    basis on which a prefix-started rollout measures the candidate rather than
    the trace. Terminating at the cut span is therefore the one invariant here.
    """
    steps: list[PrefixStep] = []
    if trace.system_prompt:
        steps.append(PrefixStep(role="system", content=trace.system_prompt))
    opening, _ = _user_steps(trace, None, markers)
    steps.extend(opening)

    for span in trace.spans:
        steps.append(_prefix_step(span, markers))
        following, _ = _user_steps(trace, span.span_id, markers)
        steps.extend(following)
        if through_span_id is not None and span.span_id == through_span_id:
            break
    return tuple(steps)


def reconstruct_state(
    trace: Trace, through_span_id: str | None, *, catalog: ToolEffectCatalog | None = None
) -> ScenarioState:
    """Everything the authentic prefix revealed about the world, and nothing else.

    Conservative by construction: we hold no production database snapshot, so a
    path no recorded result mentioned is unknown rather than absent. Reads and
    writes both reveal state — a read reports what is true, a write reports what
    became true — so both contribute, and the effect catalog is not consulted
    here. It matters when *scoring* which calls carry operational success, not
    when learning what the world looks like.

    Later results overwrite earlier ones on the same path, which is the point: a
    reservation read as confirmed and later reported cancelled is cancelled, and
    the ledger has to agree with the last thing the episode observed.
    """
    fields: dict[str, StateField] = {}
    last_call: tuple[str, dict[str, Any]] | None = None

    for span in trace.spans:
        if span.kind is SpanKind.MODEL:
            tool, arguments, _ = _render_action(span)
            if tool:
                last_call = (tool, arguments)
        elif span.status is not SpanStatus.ERROR:
            # An error reports what did *not* happen. Reading state out of it
            # would invent the very fact the failure denies.
            call_tool, call_args = last_call or (span.name, {})
            prefix = _entity_prefix(call_tool, call_args)
            for path, value in _state_paths(_json_value(span.output)).items():
                full = f"{prefix}.{path}"
                fields[full] = StateField(
                    path=full,
                    value=value,
                    origin=WorldOrigin.RECORDED,
                    revealed_by_span_id=span.span_id,
                )
            last_call = None
        if through_span_id is not None and span.span_id == through_span_id:
            break

    return ScenarioState(fields=tuple(fields.values()))


def _action_runs(trace: Trace) -> list[tuple[list[Span], list[Span]]]:
    """Group spans into (action spans, reaction spans).

    An action is a *run* of consecutive tool-call model spans plus any speech
    that opened it, because the adapter splits one assistant message carrying
    several calls into one span per call. 54 actions in the tau2 corpus are such
    batches and one carries ten calls; reading each span separately would
    invent a sequence the model never chose, and pair each call with whichever
    result happened to follow rather than its own.
    """
    runs: list[tuple[list[Span], list[Span]]] = []
    spans = list(trace.spans)
    index = 0
    while index < len(spans):
        if spans[index].kind is not SpanKind.MODEL:
            index += 1
            continue
        action = [spans[index]]
        index += 1
        # Consecutive tool-call model spans are one batched action. A speaker
        # span ends the batch: the assistant said something new.
        while (
            index < len(spans)
            and spans[index].kind is SpanKind.MODEL
            and spans[index].name not in SPEAKER_SPAN_NAMES
            and action[-1].name not in SPEAKER_SPAN_NAMES
        ):
            action.append(spans[index])
            index += 1
        reactions = []
        while index < len(spans) and spans[index].kind is not SpanKind.MODEL:
            reactions.append(spans[index])
            index += 1
        runs.append((action, reactions))
    return runs


def _calls_of(action: Sequence[Span]) -> tuple[ActionCall, ...]:
    calls: list[ActionCall] = []
    for span in action:
        tool, arguments, _ = _render_action(span)
        if tool:
            calls.append(
                ActionCall(
                    call_id=str(span.arguments.get("tool_call_id") or span.span_id),
                    tool=tool,
                    arguments=arguments,
                )
            )
    return tuple(calls)


def extract_transitions(
    trace: Trace,
    *,
    family_id: str,
    markers: Sequence[str] = (),
    task_context: str = "",
) -> tuple[GroundingTransition, ...]:
    """Every recorded action in one trace with the observations that followed.

    Lossless: structured payloads are kept whole, history is not clipped, a
    batched action keeps all its calls, and each reaction keeps the pairing back
    to the call it answers. Rendering against a context budget happens when a
    prompt is built, never here — a truncated result in the index teaches the
    simulator to emit truncated results, and a fidelity diff needs real fields.
    """
    transitions: list[GroundingTransition] = []
    prefix_all = build_prefix(trace, None, markers=markers)

    for index, (action, reactions) in enumerate(_action_runs(trace)):
        first = action[0]
        calls = _calls_of(action)
        content = next(
            (span.output for span in action if span.name in SPEAKER_SPAN_NAMES and span.output),
            None,
        )

        observations: list[GroundingObservation] = []
        for span in reactions:
            observations.append(
                GroundingObservation(
                    role="tool",
                    content=_json_value(span.output),
                    tool_name=span.name,
                    tool_call_id=str(span.arguments.get("tool_call_id") or "") or None,
                    span_id=span.span_id,
                    error=span.status is SpanStatus.ERROR,
                )
            )

        user_steps, user_markers = _user_steps(trace, action[-1].span_id, markers)
        for step in user_steps:
            observations.append(
                GroundingObservation(role="user", content=step.content, span_id=action[-1].span_id)
            )

        cut = next(
            (i for i, step in enumerate(prefix_all) if step.span_id == first.span_id),
            len(prefix_all),
        )
        transitions.append(
            GroundingTransition(
                transition_id=f"transition-{_digest(trace.trace_id, first.span_id)}",
                trace_id=trace.trace_id,
                lineage_id=trace.lineage_id,
                family_id=family_id,
                turn_index=index,
                task_context=task_context or (trace.task or ""),
                history_before=prefix_all[:cut],
                state_before=reconstruct_state(trace, _previous_span_id(trace, first.span_id)),
                action_span_id=first.span_id,
                action_span_ids=tuple(span.span_id for span in action),
                action_content=content,
                action_calls=calls,
                observations=tuple(observations),
                inferred_state_delta=_delta(reactions),
                observed=bool(observations),
                stripped_markers=user_markers,
            )
        )
    return tuple(transitions)


def _previous_span_id(trace: Trace, span_id: str) -> str | None:
    previous: str | None = None
    for span in trace.spans:
        if span.span_id == span_id:
            return previous
        previous = span.span_id
    return previous


def _delta(results: Sequence[Span]) -> dict[str, Any]:
    """State paths a reaction reports, without claiming they changed.

    Named a delta because that is what the AWM must later propose, but read off
    a recorded result it is only what the result said. Whether a field changed
    needs the value before it, which the caller holds and this does not.
    """
    delta: dict[str, Any] = {}
    for result in results:
        if result.status is SpanStatus.ERROR:
            continue
        delta.update(_state_paths(_json_value(result.output)))
    return delta


def cut_points(trace: Trace) -> tuple[str, ...]:
    """Spans a middle-prefix scenario may resume from.

    Every observed reaction is a decision point: the candidate has just been
    handed new information and must choose what to do with it. Deduplicated by
    the (tool, status) shape so a trace that called one lookup eight times does
    not contribute eight near-identical scenarios.
    """
    seen: set[tuple[str, bool]] = set()
    points: list[str] = []
    for span in trace.spans:
        if span.kind is SpanKind.MODEL:
            continue
        shape = (span.name, span.status is SpanStatus.ERROR)
        if shape in seen:
            continue
        seen.add(shape)
        points.append(span.span_id)
    return tuple(points)


def _partition_of(family: TaskFamily, trace_id: str, sealed: Iterable[str] = ()) -> Partition:
    if trace_id in set(sealed):
        return Partition.SEALED
    if trace_id in family.held_out_trace_ids:
        return Partition.HELD_OUT
    return Partition.FIT


def compile_scenarios(
    trace: Trace,
    *,
    family: TaskFamily,
    contract: SealedSuccessContract,
    hidden_user: Any = None,
    markers: Sequence[str] = (),
    lineage_group: Sequence[str] = (),
    sealed_trace_ids: Sequence[str] = (),
    max_middle: int = 4,
) -> tuple[Scenario, ...]:
    """Task-start, middle-prefix and end-prefix scenarios from one trace.

    Every scenario excludes its own trace and its whole lineage group from
    retrieval. Asserted here and again at query time: a shared index is exactly
    the thing that quietly stops being partitioned, and a rollout that can
    retrieve its own source can copy the real next observation while fidelity
    reports that the simulator learned something.
    """
    from bandits.diagnose.models import HiddenUserProfile

    partition = _partition_of(family, trace.trace_id, sealed_trace_ids)
    excluded = tuple(dict.fromkeys((trace.trace_id, *lineage_group)))
    profile = hidden_user if hidden_user is not None else HiddenUserProfile()

    def build(kind: ScenarioKind, cut: str | None) -> Scenario:
        return Scenario(
            scenario_id=f"scenario-{_digest(trace.trace_id, kind.value, cut or 'start')}",
            kind=kind,
            task=trace.task or "",
            system_policy=trace.system_prompt,
            offered_tools=tuple(
                schema.simulation_projection() for schema in (trace.tools_available or ())
            ),
            prefix=build_prefix(trace, cut, markers=markers) if cut else (),
            initial_state=reconstruct_state(trace, cut) if cut else ScenarioState(),
            success_contract=contract,
            hidden_user=profile,
            source_trace_id=trace.trace_id,
            source_task_id=contract.source_task_id,
            lineage_id=trace.lineage_id,
            family_id=family.family_id,
            partition=partition,
            cut_span_id=cut,
            retrieval_excluded_trace_ids=excluded,
        )

    scenarios = [build(ScenarioKind.TASK_START, None)]

    points = cut_points(trace)
    if points:
        # The last observed reaction is the end prefix: the candidate has seen
        # everything the real episode saw and must finish on its own.
        for span_id in points[:-1][:max_middle]:
            scenarios.append(build(ScenarioKind.MIDDLE_PREFIX, span_id))
        scenarios.append(build(ScenarioKind.END_PREFIX, points[-1]))

    return tuple(scenarios)
