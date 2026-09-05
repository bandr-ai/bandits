"""SFT export: verified success plus conservative demonstration-quality gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any

from bandits.analyze.families import normalize_instruction
from bandits.analyze.models import CorpusAnalysis, TaskSet
from bandits.export.eval import (
    _check_lineage,
    _partition_warnings,
    partition_trace_ids,
    prompt_rejection_reasons,
)
from bandits.export.models import (
    ExportBundle,
    ExportKind,
    ExportManifest,
    Partition,
    RejectedTrace,
    SFTExample,
    ToolCall,
    ToolFunction,
    TrainingMessage,
)
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, UserTurn
from bandits.verify.execute import execute_verifier
from bandits.verify.review import ReviewedVerifier


def _content(value: Any) -> str:
    """Render a span payload as message text, never as a structure.

    Chat formats carry strings. A dict left in ``content`` is silently
    ``str()``-ed by one trainer, rejected by another, and round-trips as neither.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _call_arguments(span: Span, carrier: Span | None) -> dict[str, Any]:
    """Recover the arguments an action was invoked with, from either trace shape.

    Sources disagree about where a call lives. Some record the arguments on the
    tool span itself; others record them on the model span that emitted the call
    and leave the tool span holding only the result. Reading just one of those
    shapes silently exports trajectories with no actions in them.
    """
    if span.arguments:
        return span.arguments
    if carrier is not None:
        return carrier.arguments
    return {}


def _tool_call_id(span: Span) -> str:
    """Stable within a trace, and derived from a span so two runs never collide."""
    return f"call-{span.span_id}"


def build_transcript(
    trace: Trace,
) -> tuple[tuple[TrainingMessage, ...], tuple[str, ...], tuple[str, ...]]:
    """Rebuild one episode as a chat transcript, with its defects and its warnings.

    Defects disqualify the row: emitting it would put something in the transcript
    the trace never recorded. Warnings travel with the row instead, because the
    demonstration is still faithful to what the agent did.

    Calls are never batched into a shared assistant turn. Sources hang every tool
    span off one root model span, so a shared parent says only that both calls
    happened inside that span, not that they were issued together. Batching them
    would teach an agent to commit to its second action before reading the result
    of its first.

    Every user turn the source recorded is replayed where it arrived. A run whose
    user said "use 1.9 instead" halfway through is a demonstration of following
    that correction, and a transcript holding only the opening instruction would
    teach the final action as the answer to a question nobody asked.
    """
    by_id = {span.span_id: span for span in trace.spans}
    children = {span.parent_span_id for span in trace.spans if span.parent_span_id}

    # A model span is a call carrier only when a tool span hangs off it and that
    # tool span recorded no arguments of its own. Parenthood alone is not enough:
    # a model span's own arguments may be the prompt.
    carriers: dict[str, Span] = {}
    for span in trace.spans:
        if span.kind is not SpanKind.TOOL or span.arguments:
            continue
        parent = by_id.get(span.parent_span_id or "")
        if parent is not None and parent.kind is SpanKind.MODEL and parent.arguments:
            carriers[span.span_id] = parent
    carrier_ids = {carrier.span_id for carrier in carriers.values()}

    defects: list[str] = []
    warnings: list[str] = []
    turns = trace.user_turns or (UserTurn(text=trace.task if trace.task is not None else ""),)
    anchored: dict[str | None, list[UserTurn]] = {}
    for turn in turns:
        anchored.setdefault(turn.after_span_id, []).append(turn)
    unplaceable = sorted(
        {turn.after_span_id for turn in turns if turn.after_span_id not in by_id} - {None}
    )
    if unplaceable:
        defects.append("a recorded user turn does not sit anywhere in the trajectory")
    if trace.unrepresented_user_turns:
        defects.append(
            f"the source recorded {trace.unrepresented_user_turns} user turn(s) this "
            "trace does not represent"
        )

    messages: list[TrainingMessage] = [
        TrainingMessage(role="user", content=turn.text) for turn in anchored.get(None, ())
    ]
    open_text_span: str | None = None
    """The span behind the last message, when that message is assistant text."""

    def close_turn(span_id: str) -> None:
        """Replay any user turn that arrived after this span."""
        nonlocal open_text_span
        for turn in anchored.get(span_id, ()):
            messages.append(TrainingMessage(role="user", content=turn.text))
            # What the user said next is not part of the turn before it, so the
            # next action cannot ride on that assistant message.
            open_text_span = None

    for span in trace.spans:
        if span.kind is SpanKind.TOOL:
            carrier = carriers.get(span.span_id)
            call = ToolCall(
                id=_tool_call_id(span),
                function=ToolFunction(
                    name=span.name,
                    arguments=json.dumps(
                        _call_arguments(span, carrier), sort_keys=True, default=str
                    ),
                ),
            )
            if span.output is None:
                # Emitting a result the source never recorded would train the
                # model on an observation that did not happen.
                defects.append(f"tool call {span.name!r} has no recorded result")

            owner = carrier.span_id if carrier is not None else span.parent_span_id
            if open_text_span is not None and owner == open_text_span:
                # Text from the span that emitted this call is the same turn, so
                # it rides on the call rather than becoming a second assistant
                # message in a row.
                spoken = messages.pop()
                messages.append(
                    TrainingMessage(
                        role="assistant",
                        content=spoken.content,
                        name=spoken.name,
                        tool_calls=(call,),
                    )
                )
            else:
                messages.append(TrainingMessage(role="assistant", tool_calls=(call,)))
            open_text_span = None
            messages.append(
                TrainingMessage(
                    role="tool",
                    name=span.name,
                    tool_call_id=call.id,
                    content=_content(span.output),
                )
            )
            close_turn(span.span_id)
            continue

        if span.output is not None:
            messages.append(
                TrainingMessage(role="assistant", content=_content(span.output), name=span.name)
            )
            open_text_span = span.span_id
        elif span.span_id not in carrier_ids and span.arguments and span.span_id not in children:
            defects.append("a recorded model action has neither a completion nor a tool result")
        close_turn(span.span_id)

    if not messages:
        defects.append("the trace records no messages to rebuild")
    elif messages[-1].role == "user":
        # The last thing recorded is an instruction nobody answered. There is no
        # behavior to imitate after it, and training on it would teach the model
        # that a request can end a conversation.
        defects.append("episode ends on a user turn with no recorded response")
    elif messages[-1].role == "tool":
        # Not a defect. The actions are still exactly what the agent did, and the
        # verifier has already established the outcome; what is missing is the
        # agent's own closing turn, which most exporters simply do not record.
        warnings.append("episode ends on a tool result; the closing turn was not recorded")
    return tuple(messages), tuple(dict.fromkeys(defects)), tuple(dict.fromkeys(warnings))


def generating_policy(trace: Trace) -> dict[str, Any]:
    models = tuple(dict.fromkeys(span.name for span in trace.spans if span.kind is SpanKind.MODEL))
    scaffolds = tuple(
        dict.fromkeys(
            str(span.attributes[key])
            for span in trace.spans
            for key in ("scaffold", "agent_version", "framework")
            if span.attributes.get(key)
        )
    )
    return {"models": models, "scaffolds": scaffolds or ("unknown",)}


def _quality_reasons(
    trace: Trace, messages: tuple[TrainingMessage, ...], max_steps: int
) -> list[str]:
    """Whether this run is worth imitating, given that it already passed the verifier."""
    reasons: list[str] = []
    if len(trace.spans) > max_steps:
        reasons.append(f"episode has {len(trace.spans)} spans; family quality limit is {max_steps}")
    if any(span.status is SpanStatus.ERROR for span in trace.spans):
        reasons.append("episode contains an error or recovery path")

    # Read repeats off the reconstructed calls, not off the tool spans. Two of
    # the three source shapes leave a tool span's arguments empty, which makes
    # every call in an episode look identical to every other.
    calls = [
        (call.function.name, call.function.arguments)
        for message in messages
        for call in message.tool_calls
    ]
    repeated = sorted({name for (name, _), count in Counter(calls).items() if count > 1})
    if repeated:
        reasons.append(f"episode repeats the same tool action: {', '.join(repeated)}")
    if not any(
        message.role == "assistant" and (message.content is not None or message.tool_calls)
        for message in messages[1:]
    ):
        # A turn that only calls a tool is a target like any other. Requiring
        # text here would reject exactly the trajectories worth training on.
        reasons.append("episode has no assistant target to imitate")
    if not generating_policy(trace)["models"]:
        reasons.append("generating model is not recorded")
    return reasons


def _structured(value: str) -> Any | None:
    """Parse a JSON container back out of a string field, or return None.

    Tool arguments and results reach the transcript already serialized. Without
    this, two runs of the same task differing only by an order id compare as
    different text and both survive deduplication.
    """
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _normalized(value: Any, field: str | None = None) -> Any:
    if field and (field == "id" or field.endswith("_id")) and isinstance(value, (str, int)):
        return "<id>"
    if isinstance(value, str):
        structured = _structured(value)
        if structured is not None:
            return _normalized(structured, field)
        return normalize_instruction(value)
    if isinstance(value, dict):
        return {key: _normalized(item, key) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalized(item, field) for item in value]
    return value


def _near_duplicate_signature(messages: tuple[TrainingMessage, ...]) -> str:
    normalized = [_normalized(message.model_dump(mode="json")) for message in messages]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def build_sft_export(
    corpus: TraceCorpus,
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    reviewed: ReviewedVerifier,
    reviewed_id: str,
    *,
    partition: Partition = Partition.FIT,
    max_median_multiplier: float = 1.5,
) -> ExportBundle:
    """Export demonstrations from the side of the split evaluation may not use.

    The success threshold is not a parameter here. It was frozen when the
    verifier was accepted, and a caller re-choosing it could admit as a
    demonstration a run the validation it cites recorded as a failure.
    """
    success_threshold = reviewed.success_threshold
    if max_median_multiplier < 1:
        raise ValueError("max median multiplier must be at least 1")
    _check_lineage(corpus, task_set, task_set_id, analysis, reviewed)
    spec = reviewed.spec
    family = task_set.family_by_id()[spec.family_id]
    traces = {trace.trace_id: trace for trace in corpus.traces}
    eligible = partition_trace_ids(family, partition)
    # The step-count bound is a property of the family, so it is measured across
    # the whole family rather than the partition being exported. Measuring it per
    # partition would move the bar depending on which side a caller asked for,
    # and the same trajectory would pass one export and fail the other.
    family_traces = [traces[item] for item in family.trace_ids if item in traces]
    lengths = sorted(len(trace.spans) for trace in family_traces)
    median = statistics.median(lengths) if lengths else 0
    max_steps = max(1, math.ceil(median * max_median_multiplier))
    evidence_by_trace: dict[str, list] = {}
    for item in analysis.evidence:
        evidence_by_trace.setdefault(item.trace_id, []).append(item)

    rows: list[SFTExample] = []
    rejected: list[RejectedTrace] = []
    seen: set[str] = set()
    for trace_id in eligible:
        warnings: tuple[str, ...] = ()
        reasons = prompt_rejection_reasons(trace_id, analysis)
        trace = traces.get(trace_id)
        messages: tuple[TrainingMessage, ...] = ()
        if trace is None:
            reasons.append("corpus contains no matching trace")
        else:
            result = execute_verifier(spec, tuple(evidence_by_trace.get(trace_id, ())))
            if result.score is None:
                reasons.append("reviewed verifier could not score the trajectory")
            elif result.score < success_threshold:
                reasons.append("reviewed verifier did not establish success")
            messages, defects, warnings = build_transcript(trace)
            reasons.extend(defects)
            reasons.extend(_quality_reasons(trace, messages, max_steps))
            signature = _near_duplicate_signature(messages)
            if signature in seen:
                reasons.append("near-duplicate training trajectory")
            if not reasons:
                seen.add(signature)

        if reasons:
            rejected.append(
                RejectedTrace(
                    trace_id=trace_id,
                    family_id=family.family_id,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
            continue

        digest = hashlib.sha256(f"{trace_id}\0{spec.verifier_id}".encode()).hexdigest()[:16]
        rows.append(
            SFTExample(
                example_id=f"sft-{digest}",
                messages=messages,
                generating_policy=generating_policy(trace),  # type: ignore[arg-type]
                warnings=warnings,
                corpus_id=task_set.corpus_id,
                task_set_id=task_set_id,
                family_id=family.family_id,
                trace_id=trace_id,
                verifier_id=spec.verifier_id,
                validation_id=reviewed.validation_id,
            )
        )

    return ExportBundle(
        manifest=ExportManifest(
            format=ExportKind.SFT,
            corpus_id=task_set.corpus_id,
            task_set_id=task_set_id,
            reviewed_verifier_id=reviewed_id,
            verifier_id=spec.verifier_id,
            validation_id=reviewed.validation_id,
            human_acceptance_id=reviewed.human_acceptance_id,
            verifier_status=spec.status.value,
            accepted_risks=tuple(item.code for item in reviewed.accepted_risks),
            partition=partition,
            partition_trace_count=len(eligible),
            success_threshold=success_threshold,
            max_median_multiplier=max_median_multiplier,
            rows=len(rows),
            unresolved=len(rejected),
            warnings=_partition_warnings(family, partition),
        ),
        rows=tuple(rows),
        unresolved=tuple(rejected),
    )
