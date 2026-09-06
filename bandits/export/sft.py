"""SFT export: verified success plus conservative demonstration-quality gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
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
    CompositionReport,
    DuplicateGroup,
    ExportBundle,
    ExportKind,
    ExportManifest,
    LengthSummary,
    Partition,
    PartitionComposition,
    RejectedTrace,
    SamplingCaps,
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

    A tool result the source never paired with a call is a defect, not a call to
    reconstruct. The name and arguments of the action are only inferable from
    what came back from it, and a row built that way would teach the model to
    commit to an action it can only justify by its outcome.
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
    if trace.user_turns:
        turns: tuple[UserTurn, ...] = trace.user_turns
    elif trace.task is not None:
        turns = (UserTurn(text=trace.task),)
    else:
        # No recorded instruction and no turns to fall back on. An empty user
        # message here would be a turn this episode never had, and a fabricated
        # prompt is worse than a refused row.
        turns = ()
        defects.append("trace records no user instruction")

    anchored: dict[str | None, list[UserTurn]] = {}
    for turn in turns:
        anchored.setdefault(turn.after_span_id, []).append(turn)
    unplaceable = sorted(
        {turn.after_span_id for turn in turns if turn.after_span_id not in by_id} - {None}
    )
    if unplaceable:
        defects.append("a recorded user turn does not sit anywhere in the trajectory")

    # Turns are replayed in span order, so anchors that run backwards would come
    # out reordered — the transcript would show the user saying things in an
    # order they never said them in. Refused rather than silently rearranged.
    order = {span.span_id: index for index, span in enumerate(trace.spans)}
    positions = [
        order.get(turn.after_span_id, -1)
        for turn in turns
        if turn.after_span_id in by_id or turn.after_span_id is None
    ]
    if any(later < earlier for earlier, later in zip(positions, positions[1:], strict=False)):
        defects.append("recorded user turns do not run forwards through the trajectory")
    if trace.unrepresented_user_turns:
        defects.append(
            f"the source recorded {trace.unrepresented_user_turns} user turn(s) this "
            "trace does not represent"
        )

    messages: list[TrainingMessage] = []
    if trace.system_prompt:
        # The instructions the episode ran under, where the source recorded
        # them. A row that omits them teaches behavior as if it were
        # unconditional, when it was a response to a policy the next run may
        # not be given.
        messages.append(TrainingMessage(role="system", content=trace.system_prompt))
    messages.extend(
        TrainingMessage(role="user", content=turn.text) for turn in anchored.get(None, ())
    )
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
            if not span.call_recorded:
                # A result the source never paired with a call. The only way to
                # put it in a transcript is to write the assistant turn that
                # would have produced it, which is a decision no one recorded
                # making — so nothing is emitted for it and the row is refused.
                defects.append(f"tool result {span.name!r} has no recorded assistant call")
                open_text_span = None
                close_turn(span.span_id)
                continue

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


_DECLARED_MODEL_KEYS = ("model", "gen_ai.request.model")
"""Where each adapter records the model the run was configured with."""

_DECLARED_SCAFFOLD_KEYS = ("scaffold", "agent_version", "framework", "version")
"""Where each adapter records the harness around the model."""


def _carrier_span_ids(trace: Trace) -> set[str]:
    """Model spans that stand for a tool call rather than for a completion.

    A source that records a call on the model span names that span after the
    *tool*, so reading every model span's name as a model reports ``refund`` as
    the thing that generated the episode.

    Recognised three ways, because the argument-based rule
    :func:`build_transcript` uses answers a different question — where the
    arguments live — and a call that took no arguments has none to find. A
    carrier is a model span an adapter marked as one, or that a tool span of the
    same name hangs off, or that holds the arguments for an otherwise bare tool
    span. A model span parenting a tool span of a *different* name is not a
    carrier: that is the OTLP shape, where the parent really is the model.
    """
    by_id = {span.span_id: span for span in trace.spans}
    carriers: set[str] = set()
    for span in trace.spans:
        if span.kind is SpanKind.MODEL and span.attributes.get("tool_call"):
            carriers.add(span.span_id)
            continue
        if span.kind is not SpanKind.TOOL:
            continue
        parent = by_id.get(span.parent_span_id or "")
        if parent is None or parent.kind is not SpanKind.MODEL:
            continue
        if parent.name == span.name or (parent.arguments and not span.arguments):
            carriers.add(parent.span_id)
    return carriers


def _declared(context: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Values a source stated outright, keeping only ones that read as a name.

    A structure under ``model`` is a shape this does not understand, and
    ``str()``-ing it would put ``{'name': 'gpt-5'}`` in the field a reader takes
    for a model. An empty or non-textual declaration is no declaration.
    """
    values: list[str] = []
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def generating_policy(trace: Trace) -> dict[str, Any]:
    """What produced this episode, preferring what the source declared outright.

    A span name is an inference; ``runtime_context`` is the source saying so.
    Where a wrapper declares ``model: gpt-5``, that is the answer, and reading
    span names instead reported a tool name as the generating model and the
    scaffold as unknown — on a row whose whole purpose is to record the policy a
    demonstration came from.
    """
    declared_models = tuple(dict.fromkeys(_declared(trace.runtime_context, _DECLARED_MODEL_KEYS)))
    carriers = _carrier_span_ids(trace)
    models = declared_models or tuple(
        dict.fromkeys(
            span.name
            for span in trace.spans
            if span.kind is SpanKind.MODEL and span.span_id not in carriers
        )
    )
    scaffolds = tuple(
        dict.fromkeys(
            _declared(trace.runtime_context, _DECLARED_SCAFFOLD_KEYS)
            + [
                str(span.attributes[key])
                for span in trace.spans
                for key in ("scaffold", "agent_version", "framework")
                if span.attributes.get(key)
            ]
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


_SELECTION_POLICY = (
    "trace_id ascending within the partition; caps are charged in that order, "
    "and only against a row that passed every other gate"
)
"""How rows are reached, which is what decides which ones a cap keeps.

Not a quality ranking: until the gates run, every trace the partition offers is
an equal candidate, so the only honest order is one that does not depend on how
the task set happened to list them.
"""


@dataclass(frozen=True)
class _RowFacts:
    """What the composition report needs about one rebuilt transcript."""

    trace_id: str
    family_id: str
    source: str
    lineage: str
    models: tuple[str, ...]
    scaffolds: tuple[str, ...]
    tool_calls: tuple[str, ...]
    messages: int
    characters: int


def _row_characters(messages: tuple[TrainingMessage, ...]) -> int:
    """Row size in the text it actually carries, and never in tokens.

    A tokenizer is not configured anywhere in this pipeline, and converting
    this with a rule of thumb would produce a number that reads as measured.
    """
    return sum(
        len(message.content or "")
        + sum(len(call.function.name) + len(call.function.arguments) for call in message.tool_calls)
        for message in messages
    )


def _row_facts(trace: Trace, family_id: str, messages: tuple[TrainingMessage, ...]) -> _RowFacts:
    policy = generating_policy(trace)
    return _RowFacts(
        trace_id=trace.trace_id,
        family_id=family_id,
        source=trace.source,
        # A trace whose source declared no lineage is its own, exactly as
        # grouping treats it. Pooling them under one key would read as a single
        # enormous retry chain and let one cap remove almost the whole dataset.
        lineage=trace.lineage_id or f"trace:{trace.trace_id}",
        models=tuple(policy["models"]),
        scaffolds=tuple(policy["scaffolds"]),
        tool_calls=tuple(call.function.name for message in messages for call in message.tool_calls),
        messages=len(messages),
        characters=_row_characters(messages),
    )


def _summarize(values: list[int]) -> LengthSummary:
    if not values:
        return LengthSummary(rows=0)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return LengthSummary(
        rows=len(ordered),
        minimum=ordered[0],
        median=float(statistics.median(ordered)),
        p90=float(ordered[index]),
        maximum=ordered[-1],
        total=sum(ordered),
    )


def _counted(values: Counter[str]) -> dict[str, int]:
    """Sorted by weight, heaviest first, so the skew is the first thing read."""
    return {key: count for key, count in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))}


def _compose(facts: list[_RowFacts]) -> PartitionComposition:
    lineages = Counter(item.lineage for item in facts)
    return PartitionComposition(
        rows=len(facts),
        rows_by_family=_counted(Counter(item.family_id for item in facts)),
        rows_by_source=_counted(Counter(item.source for item in facts)),
        rows_by_model=_counted(Counter(model for item in facts for model in item.models)),
        rows_by_scaffold=_counted(
            Counter(scaffold for item in facts for scaffold in item.scaffolds)
        ),
        rows_by_tool=_counted(Counter(tool for item in facts for tool in set(item.tool_calls))),
        tool_calls_by_tool=_counted(Counter(tool for item in facts for tool in item.tool_calls)),
        lineages=len(lineages),
        repeated_lineages=_counted(
            Counter({key: count for key, count in lineages.items() if count > 1})
        ),
        messages_per_row=_summarize([item.messages for item in facts]),
        characters_per_row=_summarize([item.characters for item in facts]),
    )


def _cap_reasons(
    facts: _RowFacts,
    caps: SamplingCaps,
    per_family: Counter[str],
    per_lineage: Counter[str],
) -> list[str]:
    """Which configured cap this row would exceed, named exactly.

    Charged only against a row that already passed every other gate: a rejected
    row that spent cap budget would silently cost a good one its place.
    """
    reasons: list[str] = []
    if (
        caps.max_rows_per_family is not None
        and per_family[facts.family_id] >= caps.max_rows_per_family
    ):
        reasons.append(
            f"family {facts.family_id} already contributes {caps.max_rows_per_family} row(s); "
            "excluded by max_rows_per_family"
        )
    if (
        caps.max_rows_per_lineage is not None
        and per_lineage[facts.lineage] >= caps.max_rows_per_lineage
    ):
        reasons.append(
            f"lineage {facts.lineage} already contributes {caps.max_rows_per_lineage} row(s); "
            "excluded by max_rows_per_lineage"
        )
    return reasons


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
    caps: SamplingCaps | None = None,
) -> ExportBundle:
    """Export demonstrations from the side of the split evaluation may not use.

    The success threshold is not a parameter here. It was frozen when the
    verifier was accepted, and a caller re-choosing it could admit as a
    demonstration a run the validation it cites recorded as a failure.

    Every gate below decides about one trace at a time, so none of them can see
    that most of what passed came from one lineage, one source or one tool. The
    bundle therefore carries a :class:`CompositionReport` describing both the
    partition offered and the rows selected, and ``caps`` — unset by default —
    is how a caller acts on what that report shows.
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

    limits = caps or SamplingCaps()
    rows: list[SFTExample] = []
    rejected: list[RejectedTrace] = []
    seen: set[str] = set()
    signatures: dict[str, list[str]] = {}
    offered_facts: list[_RowFacts] = []
    selected_facts: list[_RowFacts] = []
    rows_per_family: Counter[str] = Counter()
    rows_per_lineage: Counter[str] = Counter()
    warning_reasons: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()

    # Sorted rather than taken as listed: a cap keeps whichever rows it reaches
    # first, so the order has to be a property of the traces themselves and not
    # of how the task set happened to record them.
    for trace_id in sorted(eligible):
        warnings: tuple[str, ...] = ()
        reasons = prompt_rejection_reasons(trace_id, analysis)
        trace = traces.get(trace_id)
        messages: tuple[TrainingMessage, ...] = ()
        facts: _RowFacts | None = None
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
            facts = _row_facts(trace, family.family_id, messages)
            # Measured before any gate runs: what the partition offered is the
            # comparison that makes the selected dataset's skew readable.
            offered_facts.append(facts)
            signature = _near_duplicate_signature(messages)
            signatures.setdefault(signature, []).append(trace_id)
            if signature in seen:
                reasons.append("near-duplicate training trajectory")
            if limits.max_messages_per_row is not None and (
                facts.messages > limits.max_messages_per_row
            ):
                reasons.append(
                    f"transcript holds {facts.messages} message(s); excluded by "
                    f"max_messages_per_row of {limits.max_messages_per_row}"
                )
            if limits.max_characters_per_row is not None and (
                facts.characters > limits.max_characters_per_row
            ):
                reasons.append(
                    f"transcript holds {facts.characters} character(s); excluded by "
                    f"max_characters_per_row of {limits.max_characters_per_row}"
                )
            if not reasons:
                reasons.extend(_cap_reasons(facts, limits, rows_per_family, rows_per_lineage))
            if not reasons:
                seen.add(signature)
                rows_per_family[facts.family_id] += 1
                rows_per_lineage[facts.lineage] += 1
                selected_facts.append(facts)

        if reasons:
            deduplicated = tuple(dict.fromkeys(reasons))
            unresolved_reasons.update(deduplicated)
            rejected.append(
                RejectedTrace(
                    trace_id=trace_id,
                    family_id=family.family_id,
                    reasons=deduplicated,
                )
            )
            continue
        warning_reasons.update(warnings)

        digest = hashlib.sha256(f"{trace_id}\0{spec.verifier_id}".encode()).hexdigest()[:16]
        rows.append(
            SFTExample(
                example_id=f"sft-{digest}",
                messages=messages,
                tools=(
                    tuple(tool.model_dump(mode="json") for tool in trace.tools_available)
                    if trace.tools_available is not None
                    else None
                ),
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

    composition = CompositionReport(
        offered_traces=len(eligible),
        offered=_compose(offered_facts),
        selected=_compose(selected_facts),
        unresolved_reasons=_counted(unresolved_reasons),
        warning_reasons=_counted(warning_reasons),
        duplicate_groups=tuple(
            DuplicateGroup(signature=signature, trace_ids=tuple(members))
            for signature, members in sorted(
                signatures.items(), key=lambda item: (-len(item[1]), item[0])
            )
            if len(members) > 1
        ),
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
            caps=limits if limits.configured else None,
            selection_policy=_SELECTION_POLICY,
            composition_schema_version=composition.schema_version,
            rows=len(rows),
            unresolved=len(rejected),
            warnings=_partition_warnings(family, partition),
        ),
        rows=tuple(rows),
        unresolved=tuple(rejected),
        composition=composition,
    )
