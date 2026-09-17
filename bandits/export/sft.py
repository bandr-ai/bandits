"""Rebuild one episode as a chat transcript, and record what generated it.

Source-agnostic on purpose: both the next-state SFT exporter
(``export/nextstate_sft.py``) and any future exporter need the same rebuild,
and neither should own it twice.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from bandits.export.models import ToolCall, ToolFunction, TrainingMessage
from bandits.traces import Span, SpanKind, SpanStatus, Trace, UserTurn


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
