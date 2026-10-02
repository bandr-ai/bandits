"""Workflow semantics for a trace: the invocation, its task, and evidence links.

A workflow is a program calling models, not a person talking to an agent. Its
recordings hold three different things that a conversation reader conflates:

- the **invocation** — the application run, with the request it received and
  what it delivered, recorded as one span's input and output;
- the **process** — the steps and model calls nested under it;
- the **prompts** — what the program sent each model call, which carry a
  ``user`` role only because that is the slot a chat API takes input in.

Nothing here infers who spoke or what caused what. The task comes only from a
field the ingest was told to read (``WorkflowDeclaration.task_fields``);
without one it is unresolved, never taken from a model's prompt. Evidence
links record what the recording supports and on what basis — a recorded call
id, containment, text found verbatim in a later input, a shared execution
round — and mark a match ambiguous when the text could as well have come from
another call. A relationship the recording cannot establish is simply absent.

Source-specific facts (which field holds the question) arrive as declared
paths; no application's names appear here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from bandits.ingest.otlp import _TOOL_CALL_ID_KEYS, _messages, _parts
from bandits.traces import (
    EvidenceLink,
    Span,
    SpanKind,
    WorkflowDeclaration,
    WorkflowNode,
    WorkflowRequest,
)

MIN_MATCH_CHARS = 40
"""A call output shorter than this (a label like ``benefits``) found in a later
input is recorded as an ambiguous match: short text recurs for reasons that
have nothing to do with the call. The threshold reduces false matches; it never
makes a long match a proven relationship."""

ROUND_KEY = "langgraph_step"
"""Framework metadata naming the execution round a node ran in. Nodes sharing a
round and a parent ran in the same round — not necessarily overlapping in time,
and not necessarily dependent on each other."""


_MISSING = object()


def resolve(record: dict[str, Any], path: str) -> Any:
    """The value at a dotted path into ``{"input": ..., "output": ...}``, or ``_MISSING``."""
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def resolve_task(
    record: dict[str, Any], fields: Sequence[str]
) -> tuple[str | None, str, str | None, str | None]:
    """``(task, status, path, reason)`` from the declared fields, first present wins.

    Declared only when a field resolves to a non-empty string and no other
    declared field resolves to a different one. A conflict leaves the task
    unset rather than silently picking the first.
    """
    if not fields:
        return None, "unresolved", None, "no --task-field was declared"
    found: list[tuple[str, str]] = []
    wrong_type: list[str] = []
    for path in fields:
        value = resolve(record, path)
        if value is _MISSING or value is None:
            continue
        if not isinstance(value, str):
            wrong_type.append(f"{path} is {type(value).__name__}, not a string")
            continue
        if value.strip():
            found.append((path, value))
    if not found:
        reason = "; ".join(wrong_type) or (
            f"none of {', '.join(fields)} is present and non-empty in the invocation record"
        )
        return None, "unresolved", None, reason
    if len({value for _, value in found}) > 1:
        detail = "; ".join(f"{path}={value[:80]!r}" for path, value in found)
        return None, "conflict", None, f"declared fields disagree: {detail}"
    path, value = found[0]
    return value, "declared", path, None


def build_request(
    *,
    candidates: Sequence[str],
    records: dict[str, dict[str, Any]],
    declaration: WorkflowDeclaration,
) -> WorkflowRequest:
    """Choose the invocation among outermost candidates and read its request.

    One candidate is the invocation. Several are narrowed to those where a
    declared task field resolves; if that leaves exactly one, it is chosen and
    the basis says so. Otherwise nothing is chosen — an ambiguous invocation is
    recorded, not guessed.
    """
    candidates = tuple(candidates)
    chosen: str | None = None
    if len(candidates) == 1:
        chosen, basis = candidates[0], "sole outermost non-container span"
    elif not candidates:
        basis = "no outermost span that is not a container, model call or tool call"
    else:
        resolving = [
            span_id
            for span_id in candidates
            if resolve_task(records[span_id], declaration.task_fields)[1] != "unresolved"
        ]
        if len(resolving) == 1:
            chosen, basis = resolving[0], "only candidate where a declared task field resolves"
        else:
            basis = (
                f"ambiguous: {len(candidates)} outermost candidates and "
                f"{len(resolving)} resolve a declared task field"
            )
    if chosen is None:
        return WorkflowRequest(
            source_span_id=None,
            invocation_basis=basis,
            candidate_span_ids=candidates,
            task_reason="no invocation was selected",
            origin=declaration.request_origin,
        )
    record = records[chosen]
    task, status, path, reason = resolve_task(record, declaration.task_fields)
    delivered = (
        resolve(record, declaration.delivered_field) if declaration.delivered_field else _MISSING
    )
    return WorkflowRequest(
        source_span_id=chosen,
        invocation_basis=basis,
        candidate_span_ids=candidates,
        raw_input=record.get("input"),
        raw_output=record.get("output"),
        status=record.get("status"),
        task=task,
        task_status=status,  # type: ignore[arg-type]
        task_path=path,
        task_reason=reason,
        origin=declaration.request_origin,
        delivered=None if delivered is _MISSING else delivered,
    )


def _norm(text: str) -> str:
    return " ".join(text.split())


def _leaves(value: Any) -> Iterable[str]:
    """Every string inside a value, JSON strings opened, so text is matched as written."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                yield from _leaves(json.loads(stripped))
                return
            except ValueError:
                pass
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)


def _haystack(value: Any) -> str:
    return "\n".join(_norm(leaf) for leaf in _leaves(value))


def _call_text(span: Span) -> tuple[str, ...]:
    """A model call's output as the normalized strings a later input could quote."""
    return tuple(t for t in (_norm(leaf) for leaf in _leaves(span.output)) if t)


def _span_input(span: Span) -> Any:
    if span.kind is SpanKind.MODEL:
        return span.attributes.get("gen_ai.input.messages") or span.arguments
    return span.arguments


def _output_call_ids(span: Span) -> set[str]:
    ids: set[str] = set()
    for message in _messages(span.attributes.get("gen_ai.output.messages")):
        for part in _parts(message):
            call_id = part.get("id") or part.get("tool_call_id")
            if part.get("type") == "tool_call" and isinstance(call_id, str):
                ids.add(call_id)
    return ids


def build_evidence(
    spans: Sequence[Span],
    nodes: Sequence[WorkflowNode],
    request: WorkflowRequest | None,
) -> tuple[EvidenceLink, ...]:
    """Recorded relationships from each model call to what was observed after it.

    Only later records are linked — nothing produced after a call is presented
    as something it saw. The delivered value is linked separately, as a later
    outcome, and only where it contains the call's output.
    """
    calls = [span for span in spans if span.kind is SpanKind.MODEL]
    node_by_id = {node.span_id: node for node in nodes}
    parent_of: dict[str, str | None] = {s.span_id: s.parent_span_id for s in spans}
    parent_of.update({n.span_id: n.parent_span_id for n in nodes})
    links: list[EvidenceLink] = []

    # Tool results by recorded call id, or parented to the call by the adapter.
    for call in calls:
        ids = _output_call_ids(call)
        for span in spans:
            if span.kind is not SpanKind.TOOL:
                continue
            recorded_id = next(
                (v for k in _TOOL_CALL_ID_KEYS if isinstance(v := span.attributes.get(k), str)),
                None,
            )
            if recorded_id is not None and recorded_id in ids:
                links.append(
                    EvidenceLink(
                        call_span_id=call.span_id,
                        kind="tool_result",
                        target_span_id=span.span_id,
                        basis=f"tool call id {recorded_id} recorded on both",
                    )
                )
            elif span.parent_span_id == call.span_id and span.call_recorded:
                links.append(
                    EvidenceLink(
                        call_span_id=call.span_id,
                        kind="tool_result",
                        target_span_id=span.span_id,
                        basis="tool result recorded as answering this call",
                    )
                )

    # Enclosing results: the nearest enclosing node, and the nearest one carrying
    # framework metadata when that is a different node.
    round_of: dict[str, tuple[str | None, Any]] = {}
    for call in calls:
        nearest = framed = None
        parent, seen = call.parent_span_id, set()
        while parent is not None and parent not in seen:
            seen.add(parent)
            node = node_by_id.get(parent)
            if node is not None:
                nearest = nearest or node
                if node.framework and framed is None:
                    framed = node
            parent = parent_of.get(parent)
        enclosing = [(nearest, "nearest enclosing step")]
        if framed is not nearest:
            enclosing.append((framed, "nearest enclosing framework node"))
        for node, basis in enclosing:
            if node is not None:
                links.append(
                    EvidenceLink(
                        call_span_id=call.span_id,
                        kind="enclosing_result",
                        target_span_id=node.span_id,
                        basis=f"{basis}; its output may combine other calls and code",
                    )
                )
        if framed is not None and ROUND_KEY in framed.framework:
            round_of[call.span_id] = (framed.parent_span_id, framed.framework[ROUND_KEY])

    # Same execution round.
    by_round: dict[tuple[str | None, Any], list[str]] = defaultdict(list)
    for call_id, key in round_of.items():
        by_round[key].append(call_id)
    for members in by_round.values():
        for call_id in members:
            for other in members:
                if other != call_id:
                    links.append(
                        EvidenceLink(
                            call_span_id=call_id,
                            kind="same_round",
                            target_span_id=other,
                            basis=f"same {ROUND_KEY} under the same parent: same execution "
                            "round, not proof of overlap, shared outcome or dependency",
                        )
                    )

    # Text found verbatim in a later input.
    texts = {call.span_id: _call_text(call) for call in calls}
    duplicate: dict[str, bool] = {}
    for call_id, text in texts.items():
        duplicate[call_id] = bool(text) and any(
            other != call_id and texts[other] == text for other in texts
        )
    matched: dict[str, dict[str, int]] = defaultdict(dict)  # target -> call -> chars
    short: dict[str, bool] = {}
    needles_by_call: dict[str, tuple[Span, list[str]]] = {}
    for call in calls:
        text = texts[call.span_id]
        if not text:
            continue
        long_leaves = [leaf for leaf in text if len(leaf) >= MIN_MATCH_CHARS]
        short[call.span_id] = not long_leaves
        needles_by_call[call.span_id] = (call, long_leaves or list(text))

    def targets() -> Iterator[tuple[str, Any, Any]]:
        yield from ((span.span_id, span.started_at, _span_input(span)) for span in spans)
        yield from ((node.span_id, node.started_at, node.input) for node in nodes)

    # Inputs can be cumulative and large. Expand one target at a time, then
    # release it before moving to the next rather than retaining every joined
    # haystack for the whole workflow.
    for target_id, started_at, value in targets():
        haystack = _haystack(value)
        if not haystack:
            continue
        leaves = {_norm(leaf) for leaf in _leaves(value)}
        for call_id, (call, needles) in needles_by_call.items():
            if target_id == call_id or started_at < call.ended_at:
                continue
            # Long fields can be consumed independently (one query from a list).
            # Short labels must match a complete field, never a substring such
            # as "tools" inside "powertools".
            present = [n for n in needles if (n in leaves if short[call_id] else n in haystack)]
            if present:
                matched[target_id][call_id] = sum(len(n) for n in present)
    unsure = {c: short.get(c, False) or duplicate.get(c, False) for c in texts}
    for target_id, by_call in matched.items():
        for call_id, chars in by_call.items():
            # Only unambiguous contributors make a result shared: a one-word
            # label that also appears there is no evidence of a contribution.
            others = tuple(sorted(c for c in by_call if c != call_id and not unsure[c]))
            links.append(
                EvidenceLink(
                    call_span_id=call_id,
                    kind="shared_result" if others else "text_match",
                    target_span_id=target_id,
                    basis="text from this call's output appears verbatim in the target's input"
                    + ("; so does output text from other calls" if others else ""),
                    match_chars=chars,
                    ambiguous=unsure[call_id],
                    shared_with=others,
                )
            )

    # Pipeline results can supply a model prompt without a model producing them.
    # The call is the receiver; the target identifies the earlier source record.
    sources = [(s.span_id, s.ended_at, s.output) for s in spans if s.kind is SpanKind.TOOL]
    sources += [(n.span_id, n.ended_at, n.output) for n in nodes]
    for call in calls:
        haystack = _haystack(_span_input(call))
        for source_id, ended_at, output in sources:
            if ended_at > call.started_at:
                continue
            needles = {_norm(t) for t in _leaves(output) if len(_norm(t)) >= MIN_MATCH_CHARS}
            present = [t for t in needles if t in haystack]
            if present:
                links.append(
                    EvidenceLink(
                        call_span_id=call.span_id,
                        kind="input_context",
                        target_span_id=source_id,
                        basis="text from this earlier pipeline result appears verbatim in the call's input; "
                        "containment does not prove the call used it",
                        match_chars=sum(map(len, present)),
                    )
                )

    # Delivery: the declared delivered value contains the call's output.
    if request is not None and isinstance(request.delivered, str):
        delivered = _norm(request.delivered)
        delivering = [
            call.span_id
            for call in calls
            if texts[call.span_id]
            and all(t in delivered for t in texts[call.span_id])
            and sum(len(t) for t in texts[call.span_id]) >= MIN_MATCH_CHARS
        ]
        for call_id in delivering:
            links.append(
                EvidenceLink(
                    call_span_id=call_id,
                    kind="delivery",
                    basis="the declared delivered value contains this call's output "
                    "(a later outcome, never an input to earlier calls)",
                    match_chars=sum(len(t) for t in texts[call_id]),
                    ambiguous=len(delivering) > 1 or duplicate.get(call_id, False),
                    shared_with=tuple(sorted(c for c in delivering if c != call_id)),
                )
            )
    return tuple(links)


def delivery_status(request: WorkflowRequest | None, links: Sequence[EvidenceLink]) -> str:
    """How the delivered value relates to the recorded model calls."""
    if request is None or request.delivered is None:
        return "no delivered value declared or recorded"
    delivery = [link for link in links if link.kind == "delivery"]
    if not delivery:
        return "no recorded model source"
    if len(delivery) > 1:
        return "ambiguous: several calls' outputs are contained in the delivered value"
    return f"delivered by {delivery[0].call_span_id}"
