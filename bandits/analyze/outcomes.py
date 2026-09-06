"""Read outcome evidence off a trace, without deciding the outcome.

Every detector here answers "what did the source actually record?", never "did
this run succeed?". A tool result carrying a terminal-looking status is evidence
about a field, not a verdict; collapsing the two is the mistake that makes a
corpus look better than it was. Which fields mean success is a question for a
verifier, drafted and then argued with — not for a reader of spans.
"""

from __future__ import annotations

from typing import Any

from bandits.analyze.models import Evidence, EvidenceKind, Visibility, evidence_id
from bandits.analyze.terminal import terminal_spans
from bandits.traces import Span, SpanKind, SpanStatus, Trace

_EXIT_CODE_KEYS = ("exit_code", "exitcode", "returncode", "return_code")
"""Keys that carry a process result across the exporters seen so far."""

_SCORE_KEYS = ("score", "rating", "feedback", "user_feedback", "evaluation")
"""Keys under which a source may have recorded its own judgement of the run."""

_SCALAR = (str, int, float, bool)
"""Only scalars become state fields. A nested object is not a comparable value."""

_MAX_DEPTH = 4
"""How far into a nested result to walk. Deeper than this is a document, not state."""

_MAX_FIELDS = 64
"""Leaves read from one result. A large payload is read in part, and says so."""


def _evidence(
    span: Span | None,
    *,
    trace_id: str,
    claim: str,
    value: Any,
    visibility: Visibility,
    strength: str,
    kind: EvidenceKind = EvidenceKind.OBSERVED_TRACE,
    detail: str | None = None,
) -> Evidence:
    span_id = span.span_id if span is not None else None
    return Evidence(
        evidence_id=evidence_id(trace_id=trace_id, claim=claim, span_id=span_id, detail=detail),
        claim=claim,
        value=value,
        visibility=visibility,
        provenance="observed",
        strength=strength,  # type: ignore[arg-type]
        kind=kind,
        trace_id=trace_id,
        span_id=span_id,
    )


def _first_key(payload: Any, keys: tuple[str, ...]) -> tuple[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return key, payload[key]
    return None


def _qualified(span: Span, key: str) -> str:
    """The field name a check refers to: the reporting tool, then the key.

    Two tools both reporting ``status: "done"`` are two different facts. Named by
    the key alone they group as one candidate and draft as one check, which then
    passes on whichever tool happened to answer first.
    """
    return f"{_escape(span.name)}.{key}"


def _escape(segment: str) -> str:
    """Make one path segment unambiguous before joining it with dots.

    Without this, the literal key ``{"a.b": 1}`` and the nested ``{"a": {"b": 2}}``
    produce the same path, the same evidence id, and one of the two values
    silently wins. Keys carrying dots are rare and the escaping is invisible
    until one appears, which is the point.
    """
    return segment.replace("\\", "\\\\").replace(".", "\\.")


def _path(prefix: str, key: str) -> str:
    return f"{prefix}.{_escape(key)}" if prefix else _escape(key)


def _state_fields(payload: Any) -> tuple[list[tuple[str, Any]], bool]:
    """Every comparable leaf a result holds, addressed by its path in the result.

    A nested object is still not a comparable value — the leaves inside it are,
    and reading only the top level meant a source that answers
    ``{"order": {"status": "refunded"}}`` produced no terminal evidence at all
    and drafted nothing. The object is walked; the scalars in it are the fields.

    Arrays contribute their length and nothing else. Position in a list is not
    stable between runs of the same task, so a check on the third element is a
    check on whichever item happened to land third, which reads as a real rule
    right up until the ordering changes.

    The walk is bounded in depth and in count, and the second return value says
    whether a bound was hit — a field missing from a truncated read is unknown,
    not absent.
    """
    fields: list[tuple[str, Any]] = []
    truncated = False

    def walk(node: Any, prefix: str, depth: int) -> None:
        nonlocal truncated
        if len(fields) >= _MAX_FIELDS:
            truncated = True
            return
        if isinstance(node, _SCALAR):
            if prefix:
                fields.append((prefix, node))
            return
        if isinstance(node, (list, tuple)):
            fields.append((f"{prefix}[].count", len(node)))
            return
        if not isinstance(node, dict):
            # None, or something no exporter has shown us yet. Absent stays absent.
            return
        if depth >= _MAX_DEPTH:
            truncated = truncated or bool(node)
            return
        for key, value in sorted(node.items(), key=lambda pair: str(pair[0])):
            if len(fields) >= _MAX_FIELDS:
                # Stop walking, not just stop emitting: a result with a very
                # large map should not cost a full traversal per span.
                truncated = True
                return
            walk(value, _path(prefix, str(key)), depth + 1)

    walk(payload, "", 0)
    return fields[:_MAX_FIELDS], truncated


def extract_outcome_evidence(trace: Trace) -> tuple[Evidence, ...]:
    """Collect every recorded signal about how this episode went."""
    found: dict[str, Evidence] = {}
    terminal_ids = {span.span_id for span in terminal_spans(trace.spans)}
    first_tool_span_id = next(
        (span.span_id for span in trace.spans if span.kind is SpanKind.TOOL), None
    )
    if first_tool_span_id in terminal_ids:
        # A single-tool episode has no "before". Recording its one result as both
        # initial and final state would make every invariant over it compare a
        # value to itself and pass for free.
        first_tool_span_id = None

    def record(evidence: Evidence) -> None:
        found.setdefault(evidence.evidence_id, evidence)

    for span in trace.spans:
        is_terminal = span.span_id in terminal_ids
        visibility = Visibility.TERMINAL if is_terminal else Visibility.DURING

        if span.status is SpanStatus.ERROR:
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="span_error",
                    value={"name": span.name, "kind": span.kind.value},
                    visibility=visibility,
                    strength="strong",
                    kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                )
            )

        exit_code = _first_key(span.output, _EXIT_CODE_KEYS) or _first_key(
            span.attributes, _EXIT_CODE_KEYS
        )
        if exit_code is not None:
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="command_exit_code",
                    value={"key": exit_code[0], "value": exit_code[1], "tool": span.name},
                    visibility=visibility,
                    strength="strong",
                    kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                )
            )

        recorded_score = _first_key(span.attributes, _SCORE_KEYS)
        if recorded_score is not None:
            # Moderate, not strong: the source recorded a judgement but not what
            # produced it, so its trustworthiness is unknown until someone says.
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="recorded_score",
                    value={"key": recorded_score[0], "value": recorded_score[1]},
                    visibility=Visibility.POST_HOC,
                    strength="moderate",
                    kind=EvidenceKind.TRUSTED_EVALUATOR,
                )
            )

        if span.kind is SpanKind.TOOL and is_terminal:
            # Every comparable leaf the terminal tool reported, not just its
            # status. A check comparing an amount against what was charged needs
            # the amount, and recording only one field per span made the whole
            # class of before/after invariants impossible to express.
            fields, truncated = _state_fields(span.output)
            if span.output is not None and not fields:
                # A result was recorded and nothing in it can be compared: prose,
                # or a shape no rule here reads. Named, because "no terminal
                # evidence" otherwise looks identical to a tool that answered
                # nothing, and the two call for different work — one needs a
                # judge or a domain extractor, the other needs the export fixed.
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="unstructured_final_result",
                        value={"tool": span.name, "type": type(span.output).__name__},
                        visibility=Visibility.TERMINAL,
                        strength="strong",
                    )
                )
            if truncated:
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="truncated_outcome_fields",
                        value={
                            "tool": span.name,
                            "max_fields": _MAX_FIELDS,
                            "max_depth": _MAX_DEPTH,
                        },
                        visibility=Visibility.TERMINAL,
                        strength="strong",
                    )
                )
            for key, value in fields:
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="final_state_field",
                        detail=_qualified(span, key),
                        value={
                            "key": key,
                            "field": _qualified(span, key),
                            "value": value,
                            "tool": span.name,
                        },
                        visibility=Visibility.TERMINAL,
                        strength="moderate",
                        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                    )
                )

        if span.kind is SpanKind.TOOL and span.span_id == first_tool_span_id:
            # The state the episode started from, as the first tool observed it.
            # Knowable only during the run, never at_start: the agent had to call
            # a tool to learn it, so a prompt may not contain it.
            initial_fields, initial_truncated = _state_fields(span.output)
            if initial_truncated:
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="truncated_outcome_fields",
                        value={
                            "tool": span.name,
                            "max_fields": _MAX_FIELDS,
                            "max_depth": _MAX_DEPTH,
                        },
                        visibility=Visibility.DURING,
                        strength="strong",
                    )
                )
            for key, value in initial_fields:
                record(
                    _evidence(
                        span,
                        trace_id=trace.trace_id,
                        claim="initial_state_field",
                        detail=_qualified(span, key),
                        value={
                            "key": key,
                            "field": _qualified(span, key),
                            "value": value,
                            "tool": span.name,
                        },
                        visibility=Visibility.DURING,
                        strength="moderate",
                        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
                    )
                )

        if span.kind is SpanKind.MODEL and is_terminal and span.output:
            # An agent saying it finished is a claim about the episode, recorded
            # so a verifier can see it and rank it last — never so it can count.
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="final_output",
                    value={"output": span.output},
                    visibility=Visibility.TERMINAL,
                    strength="weak",
                    kind=EvidenceKind.AGENT_SELF_REPORT,
                )
            )

        if span.kind is SpanKind.TOOL and span.output is None and span.status is SpanStatus.OK:
            # An absent result is absent. It is never an empty successful result.
            record(
                _evidence(
                    span,
                    trace_id=trace.trace_id,
                    claim="missing_tool_result",
                    value={"tool": span.name},
                    visibility=visibility,
                    strength="strong",
                )
            )

    return tuple(found.values())
