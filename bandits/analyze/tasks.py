"""Turn one trace into one task candidate.

Deterministic and lossy in only one direction: anything the source did not
record stays missing and is named in ``limitations``. A trace with no declared
instruction produces a task candidate that says so, not an invented instruction.
"""

from __future__ import annotations

from bandits.analyze.models import (
    Evidence,
    TaskCandidate,
    Visibility,
    build_task_candidate,
    evidence_id,
)
from bandits.analyze.terminal import terminal_spans
from bandits.traces import SpanKind, Trace


def _observed(
    *,
    trace_id: str,
    claim: str,
    value: object,
    visibility: Visibility,
    strength: str,
    span_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id(trace_id=trace_id, claim=claim, span_id=span_id),
        claim=claim,
        value=value,
        visibility=visibility,
        provenance="observed",
        strength=strength,  # type: ignore[arg-type]
        trace_id=trace_id,
        span_id=span_id,
    )


def extract_task(trace: Trace) -> tuple[TaskCandidate, tuple[Evidence, ...]]:
    """Extract one task candidate and the evidence it is built from."""
    limitations: list[str] = []
    prompt_evidence: list[Evidence] = []
    context_evidence: list[Evidence] = []

    if trace.task is not None:
        prompt_evidence.append(
            _observed(
                trace_id=trace.trace_id,
                claim="instruction",
                value=trace.task,
                visibility=Visibility.AT_START,
                strength="strong",
            )
        )
    else:
        limitations.append("source declared no instruction; the task input is unknown")

    if not trace.spans:
        limitations.append("trace has no spans; no trajectory or outcome can be read")

    if trace.tools_available is not None:
        # The real fact a prompt may carry: what was on offer before anything
        # happened. Schemas travel with it, because a call to a tool the harness
        # cannot define is not reproducible.
        prompt_evidence.append(
            _observed(
                trace_id=trace.trace_id,
                claim="available_tools",
                value=[tool.model_dump(mode="json") for tool in trace.tools_available],
                visibility=Visibility.AT_START,
                strength="strong",
            )
        )
        undefined = [tool.name for tool in trace.tools_available if tool.parameters is None]
        if undefined:
            limitations.append(
                f"{len(undefined)} offered tool(s) were named without a schema; "
                "a call to them may not be reproducible"
            )

    if trace.system_prompt is not None:
        prompt_evidence.append(
            _observed(
                trace_id=trace.trace_id,
                claim="system_prompt",
                value=trace.system_prompt,
                visibility=Visibility.AT_START,
                strength="strong",
            )
        )
    else:
        limitations.append(
            "no system prompt is recorded; the instructions this episode ran under are unknown"
        )

    if trace.runtime_context:
        context_evidence.append(
            _observed(
                trace_id=trace.trace_id,
                claim="runtime_context",
                value=dict(sorted(trace.runtime_context.items())),
                visibility=Visibility.AT_START,
                strength="strong",
            )
        )

    tools_called = tuple(dict.fromkeys(s.name for s in trace.spans if s.kind is SpanKind.TOOL))
    if tools_called:
        # Deliberately `during`, not `at_start`: the tools an agent happened to
        # call is not the same fact as the toolset it was offered, and treating
        # it as prompt context would hand a new attempt the answer's shape.
        context_evidence.append(
            _observed(
                trace_id=trace.trace_id,
                claim="tools_called",
                value=list(tools_called),
                visibility=Visibility.DURING,
                strength="strong",
            )
        )
        if trace.tools_available is None:
            limitations.append(
                "available toolset is not recorded; only the tools this episode called are known"
            )

    terminal = terminal_spans(trace.spans)
    terminal_ids = tuple(s.span_id for s in terminal)
    trajectory_ids = tuple(s.span_id for s in trace.spans if s.span_id not in set(terminal_ids))

    context_evidence.append(
        _observed(
            trace_id=trace.trace_id,
            claim="episode_span_count",
            value=len(trace.spans),
            visibility=Visibility.POST_HOC,
            strength="strong",
        )
    )

    task = build_task_candidate(
        task_id=f"task-{trace.trace_id}",
        trace_id=trace.trace_id,
        lineage_id=trace.lineage_id,
        instruction=trace.task,
        prompt_evidence=tuple(prompt_evidence),
        trajectory_span_ids=trajectory_ids,
        terminal_span_ids=terminal_ids,
        outcome_evidence=(),
        limitations=tuple(limitations),
    )
    return task, tuple(prompt_evidence + context_evidence)
