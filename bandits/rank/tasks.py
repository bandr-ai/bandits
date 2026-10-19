"""Turn recorded traces into rank tasks: ``{id, input, payload?, expected?}`` for a rank run."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bandits.traces import SpanKind, Trace


def task_from_trace(trace: Trace) -> dict[str, Any] | None:
    """One task from one trace, or None when the trace has no instruction to replay.

    ``input`` is what the agent is given: its system prompt, the opening user turn and the tools
    that were on offer. ``expected`` keeps what actually happened (the recorded final output and the
    tool calls, in order) so a customer check, or a reference judge, can compare against it. The
    recorded outcome is evidence about the original run, not a score for the new one.
    """
    opening = next((t.text for t in trace.user_turns if t.after_span_id is None), None)
    user = opening or trace.task
    if not user:
        return None
    tools = [t.offered_projection() for t in trace.tools_available or ()]
    calls = [
        {"tool": s.name, "arguments": s.arguments, "status": s.status.value}
        for s in trace.spans
        if s.kind is SpanKind.TOOL
    ]
    model_spans = [s for s in trace.spans if s.kind is SpanKind.MODEL]
    final = model_spans[-1].output if model_spans else None
    return {
        "id": trace.trace_id,
        "input": {"system": trace.system_prompt, "user": user, "tools": tools},
        "expected": {
            "final_output": final,
            "tool_calls": calls,
            "recorded_model": trace.runtime_context.get("model"),
        },
    }


def tasks_from_traces(traces: Iterable[Trace], limit: int = 100) -> list[dict[str, Any]]:
    """Up to ``limit`` tasks, one per trace, skipping traces with no instruction.

    Traces that share a lineage (a retry chain) contribute one task, so the same request is not
    counted twice.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in traces:
        if trace.lineage_id:
            if trace.lineage_id in seen:
                continue
            seen.add(trace.lineage_id)
        task = task_from_trace(trace)
        if task is not None:
            out.append(task)
        if len(out) >= limit:
            break
    return out
