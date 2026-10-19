from datetime import UTC, datetime

from bandits.rank.tasks import task_from_trace, tasks_from_traces
from bandits.traces import Span, SpanKind, ToolSchema, Trace, UserTurn

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _span(**kw):
    base = {"span_id": "s", "kind": SpanKind.MODEL, "name": "m", "started_at": T0, "ended_at": T0}
    return Span(**{**base, **kw})


def _trace(trace_id="t1", **kw):
    base = {
        "trace_id": trace_id,
        "source": "otlp",
        "source_digest": "a" * 64,
        "spans": (
            _span(span_id="s1", output="calling"),
            _span(span_id="s2", kind=SpanKind.TOOL, name="lookup", arguments={"id": "7"}),
            _span(span_id="s3", output="Refunded."),
        ),
    }
    return Trace(**{**base, **kw})


def test_a_task_carries_the_prompt_tools_and_what_happened():
    trace = _trace(
        task="fallback",
        system_prompt="You are support.",
        user_turns=(UserTurn(text="Refund order 7"),),
        tools_available=(
            ToolSchema(name="lookup", description="d", parameters={"type": "object"}),
        ),
        runtime_context={"model": "gpt-x"},
    )
    task = task_from_trace(trace)
    assert task["id"] == "t1"
    assert task["input"] == {
        "system": "You are support.",
        "user": "Refund order 7",
        "tools": [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}],
    }
    assert task["expected"]["final_output"] == "Refunded."
    assert task["expected"]["tool_calls"] == [
        {"tool": "lookup", "arguments": {"id": "7"}, "status": "ok"}
    ]
    assert task["expected"]["recorded_model"] == "gpt-x"


def test_traces_without_an_instruction_are_skipped_and_retries_count_once():
    traces = [
        _trace("a", task="x", lineage_id="L"),
        _trace("b", task="x", lineage_id="L"),
        _trace("c"),
        _trace("d", task="y"),
    ]
    assert [t["id"] for t in tasks_from_traces(traces)] == ["a", "d"]
    assert len(tasks_from_traces([_trace(str(i), task="x") for i in range(5)], limit=2)) == 2
