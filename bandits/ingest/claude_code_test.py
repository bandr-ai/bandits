from __future__ import annotations

from pathlib import Path

from bandits.ingest.claude_code import load_claude_code
from bandits.traces import SpanKind

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MULTI_TURN = FIXTURES / "session.multiturn.jsonl"


def test_a_later_correction_is_recorded_as_its_own_turn() -> None:
    trace = load_claude_code(MULTI_TURN).traces[0]

    assert trace.task == "Update the dependency"
    assert [turn.text for turn in trace.user_turns] == [
        "Update the dependency",
        "Use version 1.9 instead",
    ]
    assert trace.user_turns[1].after_span_id == trace.spans[0].span_id


def test_a_record_carrying_only_a_tool_result_is_not_a_user_turn() -> None:
    """The harness answering the agent is not the user saying something."""
    trace = load_claude_code(MULTI_TURN).traces[0]

    assert len(trace.user_turns) == 2
    assert any(span.kind is SpanKind.TOOL for span in trace.spans)
