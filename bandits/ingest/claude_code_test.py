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


def test_an_init_record_declares_the_toolset_the_session_started_with(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        '{"type":"system","subtype":"init","cwd":"/repo","model":"claude",'
        '"tools":["Bash","Read"]}\n'
        '{"type":"user","sessionId":"s","message":{"role":"user","content":"fix the test"}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"done"}]}}\n'
    )

    trace = load_claude_code(path).traces[0]

    assert trace.tools_available is not None
    assert [tool.name for tool in trace.tools_available] == ["Bash", "Read"]
    assert all(tool.parameters is None for tool in trace.tools_available)
    assert trace.runtime_context["cwd"] == "/repo"


def test_a_log_without_an_init_record_says_nothing_about_the_toolset() -> None:
    """Never reconstructed from the calls that happen to appear."""
    trace = load_claude_code(MULTI_TURN).traces[0]

    assert trace.tools_available is None


def test_a_non_text_user_turn_is_counted_and_reported(tmp_path) -> None:
    """An image the agent acted on cannot reach a transcript, so the row must fail closed."""
    path = tmp_path / "image.jsonl"
    path.write_text(
        '{"type":"user","sessionId":"s","message":{"role":"user","content":"do the thing"}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"working"}]}}\n'
        '{"type":"user","sessionId":"s","message":{"role":"user",'
        '"content":[{"type":"image","source":{"data":"..."}}]}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"done"}]}}\n'
    )

    corpus = load_claude_code(path)

    assert corpus.traces[0].unrepresented_user_turns == 1
    assert any(issue.kind == "unrepresentable_user_turn" for issue in corpus.issues)


def test_a_tool_result_record_is_still_not_counted_against_the_trace() -> None:
    """The harness answering the agent is neither a turn nor a loss."""
    trace = load_claude_code(MULTI_TURN).traces[0]

    assert trace.unrepresented_user_turns == 0
