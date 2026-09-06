from __future__ import annotations

from pathlib import Path

from bandits.ingest.chat_json import load_chat_json
from bandits.traces import SpanKind

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
FIXTURE = FIXTURES / "traces.chat.jsonl"
MULTI_TURN = FIXTURES / "traces.multiturn.chat.json"


def test_pairs_tool_call_with_tool_result() -> None:
    corpus = load_chat_json(FIXTURE)

    assert corpus.source == "chat-json"
    assert len(corpus.traces) == 1
    trace = corpus.traces[0]
    assert trace.task == "Refund order 7741"

    tool_span = next(s for s in trace.spans if s.kind is SpanKind.TOOL)
    model_span = next(s for s in trace.spans if s.kind is SpanKind.MODEL and s.arguments)
    assert tool_span.parent_span_id == model_span.span_id
    assert tool_span.arguments == {}
    assert tool_span.output == {"status": "refunded"}
    assert model_span.arguments == {"order_id": "7741"}


def test_spans_are_tagged_synthetic_time() -> None:
    corpus = load_chat_json(FIXTURE)
    assert all(s.attributes.get("synthetic_time") is True for s in corpus.traces[0].spans)


def test_no_user_message_becomes_an_issue(tmp_path) -> None:
    path = tmp_path / "no_user.json"
    path.write_text('[{"role": "assistant", "content": "hi"}]')

    corpus = load_chat_json(path)

    assert corpus.traces == ()
    assert corpus.issues[0].kind == "no_user_message"


def test_prose_tool_result_is_kept_as_text() -> None:
    """Only JSON-shaped results are parsed; prose is genuinely prose."""
    corpus = load_chat_json(FIXTURE)
    trace = corpus.traces[0]

    final_reply = trace.spans[-1]
    assert final_reply.output == "Order 7741 has been refunded."


def test_bare_message_array_declares_no_lineage() -> None:
    corpus = load_chat_json(FIXTURE)
    assert corpus.traces[0].lineage_id is None


def test_wrapped_conversation_carries_its_session_id(tmp_path) -> None:
    source = tmp_path / "chat.json"
    source.write_text(
        '[{"session_id": "sess-9", "messages": ['
        '{"role": "user", "content": "Refund order 7741"},'
        '{"role": "assistant", "content": "Done."}]}]'
    )

    corpus = load_chat_json(source)

    assert corpus.traces[0].lineage_id == "sess-9"


def test_every_user_turn_is_recorded_in_order_with_its_position() -> None:
    """A correction halfway through is why the rest of the episode looks as it does."""
    trace = load_chat_json(MULTI_TURN).traces[0]

    assert trace.task == "Update the dependency"
    assert [turn.text for turn in trace.user_turns] == [
        "Update the dependency",
        "Use version 1.9 instead",
    ]
    assert trace.user_turns[0].after_span_id is None
    assert trace.user_turns[1].after_span_id == trace.spans[0].span_id
    assert trace.unrepresented_user_turns == 0


def test_a_user_turn_that_is_not_text_is_counted_rather_than_dropped(tmp_path) -> None:
    """Counted, so an export refuses the trace instead of losing the instruction."""
    path = tmp_path / "blocks.json"
    path.write_text(
        '[{"role": "user", "content": "Update the dependency"},'
        '{"role": "assistant", "content": "done"},'
        '{"role": "user", "content": [{"type": "image", "source": "x"}]}]'
    )

    corpus = load_chat_json(path)

    assert corpus.traces[0].unrepresented_user_turns == 1
    assert any(issue.kind == "unrepresentable_user_turn" for issue in corpus.issues)


def test_a_declared_toolset_and_system_prompt_are_captured(tmp_path) -> None:
    path = tmp_path / "wrapped.json"
    path.write_text(
        '[{"session_id": "sess-1", "model": "gpt-5", "temperature": 0.2,'
        ' "tools": [{"type": "function", "function": {"name": "refund_order",'
        ' "parameters": {"type": "object"}}}],'
        ' "messages": ['
        '{"role": "system", "content": "You are a support agent."},'
        '{"role": "user", "content": "Refund order 7741"},'
        '{"role": "assistant", "content": "Done."}]}]'
    )

    trace = load_chat_json(path).traces[0]

    assert trace.tools_available is not None
    assert [tool.name for tool in trace.tools_available] == ["refund_order"]
    assert trace.system_prompt == "You are a support agent."
    assert trace.runtime_context == {"model": "gpt-5", "temperature": 0.2}


def test_an_undeclared_toolset_stays_unknown() -> None:
    """A bare message array has nowhere to say what was on offer."""
    trace = load_chat_json(FIXTURE).traces[0]

    assert trace.tools_available is None
    assert trace.system_prompt is None
    assert trace.runtime_context == {}


def test_a_multimodal_instruction_is_refused_rather_than_crashing(tmp_path) -> None:
    """The first user turn is not always text, and ingest must survive that."""
    path = tmp_path / "multimodal.json"
    path.write_text(
        '[{"role": "user", "content": [{"type": "image", "source": "x"}]},'
        '{"role": "assistant", "content": "sure"}]'
    )

    corpus = load_chat_json(path)

    assert corpus.traces == ()
    assert {issue.kind for issue in corpus.issues} == {
        "unrepresentable_user_turn",
        "no_user_message",
    }


def test_a_later_multimodal_turn_is_counted_not_skipped(tmp_path) -> None:
    path = tmp_path / "later.json"
    path.write_text(
        '[{"role": "user", "content": "do the thing"},'
        '{"role": "assistant", "content": "working"},'
        '{"role": "user", "content": [{"type": "image", "source": "x"}]},'
        '{"role": "assistant", "content": "done"}]'
    )

    trace = load_chat_json(path).traces[0]

    assert trace.task == "do the thing"
    assert trace.unrepresented_user_turns == 1


def test_every_system_and_developer_instruction_is_kept(tmp_path) -> None:
    """The developer message is usually the one carrying the constraint."""
    path = tmp_path / "system.json"
    path.write_text(
        '[{"messages": ['
        '{"role": "system", "content": "be terse"},'
        '{"role": "developer", "content": "never refund over 100"},'
        '{"role": "user", "content": "refund 7741"},'
        '{"role": "assistant", "content": "done"}]}]'
    )

    trace = load_chat_json(path).traces[0]

    assert trace.system_prompt == "be terse\n\nnever refund over 100"


def test_an_explicitly_empty_toolset_survives_the_adapter(tmp_path) -> None:
    """`"tools": []` is falsey, and reading it as absent would say nobody declared one."""
    path = tmp_path / "empty_tools.json"
    path.write_text(
        '[{"tools": [], "messages": ['
        '{"role": "user", "content": "refund 7741"},'
        '{"role": "assistant", "content": "I cannot, I have no tools."}]}]'
    )

    trace = load_chat_json(path).traces[0]

    assert trace.tools_available == ()
