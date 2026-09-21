from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bandits.export import ToolCall, ToolFunction, TrainingMessage, build_transcript
from bandits.export.sft import _quality_reasons, generating_policy
from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.claude_code import load_claude_code
from bandits.ingest.otlp import load_otlp
from bandits.traces import Span, SpanKind, SpanStatus, Trace, UserTurn


def _trace(
    trace_id: str,
    order: int,
    status: str,
    *,
    tool_status: SpanStatus = SpanStatus.OK,
    extra_models: int = 0,
    carrier_args: bool = False,
) -> Trace:
    """One episode: a model turn that calls a tool, the result, then a closing turn.

    ``carrier_args`` switches between the two shapes real adapters produce. With
    it, the call arguments sit on the model span and the tool span holds only the
    result, as the chat-JSON and Claude Code adapters record them; without it the
    tool span carries its own arguments, as OTLP does.
    """
    started = datetime(2026, 1, 1, tzinfo=UTC)
    call_span_id = f"{trace_id}-call"
    spans: list[Span] = [
        Span(
            span_id=call_span_id,
            kind=SpanKind.MODEL,
            name="change_order" if carrier_args else "model-v1",
            started_at=started,
            ended_at=started + timedelta(seconds=1),
            # In the OTLP shape this is the prompt, not a call. A builder that
            # reads it as call arguments would export the prompt as the action.
            arguments={"order_id": order} if carrier_args else {"prompt": "change it"},
            attributes={"tool_call": True} if carrier_args else {},
        )
    ]
    for index in range(extra_models):
        spans.append(
            Span(
                span_id=f"{trace_id}-extra-{index}",
                parent_span_id=call_span_id,
                kind=SpanKind.MODEL,
                name="model-v1",
                started_at=started + timedelta(seconds=2 + index),
                ended_at=started + timedelta(seconds=3 + index),
                output=f"extra {index}",
            )
        )
    spans.append(
        Span(
            span_id=f"{trace_id}-tool",
            parent_span_id=call_span_id,
            kind=SpanKind.TOOL,
            name="change_order",
            started_at=started + timedelta(seconds=20),
            ended_at=started + timedelta(seconds=21),
            status=tool_status,
            arguments={} if carrier_args else {"order_id": order},
            output={"status": status, "order_id": order},
        )
    )
    spans.append(
        Span(
            span_id=f"{trace_id}-final",
            kind=SpanKind.MODEL,
            name="model-v1",
            started_at=started + timedelta(seconds=22),
            ended_at=started + timedelta(seconds=23),
            output="completed",
        )
    )
    return Trace(
        trace_id=trace_id,
        source="otlp",
        source_digest=trace_id.ljust(64, "0"),
        task=f"Change order {order}",
        spans=tuple(spans),
    )


FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
SUPPORT_FIXTURE = FIXTURES / "traces.support.otlp.jsonl"
MULTI_TURN_FIXTURE = FIXTURES / "traces.multiturn.chat.json"
MULTI_TURN_SESSION = FIXTURES / "session.multiturn.jsonl"


def test_transcript_puts_the_action_on_the_assistant_turn() -> None:
    """The call is a target, not context: it must not arrive as a tool message."""
    messages, defects, warnings = build_transcript(_trace("good-1", 100, "changed"))

    assert not defects and not warnings
    assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
    call = messages[1].tool_calls[0]
    assert call.function.name == "change_order"
    assert json.loads(call.function.arguments) == {"order_id": 100}
    assert messages[2].tool_call_id == call.id
    assert not messages[2].tool_calls


def test_a_later_user_turn_is_replayed_where_it_arrived() -> None:
    """The correction has to sit before the action it caused, not be dropped."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]

    messages, defects, warnings = build_transcript(trace)

    assert not defects and not warnings
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].content == "Use version 1.9 instead"
    assert json.loads(messages[3].tool_calls[0].function.arguments) == {"version": "1.9"}


def test_a_multi_turn_session_log_replays_its_correction_too() -> None:
    messages, defects, _ = build_transcript(load_claude_code(MULTI_TURN_SESSION).traces[0])

    assert not defects
    assert [message.content for message in messages if message.role == "user"] == [
        "Update the dependency",
        "Use version 1.9 instead",
    ]


def test_a_trace_that_lost_a_user_turn_is_refused() -> None:
    """Fail closed: the actions answered something this transcript cannot show."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]

    _, defects, _ = build_transcript(trace.replace(unrepresented_user_turns=1))

    assert any("does not represent" in defect for defect in defects)


def test_a_user_turn_with_no_recorded_response_is_refused() -> None:
    """A dangling instruction has no behavior to imitate after it."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]
    dangling = trace.replace(
        user_turns=(
            *trace.user_turns,
            UserTurn(text="also bump the lockfile", after_span_id=trace.spans[-1].span_id),
        )
    )

    _, defects, _ = build_transcript(dangling)

    assert any("no recorded response" in defect for defect in defects)


def test_a_trace_with_no_instruction_is_refused_rather_than_given_a_blank_turn() -> None:
    """A blank opening message is a turn the episode never had."""
    trace = _trace("good-1", 100, "changed").replace(task=None)

    _, defects, _ = build_transcript(trace)

    assert any("no user instruction" in defect for defect in defects)


def test_user_turns_that_run_backwards_are_refused_not_reordered() -> None:
    """Replay follows span order, so a backwards anchor would rewrite the conversation."""
    trace = _trace("good-1", 100, "changed")
    reversed_turns = trace.replace(
        user_turns=(
            UserTurn(text="second thought", after_span_id=trace.spans[-1].span_id),
            UserTurn(text="do it", after_span_id=trace.spans[0].span_id),
        )
    )

    _, defects, _ = build_transcript(reversed_turns)

    assert any("do not run forwards" in defect for defect in defects)


def test_an_export_refuses_a_session_whose_user_turn_was_not_text(tmp_path) -> None:
    """End to end from a real session log, not a hand-built trace."""
    path = tmp_path / "image.jsonl"
    path.write_text(
        '{"type":"user","sessionId":"s","message":{"role":"user","content":"change order 100"}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"tool_use","id":"tu-1","name":"change_order",'
        '"input":{"order_id":100}}]}}\n'
        '{"type":"user","sessionId":"s","message":{"role":"user","content":'
        '[{"type":"tool_result","tool_use_id":"tu-1","content":'
        '"{\\"status\\": \\"changed\\", \\"order_id\\": 100}"}]}}\n'
        '{"type":"user","sessionId":"s","message":{"role":"user",'
        '"content":[{"type":"image","source":{"data":"..."}}]}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"done"}]}}\n'
    )
    trace = load_claude_code(path).traces[0].replace(trace_id="good-1", source="otlp")

    _, defects, _ = build_transcript(trace)

    assert any("does not represent" in defect for defect in defects)


def test_a_user_turn_anchored_to_no_span_is_refused() -> None:
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]
    orphaned = trace.replace(
        user_turns=(*trace.user_turns, UserTurn(text="and pin it", after_span_id="not-a-span"))
    )

    _, defects, _ = build_transcript(orphaned)

    assert any("does not sit anywhere" in defect for defect in defects)


def test_a_single_turn_transcript_is_unchanged_by_recorded_turns() -> None:
    """A source that records its one user turn must export what it always did."""
    trace = _trace("good-1", 100, "changed")

    recorded = trace.replace(user_turns=(UserTurn(text=trace.task),))

    assert build_transcript(recorded) == build_transcript(trace)


@pytest.mark.parametrize("carrier_args", [False, True])
def test_both_recorded_call_shapes_yield_the_same_transcript(carrier_args: bool) -> None:
    """OTLP puts arguments on the tool span; chat JSON puts them on the model span."""
    messages, defects, _ = build_transcript(_trace("t", 100, "changed", carrier_args=carrier_args))

    assert not defects
    assert json.loads(messages[1].tool_calls[0].function.arguments) == {"order_id": 100}


def test_a_model_prompt_is_never_exported_as_the_tool_call() -> None:
    """The OTLP model span carries a prompt; only the tool span carries the action."""
    messages, _, _ = build_transcript(_trace("t", 100, "changed"))

    arguments = [json.loads(call.function.arguments) for call in messages[1].tool_calls]
    assert arguments == [{"order_id": 100}]
    assert all("prompt" not in item for item in arguments)


def test_transcript_reports_a_truncated_episode_instead_of_exporting_it() -> None:
    trace = _trace("t", 100, "changed")
    truncated = trace.replace(spans=trace.spans[:-1])

    messages, defects, warnings = build_transcript(truncated)

    # Not a defect: the actions are still exactly what the agent did.
    assert messages[-1].role == "tool"
    assert not defects
    assert any("closing turn" in warning for warning in warnings)


def test_transcript_reports_a_call_whose_result_was_never_recorded() -> None:
    trace = _trace("t", 100, "changed")
    spans = tuple(
        span.replace(output=None) if span.span_id == "t-tool" else span for span in trace.spans
    )

    _, defects, _ = build_transcript(trace.replace(spans=spans))

    assert any("no recorded result" in defect for defect in defects)


def test_tool_message_may_not_announce_tool_calls() -> None:
    call = ToolCall(id="call-1", function=ToolFunction(name="x", arguments="{}"))
    with pytest.raises(ValidationError, match="only an assistant message"):
        TrainingMessage(role="tool", tool_call_id="call-1", content="{}", tool_calls=(call,))


def test_emitted_jsonl_is_clean_chat_completions() -> None:
    """The stored artifact stays lossless; the emitted row carries no empty padding."""
    messages, defects, _ = build_transcript(_trace("t", 100, "changed"))
    assert not defects

    rendered = [message.as_chat_message() for message in messages]
    assert rendered[0] == {"role": "user", "content": "Change order 100"}
    assert "tool_calls" not in rendered[0]
    assert "content" not in rendered[1] and rendered[1]["tool_calls"][0]["type"] == "function"
    assert set(rendered[2]) == {"role", "content", "name", "tool_call_id"}
    assert all(isinstance(row.get("content", ""), str) for row in rendered)


def test_repeated_action_is_detected_when_arguments_live_on_the_model_span() -> None:
    """The carrier shape leaves tool spans argument-less; repeats must still be seen."""
    trace = _trace("t", 100, "changed", carrier_args=True)
    doubled = trace.replace(
        spans=trace.spans
        + tuple(span.replace(span_id=f"{span.span_id}-again") for span in trace.spans[:2])
    )
    messages, _, _ = build_transcript(doubled)

    assert len(_quality_reasons(doubled, messages, max_steps=99)) == 1
    assert "repeats the same tool action" in _quality_reasons(doubled, messages, 99)[0]


def test_step_count_gate_quarantines_a_long_episode() -> None:
    long_trace = _trace("good-1", 100, "changed", extra_models=6)
    messages, _, _ = build_transcript(long_trace)

    reasons = _quality_reasons(long_trace, messages, max_steps=3)

    assert any("family quality limit" in reason for reason in reasons)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SUPPORT_FIXTURE = FIXTURES / "traces.support.otlp.jsonl"
MULTI_TURN_FIXTURE = FIXTURES / "traces.multiturn.chat.json"
MULTI_TURN_SESSION = FIXTURES / "session.multiturn.jsonl"


def test_a_real_corpus_still_yields_trainable_rows() -> None:
    """A gate strict enough to quarantine every real trace is a broken gate.

    Every episode in this corpus ends on a tool result, because that is what the
    exporter recorded. Treating that as disqualifying rather than as a warning
    would silently reduce the whole export to nothing.
    """
    corpus = load_otlp(SUPPORT_FIXTURE)
    refunds = [trace for trace in corpus.traces if trace.trace_id.startswith("refund-")]

    exported = 0
    for trace in refunds:
        messages, defects, warnings = build_transcript(trace)
        if defects:
            continue
        exported += 1
        assert messages[-1].role == "tool"
        assert any("closing turn" in warning for warning in warnings)
        # lookup then refund, as two turns: the second call must follow the
        # observation that justified it, never ride alongside the first.
        calls = [call.function.name for message in messages for call in message.tool_calls]
        assert calls == ["lookup_order", "refund_order"]
        assert all(len(message.tool_calls) <= 1 for message in messages)

    assert exported == len(refunds) - 2, "only the two traces with no recorded result are defective"


def test_sequential_calls_are_never_batched_into_one_turn() -> None:
    """A shared parent span means 'emitted during', not 'issued together'."""
    trace = next(item for item in load_otlp(SUPPORT_FIXTURE).traces if item.trace_id == "refund-1")
    assert {span.parent_span_id for span in trace.spans[1:]} == {trace.spans[0].span_id}

    messages, _, _ = build_transcript(trace)

    announcing = [message for message in messages if message.tool_calls]
    assert len(announcing) == 2
    assert messages.index(announcing[1]) > messages.index(
        next(m for m in messages if m.role == "tool")
    )


def _chat_trace(tmp_path, name: str, messages: str, trace_id: str) -> Trace:
    """One chat-JSON conversation, ingested rather than hand-built.

    The pairing state these tests are about is decided by the adapter, so a
    hand-assembled trace would only ever prove that the flag it was given is the
    flag the exporter read.
    """
    path = tmp_path / f"{name}.json"
    path.write_text(messages)
    return load_chat_json(path).traces[0].replace(trace_id=trace_id)


_ORPHANED_CONVERSATION = (
    '[{"role": "user", "content": "Change order 100"},'
    ' {"role": "tool", %s"name": "change_order",'
    '  "content": "{\\"status\\": \\"changed\\", \\"order_id\\": 100}"},'
    ' {"role": "assistant", "content": "completed"}]'
)


@pytest.mark.parametrize(
    ("name", "id_field"),
    [("missing_id", ""), ("unknown_id", '"tool_call_id": "c-elsewhere", ')],
)
def test_a_tool_result_answering_no_recorded_call_is_never_given_one(
    tmp_path, name: str, id_field: str
) -> None:
    """The exporter may not reach backwards from a result to the call for it."""
    trace = _chat_trace(tmp_path, name, _ORPHANED_CONVERSATION % id_field, "good-1")

    messages, defects, _ = build_transcript(trace)

    assert any("has no recorded assistant call" in defect for defect in defects)
    assert not any(message.tool_calls for message in messages)
    assert not any(message.role == "tool" for message in messages)


_PAIRED_CONVERSATION = (
    '[{"role": "user", "content": "Change order 100"},'
    ' {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function":'
    '  {"name": "change_order", "arguments": "{\\"order_id\\": 100}"}}]},'
    ' {"role": "tool", "tool_call_id": "c1", "name": "change_order",'
    '  "content": "{\\"status\\": \\"changed\\", \\"order_id\\": 100}"},'
    ' {"role": "assistant", "content": "completed"}]'
)


def test_a_declared_model_beats_a_span_name_that_is_really_a_tool(tmp_path) -> None:
    """A chat tool call is a MODEL span named after the tool, not after the model."""
    trace = _chat_trace(
        tmp_path,
        "declared",
        json.dumps(
            [
                {
                    "session_id": "s-1",
                    "model": "gpt-5",
                    "scaffold": "agent-v2",
                    "messages": json.loads(_PAIRED_CONVERSATION),
                }
            ]
        ),
        "good-1",
    )

    policy = generating_policy(trace)

    assert trace.runtime_context["model"] == "gpt-5"
    assert policy["models"] == ("gpt-5",)
    assert policy["scaffolds"] == ("agent-v2",)
    assert "change_order" not in policy["models"]


def test_with_nothing_declared_a_call_carrier_is_not_read_as_the_model(tmp_path) -> None:
    """A bare message array has nowhere to declare a policy, and must not invent one."""
    trace = _chat_trace(tmp_path, "bare", _PAIRED_CONVERSATION, "good-1")

    policy = generating_policy(trace)

    assert trace.runtime_context == {}
    assert "change_order" not in policy["models"]
    assert policy["models"] == ("assistant",)


def test_an_otlp_trace_still_reports_the_model_its_spans_name() -> None:
    """The fallback is unchanged where span names really are model names."""
    assert generating_policy(_trace("good-1", 100, "changed"))["models"] == ("model-v1",)


def _chat_policy(tmp_path, name: str, wrapper: dict, arguments: str = "{}") -> dict:
    path = tmp_path / f"{name}.json"
    path.write_text(
        json.dumps(
            [
                {
                    **wrapper,
                    "messages": [
                        {"role": "user", "content": "Refund order 7741"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "refund", "arguments": arguments},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "name": "refund", "content": "{}"},
                        {"role": "assistant", "content": "done"},
                    ],
                }
            ]
        )
    )
    return generating_policy(load_chat_json(path).traces[0])


def test_a_call_taking_no_arguments_is_still_not_read_as_the_model(tmp_path) -> None:
    """The argument-based carrier rule has nothing to find when a call took none."""
    policy = _chat_policy(tmp_path, "zero_arg", {"session_id": "s"}, arguments="{}")

    assert "refund" not in policy["models"]


@pytest.mark.parametrize(
    ("name", "declared"),
    [("empty", ""), ("structured", {"name": "gpt-5"}), ("numeric", 0)],
)
def test_a_declaration_that_does_not_read_as_a_name_is_not_one(
    tmp_path, name: str, declared: object
) -> None:
    """str()-ing a structure would put {'name': 'gpt-5'} where a reader expects a model."""
    policy = _chat_policy(tmp_path, f"declared_{name}", {"session_id": "s", "model": declared})

    assert policy["models"] == ("assistant",)
    assert "refund" not in policy["models"]


def test_a_claude_code_tool_call_span_is_not_read_as_the_model(tmp_path) -> None:
    """That adapter marks its carriers outright; the marking is honoured."""
    path = tmp_path / "session.jsonl"
    path.write_text(
        '{"type":"user","sessionId":"s","message":{"role":"user","content":"change order 100"}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant","model":"claude-opus-5",'
        '"content":[{"type":"tool_use","id":"tu-1","name":"change_order","input":{}}]}}\n'
        '{"type":"user","sessionId":"s","message":{"role":"user","content":'
        '[{"type":"tool_result","tool_use_id":"tu-1","content":"{}"}]}}\n'
        '{"type":"assistant","sessionId":"s","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"done"}]}}\n'
    )

    policy = generating_policy(load_claude_code(path).traces[0])

    assert "change_order" not in policy["models"]


def test_a_new_tool_schema_field_cannot_leak_into_the_published_export() -> None:
    """D68. Adding ``output_schema`` to ``ToolSchema`` silently changed four
    payloads, because each dumped the whole contract. The published shape is a
    declared projection: nulls are kept, the simulator-only field is not.
    """
    from bandits.traces import ToolSchema

    tool = ToolSchema(
        name="change_order",
        parameters={"type": "object"},
        output_schema={"type": "object"},
    )
    assert tool.offered_projection() == {
        "name": "change_order",
        "description": None,
        "parameters": {"type": "object"},
    }
    # The simulator needs the result schema; the published shape must not carry it.
    assert tool.simulation_projection()["output_schema"] == {"type": "object"}
    assert "output_schema" not in tool.offered_projection()


def test_a_declared_boolean_output_schema_survives_the_simulation_projection() -> None:
    """D63 keeps boolean schemas, including ``False``. A projection written as
    an omit-empties rule would drop the wrong things; this one is explicit.
    """
    from bandits.traces import ToolSchema

    tool = ToolSchema(name="x", parameters={}, output_schema=False)
    assert tool.simulation_projection()["output_schema"] is False
