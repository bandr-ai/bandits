from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest.chat_json import load_chat_json
from bandits.traces import SpanKind

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
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


def test_a_paired_tool_result_records_that_its_call_was_recorded() -> None:
    corpus = load_chat_json(FIXTURE)

    tool_span = next(s for s in corpus.traces[0].spans if s.kind is SpanKind.TOOL)
    assert tool_span.call_recorded is True


def test_a_result_with_no_call_id_is_kept_but_marked_unpaired(tmp_path) -> None:
    """The observation is real; the call that produced it is not recorded anywhere."""
    path = tmp_path / "orphan.json"
    path.write_text(
        '[{"role": "user", "content": "Refund order 7741"},'
        ' {"role": "tool", "name": "refund", "content": "{\\"status\\": \\"refunded\\"}"},'
        ' {"role": "assistant", "content": "done"}]'
    )

    corpus = load_chat_json(path)

    tool_span = next(s for s in corpus.traces[0].spans if s.kind is SpanKind.TOOL)
    assert tool_span.output == {"status": "refunded"}
    assert tool_span.call_recorded is False
    assert any(issue.kind == "unpaired_tool_result" for issue in corpus.issues)


def test_a_result_answering_an_unknown_call_id_is_marked_unpaired(tmp_path) -> None:
    """An id naming a call this conversation never contains pairs with nothing."""
    path = tmp_path / "unknown.json"
    path.write_text(
        '[{"role": "user", "content": "Refund order 7741"},'
        ' {"role": "tool", "tool_call_id": "call-elsewhere", "name": "refund",'
        '  "content": "{\\"status\\": \\"refunded\\"}"},'
        ' {"role": "assistant", "content": "done"}]'
    )

    corpus = load_chat_json(path)

    assert next(s for s in corpus.traces[0].spans if s.kind is SpanKind.TOOL).call_recorded is False


def test_a_call_id_can_pair_with_only_one_result(tmp_path) -> None:
    path = tmp_path / "reused.json"
    path.write_text(
        '[{"role": "user", "content": "Refund order 7741"},'
        ' {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",'
        '  "function": {"name": "refund", "arguments": "{}"}}]},'
        ' {"role": "tool", "tool_call_id": "c1", "name": "refund", "content": "first"},'
        ' {"role": "tool", "tool_call_id": "c1", "name": "refund", "content": "second"}]'
    )

    corpus = load_chat_json(path)

    tool_spans = [s for s in corpus.traces[0].spans if s.kind is SpanKind.TOOL]
    assert [span.call_recorded for span in tool_spans] == [True, False]
    assert any(issue.kind == "unpaired_tool_result" for issue in corpus.issues)


def test_a_result_cannot_replace_the_recorded_call_name(tmp_path) -> None:
    path = tmp_path / "mismatched-name.json"
    path.write_text(
        '[{"role": "user", "content": "Charge order 7741"},'
        ' {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",'
        '  "function": {"name": "charge", "arguments": "{}"}}]},'
        ' {"role": "tool", "tool_call_id": "c1", "name": "refund", "content": "done"}]'
    )

    corpus = load_chat_json(path)

    tool_span = next(s for s in corpus.traces[0].spans if s.kind is SpanKind.TOOL)
    assert tool_span.name == "charge"
    assert tool_span.call_recorded is False
    assert any(issue.kind == "mismatched_tool_result" for issue in corpus.issues)


_CONVERSATION = [
    {"role": "user", "content": "Refund order 7741"},
    {"role": "assistant", "content": "Done."},
]


def _traces(tmp_path, name: str, text: str) -> tuple[int, list[str]]:
    path = tmp_path / name
    path.write_text(text)
    corpus = load_chat_json(path)
    return len(corpus.traces), [i.kind for i in corpus.issues]


def _jsonl(*records: object) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


def test_json_lines_of_wrapped_conversations_are_read(tmp_path) -> None:
    wrapped = {"id": "a", "messages": _CONVERSATION}
    assert _traces(tmp_path, "c.jsonl", _jsonl(wrapped, wrapped)) == (2, [])


def test_json_lines_of_bare_message_arrays_are_read(tmp_path) -> None:
    assert _traces(tmp_path, "c.jsonl", _jsonl(_CONVERSATION, _CONVERSATION, _CONVERSATION)) == (
        3,
        [],
    )


def test_an_array_of_message_arrays_is_read(tmp_path) -> None:
    text = json.dumps([_CONVERSATION, _CONVERSATION])
    assert _traces(tmp_path, "c.json", text) == (2, [])


def test_one_message_per_line_is_one_conversation(tmp_path) -> None:
    assert _traces(tmp_path, "c.jsonl", _jsonl(*_CONVERSATION)) == (1, [])


def test_an_array_of_wrapped_conversations_is_still_read(tmp_path) -> None:
    text = json.dumps([{"messages": _CONVERSATION}, {"messages": _CONVERSATION}])
    assert _traces(tmp_path, "c.json", text) == (2, [])


def test_a_bad_json_line_is_an_issue_with_its_line_number(tmp_path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(_jsonl(_CONVERSATION) + "{broken\n" + _jsonl(_CONVERSATION))

    corpus = load_chat_json(path)

    assert len(corpus.traces) == 2
    assert [(i.kind, i.location) for i in corpus.issues] == [("malformed_json", f"{path}:2")]


def test_an_unrecognized_item_is_an_issue_not_a_silent_skip(tmp_path) -> None:
    text = json.dumps([_CONVERSATION, "oops", 3])
    count, kinds = _traces(tmp_path, "c.json", text)
    assert (count, kinds) == (1, ["unrecognized_record", "unrecognized_record"])


def test_a_pretty_printed_malformed_document_is_not_read_as_lines(tmp_path) -> None:
    count, kinds = _traces(tmp_path, "c.json", '{\n  "messages": [\n')
    assert (count, kinds) == (0, ["malformed_json"])


def _spans(corpus):
    return [(s.kind.name, s.name, s.parent_span_id is not None) for s in corpus.traces[0].spans]


def test_text_said_before_a_tool_call_is_its_own_span_ahead_of_the_call(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "refund 7"},
                {
                    "role": "assistant",
                    "content": "Let me check the order first.",
                    "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "{}"},
                {"role": "assistant", "content": "Done."},
            ]
        )
    )
    corpus = load_chat_json(path)
    spans = corpus.traces[0].spans
    assert [s.name for s in spans] == ["assistant", "lookup", "lookup", "assistant"]
    assert spans[0].output == "Let me check the order first."
    assert spans[2].parent_span_id == spans[1].span_id
    assert corpus.issues == ()


def test_legacy_function_call_and_function_result_pair_by_name(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "refund 7"},
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "lookup", "arguments": '{"id": "7"}'},
                },
                {"role": "function", "name": "lookup", "content": '{"status": "paid"}'},
                {"role": "assistant", "content": "Done."},
            ]
        )
    )
    corpus = load_chat_json(path)
    assert _spans(corpus) == [
        ("MODEL", "lookup", False),
        ("TOOL", "lookup", True),
        ("MODEL", "assistant", False),
    ]
    assert corpus.traces[0].spans[1].output == {"status": "paid"}
    assert corpus.issues == ()


def test_an_unknown_role_is_an_issue_and_the_rest_of_the_conversation_is_kept(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "hi"},
                {"role": "narrator", "content": "meanwhile"},
                {"role": "assistant", "content": "hello"},
            ]
        )
    )
    corpus = load_chat_json(path)
    assert len(corpus.traces) == 1
    assert [i.kind for i in corpus.issues] == ["unrecognized_role"]
    assert "narrator" in corpus.issues[0].detail


def test_anthropic_content_blocks_become_text_calls_and_results(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "refund 7"}]},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "hmm"},
                                {"type": "text", "text": "checking"},
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "lookup",
                                    "input": {"id": "7"},
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "t1", "content": "paid"}
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                    ]
                }
            ]
        )
    )
    corpus = load_chat_json(path)
    trace = corpus.traces[0]
    assert trace.task == "refund 7"
    assert [(s.kind.name, s.name) for s in trace.spans] == [
        ("MODEL", "assistant"),
        ("MODEL", "lookup"),
        ("TOOL", "lookup"),
        ("MODEL", "assistant"),
    ]
    assert trace.spans[0].attributes["thinking"] == "hmm"
    assert trace.spans[1].arguments == {"id": "7"}
    assert trace.spans[2].parent_span_id == trace.spans[1].span_id
    assert corpus.issues == ()


def test_a_user_turn_with_an_image_block_is_still_counted_not_flattened(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "see"}, {"type": "image", "source": {}}],
                },
                {"role": "assistant", "content": "ok"},
            ]
        )
    )
    corpus = load_chat_json(path)
    assert [i.kind for i in corpus.issues] == ["unrepresentable_user_turn"]


@pytest.mark.parametrize(
    "export",
    [
        pytest.param(
            [{"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}],
            id="sharegpt",
        ),
        pytest.param(
            [
                {
                    "contents": [
                        {"role": "user", "parts": [{"text": "q"}]},
                        {"role": "model", "parts": [{"text": "a"}]},
                    ]
                }
            ],
            id="gemini",
        ),
        pytest.param(
            [{"type": "human", "content": "q"}, {"type": "ai", "content": "a"}], id="langchain"
        ),
        pytest.param(
            [
                {
                    "id": "1",
                    "turns": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ],
                }
            ],
            id="turns",
        ),
        pytest.param(
            [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}], id="text-key"
        ),
    ],
)
def test_other_exports_chat_shapes_are_read(tmp_path, export) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(export))
    corpus = load_chat_json(path)
    assert [t.task for t in corpus.traces] == ["q"]
    assert corpus.issues == ()


def test_gemini_function_call_and_response_pair_by_name(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "refund 7"}]},
                    {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "lookup", "args": {"id": "7"}}}],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": "lookup", "response": {"status": "paid"}}}
                        ],
                    },
                    {"role": "model", "parts": [{"text": "Done."}]},
                ]
            }
        )
    )
    corpus = load_chat_json(path)
    assert _spans(corpus) == [
        ("MODEL", "lookup", False),
        ("TOOL", "lookup", True),
        ("MODEL", "assistant", False),
    ]
    assert corpus.traces[0].spans[1].output == {"status": "paid"}


def test_a_claude_code_event_is_not_taken_for_a_chat_message(tmp_path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"type": "user", "sessionId": "s", "message": {"role": "user", "content": "x"}})
    )
    corpus = load_chat_json(path)
    assert corpus.traces == ()


def test_jsonl_with_raw_unicode_line_separators_and_a_bom_is_read(tmp_path) -> None:
    path = tmp_path / "c.jsonl"
    rows = [
        {
            "id": str(i),
            "messages": [
                {"role": "user", "content": f"a b {i}"},
                {"role": "assistant", "content": "ok"},
            ],
        }
        for i in range(2)
    ]
    path.write_bytes(
        b"\xef\xbb\xbf" + "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()
    )
    corpus = load_chat_json(path)
    assert [t.task for t in corpus.traces] == ["a b 0", "a b 1"]
    assert corpus.issues == ()


def test_an_empty_file_is_an_issue(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text("  \n")
    corpus = load_chat_json(path)
    assert corpus.traces == () and [i.kind for i in corpus.issues] == ["empty_file"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"messages": "oops"}, id="string"),
        pytest.param({"messages": [1, 2, 3]}, id="list-of-numbers"),
        pytest.param({"messages": 5}, id="number"),
        pytest.param({"messages": None}, id="null"),
        pytest.param({"messages": {"role": "user", "content": "x"}}, id="object"),
        pytest.param([{"messages": "oops"}, {"messages": [1, 2, 3]}], id="list-of-bad-wrappers"),
    ],
)
def test_a_wrapper_whose_messages_are_not_messages_is_an_issue_not_a_crash(
    tmp_path, payload
) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload))

    corpus = load_chat_json(path)

    assert corpus.traces == ()
    kinds = {i.kind for i in corpus.issues}
    assert "unrecognized_record" in kinds
    assert "no_user_message" in kinds


def test_issue_details_name_the_message_by_its_place_in_the_export(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "first"},
                {"role": "narrator", "content": "skipped"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image"}]},
                {"role": "assistant", "content": "ok"},
            ]
        )
    )

    corpus = load_chat_json(path)

    details = {i.kind: i.detail for i in corpus.issues}
    assert "message 1" in details["unrecognized_role"]
    assert "message 3" in details["unrepresentable_user_turn"]
