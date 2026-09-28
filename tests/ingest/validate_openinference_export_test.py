from scripts.validate_openinference_export import (
    _decoded_messages,
    _decoded_tool_names,
    _source_messages,
    _source_tool_names,
)


def test_oracle_compares_tool_result_and_tool_call_without_relabeling_them() -> None:
    source = {
        "llm.input_messages.0.message.role": "assistant",
        "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
        "llm.input_messages.1.message.role": "tool",
        "llm.input_messages.1.message.content": '{"ok": true}',
    }
    decoded = [
        {"role": "assistant", "parts": [{"type": "tool_call", "name": "lookup"}]},
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call-1", "result": '{"ok": true}'}],
        },
    ]
    assert _source_messages(source, "input") == _decoded_messages(decoded)
    assert _source_tool_names(source, "input") == _decoded_tool_names(decoded)
