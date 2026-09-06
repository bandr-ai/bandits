from __future__ import annotations

from bandits.ingest.toolsets import parse_toolset


def test_reads_the_openai_function_wrapper() -> None:
    tools = parse_toolset(
        [
            {
                "type": "function",
                "function": {
                    "name": "refund_order",
                    "description": "Refund an order",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            }
        ]
    )

    assert tools is not None
    assert tools[0].name == "refund_order"
    assert tools[0].description == "Refund an order"
    assert tools[0].parameters == {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
    }


def test_reads_an_anthropic_input_schema() -> None:
    tools = parse_toolset([{"name": "Edit", "input_schema": {"type": "object"}}])

    assert tools is not None
    assert tools[0].parameters == {"type": "object"}


def test_a_bare_name_keeps_no_schema_rather_than_an_empty_one() -> None:
    """An undefined tool is not a tool with no parameters."""
    tools = parse_toolset(["Bash", "Read"])

    assert tools is not None
    assert [tool.name for tool in tools] == ["Bash", "Read"]
    assert all(tool.parameters is None for tool in tools)


def test_a_serialized_declaration_is_read_as_one() -> None:
    """OTLP attributes are typed, so exporters serialize the tool array."""
    tools = parse_toolset('[{"type": "function", "function": {"name": "refund_order"}}]')

    assert tools is not None
    assert [tool.name for tool in tools] == ["refund_order"]


def test_nothing_readable_stays_unknown() -> None:
    """'The source declared no toolset' is not 'the agent had no tools'."""
    assert parse_toolset(None) is None
    assert parse_toolset([]) is None
    assert parse_toolset([{"no": "name"}]) is None
    assert parse_toolset("not json") is None
