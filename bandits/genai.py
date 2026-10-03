"""Reading the OTel GenAI message attributes a span recorded.

Shared by ingest, which decides what an episode ran under, and export, which
must refuse a transcript that leaves part of that out.
"""

from __future__ import annotations

import json
from typing import Any

PIPELINE_STEP = "bandits.pipeline_step"
"""Span attribute marking a TOOL span as a workflow step no model asked for.

A retrieval or filter node that a fixed pipeline ran between model calls. Its
result is evidence of what the next call reacted to, so it stays in the trace
for analysis and judging; it is never a call a transcript may show the model
making.
"""


def _parsed(value: object) -> object:
    if isinstance(value, str) and value.lstrip()[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _text_parts(parts: object) -> str | None:
    if not isinstance(parts, list):
        return None
    text = [
        part["content"]
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("content"), str)
    ]
    return "\n".join(text) if text else None


def system_instructions(value: object) -> str | None:
    """``gen_ai.system_instructions`` as text: a plain string or a list of parts."""
    parsed = _parsed(value)
    if isinstance(parsed, list):
        return _text_parts(parsed)
    return value if isinstance(value, str) and value else None


def system_prompt_of(attributes: dict[str, Any]) -> str | None:
    """The system instruction one model call ran under, when it recorded one.

    ``gen_ai.system_instructions`` first; otherwise the system-role messages
    that open ``gen_ai.input.messages``, which is where most exporters put it.
    """
    declared = system_instructions(attributes.get("gen_ai.system_instructions"))
    if declared is not None:
        return declared
    messages = _parsed(attributes.get("gen_ai.input.messages"))
    if not isinstance(messages, list):
        return None
    leading: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer"):
            break
        text = _text_parts(message.get("parts"))
        if text is None and isinstance(message.get("content"), str):
            text = message["content"]
        if text:
            leading.append(text)
    return "\n".join(leading) if leading else None


def input_units(attributes: dict[str, Any]) -> list[tuple[str, str] | None]:
    """A model call's recorded input as ``(role, text)`` per message, in order.

    A message with no text (a bare tool call, a tool result paired by id) is
    ``None``: its tool span carries it, and it still separates the messages on
    either side, so they are not read as one. Comparing the two renderings of
    one structured result would refuse rows over whitespace.
    """
    messages = _parsed(attributes.get("gen_ai.input.messages"))
    if not isinstance(messages, list):
        return []
    units: list[tuple[str, str] | None] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            continue
        role = "system" if message["role"] == "developer" else message["role"]
        text = _text_parts(message.get("parts"))
        if text is None and isinstance(message.get("content"), str):
            text = message["content"]
        units.append((role, text) if text and text.strip() else None)
    return units


def plain_messages(messages: object) -> list[dict[str, str]] | None:
    """*messages* as ``[{role, content}]`` text messages, when that says all of it.

    None when any message holds more than one text part or anything besides
    its role and parts: then the plain form would lose something.
    """
    messages = _parsed(messages)
    if not isinstance(messages, list):
        return None
    plain = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "parts"}:
            return None
        parts = message["parts"]
        if not (
            isinstance(parts, list)
            and len(parts) == 1
            and isinstance(parts[0], dict)
            and set(parts[0]) == {"type", "content"}
            and parts[0]["type"] == "text"
            and isinstance(parts[0]["content"], str)
        ):
            return None
        plain.append({"role": message["role"], "content": parts[0]["content"]})
    return plain
