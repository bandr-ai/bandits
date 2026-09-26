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
