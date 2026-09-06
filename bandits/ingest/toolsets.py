"""Read a declared toolset out of whatever shape a source wrote it in.

Sources agree on almost nothing here: OpenAI wraps each tool in ``{"type":
"function", "function": {...}}``, Anthropic writes ``input_schema`` instead of
``parameters``, and a session log may list nothing but names. All three are the
same fact — what the agent was offered — so they normalize into one model, and a
name offered without a schema keeps its ``parameters`` as None rather than
acquiring an empty one.
"""

from __future__ import annotations

import json
from typing import Any

from bandits.traces import ToolSchema


def _one(declared: Any) -> ToolSchema | None:
    if isinstance(declared, str) and declared:
        return ToolSchema(name=declared)
    if not isinstance(declared, dict):
        return None
    body = declared.get("function") if isinstance(declared.get("function"), dict) else declared
    name = body.get("name")
    if not isinstance(name, str) or not name:
        return None
    parameters = body.get("parameters")
    if not isinstance(parameters, dict):
        parameters = body.get("input_schema")
    description = body.get("description")
    return ToolSchema(
        name=name,
        description=description if isinstance(description, str) else None,
        parameters=parameters if isinstance(parameters, dict) else None,
    )


def parse_toolset(declared: Any) -> tuple[ToolSchema, ...] | None:
    """Normalize a declared toolset, or return None when there is nothing to read.

    A declaration that is present but holds no readable tool still returns None:
    'the source said the agent had no tools' is a claim no export actually makes,
    and reading it as one would turn an unreadable field into a fact.
    """
    if isinstance(declared, str):
        # OTLP attributes are typed, so an exporter with a list to record often
        # serializes it. The JSON is the declaration; refusing to parse it would
        # report a toolset that was recorded as one that was not.
        try:
            declared = json.loads(declared)
        except json.JSONDecodeError:
            return None
    if not isinstance(declared, (list, tuple)) or not declared:
        return None
    tools = tuple(tool for tool in (_one(item) for item in declared) if tool is not None)
    return tools or None
