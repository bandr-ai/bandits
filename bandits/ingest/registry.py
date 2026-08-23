"""Load the declared tool registry.

Why the registry matters (PLAN.md Step 6): the traces tell you what the agent
*did*, the registry tells you what it *could have done*. Traces alone give an
environment with holes exactly where production is thin. So the registry is a
first-class input, not a nicety -- it is where ``ToolProfile.declared_schema`` and
``declared_only`` (the probing candidates) come from.

Input shape is the one in ``tests/fixtures/tools.json``: a JSON array of
``{"name", "description", "input_schema"}`` objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bandits.contracts import IngestIssue, JsonObject, ToolClass


class RegistryError(ValueError):
    """The registry file is not a tool registry at all. Fail loudly, never guess."""


def _entry_schema(entry: JsonObject) -> JsonObject:
    """Return the declared JSON Schema for one entry, preserving surrounding metadata.

    We keep description alongside the schema because stage 2 uses the description as
    classification evidence, and stage 4 needs the schema to define the action space.
    A missing ``input_schema`` becomes an empty object schema rather than ``None`` so
    downstream code never has to special-case "declared but shapeless".
    """
    schema = entry.get("input_schema")
    if schema is None:
        schema = entry.get("inputSchema") or entry.get("parameters")
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, dict):
        raise RegistryError(f"input_schema for {entry.get('name')!r} is not an object")
    declared: JsonObject = {"input_schema": schema}
    description = entry.get("description")
    if isinstance(description, str):
        declared["description"] = description
    raw_class = entry.get("tool_class")
    if raw_class is not None:
        try:
            declared["tool_class"] = ToolClass(raw_class).value
        except ValueError as exc:
            raise RegistryError(
                f"tool_class for {entry.get('name')!r} must be one of "
                f"{[c.value for c in ToolClass]}, got {raw_class!r}"
            ) from exc
    return declared


def parse_registry(payload: Any, *, location: str = "<registry>") -> tuple[dict[str, JsonObject], list[IngestIssue]]:
    """Normalize an already-decoded registry payload into ``name -> declared schema``.

    Split out from :func:`load_registry` so the parsing rules can be tested without
    touching the filesystem. Malformed *entries* become issues (the rest of the
    registry is still usable); a malformed *payload* raises, because a registry that
    is not a list is a caller error we must not paper over.
    """
    if isinstance(payload, dict) and isinstance(payload.get("tools"), list):
        # Some exports wrap the array, e.g. an MCP ``list_tools`` response body.
        payload = payload["tools"]
    if not isinstance(payload, list):
        raise RegistryError(f"{location}: expected a JSON array of tools, got {type(payload).__name__}")

    registry: dict[str, JsonObject] = {}
    issues: list[IngestIssue] = []
    for index, entry in enumerate(payload):
        where = f"{location}[{index}]"
        if not isinstance(entry, dict):
            issues.append(IngestIssue(
                kind="malformed_tool_entry",
                detail=f"expected an object, got {type(entry).__name__}",
                location=where,
            ))
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(IngestIssue(
                kind="tool_missing_name",
                detail="registry entry has no usable 'name'",
                location=where,
            ))
            continue
        if name in registry:
            issues.append(IngestIssue(
                kind="duplicate_tool",
                detail=f"tool {name!r} declared more than once; keeping the first declaration",
                location=where,
            ))
            continue
        try:
            registry[name] = _entry_schema(entry)
        except RegistryError as exc:
            issues.append(IngestIssue(kind="malformed_tool_schema", detail=str(exc), location=where))
    return registry, issues


def load_registry_with_issues(path: str | Path) -> tuple[dict[str, JsonObject], list[IngestIssue]]:
    """Read a registry file and return both the mapping and any per-entry issues.

    Used by :func:`bandits.ingest.load_corpus`, which folds the issues into the
    corpus so a bad tool declaration is visible rather than silently absent.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"{path}: not valid JSON: {exc}") from exc
    return parse_registry(payload, location=str(path))


def load_registry(path: str | Path) -> dict[str, JsonObject]:
    """Return ``tool name -> declared schema dict`` for a registry file.

    The convenience form for callers that just want the action space. Entry-level
    problems are still surfaced -- as a :class:`RegistryError` here, since this
    signature has nowhere to put an issue list.
    """
    registry, issues = load_registry_with_issues(path)
    if issues:
        detail = "; ".join(f"{i.kind} at {i.location}: {i.detail}" for i in issues)
        raise RegistryError(f"{path}: unusable registry entries: {detail}")
    return registry


__all__ = [
    "RegistryError",
    "load_registry",
    "load_registry_with_issues",
    "parse_registry",
]
