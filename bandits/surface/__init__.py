"""Stage 2 -- the action space.

`build_surface` takes an ingested `TraceCorpus` (and, when available, the
declared tool registry) and produces a `ToolSurface`: for every tool, what it was
called with, what it answered, how it failed, and -- the decision that shapes
every later stage -- whether it reads, writes, or reaches outside the world.

    from bandits.surface import build_surface
    surface = build_surface(corpus, declared_tools=json.load(open("tools.json")))
    surface.by_name("refund_order").tool_class      # ToolClass.WRITE
    surface.by_name("refund_order").class_evidence  # why, in English

Design commitments:

* Deterministic. Same corpus in, byte-identical surface out. No sampling, no
  dict-order dependence, no clock, no network, no LLM.
* Declared and observed tools are unioned, never intersected. A tool that only
  shows up in traces is real (``observed_only``); a tool that only shows up in
  the registry is a probing candidate (``declared_only``) and stays UNKNOWN.
* Every classification carries human-readable evidence. A person reviews these.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from bandits.contracts import (
    CallStatus,
    InvocationPoint,
    JsonObject,
    ToolProfile,
    ToolSurface,
    TraceCorpus,
)

from .classify import Classification, classify_tools
from .errors import collect_error_modes
from .profiling import build_field_profiles, index_argument_values

__all__ = [
    "Classification",
    "build_field_profiles",
    "build_surface",
    "classify_tools",
    "collect_error_modes",
    "index_argument_values",
]


def _normalize_declared(
    declared_tools: Mapping[str, JsonObject] | Sequence[JsonObject] | None,
) -> dict[str, JsonObject | None]:
    """Accept either a registry list or a ``{name: schema}`` mapping.

    A registry list is a list of records like
    ``{"name": ..., "description": ..., "input_schema": {...}}``; the stored
    `ToolProfile.declared_schema` is the ``input_schema`` (or ``parameters``, the
    OpenAI spelling) because that is the part that defines the action space. If a
    record has neither, the whole record is kept rather than dropped.

    A record with no ``name`` is an error, not something to skip quietly
    (BUILD_PLAN rule 6).
    """
    if declared_tools is None:
        return {}
    if isinstance(declared_tools, Mapping):
        return {str(k): (dict(v) if isinstance(v, Mapping) else v) for k, v in declared_tools.items()}

    out: dict[str, JsonObject | None] = {}
    for entry in declared_tools:
        if not isinstance(entry, Mapping):
            raise ValueError(f"declared tool entry is not an object: {entry!r}")
        name = entry.get("name") or (entry.get("function") or {}).get("name")
        if not name:
            raise ValueError(f"declared tool entry has no name: {entry!r}")
        body = entry.get("function") if isinstance(entry.get("function"), Mapping) else entry
        schema = body.get("input_schema") or body.get("parameters")
        out[str(name)] = dict(schema) if isinstance(schema, Mapping) else dict(entry)
    return out


def _invocations_by_tool(corpus: TraceCorpus) -> dict[str, list[InvocationPoint]]:
    by_tool: dict[str, list[InvocationPoint]] = {}
    for trace in corpus.traces:
        for inv in trace.invocations:
            by_tool.setdefault(inv.tool, []).append(inv)
    for calls in by_tool.values():
        calls.sort(key=lambda i: (i.trace_id, i.step))
    return by_tool


def build_surface(
    corpus: TraceCorpus,
    declared_tools: Mapping[str, JsonObject] | Sequence[JsonObject] | None = None,
) -> ToolSurface:
    """Build the full action space from a corpus plus an optional tool registry.

    Profiles cover successful calls only for responses (failures are represented
    as `ErrorMode`s, which is what a rebuilt tool needs to replay adversity) and
    all calls for arguments (an argument the agent sent is part of the action
    space whether or not the backend liked it).

    Tools are emitted sorted by name so the surface is stable across runs.
    """
    by_tool = _invocations_by_tool(corpus)
    declared = _normalize_declared(declared_tools)
    names = sorted(set(by_tool) | set(declared))

    cross_tool_arg_values = index_argument_values(
        (inv.tool, inv.arguments) for calls in by_tool.values() for inv in calls
    )
    verdicts = classify_tools(corpus, names)

    profiles: list[ToolProfile] = []
    for name in names:
        calls = by_tool.get(name, [])
        ok_responses: Iterable = [
            c.response for c in calls if c.status is CallStatus.OK
        ]
        verdict = verdicts[name]
        profiles.append(
            ToolProfile(
                name=name,
                declared_schema=declared.get(name),
                tool_class=verdict.tool_class,
                class_confidence=verdict.confidence,
                class_evidence=verdict.evidence,
                call_count=len(calls),
                argument_fields=build_field_profiles(
                    [c.arguments for c in calls],
                    tool=name,
                    cross_tool_arg_values=cross_tool_arg_values,
                ),
                response_fields=build_field_profiles(
                    ok_responses, tool=name, cross_tool_arg_values=cross_tool_arg_values
                ),
                error_modes=collect_error_modes(calls),
                observed_only=name not in declared,
                declared_only=name in declared and not calls,
            )
        )
    return ToolSurface(tools=tuple(profiles))
