"""Every field in a corpus, found and read by its path: the interface agents use.

``fields`` lists each path with its types, how many steps (or traces) hold it,
and a few examples; ``values`` reads one path across traces and steps. Neither
needs to know a provider's export format: the path is the field's own name.

A step is seen as one object: its recorded fields (``name``, ``kind``,
``status``, ``started_at``, ``ended_at``, ``duration_ms``, ``output``,
``arguments``), its attributes under their own names (``gen_ai.usage.*``,
``metadata.*``, ...), and ``misc``: the fields the converter kept without
interpreting (``bandits.unmapped``). Trace-level fields are ``trace.*``. A key
that itself contains dots resolves too: the longest matching key wins.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

from bandits.traces import Span, Trace, TraceCorpus, WorkflowNode

EXAMPLES = 3
"""Distinct example values kept per path."""

MAX_DEPTH = 8
"""Levels below a step's own fields that :func:`fields` walks into."""

_HIDDEN = ("bandits.otlp.source_context",)
"""Bookkeeping about where a field came from; not a field of the step."""

ABSENT = object()
"""What :func:`resolve` returns for a path a step does not hold (unlike ``None``,
which is a recorded null)."""


def _json_text(value: Any) -> Any:
    """Bandits' own JSON-text fields as objects; anything else unchanged."""
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def step_view(step: Span | WorkflowNode) -> dict[str, Any]:
    """One step as the object paths address."""
    view: dict[str, Any] = {
        "name": step.name,
        "kind": step.kind.value if isinstance(step, Span) else "step",
        "status": step.status.value,
        "started_at": step.started_at.isoformat(),
        "ended_at": step.ended_at.isoformat(),
        "duration_ms": round((step.ended_at - step.started_at).total_seconds() * 1000),
    }
    if isinstance(step, Span):
        view["output"] = step.output
        if step.arguments:
            view["arguments"] = step.arguments
    else:
        view["input"] = step.input
        view["output"] = step.output
    for key, value in step.attributes.items():
        if key in _HIDDEN:
            continue
        if key == "bandits.unmapped":
            view["misc"] = _json_text(value)
        elif key.startswith(("bandits.", "metadata.")):
            # Objects these hold were stored as JSON text; read them back.
            view[key] = _json_text(value)
        else:
            view[key] = value
    return view


def trace_view(trace: Trace) -> dict[str, Any]:
    """A trace's own fields, as ``trace.*`` paths address them."""
    request = trace.request
    view: dict[str, Any] = {"trace_id": trace.trace_id, "task": trace.task}
    if request is not None:
        view.update(
            {
                "task_status": request.task_status,
                "task_path": request.task_path,
                "delivered": request.delivered,
                "invocation_basis": request.invocation_basis,
                "task_candidates": [c.model_dump() for c in request.task_candidates],
            }
        )
    if trace.system_prompt is not None:
        view["system_prompt"] = trace.system_prompt
    return view


def _steps(trace: Trace) -> Iterator[Span | WorkflowNode]:
    seen: set[str] = set()
    for step in (*trace.spans, *trace.workflow_nodes):
        if step.span_id not in seen:
            seen.add(step.span_id)
            yield step


def resolve(node: Any, path: str) -> Any:
    """The value at *path* inside *node*, or :data:`ABSENT`.

    At an object, the longest key that is the path or a dotted prefix of it is
    taken, so ``gen_ai.usage.input_tokens`` and ``misc.usageDetails.input``
    both resolve. ``[n]`` indexes a list. JSON text is read when stepped into.
    """
    while path:
        if path.startswith("["):
            end = path.find("]")
            node = _json_text(node)
            if end < 0 or not isinstance(node, list):
                return ABSENT
            try:
                index = int(path[1:end])
            except ValueError:
                return ABSENT
            if not -len(node) <= index < len(node):
                return ABSENT
            node, path = node[index], path[end + 1 :].removeprefix(".")
            continue
        node = _json_text(node)
        if not isinstance(node, dict):
            return ABSENT
        head = path.split("[", 1)[0]
        matches = [key for key in node if head == key or head.startswith(key + ".")]
        if not matches:
            return ABSENT
        key = max(matches, key=len)
        node, path = node[key], path[len(key) :].removeprefix(".")
    return node


def _type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _walk(value: Any, path: str, depth: int) -> Iterator[tuple[str, Any]]:
    """``(path, value)`` for *value* and everything below it; list items share
    one path, ``name[]``."""
    yield path, value
    if depth >= MAX_DEPTH:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}" if path else str(key), depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child, f"{path}[]", depth + 1)


def _example(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


def _group(path: str) -> str:
    if path.startswith("trace."):
        return "trace"
    if path.startswith("misc"):
        return "misc"
    if path.startswith("metadata."):
        return "metadata"
    if path.startswith("bandits."):
        return "bandits"
    if path.split(".", 1)[0] in (
        "name",
        "kind",
        "status",
        "started_at",
        "ended_at",
        "duration_ms",
        "input",
        "output",
        "arguments",
    ):
        return "step"
    return "normalized"


def fields(corpus: TraceCorpus | Iterable[Trace]) -> list[dict[str, Any]]:
    """Every path in the corpus: ``{path, group, types, count, of, examples}``.

    ``count`` is how many steps (for ``trace.*``, traces) hold the path, out of
    ``of``; a path held as an explicit null counts as held. Composite values
    are listed and walked into; scalars inside lists share one ``[]`` path.
    """
    traces = corpus.traces if isinstance(corpus, TraceCorpus) else list(corpus)
    types: dict[str, Counter[str]] = {}
    held: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    steps = 0

    def add(view: dict[str, Any], prefix: str) -> None:
        seen: set[str] = set()
        for key, value in view.items():
            for path, item in _walk(value, prefix + key, 1):
                types.setdefault(path, Counter())[_type(item)] += 1
                if path not in seen:
                    seen.add(path)
                    held[path] += 1
                kept = examples.setdefault(path, [])
                if (
                    len(kept) < EXAMPLES
                    and not isinstance(item, (dict, list))
                    and item is not None
                    and (text := _example(item)) not in kept
                ):
                    kept.append(text)

    for trace in traces:
        add(trace_view(trace), "trace.")
        for step in _steps(trace):
            steps += 1
            add(step_view(step), "")
    rows = [
        {
            "path": path,
            "group": _group(path),
            "types": dict(types[path]),
            "count": held[path],
            "of": len(traces) if path.startswith("trace.") else steps,
            "examples": examples.get(path, []),
        }
        for path in types
    ]
    return sorted(rows, key=lambda r: (r["group"] != "trace", r["group"], r["path"]))


def values(
    corpus: TraceCorpus | Iterable[Trace],
    path: str,
    *,
    trace_ids: Iterable[str] = (),
    present_only: bool = False,
) -> Iterator[dict[str, Any]]:
    """The value at *path* in every step (or, for ``trace.*``, every trace).

    Each row has ``trace_id``, ``step_id`` and ``step`` (None for a trace
    field), ``present``, and ``value`` when present: an absent field is
    ``present: false``, a recorded null is ``present: true, value: null``.
    """
    traces = corpus.traces if isinstance(corpus, TraceCorpus) else list(corpus)
    wanted = set(trace_ids)
    for trace in traces:
        if wanted and trace.trace_id not in wanted:
            continue
        if path.startswith("trace."):
            found = resolve(trace_view(trace), path.removeprefix("trace."))
            rows = [(None, None, found)]
        else:
            rows = [
                (step.span_id, step.name, resolve(step_view(step), path)) for step in _steps(trace)
            ]
        for step_id, name, found in rows:
            if found is ABSENT:
                if not present_only:
                    yield {
                        "trace_id": trace.trace_id,
                        "step_id": step_id,
                        "step": name,
                        "present": False,
                    }
                continue
            yield {
                "trace_id": trace.trace_id,
                "step_id": step_id,
                "step": name,
                "present": True,
                "value": found,
            }
