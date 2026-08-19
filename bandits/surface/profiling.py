"""Argument and response profiling.

Turns a set of observed JSON payloads for one tool into `FieldProfile`s.

What a profile buys us (PLAN.md, "The key concept: invocation points"): the
declared schema tells you what a tool *accepts*; the profile tells you what the
production agent actually *sent*. That gap is the "5 of 38 parameters are ever
used" signal, and it is what keeps a generated environment from having to model
33 parameters nobody ever touches.

Nesting rules
-------------
Objects are flattened to dotted paths: ``{"order": {"customer_id": 88}}`` becomes
``order.customer_id``. Arrays get a ``[]`` suffix: ``order_ids`` is profiled as
an ``array`` field (one occurrence per payload) *and* its elements are profiled
under ``order_ids[]`` (one occurrence per element). Arrays of objects therefore
produce ``items[].sku`` and friends, which is what stage 3 needs to spot
identifiers that only ever appear inside a list.

Everything here is deterministic: no sampling, no dict-ordering dependence.
Sample values are sorted by ``(json type, canonical json)`` and capped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence

from bandits.contracts import FieldProfile, JsonValue

MAX_SAMPLE_VALUES = 5
"""Cap on `FieldProfile.sample_values`. Kept small: these are hints for filler
data generation, not a data dump."""

IDENTIFIER_SUFFIXES = ("_id", "_ids", "_sku", "_skus", "_key", "_keys", "_uuid", "_code")
IDENTIFIER_NAMES = frozenset({"id", "ids", "sku", "skus", "key", "keys", "uuid"})

_ID_VALUE_MIN_OCCURRENCES = 3
_ID_VALUE_DISTINCT_RATIO = 0.6


def json_type(value: JsonValue) -> str:
    """The JSON Schema type name for a Python value.

    `bool` is checked before `int` on purpose: in Python `True` is an int, and
    calling a boolean flag an integer would make every ack-shaped response look
    like it carried a numeric field.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "unknown"


def canonical(value: JsonValue) -> str:
    """A stable, hashable string for any JSON value. Used for distinct counts."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


def _value_key(value: JsonValue) -> str:
    """Distinctness key that keeps types apart, so ``1`` and ``"1"`` differ."""
    return f"{json_type(value)}:{canonical(value)}"


def flatten(payload: JsonValue, prefix: str = "") -> list[tuple[str, JsonValue]]:
    """Flatten one JSON payload to ``(dotted_path, leaf_value)`` pairs.

    A payload that is not an object (a bare string, a bare list) is recorded
    under the path ``"$"`` so it is never silently dropped -- BUILD_PLAN rule 6.
    """
    out: list[tuple[str, JsonValue]] = []
    _walk(payload, prefix, out)
    return out


def _walk(value: JsonValue, path: str, out: list[tuple[str, JsonValue]]) -> None:
    if isinstance(value, Mapping):
        if not value and path:
            out.append((path, {}))
        for key in sorted(value):
            child = f"{path}.{key}" if path else str(key)
            _walk(value[key], child, out)
        return
    if isinstance(value, (list, tuple)):
        # The array itself is a field, and so are its elements.
        out.append((path or "$", list(value)))
        for element in value:
            _walk(element, f"{path or '$'}[]", out)
        return
    out.append((path or "$", value))


def leaf_name(path: str) -> str:
    """Last dotted segment of a path, with any ``[]`` marker removed."""
    return path.replace("[]", "").split(".")[-1]


def name_looks_like_identifier(path: str) -> bool:
    """True when the *name* alone says 'this is an entity id'."""
    leaf = leaf_name(path).lower()
    if leaf in IDENTIFIER_NAMES:
        return True
    return leaf.endswith(IDENTIFIER_SUFFIXES)


def looks_like_identifier(
    path: str,
    values: Sequence[JsonValue],
    *,
    tool: str = "",
    cross_tool_arg_values: Mapping[str, frozenset[str]] | None = None,
) -> bool:
    """Decide whether a field carries an entity identifier.

    Two independent rules, either is sufficient:

    1. **Name rule.** The leaf segment is ``id``/``sku``/``key``/``uuid`` or ends
       in ``_id`` / ``_ids`` / ``_sku`` / ``_key`` / ``_uuid`` / ``_code``.
    2. **Recurrence rule.** The field holds scalars, we saw it at least
       ``3`` times, at least ``60%`` of those values were distinct, and at least
       one of those values also appears as an *argument value of a different
       tool*. That last clause is the real signal: an id is a value that travels
       between tools. A free-text ``subject`` never does; ``7741`` does, because
       it comes out of ``search_orders`` and goes into ``refund_order``.

    Failure modes, stated plainly:

    * The name rule fires on anything called ``valid_id`` or ``request_id`` even
      when no entity is behind it. Downstream (stage 3) must confirm with real
      recurrence before creating a table.
    * The recurrence rule misses identifiers that only one tool ever touches --
      an id used in exactly one place is indistinguishable from a parameter.
    * Low-cardinality enums shared across tools (``status="delivered"``) are
      rejected by the distinct-ratio clause, but a *high*-cardinality shared
      free-text field (an email address passed to two tools) will be called an
      identifier. That is arguably correct: it is functioning as a key.
    """
    if name_looks_like_identifier(path):
        return True
    scalars = [v for v in values if json_type(v) in ("string", "integer")]
    if len(scalars) < _ID_VALUE_MIN_OCCURRENCES or len(scalars) != len(values):
        return False
    distinct = {_value_key(v) for v in scalars}
    if len(distinct) / len(scalars) < _ID_VALUE_DISTINCT_RATIO:
        return False
    if not cross_tool_arg_values:
        return False
    return any(
        len(cross_tool_arg_values.get(_value_key(v), frozenset()) - {tool}) > 0 for v in scalars
    )


def build_field_profiles(
    payloads: Iterable[JsonValue],
    *,
    tool: str = "",
    cross_tool_arg_values: Mapping[str, frozenset[str]] | None = None,
) -> tuple[FieldProfile, ...]:
    """Profile every field seen across `payloads`.

    ``occurrences`` counts payloads in which the path appeared, except for
    element paths (``foo[]``) where it counts elements -- an array of three ids
    in one response is three observations of that element field.

    Returns profiles sorted by field name so the output is byte-stable.
    """
    types: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    nulls: dict[str, int] = {}
    seen_values: dict[str, list[JsonValue]] = {}
    distinct: dict[str, set[str]] = {}

    for payload in payloads:
        for path, value in flatten(payload):
            types.setdefault(path, set()).add(json_type(value))
            counts[path] = counts.get(path, 0) + 1
            if value is None:
                nulls[path] = nulls.get(path, 0) + 1
            seen_values.setdefault(path, []).append(value)
            distinct.setdefault(path, set()).add(_value_key(value))

    profiles: list[FieldProfile] = []
    for path in sorted(counts):
        values = seen_values[path]
        samples = sorted(
            {_value_key(v): v for v in values}.items(),
            key=lambda kv: kv[0],
        )[:MAX_SAMPLE_VALUES]
        profiles.append(
            FieldProfile(
                name=path,
                json_types=tuple(sorted(types[path])),
                occurrences=counts[path],
                null_count=nulls.get(path, 0),
                distinct_values=len(distinct[path]),
                sample_values=tuple(v for _, v in samples),
                looks_like_identifier=looks_like_identifier(
                    path, values, tool=tool, cross_tool_arg_values=cross_tool_arg_values
                ),
            )
        )
    return tuple(profiles)


def index_argument_values(
    calls: Iterable[tuple[str, JsonValue]],
) -> dict[str, frozenset[str]]:
    """Map every scalar argument value to the set of tools that ever sent it.

    ``calls`` is an iterable of ``(tool_name, arguments)``. This is the corpus
    level context the recurrence rule in `looks_like_identifier` needs, and it
    is also what `classify` uses to link a write to the reads around it.
    Booleans are excluded: ``True`` is shared by every flag in the corpus and
    would make every one of them look like an identifier.
    """
    index: dict[str, set[str]] = {}
    for tool, arguments in calls:
        for _path, value in flatten(arguments):
            if json_type(value) not in ("string", "integer"):
                continue
            index.setdefault(_value_key(value), set()).add(tool)
    return {k: frozenset(v) for k, v in index.items()}
