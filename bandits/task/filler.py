"""Synthetic filler rows: copy the shapes, never the values (PLAN.md Step 10).

Two reasons the starting state cannot be only the rows the trace happens to
show. First, ``search_orders(customer_id=88)`` over a one-row table is not a
search - the agent cannot get it wrong, so the task teaches nothing. Second, a
single-row table makes a lucky guess indistinguishable from a lookup.

Two reasons the filler cannot be sampled from real data. Real values drag PII
into an artifact we intend to keep, share and train on; and they let a model
memorize the corpus instead of learning the tool loop.

So every generator here is derived from the *shape* of what was observed - the
JSON type, the string skeleton, the digit width, the numeric range - and then
emits values that were never observed. Determinism is by seed: the same
(entity, seed, count) always yields the same rows, because a task whose starting
state changes between runs is not a task.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bandits.contracts import (
    EntityRows,
    EntitySchema,
    FieldProfile,
    JsonObject,
    JsonValue,
    StateSchema,
    TaskCase,
)

__all__ = ["FillerError", "fill_task", "generate_filler", "observed_values"]

_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MAX_ATTEMPTS = 200


class FillerError(RuntimeError):
    """Raised when no unobserved value of the right shape can be produced."""


# --------------------------------------------------------------------------
# observed value collection
# --------------------------------------------------------------------------


def observed_values(
    entity: EntitySchema, rows: Iterable[JsonObject] = ()
) -> dict[str, set[str]]:
    """Every real value we know of per field, as strings.

    Sources: the profiled ``sample_values`` on the schema, plus any rows handed
    in (the task's reconstructed pre-state). Comparison is on ``str`` so that
    ``7741`` and ``"7741"`` count as the same real value - a filler row that
    collides with a real id under string comparison is still a collision as far
    as a tool that stringifies its arguments is concerned.
    """
    out: dict[str, set[str]] = {}
    for f in entity.fields:
        out[f.name] = {str(v) for v in f.sample_values if v is not None}
    for row in rows:
        for k, v in row.items():
            if v is not None:
                out.setdefault(k, set()).add(str(v))
    return out


# --------------------------------------------------------------------------
# shape inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """What a field looks like, with no real value retained except for ranges."""

    kind: str
    int_min: int = 0
    int_max: int = 0
    digits: int = 0
    float_min: float = 0.0
    float_max: float = 0.0
    decimals: int = 2
    skeleton: str = ""
    """For templated strings: 'A' upper alpha, 'a' lower alpha, '9' digit,
    anything else is a literal separator. ``SKU-RED-9`` -> ``AAA-AAA-9``."""

    year_min: int = 2000
    year_max: int = 2030


def _skeleton(value: str) -> str:
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha() and ch.isupper():
            out.append("A")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def infer_shape(profile: FieldProfile | None, values: Sequence[str]) -> _Shape:
    """Derive a value generator description from types and observed formats.

    Catches: integer ids of a fixed width, ``YYYY-MM-DD`` dates, email
    addresses, dashed SKU-style tokens, plain words, booleans, money-as-cents.
    Misses: cross-field consistency (a filler order's ``total_cents`` is not the
    filler product's price), enum semantics (see ``preserve_values``), and any
    format whose meaning is not visible in its characters.
    """
    types = set(profile.json_types) if profile and profile.json_types else set()
    if not types and values:
        types.add("string")
    if "boolean" in types and len(types) == 1:
        return _Shape(kind="bool")
    numeric = [v for v in values if re.fullmatch(r"-?\d+", v)]
    if types <= {"integer", "number", "null"} and types & {"integer", "number"}:
        if "number" in types and not all(re.fullmatch(r"-?\d+", v) for v in values):
            floats = [float(v) for v in values if _isfloat(v)] or [0.0, 100.0]
            return _Shape(kind="float", float_min=min(floats), float_max=max(floats))
        nums = [int(v) for v in numeric] or [1, 999]
        return _Shape(
            kind="int",
            int_min=min(nums),
            int_max=max(nums),
            digits=max(len(str(abs(n))) for n in nums),
        )
    strings = [v for v in values if isinstance(v, str)]
    if strings:
        if all("@" in v for v in strings):
            return _Shape(kind="email")
        dates = [_DATE.match(v) for v in strings]
        if all(dates):
            years = [int(m.group(1)) for m in dates if m]
            return _Shape(kind="date", year_min=min(years), year_max=max(years))
        return _Shape(kind="template", skeleton=_skeleton(strings[0]))
    if numeric:
        nums = [int(v) for v in numeric]
        return _Shape(kind="int", int_min=min(nums), int_max=max(nums), digits=len(str(nums[0])))
    return _Shape(kind="template", skeleton="aaaaaa")


def _isfloat(v: str) -> bool:
    try:
        float(v)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def _word(rng: random.Random, n: int) -> str:
    out = []
    for i in range(max(n, 1)):
        out.append(rng.choice(_VOWELS if i % 2 else _CONSONANTS))
    return "".join(out)


def _draw(rng: random.Random, shape: _Shape) -> JsonValue:
    if shape.kind == "bool":
        return rng.random() < 0.5
    if shape.kind == "int":
        lo = min(shape.int_min, shape.int_max)
        hi = max(shape.int_min, shape.int_max)
        span = max(hi - lo, 10)
        lo2, hi2 = lo - span, hi + span
        if shape.digits and lo >= 0:
            lo2 = max(lo2, 10 ** (shape.digits - 1) if shape.digits > 1 else 0)
            hi2 = min(hi2, 10**shape.digits - 1) if shape.digits > 1 else hi2
            if hi2 <= lo2:
                lo2, hi2 = lo, hi + span
        return rng.randint(int(lo2), int(hi2))
    if shape.kind == "float":
        lo, hi = shape.float_min, shape.float_max
        if hi <= lo:
            hi = lo + 100.0
        return round(rng.uniform(lo, hi), shape.decimals)
    if shape.kind == "email":
        return f"{_word(rng, 6)}.{_word(rng, 5)}@example.invalid"
    if shape.kind == "date":
        y = rng.randint(shape.year_min, shape.year_max)
        return f"{y:04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    out = []
    for ch in shape.skeleton or "aaaaaa":
        if ch == "9":
            out.append(rng.choice(string.digits))
        elif ch == "A":
            out.append(rng.choice(string.ascii_uppercase))
        elif ch == "a":
            out.append(rng.choice(string.ascii_lowercase))
        else:
            out.append(ch)
    return "".join(out)


def generate_filler(
    entity: EntitySchema,
    *,
    count: int,
    seed: int,
    real_rows: Sequence[JsonObject] = (),
    preserve_values: Iterable[str] = (),
) -> tuple[JsonObject, ...]:
    """Generate ``count`` synthetic rows for one entity.

    ``preserve_values`` is an explicit, off-by-default escape hatch naming fields
    whose observed literals may be reused - low-cardinality enums such as
    ``status``, which the rebuilt tools branch on, and which carry no PII. It is
    off by default because "never emit an observed real value" is the safe rule
    and re-emitting literals must be a decision someone made on purpose.

    Raises :class:`FillerError` rather than emitting a colliding value: a filler
    row that shadows a real primary key silently rewrites the task.
    """
    if entity.static_snapshot:
        # A static snapshot is materialized verbatim precisely because we refuse
        # to invent structure for it (PLAN.md Step 7). Inventing rows would be
        # the same mistake wearing a different hat.
        return ()
    preserve = set(preserve_values)
    observed = observed_values(entity, real_rows)
    profiles = {f.name: f for f in entity.fields}
    field_names = [f.name for f in entity.fields] or sorted(
        {k for r in real_rows for k in r}
    )
    if not field_names:
        return ()
    shapes = {
        name: infer_shape(profiles.get(name), sorted(observed.get(name, set())))
        for name in field_names
    }
    rng = random.Random(f"bandits-filler:{entity.name}:{seed}")
    pk = entity.primary_key
    real: dict[str, set[str]] = {name: set(observed.get(name, set())) for name in field_names}
    generated: dict[str, set[str]] = {name: set() for name in field_names}
    rows: list[JsonObject] = []
    for _ in range(count):
        row: JsonObject = {}
        for name in field_names:
            if name in preserve:
                pool = sorted(observed.get(name, set()))
                row[name] = rng.choice(pool) if pool else _draw(rng, shapes[name])
                continue
            unique = name == pk
            # Observed values are forbidden always: that is the no-real-values
            # rule. Previously generated values are forbidden only for the
            # primary key, where a collision would merge two filler rows.
            forbidden = real[name] | generated[name] if unique else real[name]
            value = _unique_draw(rng, shapes[name], forbidden, unique_required=unique)
            row[name] = value
            generated[name].add(str(value))
        rows.append(row)
    return tuple(rows)


def _unique_draw(
    rng: random.Random, shape: _Shape, forbidden: set[str], *, unique_required: bool
) -> JsonValue:
    """Draw a value avoiding ``forbidden``, or refuse.

    Booleans are exempt only when uniqueness is not required: a two-valued field
    cannot avoid its own observed values, and ``True`` is not anybody's PII. A
    boolean *primary key* still raises, because it cannot key more than two rows.
    """
    if shape.kind == "bool" and not unique_required:
        return _draw(rng, shape)
    for _ in range(_MAX_ATTEMPTS):
        v = _draw(rng, shape)
        if str(v) not in forbidden:
            return v
    raise FillerError(
        f"could not draw an unobserved value for shape {shape.kind!r} after "
        f"{_MAX_ATTEMPTS} attempts; the value space is too small to fill safely - "
        f"widen the schema, lower the filler count, or name the field in preserve_values"
    )


def fill_task(
    task: TaskCase,
    schema: StateSchema,
    *,
    seed: int,
    count_per_entity: int = 4,
    preserve_values: Iterable[str] = (),
) -> tuple[EntityRows, ...]:
    """Return the task's pre-state with filler rows appended per entity.

    Real rows always come first and are never modified. Entities the task has no
    real rows for are still filled, so a list tool over them is not empty.
    """
    preserve = set(preserve_values)
    by_entity = {er.entity: list(er.rows) for er in task.pre_state}
    out: list[EntityRows] = []
    for entity in schema.entities:
        real = by_entity.pop(entity.name, [])
        filler = generate_filler(
            entity,
            count=count_per_entity,
            seed=seed,
            real_rows=real,
            preserve_values=preserve,
        )
        if real or filler:
            out.append(EntityRows(entity=entity.name, rows=tuple(real) + filler))
    for name, real in by_entity.items():
        # Rows attributed to an entity the schema no longer carries. Keep them
        # rather than dropping evidence, and let the caller notice.
        out.append(EntityRows(entity=name, rows=tuple(real)))
    return tuple(out)
