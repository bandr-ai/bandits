"""Reconstruct a task's starting state from its own trace (PLAN.md Step 10).

The rule, stated once and enforced everywhere below:

    The trace tells you what the world looked like at the start, *because the
    agent read it*. Every read BEFORE the first write is evidence of the
    starting state. Every read AFTER a write is not - don't let it leak
    backward.

Why the leak matters. In ``ep-refund-ok`` the agent reads order 7741 at step 2
(``status == "delivered"``), refunds it at step 4, then reads it again at step 5
(``status == "refunded"``). If the step-5 observation lands in the pre-state,
the task starts in the state the agent was supposed to *produce*. Every rollout
then scores as correct without doing anything, and the reward signal is dead.

The refinement this module implements: **the flag is per entity and per row, not
global.** A write to ``orders/7741`` says nothing about the ``products`` table,
and nothing about ``orders/7750`` either. Blanking every read after the first
write throws away most of the evidence in a long trace, which produces sparse,
under-seeded environments - a different failure with the same smell. So we track
a dirty set of ``(entity, row_key)`` pairs, and only reads that touch a dirty row
(or a dirty-in-full entity) are rejected.

When a write's target row cannot be identified - no primary key in its arguments
or response - we mark the *whole entity* dirty from that step onward. Being
conservative there costs seed rows; being permissive there corrupts the task.

Two kinds of row escape the "observe a keyed body" happy path, and both are
handled here:

**Static-snapshot entities.** ``EntitySchema.static_snapshot`` entities have
``primary_key is None`` by definition - being un-cross-referenced is what makes
them a snapshot. Requiring a key to attribute a body would make the feature
unreachable at runtime, so static rows are attributed by *content* instead:
identical bodies are one row, different bodies are different rows. A static
snapshot is by definition never written, so every observation of one is valid
pre-state and the pre/post-write discipline below can never reject it.

**Partial rows.** A response can *name* a row without showing it:
``search_orders(customer_id=88) -> {"order_ids": [7741, 7742]}`` proves order
7742 exists, and proves nothing about its contents. Dropping it loses fidelity -
the replayed search returns one id where production returned two - so we seed a
*partial* row carrying only what production actually told us: the primary key,
plus any query argument that names a declared column of the entity (an equality
filter the returned rows must satisfy: ``customer_id=88`` came from the search
arguments, not from a guess).

A partial row must never be mistaken for an observed one, and ``EntityRows`` has
no flag for it, so the distinction is recorded out of band, in
``TaskCase.provenance["partial_pre_state_rows"]``: ``{entity: [key, ...]}``. The
invariant is that every key listed there addresses a row in ``pre_state`` whose
non-key fields are unknown, not merely unset. A later full read of the same row
upgrades it - the key is dropped from the partial list the moment an observed
body for it arrives. Downstream consumers that care about the difference (see
:mod:`bandits.verify.synthesize`) read that list; consumers that only need "this
row existed at the start" can ignore it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from bandits.contracts import (
    CallStatus,
    EntityRows,
    EntitySchema,
    InvocationPoint,
    JsonObject,
    JsonValue,
    StateSchema,
    Trace,
)

__all__ = [
    "PreState",
    "RowRef",
    "attribute_object",
    "content_key",
    "entity_field_names",
    "id_list_fields",
    "id_list_rows",
    "implied_columns",
    "iter_objects",
    "key_of",
    "reconstruct_final_state",
    "reconstruct_pre_state",
    "solvability_warnings",
    "write_tools_of",
]


# --------------------------------------------------------------------------
# row attribution: turning a JSON response into (entity, primary key, row)
# --------------------------------------------------------------------------

#: An object must match this many of an entity's declared fields (the primary
#: key included) before we are willing to call it a row of that entity. Two is
#: the minimum that rules out coincidental id-only payloads such as
#: ``{"error": "not_found", "order_id": 9999}``.
MIN_FIELD_OVERLAP = 2


@dataclass(frozen=True)
class RowRef:
    """A row identified inside a tool response."""

    entity: str
    key: JsonValue
    row: JsonObject
    step: int
    tool: str
    partial: bool = False
    """True when production only *named* this row (an id in a list) and never
    showed its body. ``row`` then holds the key and query-implied columns only."""

    @property
    def key_id(self) -> str:
        return json.dumps(self.key, sort_keys=True, default=str)


def entity_field_names(entity: EntitySchema) -> set[str]:
    return {f.name for f in entity.fields}


def iter_objects(value: JsonValue) -> Iterator[JsonObject]:
    """Yield every dict inside a JSON value, outermost first.

    Responses are not always a bare row: they wrap rows in envelopes
    (``{"order": {...}}``) or lists (``{"results": [{...}, {...}]}``). We look at
    all of them and let attribution decide which are rows.
    """
    if isinstance(value, dict):
        yield value
        for v in value.values():
            yield from iter_objects(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_objects(v)


def content_key(obj: JsonObject) -> str:
    """The identity of a keyless (static-snapshot) row: its own canonical body.

    Static snapshots have no primary key to address rows by, so content *is* the
    key. Two observations of the same body are one row; two different bodies are
    two rows. Canonical JSON (sorted keys, no whitespace slack) makes that
    comparison independent of the order the exporter happened to serialize in.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _attribute_static(obj: JsonObject, schema: StateSchema) -> tuple[str, JsonValue] | None:
    """Attribute a keyless body to a static-snapshot entity, by content.

    ``EntitySchema.static_snapshot`` entities have ``primary_key is None`` by
    definition: never being cross-referenced is exactly what makes them a
    snapshot. So the keyed path above can never match one, and without this the
    static-snapshot feature would exist in the schema and be unreachable at
    runtime - every environment would start with an empty table and the
    reimplemented reader would return nothing where production returned a body.

    The admission rule is containment, not overlap: every key of the object must
    be a declared field of the entity, and at least one must be present. That is
    strict enough that ``{"order_id": 7741, "status": "delivered"}`` can never be
    mistaken for a one-column ``store_policy`` row, and loose enough that the
    single-field bodies these entities almost always have still land.

    A static snapshot is by definition never written, so *every* observation of
    one is valid pre-state; the pre/post-write discipline in
    :func:`reconstruct_pre_state` still runs over it and simply never fires.

    Misses: a static entity whose schema declared no fields at all (nothing to
    contain against, so nothing is admitted), and two genuinely distinct rows
    that happen to serialize identically - they collapse into one.
    """
    best: tuple[int, str, JsonValue] | None = None
    for entity in schema.entities:
        if not entity.static_snapshot:
            continue
        names = entity_field_names(entity)
        if not names or not obj:
            continue
        if not set(obj) <= names:
            continue
        cand = (len(obj), entity.name, content_key(obj))
        if best is None or (-cand[0], cand[1]) < (-best[0], best[1]):
            best = cand
    if best is None:
        return None
    return best[1], best[2]


def attribute_object(obj: JsonObject, schema: StateSchema) -> tuple[str, JsonValue] | None:
    """Decide which entity an object is a row of, and what its key is.

    Returns ``(entity_name, key)`` or None when the object is not recognisably a
    row. For a keyed entity the key is the primary key value: the entity's
    primary key must be present and at least :data:`MIN_FIELD_OVERLAP` declared
    fields must match. For a ``static_snapshot`` entity there is no primary key
    to require, so the key is the row's canonical content (see
    :func:`_attribute_static`). Ties are broken by entity name so the result is
    deterministic, and keyed entities always win over static ones.

    Catches: rows in envelopes, rows in lists, write-response rows, and keyless
    static-snapshot bodies.
    Misses: non-static entities with no inferred primary key (they are skipped
    entirely - we cannot address their rows, so we cannot assert on them
    either), and rows whose response uses field names the schema never saw.
    """
    if not isinstance(obj, dict):
        return None
    if "error" in obj:
        # An error payload is not a row even when it echoes an id.
        return None
    best: tuple[int, str, JsonValue] | None = None
    for entity in schema.entities:
        pk = entity.primary_key
        if not pk or pk not in obj:
            continue
        names = entity_field_names(entity)
        overlap = len(names & set(obj)) if names else len(obj)
        if overlap < MIN_FIELD_OVERLAP:
            continue
        cand = (overlap, entity.name, obj[pk])
        if best is None or (-cand[0], cand[1]) < (-best[0], best[1]):
            best = cand
    if best is not None:
        return best[1], best[2]
    return _attribute_static(obj, schema)


def _project(entity: EntitySchema, obj: JsonObject) -> JsonObject:
    """Keep only fields the schema knows about.

    A write response often carries fields the entity does not have
    (``refund_amount_cents`` on ``refund_order``). Letting those through would
    make the verifier assert on columns the rebuilt store never materializes,
    and every rollout would fail on a field that does not exist.
    """
    names = entity_field_names(entity)
    if not names:
        return dict(obj)
    return {k: v for k, v in obj.items() if k in names}


def rows_in(inv: InvocationPoint, schema: StateSchema) -> list[RowRef]:
    """Every row observable in one invocation's response."""
    out: list[RowRef] = []
    seen: set[tuple[str, str]] = set()
    for obj in iter_objects(inv.response):
        hit = attribute_object(obj, schema)
        if hit is None:
            continue
        name, key = hit
        entity = schema.entity(name)
        if entity is None:  # pragma: no cover - attribute_object only names real entities
            continue
        ref = RowRef(entity=name, key=key, row=_project(entity, obj), step=inv.step, tool=inv.tool)
        ident = (ref.entity, ref.key_id)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(ref)
    return out


def key_of(entity: EntitySchema, obj: JsonObject) -> JsonValue | None:
    pk = entity.primary_key
    if pk and isinstance(obj, dict) and pk in obj:
        return obj[pk]
    return None


# --------------------------------------------------------------------------
# partial rows: entities production NAMED but never showed
# --------------------------------------------------------------------------


def id_list_fields(entity: EntitySchema) -> set[str]:
    """Response field names that would hold a list of this entity's ids.

    Derived from the primary key rather than from English: ``order_id`` implies
    ``order_ids``, which is the shape :mod:`bandits.env.tools` already emits for
    a list-form read. Kept to a small closed set on purpose - a looser rule would
    start reading arbitrary integer arrays as row ids.
    """
    pk = entity.primary_key
    if not pk:
        return set()
    names = {f"{pk}s", "ids", f"{entity.name}_ids"}
    if pk.endswith("_id"):
        names.add(f"{pk[:-3]}_ids")
    return names


def implied_columns(entity: EntitySchema, arguments: JsonObject) -> JsonObject:
    """Columns a query's own arguments prove about every row it returned.

    ``search_orders(customer_id=88)`` returned these ids *because* they match
    ``customer_id == 88``; that is a fact production stated, not an inference
    about contents. So an argument whose name is a declared column of the entity
    (and is a scalar, and is not the primary key itself) is carried onto the
    partial row.

    Misses / over-reaches: an argument that names a column but is not an equality
    filter - a ``status`` argument meaning "sort by status", or a range bound
    that happens to share a column name - would be copied verbatim. Foreign keys
    are the overwhelmingly common case and are safe; anything else is why the row
    stays marked partial and why a reviewer sees it.
    """
    pk = entity.primary_key
    names = entity_field_names(entity)
    out: JsonObject = {}
    if not isinstance(arguments, dict):
        return out
    for name, value in arguments.items():
        if name == pk or name not in names:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        out[name] = value
    return out


def id_list_rows(inv: InvocationPoint, schema: StateSchema) -> list[RowRef]:
    """Rows this invocation only NAMED, as partial rows.

    A row we know exists but have never read is a genuine epistemic middle
    ground. Dropping it costs fidelity that shows up immediately: the replayed
    ``search_orders`` returns one id where production returned two, which reads
    as a broken environment. Inventing its contents would be worse. So we seed
    exactly what production told us - the id, plus the query's own equality
    filters (:func:`implied_columns`) - and flag the row partial so that nothing
    downstream mistakes "this row exists" for "we know what is in it".

    Only a tool the schema already lists in the entity's ``read_by`` is trusted
    to be naming that entity's ids; a bare integer array from an unrelated tool
    is left alone.
    """
    out: list[RowRef] = []
    for entity in schema.entities:
        pk = entity.primary_key
        if not pk or entity.static_snapshot or inv.tool not in entity.read_by:
            continue
        wanted = id_list_fields(entity)
        if not wanted:
            continue
        implied = implied_columns(entity, inv.arguments)
        for obj in iter_objects(inv.response):
            if "error" in obj:
                continue
            for name, value in obj.items():
                if name not in wanted or not isinstance(value, (list, tuple)):
                    continue
                for item in value:
                    if item is None or isinstance(item, (dict, list, tuple)):
                        # A list of full rows is not an id list; rows_in has it.
                        continue
                    out.append(
                        RowRef(
                            entity=entity.name,
                            key=item,
                            row={pk: item, **implied},
                            step=inv.step,
                            tool=inv.tool,
                            partial=True,
                        )
                    )
    return out


# --------------------------------------------------------------------------
# tool classes
# --------------------------------------------------------------------------


def write_tools_of(schema: StateSchema) -> set[str]:
    """Write-class tools according to the schema's ``written_by`` sets."""
    return {t for e in schema.entities for t in e.written_by}


def read_tools_of(schema: StateSchema) -> set[str]:
    return {t for e in schema.entities for t in e.read_by}


# --------------------------------------------------------------------------
# the reconstruction
# --------------------------------------------------------------------------


@dataclass
class PreState:
    """Reconstructed starting state plus the reasoning that produced it."""

    trace_id: str
    rows: dict[str, dict[str, JsonObject]] = field(default_factory=dict)
    """entity -> key_id -> row. Insertion ordered; first observation wins."""

    keys: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    """entity -> key_id -> the raw primary key value."""

    partial: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    """entity -> key_id -> key, for rows production NAMED but never showed.

    A row listed here IS in :attr:`rows`, and its non-key fields are unknown
    rather than absent. Surfaces as ``provenance["partial_pre_state_rows"]``.
    A key leaves this set the instant an observed body for it arrives.
    """

    first_write_step: int | None = None
    blocked: list[JsonObject] = field(default_factory=list)
    """Observations rejected as post-state. The audit trail for the leak rule."""

    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    """Solvability warnings. A non-empty list means: do not train on this task
    until a human has looked, because every rollout may fail identically."""

    def add(self, ref: RowRef) -> bool:
        bucket = self.rows.setdefault(ref.entity, {})
        partials = self.partial.setdefault(ref.entity, {})
        if ref.key_id in bucket:
            # First observation wins. A later pre-write read of the same row can
            # only add fields, never overwrite - the earlier value is closer to
            # the true start of the episode.
            for k, v in ref.row.items():
                bucket[ref.key_id].setdefault(k, v)
            if not ref.partial:
                # An observed body supersedes a partial seed: we now know the
                # contents, so the row stops being partial. Order does not
                # matter - the id-list may come before or after the read.
                partials.pop(ref.key_id, None)
            return False
        bucket[ref.key_id] = dict(ref.row)
        self.keys.setdefault(ref.entity, {})[ref.key_id] = ref.key
        if ref.partial:
            partials[ref.key_id] = ref.key
        return True

    def partial_row_keys(self) -> dict[str, list[JsonValue]]:
        """entity -> the keys of its partial rows. The provenance payload."""
        return {name: list(keys.values()) for name, keys in self.partial.items() if keys}

    def has_value(self, value: Any) -> bool:
        """True when some pre-state row carries this value in any field."""
        target = str(value)
        for bucket in self.rows.values():
            for row in bucket.values():
                for v in row.values():
                    if str(v) == target:
                        return True
        return False

    def to_entity_rows(self) -> tuple[EntityRows, ...]:
        return tuple(
            EntityRows(entity=name, rows=tuple(bucket.values()))
            for name, bucket in self.rows.items()
            if bucket
        )


class _DirtySet:
    """Per-entity, per-row write tracking."""

    def __init__(self) -> None:
        self.rows: dict[str, set[str]] = {}
        self.entities: set[str] = set()

    def mark_row(self, entity: str, key_id: str) -> None:
        self.rows.setdefault(entity, set()).add(key_id)

    def mark_entity(self, entity: str) -> None:
        self.entities.add(entity)

    def is_dirty(self, entity: str, key_id: str) -> bool:
        return entity in self.entities or key_id in self.rows.get(entity, set())


def _write_targets(
    inv: InvocationPoint, schema: StateSchema
) -> tuple[list[tuple[str, str, JsonValue]], list[str]]:
    """What a write invocation touched.

    Returns (specific rows, whole entities). We look at the arguments first -
    ``refund_order(order_id=7741)`` names its target even when the call errored -
    then at the response. An entity that ``written_by`` claims this tool writes,
    but for which we cannot pin a row, is dirtied in full.
    """
    rows: list[tuple[str, str, JsonValue]] = []
    whole: list[str] = []
    for entity in schema.entities:
        if inv.tool not in entity.written_by:
            continue
        found: list[JsonValue] = []
        k = key_of(entity, inv.arguments)
        if k is not None:
            found.append(k)
        for obj in iter_objects(inv.response):
            k2 = key_of(entity, obj) if isinstance(obj, dict) else None
            if k2 is not None and k2 not in found:
                found.append(k2)
        if found:
            for k3 in found:
                rows.append((entity.name, json.dumps(k3, sort_keys=True, default=str), k3))
        else:
            whole.append(entity.name)
    return rows, whole


_ID_TOKEN = re.compile(r"\b(?:[A-Z0-9]+(?:[-_][A-Z0-9]+)+|\d{2,})\b")
_STOP_TOKENS = {"00"}


def solvability_warnings(instruction: str, pre: PreState) -> list[str]:
    """Flag identifiers in the instruction that have no pre-state row.

    PLAN.md Step 10 is emphatic about why this matters: if the task says "refund
    order 7741" and order 7741 does not exist in the starting state, every
    rollout fails identically. At the pass@k gate that is indistinguishable from
    "beyond the model's ability" - a false positive of exactly the kind this
    whole system exists to avoid. So we surface it as a warning on the task
    rather than shipping a silently impossible environment.

    Catches: ids the agent only learned about *after* a write (so the pre-state
    lost them), ids mentioned in the instruction but never read at all.
    Misses: identifiers written in prose ("my last order"), ids that coincide
    with an unrelated field value, and semantic preconditions - an order that
    exists but is not refundable still passes this check.
    """
    out: list[str] = []
    for token in dict.fromkeys(_ID_TOKEN.findall(instruction or "")):
        if token in _STOP_TOKENS:
            continue
        if pre.has_value(token):
            continue
        if token.isdigit() and pre.has_value(int(token)):
            continue
        out.append(f"instruction references '{token}' but no pre-state row carries that value")
    return out


def reconstruct_pre_state(
    trace: Trace,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
    include_error_responses: bool = False,
) -> PreState:
    """Reconstruct the starting state for one trace.

    ``write_tools`` defaults to the union of every entity's ``written_by``. Pass
    it explicitly when the schema is partial - a write tool we fail to recognise
    is the one way this function can leak post-state.

    Error responses are not evidence by default: ``{"error": "not_found",
    "order_id": 9999}`` describes a row that is *absent*, and seeding it would
    invert the task.
    """
    writes = set(write_tools) if write_tools is not None else write_tools_of(schema)
    pre = PreState(trace_id=trace.trace_id)
    dirty = _DirtySet()

    for inv in sorted(trace.invocations, key=lambda i: i.step):
        is_write = inv.tool in writes
        if not is_write:
            if inv.status is CallStatus.ERROR and not include_error_responses:
                continue
            for ref in (*rows_in(inv, schema), *id_list_rows(inv, schema)):
                if dirty.is_dirty(ref.entity, ref.key_id):
                    pre.blocked.append(
                        {
                            "step": inv.step,
                            "tool": inv.tool,
                            "entity": ref.entity,
                            "row_key": ref.key,
                            "partial": ref.partial,
                            "reason": "read after a write to this row; post-state, not pre-state",
                        }
                    )
                    continue
                if pre.add(ref) and ref.partial:
                    pre.notes.append(
                        f"step {inv.step}: {inv.tool} named {ref.entity}[{ref.key!r}] in an id "
                        f"list without ever reading it; seeded as a PARTIAL row "
                        f"({sorted(ref.row)}) - existence is proven, contents are not"
                    )
            continue

        # a write: everything it touched is post-state from here on
        if pre.first_write_step is None:
            pre.first_write_step = inv.step
        rows, whole = _write_targets(inv, schema)
        for entity_name, key_id, key in rows:
            dirty.mark_row(entity_name, key_id)
            pre.notes.append(
                f"step {inv.step}: {inv.tool} wrote {entity_name}[{key!r}]; "
                f"later reads of that row are post-state"
            )
        for entity_name in whole:
            dirty.mark_entity(entity_name)
            pre.notes.append(
                f"step {inv.step}: {inv.tool} wrote {entity_name} but its target row could not "
                f"be identified; the whole entity is treated as post-state from here"
            )
        if not rows and not whole:
            pre.notes.append(
                f"step {inv.step}: {inv.tool} is a write tool no entity claims; "
                f"no rows dirtied - check the schema's written_by sets"
            )

    pre.warnings = solvability_warnings(trace.instruction or "", pre)
    return pre


def reconstruct_final_state(
    trace: Trace,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
) -> dict[str, dict[str, JsonObject]]:
    """Reconstruct the state at the END of the trace: last observation wins.

    Both reads and successful write responses count here, because by the end of
    the episode there is no ordering left to violate. Write responses are
    projected onto the entity's declared fields first (see :func:`_project`).
    Errored calls are ignored - they changed nothing and returned no row.
    """
    _ = write_tools  # accepted for symmetry; final state does not depend on ordering
    state: dict[str, dict[str, JsonObject]] = {}
    for inv in sorted(trace.invocations, key=lambda i: i.step):
        if inv.status is CallStatus.ERROR:
            continue
        for ref in rows_in(inv, schema):
            bucket = state.setdefault(ref.entity, {})
            if ref.key_id in bucket:
                bucket[ref.key_id] = {**bucket[ref.key_id], **ref.row}
            else:
                bucket[ref.key_id] = dict(ref.row)
    return state
