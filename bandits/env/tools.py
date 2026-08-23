"""Generic tool reimplementation, driven by the schema and the tool class.

Three buckets, three treatments (PLAN.md Step 7):

===========  =========================================================
READ         query the materialized store
WRITE        mutate the materialized store -- this is what reward checks
EXTERNAL     never performed; recorded to the effect ledger, stubbed reply
===========  =========================================================

Anything else (``ToolClass.UNKNOWN``, a tool whose entity cannot be resolved,
an ambiguous mapping) raises :class:`UnsupportedToolError`. It is the single
most important rule in this file: a tool we cannot reimplement must fail loudly,
because a silent fake success is indistinguishable from a real one to the
verifier and therefore poisons every reward computed from it.

Rules, and why they are pluggable
---------------------------------
The generic path infers a :class:`ReadRule` / :class:`WriteRule` from the
schema. On a real customer's tools it will not always get it right -- argument
names diverge from column names, one tool writes two entities, a status value
is not derivable from the tool name. So every inference is expressed as a small
declarative rule a human can override wholesale, and the mapping from argument
to column is always *explicit and named*, never positional.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from bandits.contracts import (
    CallStatus,
    EntitySchema,
    JsonObject,
    JsonValue,
    Observation,
    StateSchema,
    ToolClass,
    ToolProfile,
    ToolSurface,
    WriteEffect,
)

from .interface import ReadOnlyEntityError, UnsupportedToolError
from .ledger import EffectLedger
from .store import Store

STATUS_COLUMN_NAMES = ("status", "state")
"""Columns treated as the lifecycle column of an entity, for the write heuristic."""


# ---------------------------------------------------------------- rules


@dataclass(frozen=True)
class ReadRule:
    """How one READ tool queries the store, and what field set it returns.

    ``key_arg``/``key_column`` give the single-row lookup. ``filter_args`` map
    argument names onto columns for the list form. ``envelope`` is the response
    key a list is wrapped in; ``list_column`` narrows a *list* response to one
    column (e.g. ``order_ids``). ``None`` on any field means "infer at call
    time from the arguments actually passed".

    ``projection`` is the response projection: the field set this tool was
    OBSERVED to return, in the traces. A table is the union of every column any
    tool ever exposed, so ``SELECT *`` returns strictly more than any single
    tool did -- ``orders`` grows a ``refund_amount_cents`` column the moment
    ``refund_order`` is profiled, and ``get_order`` would then answer with a
    key production never sent. The projection is what keeps a read honest to
    the tool, not to the table. It is derived from
    ``ToolProfile.response_fields`` when a :class:`ToolSurface` is available,
    and can be overridden here by a human exactly like the write rules can.

    Fallback when ``projection`` is ``None`` (no surface supplied): NULL-valued
    columns are omitted from the row. This is a strictly *weaker* heuristic and
    is documented as such -- it is right only because a column a tool never
    returned is usually a column that tool never populated. It is wrong for a
    column that is legitimately NULL in production and *was* sent as ``null``:
    that key gets dropped from the response even though the real tool emitted
    it. It is also wrong in the other direction: once a write populates such a
    column, the read starts emitting a key it never emitted before. Supply a
    surface and get a real projection; the fallback only keeps the common case
    from being obviously wrong.
    """

    tool: str
    entity: str
    key_arg: str | None = None
    key_column: str | None = None
    filter_args: Mapping[str, str] | None = None
    projection: tuple[str, ...] | None = None
    list_column: str | None = None
    envelope: str | None = None
    error_kind: str = "not_found"
    inferred: bool = True


@dataclass(frozen=True)
class WriteRule:
    """How one WRITE tool changes the store.

    * ``key_arg`` -> ``key_column``: which argument identifies the row.
    * ``column_map``: argument name -> column name. Explicit, never positional.
      ``None`` means "infer: an argument whose name is a column writes it".
    * ``set_values``: columns set to a constant by the mere fact of calling this
      tool -- e.g. ``refund_order`` sets ``status='refunded'``. Inferred only
      when the value was actually *observed* in the schema's sample values.
    * ``response_columns``: the exact response shape, as column names, when the
      tool was observed echoing a specific field set back (from
      ``WriteEffect.response_echoes``). ``None`` means "key plus whatever
      changed", the old default.
    * ``response_echo``: response key -> argument name, for values the store
      does not hold (``amount_cents``) but the real tool echoed back.
    * ``conflict_error_kind``: error returned when the row already carries the
      value this tool would set (the ``already_refunded`` mode).
    """

    tool: str
    entity: str
    key_arg: str | None = None
    key_column: str | None = None
    column_map: Mapping[str, str] | None = None
    set_values: Mapping[str, JsonValue] = field(default_factory=dict)
    response_columns: tuple[str, ...] | None = None
    response_echo: Mapping[str, str] = field(default_factory=dict)
    conflict_error_kind: str | None = None
    not_found_error_kind: str = "not_found"
    respond_with_row: bool = False
    inferred: bool = True


@dataclass(frozen=True)
class ExternalRule:
    """How one EXTERNAL tool is stubbed. The effect is recorded, never performed."""

    tool: str
    response: JsonValue = field(default_factory=lambda: {"ok": True})
    inferred: bool = True


@dataclass(frozen=True)
class Unsupported:
    """A tool we refuse to run, with the reason, so the manifest can say why."""

    tool: str
    reason: str


Rule = ReadRule | WriteRule | ExternalRule | Unsupported


# ---------------------------------------------------------------- inference


def _entities_for(schema: StateSchema, tool: str, attr: str) -> list[EntitySchema]:
    return [e for e in schema.entities if tool in getattr(e, attr)]


def _samples(entity: EntitySchema, column: str) -> tuple[JsonValue, ...]:
    for f in entity.fields:
        if f.name == column:
            return f.sample_values
    return ()


def _status_columns(entity: EntitySchema) -> list[str]:
    names = [f.name for f in entity.fields]
    return [n for n in names if n in STATUS_COLUMN_NAMES or n.endswith("_status")]


def _verb_forms(tool: str) -> set[str]:
    verb = tool.split("_")[0].lower()
    return {verb, verb + "ed", verb + "d", verb.rstrip("e") + "ed"}


def infer_status_effect(tool: str, entity: EntitySchema) -> tuple[str, JsonValue] | None:
    """LAST-RESORT ONLY. Guess ``(column, value)`` a write tool sets, from English.

    .. warning::

       This derives write semantics from the tool's *name*. It is not evidence,
       it is a pun that happens to work in English: ``refund_order`` ->
       ``status="refunded"`` only because the past participle of the verb
       collides with an observed status value. It does not survive
       ``cancel_order -> "cancelled"``, ``approve_return -> "authorized"``, a
       non-English tool name, or a status vocabulary that does not echo the
       verb -- and when it does fail it fails *silently*, by simply inferring
       nothing, which then surfaces as an unsupported call much later.

       The authoritative source is :class:`~bandits.contracts.WriteEffect` on
       the entity (``sets_constants``), collected from what the write was
       actually observed to change. :func:`infer_rule` uses it whenever it is
       present and only falls back here when ``write_effects`` is empty --
       which today means "schema inference has not been re-run", not "this is
       how we do it".

    Even as a fallback it refuses to invent: the value must have been
    *observed* among that column's sample values.
    """
    forms = _verb_forms(tool)
    for column in _status_columns(entity):
        for sample in _samples(entity, column):
            if isinstance(sample, str) and sample.lower() in forms:
                return column, sample
    return None


def _profile(surface: ToolSurface | None, tool: str) -> ToolProfile | None:
    return surface.by_name(tool) if surface is not None else None


def observed_response_projection(surface: ToolSurface | None, tool: str) -> tuple[str, ...] | None:
    """The field set ``tool`` was observed to return, from stage 2's profile.

    The projection is the *top-level* keys of the response object, so a nested
    path is reduced to the key that holds it rather than dropped. The profiler
    records a nested object only by its leaves -- ``address.city`` exists,
    ``address`` does not -- so filtering dotted paths out entirely deletes the
    whole ``address`` key from every replayed read, which reads as a missing
    field rather than as the shape inside a value that it is.

    Returns ``None`` when there is no surface, no profile, or no observed
    response at all -- "we do not know" is never the same as "empty".
    """
    profile = _profile(surface, tool)
    if profile is None:
        return None
    # dict.fromkeys: first-seen order, deduplicated -- `a.b` and `a.c` are one key.
    names = tuple(
        dict.fromkeys(f.name.split(".")[0].split("[")[0] for f in profile.response_fields)
    )
    return names or None


def _write_effect(entity: EntitySchema, tool: str) -> WriteEffect | None:
    for effect in entity.write_effects:
        if effect.tool == tool:
            return effect
    return None


def _rule_from_write_effect(entity: EntitySchema, effect: WriteEffect) -> WriteRule:
    """Build a :class:`WriteRule` from observed evidence. No English involved.

    ``argument_columns`` only records the arguments whose name *differs* from
    the column, so the identity mapping (``update_order_status(status=...)`` ->
    ``orders.status``) is added here for every column of the entity. Entries for
    arguments the caller does not pass are ignored at call time, so a superset
    is harmless and keeps the runtime's mapping strictly explicit.
    """
    column_map: dict[str, str] = {f.name: f.name for f in entity.fields}
    column_map.update(effect.argument_columns)
    set_values = dict(effect.sets_constants)
    conflict: str | None = None
    for value in set_values.values():
        if isinstance(value, str):
            conflict = f"already_{value}"
            break
    return WriteRule(
        tool=effect.tool,
        entity=entity.name,
        key_arg=effect.key_argument,
        key_column=entity.primary_key,
        column_map=column_map,
        set_values=set_values,
        response_columns=effect.response_echoes or None,
        conflict_error_kind=conflict,
    )


def infer_rule(
    schema: StateSchema,
    tool: str,
    tool_class: ToolClass,
    *,
    external_stub: JsonValue | None = None,
    surface: ToolSurface | None = None,
) -> Rule:
    """Derive the rule for one tool, or an :class:`Unsupported` with a reason.

    ``surface`` (stage 2) is the evidence for what a READ tool returns; without
    it reads fall back to the weaker null-omission heuristic documented on
    :class:`ReadRule`. Write semantics come from
    :attr:`EntitySchema.write_effects` when present, and only otherwise from the
    last-resort verb guess in :func:`infer_status_effect`.
    """
    if tool_class is ToolClass.UNKNOWN:
        return Unsupported(tool, "tool class is UNKNOWN; not enough evidence to reimplement it")

    if tool_class is ToolClass.EXTERNAL:
        response = external_stub if external_stub is not None else {"ok": True}
        return ExternalRule(tool, response=response, inferred=external_stub is None)

    attr = "read_by" if tool_class is ToolClass.READ else "written_by"
    candidates = _entities_for(schema, tool, attr)
    if not candidates:
        return Unsupported(
            tool,
            f"no entity lists this tool in {attr}; the schema does not say what it touches",
        )
    if len(candidates) > 1:
        names = ", ".join(sorted(e.name for e in candidates))
        return Unsupported(
            tool,
            f"ambiguous: {attr} matches several entities ({names}); supply an explicit rule",
        )
    entity = candidates[0]

    if tool_class is ToolClass.READ:
        return ReadRule(
            tool=tool,
            entity=entity.name,
            key_column=entity.primary_key,
            projection=observed_response_projection(surface, tool),
        )

    if entity.static_snapshot:
        return Unsupported(
            tool,
            f"writes {entity.name!r}, which is a static snapshot with no determined structure",
        )
    if not entity.primary_key:
        return Unsupported(
            tool, f"entity {entity.name!r} has no primary key, so the target row is undecidable"
        )
    observed = _write_effect(entity, tool)
    if observed is not None:
        # Authoritative: what this tool was actually seen to do.
        return _rule_from_write_effect(entity, observed)

    # Fallback only. See the warning on infer_status_effect: this reads the
    # tool's English name because nothing recorded what it did.
    effect = infer_status_effect(tool, entity)
    set_values: dict[str, JsonValue] = {}
    conflict: str | None = None
    if effect is not None:
        column, value = effect
        set_values[column] = value
        conflict = f"already_{value}" if isinstance(value, str) else None
    return WriteRule(
        tool=tool,
        entity=entity.name,
        key_column=entity.primary_key,
        set_values=set_values,
        conflict_error_kind=conflict,
    )


# ---------------------------------------------------------------- runtime


class ToolRuntime:
    """Executes actions against the store and the ledger, per rule."""

    def __init__(
        self,
        schema: StateSchema,
        tool_classes: Mapping[str, ToolClass],
        store: Store,
        ledger: EffectLedger,
        *,
        rules: Mapping[str, Rule] | None = None,
        external_stubs: Mapping[str, JsonValue] | None = None,
        surface: ToolSurface | None = None,
    ) -> None:
        self.schema = schema
        self.tool_classes = dict(tool_classes)
        self.store = store
        self.ledger = ledger
        self.surface = surface
        self.external_stubs = dict(external_stubs or {})
        overrides = dict(rules or {})
        self.rules: dict[str, Rule] = {}
        for tool, klass in self.tool_classes.items():
            if tool in overrides:
                self.rules[tool] = overrides[tool]
                continue
            self.rules[tool] = infer_rule(
                schema,
                tool,
                klass,
                external_stub=self.external_stubs.get(tool),
                surface=surface,
            )
        # An override may name a tool absent from the class map. Honour it.
        for tool, rule in overrides.items():
            self.rules.setdefault(tool, rule)

    @property
    def unsupported_tools(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, r in self.rules.items() if isinstance(r, Unsupported)))

    def reason(self, tool: str) -> str | None:
        rule = self.rules.get(tool)
        return rule.reason if isinstance(rule, Unsupported) else None

    # -- dispatch ----------------------------------------------------------

    def execute(self, tool: str, arguments: JsonObject, step: int) -> Observation:
        rule = self.rules.get(tool)
        if rule is None:
            raise UnsupportedToolError(
                tool, "tool is not in this environment's action space (no class, no rule)"
            )
        if isinstance(rule, Unsupported):
            raise UnsupportedToolError(tool, rule.reason)
        if isinstance(rule, ExternalRule):
            return self._external(rule, arguments, step)
        if isinstance(rule, ReadRule):
            return self._read(rule, arguments)
        return self._write(rule, arguments)

    # -- external ----------------------------------------------------------

    def _external(self, rule: ExternalRule, arguments: JsonObject, step: int) -> Observation:
        """Record the attempt, perform nothing, touch no state."""
        self.ledger.append(rule.tool, arguments, step)
        return Observation(response=rule.response, status=CallStatus.OK)

    # -- read --------------------------------------------------------------

    def _read(self, rule: ReadRule, arguments: JsonObject) -> Observation:
        entity = rule.entity
        if self.store.is_static(entity):
            return self._read_static(entity)

        key_column = rule.key_column or self.store.primary_key(entity)
        key_arg = rule.key_arg or key_column

        # Single-row lookup: the identifier column is present in the arguments.
        if key_column and key_arg and key_arg in arguments:
            value = arguments[key_arg]
            row = self.store.find_one(entity, key_column, value)
            if row is None:
                return Observation(
                    response={"error": rule.error_kind, key_arg: value},
                    status=CallStatus.ERROR,
                    error_kind=rule.error_kind,
                )
            return Observation(response=self._project(rule, row), status=CallStatus.OK)

        # List form: every argument that names a column becomes a filter.
        if rule.filter_args is not None:
            filters = {
                col: arguments[arg] for arg, col in rule.filter_args.items() if arg in arguments
            }
        else:
            filters = {k: v for k, v in arguments.items() if self.store.has_column(entity, k)}
        if not filters and arguments:
            raise UnsupportedToolError(
                rule.tool,
                f"none of the arguments {sorted(arguments)} name a column of {entity!r}; "
                "supply a ReadRule with an explicit key_arg or filter_args",
            )
        rows = self.store.find_many(entity, filters)
        list_column = rule.list_column
        if list_column is None and key_column and not self._returns_rows(rule):
            # An id-list response is the common shape; derive its key from the
            # primary key: order_id -> order_ids.
            list_column = key_column
        envelope = rule.envelope or (f"{list_column}s" if list_column else f"{entity}")
        if list_column and all(list_column in r for r in rows):
            payload: JsonValue = [r[list_column] for r in rows]
        else:
            payload = [self._project(rule, r) for r in rows]
        return Observation(response={envelope: payload}, status=CallStatus.OK)

    def _returns_rows(self, rule: ReadRule) -> bool:
        """True when the projection says this list read returns rows, not bare ids.

        ``search_orders`` was observed returning ``{"order_ids": [...]}``: its
        projected names are not columns of the entity, so the id-list shape
        stands. A projection that *does* name columns is positive evidence of a
        row-shaped response, and collapsing it to ids would contradict it.
        """
        if rule.projection is None:
            return False
        return any(self.store.has_column(rule.entity, name) for name in rule.projection)

    def _project(self, rule: ReadRule, row: JsonObject) -> JsonObject:
        """Narrow a stored row to the field set this tool actually returns.

        With a projection (evidence: ``ToolProfile.response_fields``) the row is
        cut to exactly those keys, intersected with the keys the row has -- a
        projected field the store does not hold is not invented as ``null``.

        Without one, fall back to dropping NULL columns. That fallback is a
        heuristic and a weaker one: see :class:`ReadRule` for why a legitimately
        null production field defeats it. It is applied only because the
        alternative -- returning the union of every column any tool ever
        exposed -- is wrong more often.
        """
        if rule.projection is not None:
            keep = [k for k in rule.projection if k in row]
            if keep:
                return {k: row[k] for k in keep}
            # A projection that matches nothing describes a different response
            # shape (an envelope, say), not an empty row. Leave the row alone
            # rather than answering with {}.
            return dict(row)
        return {k: v for k, v in row.items() if v is not None}

    def _read_static(self, entity: str) -> Observation:
        """Static entities are returned exactly as observed. No structure invented."""
        rows = self.store.rows(entity)
        if len(rows) == 1:
            return Observation(response=rows[0], status=CallStatus.OK)
        return Observation(response={entity: rows}, status=CallStatus.OK)

    # -- write -------------------------------------------------------------

    def _write(self, rule: WriteRule, arguments: JsonObject) -> Observation:
        entity = rule.entity
        if self.store.is_static(entity):
            raise ReadOnlyEntityError(
                rule.tool, f"{entity!r} is a static snapshot and cannot be written"
            )
        key_column = rule.key_column or self.store.primary_key(entity)
        key_arg = rule.key_arg or key_column
        if not key_column or not key_arg:
            raise UnsupportedToolError(
                rule.tool, f"no key column for {entity!r}; supply a WriteRule with key_arg/key_column"
            )
        if key_arg not in arguments:
            raise UnsupportedToolError(
                rule.tool,
                f"argument {key_arg!r} identifying the {entity!r} row was not supplied "
                f"(got {sorted(arguments)}); supply a WriteRule with the right key_arg",
            )
        key_value = arguments[key_arg]

        # argument -> column, explicit and named.
        if rule.column_map is not None:
            mapping = {a: c for a, c in rule.column_map.items() if a in arguments}
            unknown = [c for c in mapping.values() if not self.store.has_column(entity, c)]
            if unknown:
                raise UnsupportedToolError(
                    rule.tool, f"WriteRule maps onto column(s) absent from {entity!r}: {sorted(unknown)}"
                )
        else:
            mapping = {
                a: a
                for a in arguments
                if a != key_arg and self.store.has_column(entity, a)
            }
        values: dict[str, JsonValue] = dict(rule.set_values)
        for arg, column in mapping.items():
            if column == key_column:
                continue
            values[column] = arguments[arg]  # explicit args beat the tool's constant effect

        current = self.store.find_one(entity, key_column, key_value)
        if current is None:
            return Observation(
                response={"error": rule.not_found_error_kind, key_arg: key_value},
                status=CallStatus.ERROR,
                error_kind=rule.not_found_error_kind,
            )
        if rule.conflict_error_kind and rule.set_values:
            already = all(
                current.get(col) == val
                for col, val in rule.set_values.items()
                if col not in mapping.values()
            )
            if already:
                return Observation(
                    response={"error": rule.conflict_error_kind, key_arg: key_value},
                    status=CallStatus.ERROR,
                    error_kind=rule.conflict_error_kind,
                )
        if not values:
            raise UnsupportedToolError(
                rule.tool,
                f"no argument maps onto a column of {entity!r} and no state change is inferable "
                f"from the tool name; supply a WriteRule with an explicit column_map/set_values",
            )
        updated = self.store.update(entity, key_column, key_value, values)
        assert updated is not None  # matched a moment ago, single-threaded session
        if rule.respond_with_row:
            response: JsonObject = dict(updated)
        elif rule.response_columns is not None:
            # The observed response shape (WriteEffect.response_echoes), in the
            # observed order. A name that is not a column of the row falls back
            # to the argument of the same name; anything else is left out rather
            # than fabricated.
            response = {}
            for column in rule.response_columns:
                if column in updated:
                    response[column] = updated[column]
                elif column in arguments:
                    response[column] = arguments[column]
        else:
            response = {key_arg: updated[key_column]}
            for column in values:
                response[column] = updated[column]
        for resp_key, arg_name in rule.response_echo.items():
            if arg_name in arguments:
                response[resp_key] = arguments[arg_name]
        return Observation(response=response, status=CallStatus.OK)


__all__ = [
    "ExternalRule",
    "ReadRule",
    "Rule",
    "ToolRuntime",
    "Unsupported",
    "WriteRule",
    "infer_rule",
    "infer_status_effect",
    "observed_response_projection",
]
