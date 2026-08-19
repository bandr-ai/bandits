"""SQLite materialization of an inferred :class:`StateSchema`.

This is where "rebuild the state" stops being a slogan. The store owns the real
database behind the reimplemented tools:

* one table per inferred entity, with columns from the inferred
  :class:`FieldProfile` json types,
* one *verbatim* key/value table per ``static_snapshot`` entity -- we observed
  those rows but never determined their structure, so we store them exactly as
  seen and refuse to invent a schema,
* seeding from ``TaskCase.pre_state``,
* ``snapshot()`` and a stable ``digest()`` so state comparison is exact and
  independent of row insertion order.

Nothing in here knows about tools. Tool semantics live in ``tools.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from bandits.contracts import EntitySchema, FieldProfile, JsonObject, StateSchema, TaskCase

from .interface import ReadOnlyEntityError, StoreError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

STATIC_INDEX_COLUMN = "_row_index"
STATIC_PAYLOAD_COLUMN = "_payload"


def _ident(name: str, what: str) -> str:
    if not _IDENT.match(name):
        raise StoreError(f"{what} {name!r} is not a usable SQL identifier")
    return f'"{name}"'


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace slack, stable unicode."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


# ---------------------------------------------------------------- columns


@dataclass(frozen=True)
class Column:
    """One materialized column and how values cross the SQLite boundary.

    ``codec`` decides encode/decode:

    * ``scalar`` -- stored as-is (SQLite is dynamically typed, so a stray value
      of the wrong type survives round trip rather than being coerced).
    * ``bool``   -- stored as 0/1, decoded back to ``True``/``False``.
    * ``json``   -- stored as canonical JSON text. Used for arrays, objects and
      for any field whose observed json types were mixed or unknown. Encoding a
      field we do not understand as TEXT-holding-JSON is lossless; guessing a
      concrete type is not.
    """

    name: str
    sql_type: str
    codec: str

    def encode(self, value: Any) -> Any:
        if value is None:
            return None
        if self.codec == "bool":
            return 1 if value else 0
        if self.codec == "json":
            return canonical_json(value)
        return value

    def decode(self, value: Any) -> Any:
        if value is None:
            return None
        if self.codec == "bool":
            return bool(value)
        if self.codec == "json":
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value
        return value


def column_for(profile: FieldProfile) -> Column:
    """Map an inferred field profile onto a SQLite column."""
    types = {t for t in profile.json_types if t != "null"}
    if types == {"integer"}:
        return Column(profile.name, "INTEGER", "scalar")
    if types in ({"number"}, {"integer", "number"}):
        return Column(profile.name, "REAL", "scalar")
    if types == {"string"}:
        return Column(profile.name, "TEXT", "scalar")
    if types == {"boolean"}:
        return Column(profile.name, "INTEGER", "bool")
    # arrays, objects, mixed, or nothing observed at all -> honest TEXT/JSON.
    return Column(profile.name, "TEXT", "json")


def _key_candidates(value: Any) -> list[Any]:
    """Values to try when matching an identifier.

    Traces stringify ids inconsistently (``7741`` vs ``"7741"``) depending on
    the exporter and on the model's own JSON. Matching on both is a fidelity
    fix, not laxness -- but it is deliberately narrow: only int/str, never a
    fuzzy match.
    """
    out: list[Any] = [value]
    if isinstance(value, bool):
        return out
    if isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, str):
        out.append(value)
        try:
            out.append(int(value))
        except ValueError:
            pass
    seen: list[Any] = []
    for v in out:
        if not any(type(v) is type(s) and v == s for s in seen):
            seen.append(v)
    return seen


# ---------------------------------------------------------------- store


class Store:
    """The materialized database for one environment."""

    def __init__(self, schema: StateSchema, *, path: str = ":memory:") -> None:
        self.schema = schema
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._columns: dict[str, dict[str, Column]] = {}
        self._entities: dict[str, EntitySchema] = {}
        for e in schema.entities:
            if e.name in self._entities:
                raise StoreError(f"duplicate entity {e.name!r} in schema")
            self._entities[e.name] = e

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> Store:
        if self._conn is not None:
            return self
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        self._create_tables()
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError("store is not open")
        return self._conn

    # -- schema ------------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        for entity in self.schema.entities:
            table = _ident(entity.name, "entity")
            if entity.static_snapshot:
                # Underdetermined entity: rows are kept verbatim as JSON, with
                # no invented columns. Read-only by construction.
                cur.execute(
                    f"CREATE TABLE {table} ("
                    f'"{STATIC_INDEX_COLUMN}" INTEGER PRIMARY KEY, '
                    f'"{STATIC_PAYLOAD_COLUMN}" TEXT NOT NULL)'
                )
                self._columns[entity.name] = {}
                continue
            cols: dict[str, Column] = {}
            for profile in entity.fields:
                col = column_for(profile)
                cols[col.name] = col
            if entity.primary_key and entity.primary_key not in cols:
                # A key we know exists but never profiled. Keep it, as TEXT/JSON.
                cols[entity.primary_key] = Column(entity.primary_key, "TEXT", "json")
            if not cols:
                raise StoreError(
                    f"entity {entity.name!r} has no fields and is not a static snapshot"
                )
            defs = []
            for name, col in cols.items():
                piece = f"{_ident(name, 'field')} {col.sql_type}"
                if entity.primary_key == name:
                    piece += " PRIMARY KEY"
                defs.append(piece)
            cur.execute(f"CREATE TABLE {table} ({', '.join(defs)})")
            self._columns[entity.name] = cols
        self.conn.commit()

    def entity(self, name: str) -> EntitySchema:
        try:
            return self._entities[name]
        except KeyError:
            raise StoreError(f"unknown entity {name!r}") from None

    @property
    def entity_names(self) -> tuple[str, ...]:
        return tuple(self._entities)

    def is_static(self, entity: str) -> bool:
        return self.entity(entity).static_snapshot

    def primary_key(self, entity: str) -> str | None:
        return self.entity(entity).primary_key

    def columns(self, entity: str) -> dict[str, Column]:
        self.entity(entity)
        return dict(self._columns.get(entity, {}))

    def has_column(self, entity: str, name: str) -> bool:
        return name in self._columns.get(entity, {})

    # -- seeding -----------------------------------------------------------

    def seed(self, task: TaskCase) -> None:
        """Load ``TaskCase.pre_state`` into the store.

        Unknown entities and unknown columns are errors, not warnings: a task
        whose starting state does not fit the schema would silently produce an
        environment that cannot be solved.
        """
        for block in task.pre_state:
            if block.entity not in self._entities:
                raise StoreError(
                    f"pre_state references entity {block.entity!r} absent from the schema"
                )
            for row in block.rows:
                self.insert(block.entity, row)
        self.conn.commit()

    def insert(self, entity: str, row: JsonObject) -> None:
        schema = self.entity(entity)
        if schema.static_snapshot:
            self._insert_static(entity, row)
            return
        cols = self._columns[entity]
        unknown = [k for k in row if k not in cols]
        if unknown:
            raise StoreError(
                f"entity {entity!r} row has columns not in the inferred schema: "
                f"{sorted(unknown)}. Fix the schema or the task, do not guess."
            )
        names = list(row)
        values = [cols[n].encode(row[n]) for n in names]
        placeholders = ", ".join("?" for _ in names)
        quoted = ", ".join(_ident(n, "field") for n in names)
        try:
            self.conn.execute(
                f"INSERT INTO {_ident(entity, 'entity')} ({quoted}) VALUES ({placeholders})", values
            )
        except sqlite3.DatabaseError as exc:
            # A constraint the inferred schema imposed but the data does not
            # satisfy -- almost always a wrong primary key. Surface it as a
            # StoreError so callers see "this schema does not hold" rather than
            # a raw driver exception leaking out of the environment.
            raise StoreError(
                f"could not seed {entity!r} row {canonical_json(row)}: {exc}. "
                "The inferred schema contradicts the data; fix the schema, do not relax it."
            ) from exc
        self.conn.commit()

    def _insert_static(self, entity: str, row: JsonObject) -> None:
        cur = self.conn.execute(
            f'SELECT COALESCE(MAX("{STATIC_INDEX_COLUMN}"), -1) + 1 FROM {_ident(entity, "entity")}'
        )
        idx = cur.fetchone()[0]
        self.conn.execute(
            f"INSERT INTO {_ident(entity, 'entity')} "
            f'("{STATIC_INDEX_COLUMN}", "{STATIC_PAYLOAD_COLUMN}") VALUES (?, ?)',
            (idx, canonical_json(row)),
        )
        self.conn.commit()

    # -- reads -------------------------------------------------------------

    def rows(self, entity: str) -> list[JsonObject]:
        schema = self.entity(entity)
        table = _ident(entity, "entity")
        if schema.static_snapshot:
            cur = self.conn.execute(
                f'SELECT "{STATIC_PAYLOAD_COLUMN}" FROM {table} ORDER BY "{STATIC_INDEX_COLUMN}"'
            )
            return [json.loads(r[0]) for r in cur.fetchall()]
        cols = self._columns[entity]
        cur = self.conn.execute(f"SELECT * FROM {table}")
        out = []
        for raw in cur.fetchall():
            out.append({k: cols[k].decode(raw[k]) for k in raw.keys()})
        return out

    def find_one(self, entity: str, column: str, value: Any) -> JsonObject | None:
        matches = self.find_many(entity, {column: value})
        return matches[0] if matches else None

    def find_many(self, entity: str, filters: dict[str, Any]) -> list[JsonObject]:
        schema = self.entity(entity)
        if schema.static_snapshot:
            raise StoreError(
                f"{entity!r} is a static snapshot: it has no columns to query, read it whole"
            )
        cols = self._columns[entity]
        unknown = [k for k in filters if k not in cols]
        if unknown:
            raise StoreError(f"entity {entity!r} has no column(s) {sorted(unknown)}")
        rows = self.rows(entity)
        out = []
        for row in rows:
            if all(self._matches(row.get(k), v) for k, v in filters.items()):
                out.append(row)
        return out

    @staticmethod
    def _matches(actual: Any, wanted: Any) -> bool:
        for cand in _key_candidates(wanted):
            if type(cand) is type(actual) and cand == actual:
                return True
        return actual == wanted

    # -- writes ------------------------------------------------------------

    def update(
        self, entity: str, key_column: str, key_value: Any, values: dict[str, Any]
    ) -> JsonObject | None:
        """Apply ``values`` to the row matched by ``key_column``.

        Returns the updated row, or ``None`` when no row matched. Never creates
        a row: an update against a missing id is a ``not_found``, which is real
        environment behaviour the agent must learn to handle.
        """
        schema = self.entity(entity)
        if schema.static_snapshot:
            raise ReadOnlyEntityError(
                entity,
                "entity is a static snapshot (structure was never determined); it cannot be written",
            )
        cols = self._columns[entity]
        if key_column not in cols:
            raise StoreError(f"entity {entity!r} has no key column {key_column!r}")
        unknown = [k for k in values if k not in cols]
        if unknown:
            raise StoreError(f"entity {entity!r} has no column(s) {sorted(unknown)}")
        current = self.find_one(entity, key_column, key_value)
        if current is None:
            return None
        if values:
            assignments = ", ".join(f"{_ident(k, 'field')} = ?" for k in values)
            params = [cols[k].encode(v) for k, v in values.items()]
            params.append(cols[key_column].encode(current[key_column]))
            self.conn.execute(
                f"UPDATE {_ident(entity, 'entity')} SET {assignments} "
                f"WHERE {_ident(key_column, 'field')} = ?",
                params,
            )
            self.conn.commit()
        return self.find_one(entity, key_column, current[key_column])

    # -- state identity ----------------------------------------------------

    def snapshot(self) -> dict[str, list[JsonObject]]:
        """Detached copy of the whole store, entity -> rows."""
        return {name: self.rows(name) for name in self._entities}

    def digest(self) -> str:
        """sha256 over a canonical, order-independent rendering of the state."""
        snap = self.snapshot()
        payload = {
            entity: sorted(canonical_json(row) for row in rows) for entity, rows in snap.items()
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def schema_digest(self) -> str:
        """sha256 of the schema itself. Two envs with the same digest are comparable."""
        payload = self.schema.model_dump(mode="json")
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = ["Column", "Store", "canonical_json", "column_for"]
