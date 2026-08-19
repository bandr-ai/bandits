"""Synthesize a ``Verifier`` by diffing pre-state against final state.

PLAN.md Step 11: reward is an assertion over the final state and the effect
ledger, never a judge. This module writes those assertions automatically, from
one trace, deterministically - no model, no network, no heuristic scoring.

The shape of the diff::

    field changed        -> STATE_EQUALS
    row appeared         -> ROW_EXISTS
    row disappeared      -> ROW_ABSENT
    external tool called -> EFFECT_COUNT
    nothing changed      -> STATE_UNCHANGED

That last line is the one people skip. If the verifier only checks that
``orders[7741].status == 'refunded'``, an agent that also empties the customers
table scores full marks. Collateral damage has to cost something or the policy
learns it is free. So every entity and every row we have pre-state evidence for,
and which the trace did not change, gets an explicit STATE_UNCHANGED.

One asymmetry is deliberate too. A *partial* pre-state row - one production
named in an id list but never read - still gets assertions, but only over the
fields that are actually known (its key and the query filters that produced it).
Existence and those fields are evidence; the rest of the row is not, and is never
asserted to be empty. See :func:`partial_key_ids`.

Two refusals are deliberate:

* **Only positively-labeled traces.** "Do what production did" is not a reward
  function unless production was labeled correct. Synthesizing from an
  unlabeled or failed trace bakes the failure in as the target.
* **``reviewed_by`` starts None.** A generated reward function nobody has read
  is an unexamined reward function. :func:`bandits.verify.run.evaluate` refuses
  to grade with one unless the caller explicitly opts out for testing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from bandits.contracts import (
    Assertion,
    AssertionKind,
    CallStatus,
    EntityRows,
    JsonObject,
    JsonValue,
    StateSchema,
    TaskCase,
    Trace,
    Verifier,
)
from bandits.task.prestate import iter_objects, key_of, reconstruct_final_state

__all__ = [
    "UnlabeledTraceError",
    "external_tools_of",
    "partial_key_ids",
    "row_key_for",
    "synthesize_verifier",
]


#: Appended to the description of any assertion built over a partial pre-state
#: row, so the reviewer can see that the row's unlisted columns are *unknown*
#: rather than known-empty, and that the assertion is deliberately narrow.
_PARTIAL_NOTE = (
    " (partial pre-state row: production named this row in an id list but never "
    "read it, so only the fields listed here are known)"
)


class UnlabeledTraceError(ValueError):
    """Raised when asked to synthesize from a trace that is not labeled correct."""


def row_key_for(primary_key: str, value: JsonValue) -> JsonObject:
    """Row keys travel as ``{pk_name: value}``.

    A bare value would force the grader to know each entity's primary key from
    somewhere else; a dict makes every assertion self-describing, and leaves room
    for composite keys later without changing the contract.
    """
    return {primary_key: value}


def external_tools_of(schema: StateSchema, trace: Trace) -> list[str]:
    """Tools the trace called that no entity reads or writes.

    Default only. Pass ``external_tools`` explicitly when stage 2's
    classification is available - this fallback would call an unmodeled read
    tool 'external' and assert an effect count on it.
    """
    known = {t for e in schema.entities for t in (*e.read_by, *e.written_by)}
    seen: list[str] = []
    for inv in sorted(trace.invocations, key=lambda i: i.step):
        if inv.tool not in known and inv.tool not in seen:
            seen.append(inv.tool)
    return seen


def partial_key_ids(task: TaskCase, entity: str) -> set[str]:
    """Key ids of ``entity``'s partial pre-state rows, from the task provenance.

    A partial row is one production only *named* - an id in a list - so its key
    and the query filters that produced it are known and nothing else is (see
    :mod:`bandits.task.prestate`). Synthesis still asserts on it, because
    existence and those known fields are real evidence and an agent that deletes
    or rewrites the row is doing collateral damage. What it must NOT do is treat
    the row's *unknown* columns as absent: every assertion below is written only
    over fields the pre-state row actually carries, so a partial row yields a
    narrower assertion rather than a wrong one. The description says so out loud,
    because a human reads these before they are allowed to grade.
    """
    raw = task.provenance.get("partial_pre_state_rows") or {}
    if not isinstance(raw, dict):
        return set()
    return {json.dumps(k, sort_keys=True, default=str) for k in raw.get(entity, ())}


def _index(rows: Iterable[JsonObject], pk: str | None) -> dict[str, JsonObject]:
    idx: dict[str, JsonObject] = {}
    for row in rows:
        if pk is None or pk not in row:
            continue
        idx[json.dumps(row[pk], sort_keys=True, default=str)] = dict(row)
    return idx


def _deleted_rows(trace: Trace, schema: StateSchema) -> set[tuple[str, str]]:
    """Rows a read reported missing AFTER something wrote to them.

    Reads cannot prove deletion on their own - a row we never looked at again is
    not gone. We only claim absence when the trace explicitly asked for the row
    later and got ``not_found``.
    """
    gone: set[tuple[str, str]] = set()
    for inv in sorted(trace.invocations, key=lambda i: i.step):
        if inv.status is not CallStatus.ERROR:
            continue
        if (inv.error_kind or "").lower() not in {"not_found", "missing", "deleted"}:
            continue
        for entity in schema.entities:
            k = key_of(entity, inv.arguments)
            if k is None:
                for obj in iter_objects(inv.response):
                    k = key_of(entity, obj)
                    if k is not None:
                        break
            if k is not None:
                gone.add((entity.name, json.dumps(k, sort_keys=True, default=str)))
    return gone


def synthesize_verifier(
    task: TaskCase,
    trace: Trace,
    schema: StateSchema,
    *,
    external_tools: Iterable[str] | None = None,
    verifier_id: str | None = None,
) -> Verifier:
    """Build the reward function for one task from its own positively-labeled trace.

    Raises :class:`UnlabeledTraceError` when ``trace.outcome`` is not exactly
    True - including None, which means nobody has labeled it (PLAN.md Step 8).
    """
    if trace.outcome is not True:
        raise UnlabeledTraceError(
            f"refusing to synthesize a verifier from trace {trace.trace_id!r} with "
            f"outcome={trace.outcome!r}. 'Do what production did' is only a reward "
            f"function when production was labeled correct; a failed or unlabeled "
            f"trajectory would become the target."
        )
    if trace.trace_id != task.trace_id:
        raise ValueError(
            f"task {task.task_id!r} came from trace {task.trace_id!r}, not {trace.trace_id!r}"
        )

    externals = list(external_tools) if external_tools is not None else external_tools_of(schema, trace)
    final = reconstruct_final_state(trace, schema)
    deleted = _deleted_rows(trace, schema)
    pre_by_entity = {er.entity: er for er in task.pre_state}
    assertions: list[Assertion] = []

    for entity in schema.entities:
        pk = entity.primary_key
        pre_er: EntityRows | None = pre_by_entity.get(entity.name)
        pre_idx = _index(pre_er.rows, pk) if pre_er else {}
        post_idx = final.get(entity.name, {})
        partial_ids = partial_key_ids(task, entity.name)

        if entity.static_snapshot:
            # Read-only, never cross-referenced. We refuse to invent structure
            # for it upstream; asserting on it here would assert on a guess.
            continue
        if pk is None:
            continue

        changed_any = False
        row_assertions: list[Assertion] = []

        for key_id, post in post_idx.items():
            if (entity.name, key_id) in deleted:
                # A later read reported this row missing. Absence wins over any
                # earlier write response, and the two must never both be
                # asserted - a verifier that contradicts itself can never pass.
                continue
            pre = pre_idx.get(key_id)
            key_value = post.get(pk)
            rk = row_key_for(pk, key_value)
            if pre is None:
                changed_any = True
                row_assertions.append(
                    Assertion(
                        kind=AssertionKind.ROW_EXISTS,
                        entity=entity.name,
                        row_key=rk,
                        expected=post,
                        description=(
                            f"{entity.name}[{key_value!r}] must exist at the end of the episode; "
                            f"it was not in the starting state"
                        ),
                    )
                )
                continue
            diffs = {k: v for k, v in post.items() if k in pre and pre[k] != v}
            same = {k: v for k, v in pre.items() if k not in diffs}
            if diffs:
                changed_any = True
                for fname, value in diffs.items():
                    row_assertions.append(
                        Assertion(
                            kind=AssertionKind.STATE_EQUALS,
                            entity=entity.name,
                            row_key=rk,
                            field=fname,
                            expected=value,
                            description=(
                                f"{entity.name}[{key_value!r}].{fname} must be {value!r} "
                                f"(was {pre[fname]!r} in the starting state)"
                            ),
                        )
                    )
                if same:
                    row_assertions.append(
                        Assertion(
                            kind=AssertionKind.STATE_UNCHANGED,
                            entity=entity.name,
                            row_key=rk,
                            expected=same,
                            description=(
                                f"the other fields of {entity.name}[{key_value!r}] must be "
                                f"untouched: {sorted(same)}"
                                + (_PARTIAL_NOTE if key_id in partial_ids else "")
                            ),
                        )
                    )

        for key_id, pre in pre_idx.items():
            key_value = pre.get(pk)
            rk = row_key_for(pk, key_value)
            if (entity.name, key_id) in deleted:
                changed_any = True
                row_assertions.append(
                    Assertion(
                        kind=AssertionKind.ROW_ABSENT,
                        entity=entity.name,
                        row_key=rk,
                        description=(
                            f"{entity.name}[{key_value!r}] must be gone; the trace read it back "
                            f"as not_found after a write"
                        ),
                    )
                )
            elif key_id not in post_idx:
                # Never looked at again. Silence is not deletion: assert it is
                # still exactly as it started.
                row_assertions.append(
                    Assertion(
                        kind=AssertionKind.STATE_UNCHANGED,
                        entity=entity.name,
                        row_key=rk,
                        expected=pre,
                        description=(
                            f"{entity.name}[{key_value!r}] was never written and must be "
                            f"unchanged"
                            + (_PARTIAL_NOTE if key_id in partial_ids else "")
                        ),
                    )
                )

        if not changed_any and pre_idx:
            # Whole entity untouched. One assertion instead of N, and it also
            # covers rows the episode never read back.
            assertions.append(
                Assertion(
                    kind=AssertionKind.STATE_UNCHANGED,
                    entity=entity.name,
                    expected=list(pre_idx.values()),
                    description=(
                        f"the {entity.name} entity must be untouched; the reference episode "
                        f"changed nothing in it"
                        + (
                            _PARTIAL_NOTE
                            if partial_ids & set(pre_idx)
                            else ""
                        )
                    ),
                )
            )
        else:
            assertions.extend(row_assertions)

    for tool in externals:
        count = sum(1 for inv in trace.invocations if inv.tool == tool)
        assertions.append(
            Assertion(
                kind=AssertionKind.EFFECT_COUNT,
                tool=tool,
                expected=count,
                description=(
                    f"{tool} must be attempted exactly {count} time(s); the effect ledger "
                    f"records the attempt, the effect is never performed"
                ),
            )
        )

    return Verifier(
        verifier_id=verifier_id or f"ver-{task.task_id}",
        task_id=task.task_id,
        assertions=tuple(assertions),
        reviewed_by=None,
    )
