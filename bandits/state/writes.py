"""Infer *what* a write tool writes, not merely *that* it writes.

:func:`bandits.state.entities.infer_writers` and
:func:`~bandits.state.entities.split_by_surface` answer "does this tool mutate
the entity?" and stop there. The evidence they consult - the before/after
windows around a call, inside one trace, on one row - also carries the *content*
of the write, and this module keeps it instead of throwing it away.

Why it matters: without this, the only thing downstream has left is the tool's
English name, and ``refund_order -> status="refunded"`` is a verb-tense pun that
breaks the first time it meets ``approve_return -> "authorized"`` or a localized
tool name. Everything below is derived from observed values.

The chain of reasoning, one documented rule per step:

1. **Window.** For each successful write, reconstruct the row as it was
   *before* (:func:`_pre_state`) and as it was *after* (:func:`_post_state`).
2. **Diff.** The columns that differ, or that appear for the first time, are
   what the call changed (:func:`_changed_columns`).
3. **Key.** The argument whose value is the row's key names the row
   (:func:`_infer_key_argument`).
4. **Arguments -> columns.** An argument whose *value* turns up in a changed
   column is the source of that column (:func:`_infer_argument_columns`).
5. **Constants.** A changed column that took the same value on every single
   successful call, and that no argument supplied, is a constant the tool sets
   (:func:`_infer_constants`).
6. **Echoes.** The columns the write's own response body hands back
   (:func:`_response_echoes`).

**Only ``CallStatus.OK`` calls are evidence.** An errored ``refund_order``
changed nothing, and its ``{"error": "already_refunded", "order_id": 7741}``
body would otherwise donate a bogus ``error`` column and a phantom diff. Error
calls never reach this module: they are already filtered out of
:class:`~bandits.state.entities.Observation` by
:func:`~bandits.state.entities.group_observations`.

**When evidence is absent.** A tool that ``written_by`` names but that produced
no attributable successful response still gets a
:class:`~bandits.contracts.WriteEffect`, with ``evidence_count == 0``,
``confidence == 0.0``, empty inferences and an ``evidence`` string saying so.
Emitting the empty record rather than nothing keeps the invariant
``{e.tool for e in write_effects} == set(written_by)``, so a consumer that
finds no effect for a writer is looking at a bug, not at silence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from bandits.contracts import JsonObject, JsonValue, WriteEffect

from .entities import EntityDraft, Observation, _sort_key
from .identifiers import Scalar, is_identifier_scalar

#: Confidence for a write backed by one successful call with a before/after
#: diff, plus the bonus per additional corroborating call, and the ceiling.
#: Never 1.0: the corpus is a sample, and a later trace can always contradict.
BASE_CONFIDENCE = 0.5
PER_EXTRA_CALL = 0.15
MAX_CONFIDENCE = 0.95

#: Ceiling when *no* call had a prior observation of the row, so every claim
#: rests on the write's own response rather than on an observed change.
NO_DIFF_CONFIDENCE = 0.4


def _same(a: JsonValue, b: JsonValue) -> bool:
    """Value equality that survives unhashable JSON (lists, nested objects)."""
    return _sort_key(a) == _sort_key(b)


@dataclass
class CallEvidence:
    """One successful write call, with the row before and after it."""

    obs: Observation
    pre: JsonObject
    """Merged body of the row from every earlier observation in the same trace."""

    post: JsonObject
    """The write's own response, merged with later reads of the same row."""

    changed: JsonObject
    """Columns ``post`` shows that ``pre`` did not have, or had differently."""

    had_pre: bool
    """False when this call was the first sighting of the row - weaker evidence."""

    @property
    def tool(self) -> str:
        return self.obs.tool

    @property
    def arguments(self) -> JsonObject:
        return self.obs.inv.arguments or {}

    @property
    def body(self) -> JsonObject:
        return self.obs.body


# --------------------------------------------------------------------------
# rule 1+2 - the before/after window and its diff
# --------------------------------------------------------------------------


def _row_timelines(draft: EntityDraft) -> dict[tuple[str, Scalar], list[Observation]]:
    """Observations grouped by ``(trace_id, row key)``, in step order.

    State does not change on its own *within* one episode, so a difference
    between two observations of the same row in the same trace is a write. Rows
    are never joined *across* traces: two traces are two independent worlds, and
    order 7741 is ``delivered`` at the start of one and ``refunded`` at the start
    of another.
    """
    rows: dict[tuple[str, Scalar], list[Observation]] = defaultdict(list)
    for o in draft.observations:
        rows[(o.inv.trace_id, o.anchor_value)].append(o)
    for obs in rows.values():
        obs.sort(key=lambda o: o.inv.step)
    return dict(rows)


def _pre_state(timeline: list[Observation], position: int) -> JsonObject:
    """The row as last seen before ``timeline[position]``.

    Every earlier body is merged left to right, because different tools project
    different subsets: ``search`` shows an id, ``get_order`` shows six columns.
    The merge is the most complete picture of the row we had before the write.
    """
    pre: JsonObject = {}
    for earlier in timeline[:position]:
        pre.update(earlier.body)
    return pre


def _post_state(
    timeline: list[Observation],
    position: int,
    writers: set[str],
) -> JsonObject:
    """The row as seen after ``timeline[position]``, up to the next write.

    **Rule.** Start from the write's own response, then merge the reads that
    follow it in the same trace, stopping at the next observation produced by
    *another* writing tool. The stop is what keeps ``update_order_status``'s
    change from being credited to the ``refund_order`` that ran before it.

    **Why merge later reads at all.** A write that echoes nothing but the id
    (``{"ok": true, "order_id": 7741}``) is still fully observable if a
    ``get_order`` follows it. Without the merge, such a tool would yield an
    empty ``WriteEffect`` despite the corpus containing the answer.

    **Failure mode.** A later read also reflects changes made by something we
    did *not* record - a background job, a second agent, a webhook. In a
    single-agent episode that is rare; in a multi-actor system it will
    misattribute. ``confidence`` never reaches 1.0 for this reason.
    """
    post: JsonObject = dict(timeline[position].body)
    for later in timeline[position + 1 :]:
        if later.tool in writers:
            break
        post.update(later.body)
    return post


def _changed_columns(pre: JsonObject, post: JsonObject, anchor_field: str | None) -> JsonObject:
    """Columns the write introduced or altered.

    **Rule.** ``post[c]`` counts as changed when ``c`` is absent from ``pre`` or
    holds a different value there. The anchor is always excluded: echoing the
    key back is addressing, not writing.

    Note this rule handles projection width for free. ``refund_order`` answers
    with three columns where ``get_order`` answered with six; the three are
    compared against what was known and only the genuinely different ones
    survive, so a narrow projection is never mistaken for a deletion.
    """
    changed = {
        k: v
        for k, v in post.items()
        if k != anchor_field and (k not in pre or not _same(pre[k], v))
    }
    return changed


def collect_evidence(draft: EntityDraft, writers: set[str]) -> dict[str, list[CallEvidence]]:
    """Build the before/after evidence for every writing tool on this entity.

    Only successful calls are present at all (see the module docstring). A call
    that is the first sighting of its row has no ``pre`` to diff against; rather
    than declare every column of the merged window "changed" - which would sweep
    in untouched columns contributed by a following read - we fall back to the
    write's *own* response body as the change set, and mark ``had_pre`` False so
    the confidence is capped.
    """
    evidence: dict[str, list[CallEvidence]] = defaultdict(list)
    for timeline in _row_timelines(draft).values():
        for i, obs in enumerate(timeline):
            if obs.tool not in writers:
                continue
            pre = _pre_state(timeline, i)
            post = _post_state(timeline, i, writers)
            had_pre = bool(pre)
            if had_pre:
                changed = _changed_columns(pre, post, draft.anchor_field)
            else:
                changed = _changed_columns({}, dict(obs.body), draft.anchor_field)
            evidence[obs.tool].append(
                CallEvidence(obs=obs, pre=pre, post=post, changed=changed, had_pre=had_pre)
            )
    for calls in evidence.values():
        calls.sort(key=lambda c: (c.obs.inv.trace_id, c.obs.inv.step))
    return dict(evidence)


# --------------------------------------------------------------------------
# rule 3 - which argument names the row
# --------------------------------------------------------------------------


def _infer_key_argument(calls: list[CallEvidence], anchor_field: str | None) -> str | None:
    """The argument that identifies the row being written.

    **Rule.** An argument qualifies when its value equals the row's key on
    *every* successful call. Among the qualifiers, an argument named exactly
    like the anchor field wins; otherwise the alphabetically first, for
    determinism.

    **Evidence required.** Agreement on all calls. One disagreement disqualifies
    the argument outright - a key that is right most of the time is not a key.

    **When evidence is insufficient.** ``None``. A write whose row handle we
    cannot name is still recorded as a write; the caller must not invent one.
    """
    if not calls:
        return None
    qualifying: set[str] | None = None
    for call in calls:
        matches = {
            name
            for name, value in call.arguments.items()
            if is_identifier_scalar(value) and _same(value, call.obs.anchor_value)
        }
        qualifying = matches if qualifying is None else (qualifying & matches)
    if not qualifying:
        return None
    if anchor_field in qualifying:
        return anchor_field
    return sorted(qualifying)[0]


# --------------------------------------------------------------------------
# rule 4 - which argument feeds which column
# --------------------------------------------------------------------------


def _name_affinity(argument: str, column: str) -> int:
    """Crude, explicit name-similarity score, used **only** to break ties.

    Never to make a match: names are a naming convention, values are evidence.
    ``amount_cents`` and ``refund_amount_cents`` score high because one is a
    suffix of the other; two unrelated names score 0.
    """
    if argument == column:
        return 3
    if column.endswith("_" + argument) or argument.endswith("_" + column):
        return 2
    if argument in column or column in argument:
        return 1
    return 0


def _infer_argument_columns(
    calls: list[CallEvidence],
    key_argument: str | None,
    anchor_field: str | None,
) -> tuple[dict[str, str], list[str]]:
    """Map argument name -> column name by matching **values**, not names.

    **Rule.** Argument ``a`` feeds column ``c`` when, on every successful call
    that passed ``a``, ``c`` is among the columns the call changed and
    ``post[c] == arguments[a]``. ``refund_order(amount_cents=4200)`` and
    ``orders.refund_amount_cents`` becoming ``4200`` in that same call is the
    whole of the evidence.

    Per the contract, only mappings where the two names *differ* are published:
    an argument that feeds an identically named column needs no map, and the
    identity case is recorded in ``evidence`` instead so a reviewer still sees
    it. The key argument is excluded - it addresses the row, it does not set it.

    **Tie-break.** If two changed columns took the same value in every call,
    :func:`_name_affinity` picks; a genuine tie on that too is left unmapped
    rather than guessed.

    **Evidence required.** A value match on every call that supplied the
    argument, and at least one such call.

    **Failure modes, all inherent to matching on values:**

    * *Coincidence.* ``amount_cents=4200`` and an unrelated ``total_cents``
      that also happens to be ``4200`` and also happens to change. Requiring
      agreement across every call makes this unlikely but not impossible; two
      calls with distinct amounts kill it.
    * *Transformation.* ``amount_cents=4200`` stored as ``refund_amount=42.00``
      or as ``"$42.00"`` is invisible here. We publish nothing rather than
      guess a conversion.
    * *Single-call corpora.* One call cannot distinguish a mapping from a
      coincidence at all; ``confidence`` reports that honestly instead of the
      mapping being withheld.
    """
    if not calls:
        return {}, []

    per_argument: dict[str, list[set[str]]] = defaultdict(list)
    examples: dict[str, JsonValue] = {}
    for call in calls:
        for name, value in call.arguments.items():
            if name == key_argument:
                continue
            cols = {
                c
                for c, cv in call.changed.items()
                if c != anchor_field and _same(cv, value)
            }
            per_argument[name].append(cols)
            examples.setdefault(name, value)

    mapping: dict[str, str] = {}
    notes: list[str] = []
    for name, per_call in sorted(per_argument.items()):
        common: set[str] = set.intersection(*per_call) if per_call else set()
        if not common:
            continue
        ranked = sorted(common, key=lambda c: (-_name_affinity(name, c), c))
        if len(ranked) > 1 and _name_affinity(name, ranked[0]) == _name_affinity(name, ranked[1]):
            notes.append(
                f"argument {name!r} matched columns {sorted(common)} equally on "
                f"{len(per_call)} call(s); left unmapped rather than guessed"
            )
            continue
        column = ranked[0]
        notes.append(
            f"argument {name!r} -> column {column!r}: value matched on "
            f"{len(per_call)}/{len(per_call)} call(s) (e.g. {examples[name]!r})"
            + ("; names identical, not published in argument_columns" if column == name else "")
        )
        if column != name:
            mapping[name] = column
    return mapping, notes


# --------------------------------------------------------------------------
# rule 5 - the constants a tool always sets
# --------------------------------------------------------------------------


def _infer_constants(
    calls: list[CallEvidence],
    anchor_field: str | None,
    argument_columns: dict[str, str],
) -> tuple[JsonObject, list[str]]:
    """Columns this tool set to the *same observed value* on every call.

    **Rule.** Column ``c`` is a constant when

    1. ``c`` is among the changed columns of **every** successful call,
    2. every call set it to the same value,
    3. ``c`` is not the anchor, and
    4. no argument supplied that value. A column whose value tracks an argument
       is argument-driven, not constant - this is what stops
       ``update_order_status(status="delivered")`` from being written down as
       "this tool always sets status to delivered" on the strength of a corpus
       that happened to call it once.

    The value comes from the observed diff and from the write's own response.
    It never comes from the tool's name.

    **Evidence required.** Unanimity. A column that changed to ``"refunded"``
    twice and ``"partially_refunded"`` once is not a constant, and is omitted
    entirely rather than reported with the majority value.

    **When evidence is insufficient.** The column is left out. A missing
    constant costs a downstream stage a hint; a wrong one makes the rebuilt
    environment confidently lie.
    """
    if not calls:
        return {}, []

    candidates: set[str] = set(calls[0].changed)
    for call in calls[1:]:
        candidates &= set(call.changed)
    candidates.discard(anchor_field)
    candidates -= set(argument_columns.values())

    constants: JsonObject = {}
    notes: list[str] = []
    for column in sorted(candidates):
        values = [c.changed[column] for c in calls]
        if len({_sort_key(v) for v in values}) != 1:
            notes.append(
                f"column {column!r} changed to {len({_sort_key(v) for v in values})} different "
                f"values across {len(values)} call(s); not a constant"
            )
            continue
        value = values[0]
        if all(
            any(_same(v, value) for v in call.arguments.values()) for call in calls
        ):
            notes.append(
                f"column {column!r} was always equal to a supplied argument; "
                "argument-driven, not a constant"
            )
            continue
        constants[column] = value
        befores = sorted(
            {_sort_key(c.pre[column])[1] for c in calls if column in c.pre}
        )
        notes.append(
            f"column {column!r} = {value!r} on all {len(calls)} successful call(s)"
            + (f"; changed from {', '.join(befores)}" if befores else "")
        )
    return constants, notes


# --------------------------------------------------------------------------
# rule 6 - what the write hands back
# --------------------------------------------------------------------------


def _response_echoes(calls: list[CallEvidence], columns: set[str]) -> tuple[str, ...]:
    """Entity columns the write's own response body contains.

    Restricted to columns the entity actually has, so a protocol field the tool
    tacks on (``{"ok": true}``) is not published as state. This is what lets a
    rebuilt tool answer in the shape the recorded one did.
    """
    seen: set[str] = set()
    for call in calls:
        seen |= {k for k in call.body if k in columns}
    return tuple(sorted(seen))


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _confidence(calls: list[CallEvidence]) -> float:
    """Confidence in a write's inferred semantics.

    One corroborated call earns :data:`BASE_CONFIDENCE`; each additional one
    adds :data:`PER_EXTRA_CALL` up to :data:`MAX_CONFIDENCE`. Never 1.0 - see
    :func:`_post_state`. If no call had a prior observation of its row, every
    claim rests on the write's own response with nothing to diff against, and
    the result is capped at :data:`NO_DIFF_CONFIDENCE`.
    """
    if not calls:
        return 0.0
    score = min(MAX_CONFIDENCE, BASE_CONFIDENCE + PER_EXTRA_CALL * (len(calls) - 1))
    if not any(c.had_pre for c in calls):
        score = min(score, NO_DIFF_CONFIDENCE)
    return round(score, 2)


def infer_write_effects(
    draft: EntityDraft,
    writers: set[str],
    columns: set[str],
) -> tuple[WriteEffect, ...]:
    """One :class:`~bandits.contracts.WriteEffect` per tool in ``writers``.

    Returned sorted by tool name. Writers with no attributable successful call
    get an explicitly empty effect (see the module docstring), so the result
    always covers exactly ``EntitySchema.written_by``.
    """
    evidence = collect_evidence(draft, writers)
    effects: list[WriteEffect] = []
    for tool in sorted(writers):
        calls = evidence.get(tool, [])
        if not calls:
            effects.append(
                WriteEffect(
                    tool=tool,
                    evidence=(
                        f"no successful {tool} call produced a response attributable to "
                        f"{draft.name!r}; nothing inferred about what it writes",
                    ),
                )
            )
            continue

        key_argument = _infer_key_argument(calls, draft.anchor_field)
        argument_columns, arg_notes = _infer_argument_columns(
            calls, key_argument, draft.anchor_field
        )
        constants, const_notes = _infer_constants(calls, draft.anchor_field, argument_columns)
        echoes = _response_echoes(calls, columns)

        diffed = sum(1 for c in calls if c.had_pre)
        lines = [
            f"{len(calls)} successful {tool} call(s) attributed to {draft.name!r}; "
            f"{diffed} with a prior observation of the row to diff against",
        ]
        if key_argument:
            lines.append(
                f"argument {key_argument!r} matched the row key on all {len(calls)} call(s)"
            )
        else:
            lines.append("no argument matched the row key on every call; key_argument left unset")
        lines.extend(const_notes)
        lines.extend(arg_notes)
        if echoes:
            lines.append(f"response echoes entity columns: {', '.join(echoes)}")

        effects.append(
            WriteEffect(
                tool=tool,
                key_argument=key_argument,
                argument_columns=argument_columns,
                sets_constants=constants,
                response_echoes=echoes,
                evidence_count=len(calls),
                confidence=_confidence(calls),
                evidence=tuple(lines),
            )
        )
    return tuple(effects)


__all__ = [
    "BASE_CONFIDENCE",
    "CallEvidence",
    "MAX_CONFIDENCE",
    "NO_DIFF_CONFIDENCE",
    "PER_EXTRA_CALL",
    "collect_evidence",
    "infer_write_effects",
]
