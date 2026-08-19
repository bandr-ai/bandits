"""Infer the tables behind the tools.

This is the core of stage 3 and the hardest step in PLAN.md (Step 7). The input
is a stream of ``(tool, arguments, response)``. The output is a set of
:class:`~bandits.contracts.EntitySchema` objects that make that stream
consistent.

The chain of reasoning, each link of which is a documented rule below:

1. **Anchor.** A response body describes a row, and the row is identified by
   the identifier the caller passed in and the response echoed back
   (:func:`choose_anchor`). ``get_order(order_id=7741)`` -> body echoing
   ``order_id: 7741`` means "this body is the row ``orders#7741``".
2. **Group.** All bodies with the same anchor field are the same table
   (:func:`group_observations`).
3. **Name.** The table is named by pluralizing the anchor's stem, with the
   noun taken from the tools when the anchor carries no stem (:func:`entity_name`).
4. **Fields.** The columns are the *union* of every field any response ever
   showed, because different tools project different subsets
   (:func:`build_field_profiles`).
5. **Reads and writes.** From the :class:`~bandits.contracts.ToolSurface` when
   we have one; from observed body changes when we do not (:func:`infer_writers`).
6. **Degrade honestly.** A body with no identifier at all is not forced into a
   table; it becomes a static snapshot or is left unresolved
   (:func:`snapshot_drafts`).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from bandits.contracts import (
    FieldProfile,
    InvocationPoint,
    JsonObject,
    JsonValue,
    ToolClass,
    ToolSurface,
    TraceCorpus,
)

from .identifiers import (
    IdentifierIndex,
    Scalar,
    _singular_candidates,
    find_identifiers,
    ok_invocations,
    scalar_fields,
    scalar_list_fields,
)

#: Verb prefixes stripped when deriving a noun from a tool name. Explicit list,
#: no stemming library: an unknown verb simply yields no noun, and we fall back
#: to the field name rather than mangling it.
TOOL_VERBS = (
    "get",
    "fetch",
    "read",
    "load",
    "lookup",
    "look_up",
    "find",
    "search",
    "list",
    "query",
    "describe",
    "show",
    "retrieve",
    "create",
    "new",
    "add",
    "insert",
    "update",
    "set",
    "edit",
    "modify",
    "patch",
    "delete",
    "remove",
    "cancel",
    "refund",
    "place",
    "submit",
)

MAX_SAMPLE_VALUES = 5


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


@dataclass
class Observation:
    """One successful response body, attributed to one row of one entity."""

    inv: InvocationPoint
    body: JsonObject
    anchor_field: str
    anchor_value: Scalar

    @property
    def tool(self) -> str:
        return self.inv.tool


@dataclass
class EntityDraft:
    """Everything gathered about one candidate entity before it is frozen."""

    name: str
    anchor_field: str | None
    observations: list[Observation] = field(default_factory=list)
    reference_tools: set[str] = field(default_factory=set)
    """Tools that only mentioned this entity's ids in a list, never a body."""

    snapshot_rows: list[JsonObject] = field(default_factory=list)
    """Verbatim rows, for entities we refuse to model as tables."""

    forced_static: bool = False
    """True for identifier-less snapshots: static regardless of relations."""

    snapshot_observations: int = 0
    """Calls backing a snapshot, before duplicate rows were collapsed."""

    @property
    def tools(self) -> set[str]:
        return {o.tool for o in self.observations} | self.reference_tools

    @property
    def evidence_count(self) -> int:
        return len(self.observations) + self.snapshot_observations

    def key_values(self) -> set[Scalar]:
        return {o.anchor_value for o in self.observations}

    def field_values(self) -> dict[str, list[JsonValue]]:
        """field name -> every value observed for it, in observation order."""
        out: dict[str, list[JsonValue]] = defaultdict(list)
        for o in self.observations:
            for k, v in o.body.items():
                out[k].append(v)
        return dict(out)


# --------------------------------------------------------------------------
# rule 1 - anchors
# --------------------------------------------------------------------------


def echo_candidates(inv: InvocationPoint, index: IdentifierIndex) -> list[str]:
    """Identifier fields a call passed *in* and got the same value back *out*.

    **Rule.** ``F`` is an echo candidate for a call when ``F`` is an identifier
    field (see :mod:`bandits.state.identifiers`), ``arguments[F] == response[F]``,
    and both are identifier-shaped scalars. This is the strongest evidence we
    can get without a schema: the caller named a row and the tool handed that
    row back.

    **Evidence required.** One call. The echo is self-evidencing.

    **When evidence is insufficient.** An empty list, which sends the caller to
    :func:`choose_anchor`'s weaker body-only path.
    """
    args = scalar_fields(inv.arguments or {})
    body = scalar_fields(inv.response)
    return sorted(
        name
        for name, value in body.items()
        if index.has(name) and name in args and args[name] == value
    )


def _anchor_consensus(invocations: list[InvocationPoint], index: IdentifierIndex) -> Counter:
    """How often each field was the *sole* echo candidate of a call.

    A field that alone identifies a row somewhere in the corpus has earned the
    right to break ties elsewhere. ``update_order_status(order_id, status)``
    echoes both ``order_id`` and ``status``; only ``order_id`` is ever a sole
    echo (in ``get_order``, ``refund_order``), so it wins and ``status`` stays
    an attribute.
    """
    votes: Counter = Counter()
    for inv in invocations:
        cands = echo_candidates(inv, index)
        if len(cands) == 1:
            votes[cands[0]] += 1
    return votes


def choose_anchor(
    inv: InvocationPoint,
    index: IdentifierIndex,
    consensus: Counter,
) -> tuple[str, Scalar] | None:
    """Pick the identifier field that names the row a response describes.

    **Rule, in priority order.**

    1. If exactly one field is echoed from arguments to response, that is the
       anchor.
    2. If several are echoed, take the one with the most sole-echo votes
       corpus-wide (:func:`_anchor_consensus`); ties break on the number of
       distinct values the field takes, then alphabetically, so the result is
       deterministic.
    3. If none is echoed (the call did not name the row - e.g. a create), fall
       back to identifier fields *present in the body*, ranked the same way,
       and require the field to have won at least one sole-echo vote somewhere
       else in the corpus. Without that vote we have no evidence the field is a
       key rather than a reference (``get_order``'s body carries ``customer_id``
       and ``sku``, but neither identifies the order).

    **Evidence required.** Either an echo, or corpus-wide proof that the field
    anchors rows elsewhere.

    **When evidence is insufficient.** ``None`` - the body is not attributed to
    any table, and it ends up as a snapshot or in ``unresolved``. We do not
    pick "the first ``*_id``-looking field": name shape is not evidence.
    """
    cands = echo_candidates(inv, index)
    body = scalar_fields(inv.response)
    if not cands:
        cands = sorted(n for n in body if index.has(n) and consensus.get(n, 0) > 0)
    if not cands:
        return None
    best = max(cands, key=lambda n: (consensus.get(n, 0), len(index.values(n)), n))
    return best, body[best]


# --------------------------------------------------------------------------
# rule 3 - naming
# --------------------------------------------------------------------------


def pluralize(word: str) -> str:
    """Pluralize an English noun with a small, explicit rule.

    Deliberately not a library - the rule must be readable and reviewable:

    * ends in ``s``, ``x``, ``z``, ``ch``, ``sh`` -> ``+es``
    * ends in consonant + ``y``                   -> ``y`` becomes ``ies``
    * otherwise                                   -> ``+s``

    Irregular English nouns (``person`` -> ``people``) come out wrong
    (``persons``). That is a cosmetic miss, not a structural one: the name is a
    label, and every downstream stage keys off it consistently.
    """
    if not word:
        return word
    lower = word.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if lower.endswith("y") and len(word) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


#: Word endings that look plural but are not. Stripping the ``s`` from these
#: produces nonsense (``status`` -> ``statu``), so they are left alone.
NOT_PLURAL_ENDINGS = ("ss", "us", "is", "as", "os")


def singularize(word: str) -> str:
    """Inverse of :func:`pluralize`, same explicit-rule discipline.

    Words ending in :data:`NOT_PLURAL_ENDINGS` are returned unchanged: they end
    in ``s`` without being plural.
    """
    lower = word.lower()
    if lower.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if lower.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if lower.endswith(NOT_PLURAL_ENDINGS):
        return word
    if lower.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def tool_noun(tool: str) -> str | None:
    """The noun in a tool name, or ``None`` when no known verb prefixes it.

    ``get_product`` -> ``product``, ``search_orders`` -> ``order``,
    ``update_order_status`` -> ``order_status``. A tool whose name starts with
    no verb in :data:`TOOL_VERBS` yields ``None`` rather than a guess.
    """
    for verb in TOOL_VERBS:
        prefix = verb + "_"
        if tool.startswith(prefix) and len(tool) > len(prefix):
            return singularize(tool[len(prefix) :])
    return None


def entity_name(anchor_field: str, tools: set[str]) -> str:
    """Name the table an anchor field identifies.

    **Rule.**

    1. ``<stem>_id`` -> ``pluralize(stem)``. ``order_id`` -> ``orders``,
       ``customer_id`` -> ``customers``.
    2. Otherwise the anchor carries no stem (``sku``, ``id``), so take the noun
       from the tools that anchor on it: the most common
       :func:`tool_noun`, ties broken by shortest then alphabetical, and
       pluralize it. ``sku`` is anchored only by ``get_product`` -> ``products``.
    3. If no tool yields a noun, pluralize the field name itself (``sku`` ->
       ``skus``). Ugly, but it is the observed evidence and nothing else.

    **Evidence required.** For rule 2, at least one tool name with a known verb
    prefix.
    """
    if anchor_field.endswith("_id") and len(anchor_field) > 3:
        return pluralize(anchor_field[: -len("_id")])
    nouns = Counter(n for n in (tool_noun(t) for t in sorted(tools)) if n)
    if nouns:
        best = min(nouns.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0]
        return pluralize(best)
    if anchor_field == "id":
        return "rows"
    return pluralize(anchor_field)


def snapshot_name(tool: str) -> str:
    """Name for an identifier-less snapshot: the tool's noun, kept singular.

    ``get_store_policy`` -> ``store_policy``. It is **not** pluralized: we have
    no evidence this is a table with rows, and a plural name would imply one.
    A tool with no recognizable verb keeps its own name.
    """
    noun = tool_noun(tool)
    return noun if noun else tool


# --------------------------------------------------------------------------
# rule 4 - fields
# --------------------------------------------------------------------------


def json_type(value: JsonValue) -> str:
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
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _sort_key(value: JsonValue) -> tuple[str, str]:
    return (json_type(value), json.dumps(value, sort_keys=True, default=str))


def build_field_profiles(
    bodies: list[JsonObject],
    primary_keys: set[str],
) -> tuple[FieldProfile, ...]:
    """Profile the **union** of every field any body ever showed.

    **Rule.** Union, never intersection. Different tools project different
    subsets of the same row: ``refund_order`` is the only tool that ever
    returns ``refund_amount_cents``, and dropping it because ``get_order``
    never showed it would delete a real column. ``occurrences`` records how
    thin the evidence for each field is, so a downstream stage can tell a
    universal column from one seen once.

    **Evidence required.** One body containing the field.

    **When evidence is insufficient.** Nothing is inferred about nullability or
    domain beyond what was literally observed: ``null_count`` counts observed
    nulls only, and absence from a projection is *not* recorded as a null.
    """
    order: list[str] = []
    seen: set[str] = set()
    for body in bodies:
        for k in body:
            if k not in seen:
                seen.add(k)
                order.append(k)

    profiles: list[FieldProfile] = []
    for name in order:
        values = [b[name] for b in bodies if name in b]
        types = tuple(sorted({json_type(v) for v in values}))
        distinct: list[JsonValue] = []
        distinct_keys: set[str] = set()
        for v in values:
            k = _sort_key(v)[1]
            if k not in distinct_keys:
                distinct_keys.add(k)
                distinct.append(v)
        samples = tuple(sorted(distinct, key=_sort_key)[:MAX_SAMPLE_VALUES])
        profiles.append(
            FieldProfile(
                name=name,
                json_types=types,
                occurrences=len(values),
                null_count=sum(1 for v in values if v is None),
                distinct_values=len(distinct),
                sample_values=samples,
                looks_like_identifier=name in primary_keys,
            )
        )
    return tuple(profiles)


# --------------------------------------------------------------------------
# rule 5 - who reads, who writes
# --------------------------------------------------------------------------


def infer_writers(draft: EntityDraft) -> set[str]:
    """Writers inferred from the corpus alone, when no ToolSurface is given.

    **Rule.** A tool writes an entity when, inside a single trace, its response
    for a row disagrees with what was already observed for that same row on a
    field they share. State does not change by itself within an episode, so a
    disagreement is a write, and the call that produced the new value is the
    writer.

    Bodies are merged as the trace proceeds, so a later projection is compared
    against everything seen so far. Comparison is on *shared* fields only:
    ``refund_order`` returning ``{order_id, status, refund_amount_cents}`` after
    ``get_order`` returned the full row is a narrower projection, not a change -
    it counts as a write only because ``status`` actually differs.

    **Evidence required.** One in-trace disagreement. A single observation of a
    row can never demonstrate a write.

    **When evidence is insufficient.** The tool is not called a writer. A write
    tool whose effect is never observed by a later read - and which does not
    echo the change itself - is indistinguishable from a read in the trace, and
    we say so by leaving it out rather than guessing from its name. Provide a
    ``ToolSurface`` to do better.
    """
    writers: set[str] = set()
    by_trace: dict[str, list[Observation]] = defaultdict(list)
    for o in draft.observations:
        by_trace[o.inv.trace_id].append(o)
    for observations in by_trace.values():
        observations.sort(key=lambda o: o.inv.step)
        known: dict[Scalar, JsonObject] = {}
        for o in observations:
            prev = known.get(o.anchor_value)
            if prev is not None:
                changed = any(
                    k in prev and _sort_key(prev[k]) != _sort_key(v) for k, v in o.body.items()
                )
                if changed:
                    writers.add(o.tool)
                known[o.anchor_value] = {**prev, **o.body}
            else:
                known[o.anchor_value] = dict(o.body)
    return writers


def split_by_surface(
    tools: set[str], surface: ToolSurface
) -> tuple[set[str], set[str]]:
    """Split an entity's tools into (writers, readers) using stage 2's classes.

    **Rule.** ``ToolClass.WRITE`` -> writer, ``ToolClass.READ`` -> reader.
    ``EXTERNAL`` and ``UNKNOWN`` are placed in **neither** list. An unknown tool
    must never be probed (PLAN.md Step 9) and must not be silently assumed
    harmless; leaving it out of both lists is the honest record of what we know.
    """
    writers, readers = set(), set()
    for name in tools:
        profile = surface.by_name(name)
        if profile is None:
            continue
        if profile.tool_class is ToolClass.WRITE:
            writers.add(name)
        elif profile.tool_class is ToolClass.READ:
            readers.add(name)
    return writers, readers


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def _entity_tools(surface: ToolSurface | None) -> set[str] | None:
    """Tools that may touch the store at all, or ``None`` when unknown.

    External tools do not read or write the reconstructed database - their
    responses are acknowledgements from a third party - so their bodies must
    never become entities. Without a surface we cannot tell, and every tool is
    considered.
    """
    if surface is None:
        return None
    return {
        p.name
        for p in surface.tools
        if p.tool_class in (ToolClass.READ, ToolClass.WRITE, ToolClass.UNKNOWN)
    }


def group_observations(
    corpus: TraceCorpus,
    index: IdentifierIndex,
    surface: ToolSurface | None = None,
) -> tuple[dict[str, EntityDraft], list[InvocationPoint]]:
    """Group successful response bodies into entity drafts.

    Returns ``(drafts_by_anchor_field, unattributed_invocations)``. Only
    ``CallStatus.OK`` calls are considered: an error body is a protocol
    response, and letting ``{"error": "not_found", "order_id": 9999}`` through
    would put an ``error`` column on ``orders`` and a phantom row 9999 in it.
    """
    allowed = _entity_tools(surface)
    invocations = [
        inv for inv in ok_invocations(corpus) if allowed is None or inv.tool in allowed
    ]
    consensus = _anchor_consensus(invocations, index)

    drafts: dict[str, EntityDraft] = {}
    unattributed: list[InvocationPoint] = []
    for inv in invocations:
        if not isinstance(inv.response, dict):
            unattributed.append(inv)
            continue
        anchor = choose_anchor(inv, index, consensus)
        if anchor is None:
            unattributed.append(inv)
            continue
        field_name, value = anchor
        draft = drafts.get(field_name)
        if draft is None:
            draft = EntityDraft(name="", anchor_field=field_name)
            drafts[field_name] = draft
        draft.observations.append(
            Observation(inv=inv, body=dict(inv.response), anchor_field=field_name, anchor_value=value)
        )

    for field_name, draft in drafts.items():
        draft.name = entity_name(field_name, {o.tool for o in draft.observations})
    return drafts, unattributed


def attach_reference_reads(
    drafts: dict[str, EntityDraft],
    unattributed: list[InvocationPoint],
    index: IdentifierIndex,
) -> list[InvocationPoint]:
    """Credit list-of-id responses as reads of the entity they list.

    **Rule.** A response field whose value is a list of identifier-shaped
    scalars, whose name is a plural of an anchor field ``F``
    (``order_ids`` -> ``order_id``), and **all** of whose elements are known
    values of ``F``, means the tool read that entity. It contributes no fields
    and no rows - a list of ids is not a row body - only ``read_by`` evidence.

    **Evidence required.** Full containment in the identifier's observed value
    set. Partial containment could just as easily be a different id space that
    happens to overlap.

    **When evidence is insufficient.** The call stays unattributed and its tool
    may end up in ``StateSchema.unresolved``.
    """
    by_field = {d.anchor_field: d for d in drafts.values() if d.anchor_field}
    still_unattributed: list[InvocationPoint] = []
    for inv in unattributed:
        matched = False
        for list_name, values in scalar_list_fields(inv.response).items():
            for singular in _singular_candidates(list_name):
                draft = by_field.get(singular)
                if draft is None:
                    continue
                if set(values) <= index.values(singular):
                    draft.reference_tools.add(inv.tool)
                    matched = True
        if not matched:
            still_unattributed.append(inv)
    return still_unattributed


def snapshot_drafts(
    unattributed: list[InvocationPoint],
) -> tuple[list[EntityDraft], list[str]]:
    """Turn identifier-less-but-stable responses into static snapshots.

    This is PLAN.md's "degrade honestly rather than invent structure". A tool
    like ``get_store_policy`` returns ``{"policy": "..."}``: no id, nothing to
    join on, no way to know what a row even is. We keep the observed bodies
    verbatim and mark the entity ``static_snapshot``.

    **Rule.** All of a tool's unattributed OK responses become one snapshot when:

    1. every response is a JSON object, and
    2. at least one field is not a bare boolean - a body of only booleans is an
       acknowledgement (``{"sent": true}``), not state, and
    3. the tool is argument-deterministic: identical arguments always produced
       identical bodies. Otherwise the response depends on something we cannot
       see, and pinning a snapshot would make the rebuilt environment lie.

    **Evidence required.** One observation. A single verbatim row is honest;
    the dishonest move would be generalizing it into a schema.

    **When evidence is insufficient.** The tool's name is returned in the second
    element, destined for ``StateSchema.unresolved``.
    """
    by_tool: dict[str, list[InvocationPoint]] = defaultdict(list)
    for inv in unattributed:
        by_tool[inv.tool].append(inv)

    snapshots: list[EntityDraft] = []
    unresolved: list[str] = []
    for tool, invs in sorted(by_tool.items()):
        bodies = [inv.response for inv in invs]
        if not all(isinstance(b, dict) and b for b in bodies):
            unresolved.append(tool)
            continue
        if all(isinstance(v, bool) for b in bodies for v in b.values()):
            unresolved.append(tool)
            continue
        deterministic = True
        seen: dict[str, str] = {}
        for inv in invs:
            akey = json.dumps(inv.arguments or {}, sort_keys=True, default=str)
            bkey = json.dumps(inv.response, sort_keys=True, default=str)
            if seen.setdefault(akey, bkey) != bkey:
                deterministic = False
                break
        if not deterministic:
            unresolved.append(tool)
            continue

        draft = EntityDraft(name=snapshot_name(tool), anchor_field=None, forced_static=True)
        draft.reference_tools.add(tool)
        rows: list[JsonObject] = []
        row_keys: set[str] = set()
        for b in bodies:
            k = json.dumps(b, sort_keys=True, default=str)
            if k not in row_keys:
                row_keys.add(k)
                rows.append(dict(b))
        draft.snapshot_rows = rows
        draft.snapshot_observations = len(invs)
        snapshots.append(draft)
    return snapshots, unresolved


def infer_drafts(
    corpus: TraceCorpus,
    surface: ToolSurface | None = None,
    index: IdentifierIndex | None = None,
) -> tuple[list[EntityDraft], list[str]]:
    """Full grouping pass: drafts plus the tools that could not be attributed."""
    index = index if index is not None else find_identifiers(corpus)
    drafts, unattributed = group_observations(corpus, index, surface)
    unattributed = attach_reference_reads(drafts, unattributed, index)
    snapshots, unresolved = snapshot_drafts(unattributed)

    all_drafts = list(drafts.values()) + snapshots
    attributed_tools = {t for d in all_drafts for t in d.tools}
    unresolved = sorted({t for t in unresolved if t not in attributed_tools})
    return all_drafts, unresolved


__all__ = [
    "EntityDraft",
    "MAX_SAMPLE_VALUES",
    "NOT_PLURAL_ENDINGS",
    "Observation",
    "TOOL_VERBS",
    "attach_reference_reads",
    "build_field_profiles",
    "choose_anchor",
    "echo_candidates",
    "entity_name",
    "group_observations",
    "infer_drafts",
    "infer_writers",
    "json_type",
    "pluralize",
    "singularize",
    "snapshot_drafts",
    "snapshot_name",
    "split_by_surface",
    "tool_noun",
]
