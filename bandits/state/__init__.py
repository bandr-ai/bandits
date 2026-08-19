"""Stage 3 - state schema inference.

Reconstructs the database behind a set of tools from nothing but recorded
``(tool, arguments, response)`` triples, using the fact that **IDs repeat**
(PLAN.md Step 7).

    from bandits.state import infer_schema
    schema = infer_schema(corpus)               # corpus only
    schema = infer_schema(corpus, surface)      # better: use stage 2's classes

Nothing here touches the network, calls a model, or probes a tool. Every claim
in the returned :class:`~bandits.contracts.StateSchema` is backed by an
observation in the corpus, and the things the corpus cannot settle are marked
rather than guessed: an entity we cannot model becomes a ``static_snapshot``,
and a tool we cannot attribute lands in ``unresolved``.

Module map:

* :mod:`bandits.state.identifiers` - which fields carry row handles.
* :mod:`bandits.state.entities`    - grouping, naming, fields, reads/writes.
* :mod:`bandits.state.relations`   - foreign keys.
* :mod:`bandits.state.writes`      - what each write tool actually writes.
"""

from __future__ import annotations

from bandits.contracts import EntitySchema, StateSchema, ToolSurface, TraceCorpus

from .entities import (
    EntityDraft,
    build_field_profiles,
    infer_drafts,
    infer_writers,
    split_by_surface,
)
from .identifiers import find_identifiers
from .relations import cross_referenced, infer_foreign_keys
from .writes import infer_write_effects


def _is_static(draft: EntityDraft, writers: set[str], is_cross_referenced: bool) -> bool:
    """Decide whether an entity must degrade to a verbatim snapshot.

    **Rule.** Static when either

    1. the responses carry no identifier at all, so there is no row handle and
       no way to know what a row even is (``get_store_policy`` ->
       ``{"policy": ...}``), or
    2. the entity is only ever read *and* never cross-referenced: nothing
       writes it and no foreign key touches it in either direction. Its rows
       are then unconstrained by anything else we reconstructed, and any schema
       we published for it would be a guess dressed up as structure.

    **Evidence required.** To *avoid* being static, an entity needs either an
    observed write or a foreign key - both of which are themselves
    evidence-gated (see :mod:`bandits.state.relations`).

    **When evidence is insufficient.** Static wins. Materializing the observed
    rows verbatim is always safe; publishing a table that nothing constrains is
    how a rebuilt environment starts inventing answers.
    """
    if draft.forced_static or draft.anchor_field is None:
        return True
    return not writers and not is_cross_referenced


def infer_schema(corpus: TraceCorpus, surface: ToolSurface | None = None) -> StateSchema:
    """Infer the :class:`~bandits.contracts.StateSchema` behind a corpus.

    ``surface`` is optional. When given, stage 2's ``ToolClass`` decides which
    tools write and which read - that is real classification evidence, and it
    also keeps ``EXTERNAL`` tools (``send_email``) from ever contributing rows.
    When absent, writers are inferred from observed state changes inside a
    trace (:func:`bandits.state.entities.infer_writers`), which can only see a
    write whose effect is later observed.

    Each entity also carries a :class:`~bandits.contracts.WriteEffect` for
    every tool in ``written_by``, recording *what* that tool was observed to
    write - the key argument, the columns it always sets to a constant, the
    arguments whose values land in columns, and the columns it echoes back. See
    :mod:`bandits.state.writes`. Downstream stages must use this instead of
    reading write semantics out of the tool's English name.

    Entities are returned sorted with the modelled tables first, then the static
    snapshots, alphabetically within each group. ``unresolved`` holds tools
    whose successful responses could not be attributed to any entity - without a
    surface, that is ``send_email`` in the golden corpus, whose
    ``{"sent": true}`` is an acknowledgement rather than state. With a surface
    it is empty, because ``send_email`` is already classified ``EXTERNAL``: it
    is accounted for by the effect ledger, not unexplained.
    """
    index = find_identifiers(corpus)
    drafts, unresolved = infer_drafts(corpus, surface=surface, index=index)
    foreign_keys, _evidence = infer_foreign_keys(drafts)

    primary_keys = {d.anchor_field for d in drafts if d.anchor_field}

    entities: list[EntitySchema] = []
    for draft in drafts:
        if surface is not None:
            writers, readers = split_by_surface(draft.tools, surface)
        else:
            writers = infer_writers(draft)
            readers = draft.tools - writers

        static = _is_static(draft, writers, cross_referenced(draft.name, foreign_keys))
        bodies = (
            list(draft.snapshot_rows)
            if draft.anchor_field is None
            else [o.body for o in draft.observations]
        )
        fields = build_field_profiles(bodies, primary_keys)
        entities.append(
            EntitySchema(
                name=draft.name,
                primary_key=draft.anchor_field,
                fields=fields,
                foreign_keys=foreign_keys.get(draft.name, ()),
                written_by=tuple(sorted(writers)),
                read_by=tuple(sorted(readers)),
                write_effects=infer_write_effects(draft, writers, {f.name for f in fields}),
                evidence_count=draft.evidence_count,
                static_snapshot=static,
            )
        )

    entities.sort(key=lambda e: (e.static_snapshot, e.name))
    return StateSchema(entities=tuple(entities), unresolved=tuple(sorted(unresolved)))


__all__ = ["infer_schema", "infer_write_effects"]
