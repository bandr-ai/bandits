"""Infer foreign keys between the discovered entities.

PLAN.md Step 7: "Foreign keys, wherever an ID shows up as a field in one place
and a primary key in another." The whole inference is value-based - we join the
observed value sets and ask whether one is drawn from another - with a name
match used as corroborating evidence, never on its own.

Two rules, in decreasing strength, both of which refuse to fire on a single
observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bandits.contracts import ForeignKey

from .entities import EntityDraft
from .identifiers import Scalar, is_identifier_scalar

#: A foreign key is a claim about a *relationship*, and one observation cannot
#: distinguish a relationship from a coincidence. Two independent observations
#: is the floor for the name-corroborated rule.
MIN_FK_OBSERVATIONS = 2

#: The value-only rule has no name to corroborate it, so it needs more.
MIN_FK_OBSERVATIONS_VALUE_ONLY = 3
MIN_FK_DISTINCT_VALUE_ONLY = 2

#: Ceiling on the confidence the weaker, value-only rule may claim.
VALUE_ONLY_CONFIDENCE_CEILING = 0.7


@dataclass(frozen=True)
class FKEvidence:
    """Why a foreign key was (or was not) emitted. Useful for review and tests."""

    source_entity: str
    field: str
    target_entity: str
    target_field: str
    observations: int
    distinct_values: int
    matched_distinct: int
    rule: str

    @property
    def coverage(self) -> float:
        return self.matched_distinct / self.distinct_values if self.distinct_values else 0.0


def _distinct(values: list[Scalar]) -> set[Scalar]:
    return {v for v in values if is_identifier_scalar(v)}


def _confidence(observations: int, coverage: float, ceiling: float, floor: float) -> float:
    """Confidence from the amount and quality of evidence.

    ``evidence = min(1, observations / 5)`` - five independent observations is
    where we stop paying for more - multiplied by the fraction of the field's
    distinct values that were actually found in the target's key. The result is
    mapped into ``[floor, ceiling]``, so a weak rule can never out-rank a strong
    one no matter how much data it sees.
    """
    evidence = min(1.0, observations / 5.0)
    return round(floor + (ceiling - floor) * coverage * evidence, 3)


def infer_foreign_keys(
    drafts: list[EntityDraft],
) -> tuple[dict[str, tuple[ForeignKey, ...]], list[FKEvidence]]:
    """Infer foreign keys across all entity drafts.

    **Rule FK-1 (name-corroborated).** Field ``f`` of entity ``A`` references
    ``B`` when ``f`` has the same name as ``B``'s primary key, ``f`` is not
    ``A``'s own primary key, and at least one of ``f``'s observed values was
    actually seen as a key of ``B``. Requires
    :data:`MIN_FK_OBSERVATIONS` observations of ``f``.

    Full containment is deliberately *not* required here. ``orders.sku`` takes
    four values in the golden corpus while ``get_product`` was only ever called
    for two of them: the unseen skus are rows we never observed, not evidence
    against the relationship. Partial coverage is recorded honestly by lowering
    ``confidence`` instead of dropping the key.

    **Rule FK-2 (value-only).** With no name match, we require much more: every
    distinct value of ``f`` must be a known key of ``B``, over at least
    :data:`MIN_FK_DISTINCT_VALUE_ONLY` distinct values and
    :data:`MIN_FK_OBSERVATIONS_VALUE_ONLY` observations, and the resulting
    confidence is capped at :data:`VALUE_ONLY_CONFIDENCE_CEILING`. Small integer
    columns land inside each other's ranges by accident constantly; this rule
    exists to catch a renamed key (``buyer`` -> ``customers.customer_id``), not
    to trawl.

    **When evidence is insufficient.** No ``ForeignKey`` is emitted. A missing
    foreign key costs a downstream join; an invented one corrupts the rebuilt
    database and every verifier written against it. The returned
    :class:`FKEvidence` list records what was considered either way.

    Static snapshots (no primary key) are never a target: we do not claim to
    know what their rows are keyed by. Self-references are skipped - the
    corpus cannot distinguish a genuine self-reference from a row echoing its
    own key.
    """
    keyed = [d for d in drafts if d.anchor_field and not d.forced_static]
    key_values: dict[str, set[Scalar]] = {d.name: _distinct(list(d.key_values())) for d in keyed}
    key_field: dict[str, str] = {d.name: d.anchor_field for d in keyed if d.anchor_field}

    out: dict[str, tuple[ForeignKey, ...]] = {}
    evidence: list[FKEvidence] = []

    for draft in keyed:
        found: list[ForeignKey] = []
        values_by_field = draft.field_values()
        for fname, raw_values in sorted(values_by_field.items()):
            if fname == draft.anchor_field:
                continue
            values = [v for v in raw_values if is_identifier_scalar(v)]
            if not values:
                continue
            distinct = _distinct(values)
            for target in sorted(key_values):
                if target == draft.name:
                    continue
                target_key = key_field[target]
                matched = distinct & key_values[target]
                name_match = fname == target_key
                ev = FKEvidence(
                    source_entity=draft.name,
                    field=fname,
                    target_entity=target,
                    target_field=target_key,
                    observations=len(values),
                    distinct_values=len(distinct),
                    matched_distinct=len(matched),
                    rule="FK-1" if name_match else "FK-2",
                )
                if name_match:
                    ok = len(values) >= MIN_FK_OBSERVATIONS and bool(matched)
                    conf = _confidence(len(values), ev.coverage, ceiling=1.0, floor=0.5)
                else:
                    ok = (
                        len(values) >= MIN_FK_OBSERVATIONS_VALUE_ONLY
                        and len(distinct) >= MIN_FK_DISTINCT_VALUE_ONLY
                        and matched == distinct
                    )
                    conf = _confidence(
                        len(values), ev.coverage, ceiling=VALUE_ONLY_CONFIDENCE_CEILING, floor=0.3
                    )
                evidence.append(ev)
                if ok:
                    found.append(
                        ForeignKey(
                            field=fname,
                            references_entity=target,
                            references_field=target_key,
                            confidence=conf,
                        )
                    )
        if found:
            out[draft.name] = tuple(
                sorted(found, key=lambda fk: (fk.field, fk.references_entity))
            )
    return out, evidence


def cross_referenced(
    entity: str, foreign_keys: dict[str, tuple[ForeignKey, ...]]
) -> bool:
    """True when the entity takes part in any relationship, either direction.

    An entity that is only ever read and joins to nothing is exactly the case
    PLAN.md says we cannot model; :func:`bandits.state.infer_schema` uses this
    to decide whether to degrade it to a static snapshot.
    """
    if foreign_keys.get(entity):
        return True
    return any(fk.references_entity == entity for fks in foreign_keys.values() for fk in fks)


__all__ = [
    "FKEvidence",
    "MIN_FK_DISTINCT_VALUE_ONLY",
    "MIN_FK_OBSERVATIONS",
    "MIN_FK_OBSERVATIONS_VALUE_ONLY",
    "VALUE_ONLY_CONFIDENCE_CEILING",
    "cross_referenced",
    "infer_foreign_keys",
]
