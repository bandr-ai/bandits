"""Group episodes into task families and select a defensible set from them.

The goal is not to find every task. It is to choose a small set that stands for a
large share of real production volume, and to say plainly how much it leaves out.

Two rules shape everything here. Grouping reads only ``at_start`` content, so a
family can never be defined by something a live request would not know. And the
fit/held-out split moves whole lineage groups, never individual traces, so a
retry of the same request cannot land on both sides of the boundary.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Literal

from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    FamilyCoherence,
    MissingSlot,
    SelectedTask,
    SlotKind,
    TaskFamily,
    TaskSet,
    kind_rank,
)
from bandits.store import DerivedEnvelope, DerivedStore

DEFAULT_BUDGET = 20
DEFAULT_HELD_OUT = 0.3
DEFAULT_NEIGHBORS = 3
"""Maximum neighbors per descriptor in the mutual-kNN grouping graph.

This value has not yet been calibrated for the embedding backend. Raising it
reduces fragmentation but widens the transitive components that :func:`_cluster`
builds, so it cannot be tuned on its own — the two effects need choosing together
against a labelled corpus."""

DEFAULT_DIAMETER_FACTOR = 1.25
"""Multiple of the link threshold above which a family is reported as over-merged.

Mutual-kNN bounds each *edge* but not the *span* of the component it builds, so a
family can end up wider than the threshold that admitted any of its links without
a single edge having broken the rule. A span at or under the threshold is always
explicable as one legal edge; past it, the family is only held together
transitively, and the further past, the less any single pair accounts for it.

Measured on thirteen support instructions embedded with qwen3-embedding-8b: the
families that collapsed eleven-of-thirteen at ``neighbors=5`` span 1.44x the
threshold, while every family that survived as a coherent task spans 1.00x or
less. This sits between them.

Advisory and provisional. What counts as implausibly wide depends on the same
``(similarity, neighbors)`` pair that has never been measured against a labelled
corpus (#2, #16), and thirteen hand-written instructions are not that corpus.
It attaches a limitation and changes no grouping, so a wrong value costs a
reviewer one glance rather than a corrupted family."""

DEFAULT_TAIL_RESERVE = 0.3

_TAIL_SLOTS: tuple[SlotKind, ...] = (
    SlotKind.KNOWN_FAILURE,
    SlotKind.RARE_TOOL,
    SlotKind.LONG_EPISODE,
    SlotKind.UNMEASURABLE,
)

# The reserve is filled from structural signals alone: an error status, a rare
# tool, an unusual length, absent terminal evidence. Categories that need to know
# what the work *meant* — a handoff, a refused request, a partial success — are
# invisible here, and no fixed list of them holds across domains. The selection
# says so as a limitation rather than naming categories from one domain and
# reporting them missing everywhere else.
_SEMANTIC_RESERVE_LIMIT = (
    "the tail reserve is filled from structural signals only; categories that "
    "depend on what the work meant need a domain extension to select for"
)

_MASKS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), " <email> "),
    (re.compile(r"\bhttps?://\S+"), " <url> "),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), " <uuid> "),
    (re.compile(r"\b[0-9a-f]{16,}\b"), " <hash> "),
    (re.compile(r"\b\w*\d[\w-]*\b"), " <id> "),
)
"""Value masks, applied in order. Identifiers are what vary between runs of the
same task, so masking them is what makes two runs recognizable as one family."""

_NON_TOKEN = re.compile(r"[^a-z0-9<>_]+")


def normalize_instruction(instruction: str) -> str:
    """Reduce an instruction to the shape it shares with others like it."""
    text = instruction.lower()
    for pattern, replacement in _MASKS:
        text = pattern.sub(replacement, text)
    return " ".join(_NON_TOKEN.sub(" ", text).split())


def fingerprint(instruction: str) -> str:
    normalized = normalize_instruction(instruction)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


Distance = Callable[[str, str], float]
"""How far apart two normalized instructions are, in [0, 1].

The callable receives the normalized descriptors used as embedding-cache keys.
"""


def _stable_fraction(*parts: str) -> float:
    """A deterministic value in [0, 1) for one key, so splits are reproducible."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


_MEASURABLE_FLOOR = EvidenceKind.HUMAN_LABEL
"""Below this, an outcome rests on an uncalibrated judge or the agent's own word."""

_FAILURE_CLAIMS = ("span_error",)


class _TraceFeatures:
    """Everything grouping and slot-filling need about one trace, read from the analysis."""

    __slots__ = (
        "trace_id",
        "lineage_id",
        "instruction",
        "normalized",
        "tools",
        "span_count",
        "has_failure",
        "measurable",
    )

    def __init__(
        self,
        *,
        trace_id: str,
        lineage_id: str | None,
        instruction: str,
        tools: frozenset[str],
        span_count: int,
        has_failure: bool,
        measurable: bool,
    ) -> None:
        self.trace_id = trace_id
        self.lineage_id = lineage_id
        self.instruction = instruction
        self.normalized = normalize_instruction(instruction)
        self.tools = tools
        self.span_count = span_count
        self.has_failure = has_failure
        self.measurable = measurable

    @property
    def lineage_group(self) -> str:
        """A trace with no declared lineage is its own group, never merged with another."""
        return self.lineage_id or f"trace:{self.trace_id}"


def _is_failure(evidence: Evidence) -> bool:
    if evidence.claim in _FAILURE_CLAIMS:
        return True
    if evidence.claim == "command_exit_code":
        return bool(evidence.value.get("value"))
    return False


def _features(analysis: CorpusAnalysis) -> tuple[list[_TraceFeatures], list[str]]:
    """Per-trace features, plus the ids of traces that cannot be grouped at all."""
    by_trace: dict[str, list[Evidence]] = {}
    for item in analysis.evidence:
        by_trace.setdefault(item.trace_id, []).append(item)

    features: list[_TraceFeatures] = []
    ungroupable: list[str] = []
    for task in analysis.tasks:
        if not task.instruction:
            # Nothing at_start to group on. Excluded from families but still
            # counted as production volume, so coverage reflects the gap.
            ungroupable.append(task.trace_id)
            continue

        evidence = by_trace.get(task.trace_id, [])
        outcome = [e for e in evidence if e.evidence_id in set(task.outcome_evidence_ids)]
        tools = next(
            (frozenset(e.value) for e in evidence if e.claim == "tools_called"), frozenset()
        )
        span_count = next((int(e.value) for e in evidence if e.claim == "episode_span_count"), 0)
        features.append(
            _TraceFeatures(
                trace_id=task.trace_id,
                lineage_id=task.lineage_id,
                instruction=task.instruction,
                tools=tools,
                span_count=span_count,
                has_failure=any(_is_failure(e) for e in outcome),
                measurable=any(kind_rank(e.kind) >= kind_rank(_MEASURABLE_FLOOR) for e in outcome),
            )
        )
    return features, ungroupable


def _cluster(
    features: list[_TraceFeatures],
    similarity: float,
    neighbors: int,
    distance: Distance,
) -> list[list[_TraceFeatures]]:
    """Collapse exact duplicates, then connect mutual nearest descriptors.

    The similarity threshold refuses weak edges; requiring both endpoints to
    select one another avoids most single-link chains. Sorting every tie makes
    the graph and its connected components independent of input order.
    """
    exact: dict[str, list[_TraceFeatures]] = {}
    for feature in features:
        exact.setdefault(feature.normalized, []).append(feature)

    descriptors = sorted(exact)
    max_distance = 1.0 - similarity
    nearest: dict[str, set[str]] = {}
    for descriptor in descriptors:
        ranked = sorted(
            ((distance(descriptor, other), other) for other in descriptors if other != descriptor),
            key=lambda item: (item[0], item[1]),
        )
        nearest[descriptor] = {
            other for score, other in ranked[:neighbors] if score <= max_distance
        }

    graph = {descriptor: set() for descriptor in descriptors}
    for descriptor in descriptors:
        for other in nearest[descriptor]:
            if descriptor in nearest[other]:
                graph[descriptor].add(other)
                graph[other].add(descriptor)

    components: list[list[str]] = []
    unseen = set(descriptors)
    while unseen:
        root = min(unseen)
        pending = [root]
        component: list[str] = []
        unseen.remove(root)
        while pending:
            current = pending.pop()
            component.append(current)
            for other in sorted(graph[current], reverse=True):
                if other in unseen:
                    unseen.remove(other)
                    pending.append(other)
        components.append(sorted(component))

    return [
        [feature for descriptor in group for feature in exact[descriptor]] for group in components
    ]


def _diameter(members: list[_TraceFeatures], distance: Distance) -> tuple[float, str, str]:
    """The widest distance inside one family, and the two descriptors that span it.

    Reported rather than acted on: naming the pair is what lets a reviewer decide
    whether the family is one task, and reach for ``split-family`` if it is not.
    """
    descriptors = sorted({member.normalized for member in members})
    widest = (0.0, descriptors[0], descriptors[0])
    for index, left in enumerate(descriptors):
        for right in descriptors[index + 1 :]:
            span = distance(left, right)
            if span > widest[0]:
                widest = (span, left, right)
    return widest


def _medoid(members: list[_TraceFeatures], distance: Distance) -> str:
    """Find the exact weighted medoid over distinct descriptors, not every trace."""
    by_descriptor: dict[str, list[_TraceFeatures]] = {}
    for member in members:
        by_descriptor.setdefault(member.normalized, []).append(member)

    representatives = {
        descriptor: min(group, key=lambda member: member.trace_id)
        for descriptor, group in by_descriptor.items()
    }
    winning_descriptor = min(
        by_descriptor,
        key=lambda candidate: (
            sum(
                distance(representatives[candidate].normalized, representatives[other].normalized)
                * len(group)
                for other, group in by_descriptor.items()
            ),
            representatives[candidate].trace_id,
        ),
    )
    return representatives[winning_descriptor].trace_id


def _split_by_lineage(
    members: list[_TraceFeatures], family_id: str, held_out: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition whole lineage groups, never individual traces."""
    groups: dict[str, list[str]] = {}
    for feature in members:
        groups.setdefault(feature.lineage_group, []).append(feature.trace_id)

    target = int(round(held_out * len(members)))
    ordered = sorted(groups, key=lambda key: (_stable_fraction(family_id, key), key))

    held: list[str] = []
    fit: list[str] = []
    for key in ordered:
        if len(held) + len(groups[key]) <= target:
            held.extend(groups[key])
        else:
            fit.extend(groups[key])
    return tuple(sorted(fit)), tuple(sorted(held))


def _slot_candidates(
    slot: SlotKind, pool: list[_TraceFeatures], all_features: list[_TraceFeatures]
) -> list[_TraceFeatures]:
    """Traces eligible for one reserved slot, most extreme first."""
    if slot is SlotKind.KNOWN_FAILURE:
        return sorted((f for f in pool if f.has_failure), key=lambda f: (-f.span_count, f.trace_id))

    if slot is SlotKind.RARE_TOOL:
        usage = Counter(tool for f in all_features for tool in f.tools)
        threshold = max(1, int(0.25 * len(all_features)))
        rarity = {f.trace_id: min((usage[tool] for tool in f.tools), default=0) for f in pool}
        return sorted(
            (f for f in pool if f.tools and rarity[f.trace_id] <= threshold),
            key=lambda f: (rarity[f.trace_id], f.trace_id),
        )

    if slot is SlotKind.LONG_EPISODE:
        # Strictly longer than the 90th percentile. A trace merely at the cutoff
        # is typical, and filling the slot with one would hide that the corpus
        # has no unusually long episode left to offer.
        cutoff = _percentile([f.span_count for f in all_features], 0.9)
        return sorted(
            (f for f in pool if f.span_count > cutoff),
            key=lambda f: (-f.span_count, f.trace_id),
        )

    if slot is SlotKind.UNMEASURABLE:
        return sorted((f for f in pool if not f.measurable), key=lambda f: f.trace_id)

    return []


def _farthest_first(
    pool: list[_TraceFeatures], chosen: list[_TraceFeatures], count: int, distance: Distance
) -> list[_TraceFeatures]:
    """Greedy k-center fill: each pick is the one least like everything already in."""
    picked: list[_TraceFeatures] = []
    remaining = list(pool)
    selected = list(chosen)

    for _ in range(count):
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda candidate: (
                min(
                    (distance(candidate.normalized, other.normalized) for other in selected),
                    default=1.0,
                ),
                candidate.trace_id,
            ),
        )
        picked.append(best)
        selected.append(best)
        remaining.remove(best)
    return picked


def mine_task_set(
    analysis: CorpusAnalysis,
    analysis_id: str,
    *,
    distance: Distance,
    similarity: float,
    budget: int = DEFAULT_BUDGET,
    held_out: float = DEFAULT_HELD_OUT,
    neighbors: int = DEFAULT_NEIGHBORS,
    tail_reserve: float = DEFAULT_TAIL_RESERVE,
    diameter_factor: float = DEFAULT_DIAMETER_FACTOR,
    proposed_by: Literal["rule", "model", "human"] = "rule",
) -> TaskSet:
    """Group an analysis into families and select a set that stands for the workload."""
    features, ungroupable = _features(analysis)
    total_mass = len(analysis.tasks)

    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    if not 0.0 <= similarity <= 1.0:
        # A similarity outside [0, 1] inverts the link threshold, which silently
        # admits every pair and reports every family as over-merged.
        raise ValueError(f"similarity must be between 0 and 1, got {similarity}")
    if diameter_factor <= 0:
        raise ValueError(f"diameter_factor must be positive, got {diameter_factor}")
    clusters = _cluster(features, similarity, neighbors, distance)
    families: list[TaskFamily] = []
    family_of: dict[str, str] = {}

    for members in sorted(clusters, key=lambda c: (-len(c), c[0].trace_id)):
        descriptor = Counter(f.normalized for f in members).most_common(1)[0][0]
        family_id = f"family-{fingerprint(descriptor)}"
        fit, held = _split_by_lineage(members, family_id, held_out)

        limitations: list[str] = []
        span, left, right = _diameter(members, distance)
        coherence = FamilyCoherence(
            diameter=span,
            link_threshold=1.0 - similarity,
            diameter_factor=diameter_factor,
            widest_pair=(left, right),
        )
        if coherence.over_merged:
            # Every edge cleared the threshold; the component did not. Verifiers
            # are drafted per family, so an over-merged one is read for terminal
            # evidence across unrelated tasks and keyed to whichever value was
            # most common. Fragmentation is recoverable with merge-families;
            # this is not visible at all unless it is said out loud.
            limitations.append(
                f"widest pair inside this family is {span:.2f} apart, over "
                f"{diameter_factor:g}x the {1.0 - similarity:.2f} that admitted any "
                f"single link: {left!r} vs {right!r}; mutual-kNN bounds each edge "
                "but not the span of the component, so check these are one task"
            )
        if not held and len(members) > 1:
            limitations.append(
                "every trace shares one lineage group; no held-out split is possible "
                "without splitting a retry chain"
            )
        if all(f.lineage_id is None for f in members):
            limitations.append(
                "no trace declares a lineage id; each is treated as independent, "
                "which the source does not actually prove"
            )

        families.append(
            TaskFamily(
                family_id=family_id,
                descriptor=descriptor,
                trace_ids=tuple(sorted(f.trace_id for f in members)),
                medoid_trace_id=_medoid(members, distance),
                workload_mass=len(members),
                fit_trace_ids=fit,
                held_out_trace_ids=held,
                proposed_by=proposed_by,
                coherence=coherence,
                limitations=tuple(limitations),
            )
        )
        for feature in members:
            family_of[feature.trace_id] = family_id

    by_trace = {f.trace_id: f for f in features}
    selected: list[SelectedTask] = []
    chosen: list[_TraceFeatures] = []
    missing: list[MissingSlot] = []

    # The tail reserve never takes the last slot: a selection with no medoid in
    # it represents nothing, however well it covers the edges.
    tail_budget = max(0, min(budget - 1, math.ceil(budget * tail_reserve)))
    medoid_budget = budget - tail_budget

    # Medoids first, largest workload first: this is what makes the set
    # representative rather than merely diverse.
    for family in sorted(families, key=lambda f: (-f.workload_mass, f.family_id))[:medoid_budget]:
        feature = by_trace[family.medoid_trace_id]
        selected.append(
            SelectedTask(
                trace_id=feature.trace_id, family_id=family.family_id, slot=SlotKind.MEDOID
            )
        )
        chosen.append(feature)

    # Then the tail, round-robin so no single category eats the reserve.
    queues = {slot: _slot_candidates(slot, features, features) for slot in _TAIL_SLOTS}
    taken = {f.trace_id for f in chosen}
    filled = 0
    while filled < tail_budget:
        progressed = False
        for slot in _TAIL_SLOTS:
            if filled >= tail_budget:
                break
            candidate = next((f for f in queues[slot] if f.trace_id not in taken), None)
            if candidate is None:
                continue
            selected.append(
                SelectedTask(
                    trace_id=candidate.trace_id,
                    family_id=family_of[candidate.trace_id],
                    slot=slot,
                )
            )
            chosen.append(candidate)
            taken.add(candidate.trace_id)
            filled += 1
            progressed = True
        if not progressed:
            break

    filled_slots = {s.slot for s in selected}
    for slot in _TAIL_SLOTS:
        if slot in filled_slots:
            continue
        missing.append(
            MissingSlot(
                slot=slot.value,
                reason=(
                    "no trace in the corpus matches this category"
                    if not queues[slot]
                    else "every matching trace was already selected for another slot"
                ),
            )
        )

    # Whatever budget is left goes to whatever is least like what is already in.
    remaining_pool = [f for f in features if f.trace_id not in taken]
    for feature in _farthest_first(remaining_pool, chosen, budget - len(selected), distance):
        selected.append(
            SelectedTask(
                trace_id=feature.trace_id,
                family_id=family_of[feature.trace_id],
                slot=SlotKind.FARTHEST_FIRST,
            )
        )
        taken.add(feature.trace_id)

    represented = {s.family_id for s in selected}
    covered_mass = sum(f.workload_mass for f in families if f.family_id in represented)

    limitations: list[str] = [*analysis.limitations, _SEMANTIC_RESERVE_LIMIT]
    if ungroupable:
        limitations.append(
            f"{len(ungroupable)} trace(s) declare no instruction and cannot be grouped; "
            "they count as production volume that no family represents"
        )

    return TaskSet(
        corpus_id=analysis.corpus_id,
        analysis_id=analysis_id,
        families=tuple(families),
        selected=tuple(selected),
        total_workload_mass=total_mass,
        workload_coverage=(covered_mass / total_mass) if total_mass else 0.0,
        missing_slots=tuple(missing),
        underfilled=len(selected) < budget,
        limitations=tuple(limitations),
    )


def merge_families(task_set: TaskSet, family_ids: tuple[str, ...]) -> TaskSet:
    """Combine families a reviewer says are one task. Recorded as a human correction."""
    known = task_set.family_by_id()
    unknown = [fid for fid in family_ids if fid not in known]
    if unknown:
        raise ValueError(f"unknown family id(s): {sorted(unknown)}")
    if len(set(family_ids)) < 2:
        raise ValueError("merging needs at least two distinct families")

    merged_from = [known[fid] for fid in family_ids]
    trace_ids = tuple(sorted({tid for f in merged_from for tid in f.trace_ids}))
    survivor = max(merged_from, key=lambda f: (f.workload_mass, f.family_id))

    merged = TaskFamily(
        family_id=survivor.family_id,
        descriptor=survivor.descriptor,
        trace_ids=trace_ids,
        medoid_trace_id=survivor.medoid_trace_id,
        workload_mass=len(trace_ids),
        # The split is rebuilt from the union rather than concatenated, so a
        # lineage group present in two merged families cannot end up on both sides.
        fit_trace_ids=tuple(sorted({t for f in merged_from for t in f.fit_trace_ids})),
        held_out_trace_ids=tuple(sorted({t for f in merged_from for t in f.held_out_trace_ids})),
        proposed_by="human",
        review_status="merged",
        # Deliberately dropped: the merged family is wider than either input, and
        # recomputing needs the distance function that grouped them, which a
        # correction does not have. A stale narrower figure would understate it.
        coherence=None,
        limitations=(
            *sorted({limit for f in merged_from for limit in f.limitations}),
            "coherence was not recomputed after this merge; the family is at least "
            "as wide as the widest it was built from",
        ),
    )
    remap = {fid: survivor.family_id for fid in family_ids}
    families = tuple(
        merged if f.family_id == survivor.family_id else f
        for f in task_set.families
        if f.family_id not in remap or f.family_id == survivor.family_id
    )
    return _rebuild(task_set, families, remap)


def split_family(task_set: TaskSet, family_id: str, analysis: CorpusAnalysis) -> TaskSet:
    """Split a family back into its exact-instruction groups.

    Deterministic on purpose: a reviewer rejecting a grouping needs a defensible
    smaller one, not a differently-guessed larger one.
    """
    known = task_set.family_by_id()
    if family_id not in known:
        raise ValueError(f"unknown family id: {family_id!r}")

    target = known[family_id]
    features, _ = _features(analysis)
    members = [f for f in features if f.trace_id in set(target.trace_ids)]
    groups: dict[str, list[_TraceFeatures]] = {}
    for feature in members:
        groups.setdefault(feature.normalized, []).append(feature)

    if len(groups) < 2:
        raise ValueError(f"family {family_id} holds one distinct instruction; nothing to split")

    held_out_ids = set(target.held_out_trace_ids)
    replacements = [
        TaskFamily(
            family_id=f"family-{fingerprint(descriptor)}",
            descriptor=descriptor,
            trace_ids=tuple(sorted(f.trace_id for f in group)),
            # Every member in this correction group has the same normalized
            # descriptor, so every candidate has identical medoid cost. The
            # stable trace-id tie-break is the complete answer and needs no
            # clustering backend that the old task-set artifact did not record.
            medoid_trace_id=min(group, key=lambda feature: feature.trace_id).trace_id,
            workload_mass=len(group),
            # The parent's split is carried through rather than redrawn, so a
            # trace cannot change sides as a side effect of the correction.
            fit_trace_ids=tuple(
                sorted(f.trace_id for f in group if f.trace_id not in held_out_ids)
            ),
            held_out_trace_ids=tuple(
                sorted(f.trace_id for f in group if f.trace_id in held_out_ids)
            ),
            proposed_by="human",
            review_status="split",
            # A subfamily is narrower than the family it came from, so the
            # parent's measurement would overstate it; there is no distance
            # function here to recompute one with.
            coherence=None,
            limitations=tuple(
                limit for limit in target.limitations if not limit.startswith("widest pair")
            ),
        )
        for descriptor, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]

    remap = {
        feature.trace_id: replacement.family_id
        for replacement in replacements
        for feature in members
        if feature.trace_id in set(replacement.trace_ids)
    }
    families = tuple(f for f in task_set.families if f.family_id != family_id) + tuple(replacements)
    return _rebuild(task_set, families, {}, trace_remap=remap)


def _rebuild(
    task_set: TaskSet,
    families: tuple[TaskFamily, ...],
    family_remap: dict[str, str],
    *,
    trace_remap: dict[str, str] | None = None,
) -> TaskSet:
    """Reissue a task set after a correction, keeping the selection intact.

    A correction changes how episodes are grouped. It must not silently change
    which episodes were selected, or a reviewer's edit would quietly reshape the
    eval set underneath them.
    """
    trace_remap = trace_remap or {}
    selected = tuple(
        s.model_copy(
            update={
                "family_id": trace_remap.get(s.trace_id, family_remap.get(s.family_id, s.family_id))
            }
        )
        for s in task_set.selected
    )
    represented = {s.family_id for s in selected}
    covered = sum(f.workload_mass for f in families if f.family_id in represented)
    # replace(), not model_copy(): a correction that broke a family invariant
    # would otherwise be written to the store and only fail on the next read.
    return task_set.replace(
        families=families,
        selected=selected,
        workload_coverage=(
            covered / task_set.total_workload_mass if task_set.total_workload_mass else 0.0
        ),
    )


def compute_task_set_id(task_set: TaskSet) -> str:
    digest = hashlib.sha256(task_set.model_dump_json().encode("utf-8")).hexdigest()
    return f"taskset-{digest[:16]}"


def save_task_set(task_set: TaskSet, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_task_set_id(task_set),
        kind="taskset",
        parent_artifact_id=task_set.analysis_id,
        payload=task_set.model_dump_json().encode("utf-8"),
        summary={
            "families": len(task_set.families),
            "selected": len(task_set.selected),
            "missing_slots": len(task_set.missing_slots),
        },
    )


def load_task_set(task_set_id: str, store: DerivedStore) -> TaskSet:
    return TaskSet.model_validate_json(store.read_payload(task_set_id))
