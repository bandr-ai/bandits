#!/usr/bin/env python3
"""Static coverage audit: how much of the corpus could this AWM simulate?

No model calls, no cost. For every recorded tool transition, ask what the
grounding tools would have been able to establish, and classify it:

    EXACT          every field the recorded answer asserts is already
                   available for this entity (compiled state or a prior
                   recorded observation) -- the AWM only has to report it
    TRANSFORM      the pre-state exists and the tool has behavioral
                   evidence, but the answer asserts values no source shows
                   (a refund amount, a new status). Identifiable in
                   principle; NOT executable today, since the claim auditor
                   has no transformation rule -- see `potential` below
    BEHAVIOR_ONLY  similar calls of this tool are recorded, but nothing
                   entity-specific. Only the shape is known
    UNSUPPORTED    neither entity facts nor tool behavior

Coverage counts EXACT only. TRANSFORM is reported separately as potential
coverage: counting it as coverage would overstate what the environment can
execute today by exactly the size of the bucket a transformation rule would
unlock, which is the point of measuring it.

tau2-specific by construction (entity kinds, id arguments, tool names), like
the smoke runner it sits beside.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_awm_fidelity import excluded_trace_ids_for, load_family

from bandits.emulate.agentic import (
    AWMRuntimeContext,
    _flatten_paths,
    _prior_observation_fields,
)
from bandits.emulate.models import GroundingTransition

DEFAULT_TAU_ROOT = Path("work/tau/run/proj/.bandits")
DEFAULT_CORPUS_ID = "corpus-ee3b33086ef177d7"
DEFAULT_TASK_SET_ID = "taskset-37654fb6a6bd578f"
DEFAULT_FAMILY_ID = "family-451ae91f975c"
DEFAULT_MARKERS = ()

EXACT = "EXACT"
TRANSFORM = "TRANSFORM"
BEHAVIOR_ONLY = "BEHAVIOR_ONLY"
UNSUPPORTED = "UNSUPPORTED"

_ENTITY_ARGS = {"user_id": "user", "reservation_id": "reservation"}


def _target_entities(transition: GroundingTransition) -> set[tuple[str, str]]:
    found = set()
    for call in transition.action_calls:
        for arg, kind in _ENTITY_ARGS.items():
            value = call.arguments.get(arg)
            if isinstance(value, str) and value:
                found.add((kind, value))
    return found


def _available_values(
    transition: GroundingTransition, entity: tuple[str, str]
) -> dict[str, Any]:
    """Everything read_world_state would return for this entity: compiled
    state plus prior recorded observations, exactly as the real tool merges
    them (so the audit cannot claim coverage the tool would not deliver)."""
    kind, entity_id = entity
    tool_prefix = {"user": "get_user_details", "reservation": "get_reservation_details"}.get(kind)
    found: dict[str, Any] = {}
    if tool_prefix:
        prefix = f"{tool_prefix}.{entity_id}."
        for field in transition.state_before.fields:
            if field.path.startswith(prefix):
                found[field.path[len(prefix) :]] = field.value
    context = AWMRuntimeContext(
        candidate_calls=transition.action_calls,
        current_state=transition.state_before,
        history_before=transition.history_before,
    )
    for path, prior in _prior_observation_fields(context, kind, entity_id).items():
        if path not in found and not prior.conflicting_values:
            found[path] = prior.value
    return found


def _behavioral_evidence(
    transition: GroundingTransition, fit_index: tuple[GroundingTransition, ...], tool: str
) -> bool:
    """Whether any OTHER trace records this tool -- the shape evidence
    search_transitions would surface. Excludes this transition's own lineage,
    mirroring the runtime's fit/held-out isolation: counting a transition's
    own trace as evidence for itself would inflate every row."""
    excluded = set(excluded_trace_ids_for(transition, fit_index))
    return any(
        other.trace_id not in excluded and any(c.tool == tool for c in other.action_calls)
        for other in fit_index
    )


def classify(
    transition: GroundingTransition, *, fit_index: tuple[GroundingTransition, ...]
) -> list[dict[str, Any]]:
    """Classify every call/result pair; never collapse a batch to its first call."""
    rows = []
    for call in transition.action_calls:
        recorded = next(
            (
                observation
                for observation in transition.observations
                if observation.role == "tool"
                and observation.tool_call_id == call.call_id
            ),
            None,
        )
        if recorded is None and len(transition.action_calls) == 1:
            candidates = [
                observation
                for observation in transition.observations
                if observation.role == "tool"
                and (
                    observation.tool_name == call.tool
                    or observation.tool_name is None
                )
            ]
            recorded = candidates[0] if len(candidates) == 1 else None
        rows.append(
            _classify_call(
                transition,
                call=call,
                recorded=recorded,
                fit_index=fit_index,
            )
        )
    return rows


def _classify_call(
    transition: GroundingTransition, *, call, recorded, fit_index
) -> dict[str, Any]:
    tool = call.tool
    asserted = _flatten_paths(recorded.content) if recorded and recorded.content is not None else {}

    one_call = transition.replace(action_calls=(call,))
    entities = _target_entities(one_call)
    available: dict[str, Any] = {}
    for entity in entities:
        available.update(_available_values(transition, entity))

    matched = {p: v for p, v in asserted.items() if p in available and available[p] == v}
    known_path_wrong_value = {
        p: v for p, v in asserted.items() if p in available and available[p] != v
    }
    missing = {p: v for p, v in asserted.items() if p not in available}
    has_behavior = _behavioral_evidence(transition, fit_index, tool)

    # The corpus records NO measured state deltas (inferred_state_delta is
    # empty for every transition; delta_ground_truth_status is UNAVAILABLE or
    # NOT_APPLICABLE throughout), so a write cannot be read off the ledger.
    # It is inferred instead from the recorded answer itself: an asserted
    # value that differs from this entity's known pre-state is a mutation.
    # Conservative -- a write whose pre-state is unknown looks like a read --
    # which understates writes rather than inventing them.
    is_write = bool(known_path_wrong_value)

    # Tools that take no entity id (calculate, transfer_to_human_agents,
    # search_*) are not entity lookups: no amount of state reconstruction
    # makes them EXACT, so they are reported apart rather than diluting the
    # entity-coverage number they cannot contribute to.
    is_entity_scoped = bool(entities)

    if not asserted:
        verdict = EXACT if available else (BEHAVIOR_ONLY if has_behavior else UNSUPPORTED)
    elif not missing and not known_path_wrong_value:
        verdict = EXACT
    elif not missing and known_path_wrong_value and has_behavior:
        # Every asserted path is known for this entity; some values differ.
        # That is a transformation of known pre-state, which is what a
        # mutation rule would license -- and what the auditor rejects today.
        verdict = TRANSFORM
    elif available and has_behavior:
        verdict = TRANSFORM if is_write else BEHAVIOR_ONLY
    elif has_behavior:
        verdict = BEHAVIOR_ONLY
    else:
        verdict = UNSUPPORTED

    return {
        "transition_id": transition.transition_id,
        "call_id": call.call_id,
        "tool": tool,
        "verdict": verdict,
        "is_write": is_write,
        "is_entity_scoped": is_entity_scoped,
        "is_batch": len(transition.action_calls) > 1,
        "is_error": bool(recorded.error) if recorded else False,
        "turn_index": transition.turn_index,
        "asserted_fields": len(asserted),
        "fields_available": len(matched),
        "fields_missing": len(missing),
        "fields_transformed": len(known_path_wrong_value),
        "entities": sorted(f"{k}:{i}" for k, i in entities),
    }


def _table(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[Any, dict[str, int]] = defaultdict(
        lambda: {EXACT: 0, TRANSFORM: 0, BEHAVIOR_ONLY: 0, UNSUPPORTED: 0, "total": 0}
    )
    for row in rows:
        bucket = buckets[row[key]]
        bucket[row["verdict"]] += 1
        bucket["total"] += 1
    return {str(k): v for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]["total"])}


def _render(title: str, table: dict[str, Any]) -> str:
    lines = [f"\n{title}", f"{'':32s} {'n':>5s} {'exact':>7s} {'xform':>7s} {'behav':>7s} {'unsup':>7s}"]
    for name, counts in table.items():
        total = counts["total"]
        lines.append(
            f"{name[:32]:32s} {total:5d} "
            f"{counts[EXACT]:7d} {counts[TRANSFORM]:7d} "
            f"{counts[BEHAVIOR_ONLY]:7d} {counts[UNSUPPORTED]:7d}"
        )
    return "\n".join(lines)


def run(
    *,
    tau_root: Path,
    corpus_id: str,
    task_set_id: str,
    family_id: str,
    markers: tuple[str, ...],
    output: Path | None,
) -> None:
    family, transitions, _schemas = load_family(
        tau_root, corpus_id=corpus_id, task_set_id=task_set_id, family_id=family_id, markers=markers
    )
    # Only transitions the tool world is ever asked to simulate: an action
    # with at least one call AND a recorded tool observation to check against.
    # Assistant/user turns are not tool transitions, and auditing all 371
    # rows would understate coverage on work the AWM never does.
    auditable = tuple(
        t
        for t in transitions
        if t.action_calls and any(o.role == "tool" for o in t.observations)
    )
    fit_ids = set(family["fit_trace_ids"])
    fit_index = tuple(t for t in transitions if t.trace_id in fit_ids)
    rows = [
        row
        for transition in auditable
        for row in classify(transition, fit_index=fit_index)
    ]

    totals = {EXACT: 0, TRANSFORM: 0, BEHAVIOR_ONLY: 0, UNSUPPORTED: 0}
    for row in rows:
        totals[row["verdict"]] += 1
    n = len(rows)

    # Entity-scoped rows are the ones state reconstruction can actually move;
    # the rest are reported so the headline number is not diluted by tools no
    # fixture could ever cover.
    entity_rows = [r for r in rows if r["is_entity_scoped"]]
    entity_totals = {EXACT: 0, TRANSFORM: 0, BEHAVIOR_ONLY: 0, UNSUPPORTED: 0}
    for row in entity_rows:
        entity_totals[row["verdict"]] += 1
    en = len(entity_rows)

    report = {
        "corpus_id": corpus_id,
        "family_id": family_id,
        "transitions_total": len(transitions),
        "transitions_auditable": n,
        "verdicts": totals,
        # Coverage today counts EXACT only. TRANSFORM is what a mutation rule
        # would unlock, reported separately rather than folded in.
        "coverage_today": (totals[EXACT] / n) if n else None,
        "coverage_potential_with_transforms": ((totals[EXACT] + totals[TRANSFORM]) / n) if n else None,
        "entity_scoped": {
            "transitions": en,
            "verdicts": entity_totals,
            "coverage_today": (entity_totals[EXACT] / en) if en else None,
            "coverage_potential_with_transforms": (
                (entity_totals[EXACT] + entity_totals[TRANSFORM]) / en if en else None
            ),
        },
        "by_tool": _table(rows, "tool"),
        "by_entity_scoped": _table(rows, "is_entity_scoped"),
        "by_write": _table(rows, "is_write"),
        "by_batch": _table(rows, "is_batch"),
        "by_error": _table(rows, "is_error"),
        "rows": rows,
    }

    print(f"auditable tool transitions: {n} (of {len(transitions)} total)")
    print(f"coverage today (EXACT):              {report['coverage_today']:.1%}")
    print(f"coverage potential (EXACT+TRANSFORM): {report['coverage_potential_with_transforms']:.1%}")
    if en:
        print(
            f"entity-scoped only ({en}):           "
            f"exact {entity_totals[EXACT] / en:.1%}, "
            f"potential {(entity_totals[EXACT] + entity_totals[TRANSFORM]) / en:.1%}"
        )
    print(_render("by tool", report["by_tool"]))
    print(_render("by entity-scoped (True = takes an entity id)", report["by_entity_scoped"]))
    print(_render("by write (True = mutation)", report["by_write"]))
    print(_render("by batch (True = multi-call)", report["by_batch"]))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
        print(f"\nWrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, default=DEFAULT_TAU_ROOT)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--task-set-id", default=DEFAULT_TASK_SET_ID)
    parser.add_argument("--family-id", default=DEFAULT_FAMILY_ID)
    parser.add_argument("--marker", action="append", dest="markers", default=None)
    parser.add_argument("--output", type=Path, default=Path("work/awm-agentic/coverage-audit.json"))
    args = parser.parse_args()
    run(
        tau_root=args.tau_root,
        corpus_id=args.corpus_id,
        task_set_id=args.task_set_id,
        family_id=args.family_id,
        markers=tuple(args.markers) if args.markers else DEFAULT_MARKERS,
        output=args.output,
    )


if __name__ == "__main__":
    main()
