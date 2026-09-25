"""What the verifier actually cost per decision, read from the Bandits ledger.

The ledger records every judge call: which trace and turn it was for, the
tokens it used and how long it took. It records no prices -- those are the
provider's, change over time, and are given here explicitly by whoever runs
this, never looked up or guessed. The result is saved as its own artifact so
the report can show "N× cheaper" and still be rebuilt from saved artifacts
alone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from bandits.store import Contract, DerivedEnvelope, DerivedStore
from bandits_jev.dataset import DecisionDataset


class DecisionCost(Contract):
    calls: int
    """Model calls made for this decision: one per vote, plus any that
    failed after the transport gave up retrying."""
    failed_calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    """Sum of the calls' own durations. Votes may run in parallel, so this is
    compute time, not necessarily what a caller waited; waits between
    rate-limit retries are not included (see ``VerifierCost.retries``)."""
    usd: float


class VerifierCost(Contract):
    dataset_id: str
    ledger: str
    """The ledger file this was read from, as given."""
    models: tuple[str, ...]
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    per_decision: dict[str, DecisionCost]
    """Keyed by decision id; only decisions of ``dataset_id`` the ledger
    covers."""
    uncovered_decisions: int
    """Dataset decisions the ledger has no judge call for."""
    retries: int
    """Retry events (e.g. HTTP 429) recorded for the covered decisions."""
    over_counted_decisions: tuple[str, ...] = ()
    """Decisions with more successful calls than the judge asked votes for.
    Usually the ledger (or several chained ledgers) holds more than one
    judge run over the same steps, so their cost is summed into one
    decision and overstated; occasionally a re-asked unparsable reply. A
    warning, not a refusal -- see ``verifier_cost_from_ledger``."""


def verifier_cost_from_ledger(
    lines: Iterable[str],
    dataset: DecisionDataset,
    dataset_id: str,
    *,
    ledger: str,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> VerifierCost:
    """Attribute every ``judge_turn`` model call in the ledger to the
    dataset decision for the same (trace, turn). Calls for turns the
    dataset does not contain are ignored; dataset decisions with no call
    are counted, never priced at zero. Decisions with more successful calls
    than the judge asked votes for are listed in ``over_counted_decisions``
    (two judge runs over the same steps in one ledger would double their
    cost). Refuses a ledger whose judge model
    differs from the one that labeled the dataset -- that would price a
    different verifier."""
    if input_usd_per_mtok < 0 or output_usd_per_mtok < 0:
        raise ValueError("prices must be non-negative")
    by_turn = {
        (e.lineage.trace_id, e.lineage.turn_index): e.decision_id
        for e in dataset.examples
        if e.lineage.trace_id is not None and e.lineage.turn_index is not None
    }
    votes_asked = {e.decision_id: e.judge.votes_requested for e in dataset.examples if e.judge is not None}
    if not by_turn:
        raise ValueError(f"{dataset_id} has no judge-labeled rows (no trace/turn lineage) to price")
    labeling_models = {e.judge.model for e in dataset.examples if e.judge is not None}

    totals: dict[str, dict[str, float]] = {}
    retries = 0
    models: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ledger}:{line_number}: not valid JSON ({exc})") from exc
        if row.get("stage") != "judge_turn":
            continue
        decision_id = by_turn.get((row.get("trace_id"), row.get("turn_index")))
        if decision_id is None:
            continue
        if row.get("event_type") == "retry":
            retries += 1
            continue
        if row.get("event_type") != "model_call":
            continue
        models.add(str(row.get("model")))
        usage = row.get("usage") or {}
        t = totals.setdefault(
            decision_id, {"calls": 0, "failed": 0, "succeeded": 0, "prompt": 0, "completion": 0, "seconds": 0.0}
        )
        t["calls"] += 1
        succeeded = row.get("status") == "success"
        t["failed"] += 0 if succeeded else 1
        t["succeeded"] += 1 if succeeded else 0
        t["prompt"] += int(usage.get("prompt_tokens") or 0)
        t["completion"] += int(usage.get("completion_tokens") or 0)
        t["seconds"] += float(row.get("duration_seconds") or 0.0)

    if labeling_models and models and not models <= labeling_models:
        raise ValueError(
            f"ledger calls used {sorted(models)}, but the dataset was labeled by {sorted(labeling_models)}"
        )
    per_decision = {
        decision_id: DecisionCost(
            calls=int(t["calls"]),
            failed_calls=int(t["failed"]),
            prompt_tokens=int(t["prompt"]),
            completion_tokens=int(t["completion"]),
            seconds=t["seconds"],
            usd=(t["prompt"] * input_usd_per_mtok + t["completion"] * output_usd_per_mtok) / 1e6,
        )
        for decision_id, t in sorted(totals.items())
    }
    return VerifierCost(
        dataset_id=dataset_id,
        ledger=ledger,
        models=tuple(sorted(models)),
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        per_decision=per_decision,
        uncovered_decisions=len(set(by_turn.values()) - set(per_decision)),
        retries=retries,
        over_counted_decisions=tuple(
            decision_id
            for decision_id, t in sorted(totals.items())
            if decision_id in votes_asked and t["succeeded"] > votes_asked[decision_id]
        ),
    )


def compute_verifier_cost_id(cost: VerifierCost) -> str:
    return f"decision-verifier-cost-{hashlib.sha256(cost.model_dump_json().encode()).hexdigest()[:16]}"


def save_verifier_cost(cost: VerifierCost, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_verifier_cost_id(cost),
        kind="decision_verifier_cost",
        parent_artifact_id=cost.dataset_id,
        payload=cost.model_dump_json().encode(),
        summary={"decisions": len(cost.per_decision), "uncovered": cost.uncovered_decisions},
    )


def load_verifier_cost(cost_id: str, store: DerivedStore) -> VerifierCost:
    return VerifierCost.model_validate_json(store.read_payload(cost_id))
