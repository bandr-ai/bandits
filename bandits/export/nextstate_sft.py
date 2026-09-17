"""Turn a next-state verifier's scored traces into labeled SFT rows.

``score-traces`` already decides, per trace, whether it passes and which turns
were flagged and why (``verify/propose.py``). This module's only job is to
carry that decision into a row a training job can read: one full trajectory
per row, labeled ``positive`` or ``negative``, with the flagged turns kept as
metadata rather than used to cut the trajectory apart. It does not re-derive
or second-guess the label — a family-specific disagreement with the judge is
what ``review-checks`` is for, not this.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from bandits.export.models import RejectedTrace, TrainingMessage
from bandits.export.sft import build_transcript, generating_policy
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract, Trace
from bandits.verify.propose import FamilyVerifier, VerifierScores


class NextStateSFTExample(Contract):
    example_id: str
    label: Literal["positive", "negative"]
    messages: tuple[TrainingMessage, ...]
    tools: tuple[dict[str, Any], ...] | None = None
    generating_policy: dict[str, Any]

    corpus_id: str
    family_id: str
    archetype: str
    verifier_id: str
    scores_id: str
    judge_run_id: str
    checks_applied: tuple[str, ...]
    all_checks_reviewed: bool
    """False when any applied check was not a human-accepted ``FamilyCheck``
    (e.g. this export drew from ``--survivors``). Carried onto the row so a
    dataset built before real review can't be mistaken for one that was.
    """

    trace_id: str
    turns: int
    observed: int
    score: float | None
    flagged_turns: tuple[int, ...]
    flagged_by: dict[str, tuple[str, ...]]
    """Turn index (as a string, for JSON) -> check names (and/or ``judge``)
    that flagged it. Empty for a positive row."""

    def jsonl_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["messages"] = [message.as_chat_message() for message in self.messages]
        return row


class NextStateSFTBundle(Contract):
    schema_version: int = 1
    verifier_id: str
    scores_id: str
    all_checks_reviewed: bool
    positive: int
    negative: int
    rows: tuple[NextStateSFTExample, ...]
    unresolved: tuple[RejectedTrace, ...] = ()

    @model_validator(mode="after")
    def counts_match_rows(self) -> NextStateSFTBundle:
        if self.positive + self.negative != len(self.rows):
            raise ValueError("positive/negative counts do not match the rows")
        return self


def _all_checks_reviewed(verifier: FamilyVerifier, checks_applied: Sequence[str]) -> bool:
    accepted_names = {c.name for c in verifier.checks if c.decision == "accepted"}
    return all(name in accepted_names for name in checks_applied)


def compute_example_id(scores_id: str, trace_id: str) -> str:
    digest = hashlib.sha256(f"{scores_id}:{trace_id}".encode()).hexdigest()
    return f"nextstate-sft-{digest[:16]}"


def build_nextstate_sft_export(
    traces: Sequence[Trace],
    verifier: FamilyVerifier,
    verifier_id: str,
    scores: VerifierScores,
    scores_id: str,
) -> NextStateSFTBundle:
    by_id = {trace.trace_id: trace for trace in traces}
    reviewed = _all_checks_reviewed(verifier, scores.checks_applied)
    rows: list[NextStateSFTExample] = []
    unresolved: list[RejectedTrace] = []
    for trace_score in scores.scores:
        trace = by_id.get(trace_score.trace_id)
        if trace is None:
            unresolved.append(
                RejectedTrace(
                    trace_id=trace_score.trace_id,
                    family_id=verifier.family_id,
                    reasons=("trace not found in the verifier's corpus",),
                )
            )
            continue
        if trace_score.observed == 0:
            unresolved.append(
                RejectedTrace(
                    trace_id=trace_score.trace_id,
                    family_id=verifier.family_id,
                    reasons=("no observed turns; the verifier scored nothing",),
                )
            )
            continue
        messages, defects, _warnings = build_transcript(trace)
        if defects:
            unresolved.append(
                RejectedTrace(
                    trace_id=trace_score.trace_id,
                    family_id=verifier.family_id,
                    reasons=defects,
                )
            )
            continue
        rows.append(
            NextStateSFTExample(
                example_id=compute_example_id(scores_id, trace_score.trace_id),
                label="positive" if trace_score.passes else "negative",
                messages=messages,
                tools=(
                    tuple(tool.model_dump(mode="json") for tool in trace.tools_available)
                    if trace.tools_available is not None
                    else None
                ),
                generating_policy=generating_policy(trace),
                corpus_id=verifier.corpus_id,
                family_id=verifier.family_id,
                archetype=verifier.archetype.value,
                verifier_id=verifier_id,
                scores_id=scores_id,
                judge_run_id=scores.judge_run_id,
                checks_applied=scores.checks_applied,
                all_checks_reviewed=reviewed,
                trace_id=trace_score.trace_id,
                turns=trace_score.turns,
                observed=trace_score.observed,
                score=trace_score.score,
                flagged_turns=tuple(f.index for f in trace_score.flagged),
                flagged_by={str(f.index): f.by for f in trace_score.flagged},
            )
        )
    positive = sum(1 for row in rows if row.label == "positive")
    return NextStateSFTBundle(
        verifier_id=verifier_id,
        scores_id=scores_id,
        all_checks_reviewed=reviewed,
        positive=positive,
        negative=len(rows) - positive,
        rows=tuple(rows),
        unresolved=tuple(unresolved),
    )


def compute_bundle_id(bundle: NextStateSFTBundle) -> str:
    digest = hashlib.sha256(bundle.model_dump_json().encode()).hexdigest()
    return f"nextstate-sft-export-{digest[:16]}"


def save_nextstate_sft(bundle: NextStateSFTBundle, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_bundle_id(bundle),
        kind="nextstate_sft_export",
        parent_artifact_id=bundle.scores_id,
        payload=bundle.model_dump_json().encode(),
        summary={
            "positive": bundle.positive,
            "negative": bundle.negative,
            "unresolved": len(bundle.unresolved),
        },
    )


def load_nextstate_sft(export_id: str, store: DerivedStore) -> NextStateSFTBundle:
    return NextStateSFTBundle.model_validate_json(store.read_payload(export_id))


def _atomic_jsonl(path: Path, values: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    lines = (
        json.dumps(value.jsonl_row(), sort_keys=True, separators=(",", ":")) for value in values
    )
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    os.replace(temporary, path)


def write_nextstate_sft(bundle: NextStateSFTBundle, output: Path) -> tuple[Path, Path]:
    """Write labeled rows plus a sibling quarantine file. Both written even if empty."""
    unresolved = output.with_name(f"{output.stem}.unresolved.jsonl")
    _atomic_jsonl(output, bundle.rows)
    _atomic_jsonl(unresolved, bundle.unresolved)
    return output, unresolved
