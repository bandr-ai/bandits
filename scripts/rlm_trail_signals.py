#!/usr/bin/env python3
"""Run the fit/held-out RLM signal-discovery harness on TRAIL, not tau2.

TRAIL (Patronus AI, arXiv:2505.08638): 148 real GAIA + SWE-Bench agent traces,
each independently reviewed by a human QA annotator who names every concrete
error (category, the span it happened in, its impact) and gives a continuous
1-5 overall reliability score. There is no hidden database check behind that
score and no repeated trials of the same task the way tau2 or AppWorld have,
so there is nothing here for the RLM family-miner to mine: each task is
essentially one-off. What TRAIL gives instead is a *domain* where earlier work
in this repo found failures are largely visible in the trace itself
(hallucinated tool results, skipped planned steps, ignored instructions) rather
than facts that live outside it — which is exactly the condition under which a
trace-only signal has a real chance, unlike tau2's wrong-refund/wrong-recipient
failures.

Because there is no repeated-task structure, this script does not call the
family miner. It builds one synthetic "family" per benchmark split (GAIA,
SWE-Bench), each a stratified random 70/30 fit/held-out partition of the
traces that fall in the top or bottom score tertile (the same binarization
`docs/sft-signal-experiment.md` already used for the holistic-judge baseline,
so results here are comparable to it). That partition is declared honestly on
the family's `limitations`, not silently presented as a mined family.

The first version of this script called `evaluate_family` from
`trail_rlm_verifier.py` unmodified. That single-shot loop only ever repaired a
proposal that failed to *compile* — never one that compiled fine and scored a
coin flip. Every signal it produced was a literal keyword match lifted
straight from the few shown examples ("contains 'I apologize'", "contains
'let me try'"), and every one scored ~0.50 AUC: a check that only recognizes
the exact wording of the trajectories it was shown cannot be expected to
recognize the same underlying failure stated differently, and TRAIL's own
held-out design (a live benchmark's evaluation gates a submission on
differently-worded probes than the ones it can rehearse against) is the
concrete precedent for why that distinction matters and how to enforce it.

This version fixes both. `evaluate_family_recursive` below actually recurses:
each round is told exactly which fit traces its last proposal got wrong, by
label and by what the trace actually contains, and is asked to name a more
general property — not to patch the same keyword. And every proposal is
grounds-checked before it is even executed: any string literal in the
generated code that was copied verbatim (case-insensitively, 12+ characters)
from the shown few-shot text is rejected outright, with the offending literal
named in the correction, so the model cannot "solve" generalization by simply
being told to and quietly keeping the substring anyway.

Usage:
    git clone --depth 1 https://github.com/patronus-ai/trail-benchmark /tmp/trail-benchmark
    uv run python scripts/rlm_trail_signals.py \
        --trail-dir /tmp/trail-benchmark/benchmarking --out work/trail-rlm/report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from signal_experiment import score_signal  # noqa: E402
from signal_synth import _compile_signal, _run_over_corpus  # noqa: E402
from trail_rlm_verifier import (  # noqa: E402
    ProposedSignal,
    _balanced_example_ids,
    _baseline_values,
    _binary_labels,
    _examples,
    build_predictor,
)
from trail_signal import TrailTrace, load_trail  # noqa: E402

from bandits.analyze.models import TaskFamily

_MIN_LITERAL_LEN = 12
"""Below this, a shared substring ("the answer is", punctuation runs) is too
common to prove memorization; at or above it, verbatim reuse from the
few-shot text is copying, not coincidence."""


def _to_signal_trace(traj: TrailTrace) -> dict[str, Any]:
    """TrailTrace -> the {"task", "spans": [...]} shape the sandboxed signal
    helpers (`spans`, `tool_spans`, `model_texts`) and the structural baselines
    expect."""
    return {
        "trace_id": traj.trace_id,
        "task": traj.task,
        "user_turns": [],
        "spans": [
            {
                "kind": "model" if step["kind"] == "llm" else "tool",
                "name": step["name"],
                "status": "error" if step["is_error"] else "ok",
                "arguments": None,
                "output": step["text"],
            }
            for step in traj.steps
        ],
    }


def _stratified_split(
    labels: dict[str, bool], trace_ids: list[str], *, fit_fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """70/30 by default, kept stratified so both classes land on both sides."""
    rng = random.Random(seed)
    fit: list[str] = []
    held: list[str] = []
    for label_value in (True, False):
        bucket = [t for t in trace_ids if labels.get(t) is label_value]
        rng.shuffle(bucket)
        cut = (
            max(1, min(len(bucket) - 1, round(len(bucket) * fit_fraction)))
            if len(bucket) > 1
            else len(bucket)
        )
        fit.extend(bucket[:cut])
        held.extend(bucket[cut:])
    return tuple(fit), tuple(held)


def build_family(
    split_name: str, trajectories: list[TrailTrace], labels: dict[str, bool], *, seed: int
) -> TaskFamily | None:
    trace_ids = [t.trace_id for t in trajectories]
    labeled_ids = [t for t in trace_ids if t in labels]
    if len({labels[t] for t in labeled_ids}) < 2:
        return None
    fit_ids, held_ids = _stratified_split(labels, labeled_ids, fit_fraction=0.7, seed=seed)
    if len({labels[t] for t in fit_ids}) < 2 or len({labels[t] for t in held_ids}) < 2:
        return None
    return TaskFamily(
        family_id=f"family-trail-{split_name}",
        descriptor=f"TRAIL {split_name} traces, top/bottom score tertile only",
        trace_ids=tuple(trace_ids),
        medoid_trace_id=trace_ids[0],
        workload_mass=len(trace_ids),
        fit_trace_ids=fit_ids,
        held_out_trace_ids=held_ids,
        proposed_by="rule",
        limitations=(
            "not a mined family: TRAIL tasks are one-off, not repeated trials of a "
            "template, so this is a stratified random 70/30 split of the top/bottom "
            "score tertile, seeded for reproducibility, not evidence of task recurrence",
        ),
    )


# --- generalization guard ---------------------------------------------------


def _string_literals(code: str) -> list[str]:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _copied_literal(code: str, shown_text: str) -> str | None:
    """The first literal in ``code`` that is verbatim, memorizable copying from
    the few-shot text rather than a generic word or phrase the model would
    write regardless of which examples it happened to see."""
    haystack = shown_text.lower()
    for literal in _string_literals(code):
        candidate = literal.strip().lower()
        if len(candidate) >= _MIN_LITERAL_LEN and candidate in haystack:
            return literal
    return None


# --- recursive, feedback-driven discovery -----------------------------------


def _mismatches(
    values: dict[str, float | None],
    labels: dict[str, bool],
    traces: dict[str, dict],
    limit: int = 4,
) -> list[dict[str, Any]]:
    rows = []
    for trace_id, label in labels.items():
        score = values.get(trace_id)
        if score is None or (score >= 0.5) == label:
            continue
        task = (traces[trace_id].get("task") or "")[:160]
        rows.append(
            {
                "trace_id": trace_id,
                "true_label": "RELIABLE" if label else "UNRELIABLE",
                "signal_scored": score,
                "task": task,
            }
        )
    rows.sort(key=lambda row: row["trace_id"])
    return rows[:limit]


def _round_correction(rejections: list[str], scored_rows: list[dict[str, Any]]) -> str:
    lines = []
    if rejections:
        lines.append("HOST REJECTED THESE PROPOSALS:")
        lines.extend(f"  - {item}" for item in rejections)
    for row in scored_rows:
        lines.append(
            f"'{row['name']}' scored fit AUC={row['auc']}: right about half the time, no "
            f"better than guessing. Wrong on, among others: "
            + "; ".join(f"{m['trace_id']} (true={m['true_label']})" for m in row["mismatches"])
        )
    if scored_rows or rejections:
        lines.append(
            "Do not adjust the wording of the same keyword check. Name a different, more "
            "general property of a reliable trajectory — one that would still hold if these "
            "traces used entirely different phrasing for the same failure — and write a signal "
            "for that instead."
        )
    return "\n".join(lines)


def evaluate_family_recursive(
    family: TaskFamily,
    traces: dict[str, dict[str, Any]],
    labels: dict[str, bool],
    predict,
    *,
    keep_auc: float,
    max_fit_examples: int,
    rounds: int,
) -> dict[str, Any]:
    fit_labels = {t: labels[t] for t in family.fit_trace_ids if t in labels}
    held_labels = {t: labels[t] for t in family.held_out_trace_ids if t in labels}
    if len(fit_labels) < 4 or len({*fit_labels.values()}) < 2 or len({*held_labels.values()}) < 2:
        return {
            "family_id": family.family_id,
            "status": "skipped",
            "reason": "fit needs >=4 labelled traces and both classes; held-out needs both classes",
            "fit_labels": len(fit_labels),
            "held_out_labels": len(held_labels),
        }

    contract = json.dumps(
        {
            "family_id": family.family_id,
            "descriptor": family.descriptor,
            "limitations": family.limitations,
        },
        indent=2,
    )
    example_ids = _balanced_example_ids(fit_labels, max_fit_examples)
    rendered_examples = _examples(traces, fit_labels, example_ids)
    all_family = [traces[t] for t in family.trace_ids if t in traces]

    correction = ""
    history: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for round_index in range(rounds):
        reply = predict(
            family_contract=contract, fit_examples=rendered_examples, correction=correction
        )
        raw = getattr(reply, "signals", [])
        proposed = [
            item if isinstance(item, ProposedSignal) else ProposedSignal.model_validate(item)
            for item in raw
        ]
        rejections: list[str] = []
        scored_rows: list[dict[str, Any]] = []
        for proposal in proposed:
            if "trace_id" in proposal.code:
                rejections.append(
                    f"{proposal.name}: reads trace_id directly, not a general property"
                )
                continue
            copied = _copied_literal(proposal.code, rendered_examples)
            if copied is not None:
                rejections.append(
                    f"{proposal.name}: hardcodes {copied!r}, copied verbatim from the shown "
                    "examples rather than a property that generalizes past their exact wording"
                )
                continue
            try:
                signal_fn = _compile_signal(proposal.code)
            except Exception as exc:  # noqa: BLE001
                rejections.append(f"{proposal.name}: {exc}")
                continue
            values = _run_over_corpus(signal_fn, all_family)
            fit_metric = score_signal(proposal.name, values, fit_labels)
            row = {
                "name": proposal.name,
                "hypothesis": proposal.hypothesis,
                "code": proposal.code,
                "blind_spots": proposal.blind_spots,
                "gaming_hypotheses": proposal.gaming_hypotheses,
                "auc": fit_metric.auc,
                "fit": fit_metric.__dict__,
                "mismatches": _mismatches(values, fit_labels, traces),
            }
            scored_rows.append(row)
            if fit_metric.auc is not None and fit_metric.auc >= keep_auc:
                # This is the only point held-out labels are consulted, and only
                # for a signal that already cleared the fit bar without seeing them.
                held_metric = score_signal(proposal.name, values, held_labels)
                kept.append({**row, "held_out": held_metric.__dict__})
        history.append({"round": round_index, "rejections": rejections, "scored": scored_rows})
        if kept:
            break
        correction = _round_correction(rejections, scored_rows)

    baselines = {
        name: {
            "fit": score_signal(name, values, fit_labels).__dict__,
            "held_out": score_signal(name, values, held_labels).__dict__,
        }
        for name, values in _baseline_values(all_family).items()
    }
    return {
        "family_id": family.family_id,
        "status": "evaluated",
        "fit_labels": len(fit_labels),
        "held_out_labels": len(held_labels),
        "rounds_used": len(history),
        "history": history,
        "kept": kept,
        "baselines": baselines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash-0731")
    parser.add_argument("--keep-auc", type=float, default=0.62)
    parser.add_argument("--max-fit-examples", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=16_000)
    args = parser.parse_args()

    predict = build_predictor(
        model=args.model,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_tokens=args.max_tokens,
    )

    report: dict[str, Any] = {"source": "TRAIL (arXiv:2505.08638)", "families": []}
    for split_name in ("gaia", "swe_bench"):
        trajectories = load_trail(args.trail_dir, split_name)
        scores = {t.trace_id: t.overall for t in trajectories if t.overall is not None}
        labels, provenance = _binary_labels(scores)
        family = build_family(split_name, trajectories, labels, seed=args.seed)
        if family is None:
            report["families"].append(
                {
                    "family_id": f"family-trail-{split_name}",
                    "status": "skipped",
                    "reason": "not enough traces with both classes on both sides",
                    "n_traces": len(trajectories),
                    "n_labeled": len(labels),
                }
            )
            print(f"{split_name}: skipped, n_traces={len(trajectories)} n_labeled={len(labels)}")
            continue
        traces = {t.trace_id: _to_signal_trace(t) for t in trajectories}
        result = evaluate_family_recursive(
            family,
            traces,
            labels,
            predict,
            keep_auc=args.keep_auc,
            max_fit_examples=args.max_fit_examples,
            rounds=args.rounds,
        )
        result["label_provenance"] = provenance
        report["families"].append(result)
        print(
            f"{split_name}: {result['status']} fit={result.get('fit_labels', 0)} "
            f"held={result.get('held_out_labels', 0)} rounds={result.get('rounds_used', 0)} "
            f"kept={len(result.get('kept', []))}"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
