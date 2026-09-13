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

Everything downstream — fit-only RLM discovery, sandboxed signal execution,
fit-threshold selection, held-out reveal only for kept signals — reuses
`evaluate_family` from `trail_rlm_verifier.py` unchanged.

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
from trail_rlm_verifier import _binary_labels, build_predictor, evaluate_family  # noqa: E402
from trail_signal import TrailTrace, load_trail  # noqa: E402

from bandits.analyze.models import TaskFamily


def _to_signal_trace(traj: TrailTrace) -> dict[str, Any]:
    """TrailTrace -> the {"task", "spans": [{"kind","name","status","output"}]} shape
    `evaluate_family`'s baselines and renderer expect (the same shape the earlier
    tau2 signal-discovery smoke tests used)."""
    return {
        "trace_id": traj.trace_id,
        "task": traj.task,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash-0731")
    parser.add_argument("--keep-auc", type=float, default=0.62)
    parser.add_argument("--max-fit-examples", type=int, default=12)
    parser.add_argument("--repair-attempts", type=int, default=1)
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
        result = evaluate_family(
            family,
            traces,
            labels,
            predict,
            keep_auc=args.keep_auc,
            max_fit_examples=args.max_fit_examples,
            repair_attempts=args.repair_attempts,
        )
        result["label_provenance"] = provenance
        report["families"].append(result)
        print(
            f"{split_name}: {result['status']} fit={result.get('fit_labels', 0)} "
            f"held={result.get('held_out_labels', 0)}"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
