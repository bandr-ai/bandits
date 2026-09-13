#!/usr/bin/env python3
"""Score a turn-judge run (and optionally verifier scores) against tau2's sealed truth.

tau2 grades an episode against a hidden goal state: ``success`` is binary per
trace, and 74% of traces succeed, so a scorer has to beat that just by saying
"pass" to everything. The question here is whether the reaction-based score —
the share of turns the judge or the checks did *not* flag — ranks successes
above failures (AUC), and whether the traces it passes outright are cleaner
than the base rate (precision of ``passes``).

Usage:
    uv run python scripts/tau2_nextstate_eval.py --project work/tau2-ns \
        --judge-run turn-judge-… [--scores verifier-scores-…] \
        --labels work/tau2-run/tau2.labels.json --out work/tau2-ns/eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bandits.store import DerivedStore
from bandits.verify.nextstate import load_turn_judge_run
from bandits.verify.propose import load_verifier_scores


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(
    name: str, score_of: dict[str, float | None], passes_of: dict[str, bool], truth: dict[str, bool]
) -> dict:
    ids = [t for t in score_of if t in truth]
    pairs = [(score_of[t], truth[t]) for t in ids if score_of[t] is not None]
    passed = [t for t in ids if passes_of[t]]
    failed = [t for t in ids if not passes_of[t]]
    base = sum(truth[t] for t in ids) / len(ids) if ids else None
    return {
        "scorer": name,
        "traces": len(ids),
        "scored": len(pairs),
        "base_rate_success": base,
        "auc(score, success)": auc(pairs),
        "passes": len(passed),
        "precision_of_passes": sum(truth[t] for t in passed) / len(passed) if passed else None,
        "coverage_of_successes": sum(truth[t] for t in passed) / sum(truth[t] for t in ids)
        if ids
        else None,
        "failure_rate_among_flagged": (1 - sum(truth[t] for t in failed) / len(failed))
        if failed
        else None,
        "accuracy(passes==success)": sum(passes_of[t] == truth[t] for t in ids) / len(ids)
        if ids
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--judge-run", required=True)
    parser.add_argument("--scores", default=None)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    derived = DerivedStore(args.project / ".bandits")
    run = load_turn_judge_run(args.judge_run, derived)
    labels = json.loads(args.labels.read_text())
    truth = {t: bool(v["success"]) for t, v in labels.items()}

    signals = run.signal_by_trace()
    rows = [
        summarize(
            "judge",
            {t: s.score for t, s in signals.items()},
            {t: s.passes for t, s in signals.items()},
            truth,
        )
    ]
    # A judge that is right about which turns are wrong should be most right
    # about the first one; the count and the position are two readings.
    rows.append(
        summarize(
            "judge, negatives only",
            {t: -float(s.negative) for t, s in signals.items()},
            {t: s.negative == 0 for t, s in signals.items()},
            truth,
        )
    )
    if args.scores:
        scores = load_verifier_scores(args.scores, derived)
        by = {s.trace_id: s for s in scores.scores}
        rows.append(
            summarize(
                "verifier" + (" + judge" if scores.include_judge else " (checks only)"),
                {t: s.score for t, s in by.items()},
                {t: s.passes for t, s in by.items()},
                truth,
            )
        )
    report = {"judge_run": args.judge_run, "scores": args.scores, "model": run.model, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
