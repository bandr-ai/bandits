"""Paired held-out comparison of evaluation runs on the same questions.

    python recipes/jev/scripts/compare_runs.py base=work/step-rl/rollouts/base.json \
        outcome=work/step-rl/grpo/outcome-seed1-eval.json step=work/step-rl/grpo/step-seed1-eval.json

Each file holds rollouts under "results" (rollout harness) or "eval" (trainer).
Per question, each metric is averaged over that question's rollouts; arms are
then compared question by question, and the 95% interval comes from resampling
questions. Pure Python.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

METRICS = {
    "evidence_seen": lambda r: r["evidence_seen_recall"],
    "evidence_opened": lambda r: r["evidence_opened_recall"],
    "accuracy_proxy": lambda r: float(r["correct"]),
    "tool_calls": lambda r: float(r["tool_calls"]),
    "repeated_action": lambda r: float(len({e["action"] for e in r["events"]}) < len(r["events"])),
}


def per_question(rollouts: list[dict]) -> dict[str, dict[str, float]]:
    groups = defaultdict(list)
    for r in rollouts:
        groups[r["query_id"]].append(r)
    return {q: {m: sum(f(r) for r in rs) / len(rs) for m, f in METRICS.items()} for q, rs in groups.items()}


def paired(a: dict, b: dict, metric: str, rounds: int = 2000, seed: int = 0) -> dict:
    questions = sorted(set(a) & set(b))
    diffs = [b[q][metric] - a[q][metric] for q in questions]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(rounds))
    return {
        "questions": len(questions),
        "difference": round(sum(diffs) / len(diffs), 4),
        "ci95": [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds) - 1], 4)],
    }


def main(specs: list[str]) -> dict:
    arms = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        arms[name] = per_question(data.get("eval") or data["results"])
    names = list(arms)
    out = {
        "means": {
            n: {m: round(sum(q[m] for q in arms[n].values()) / len(arms[n]), 4) for m in METRICS} for n in names
        },
        "paired": {},
    }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out["paired"][f"{b} - {a}"] = {m: paired(arms[a], arms[b], m) for m in METRICS}
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:]), indent=1))
