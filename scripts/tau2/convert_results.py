"""Flatten a published τ²-bench results file into the JSONL bandits.ingest --source tau2 expects.

τ²-bench ships one JSON *object* per run (keys: timestamp/info/tasks/simulations),
with each simulation's outcome nested at `reward_info.reward` (a float, not the
top-level `success`/`reward` bandits.ingest.tau2.parse_tau2_record looks for).
This script does only that reshaping -- one record per line, task_id/success/
reward/model promoted to the top level, messages passed through unchanged so the
existing tau2 adapter's tool_call_id<-id remap still does the rest.

Usage:
    python scripts/tau2/convert_results.py \
        /tmp/tau2-bench/data/tau2/results/final/gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json \
        -o work/tau2-retail/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("info", {}).get("agent_info", {}).get("llm")
    records = []
    for sim in data.get("simulations", []):
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")
        success = reward == 1.0 if isinstance(reward, (int, float)) else None
        records.append({
            "task_id": sim.get("task_id"),
            "success": success,
            "reward": reward,
            "model": model,
            "messages": sim.get("messages", []),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--only-success", action="store_true", help="Drop failed/unlabeled trials.")
    args = ap.parse_args()

    records = convert(args.results_json)
    if args.only_success:
        records = [r for r in records if r["success"] is True]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    n_success = sum(1 for r in records if r["success"] is True)
    print(f"wrote {len(records)} trajectories ({n_success} success) -> {args.out}")


if __name__ == "__main__":
    main()
