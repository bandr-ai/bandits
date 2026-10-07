"""Spawn a held-out evaluation of a saved adapter on the deployed `jev-grpo` app and return.

The evaluation runs server-side, so the launching machine may sleep. The result is written
to /runs/step-rl/grpo/<run>/eval.json on the `jev-runs` volume.

    uvx modal deploy recipes/jev/scripts/modal_grpo.py
    uvx --from modal python recipes/jev/scripts/spawn_eval.py oracle-seed1-oracle
    uvx --from modal python recipes/jev/scripts/spawn_eval.py base --untrained   # the no-RL baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

run = sys.argv[1]
untrained = "--untrained" in sys.argv[2:]
work = Path(__file__).resolve().parents[3] / "work/step-rl"
tasks = [json.loads(line) for line in (work / "tasks.jsonl").read_text().splitlines()]
eval_tasks = [t for t in tasks if t["split"] == "held_out_eval"][:60]
call = modal.Cls.from_name("jev-grpo", "Evaluator")().evaluate.spawn(
    run, None if untrained else f"/runs/step-rl/grpo/{run}/adapter-step10", eval_tasks, 8
)
print(json.dumps({"run": run, "function_call_id": call.object_id, "eval_questions": len(eval_tasks)}))
