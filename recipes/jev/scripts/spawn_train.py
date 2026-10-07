"""Spawn a training run on the deployed `jev-grpo` app and return.

The run executes server-side, so the launching machine may sleep. Progress goes to
/runs/step-rl/grpo/<run>/metrics.jsonl and the result to eval.json on `jev-runs`.

    uvx modal deploy recipes/jev/scripts/modal_grpo.py
    uvx --from modal python recipes/jev/scripts/spawn_train.py jev_final 1 0.3 0.20
    uvx --from modal python recipes/jev/scripts/spawn_train.py jev_final_step 1 0.3 -0.17 api realjev
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import modal

spec = importlib.util.spec_from_file_location("modal_grpo", Path(__file__).with_name("modal_grpo.py"))
grpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grpo)

arm, seed, step_weight, baseline = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
backend = sys.argv[5] if len(sys.argv) > 5 else "local"
tag = sys.argv[6] if len(sys.argv) > 6 else ""
config, train_pool, eval_tasks = grpo.make_job(arm, seed, step_weight=step_weight, baseline=baseline, judge_backend=backend, tag=tag)
trainer = "ApiTrainer" if backend == "api" else "Trainer"
call = modal.Cls.from_name("jev-grpo", trainer)().train.spawn(config, train_pool, eval_tasks)
print(json.dumps({"run": config["run"], "function_call_id": call.object_id}))
