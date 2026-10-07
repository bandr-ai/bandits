"""Train the outcome Jev on Modal and gate it on held-out final answers.

    uvx modal run --detach recipes/jev/scripts/modal_outcome_judge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO))

from recipes.jev.scripts.modal_run import hf_cache, image, runs  # noqa: E402

if modal.is_local():
    image = image.add_local_file(str(REPO / "work/step-rl/outcome-train.jsonl"), "/work/outcome-train.jsonl", copy=True)
    image = image.add_local_file(str(REPO / "work/step-rl/outcome-heldout.jsonl"), "/work/outcome-heldout.jsonl", copy=True)

app = modal.App("jev-outcome-judge", image=image)


@app.function(gpu="L40S", timeout=2 * 3600, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import subprocess

    subprocess.run(
        ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/outcome_judge_worker.py"],
        cwd="/repo/recipes/jev", check=True,
    )
    report = json.loads(Path("/runs/outcome-judge-v1/report.json").read_text())
    runs.commit()
    return {"best_adapter": report["best_adapter"], "gate": report["gate"]}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run.remote(), indent=2))
