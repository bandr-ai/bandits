"""Train Jev v2 on gold-evidence step labels on Modal and compare it with v1 on held-out steps.

    uvx modal run --detach recipes/jev/scripts/modal_judge_v2.py
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
    image = image.add_local_file(str(REPO / "work/step-rl/judge-v2-train.jsonl"), "/work/judge-v2-train.jsonl", copy=True)
    image = image.add_local_file(str(REPO / "work/step-rl/judge-v2-heldout.jsonl"), "/work/judge-v2-heldout.jsonl", copy=True)

app = modal.App("jev-judge-v2", image=image)


@app.function(gpu="L40S", timeout=2 * 3600, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import subprocess

    subprocess.run(
        ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/judge_v2_worker.py"],
        cwd="/repo/recipes/jev", check=True,
    )
    report = json.loads(Path("/runs/search-judge-v2/report.json").read_text())
    runs.commit()
    return {
        "best_adapter": report["best_adapter"],
        **{name: {k: report[name][k] for k in ("scored", "accuracy", "auc_positive_vs_rest",
                                               "share_scored_above_0.25_among_non_positive")} for name in ("v1", "v2")},
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run.remote(), indent=2))
