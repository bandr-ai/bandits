"""Score both 4B judge checkpoints on the fresh rubric-v3 search audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO))

from recipes.jev.scripts.modal_run import hf_cache, image, runs  # noqa: E402

if modal.is_local():
    image = image.add_local_file(str(REPO / "work/step-rl/openresearcher-v3-audit-labeled.jsonl"), "/work/search-audit-v3.jsonl", copy=True)

app = modal.App("jev-search-expanded-audit", image=image)


@app.function(gpu="L40S", timeout=3600, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import subprocess

    subprocess.run(
        ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/search_expanded_worker.py"],
        cwd="/repo/recipes/jev", check=True,
    )
    report = json.loads(Path("/runs/search-judge-pilot/v3-audit.json").read_text())
    runs.commit()
    return {
        "rows": report["new_judge"]["rows"],
        "majority_baseline": report["majority_baseline"],
        "old_accuracy": report["old_judge"]["accuracy"],
        "new_accuracy": report["new_judge"]["accuracy"],
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run.remote(), indent=2))
