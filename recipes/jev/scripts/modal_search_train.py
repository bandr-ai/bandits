"""Run the search-specific judge pilot on Modal after preparing input JSONL files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO))

from recipes.jev.scripts.modal_run import hf_cache, image, runs  # noqa: E402

if modal.is_local():
    image = image.add_local_file(str(REPO / "work/step-rl/search-train-combined.jsonl"), "/work/search-train.jsonl", copy=True)
    image = image.add_local_file(str(REPO / "work/step-rl/openresearcher-labeled-v2.jsonl"), "/work/search-audit.jsonl", copy=True)
    image = image.add_local_file(str(REPO / "work/step-rl/search-reward-probes.jsonl"), "/work/search-probes.jsonl", copy=True)

app = modal.App("jev-search-judge-pilot", image=image)


@app.function(gpu="L40S", timeout=7200, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def run() -> dict:
    import subprocess

    subprocess.run(
        ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/search_train_worker.py"],
        cwd="/repo/recipes/jev", check=True,
    )
    report = json.loads(Path("/runs/search-judge-pilot/report.json").read_text())
    runs.commit()
    return {
        "dataset_id": report["dataset_id"], "best_adapter": report["best_adapter"],
        "external_audit_accuracy": report["external_audit"]["accuracy"],
        "attack_probe_accuracy": report["attack_probes"]["accuracy"],
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run.remote(), indent=2))
