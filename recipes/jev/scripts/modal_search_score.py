"""Score an external search-step audit set with the existing APB Jev adapter.

Run from the repo root with ``uvx modal run recipes/jev/scripts/modal_search_score.py``.
The labeled JSONL is copied into the image; the existing adapter is read from
the persistent Jev runs volume. This job does not train or touch BrowseComp-Plus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO))

from recipes.jev.scripts.modal_run import hf_cache, image, runs  # noqa: E402

SOURCE = REPO / "work/step-rl/openresearcher-labeled-v2.jsonl"
IMAGE = image.add_local_file(str(SOURCE), "/work/search-labeled.jsonl", copy=True) if modal.is_local() else image
APP = modal.App("jev-search-reward-gate", image=IMAGE)


@APP.function(gpu="L40S", timeout=3600, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def score() -> dict:
    import subprocess

    subprocess.run(
        ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/search_score_worker.py"],
        cwd="/repo/recipes/jev", check=True,
    )
    report = json.loads(Path("/runs/search-reward-gate.json").read_text())
    runs.commit()
    return {k: v for k, v in report.items() if k != "predictions"}


@APP.local_entrypoint()
def main() -> None:
    print(json.dumps(score.remote(), indent=2))
