"""Run a launch-plan phase (scripts/gpu_run.sh) on a Modal GPU.

From the repo root, with the Bandits work projects in ./work (gitignored):

    uvx modal run recipes/jev/scripts/modal_run.py --phase 1
    uvx modal volume ls jev-runs /
    uvx modal volume get jev-runs /<phase-dir>.tar.gz .

The repo and the two work projects (work/trail-ns, work/tau2-ns) are copied
into the image, and the recipe's train environment is installed at build
time. Runs, the shared project (so phase 2 reuses phase 1's finished
training) and the Hugging Face cache live on persistent volumes.

Prices default to list prices on 2026-09-25: Modal L40S $0.000542/s, and
the judge model (Fireworks nemotron-lightning-3p5-30b-a3b) $0.05 in /
$0.01 cached in / $0.20 out per Mtok. Override them if your bill differs.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3]
WORK = REPO / "work"

JUDGE_RUNS = "turn-judge-0044b8f5c041155c turn-judge-b83480800ad90a25 turn-judge-d450cf34d97b27a2"
"""TRAIL GAIA (v2), TRAIL SWE (v3) and tau2: the launch data (no shared traces)."""
LEDGERS = (
    "/work/tau2-ns/ledger-judge-sub.jsonl /work/trail-ns/ledger-gaia-v2.jsonl /work/trail-ns/ledger-swe-v3.jsonl"
)
SWE_JUDGE_RUN = "turn-judge-b83480800ad90a25"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("uv")
    .add_local_dir(
        str(REPO),
        "/repo",
        copy=True,
        ignore=["**/.venv", "**/__pycache__", "**/.pytest_cache", "**/.ruff_cache", ".git", "work", "runs"],
    )
    .add_local_dir(str(WORK / "trail-ns"), "/work/trail-ns", copy=True)
    .add_local_dir(str(WORK / "tau2-ns"), "/work/tau2-ns", copy=True)
    # flash-linear-attention is Triton only (no CUDA build) and speeds up
    # Qwen3.5's linear-attention layers; causal-conv1d is skipped (it needs
    # a CUDA toolchain), so the conv falls back to PyTorch.
    .run_commands(
        "cd /repo/recipes/jev && uv sync --extra dev --extra train",
        "cd /repo/recipes/jev && uv pip install flash-linear-attention",
    )
)

runs = modal.Volume.from_name("jev-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("jev-hf-cache", create_if_missing=True)
app = modal.App("jev-launch-run", image=image)


@app.function(gpu="L40S", timeout=4 * 60 * 60, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def run_phase(phase: int, extra_env: dict[str, str]) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = f"/runs/phase{phase}-{stamp}"
    env = {
        **os.environ,
        "PHASE": str(phase),
        "PROJECTS": "/work/trail-ns /work/tau2-ns",
        "JUDGE_RUNS": JUDGE_RUNS,
        "PROJECT_DIR": "/runs/jev-project",
        "OUT": out,
        "SKIP_INSTALL": "1",
        **extra_env,
    }
    result = subprocess.run(
        ["bash", "recipes/jev/scripts/gpu_run.sh"], cwd="/repo", env=env, text=True, capture_output=True
    )
    Path(f"{out}.console.log").write_text(result.stdout + "\n--- stderr ---\n" + result.stderr)
    runs.commit()
    hf_cache.commit()
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
    if result.returncode != 0:
        raise RuntimeError(f"gpu_run.sh exited {result.returncode}; log at {out}.console.log\n{tail}")
    return f"{out}\n{tail}"


@app.local_entrypoint()
def main(
    phase: int = 1,
    seed: str = "1",
    models: str = "",
    held_out_swe: bool = False,
    gpu_usd_per_hour: str = "1.9512",
    input_usd_per_mtok: str = "0.05",
    cached_input_usd_per_mtok: str = "0.01",
    output_usd_per_mtok: str = "0.20",
):
    """--models overrides gpu_run.sh's default (Base and Instruct 4B);
    --held-out-swe adds the unseen-source run (phase 2 only)."""
    extra = {
        "SEED": seed,
        "GPU_USD_PER_HOUR": gpu_usd_per_hour,
        "LEDGERS": LEDGERS,
        "INPUT_USD_PER_MTOK": input_usd_per_mtok,
        "CACHED_INPUT_USD_PER_MTOK": cached_input_usd_per_mtok,
        "OUTPUT_USD_PER_MTOK": output_usd_per_mtok,
    }
    if models:
        extra["MODELS"] = models
    if held_out_swe:
        extra["HELD_OUT_RUNS"] = SWE_JUDGE_RUN
    print(run_phase.remote(phase, extra))
