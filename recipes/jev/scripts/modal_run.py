"""Run a launch-plan phase (scripts/gpu_run.sh) on a Modal GPU.

From the repo root, with the Bandits work projects in ./work (gitignored):

    uvx modal run recipes/jev/scripts/modal_run.py --phase 1
    uvx modal volume ls jev-runs /
    uvx modal volume get jev-runs /<run-dir>.tar.gz .

The checkout this script sits in and the two work projects (work/trail-ns,
work/tau2-ns) are copied into the image, and the recipe's train environment
is installed at build time. From a git worktree, which has no work/, set
JEV_WORK to the main checkout's work/. .env files and gitignored data are
left out of the image, and of each work project only its .bandits store and
the launch ledgers go in.

Runs, the shared project (so phase 2 reuses phase 1's finished training) and
the Hugging Face cache live on persistent volumes. Launch runs one after
another: each container sees the volume as it was when it started, so a run
started before another finished retrains instead of reusing its work.

The script's output streams to the console and to <run-dir>.console.log on
the runs volume as it runs.

Prices default to list prices on 2026-09-25: Modal L40S $0.000542/s, and
the judge model (Fireworks nemotron-lightning-3p5-30b-a3b) $0.05 in /
$0.01 cached in / $0.20 out per Mtok. Override them if your bill differs.
"""

from __future__ import annotations

import collections
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

if modal.is_local():
    REPO = Path(__file__).resolve().parents[3]
    WORK = Path(os.environ.get("JEV_WORK", REPO / "work")).resolve()
else:
    # In the container this file sits at /root, and the image is already built.
    REPO = WORK = Path("/")

JUDGE_RUNS = "turn-judge-0044b8f5c041155c turn-judge-b83480800ad90a25 turn-judge-d450cf34d97b27a2"
"""TRAIL GAIA (v2), TRAIL SWE (v3) and tau2: the launch data (no shared traces)."""
LEDGERS = "/work/tau2-ns/ledger-judge-sub.jsonl /work/trail-ns/ledger-gaia-v2.jsonl /work/trail-ns/ledger-swe-v3.jsonl"
SWE_JUDGE_RUN = "turn-judge-b83480800ad90a25"


def _not_read_by_the_run(path: Path) -> bool:
    """A work project is uploaded as only what gpu_run.sh reads from it: its
    .bandits store and the launch ledgers. Old ledgers, logs, evaluations and
    anything else there (a .env included) stay local."""
    kept = path.parts[:1] == (".bandits",) or path.name in {
        Path(ledger).name for ledger in LEDGERS.split()
    }
    return not kept or path.name == ".env"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install("uv")
    .add_local_dir(
        str(REPO),
        "/repo",
        copy=True,
        ignore=[
            "**/.venv",
            "**/__pycache__",
            "**/.pytest_cache",
            "**/.ruff_cache",
            "**/.coverage",
            ".git",
            # .env holds the Fireworks key; the rest is gitignored local data the run doesn't read.
            "**/.env",
            "work",
            "runs",
            "datasets",
            "handoffs",
            ".bandits",
        ],
    )
    .add_local_dir(str(WORK / "trail-ns"), "/work/trail-ns", copy=True, ignore=_not_read_by_the_run)
    .add_local_dir(str(WORK / "tau2-ns"), "/work/tau2-ns", copy=True, ignore=_not_read_by_the_run)
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


TIMEOUT_S = 4 * 60 * 60
SCRIPT_TIMEOUT_S = TIMEOUT_S - 10 * 60
"""Stop gpu_run.sh before Modal kills the container, so its log is saved and the volume committed."""


@app.function(
    gpu="L40S", timeout=TIMEOUT_S, volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache}
)
def run_phase(phase: int, extra_env: dict[str, str], tag: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = "-".join(
        part for part in (f"phase{phase}", f"seed{extra_env.get('SEED', '1')}", tag, stamp) if part
    )
    out = f"/runs/{name}"
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
    tail: collections.deque[str] = collections.deque(maxlen=40)
    # coreutils timeout signals the whole process group, so jev's Python children stop too.
    cmd = [
        "timeout",
        "--kill-after=60",
        str(SCRIPT_TIMEOUT_S),
        "bash",
        "recipes/jev/scripts/gpu_run.sh",
    ]
    try:
        with (
            open(f"{out}.console.log", "w") as log,
            subprocess.Popen(
                cmd,
                cwd="/repo",
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            ) as proc,
        ):
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
        returncode = proc.returncode
    finally:
        runs.commit()
        hf_cache.commit()
    last = "".join(tail)
    if returncode in (124, 137):  # 137: still running at the timeout's --kill-after
        raise RuntimeError(
            f"gpu_run.sh stopped after {SCRIPT_TIMEOUT_S}s; log at {out}.console.log\n{last}"
        )
    if returncode != 0:
        raise RuntimeError(f"gpu_run.sh exited {returncode}; log at {out}.console.log\n{last}")
    return f"{out}\n{last}"


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
    tag: str = "",
):
    """--models overrides gpu_run.sh's default (Base and Instruct 4B); pin a
    model's revision as model@sha (phase 2 passes the sha phase 1 printed).
    --held-out-swe adds the unseen-source run (phase 2 only). --tag names the
    run directory; by default it is the model's name when --models names one."""
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
    if not tag and len(models.split()) == 1:
        tag = models.split("@")[0].split("/")[-1]
    if held_out_swe:
        tag = f"{tag}-held-out-swe" if tag else "held-out-swe"
    print(run_phase.remote(phase, extra, tag))
