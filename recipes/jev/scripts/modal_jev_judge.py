"""Serve the frozen search judge on Modal and check it against the saved audit.

Deploy once so the RL reward manager can call it by name:

    uvx modal deploy recipes/jev/scripts/modal_jev_judge.py

then, from anywhere authenticated with the same Modal workspace:

    modal.Cls.from_name("jev-step-judge", "JevJudge")().score.remote(steps)

Fidelity check (scores the 100 audited steps through the server and compares
every probability with the saved report):

    uvx modal run recipes/jev/scripts/modal_jev_judge.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO))

from recipes.jev.scripts.modal_run import hf_cache, image, runs  # noqa: E402

GPU = "L40S"
RESULT_PREFIX = "@@RESULT@@ "

app = modal.App("jev-step-judge", image=image)


@app.cls(
    gpu=GPU,
    timeout=3600,
    scaledown_window=300,
    max_containers=2,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
)
class JevJudge:
    @modal.enter()
    def load(self) -> None:
        # Torch lives in the recipe's uv venv, and Modal's own Python shadows
        # that venv's packages, so the model runs in a venv subprocess.
        self.lock = threading.Lock()
        self.worker = subprocess.Popen(
            ["/repo/recipes/jev/.venv/bin/python", "/repo/recipes/jev/scripts/judge_serve_worker.py"],
            cwd="/repo/recipes/jev",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.provenance = self._read()["ready"]

    def _read(self) -> dict:
        for line in self.worker.stdout:
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX) :])
            print(line, end="", flush=True)
        raise RuntimeError(f"judge worker exited with code {self.worker.wait()}")

    @modal.method()
    def score(self, steps: list[dict]) -> dict:
        """Score observed steps. Each step: question, previous [[tool, action,
        observation], ...], tool, action, observation. Overlong or unreadable
        steps come back with a reason and no score; they are never truncated."""
        with self.lock:
            self.worker.stdin.write(json.dumps({"steps": steps}) + "\n")
            self.worker.stdin.flush()
            return {"judge": self.provenance, "results": self._read()["results"]}


@app.local_entrypoint()
def main() -> None:
    work = REPO / "work/step-rl"
    raw = {r["id"]: r for r in map(json.loads, (work / "openresearcher-v3-audit-candidates.jsonl").read_text().splitlines())}
    saved = json.loads((work / "search-judge-v3-audit.json").read_text())["new_judge"]["predictions"]
    steps = []
    for row in saved:
        s = raw[row["id"]]
        steps.append(
            {
                "question": s["question"],
                "previous": [[p["tool"], p["action"], p["observation"]] for p in s["previous"]],
                "tool": s["tool"],
                "action": s["action"],
                "observation": s["observation"],
            }
        )
    reply = JevJudge().score.remote(steps)
    worst, mismatched, rejected = 0.0, 0, 0
    for row, got in zip(saved, reply["results"], strict=True):
        if got["probabilities"] is None:
            rejected += 1
            continue
        worst = max(worst, *(abs(got["probabilities"][k] - v) for k, v in row["probabilities"].items()))
        mismatched += max(got["probabilities"], key=got["probabilities"].get) != row["predicted"]
    print(json.dumps({"provenance": reply["judge"], "steps": len(saved), "rejected": rejected,
                      "predicted_class_mismatches": mismatched, "max_abs_probability_diff": worst}, indent=2))
