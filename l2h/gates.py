"""Execution gates: run a task's solution/verifier and judge the outcome.

The default LocalRunner runs the task's real tests/test.sh (and solve.sh for the
oracle) in a subprocess with L2H_* env vars pointing at a scratch dir: no
Docker, about a second. Spawn failures, timeouts and a verifier that exits
without writing a reward are INFRA ERRORS, never task failures.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .harbor import ANSWER_DIR


@dataclass
class RunOutcome:
    state: str  # "ok" | "infra_error"
    reward: float | None = None
    detail: str = ""
    log: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reward": self.reward,
            "detail": self.detail,
            "log": self.log[-2000:],
        }


class Runner:
    """Interface. `run_verifier(task_dir, answer_files)` -> RunOutcome."""

    name = "abstract"

    def run_solution(self, task_dir: Path) -> RunOutcome: ...
    def run_verifier(self, task_dir: Path, answer: dict[str, str] | None) -> RunOutcome: ...


@dataclass
class LocalRunner(Runner):
    timeout: float = 60.0
    name: str = "local"
    bash: str = field(default_factory=lambda: shutil.which("bash") or "bash")

    def _exec(
        self, script: Path, env: dict, cwd: Path
    ) -> tuple[subprocess.CompletedProcess | None, str]:
        try:
            cp = subprocess.run(
                [self.bash, str(script)],
                env=env,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return cp, ""
        except subprocess.TimeoutExpired:
            return None, f"timed out after {self.timeout}s"
        except OSError as e:
            return None, f"could not start {script.name}: {e}"

    def _env(self, task_dir: Path, scratch: Path) -> dict:
        import os

        env = dict(os.environ)
        env.update(
            L2H_TESTS_DIR=str(task_dir / "tests"),
            L2H_ANSWER_DIR=str(scratch / "answer"),
            L2H_REWARD_DIR=str(scratch / "verifier"),
        )
        return env

    def run_verifier(self, task_dir: Path, answer: dict[str, str] | None) -> RunOutcome:
        """answer: {filename: content} written into the answer dir; None/{} = no-op."""
        script = task_dir / "tests" / "test.sh"
        if not script.is_file():
            return RunOutcome("infra_error", detail="tests/test.sh is missing")
        with tempfile.TemporaryDirectory(prefix="l2h-gate-") as tmp:
            scratch = Path(tmp)
            (scratch / "answer").mkdir()
            for name, content in (answer or {}).items():
                (scratch / "answer" / name).write_text(content)
            return self._verify(task_dir, scratch)

    def _verify(self, task_dir: Path, scratch: Path) -> RunOutcome:
        cp, err = self._exec(task_dir / "tests" / "test.sh", self._env(task_dir, scratch), scratch)
        if cp is None:
            return RunOutcome("infra_error", detail=err)
        log = (cp.stdout or "") + (cp.stderr or "")
        reward_file = scratch / "verifier" / "reward.txt"
        if not reward_file.is_file():
            return RunOutcome(
                "infra_error",
                detail=f"verifier exited {cp.returncode} without writing /logs/verifier/reward.txt",
                log=log,
            )
        try:
            reward = float(reward_file.read_text().strip())
        except ValueError:
            return RunOutcome("infra_error", detail="reward.txt is not a number", log=log)
        if not 0.0 <= reward <= 1.0:
            return RunOutcome("infra_error", detail=f"reward {reward} outside 0..1", log=log)
        return RunOutcome("ok", reward=reward, log=log)

    def run_solution(self, task_dir: Path) -> RunOutcome:
        """Oracle: run solution/solve.sh, then the verifier on what it wrote."""
        script = task_dir / "solution" / "solve.sh"
        if not script.is_file():
            return RunOutcome("infra_error", detail="solution/solve.sh is missing")
        with tempfile.TemporaryDirectory(prefix="l2h-gate-") as tmp:
            scratch = Path(tmp)
            (scratch / "answer").mkdir()
            cp, err = self._exec(script, self._env(task_dir, scratch), scratch)
            if cp is None:
                return RunOutcome("infra_error", detail=err)
            if cp.returncode != 0:
                # the reference solution itself broke: that is a task defect
                return RunOutcome(
                    "ok", reward=0.0, detail=f"solve.sh exited {cp.returncode}", log=cp.stderr or ""
                )
            return self._verify(task_dir, scratch)


def verdict(outcome: RunOutcome, want_reward: bool) -> str:
    """Map a run to a gate state: pass / fail / error."""
    if outcome.state != "ok":
        return "error"
    got = (outcome.reward or 0.0) >= 1.0
    return "pass" if got == want_reward else "fail"


def derive_wrong(expected_canonical: str, check: str) -> str:
    """Fallback wrong answer when the generator supplied none (flagged as auto)."""
    if check == "numeric":
        try:
            return str(float(expected_canonical) * 2 + 1)
        except ValueError:
            pass
    return expected_canonical + " (wrong)"


def run_exec_gates(
    runner: Runner,
    task_dir: Path,
    answer_file: str,
    wrong: list[str],
    alternatives: list[str],
    auto_wrong: bool,
) -> dict:
    """Run oracle / no-op / wrong-answer / alternatives. Value = gate satisfied?"""
    detail: dict[str, dict] = {}

    o = runner.run_solution(task_dir)
    detail["oracle"] = {"gate": verdict(o, True), **o.to_dict()}
    n = runner.run_verifier(task_dir, {})
    detail["noop"] = {"gate": verdict(n, False), **n.to_dict()}
    wrongs = []
    for w in wrong:
        r = runner.run_verifier(task_dir, {answer_file: str(w)})
        wrongs.append({"answer": str(w), "gate": verdict(r, False), **r.to_dict()})
    detail["wrong"] = {
        "gate": _combine([w["gate"] for w in wrongs]) if wrongs else "error",
        "auto": auto_wrong,
        "runs": wrongs,
    }
    alts = []
    for a in alternatives:
        r = runner.run_verifier(task_dir, {answer_file: str(a)})
        alts.append({"answer": str(a), "gate": verdict(r, True), **r.to_dict()})
    detail["alternatives"] = {
        "gate": _combine([a["gate"] for a in alts]) if alts else "not_run",
        "runs": alts,
    }
    gates = [detail[k]["gate"] for k in ("oracle", "noop", "wrong", "alternatives")]
    if "fail" in gates:
        exec_state = "fail"
    elif "error" in gates:
        exec_state = "error"
    else:
        exec_state = "pass"
    return {"exec": exec_state, "detail": detail, "runner": runner.name}


def _combine(states: list[str]) -> str:
    if "fail" in states:
        return "fail"
    if "error" in states:
        return "error"
    return "pass"


__all__ = [
    "LocalRunner",
    "Runner",
    "RunOutcome",
    "run_exec_gates",
    "derive_wrong",
    "verdict",
    "ANSWER_DIR",
]
