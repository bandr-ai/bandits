"""Execution gates: run a task's solution/verifier and judge the outcome.

The default LocalRunner runs the task's real tests/test.sh (and solve.sh for the
oracle) in a subprocess with MINT_* env vars pointing at a scratch dir: no
Docker, about a second. Spawn failures, timeouts and a verifier that exits
without writing a reward are INFRA ERRORS, never task failures.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .harbor import ANSWER_DIR


@dataclass
class RunOutcome:
    state: str  # "ok" | "infra_error"
    reward: float | None = None
    detail: str = ""
    log: str = ""
    trial: str = ""  # Harbor trial id; only a real Harbor run sets it (evidence for `validated`)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reward": self.reward,
            "detail": self.detail,
            "trial": self.trial,
            "log": self.log[-2000:],
        }


class Runner:
    """Interface. `run_verifier(task_dir, answer_files)` -> RunOutcome."""

    name = "abstract"

    def run_solution(self, task_dir: Path) -> RunOutcome: ...
    def run_verifier(self, task_dir: Path, answer: dict[str, str] | None) -> RunOutcome: ...

    def run_script(self, task_dir: Path, script: str) -> RunOutcome:
        """Run an executable outcome: a copy of the task whose solve.sh is `script`."""
        with tempfile.TemporaryDirectory(prefix="mint-outcome-") as tmp:
            copy = Path(tmp) / task_dir.name
            shutil.copytree(task_dir, copy)
            solve = copy / "solution" / "solve.sh"
            solve.parent.mkdir(exist_ok=True)
            body = script if script.startswith("#!") else "#!/bin/bash\n" + script
            solve.write_text(body.rstrip("\n") + "\n")
            solve.chmod(0o755)
            return self.run_solution(copy)


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
            MINT_TESTS_DIR=str(task_dir / "tests"),
            MINT_ANSWER_DIR=str(scratch / "answer"),
            MINT_REWARD_DIR=str(scratch / "verifier"),
        )
        return env

    def run_verifier(self, task_dir: Path, answer: dict[str, str] | None) -> RunOutcome:
        """answer: {filename: content} written into the answer dir; None/{} = no-op."""
        script = task_dir / "tests" / "test.sh"
        if not script.is_file():
            return RunOutcome("infra_error", detail="tests/test.sh is missing")
        with tempfile.TemporaryDirectory(prefix="mint-gate-") as tmp:
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
        with tempfile.TemporaryDirectory(prefix="mint-gate-") as tmp:
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


@dataclass
class HarborRunner(Runner):
    """Real execution: `harbor run` builds the task's images and runs the
    oracle/nop agent plus the (separate) verifier in containers. Only passes
    from this runner may make a task `validated`.

    Gates map to Harbor agents: oracle -> `-a oracle` (runs solution/solve.sh);
    no-op -> `-a nop`; a specific answer -> `-a oracle` on a scratch copy whose
    solve.sh just writes that answer, so the real artifact hand-off to the
    verifier is exercised. Anything that stops the run from producing a
    verifier reward (harbor missing, timeout, build failure, agent exception)
    is an INFRA ERROR, never a task failure.
    """

    timeout: float = 1800.0
    name: str = "harbor"
    binary: str = field(default_factory=lambda: shutil.which("harbor") or "harbor")

    def _harbor(self, task_dir: Path, agent: str) -> RunOutcome:
        with tempfile.TemporaryDirectory(prefix="mint-harbor-") as tmp:
            jobs = Path(tmp) / "jobs"
            cmd = [
                self.binary, "run", "-p", str(task_dir), "-a", agent, "-o", str(jobs),
                "--job-name", "gate", "-n", "1", "-k", "1", "-q", "--delete",
            ]  # fmt: skip
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return RunOutcome("infra_error", detail=f"harbor timed out after {self.timeout}s")
            except OSError as e:
                return RunOutcome("infra_error", detail=f"could not start harbor: {e}")
            log = (cp.stdout or "") + (cp.stderr or "")
            results = sorted((jobs / "gate").glob("*/result.json"))
            if not results:
                return RunOutcome(
                    "infra_error",
                    detail=f"harbor exited {cp.returncode} without a trial result (Docker running? image build ok?)",
                    log=log,
                )
            try:
                res = json.loads(results[0].read_text())
            except (OSError, ValueError):
                return RunOutcome("infra_error", detail="trial result.json is unreadable", log=log)
            exc = res.get("exception_info")
            if exc:
                kind = exc.get("exception_type") or "exception"
                msg = str(exc.get("exception_message") or "")[:300]
                return RunOutcome("infra_error", detail=f"trial raised {kind}: {msg}", log=log)
            rewards = (res.get("verifier_result") or {}).get("rewards")
            if not rewards:
                return RunOutcome("infra_error", detail="verifier produced no reward", log=log)
            reward = rewards.get("reward", next(iter(rewards.values())))
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                return RunOutcome("infra_error", detail="reward is not a number", log=log)
            if not 0.0 <= reward <= 1.0:
                return RunOutcome("infra_error", detail=f"reward {reward} outside 0..1", log=log)
            trial = str(res.get("trial_name") or results[0].parent.name)
            return RunOutcome("ok", reward=reward, log=log, trial=trial)

    def run_solution(self, task_dir: Path) -> RunOutcome:
        return self._harbor(task_dir, "oracle")

    def run_verifier(self, task_dir: Path, answer: dict[str, str] | None) -> RunOutcome:
        if not answer:
            return self._harbor(task_dir, "nop")
        with tempfile.TemporaryDirectory(prefix="mint-harbor-copy-") as tmp:
            copy = Path(tmp) / task_dir.name
            shutil.copytree(task_dir, copy)
            lines = ["#!/bin/bash", "set -e", f"mkdir -p {shlex.quote(ANSWER_DIR)}"]
            for fname, content in answer.items():
                dest = shlex.quote(f"{ANSWER_DIR}/{fname}")
                lines.append(f"printf '%s' {shlex.quote(content)} > {dest}")
            solve = copy / "solution" / "solve.sh"
            solve.parent.mkdir(exist_ok=True)
            solve.write_text("\n".join(lines) + "\n")
            solve.chmod(0o755)
            return self._harbor(copy, "oracle")


def harbor_available() -> bool:
    return bool(shutil.which("harbor") and shutil.which("docker"))


def pick_runner(choice: str, cfg: dict) -> Runner:
    """auto: Harbor when it and Docker are installed, otherwise the host-local
    runner (whose passes leave tasks `generated`, never `validated`)."""
    from .output import MINTError

    if choice == "harbor" or (choice == "auto" and harbor_available()):
        if not harbor_available():
            raise MINTError(
                "--runner harbor needs both `harbor` and `docker` on PATH",
                "install Harbor (`uv tool install harbor`) and start Docker, or use --runner local",
            )
        return HarborRunner(timeout=cfg.get("harbor_timeout_sec", 1800))
    return LocalRunner(timeout=cfg["gate_timeout_sec"])


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


def _label(item) -> dict:
    """How a wrong answer / alternative is named in the evidence."""
    if isinstance(item, dict):
        return {"script": item.get("note") or item["script"][:80]}
    return {"answer": str(item)}


def _try(runner: Runner, task_dir: Path, answer_file: str | None, item) -> RunOutcome:
    """Run one wrong answer / alternative: a string goes in the answer file, an Outcome runs."""
    if isinstance(item, dict):
        return runner.run_script(task_dir, item["script"])
    if not answer_file:
        return RunOutcome(
            "infra_error", detail="a string answer needs an answer file; use an Outcome script"
        )
    return runner.run_verifier(task_dir, {answer_file: str(item)})


def plan_exec_gates(
    runner: Runner,
    task_dir: Path,
    answer_file: str | None,
    wrong: list,
    alternatives: list,
    auto_wrong: bool,
) -> tuple[list[Callable[[], RunOutcome]], Callable[[list[RunOutcome]], dict]]:
    """Split the exec gates into independent runs plus a pure `assemble(outcomes)`.

    `calls` are zero-argument callables (oracle if any, no-op, each wrong answer, each
    alternative, in that order); each is an independent runner call and can run in any
    order or concurrently. `assemble` takes their outcomes in the same order and builds the
    exact result `run_exec_gates` returns, so parallel and sequential runs agree."""
    has_oracle = (task_dir / "solution" / "solve.sh").is_file()
    calls: list[Callable[[], RunOutcome]] = []
    if has_oracle:
        calls.append(lambda: runner.run_solution(task_dir))
    calls.append(lambda: runner.run_verifier(task_dir, {}))
    for w in wrong:
        calls.append(lambda w=w: _try(runner, task_dir, answer_file, w))
    for a in alternatives:
        calls.append(lambda a=a: _try(runner, task_dir, answer_file, a))

    def assemble(outs: list[RunOutcome]) -> dict:
        it = iter(outs)
        detail: dict[str, dict] = {}
        if has_oracle:
            o = next(it)
            detail["oracle"] = {"gate": verdict(o, True), **o.to_dict()}
        else:
            detail["oracle"] = {"gate": "not_run", "detail": "no oracle (oracle_kind none)"}
        n = next(it)
        detail["noop"] = {"gate": verdict(n, False), **n.to_dict()}
        wrongs = []
        for w in wrong:
            r = next(it)
            wrongs.append({**_label(w), "gate": verdict(r, False), **r.to_dict()})
        detail["wrong"] = {
            "gate": _combine([w["gate"] for w in wrongs]) if wrongs else "error",
            "auto": auto_wrong,
            "runs": wrongs,
        }
        alts = []
        for a in alternatives:
            r = next(it)
            alts.append({**_label(a), "gate": verdict(r, True), **r.to_dict()})
        detail["alternatives"] = {
            "gate": _combine([a["gate"] for a in alts]) if alts else "not_run",
            "runs": alts,
        }
        gates = [detail[k]["gate"] for k in ("oracle", "noop", "wrong", "alternatives")]
        if "fail" in gates:
            exec_state = "fail"
        elif "error" in gates:
            exec_state = "error"
        elif detail["oracle"]["gate"] == "not_run":
            exec_state = "no_oracle"
        else:
            exec_state = "pass"
        trials = sorted({x.trial for x in outs if x.trial}) if all(x.trial for x in outs) else []
        return {"exec": exec_state, "detail": detail, "runner": runner.name, "trials": trials}

    return calls, assemble


def run_exec_gates(
    runner: Runner,
    task_dir: Path,
    answer_file: str | None,
    wrong: list,
    alternatives: list,
    auto_wrong: bool,
) -> dict:
    """Run oracle / no-op / wrong outcomes / alternatives. Value = gate satisfied?

    No solution/solve.sh (oracle_kind none): the oracle is not run and, if every other gate
    passes, exec is `no_oracle` (the task can be generated, never validated)."""
    calls, assemble = plan_exec_gates(
        runner, task_dir, answer_file, wrong, alternatives, auto_wrong
    )
    return assemble([c() for c in calls])


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
    "plan_exec_gates",
    "run_exec_gates",
    "derive_wrong",
    "verdict",
    "ANSWER_DIR",
]
