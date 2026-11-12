"""Parallel, resumable validation: same results as one-at-a-time, saved as it goes."""

import json
import threading
import time
from dataclasses import dataclass, field, replace

import pytest

from mint.gates import LocalRunner
from mint.validate import run_validate
from mint.workspace import Workspace

from .conftest import _HarborSim


def rows(env):
    return {
        r["task_id"]: r
        for r in (json.loads(x) for x in (env.ws / "out/results.jsonl").read_text().splitlines())
    }


@dataclass
class Probe(_HarborSim):
    """Harbor stand-in that counts calls, tracks overlap and can die after N calls."""

    die_after: int | None = None
    delay: float = 0.0
    calls: int = 0
    solutions: int = 0
    live: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _enter(self, solution: bool):
        with self.lock:
            if self.die_after is not None and self.calls >= self.die_after:
                raise RuntimeError("runner killed")
            self.calls += 1
            self.solutions += solution
            self.live += 1
            self.peak = max(self.peak, self.live)

    def _leave(self):
        with self.lock:
            self.live -= 1

    def run_solution(self, task_dir):
        self._enter(True)
        try:
            time.sleep(self.delay)
            return super().run_solution(task_dir)
        finally:
            self._leave()

    def run_verifier(self, task_dir, answer):
        self._enter(False)
        try:
            time.sleep(self.delay)
            return super().run_verifier(task_dir, answer)
        finally:
            self._leave()


def validate(env, runner, jobs, sample=3, dry=False):
    return run_validate(Workspace(env.ws), None, sample, None, {2}, [], dry, runner, jobs)


def stable(summary):
    return {k: v for k, v in summary.items() if k not in ("wall_seconds", "jobs")}


def prepared(env):
    env.build_pipeline()
    env.run("run", "generate")
    return env


def test_parallel_matches_sequential(env):
    prepared(env)
    one = validate(env, Probe(), 1, dry=True)
    probe = Probe(delay=0.05)
    many = validate(env, probe, 4, dry=True)
    assert stable(many) == stable(one) and many["jobs"] == 4
    assert probe.peak > 1  # gate runs really overlapped
    assert many["wall_seconds"] >= 0
    assert not (env.ws / "out/results.jsonl").read_text().count("validated")  # dry saved nothing
    seq = Probe(delay=0.01)
    validate(env, seq, 1, dry=True)
    assert seq.peak == 1
    # a real parallel run lands every sampled task as validated, like a serial one
    validate(env, Probe(), 4)
    assert sorted(r["status"] for r in rows(env).values()) == ["validated"] * 3
    assert all(r["gates"]["exec_detail"]["oracle"] == "pass" for r in rows(env).values())


def test_jobs_flag_and_config(env, capsys):
    prepared(env)
    code, out, err, res = env.run("validate", "-j", "2", "--sample-size", "1", "--json")
    assert code == 0 and res["data"]["jobs"] == 2 and "wall_seconds" in res["data"]
    assert "[1/1] addnums/" in err  # per-task progress on stderr
    env.write(
        "mint.json",
        json.dumps({**json.loads((env.ws / "mint.json").read_text()), "validate_jobs": 3}),
    )
    assert env.run("validate", "--sample-size", "1", "--json")[3]["data"]["jobs"] == 3
    code, _, _, res = env.run("validate", "--jobs", "0", "--json")
    assert code != 0 and "jobs" in json.dumps(res)


def test_killed_run_keeps_finished_tasks_and_rerun_does_the_rest(env):
    prepared(env)
    per_task = 4  # oracle, no-op, one wrong, one alternative
    dying = Probe(die_after=2 * per_task)
    with pytest.raises(RuntimeError, match="killed"):
        validate(env, dying, 1)
    saved = [r for r in rows(env).values() if r["gates"]["exec"] == "pass"]
    assert len(saved) == 2
    assert {r["status"] for r in saved} == {"validated"}
    gates_dir = Workspace(env.ws).internal / "gates"
    assert len(list(gates_dir.glob("*.json"))) == 2
    # rerun: the two finished tasks are not run again, only the remaining one
    again = Probe()
    s = validate(env, again, 4)
    assert again.solutions == 1 and again.calls == per_task
    assert sorted(r["status"] for r in rows(env).values()) == ["validated"] * 3
    assert s["types"]["addnums"]["state"] == "passed"


def test_killed_parallel_run_saves_what_finished(env):
    prepared(env)
    with pytest.raises(RuntimeError):
        validate(env, Probe(die_after=5, delay=0.02), 4)
    done = [r for r in rows(env).values() if r["gates"]["exec"] == "pass"]
    assert len(done) <= 1  # at most the one task whose 4 runs all finished
    again = Probe()
    validate(env, again, 4)
    assert sorted(r["status"] for r in rows(env).values()) == ["validated"] * 3
    assert again.solutions == 3 - len(done)


def test_unchanged_packages_are_not_rerun(env):
    prepared(env)
    validate(env, Probe(), 4)
    for args in (("run", "generate"), ("run", "generate")):  # no-op regenerate
        env.run(*args)
    p = Probe()
    s = validate(env, p, 4)
    assert p.calls == 0 and s["types"]["addnums"]["sampled"] == []
    assert not any(r.get("changed") for r in rows(env).values())
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    # a changed package is the only one re-run
    src = (env.ws / "tasktypes/addnums.py").read_text()
    env.write("tasktypes/addnums.py", src.replace("Add up the numbers", "Sum the numbers"))
    env.run("run", "generate")
    p = Probe()
    validate(env, p, 4, sample=0)
    assert p.solutions == 3  # all three instructions changed
    # a different runner's pass does not count
    local = LocalRunner()
    p = Probe()
    validate(env, replace(p, name="harbor"), 2)  # same runner: nothing to do
    assert p.calls == 0
    s = validate(env, local, 2, sample=1)
    assert s["runner"] == "local" and len(s["types"]["addnums"]["sampled"]) == 1


FAULTY = (
    "return BuildResult(files={'tests/test.sh': '#!/bin/bash\\nmkdir -p \"$MINT_REWARD_DIR\"\\n"
    'echo 1 > "$MINT_REWARD_DIR/reward.txt"\\n# /logs/verifier/reward.txt\\n\'},'
)


def test_failure_saved_by_killed_run_is_counted_on_rerun(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py").read_text().replace("return BuildResult(", FAULTY),
    )
    env.run("run", "generate")
    with pytest.raises(RuntimeError):
        validate(env, Probe(die_after=4), 1)  # first task finished (failed), then killed
    assert [r["gates"]["exec"] for r in rows(env).values()].count("fail") == 1
    p = Probe()
    s = validate(env, p, 2)
    t = s["types"]["addnums"]
    assert t["state"] == "failed" and t["attempts"] == 1  # one attempt, not zero, not two
    assert len(t["resumed"]) == 1 and p.solutions == len(t["sampled"]) - 1
    assert {r["status"] for r in rows(env).values()} == {"exec_failed"}
    # rerun without regenerating: saved failures are reused, nothing executes
    p = Probe()
    validate(env, p, 2)
    assert p.calls == 0
