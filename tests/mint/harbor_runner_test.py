"""HarborRunner against a fake `harbor` executable that writes result.json in Harbor's
real shape (jobs/<job>/<trial>/result.json). Real Docker runs are out of unit scope."""

import os
import stat
import sys
from pathlib import Path

import pytest

import mint

from .validate_test import rows

FAKE = """#!{py}
import json, os, re, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r})
from mint.gates import LocalRunner

a = sys.argv[1:]
assert a[0] == "run", a
opt = lambda k: a[a.index(k) + 1]
task, agent, jobs, job = Path(opt("-p")), opt("-a"), Path(opt("-o")), opt("--job-name")
mode = os.environ.get("FAKE_HARBOR_MODE", "ok")
open(os.environ["FAKE_HARBOR_LOG"], "a").write(agent + "\\n")
if mode == "crash":
    sys.exit(1)
trial = jobs / job / "trial_1"
trial.mkdir(parents=True)
res = {{"task_name": task.name, "trial_name": "trial_1", "exception_info": None,
        "verifier_result": None}}
if mode == "exception":
    res["exception_info"] = {{"exception_type": "EnvironmentStartTimeoutError",
                              "exception_message": "docker build failed"}}
else:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "t"
        shutil.copytree(task, copy)
        solve = copy / "solution" / "solve.sh"
        if solve.exists():  # the container maps /app/answer; here it is the scratch dir
            solve.write_text(solve.read_text().replace("/app/answer", "${{MINT_ANSWER_DIR}}"))
        r = LocalRunner()
        out = r.run_solution(copy) if agent == "oracle" else r.run_verifier(copy, {{}})
    assert out.state == "ok", out
    res["verifier_result"] = {{"rewards": {{"reward": out.reward}}}}
(trial / "result.json").write_text(json.dumps(res))
"""


@pytest.fixture
def fake_harbor(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "harbor_calls.log"
    log.write_text("")
    root = str(Path(mint.__file__).parent.parent)
    h = bin_dir / "harbor"
    h.write_text(FAKE.format(py=sys.executable, root=root))
    h.chmod(h.stat().st_mode | stat.S_IEXEC)
    d = bin_dir / "docker"
    d.write_text("#!/bin/sh\nexit 0\n")
    d.chmod(d.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_HARBOR_LOG", str(log))
    monkeypatch.setattr("mint.gates.harbor_available", lambda: True)
    return log


def test_harbor_runner_gates_make_tasks_validated(env, fake_harbor):
    env.build_pipeline()
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--sample-size", "3", "--runner", "harbor", "--json")
    assert code == 0, res
    assert res["data"]["runner"] == "harbor"
    rs = rows(env).values()
    assert {r["status"] for r in rs} == {"validated"}
    assert {r["gates"]["runner"] for r in rs} == {"harbor"}
    calls = fake_harbor.read_text().split()
    assert "nop" in calls and "oracle" in calls  # nop gate + oracle (also used for wrong answers)
    # per task: oracle, nop, wrong, alternatives -> one nop, the rest oracle
    assert calls.count("oracle") > calls.count("nop") >= 3


def test_auto_picks_harbor_when_installed(env, fake_harbor):
    env.build_pipeline()
    env.run("run", "generate")
    _, _, _, res = env.run("validate", "--sample-size", "1", "--json")
    assert res["data"]["runner"] == "harbor"


@pytest.mark.parametrize("mode", ["crash", "exception"])
def test_harbor_failures_are_infra_errors_not_task_failures(env, fake_harbor, monkeypatch, mode):
    monkeypatch.setenv("FAKE_HARBOR_MODE", mode)
    env.build_pipeline()
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--sample-size", "3", "--runner", "harbor", "--json")
    assert res["data"]["infra_errors"]
    assert {r["status"] for r in rows(env).values()} == {"generated"}
    assert {r["gates"]["exec"] for r in rows(env).values()} == {"error"}


def test_explicit_harbor_runner_without_harbor_is_a_clear_error(env):
    env.build_pipeline()
    env.run("run", "generate")
    code, _, err, _ = env.run("validate", "--runner", "harbor")
    assert code != 0 and "harbor" in err and "--runner local" in err


def test_auto_without_harbor_falls_back_to_local_and_says_so(env, local_runner):
    env.build_pipeline()
    env.run("run", "generate")
    code, out, _, _ = env.run("validate", "--sample-size", "1")
    assert code == 0 and "host-local runner" in out
    assert {r["status"] for r in rows(env).values()} == {"generated"}


def test_local_passes_do_not_count_when_harbor_is_selected(env, local_runner, fake_harbor):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3", "--runner", "local")
    assert {r["gates"]["runner"] for r in rows(env).values()} == {"local"}
    assert {r["status"] for r in rows(env).values()} == {"generated"}
    _, _, _, st = env.run("status", "--json")
    v = st["data"]["stages"]["5_validate"]
    assert v["done"] is False and v["local_only"] is True  # local-only is not "validation done"
    assert fake_harbor.read_text() == ""
    env.run("validate", "--sample-size", "3", "--runner", "harbor")
    assert fake_harbor.read_text().split()  # Harbor really ran
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    assert {r["gates"]["runner"] for r in rows(env).values()} == {"harbor"}
