"""Full jobs end to end: custom verifier on the end state, executable wrong outcomes,
no-oracle packages, candidates instead of blocks, and the full-job report."""

import tomllib

from .validate_test import rows

FULL_PY = r'''
from mint import Blocked, BuildResult, Candidate, Outcome

SPEC = {"unit": "episode", "env": "rebuilt", "job": "main",
        "check": "behavioral", "gold": "independent", "hidden": [], "applies_when": "any"}
VERSION = 1

TEST_SH = """#!/bin/bash
# writes /logs/verifier/reward.txt
mkdir -p "${MINT_REWARD_DIR:-/logs/verifier}"
if [ "$(cat out.txt 2>/dev/null)" = "42" ]; then r=1; else r=0; fi
echo $r > "${MINT_REWARD_DIR:-/logs/verifier}/reward.txt"
"""

def applies(t, findings):
    return True

def group_key(t):
    return t["id"]

def build(group, out):
    tid = group[0]["id"]
    files = {"tests/test.sh": TEST_SH,
             "environment/Dockerfile": "FROM ubuntu:24.04\nWORKDIR /app\n"}
    common = dict(instruction="Produce out.txt with the result.", gold_type="independent",
                  wrong_answers=[Outcome("echo 41 > out.txt", note="off by one",
                                         source="synthetic shortcut")],
                  alternatives=[Outcome("printf 42 > out.txt")])
    if tid == "q1":
        return BuildResult(task_id="full_q1", oracle_kind="replay",
                           oracle_source="logged attempt a1, outcome known from its final check",
                           verifier_timeout_sec=600,
                           files={**files, "solution/solve.sh": "#!/bin/bash\necho 42 > out.txt\n"},
                           **common)
    if tid == "q2":
        return BuildResult(task_id="full_q2", oracle_kind="none", env_source="partial",
                           gaps=[{"what": "input file seen only in part", "affects": "fidelity"}],
                           files=files, **common)
    if tid == "q3":
        return Candidate("source archive not in the logs",
                         gaps=[{"what": "source archive", "affects": "execution"}],
                         next_action="fetch the archive the logged run downloaded")
    return Blocked("needs a live system", missing=["service"])
'''


def setup(env):
    env.build_pipeline()
    env.write("tasktypes/addnums.py", FULL_PY)
    _, _, _, res = env.run("run", "generate", "--json")
    return res["data"]


def test_full_job_custom_verifier_outcomes_candidates_and_report(env):
    gen = setup(env)
    r = rows(env)
    assert set(r) == {"full_q1", "full_q2"}

    # package: no answer file, verifier runs in the task environment, provenance recorded
    d1 = env.ws / "out/tasks/addnums/full_q1"
    assert not (d1 / "tests/expected.json").exists()
    toml = tomllib.loads((d1 / "task.toml").read_text())
    assert "artifacts" not in toml and "environment_mode" not in toml["verifier"]
    meta = toml["metadata"]["mint"]
    assert (meta["oracle"], meta["env_source"], meta["grading_method"]) == (
        "replay",
        "rebuilt",
        "behavioral",
    )
    assert meta["job"] == "main" and "task_scope" not in meta
    assert meta["oracle_source"].startswith("logged attempt a1")
    assert toml["verifier"]["timeout_sec"] == 600.0  # its own budget
    d2 = env.ws / "out/tasks/addnums/full_q2"
    assert tomllib.loads((d2 / "task.toml").read_text())["verifier"]["timeout_sec"] == 120.0
    assert not (d2 / "solution/solve.sh").exists()

    # a full job blocked for a non-allowed reason is an error, not a silent block
    queue = (env.ws / "out/queue.jsonl").read_text()
    assert '"kind": "candidate"' in queue and "fetch the archive" in queue
    assert "a job may be Blocked only for" in queue
    assert gen["queued"] == 3  # q3 candidate + two disallowed full-job blocks

    # gates: scripts run in place of solve.sh; no oracle -> no_oracle, stays generated
    env.run("validate", "--sample-size", "5")
    r = rows(env)
    assert r["full_q1"]["gates"]["exec"] == "pass"
    assert r["full_q1"]["status"] == "validated"
    assert r["full_q2"]["gates"]["exec"] == "no_oracle"
    assert r["full_q2"]["status"] == "generated"

    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "## Jobs" in md
    row = next(line for line in md.splitlines() if line.startswith("| main | addnums |"))
    assert "| 2 | 1 | 1 | 1 |" in row  # 2 tasks, 1 validated, 1 ready, 1 candidate
    assert "fidelity: input file seen only in part" in row
    assert "candidate q3: source archive not in the logs" in md
    assert "| 2 | 1 | 1 | 1 | 0 | 2 |" in row  # ...candidates, blocked, errors (2 bad blocks)
    # provenance shown: oracle kind + source, outcomes with their sources
    assert "full_q1: oracle replay, source: logged attempt a1" in md
    assert "full_q2: oracle none, source: none given" in md
    assert "wrong x1 (script 1; 1 source(s)): off by one [synthetic shortcut]" in md
    assert "alternatives x1" in md and "[no source given]" in md
    # runs whose full job errored are not covered: they show up instead of disappearing
    assert "Trajectories with no task or candidate: **2 of 5**" in md


def test_broken_verifier_is_caught_by_an_executable_wrong_outcome(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        FULL_PY.replace('if [ "$(cat out.txt 2>/dev/null)" = "42" ]', "if true"),
    )
    env.run("run", "generate")
    env.run("validate", "--sample-size", "5")
    r = rows(env)["full_q1"]
    assert r["gates"]["exec"] == "fail" and r["status"] == "exec_failed"


def test_outcome_source_round_trips_and_verifier_timeout_is_validated():
    import pytest

    from mint import BuildResult, ContractError, Outcome
    from mint.harbor import resolve, task_toml
    from mint.models import normalize_outcome

    o = Outcome("echo 1", note="n", source="logged attempt")
    assert normalize_outcome(o)["source"] == "logged attempt"
    assert normalize_outcome({"script": "x"})["source"] == ""
    b = BuildResult(task_id="t", files={"tests/test.sh": "#!/bin/bash\n"}, verifier_timeout_sec=0)
    with pytest.raises(ContractError, match="verifier_timeout_sec"):
        resolve(b)
    b.verifier_timeout_sec = 45
    resolve(b)
    assert (
        tomllib.loads(task_toml(b, "ty", "g", [], separate=False))["verifier"]["timeout_sec"]
        == 45.0
    )


def test_ready_needs_strong_trust_for_an_answer_oracle():
    from mint.report import ready

    def row(gold, oracle):
        return {
            "status": "validated",
            "gold": gold,
            "provenance": {"oracle": oracle, "env_source": "evidence", "gaps": []},
        }

    assert ready(row("independent", "reference"))[0]
    assert not ready(row("independent", "answer"))[0]
    assert not ready(row("solved", "answer"))[0]
    assert ready(row("computed", "answer"))[0] and ready(row("verified", "answer"))[0]


def test_report_separates_infra_errors_from_grading_failures():
    from mint.report import _exec_problems

    def g(exec_, **det):
        return {"gates": {"exec": exec_, "exec_detail": det}}

    assert _exec_problems(g("fail", oracle="pass", wrong="fail")) == (["wrong"], [])
    assert _exec_problems(g("error", oracle="error", wrong="pass")) == ([], ["oracle"])
    assert _exec_problems(g("fail", noop="fail", oracle="error")) == (["noop"], ["oracle"])
    assert _exec_problems(g("pass", oracle="pass")) == ([], [])


def test_runtime_env_and_agent_timeout_reach_task_toml():
    import pytest

    from mint import BuildResult, ContractError
    from mint.harbor import resolve, task_toml

    b = BuildResult(
        task_id="t",
        files={"tests/test.sh": "#!/bin/bash\n"},
        runtime_env={
            "USER_SIM_MODEL": "${USER_SIM_MODEL:-}",
            "USER_SIM_API_KEY": "${USER_SIM_API_KEY:-}",
        },
        agent_timeout_sec=3600,
    )
    resolve(b)
    d = tomllib.loads(task_toml(b, "ty", "g", [], separate=False))
    assert d["environment"]["env"]["USER_SIM_API_KEY"] == "${USER_SIM_API_KEY:-}"
    assert d["agent"]["timeout_sec"] == 3600.0
    b.runtime_env = {"USER_SIM_API_KEY": "sk-live-123"}
    with pytest.raises(ContractError, match="host reference"):
        resolve(b)
    b.runtime_env, b.agent_timeout_sec = {}, 0
    with pytest.raises(ContractError, match="agent_timeout_sec"):
        resolve(b)
