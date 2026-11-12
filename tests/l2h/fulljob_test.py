"""Full jobs end to end: custom verifier on the end state, executable wrong outcomes,
no-oracle packages, candidates instead of blocks, and the full-job report."""

import tomllib

from .validate_test import rows

FULL_PY = r'''
from l2h import Blocked, BuildResult, Candidate, Outcome

SPEC = {"unit": "episode", "env": "rebuilt", "job": "main", "task_scope": "full_job",
        "check": "behavioral", "gold": "independent", "hidden": [], "applies_when": "any"}
VERSION = 1

TEST_SH = """#!/bin/bash
# writes /logs/verifier/reward.txt
mkdir -p "${L2H_REWARD_DIR:-/logs/verifier}"
if [ "$(cat out.txt 2>/dev/null)" = "42" ]; then r=1; else r=0; fi
echo $r > "${L2H_REWARD_DIR:-/logs/verifier}/reward.txt"
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
                  wrong_answers=[Outcome("echo 41 > out.txt", note="off by one")],
                  alternatives=[Outcome("printf 42 > out.txt")])
    if tid == "q1":
        return BuildResult(task_id="full_q1", oracle_kind="replay",
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
    meta = toml["metadata"]["l2h"]
    assert (meta["oracle"], meta["env_source"], meta["grading_method"]) == (
        "replay",
        "rebuilt",
        "behavioral",
    )
    assert (meta["job"], meta["task_scope"]) == ("main", "full_job")
    d2 = env.ws / "out/tasks/addnums/full_q2"
    assert not (d2 / "solution/solve.sh").exists()

    # a full job blocked for a non-allowed reason is an error, not a silent block
    queue = (env.ws / "out/queue.jsonl").read_text()
    assert '"kind": "candidate"' in queue and "fetch the archive" in queue
    assert "a full job may be Blocked only for" in queue
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
    assert "## Full jobs" in md
    row = next(line for line in md.splitlines() if line.startswith("| main | addnums |"))
    assert "| 2 | 1 | 1 | 1 |" in row  # 2 tasks, 1 validated, 1 ready, 1 candidate
    assert "fidelity: input file seen only in part" in row
    assert "candidate q3: source archive not in the logs" in md
    assert "| 2 | 1 | 1 | 1 | 0 | 2 |" in row  # ...candidates, blocked, errors (2 bad blocks)
    # runs whose full job errored are not covered: they show up instead of disappearing
    assert "Trajectories with no full-job task or candidate: **2 of 5**" in md


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
