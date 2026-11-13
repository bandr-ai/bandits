import json

from mint.validate import failure_causes

from .conftest import TASKTYPE_PY
from .secrets_test import FAKE
from .validate_test import rows


def test_blocked_exports_and_job_counts_survive_rescan_and_regeneration(env):
    env.build_pipeline()
    unsafe = TASKTYPE_PY.replace("instruction=instr,", f"instruction=instr + {FAKE['sk']!r},")
    env.write("tasktypes/addnums.py", unsafe)
    assert env.run("run", "generate")[0] == 0
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    assert not list((env.ws / "out/tasks").rglob("instruction.md"))
    assert FAKE["sk"] not in (env.ws / "out/results.jsonl").read_text()
    assert env.run("validate", "--level", "1")[0] != 0
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    env.run("report")
    report = (env.ws / "report.md").read_text()
    job = next(line for line in report.splitlines() if line.startswith("| main | addnums |"))
    assert job.split("|")[7].strip() == "3"  # blocked column
    assert "Trajectories with no task, candidate, or recorded exclusion: **0 of 5**" in report
    assert "unaccounted=0" in report
    env.write("tasktypes/addnums.py", TASKTYPE_PY)
    assert env.run("run", "generate")[0] == 0
    assert {r["status"] for r in rows(env).values()} == {"generated"}
    assert all(not r.get("withheld") for r in rows(env).values())
    assert len(list((env.ws / "out/tasks").rglob("instruction.md"))) == 3


def test_secret_added_after_generation_is_removed(env):
    env.build_pipeline()
    env.run("run", "generate")
    folder = env.ws / "out/tasks/addnums/t_q1"
    instruction = folder / "instruction.md"
    instruction.write_text(instruction.read_text() + "\n" + FAKE["sk"])
    env.run("validate", "--level", "1")
    assert not folder.exists()
    assert rows(env)["t_q1"]["status"] == "blocked"
    env.run("validate", "--level", "1")
    assert rows(env)["t_q1"]["status"] == "blocked"


def test_status_does_not_complete_optional_findings_before_running(env):
    env.init()
    _, _, _, result = env.run("status", "--json")
    stage = result["data"]["stages"]["2_findings"]
    assert not stage["done"] and stage["optional"]


def test_failed_wrong_answer_is_identified():
    result = {
        "detail": {
            "wrong": {
                "gate": "fail",
                "runs": [
                    {"gate": "pass", "answer": "rejected"},
                    {"gate": "fail", "answer": "accepted shortcut"},
                ],
            }
        }
    }
    message = json.dumps(failure_causes([("task1", result)]))
    assert "accepted shortcut" in message
    assert '"rejected"' not in message
