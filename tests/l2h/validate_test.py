"""validate levels, status transitions, attempts, decisions, report, task commands."""

import json

from .conftest import CHECKS_PY, PARSE_PY, TASKTYPE_PY


def rows(env):
    return {
        r["task_id"]: r
        for r in (json.loads(x) for x in (env.ws / "out/results.jsonl").read_text().splitlines())
    }


def test_generate_then_validate_promotes_to_validated(env):
    env.build_pipeline()
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"generated"}
    code, _, _, res = env.run("validate", "--sample-size", "2", "--json")
    assert code == 0, res
    t = res["data"]["types"]["addnums"]
    assert t["state"] == "passed" and len(t["sampled"]) == 2
    st = {tid: r["status"] for tid, r in rows(env).items()}
    assert sorted(st.values()) == ["generated", "validated", "validated"]
    r = next(r for r in rows(env).values() if r["status"] == "validated")
    assert r["gates"]["exec"] == "pass" and r["gates"]["runner"] == "local"
    assert r["gates"]["exec_detail"] == {
        "oracle": "pass",
        "noop": "pass",
        "wrong": "pass",
        "alternatives": "pass",
    }
    # the sample is seeded: a second environment picks the same tasks
    assert res["data"]["review_for_derivability"]


def test_validate_is_seeded_and_incremental(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "1")
    first = {t for t, r in rows(env).items() if r["status"] == "validated"}
    assert len(first) == 1
    code, _, _, res = env.run("validate", "--sample-size", "1", "--json")
    assert {
        t for t, r in rows(env).items() if r["status"] == "validated"
    } == first  # nothing more needed
    env.run("validate", "--sample-size", "3")
    assert {r["status"] for r in rows(env).values()} == {"validated"}


def test_generate_preserves_validation_when_unchanged_and_resets_on_change(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    # change the instruction wording -> new content hash -> must be revalidated, flagged changed
    src = (env.ws / "tasktypes/addnums.py").read_text()
    env.write(
        "tasktypes/addnums.py",
        src.replace("Add up the numbers", "Sum the numbers").replace("VERSION = 1", "VERSION = 2"),
    )
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"generated"}
    assert all(
        r["gates"]["exec"] == "not_run" and r["generator"] == "addnums@2"
        for r in rows(env).values()
    )
    code, _, _, res = env.run("validate", "--sample-size", "0", "--json")
    assert (
        len(res["data"]["types"]["addnums"]["sampled"]) == 3
    )  # changed instances are always validated


def test_leak_blocks_task_at_generate_and_validate_cannot_rescue_it(env):
    env.build_pipeline(leak=True)
    code, _, _, res = env.run("run", "generate", "--json")
    d = res["data"]
    assert d["by_status"] == {"blocked": 3}
    assert d["structure_or_leak_failures"][0]["problems"][0].startswith(
        "leak: hidden_string"
    ) or "leak" in " ".join(d["structure_or_leak_failures"][0]["problems"])
    code, _, _, res = env.run("validate", "--json")
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    assert all(
        r["gates"]["exec"] == "not_run" for r in rows(env).values()
    )  # no exec on leaky tasks
    _, out, _, _ = env.run("task", "show", "t_q1")
    assert "LEAKS FOUND" in out


def test_verifier_fault_blocks_type_with_cause_and_attempt_budget(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace(
            "return BuildResult(",
            "return BuildResult(files={'tests/test.sh': '#!/bin/bash\\nmkdir -p \"$L2H_REWARD_DIR\"\\necho 1 > \"$L2H_REWARD_DIR/reward.txt\"\\n# /logs/verifier/reward.txt\\n'},",
        ),
    )
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--max-fix-attempts", "2", "--json")
    assert code == 1
    t = res["data"]["types"]["addnums"]
    assert t["state"] == "failed" and t["attempts"] == 1
    causes = " ".join(c["cause"] for c in t["causes"])
    assert "no-op answer was accepted" in causes and "wrong answer was accepted" in causes
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    assert "attempt 1/2" in next(iter(rows(env).values()))["reason"]
    # regenerate (unchanged generator) retries: blocked rows are re-evaluated, not preserved
    env.run("run", "generate")
    assert env.run("validate", "--max-fix-attempts", "2")[0] == 1  # attempt 2
    env.run("run", "generate")
    code, _, err, res = env.run("validate", "--max-fix-attempts", "2", "--json")
    assert code == 1 and "fix attempts" in res["error"]["message"]  # budget spent: stop and report
    assert "decide" in res["error"]["fix"]


def test_infra_error_is_error_not_failure(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace(
            "return BuildResult(",
            "return BuildResult(files={'tests/test.sh': '#!/bin/bash\\n# /logs/verifier/reward.txt\\nexit 3\\n'},",
        ),
    )
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--json")
    d = res["data"]
    assert code == 1 and d["any_infra_error"] and not d["any_failure"]
    assert d["types"]["addnums"]["state"] == "infra_error"
    assert {r["status"] for r in rows(env).values()} == {"generated"}  # not blocked, not validated
    assert {r["gates"]["exec"] for r in rows(env).values()} <= {"error", "not_run"}
    assert any("INFRA" in c["cause"] for c in d["types"]["addnums"]["causes"])
    _, out, _, _ = env.run("status")
    assert "infrastructure errors" in out


def test_level3_audit_fail_sends_type_back(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    _, _, _, res = env.run("validate", "--level", "3", "--json")
    assert len(res["data"]["level3_audit_sample"]["addnums"]) == 3
    code, _, _, res = env.run(
        "decide",
        "instruction ambiguous",
        "--why",
        "t_q1 says add but nums are mixed units",
        "--kind",
        "audit",
        "--verdict",
        "fail",
        "--type",
        "addnums",
        "--json",
    )
    assert code == 0 and res["data"]["effects"]
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    assert "batch audit failed" in next(iter(rows(env).values()))["reason"]
    assert env.run("decide", "x", "--why", "y", "--kind", "audit", "--verdict", "fail")[0] == 2


def test_decide_dedupes_and_report_lists_everything(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "1")
    for _ in range(2):
        env.run("decide", "drop q4", "--why", "no logged answer", "--kind", "exclusion")
    assert len((env.ws / "decisions.jsonl").read_text().splitlines()) == 1
    env.run(
        "decide",
        "has_answer fine",
        "--why",
        "read 5 fresh matches",
        "--kind",
        "spot_check",
        "--check",
        "has_answer",
        "--verdict",
        "pass",
    )
    assert env.run("report")[0] == 0
    md = (env.ws / "report.md").read_text()
    for needle in [
        "raw records that landed in no trajectory: 4",
        "has_answer",
        "boom",
        "| addnums | 3 | 1 | 2 | 0 |",
        "imitation",
        "needs a live system",
        "drop q4",
        "no logged answer",
        "out/tasks/addnums/",
        "unaccounted=0",
        "used=3, excluded=1, blocked=1",
    ]:
        assert needle in md, needle
    assert "| has_answer | 1 | 4 | 1 | 0 | 20.0% | yes |" in md
    assert "| boom | 1 | 0 | 0 | 5 | 0.0% | **NO** |" in md  # unchecked check flagged
    assert "check error x" in md


def test_task_show_and_try(env):
    env.build_pipeline()
    env.run("run", "generate")
    code, _, _, res = env.run("task", "show", "t_q1", "--json")
    s = res["data"]["solver_sees"]
    assert "instruction.md" in s and "environment/data/nums.txt" in [
        f["path"] for f in s["environment"]
    ]
    assert "tests/" in res["data"]["hidden_from_solver"] and res["data"]["leaks"] == []
    assert "42" not in json.dumps(s)
    assert env.run("task", "try", "t_q1", "--answer", "42", "--json")[3]["data"]["reward"] == 1.0
    assert env.run("task", "try", "t_q1", "--answer", "41", "--json")[3]["data"]["reward"] == 0.0
    assert env.run("task", "try", "t_q1", "--json")[3]["data"]["reward"] == 0.0  # no-op
    code, _, _, res = env.run("task", "try", "nope", "--answer", "1", "--json")
    assert code == 1 and "unknown task" in res["error"]["message"]


def test_task_try_infra_error_exit_1(env):
    env.build_pipeline()
    env.run("run", "generate")
    (env.ws / "out/tasks/addnums/t_q1/tests/test.sh").write_text("#!/bin/bash\nexit 7\n")
    code, _, _, res = env.run("task", "try", "t_q1", "--answer", "42", "--json")
    assert code == 1 and "infrastructure" in res["error"]["message"]


def test_status_tracks_stages_staleness_and_resumes(env):
    code, _, _, res = env.init()
    _, _, _, res = env.run("status", "--json")
    assert res["next"] == ["l2h sample"]
    env.write("parse.py", PARSE_PY)
    assert "l2h run parse" in env.run("status", "--json")[3]["next"][0]
    env.write("checks.py", CHECKS_PY)
    env.write("tasktypes/addnums.py", TASKTYPE_PY)
    env.write("source_notes.md", "a record is one jsonl line")
    env.run("run", "parse")
    env.run("run", "checks")
    env.run("run", "generate")
    _, _, _, res = env.run("status", "--json")
    s = res["data"]["stages"]
    assert s["1_understand"]["done"] and s["2_findings"]["done"]
    # editing parse.py makes the parse stage stale until re-run
    env.write("parse.py", (env.ws / "parse.py").read_text() + "\n# tweak\n")
    s = env.run("status", "--json")[3]["data"]["stages"]
    assert s["1_understand"]["stale"] and not s["1_understand"]["done"]
    assert env.run("status", "--json")[3]["next"] == ["l2h run parse"]


def test_run_generate_type_filter_keeps_other_types(env):
    env.build_pipeline()
    env.write("tasktypes/other.py", TASKTYPE_PY.replace('"t_" + t["id"]', '"o_" + t["id"]'))
    env.run("run", "generate")
    assert {r["type"] for r in rows(env).values()} == {"addnums", "other"}
    env.run("run", "generate", "--type", "other")
    assert {r["type"] for r in rows(env).values()} == {"addnums", "other"}
    assert env.run("run", "generate", "--type", "nope")[0] == 1
