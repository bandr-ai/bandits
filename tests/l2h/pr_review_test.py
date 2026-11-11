"""Regression tests for the PR review fixes (validate budget/evidence, explore, status)."""

import json

from .conftest import TASKTYPE_PY
from .validate_test import rows

ALWAYS_PASS = (
    "{'tests/test.sh': '#!/bin/bash\\nmkdir -p \"$L2H_REWARD_DIR\"\\n"
    'echo 1 > "$L2H_REWARD_DIR/reward.txt"\\n# /logs/verifier/reward.txt\\n\'}'
)


def one_bad_verifier(env):
    """q2's verifier accepts anything; q1 and q3 are fine."""
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace(
            "return BuildResult(",
            f"return BuildResult(files={ALWAYS_PASS} if t['id'] == 'q2' else {{}},",
        ),
    )
    env.run("run", "generate")


def test_rerun_after_failed_sample_is_not_passed_and_keeps_budget(env):
    one_bad_verifier(env)
    code, _, _, res = env.run("validate", "--sample-size", "3", "--json")
    t = res["data"]["types"]["addnums"]
    assert code == 1 and t["state"] == "failed" and t["attempts"] == 1
    code, _, _, res = env.run("validate", "--sample-size", "3", "--json")
    t = res["data"]["types"]["addnums"]
    assert code == 1 and t["state"] == "failed" and t["attempts"] == 1  # not reset, not "passed"
    st = {r["status"] for r in rows(env).values()}
    assert "blocked" in st


def test_changed_package_reruns_leak_even_with_level_2_only(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    ip = env.ws / "out/tasks/addnums/t_q1/instruction.md"
    ip.write_text(ip.read_text() + "\nanswer: 42\n")
    code, _, _, res = env.run("validate", "--level", "2", "--sample-size", "0", "--json")
    r = rows(env)["t_q1"]
    assert r["gates"]["leak"] == "fail" and r["status"] == "blocked"
    assert r["gates"]["exec"] == "not_run"
    assert "t_q1" not in res["data"]["types"]["addnums"]["sampled"]


def test_exhausted_type_does_not_abort_other_types(env):
    env.build_pipeline()
    bad = TASKTYPE_PY.replace("return BuildResult(", f"return BuildResult(files={ALWAYS_PASS},")
    env.write("tasktypes/addnums.py", bad)
    env.write("tasktypes/other.py", TASKTYPE_PY.replace('"t_" + t["id"]', '"o_" + t["id"]'))
    env.run("run", "generate")
    assert env.run("validate", "--max-fix-attempts", "1")[0] == 1  # addnums: attempt 1/1
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--max-fix-attempts", "1", "--json")
    d = res["data"]
    assert code == 1 and d["any_failure"]
    assert d["types"]["addnums"]["state"] == "exhausted"
    assert d["types"]["other"]["state"] == "passed"  # still validated, results written
    assert {r["status"] for r in rows(env).values() if r["type"] == "other"} == {"validated"}


def test_status_ignores_underscore_helpers_in_tasktypes(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.write("tasktypes/_common.py", "X = 1\n")
    _, _, _, res = env.run("status", "--json")
    s = res["data"]["stages"]
    assert s["3_task_types"]["types"] == ["addnums"]
    assert not s["4_generate"]["stale"]


def test_attachment_ids_cannot_escape_attachments_dir(env):
    env.build_pipeline()
    traj = {
        "id": "..",
        "source_refs": [{"file": "a.jsonl", "offset": 0}],
        "events": [],
        "meta": {},
        "ended": "done",
        "attachments": [{"name": "x", "text": "t"}, {"name": "..", "text": "u"}],
    }
    (env.ws / "out/trajectories.jsonl").write_text(json.dumps(traj) + "\n")
    code, *_ = env.run("show", "..", "--attachments")
    assert code == 0
    assert not (env.ws / "out/x").exists() and not (env.ws / "x").exists()
    inside = {
        p.relative_to(env.ws / "out/attachments").parts[0]
        for p in (env.ws / "out/attachments").rglob("*")
    }
    assert inside == {"_.."}


def test_query_expression_sees_r_inside_comprehensions(env):
    env.build_pipeline()
    code, _, _, res = env.run(
        "query",
        "any(e['role'] == 'user' and len(r.events) == 2 for e in r.events)",
        "--count",
        "--json",
    )
    assert code == 0 and res["data"]["matched"] == 5 and "expression_errors" not in res["data"]
    code, _, _, res = env.run(
        "query", "True", "--group-by", "len([e for e in r.events if len(r.events)])", "--json"
    )
    assert res["data"]["groups"] == {"2": 5} or list(res["data"]["groups"]) == ["2"]


def test_show_field_index_out_of_range_is_a_clean_error(env):
    env.build_pipeline()
    code, _, _, res = env.run("show", "q1", "--field", "events.5", "--json")
    assert code == 1 and "no field 'events.5'" in res["error"]["message"]


def test_invalid_grep_regex_is_a_usage_error(env):
    env.build_pipeline()
    code, _, _, res = env.run("query", "True", "--grep", "(unclosed", "--json")
    assert code == 2 and "bad --grep regex" in res["error"]["message"]


def test_status_and_report_show_gold_next_to_status_and_warn_when_unaudited(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "2")
    _, out, _, res = env.run("status", "--json")
    assert res["data"]["stages"]["5_validate"]["by_status_gold"] == {
        "validated (imitation)": 2,
        "generated (imitation)": 1,
    }
    _, out, _, _ = env.run("status")
    assert "validated (imitation)=2" in out and "does not mean the answer is correct" in out
    assert "warning: 2 validated task(s) have no audit/derivability decision" in out
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "- validated (imitation): 2" in md and "does not mean the answer is correct" in md
    assert (
        "**WARNING**" in md
        and "Validated tasks without a derivability/audit decision: 2 of 2" in md
    )
    done = sorted(t for t, r in rows(env).items() if r["status"] == "validated")
    env.run(
        "decide",
        "derivable",
        "--why",
        "read it",
        "--kind",
        "derivability",
        "--task",
        done[0],
        "--task",
        done[1],
    )
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "without a derivability/audit decision: 0 of 2" in md and "**WARNING**" not in md
