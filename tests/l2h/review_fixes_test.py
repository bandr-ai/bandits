"""Regression tests: local runs aren't `validated`, limited runs don't prune,
partial parses aren't cached as complete, edited packages lose exec evidence."""

import json

from .validate_test import rows


def test_local_runner_passes_stay_generated(env, local_runner):
    env.build_pipeline()
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--sample-size", "3", "--json")
    assert code == 0, res
    rs = rows(env).values()
    assert {r["status"] for r in rs} == {"generated"}
    assert {r["gates"]["exec"] for r in rs} == {"pass"}
    assert {r["gates"]["runner"] for r in rs} == {"local"}


def test_limited_generate_keeps_other_tasks(env):
    env.build_pipeline()
    env.run("run", "generate")
    before = set(rows(env))
    assert len(before) == 3
    code, *_ = env.run("run", "generate", "--limit", "1")
    assert code == 0
    assert set(rows(env)) == before
    for tid in before:
        assert (env.ws / "out/tasks/addnums" / tid).is_dir()


def test_limited_parse_is_not_reused_as_complete(env):
    env.build_pipeline()
    env.run("run", "parse", "--limit", "1", "--no-cache")
    code, _, _, res = env.run("run", "parse", "--json")
    assert code == 0
    assert res["data"].get("cached") is not True
    assert res["data"]["trajectories"] > 1


def test_editing_a_generated_task_invalidates_exec_evidence(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    tid = sorted(rows(env))[0]
    df = env.ws / "out/tasks/addnums" / tid / "environment/Dockerfile"
    df.write_text(df.read_text() + "\n# edited\n")
    env.run("validate", "--level", "1")
    r = rows(env)[tid]
    assert r["gates"]["exec"] == "not_run" and r["status"] == "generated"
    code, _, _, res = env.run("validate", "--sample-size", "0", "--json")
    assert tid in res["data"]["types"]["addnums"]["sampled"]
    assert rows(env)[tid]["gates"]["exec"] == "pass"


def test_limited_type_run_keeps_unrelated_queue_entries(env):
    env.build_pipeline()
    env.run("run", "generate")

    def q():
        return [json.loads(x) for x in (env.ws / "out/queue.jsonl").read_text().splitlines()]

    before = q()
    assert before
    env.run("run", "generate", "--type", "addnums", "--limit", "1")
    assert {x["group"] for x in before} <= {x["group"] for x in q()}


def test_level2_only_run_rechecks_package_freshness(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    tid = sorted(rows(env))[0]
    t = env.ws / "out/tasks/addnums" / tid / "tests/test.sh"
    t.write_text(t.read_text() + "\n# edited\n")
    code, _, _, res = env.run("validate", "--level", "2", "--sample-size", "3", "--json")
    assert tid in res["data"]["types"]["addnums"]["sampled"]
    assert rows(env)[tid]["gates"]["exec"] == "pass"


def _read(env, rel):
    f = env.ws / rel
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


GROUPED = """
from l2h import Answer, Blocked, BuildResult
SPEC = {"unit": "episode", "env": "static", "check": "numeric", "gold": "imitation",
        "hidden": ["answer"], "applies_when": "has an answer"}
VERSION = 1

def applies(t, findings):
    return findings["has_answer"]["result"] == "pass"

def group_key(t):
    return "BLOCK" if t["meta"]["nums"] == "BLOCK" else "ALL"

def build(group, out):
    if group[0]["meta"]["nums"] == "BLOCK":
        return Blocked("needs a live system", missing=["running service"])
    return BuildResult(
        task_id="t_group",
        instruction="Add up the numbers in /app/nums.txt.",
        evidence={"nums.txt": "\\n".join(t["meta"]["nums"] for t in group)},
        answer=Answer(expected=group[0]["meta"]["answer"], check="numeric", tolerance=0.01),
        wrong_answers=["-1"],
    )
"""


def test_limited_run_rebuilds_the_whole_group(env):
    from l2h.stages import package_hash

    env.build_pipeline()
    env.write("tasktypes/addnums.py", GROUPED)
    env.run("run", "generate")
    d = env.ws / "out/tasks/addnums/t_group"
    full = package_hash(d)
    row = rows(env)["t_group"]
    assert len(row["sources"]) > 1
    env.run("run", "generate", "--limit", "1")
    assert package_hash(d) == full  # not rebuilt from the one selected member
    assert rows(env)["t_group"]["sources"] == row["sources"]


def test_failed_rebuild_drops_the_stale_row(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    assert rows(env)["t_q1"]["status"] == "validated"
    src = (env.ws / "tasktypes/addnums.py").read_text()
    env.write(
        "tasktypes/addnums.py", src.replace('nums == "BLOCK"', 'nums in ("BLOCK", "17\\n25")')
    )
    env.run("run", "generate", "--limit", "1")
    assert "t_q1" not in rows(env)
    assert not (env.ws / "out/tasks/addnums/t_q1").exists()
    assert any("q1" in q["trajectory_ids"] for q in _read(env, "out/queue.jsonl"))
    assert {"t_q2", "t_q3"} <= set(rows(env))  # unrelated tasks untouched


def test_id_collision_with_a_retained_type_is_rejected(env):
    from .conftest import TASKTYPE_PY

    env.build_pipeline()
    env.run("run", "generate")
    before = rows(env)
    env.write("tasktypes/other.py", TASKTYPE_PY)  # same task ids as addnums
    code, _, err, _ = env.run("run", "generate", "--type", "other", "--limit", "1")
    assert code != 0 and "already owned by another type" in err
    assert rows(env) == before


def test_type_filtered_limited_run_keeps_exclusions_and_other_types_queue(env):
    from .conftest import TASKTYPE_PY

    env.build_pipeline()
    env.write("tasktypes/other.py", TASKTYPE_PY.replace('"t_"', '"o_"'))
    env.run("run", "generate")
    ex, q = _read(env, "out/excluded.jsonl"), _read(env, "out/queue.jsonl")
    assert ex and {x["type"] for x in q} == {"addnums", "other"}
    env.run("run", "generate", "--type", "addnums", "--limit", "100")
    assert _read(env, "out/excluded.jsonl") == ex
    assert {x["type"] for x in _read(env, "out/queue.jsonl")} == {"addnums", "other"}


def test_reused_row_gets_fresh_leak_result(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    src = (env.ws / "tasktypes/addnums.py").read_text()
    # same package bytes, but the generator now declares the instruction text as hidden
    env.write(
        "tasktypes/addnums.py",
        src.replace(
            "assumptions=",
            'hidden=["Add up the numbers in /app/nums.txt"], assumptions=',
        ),
    )
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"blocked"}
    assert all(r["gates"]["leak"] == "fail" for r in rows(env).values())


def test_schema_documents_both_runners(env):
    env.init()
    _, out, _, _ = env.run("schema")
    assert "harbor" in out and "local" in out
