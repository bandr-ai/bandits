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
