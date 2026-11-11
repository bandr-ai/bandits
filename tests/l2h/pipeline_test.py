"""Script contracts and the end-to-end flow on the synthetic corpus."""

import json

from l2h.stages import cause_of, group_errors


def test_parse_contract_roundtrip_and_error_grouping(env):
    env.build_pipeline()
    traj = [json.loads(x) for x in (env.ws / "out/trajectories.jsonl").read_text().splitlines()]
    assert [t["id"] for t in traj] == ["q1", "q2", "q3", "q4", "q6"]
    t = traj[0]
    assert set(t) >= {"id", "source_refs", "events", "attachments", "meta", "ended"}
    assert t["source_refs"] == [{"file": "a.jsonl", "offset": 0}]
    assert traj[3]["ended"] == "error"
    code, _, _, res = env.run("run", "parse", "--no-cache", "--json")
    d = res["data"]
    assert d["parse_errors"] == 2  # bad json line + record without prompt
    # two distinct messages, grouped by cause (digits normalised), examples kept
    causes = {g["cause"]: g for g in d["error_groups"]}
    assert causes["bad json on record N"]["examples"] == ["a.jsonl#4"]
    # heartbeat record, notes.txt and the two unparseable records (so 4) land in no trajectory
    assert d["records_in_no_trajectory"] == 4
    assert "notes.txt#0" in d["orphan_examples"]


def test_error_grouping_unit():
    g = group_errors([("bad 'x1' on 12", "a"), ("bad 'y' on 99", "b"), ("other", "c")])
    assert g[0]["count"] == 2 and g[0]["examples"] == ["a", "b"]
    assert cause_of("id deadbeefdeadbeef at 5") == "id <hex> at N"


def test_checks_findings_and_crash_isolated(env):
    env.build_pipeline()
    rows = [json.loads(x) for x in (env.ws / "out/findings.jsonl").read_text().splitlines()]
    assert len(rows) == 5 * 2  # every record x every check
    boom = [r for r in rows if r["check"] == "boom"]
    assert {r["result"] for r in boom} == {"unknown"}
    assert "check crashed" in boom[0]["note"]
    ha = {r["trajectory_id"]: r["result"] for r in rows if r["check"] == "has_answer"}
    assert ha["q4"] == "fail" and ha["q1"] == "pass"


def test_generate_build_blocked_exception_and_exclusion(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py").read_text().replace('"BLOCK"', '"BLOCK"'),
    )
    code, _, _, res = env.run("run", "generate", "--json")
    assert code == 0
    d = res["data"]
    assert d["by_status"] == {"generated": 3} and d["queued"] == 1
    q = [json.loads(x) for x in (env.ws / "out/queue.jsonl").read_text().splitlines()]
    assert q[0]["kind"] == "blocked" and q[0]["missing"] == ["running service"]
    ex = [json.loads(x) for x in (env.ws / "out/excluded.jsonl").read_text().splitlines()]
    assert ex[0]["trajectory_id"] == "q4"
    assert "no logged answer" in ex[0]["reason"]
    rows = [json.loads(x) for x in (env.ws / "out/results.jsonl").read_text().splitlines()]
    assert (
        rows[0]["sources"]
        and rows[0]["generator"] == "addnums@1"
        and rows[0]["gold"] == "imitation"
    )
    assert rows[0]["gates"] == {
        "structure": "pass",
        "leak": "pass",
        "secrets": "pass",
        "exec": "not_run",
    }


def test_builder_exception_is_queued_not_fatal(env):
    env.build_pipeline()
    # q6 now crashes instead of blocking
    src = (env.ws / "tasktypes/addnums.py").read_text()
    env.write(
        "tasktypes/addnums.py",
        src.replace('if nums == "BLOCK":', 'if nums == "NEVER":').replace('"CRASH"', '"BLOCK"'),
    )
    code, _, _, res = env.run("run", "generate", "--json")
    assert code == 0
    q = [json.loads(x) for x in (env.ws / "out/queue.jsonl").read_text().splitlines()]
    assert q[0]["kind"] == "exception" and "generator bug" in q[0]["reason"]


def test_package_matches_harbor_layout(env):
    env.build_pipeline()
    env.run("run", "generate", "--json")
    d = env.ws / "out/tasks/addnums/t_q1"
    for f in [
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/data/nums.txt",
        "tests/test.sh",
        "tests/Dockerfile",
        "tests/grade.py",
        "tests/expected.json",
        "solution/solve.sh",
    ]:
        assert (d / f).is_file(), f
    toml = (d / "task.toml").read_text()
    assert 'environment_mode = "separate"' in toml and 'artifacts = ["/app/answer"]' in toml
    assert "/logs/verifier/reward.txt" in (d / "tests/test.sh").read_text()
    assert "/app/answer/answer.txt" in (d / "instruction.md").read_text()
    assert "42" not in (d / "instruction.md").read_text()
    assert "42" not in (d / "environment/data/nums.txt").read_text()
    import tomllib

    cfg = tomllib.loads(toml)
    assert cfg["task"]["name"] == "l2h/addnums-t_q1"
    assert cfg["metadata"]["l2h"]["sources"] == ["a.jsonl#0"]  # provenance in task.toml only
    assert (d / "tests/test.sh").stat().st_mode & 0o111
