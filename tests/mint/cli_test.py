"""CLI behaviour: --json shape, exit codes, --dry-run, idempotence, next hints."""

import json

from mint import cli
from mint.cli import main

from .conftest import CHECKS_PY, JOBS_JSON, PARSE_PY, TASKTYPE_PY


def test_json_shape_stdout_only_progress_on_stderr(env):
    env.build_pipeline()
    code, out, err, res = env.run("run", "generate", "--json")
    assert code == 0
    assert set(res) == {"ok", "command", "data", "warnings", "next"}
    assert res["ok"] is True and res["command"] == "run" and res["next"]
    assert json.loads(out)  # stdout is exactly one JSON document
    assert "generate addnums" in err  # progress went to stderr


def test_human_output_ends_with_next_hint(env):
    env.build_pipeline()
    code, out, _, _ = env.run("status")
    assert code == 0 and out.strip().splitlines()[-1].startswith("next: ")


def test_every_command_gives_next_hint(env):
    env.build_pipeline()
    env.run("run", "generate")
    for cmd in (
        ["status"],
        ["schema"],
        ["report"],
        ["sample"],
        ["query", "True", "--count"],
        ["task", "show", "t_q1"],
        ["task", "try", "t_q1", "--answer", "42"],
        ["docs"],
    ):
        _, out, _, res = env.run(*cmd, "--json")
        assert res["next"], cmd
        assert env.run(*cmd)[1].strip().splitlines()[-1].startswith("next: "), cmd


def test_exit_codes(env, monkeypatch):
    # 2: bad args / not a workspace / bad config
    assert main(["bogus"]) == 2
    assert main(["status", "--workspace", str(env.tmp / "nowhere")]) == 2
    code, out, err, res = env.run("sample", "-n", "x", "--json", ws=True)
    assert code == 2 and res["ok"] is False and res["error"]["fix"]
    env.init()
    assert env.run("validate", "--level", "9")[0] == 2
    # 1: runtime error with a fix message
    code, out, err, res = env.run("run", "parse", "--json")
    assert code == 1 and "parse.py" in res["error"]["message"] and res["error"]["fix"]
    # 0
    assert env.run("status")[0] == 0
    # 130: interrupted
    monkeypatch.setattr(
        cli.explore, "sample_native", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    code, _, err, _ = env.run("sample")
    assert code == 130 and "status" in err


def test_init_requires_runtime_unless_allowed(env, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    code, _, err, _ = env.run("init", str(env.logs), str(env.ws))
    assert code == 1 and "docker" in err and "fix:" in err
    assert not env.ws.exists()  # stopped before creating anything
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    assert env.run("init", str(env.logs), str(env.ws))[0] == 0
    code, _, err, _ = env.run("init", str(env.tmp / "missing"), str(env.tmp / "ws2"))
    assert code == 2 and "does not exist" in err


def test_init_is_idempotent_and_keeps_user_files(env):
    assert env.init()[0] == 0
    env.write("source_notes.md", "my notes")
    before = env.snapshot()
    code, _, _, res = env.init()
    assert code == 0 and res["data"]["already_initialized"] is True
    assert env.snapshot() == before
    assert (env.ws / "source_notes.md").read_text() == "my notes"
    code, _, err, _ = env.run("init", str(env.tmp), str(env.ws), "--allow-missing-runtime")
    assert code == 1 and "already a workspace" in err


def test_dry_run_writes_nothing(env):
    env.init()
    before = env.snapshot()
    env.write("parse.py", PARSE_PY)
    env.write("checks.py", CHECKS_PY)
    env.write("tasktypes/addnums.py", TASKTYPE_PY)
    env.write("jobs.json", JOBS_JSON)
    base = env.snapshot()
    assert set(base) - set(before) == {
        "ws/parse.py",
        "ws/checks.py",
        "ws/tasktypes/addnums.py",
        "ws/jobs.json",
    }
    for cmd in (
        ["new", "check", "foo"],
        ["new", "parser", "--force"],
        ["run", "parse"],
        ["decide", "x", "--why", "y"],
        ["report"],
        ["sample", "--unseen"],
    ):
        code, out, _, _ = env.run(*cmd, "--dry-run")
        assert code == 0, (cmd, out)
        assert env.snapshot() == base, cmd
    env.run("run", "parse")
    env.run("run", "checks")
    base = env.snapshot()
    for cmd in (["run", "generate"], ["show", "q1", "--attachments"]):
        assert env.run(*cmd, "--dry-run")[0] == 0, cmd
        assert env.snapshot() == base, cmd
    env.run("run", "generate")
    base = env.snapshot()
    assert env.run("validate", "--dry-run")[0] == 0
    assert env.snapshot() == base


def test_reruns_are_idempotent(env):
    env.build_pipeline()
    for cmd in (
        ["run", "generate"],
        ["validate"],
        ["decide", "keep going", "--why", "evidence"],
        ["report"],
    ):
        env.run(*cmd)
        first = env.snapshot()
        code, _, _, _ = env.run(*cmd)
        assert code == 0
        assert env.snapshot() == first, cmd
    # parse reruns too (ignoring cache bookkeeping, outputs are byte-identical)
    a = (env.ws / "out/trajectories.jsonl").read_bytes()
    env.run("run", "parse", "--no-cache")
    assert (env.ws / "out/trajectories.jsonl").read_bytes() == a


def test_parse_is_cached(env):
    env.build_pipeline()
    _, _, _, res = env.run("run", "parse", "--json")
    assert res["data"]["cached"] is True
    _, _, _, res = env.run("run", "parse", "--no-cache", "--json")
    assert "cached" not in res["data"]
    env.write("parse.py", (env.ws / "parse.py").read_text() + "\n# edit\n")
    assert "cached" not in env.run("run", "parse", "--json")[3]["data"]


def test_raw_logs_are_never_modified(env):
    before = {p: p.read_bytes() for p in env.logs.rglob("*") if p.is_file()}
    env.build_pipeline()
    for cmd in (
        ["run", "generate"],
        ["validate"],
        ["sample"],
        ["query", "True"],
        ["show", "a.jsonl#0"],
        ["report"],
    ):
        env.run(*cmd)
    assert {p: p.read_bytes() for p in env.logs.rglob("*") if p.is_file()} == before
    assert sorted(p.name for p in env.logs.iterdir()) == ["a.jsonl", "notes.txt"]


def test_new_scaffolds_and_does_not_clobber(env):
    env.init()
    assert env.run("new", "parser")[0] == 0 and (env.ws / "parse.py").exists()
    assert env.run("new", "parser")[0] == 1  # exists: refuse, change nothing
    assert env.run("new", "check", "loops")[0] == 0
    assert env.run("new", "check", "loops")[0] == 1
    assert env.run("new", "check", "other")[0] == 0
    txt = (env.ws / "checks.py").read_text()
    assert txt.count("def check_") == 2 and 'Check("other"' in txt
    assert env.run("new", "tasktype", "bad name!")[0] in (1, 2)
    assert env.run("new", "tasktype", "qa")[0] == 0
    assert "def applies" in (env.ws / "tasktypes/qa.py").read_text()
    compile((env.ws / "tasktypes/qa.py").read_text(), "qa.py", "exec")


def test_scaffolded_templates_load(env):
    env.init()
    env.run("new", "parser")
    env.run("new", "check", "c1")
    env.run("new", "tasktype", "qa")
    env.run("new", "jobs")
    from mint.output import MINTError
    from mint.scripts import load_checks, load_jobs, load_parser, load_tasktypes
    from mint.workspace import Workspace

    ws = Workspace(env.ws)
    assert callable(load_parser(ws).module.parse)
    assert load_checks(ws)[1][0].id == "c1"
    try:  # the scaffolded job map must be filled in before task types load
        load_jobs(ws)
    except MINTError as e:
        assert "TODO" in e.message
    else:
        raise AssertionError("an unfilled jobs.json must not load")
    # only the scaffold's placeholders are rejected, not user text that says TODO
    todo_app = json.loads(JOBS_JSON)
    todo_app["jobs"][0]["description"] = "build a TODO app from the spec"
    env.write("jobs.json", json.dumps(todo_app))
    assert "main" in load_jobs(ws)
    env.write("jobs.json", JOBS_JSON)
    tt = env.ws / "tasktypes/qa.py"
    try:  # a task type must name a real job
        load_tasktypes(ws)
    except MINTError as e:
        assert "SPEC['job']" in e.message
    else:
        raise AssertionError("a task type without a known job must not load")
    tt.write_text(tt.read_text().replace('"job": "TODO"', '"job": "main"'))
    assert load_tasktypes(ws)[0].generator == "qa@1"
    code, _, _, res = env.run("run", "parse", "--json")
    assert code == 1 and "NotImplementedError" in res["error"]["message"]


def test_schema_and_docs_ship_with_package(env):
    _, _, _, res = env.run("schema", "--json")
    d = res["data"]
    assert "parse.py" in d and "tasktypes/<type>.py" in d and d["exit codes"]["130"]
    _, _, _, res = env.run("docs", "--json")
    assert "mint" in res["data"]["skill"]
    assert {"parsing", "checks", "tasktypes", "validation", "harbor"} <= set(res["data"]["topics"])
    assert "oracle" in env.run("docs", "--topic", "validation")[1]
    assert env.run("docs", "--topic", "nope")[0] == 2
    skill = res["data"]["skill"]
    assert 60 <= len(skill.splitlines()) <= 130
