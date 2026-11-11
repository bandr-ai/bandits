"""Parsed trajectories are ATIF documents plus an extra['l2h'] block (synthetic data only)."""

import json
from pathlib import Path

import pytest

from l2h import ContractError, Trajectory, message_text
from l2h.models import ATIF_VERSION, normalize_trajectory

DOC = {
    "agent": {"name": "bot", "version": "unknown", "model_name": "m1"},
    "steps": [
        {"source": "user", "message": "add 2 and 3"},
        {
            "source": "agent",
            "message": "calling",
            "tool_calls": [
                {"tool_call_id": "c1", "function_name": "calc", "arguments": {"x": 2}},
                {"tool_call_id": "c2", "function_name": "calc", "arguments": {"x": 3}},
            ],
            "observation": {
                "results": [
                    {"source_call_id": "c2", "content": [{"type": "text", "text": "3"}]},
                    {"source_call_id": "c1", "content": "2"},
                ]
            },
        },
        {"source": "agent", "message": "5"},
    ],
}
REFS = [{"file": "a.jsonl", "offset": 0}]


def make(doc=None, **kw):
    return Trajectory.from_atif(json.loads(json.dumps(doc or DOC)), id="s1", source_refs=REFS, **kw)


def test_from_atif_adds_block_version_and_step_ids():
    t = make(ended="done", meta={"k": 1})
    assert t["schema_version"] == ATIF_VERSION and [s["step_id"] for s in t["steps"]] == [1, 2, 3]
    assert t["extra"]["l2h"]["id"] == "s1"
    assert (t["id"], t.meta, t.ended, t["source_refs"], t.attachments) == (
        "s1",
        {"k": 1},
        "done",
        REFS,
        [],
    )
    assert "id" not in dict(t) and t.get("nope") is None  # l2h fields live under extra only
    assert t.flat()["meta"] == {"k": 1} and t.flat()["steps"] is t["steps"]
    assert normalize_trajectory(t)["id"] == "s1"


def test_accessors_match_results_to_calls_and_find_final_message():
    t = make()
    calls = t.calls()
    assert [(c.id, c.name, c.arguments, c.output) for c in calls] == [
        ("c1", "calc", {"x": 2}, "2"),
        ("c2", "calc", {"x": 3}, "3"),
    ]
    assert calls[0].step is t["steps"][1]
    assert t.final_message == "5"
    cut = make({**DOC, "steps": DOC["steps"][:2]})
    assert cut.final_message is None  # last agent step still wanted a tool
    assert message_text([{"type": "text", "text": "a"}, {"type": "image"}]) == "a"
    assert make({**DOC, "steps": DOC["steps"][:1]}).calls() == []


def test_unanswered_call_has_no_result():
    doc = {
        **DOC,
        "steps": [
            {"source": "agent", "message": "", "tool_calls": DOC["steps"][1]["tool_calls"][:1]}
        ],
    }
    (c,) = make(doc).calls()
    assert c.result is None and c.output is None


def test_subagents_have_the_same_accessors():
    sub = {"trajectory_id": "sub1", **DOC}
    t = make({**DOC, "subagent_trajectories": [sub]})
    assert len(t.subagents) == 1 and t.subagents[0].final_message == "5"


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda t: t["extra"].pop("l2h"), "missing extra"),
        (lambda t: t["extra"]["l2h"].update(ended="great"), "'ended' must be one of"),
        (lambda t: t["extra"]["l2h"].update(source_refs=[]), "source_refs"),
        (lambda t: t.update(steps=[]), "not ATIF: 'steps'"),
        (lambda t: t["steps"][0].update(source="robot"), "not ATIF"),
    ],
)
def test_contract_errors(mutate, msg):
    t = make()
    mutate(t)
    with pytest.raises(ContractError, match=msg):
        normalize_trajectory(t)


def test_harbor_model_rejects_invalid_atif_with_a_stable_cause():
    pytest.importorskip("harbor.models.trajectories")
    t = make()
    del t["agent"]
    with pytest.raises(ContractError, match=r"not ATIF: agent: Field required"):
        normalize_trajectory(t)
    t = make()
    t["steps"][1]["observation"]["results"][0]["source_call_id"] = "zzz"
    with pytest.raises(ContractError, match="not ATIF: .*source_call_id"):
        normalize_trajectory(t)
    t = make()
    t["steps"][0]["tool_calls"] = []  # only agent steps may carry tool calls
    with pytest.raises(ContractError, match="not ATIF"):
        normalize_trajectory(t)
    normalize_trajectory(make())  # the valid document passes the real model


def test_without_harbor_only_the_basics_are_checked(monkeypatch):
    monkeypatch.setattr("l2h.models._atif_model", lambda: None)
    t = make()
    del t["agent"]
    assert normalize_trajectory(t)["id"] == "s1"


GUIDE = Path(__file__).parents[2] / "l2h/guides/parsing.md"
ATIF_NATIVE_PARSE = GUIDE.read_text().split("```python\n")[1].split("```")[0]  # the guide's example


BAD_PARSE = """
from l2h import Trajectory

def parse(paths):
    for i in (1, 2, 3):
        steps = [] if i < 3 else [{"source": "robot", "message": "m"}]
        yield Trajectory.from_atif({"agent": {"name": "b", "version": "1"}, "steps": steps},
                                   id=f"bad{i}", source_refs=[{"file": "a.jsonl", "offset": 0}])
"""


def test_atif_native_logs_parse_in_a_few_lines(env):
    (env.logs / "job/t1/agent").mkdir(parents=True)
    (env.logs / "job/t1/agent/trajectory.json").write_text(json.dumps(DOC))
    assert env.init()[0] == 0
    env.write("parse.py", ATIF_NATIVE_PARSE)
    code, _, _, res = env.run("run", "parse", "--json")
    d = res["data"]
    assert code == 0 and d["trajectories"] == 1 and d["parse_errors"] == 0, d
    stored = json.loads((env.ws / "out/trajectories.jsonl").read_text())
    assert stored["steps"][2]["message"] == "5" and stored["extra"]["l2h"]["id"] == "t1"
    assert stored["extra"]["l2h"]["source_refs"][0]["file"] == "job/t1/agent/trajectory.json"


def test_invalid_atif_becomes_a_parse_error_grouped_by_cause(env):
    assert env.init()[0] == 0
    env.write("parse.py", BAD_PARSE)
    d = env.run("run", "parse", "--json")[3]["data"]
    assert d["trajectories"] == 0 and d["parse_errors"] == 3
    assert [(g["count"], g["examples"]) for g in d["error_groups"]] == [
        (2, ["bad1", "bad2"]),
        (1, ["bad3"]),
    ]
    assert all(g["cause"].startswith("not ATIF") for g in d["error_groups"])


def test_task_scope_is_required_and_shown_in_report(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace('"job": "main", "task_scope": "decision", ', ""),
    )
    code, _, err, res = env.run("run", "generate", "--json")
    assert code == 1 and "task_scope" in res["error"]["message"]
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace('"env": "static",', '"env": "static", "job": "main", "task_scope": "everything",'),
    )
    assert env.run("run", "generate", "--json")[0] == 1
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace(
            '"job": "main", "task_scope": "everything"', '"job": "main", "task_scope": "full_job"'
        ),
    )
    assert env.run("run", "generate", "--json")[0] == 0
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "| addnums | full job |" in md and "decision-from-evidence" in md


def test_schema_and_scaffolds_describe_atif(env):
    d = env.run("schema", "--json", ws=False)[3]["data"]["parse.py"]["Trajectory"]
    assert ATIF_VERSION in d["format"] and "source_call_id" in json.dumps(d) and "extra.l2h" in d
    assert (
        "task_scope"
        in env.run("schema", "--json", ws=False)[3]["data"]["tasktypes/<type>.py"]["SPEC"]
    )
    env.init()
    env.run("new", "parser")
    env.run("new", "tasktype", "x")
    assert "from_atif" in (env.ws / "parse.py").read_text()
    assert '"task_scope"' in (env.ws / "tasktypes/x.py").read_text()


def test_show_field_accepts_l2h_shortcuts(env):
    env.build_pipeline()
    assert env.run("show", "q1", "--field", "meta.answer", "--json")[3]["data"]["record"] == "42"
    d = env.run("show", "q1", "--field", "extra.l2h.ended", "--json")[3]["data"]
    assert d["record"] == "done"


def test_report_lists_jobs_and_unbuilt_ones(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "## Jobs" in md and "| main | full_job |  | addnums |" in md
    assert "Jobs with no task type yet (say why in a decision): sub" in md


def test_old_events_format_gets_a_migration_error():
    import pytest

    from l2h.models import ContractError, normalize_trajectory

    with pytest.raises(ContractError, match="pre-ATIF format"):
        normalize_trajectory({"id": "a", "source_refs": [{"file": "x", "offset": 0}], "events": []})
