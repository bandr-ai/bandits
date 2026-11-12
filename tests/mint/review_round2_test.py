"""Second review round: id-less tool calls, `in` on mint fields, run-summary diffs, next steps."""

import json

from mint import Trajectory

from .conftest import CHECKS_PY, JOBS_JSON, PARSE_PY, TASKTYPE_PY

REFS = [{"file": "a.jsonl", "offset": 0}]


def traj(steps):
    return Trajectory.from_atif(
        {"agent": {"name": "b", "version": "unknown"}, "steps": steps}, id="s", source_refs=REFS
    )


def test_calls_without_ids_pair_with_results_by_position():
    t = traj(
        [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"function_name": "a", "arguments": {}},
                    {"function_name": "b", "arguments": {}},
                ],
                "observation": {"results": [{"content": "ra"}, {"content": "rb"}]},
            }
        ]
    )
    assert [(c.name, c.output) for c in t.calls()] == [("a", "ra"), ("b", "rb")]
    # ids win over position; a count mismatch is ambiguous and pairs nothing
    t2 = traj(
        [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "x", "function_name": "a", "arguments": {}},
                    {"function_name": "b", "arguments": {}},
                    {"function_name": "c", "arguments": {}},
                ],
                "observation": {
                    "results": [{"content": "rb?"}, {"source_call_id": "x", "content": "ra"}]
                },
            }
        ]
    )
    assert [c.output for c in t2.calls()] == ["ra", None, None]


def test_partial_ids_never_pair_by_position():
    # c2 has an id but was never answered; the id-less result belongs to something else
    t = traj(
        [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "a", "arguments": {}},
                    {"tool_call_id": "c2", "function_name": "b", "arguments": {}},
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "ra"}, {"content": "other"}]
                },
            }
        ]
    )
    assert [c.output for c in t.calls()] == ["ra", None]
    # an id-less call next to an id-less result still doesn't pair when any id is present
    t2 = traj(
        [
            {
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "a", "arguments": {}},
                    {"function_name": "b", "arguments": {}},
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "ra"}, {"content": "rb?"}]
                },
            }
        ]
    )
    assert [c.output for c in t2.calls()] == ["ra", None]


def test_in_agrees_with_getitem_for_mint_fields():
    t = traj([{"source": "user", "message": "hi"}])
    for k in ("id", "meta", "ended", "source_refs", "attachments", "steps"):
        assert k in t and t[k] is not None
    assert "nope" not in t and "meta" not in dict(t)


def test_run_parse_warns_on_unanswered_calls_and_suggests_jobs_first(env):
    env.init()
    env.write("parse.py", PARSE_PY.replace('"source_call_id": "c1", ', '"source_call_id": "zz", '))
    env.write("source_notes.md", "a record is one line\n")
    code, out, _, _ = env.run("run", "parse")
    assert code == 0
    assert "5 of 5 tool calls have no matching result" in out
    nxt = out[out.index("next:") :]
    assert "mint new jobs" in nxt and "mint new check" not in nxt


def test_changed_since_last_run_reports_content_and_version_changes(env):
    env.init()
    env.write("parse.py", PARSE_PY)
    env.write("checks.py", CHECKS_PY)
    env.write("jobs.json", JOBS_JSON)
    env.write("tasktypes/addnums.py", TASKTYPE_PY)
    env.run("run", "parse")
    # same counts, different content
    env.write("parse.py", PARSE_PY.replace('"name": "qa-bot"', '"name": "qa-bot-2"'))
    s = env.run("run", "parse", "--json")[3]["data"]
    assert s["changed_since_last_run"] == {"trajectories_changed": 5}
    s = env.run("run", "parse", "--json", "--no-cache")[3]["data"]
    assert s["changed_since_last_run"] == {"none": "same results as last run"}
    env.run("run", "checks")
    env.run("run", "generate")
    env.write("tasktypes/addnums.py", TASKTYPE_PY.replace("VERSION = 1", "VERSION = 2"))
    s = env.run("run", "generate", "--json")[3]["data"]
    ch = s["changed_since_last_run"]
    assert ch.get("tasks_generator_changed") == 3, ch
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace("VERSION = 1", "VERSION = 2").replace(
            "Add up the numbers", "Sum the numbers"
        ),
    )
    ch = env.run("run", "generate", "--json")[3]["data"]["changed_since_last_run"]
    assert ch == {"tasks_content_changed": 3}, ch
    assert json.dumps(ch)


def test_solve_sh_comment_names_the_gold_kind():
    from mint.harbor import solve_sh
    from mint.models import Answer

    a = Answer(expected="7")
    assert "the logged answer" in solve_sh(a) and "not that the answer" in solve_sh(a)
    v = solve_sh(a, gold_type="verified")
    assert "Oracle (verified gold)" in v and "logged answer" not in v
    assert "not that the answer" not in v
    assert "computed from the task inputs" in solve_sh(a, gold_type="solved")


def test_skill_documents_missing_runtime_and_jobs_scaffold():
    from pathlib import Path

    import mint

    skill = (Path(mint.__file__).parent / "SKILL.md").read_text()
    guide = (Path(mint.__file__).parent / "guides/tasktypes.md").read_text()
    assert "--allow-missing-runtime" in skill and "`mint new jobs`" in skill
    for doc in (skill, guide):
        assert "`full_job` | `subtask` | `decision`" in doc
