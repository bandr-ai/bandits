"""Negative tasks: the verifier rewards any well-formed answer except the known-wrong ones."""

import json

from .validate_test import rows

NEG_PY = """
from mint import Answer, BuildResult

SPEC = {"unit": "episode", "env": "static", "job": "main", "task_scope": "decision", "check": "forbidden", "gold": "negative",
        "hidden": [], "applies_when": "any"}
VERSION = 1
KW = {}

def applies(t, findings):
    return t["meta"]["nums"] != "BLOCK"

def group_key(t):
    return t["id"]

def build(group, out):
    t = group[0]
    kw = dict(gold_type="negative", alternatives=["99"],
              answer=Answer(expected=["13", "14"], check="forbidden", pattern=r"\\d+"))
    kw.update(KW)
    return BuildResult(
        task_id="n_" + t["id"],
        instruction="Pick any whole number that is not what the failed run reported (see /app/nums.txt).",
        evidence={"nums.txt": t["meta"]["nums"]},
        **kw,
    )
"""


def setup(env, py=NEG_PY):
    env.build_pipeline()
    (env.ws / "tasktypes/addnums.py").unlink()
    env.write("tasktypes/neg.py", py)


def test_negative_task_gates_and_verifier(env):
    setup(env)
    code, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"generated": 4}, res["data"]
    assert rows(env)["n_q1"]["gold"] == "negative"
    spec = json.loads((env.ws / "out/tasks/neg/n_q1/tests/expected.json").read_text())
    assert spec["check"] == "forbidden" and spec["pattern"] == "\\d+"
    # the oracle writes a valid alternative, not a forbidden answer
    assert "99" in (env.ws / "out/tasks/neg/n_q1/solution/solve.sh").read_text()

    def reward(*a):
        return env.run("task", "try", "n_q1", *a, "--json")[3]["data"]["reward"]

    assert reward("--answer", "99") == 1.0 and reward("--answer", " 7 ") == 1.0
    assert reward("--answer", "13") == 0.0 and reward("--answer", "14") == 0.0
    assert reward("--answer", "not a number") == 0.0  # not well-formed
    assert reward() == 0.0  # no-op
    code, _, _, res = env.run("validate", "--sample-size", "4", "--json")
    assert code == 0, res
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    detail = next(iter(rows(env).values()))["gates"]["exec_detail"]
    assert detail == {"oracle": "pass", "noop": "pass", "wrong": "pass", "alternatives": "pass"}
    assert not any("alternatives:" in w for r in rows(env).values() for w in r.get("warnings", []))
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert (
        "| negative |" in md
        and "| neg | decision-from-evidence | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 4 |" in md
    )
    assert "validated (negative): 4" in md


def test_verifier_that_accepts_a_forbidden_answer_fails_level_2(env):
    setup(env)
    env.write(
        "tasktypes/neg.py",
        NEG_PY.replace(
            "kw.update(KW)",
            "kw['files'] = {'tests/test.sh': '#!/bin/bash\\nmkdir -p \"$MINT_REWARD_DIR\"\\n"
            'echo 1 > "$MINT_REWARD_DIR/reward.txt"\\n# /logs/verifier/reward.txt\\n\'}',
        ),
    )
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--sample-size", "2", "--json")
    causes = " ".join(c["cause"] for c in res["data"]["types"]["neg"]["causes"])
    assert code == 1 and "wrong answer was accepted" in causes


def test_negative_contract_errors(env):
    setup(env, NEG_PY.replace("KW = {}", "KW = {'alternatives': []}"))
    env.run("run", "generate")
    assert "needs `alternatives`" in (env.ws / "out/queue.jsonl").read_text()
    env.write("tasktypes/neg.py", NEG_PY.replace("KW = {}", "KW = {'alternatives': ['13']}"))
    env.run("run", "generate")
    assert "must differ from the known-wrong" in (env.ws / "out/queue.jsonl").read_text()
    env.write("tasktypes/neg.py", NEG_PY.replace("KW = {}", "KW = {'gold_type': 'imitation'}"))
    env.run("run", "generate")
    assert "go together" in (env.ws / "out/queue.jsonl").read_text()
