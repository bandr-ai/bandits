"""Level-1 warnings: allowed-set label vocabularies, giveaways, missing alternatives,
dangling /app paths. Warnings never block a task."""

from .conftest import TASKTYPE_PY
from .validate_test import rows

LABELS_PY = """
from l2h import Answer, BuildResult

SPEC = {"unit": "episode", "env": "static", "task_scope": "decision", "check": "allowed_set", "gold": "imitation",
        "hidden": [], "applies_when": "any"}
VERSION = 1
EXTRA = ""
ALT = []

def applies(t, findings):
    return True

def group_key(t):
    return t["id"]

def build(group, out):
    t = group[0]
    return BuildResult(
        task_id="l_" + t["id"],
        instruction="Classify the ticket in /app/ticket.txt as one of: account_suspension, "
                    "refund_requested, billing_dispute. Labels are defined in /app/rubric.md."
                    + EXTRA,
        evidence={"ticket.txt": "please help", "rubric.md": "- billing_dispute\\n- account_suspension"},
        answer=Answer(expected=["account_suspension"], check="allowed_set"),
        wrong_answers=["refund_requested"],
        alternatives=ALT,
    )
"""


def test_label_vocabulary_passes_leak_scan_without_allow_list(env):
    env.build_pipeline()
    env.write("tasktypes/addnums.py", LABELS_PY)
    code, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"generated": 5}, res["data"]["structure_or_leak_failures"]
    r = next(iter(rows(env).values()))
    assert r["gates"]["leak"] == "pass"
    code, _, _, res = env.run("validate", "--sample-size", "2", "--json")
    assert code == 0, res


def test_real_leak_still_blocks_with_labels(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        LABELS_PY.replace('EXTRA = ""', 'EXTRA = " The answer: account_suspension"'),
    )
    _, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"blocked": 5}


def test_missing_alternatives_warns_and_is_reported(env):
    env.build_pipeline()
    env.write("tasktypes/addnums.py", LABELS_PY)
    env.run("run", "generate")
    r = next(iter(rows(env).values()))
    assert any(w.startswith("alternatives:") for w in r["warnings"])
    code, out, _, res = env.run("validate", "--sample-size", "1", "--json")
    assert code == 0 and res["data"]["warnings"]
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "## Warnings" in md and "alternatives x5" in md
    # giving alternatives clears the warning
    env.write("tasktypes/addnums.py", LABELS_PY.replace("ALT = []", 'ALT = ["refund_requested"]'))
    env.write(
        "tasktypes/addnums.py",
        (env.ws / "tasktypes/addnums.py")
        .read_text()
        .replace('wrong_answers=["refund_requested"]', 'wrong_answers=["billing_dispute"]'),
    )
    env.run("run", "generate")
    assert not any("warnings" in r for r in rows(env).values())


def test_missing_referenced_file_warns(env):
    env.build_pipeline()
    env.write("tasktypes/addnums.py", LABELS_PY.replace("/app/rubric.md", "/app/missing.md"))
    env.run("run", "generate")
    r = next(iter(rows(env).values()))
    ws = [w for w in r["warnings"] if w.startswith("missing_file:")]
    assert len(ws) == 1 and "/app/missing.md" in ws[0]
    assert r["status"] == "generated"  # a warning never blocks
    assert not any("/app/ticket.txt" in w or "/app/answer" in w for w in r["warnings"])


def test_giveaway_when_exact_answer_is_in_evidence(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace(
            'evidence={"nums.txt": nums}',
            'evidence={"nums.txt": nums + "\\nsummary total: " + t["meta"]["answer"]}',
        ),
    )
    env.run("run", "generate")
    r = rows(env)["t_q1"]
    assert r["gates"]["leak"] == "pass" and r["status"] == "generated"
    assert any(w.startswith("giveaway:") and "nums.txt" in w for w in r["warnings"])
    env.run("report")
    assert "giveaway x3" in (env.ws / "report.md").read_text()
