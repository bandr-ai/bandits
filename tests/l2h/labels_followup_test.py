"""Follow-up regressions from running the fixed tool on real-shaped data:
exact-label tasks whose rubric defines every label in prose, secret-scan false alarms
on ordinary code, and negative tasks that would accept any string."""

import json

from l2h import Answer, secrets
from l2h.models import ContractError

from .negative_test import NEG_PY, setup
from .validate_test import rows

RUBRIC = (
    "You judge whether the element is on the page.\n"
    "If the element is visible where it was recorded, conclude 'found_in_place'.\n"
    "If several elements match the locator, conclude 'found_over_matched'.\n"
    "If nothing matches and nothing explains it, conclude 'absent'; if the page explains why, "
    "use 'absent_explained' instead.\n"
)

EXACT_PY = """
from l2h import Answer, BuildResult

SPEC = {"unit": "episode", "env": "static", "task_scope": "decision", "check": "exact", "gold": "imitation",
        "hidden": [], "applies_when": "any"}
VERSION = 1
EVIDENCE_EXTRA = ""
CHOICES = ["found_in_place", "found_over_matched", "absent", "absent_explained"]

def applies(t, findings):
    return True

def group_key(t):
    return t["id"]

def build(group, out):
    t = group[0]
    return BuildResult(
        task_id="x_" + t["id"],
        instruction="Read /app/rubric.md and /app/log.md. Write one label (one of: "
                    + ", ".join(CHOICES) + ").",
        evidence={"rubric.md": RUBRIC, "log.md": "probe returned 3 matches" + EVIDENCE_EXTRA},
        answer=Answer(expected="found_over_matched", choices=CHOICES),
        wrong_answers=["absent"],
    )
"""


def exact_setup(env, py=EXACT_PY):
    env.build_pipeline()
    env.write("tasktypes/addnums.py", py.replace("RUBRIC", repr(RUBRIC), 1))


def test_exact_label_task_with_rubric_defining_labels_passes_leak_scan(env):
    exact_setup(env)
    _, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"generated": 5}, res["data"]
    r = next(iter(rows(env).values()))
    assert r["gates"]["leak"] == "pass"
    w = " ".join(r.get("warnings", []))
    assert "giveaway" not in w  # the rubric defines labels; that is not a giveaway
    assert "alternatives" in w  # label task without alternatives is flagged


def test_label_stated_in_evidence_is_still_a_leak(env):
    exact_setup(
        env,
        EXACT_PY.replace('EVIDENCE_EXTRA = ""', 'EVIDENCE_EXTRA = "; tagged found_over_matched"'),
    )
    _, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"blocked": 5}


def test_choices_must_contain_expected():
    try:
        Answer(expected="nope", choices=["a", "b"]).validate()
    except ContractError as e:
        assert "not in answer.choices" in str(e)
    else:
        raise AssertionError("expected a ContractError")


def test_secret_scan_ignores_ordinary_code_but_not_literals():
    code = [
        "password = getpass.getpass()",
        "secret = generate_secret(32)",
        "token = tokens[index]",
        "token = tok.encode(text)",
        "self.token = self.session.token",
    ]
    for line in code:
        assert not secrets.scan_text(line), line
    for line in ["password: Tr0ub4dor3x", 'token = "q9w8e7r6t5y4"', "password=hunter22"]:
        assert secrets.scan_text(line), line


def test_negative_task_needs_pattern_or_choices():
    try:
        Answer(expected=["13"], check="forbidden").validate()
    except ContractError as e:
        assert "pattern or answer.choices" in str(e)
    else:
        raise AssertionError("expected a ContractError")


def test_negative_task_with_choices_rejects_answers_outside_them(env):
    py = NEG_PY.replace(
        "KW = {}",
        "KW = {'answer': Answer(expected=['red'], check='forbidden', "
        "choices=['red', 'green', 'blue']), 'alternatives': ['green']}",
    )
    setup(env, py)
    env.run("run", "generate")
    tid, _ = next(iter(rows(env).items()))
    for answer, want in (("green", 0), ("red", 1), ("purple", 1), ("", 1)):
        code, _, _, res = env.run("task", "try", tid, "--answer", answer, "--json")
        assert res["data"]["reward"] == (1.0 if want == 0 else 0.0), (answer, res)
    spec = json.loads(next((env.ws / "out/tasks").rglob("expected.json")).read_text())
    assert spec["choices"] == ["red", "green", "blue"]
