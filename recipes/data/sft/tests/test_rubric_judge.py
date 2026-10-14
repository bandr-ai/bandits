"""Rubric judge as the pipeline gate: offline, a fake judge model answers."""
from __future__ import annotations

import json

import pytest
from test_pipeline import seed

from simia_plus.config import load_config
from simia_plus.llm import set_llm_factory
from simia_plus.rubric_judge import describe_change, normalize
from simia_plus.schema import from_sharegpt
from simia_plus.verify import verify

CLEAN = {"facts": [], "contradictions": [], "impossible_results": [], "unsupported_claims": []}


def judge_says(payload):
    """The rubric judge answers `payload`; the consistency check (same judge model) finds nothing wrong."""
    def chat(self, messages, tools=None, json_mode=False):
        if isinstance(payload, Exception):
            raise payload
        if "internal consistency" in messages[0]["content"]:
            return {"content": json.dumps(CLEAN), "tool_calls": []}
        return {"content": json.dumps(payload), "tool_calls": []}
    set_llm_factory(lambda cfg, role: type("J", (), {"chat": chat})())


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.json"
    (tmp_path / "rubric.md").write_text("# rubric")
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "features": {"rubric_judge": True},
                             "rubric_path": str(tmp_path / "rubric.md"), "models": {"default": {"model": "fake"}}}))
    yield load_config(p)
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


CHECKS = {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}
GOOD = {"agent": "x", "situation": "READY", "integrity": CHECKS, "answer_correct": True,
        "label": "good", "minor_issues": [], "reason": "r"}


def test_good_is_kept(cfg):
    judge_says(GOOD)
    v = verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is True and v["rubric"]["label"] == "good"


def test_integrity_failure_is_dropped_even_if_labelled_good(cfg):
    judge_says({**GOOD, "integrity": {**CHECKS, "I1_consistent": "fail"}})
    v = verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is False and v["rubric"]["integrity_failed"] == ["I1_consistent"]


def test_wrong_answer_is_dropped(cfg):
    judge_says({**GOOD, "answer_correct": False})
    assert verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]["kept"] is False


def test_judge_failure_fails_closed(cfg):
    judge_says(RuntimeError("boom"))
    v = verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is False and v["rubric"]["label"] == "error"


def test_rule_failure_skips_the_judge(cfg):
    calls = []
    set_llm_factory(lambda c, role: calls.append(role) or (_ for _ in ()).throw(AssertionError("judge called")))
    t = from_sharegpt(seed(0), 0)
    t["meta"]["format_issues"] = ["broken"]
    assert verify(cfg, t, set())["meta"]["verify"]["kept"] is False and not calls


def test_unclear_is_not_kept():
    assert normalize({**GOOD, "label": "unclear"})["label"] == "unclear"


def test_incomplete_verdicts_are_errors_not_passes():
    """Review repro: {"label": "good"} without integrity checks or answer_correct used to pass."""
    for bad in ({"label": "good"}, {**GOOD, "integrity": {"I1_consistent": "pass"}}, {**GOOD, "answer_correct": None},
                {**GOOD, "integrity": {**CHECKS, "I3_decisive_kept": "maybe"}}):
        with pytest.raises(ValueError):
            normalize(json.loads(json.dumps(bad)))
    assert normalize({**GOOD, "integrity": {**CHECKS, "I2_possible": None, "I2_possible_results": "pass"}})["label"] == "good"


def test_wrong_answer_is_bad_even_if_labelled_unclear():
    assert normalize({**GOOD, "label": "unclear", "answer_correct": False})["label"] == "bad"


def test_incomplete_verdict_fails_closed_in_verify(cfg):
    judge_says({"label": "good"})
    v = verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is False and v["rubric"]["label"] == "error"


def test_describe_change_keeps_only_the_change():
    assert json.loads(describe_change({"name": "n", "change": "c", "correct_when": "secret"})) == {"name": "n", "change": "c"}
    assert describe_change(None) is None


def test_no_rubric_fails_closed(cfg):
    """No default rubric: without one the judge errors and nothing is kept (never another domain's rubric)."""
    from simia_plus.rubric_judge import load_rubric
    with pytest.raises(ValueError, match="no rubric"):
        load_rubric(None)


def test_change_gate_drops_noops_cosmetic_edits_and_copies():
    from simia_plus.verify import change_gate
    s = from_sharegpt(seed(0), 0)
    same = json.loads(json.dumps(s))
    assert any(i.startswith("no_input_change") for i in change_gate(same, s, 0.02))
    assert any(i.startswith("assistant_unchanged") for i in change_gate(same, s, 0.02))
    cosmetic = json.loads(json.dumps(s))
    u = next(m for m in cosmetic["messages"] if m["role"] == "user")
    u["content"] = u["content"].replace(" ", "  ") + " "  # whitespace only
    assert any(i.startswith("no_input_change") for i in change_gate(cosmetic, s, 0.02))
    real = json.loads(json.dumps(s))
    next(m for m in real["messages"] if m["role"] == "user")["content"] += " Also, I am a former employee."
    real["messages"][-1]["content"] = "A completely different reply that follows the new input."
    assert change_gate(real, s, 0.02) == []
    assert not any(i.startswith("assistant_unchanged") for i in change_gate(same, s, 0))  # 0 disables the copy check


def test_v2_label_follows_parts_not_the_judge():
    """v2 judges write no label: code derives it, so a right answer cannot excuse a failed integrity check."""
    v2 = {"integrity": {**CHECKS, "I2_possible": "fail"}, "answer_correct": True, "integrity_critique": "a", "answer_critique": "b"}
    out = normalize(v2)
    assert out["label"] == "bad" and out["integrity_failed"] == ["I2_possible"] and out["critique"] == "a\nb"
    assert normalize({"integrity": CHECKS, "answer_correct": True})["label"] == "good"
