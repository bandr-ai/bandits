"""Consistency check: code decides the label from the model's lists; offline fake."""
from __future__ import annotations

import json

from test_pipeline import seed

from bandits_sft.config import load_config
from bandits_sft.consistency import check, summarize
from bandits_sft.llm import set_llm_factory
from bandits_sft.schema import from_sharegpt


def fake(reply: dict, prompts: list):
    def factory(cfg, role):
        def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
            prompts.append(messages[-1]["content"])
            return {"content": json.dumps(reply), "tool_calls": []}
        return type("F", (), {"chat": chat})()
    set_llm_factory(factory)


def test_label_follows_the_lists(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "models": {"default": {"model": "fake"}}}))
    cfg = load_config(p)
    t = from_sharegpt(seed(0), 0)
    prompts = []
    fake({"facts": [{"msg": 1, "facts": ["a"]}], "contradictions": [], "impossible_results": [], "unsupported_claims": []}, prompts)
    assert check(cfg, t)["label"] == "good" and "<conversation" in prompts[0] and "<reference_seed" not in prompts[0]
    check(cfg, t, seed=t)
    assert "<reference_seed" in prompts[-1]
    fake({"facts": [], "contradictions": [{"msgs": [1, 3], "why": "x"}], "impossible_results": [], "unsupported_claims": []}, prompts)
    r = check(cfg, t)
    assert r["label"] == "bad" and r["problems"]["contradictions"]
    rows = [{"expected": "bad", **r}, {"expected": "good", "label": "good", "problems": {k: [] for k in r["problems"]}}]
    assert summarize(rows)["bad_caught"] == "1/1" and summarize(rows)["good_rejected"] == "0/1"
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def test_verify_runs_judge_and_consistency_and_either_rejects(tmp_path):
    from test_pipeline import write_cfg

    from bandits_sft.verify import verify
    rubric = tmp_path / "r.md"
    rubric.write_text("rubric\n## Output\n{}")
    t = from_sharegpt(seed(0), 0)
    for judge_out, cons_out, kept in ((
            {"answer_correct": True, "integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}}, {"contradictions": [], "impossible_results": [], "unsupported_claims": []}, True),
            ({"answer_correct": True, "integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}}, {"contradictions": [{"msgs": [1, 2], "why": "x"}]}, False),
            ({"answer_correct": False, "integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}}, {"contradictions": [], "impossible_results": [], "unsupported_claims": []}, False)):
        cfg = load_config(write_cfg(tmp_path, {"rubric_judge": True, "consistency_check": True, "change_gate": False}, target=1,
                                    rubric_path=str(rubric)))

        def chat(self, messages, *a, j=judge_out, c=cons_out, **k):
            is_cons = "internal consistency" in messages[0]["content"]
            return {"content": json.dumps(c if is_cons else j), "tool_calls": []}
        set_llm_factory(lambda c, role, chat=chat: type("F", (), {"chat": chat})())
        v = verify(cfg, json.loads(json.dumps(t)), set())["meta"]["verify"]
        assert v["kept"] is kept and "consistency" in v and "rubric" in v
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def test_incomplete_answer_is_an_error_not_good(tmp_path):
    """Review repro: {} used to become label good. Now one retry, then an error (fail closed)."""
    import pytest
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "models": {"default": {"model": "fake"}}}))
    cfg = load_config(p)
    prompts = []
    for reply in ({}, {"contradictions": []}, {"contradictions": None, "impossible_results": [], "unsupported_claims": []}):
        prompts.clear()
        fake(reply, prompts)
        with pytest.raises(ValueError):
            check(cfg, from_sharegpt(seed(0), 0))
        assert len(prompts) == 2  # retried once
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))
