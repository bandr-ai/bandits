"""Offline tests of one scenario job: leakage, pairs, control outcome, resume and shared gates."""
import json

import pytest

from bandits_sft import scenarios as s
from bandits_sft.llm import log_call, set_budget, set_llm_factory


def seed(i="s"):
    return {"id": i, "system": 'Return ONLY JSON with root_cause, confidence and detail.', "tools": [],
            "messages": [{"role": "user", "content": "A mailbox request failed after navigation"},
                         {"role": "assistant", "content": '{"root_cause":"X","confidence":0.9,"detail":"recorded fact"}'}], "meta": {}}


class Fake:
    def __init__(self, role, opts):
        self.role, self.opts = role, opts

    def chat(self, messages, **kwargs):
        self.opts["calls"].append(self.role)
        text = messages[0]["content"]
        if self.role == "analyzer":
            out = {k: "new workflow with a changed observation" for k in
                   ("situation", "workflow", "decisive_evidence", "dependencies", "novelty_reason")}
            out["variation"] = {k: "seed" if k == "basis" else "changed evidence" for k in
                                ("name", "change", "preserve", "dependencies", "correct_when", "teaches", "basis")}
            out["counterfactual_flip"] = "flip the observed state and its dependents"
            out["episode_state"] = {"context": {}, "facts": ["item present"]}
        elif self.role == "generator":
            row = seed()
            row["messages"][0]["content"] = "Inventory synchronization stopped after warehouse transfer"
            out = {"traces": [{"messages": row["messages"]}]}
            if self.opts.get("pair"):
                other = seed()
                other["messages"][0]["content"] = "Inventory synchronization continued after warehouse transfer"
                other["messages"][-1]["content"] = '{"root_cause":"Y","confidence":0.9,"detail":"continued"}'
                out["traces"].append({"messages": other["messages"]})
            if self.opts.get("wrong_target"):
                out["traces"][0]["messages"][-1]["content"] = '{"root_cause":"Y"}'
        elif "internal consistency" in text:
            out = {"facts": [], "contradictions": [], "impossible_results": [], "unsupported_claims": []}
        elif "generation dimensions" in text:
            out = {"supported": True, "novelty_ok": not self.opts.get("cosmetic"), "dimensions_ok": True,
                   "pair_ok": True, "reason": "changed workflow"}
        else:
            if self.opts.get("judge_down"):
                raise RuntimeError("judge unavailable")
            out = {"integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a",
                                  "I4_same_situation": "n/a"}, "answer_correct": True}
        log_call({"role": self.role, "ok": True, "cost": 0.001})
        return {"content": json.dumps(out), "tool_calls": [], "cost": 0.001}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("SCENARIO_TEST_KEY", "fake")
    opts = {"calls": []}
    set_llm_factory(lambda cfg, role: Fake(role, opts))
    yield opts
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real API in tests")))
    set_budget(None)


def config(tmp_path):
    from bandits_sft.config import load_config
    (tmp_path / "rubric.md").write_text("Check observations and answer correctness.")
    model = {"model": "fake", "api_key_env": "SCENARIO_TEST_KEY", "price_input_per_mtok": 0.01, "price_output_per_mtok": 0.01,
             "max_tokens": 1000}
    (tmp_path / "config.json").write_text(json.dumps({"seeds_path": str(tmp_path / "seeds.jsonl"),
           "features": {"rubric_judge": True, "consistency_check": True}, "rubric_path": str(tmp_path / "rubric.md"),
           "models": {"default": model}}))
    return load_config(str(tmp_path / "config.json"))


def job(i=1, **kw):
    return {"job_id": f"scenarios/{i:04d}", "cell": "scenario_broad", "seed_id": "s", "pair": False, "target": {},
            "failure_id": None, "dimensions": {**s.DEFAULTS, "targeting": "coverage"}, "allocation": "broad",
            "capability": None, "outcome": None, **kw}


def run(tmp_path, **kw):
    return s.run_job(config(tmp_path), job(**kw), [seed()], "X allowed", tmp_path / "out", [], {}, {})


def test_job_resume_reuses_card_and_generation(tmp_path, fake):
    fake["judge_down"] = True
    with pytest.raises(RuntimeError, match="evaluator error"):
        run(tmp_path)
    fake["judge_down"] = False
    row = run(tmp_path)
    assert row["kept"] is True and len(row["traces"]) == 1
    assert "conclusion" not in json.loads(row["traces"][0]["messages"][-1]["content"])
    assert fake["calls"].count("analyzer") == 1 and fake["calls"].count("generator") == 1


def test_cosmetic_scenario_rejected(tmp_path, fake):
    fake["cosmetic"] = True
    assert run(tmp_path)["kept"] is False


def test_leakage_and_missing_target_keys():
    t = seed()
    assert s.leakage(t, [t])
    assert not s.target_matches(t, {"conclusion": None}, {})


def test_target_enforced_after_quality_gates(tmp_path, fake):
    fake["wrong_target"] = True
    row = run(tmp_path, target={"root_cause": "X"})
    assert row["reason"] == "target not reached" and row["kept"] is False


def test_scenario_pair_passes_atomically(tmp_path, fake):
    fake["pair"] = True
    row = run(tmp_path, pair=True)
    assert row["kept"] is True and len(row["traces"]) == 2
    assert row["traces"][0]["meta"]["pair_id"] == row["traces"][1]["meta"]["pair_id"]


def test_pair_dedup_keeps_unit_atomic(tmp_path):
    a, b = seed("a"), seed("b")
    for t in (a, b):
        t["meta"] = {"cell": "c"}
    rows = [{"kept": True, "job": {"cell": "c"}, "traces": [a], "reason": "ok"},
            {"kept": True, "job": {"cell": "c"}, "traces": [b, seed("new")], "reason": "ok"}]
    report = s.export(tmp_path, rows, {})
    assert report["traces_kept"] == 1 and report["duplicate_or_conflicting_traces"] == 2


def test_control_outcome_mismatch_rejected_before_judges(tmp_path, fake):
    row = run(tmp_path, outcome={"root_cause": "Z"})
    assert row["reason"] == "control outcome mismatch" and fake["calls"] == ["analyzer", "generator"]
    assert run(tmp_path, i=2, outcome={"root_cause": "X"})["reason"] != "control outcome mismatch"


def test_leakage_ignores_shared_template():
    from bandits_sft.verify import boilerplate
    template = " ".join(f"field{i} value{i}" for i in range(60))  # long input template shared by every trace
    def t(i, body):
        return {"id": i, "messages": [{"role": "user", "content": f"{template} {body}"}]}
    seeds = [t(i, f"case {i} " + " ".join(f"w{i}x{k}" for k in range(20))) for i in range(10)]
    common = boilerplate(seeds)
    fresh = t("new", " ".join(f"new{k}" for k in range(20)))
    assert not s.leakage(fresh, seeds[:1], common=common)  # template alone is not a copy
    assert s.leakage(t("dup", seeds[3]["messages"][0]["content"].split(template)[1]), seeds, common=common)
    assert s.leakage(fresh, seeds[:1])  # without template removal everything looks copied (the old failure)


def test_truncated_generation_rejects_job_only(tmp_path, fake, monkeypatch):
    from bandits_sft.llm import Truncated
    real = s.chat_json

    def chat(llm, system, user, **kw):
        if system == s.prompts.GENERATE:
            raise Truncated("response hit the token limit (64000)")
        return real(llm, system, user, **kw)
    monkeypatch.setattr(s, "chat_json", chat)
    row = run(tmp_path)
    assert row["kept"] is False and "Truncated" in row["reason"]
    assert (tmp_path / "out/jobs/scenarios/0001/generated_raw.json").exists()  # resume does not pay again
