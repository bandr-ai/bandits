"""Wider variation (ab-vary): proposal validation, task note, stacking, variety measures; offline fakes."""
from __future__ import annotations

import json

import pytest
from test_pipeline import seed

from simia_plus.analyzer import _stack_pairs, coverage_summary, generate_analyzer, propose_wide, system_labels, variety
from simia_plus.config import load_config
from simia_plus.llm import set_llm_factory
from simia_plus.schema import from_sharegpt


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "models": {"default": {"model": "fake"}}}))
    yield load_config(p)
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def reply(content: str, prompts: list):
    def factory(cfg, role):
        def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
            prompts.append((role, messages))
            return {"content": content, "tool_calls": []}
        return type("F", (), {"chat": chat})()
    set_llm_factory(factory)


def var(name, kind, **kw):
    return {"name": name, "kind": kind, "requires_tools": [], "basis": "seed", "decision_type": f"d_{name}",
            "answer_transition": "changes", "teaches": "t", "change": "c", "preserve": "p", "dependencies": "d",
            "correct_when": "w", **kw}


def test_proposals_keep_valid_kinds_only(cfg):
    seeds = [from_sharegpt(seed(i), i) for i in range(3)]
    labels = system_labels(seeds)
    label = next(iter(labels))
    out = {"variations": [var("a", "task"), var("c", "cross_seed"), var("d", "made_up"), var("e", "evidence")]}
    prompts = []
    reply(json.dumps(out), prompts)
    good = propose_wide(cfg, seeds[0], label, labels, seeds, "contracts", [], "coverage text", 4)
    assert [v["name"] for v in good] == ["a", "e"] and good[0]["eligible_seeds"] == [seeds[0]["id"]]
    assert "coverage text" in prompts[0][1][-1]["content"]
    assert {"c", "d"} <= {r["name"] for r in propose_wide.rejected}


def test_task_note_only_for_task_kind(cfg):
    from simia_plus.banks import ObsBank
    s = from_sharegpt(seed(0), 0)
    prompts = []
    reply(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}), prompts)
    for v, expect in ((var("x", "task"), True), (var("y", "evidence"), False)):
        generate_analyzer(cfg, {"job_id": v["name"], "seed_id": "s0", "variation": v}, s, "{seed_trace}{variation}{generation_id}",
                          ObsBank.from_traces([s]))
        assert ("## A different request" in prompts[-1][1][-1]["content"]) is expect


def test_stack_pairs_differ_in_decision_type():
    s = from_sharegpt(seed(0), 0)
    labels = system_labels([s])
    vs = [{**var(n, "evidence"), "applies_to": ["all"], "decision_type": dt} for n, dt in (("a", "x"), ("b", "x"), ("c", "y"), ("d", "z"))]
    pairs = _stack_pairs(vs, s, labels)
    assert [p["combines"] for p in pairs] == [["a", "c"], ["b", "d"]]


def test_variety_and_coverage_measures():
    seeds = [from_sharegpt(seed(i), i) for i in range(2)]
    kept = [{**seeds[0], "id": "k0", "meta": {"seed_id": seeds[0]["id"]}}]
    v = variety(kept, seeds, None)
    assert v["mean_share_found_in_closest_seed"] == 1.0 and v["at_least_0.6_found_in_a_seed"] == 1 and v["action_signatures_in_no_seed"] == 0
    cov = coverage_summary(seeds, [var("a", "task")], None)
    assert "d_a (1)" in cov and "tool calls in the seeds" in cov


def test_judge_sees_the_variation_kind():
    from simia_plus.rubric_judge import build_prompt, describe_change
    s = from_sharegpt(seed(0), 0)
    _, user = build_prompt(s, "rubric", s, describe_change({"name": "n", "change": "c", "kind": "task"}))
    assert '"kind": "task"' in user


def test_tools_the_seed_lacks_are_rejected(cfg):
    seeds = [from_sharegpt(seed(i), i) for i in range(2)]
    labels = system_labels(seeds)
    prompts = []
    reply(json.dumps({"variations": [var("ok", "task"), var("bad", "task", requires_tools=["not_a_seed_tool"])]}), prompts)
    good = propose_wide(cfg, seeds[0], next(iter(labels)), labels, seeds, "contracts", [], "", 2)
    assert [v["name"] for v in good] == ["ok"] and "<seed_tools" in prompts[0][1][-1]["content"]


def test_llm_client_per_model_settings(tmp_path):
    import dataclasses
    from simia_plus.llm import get_llm, set_llm_factory
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "models": {"default": {"model": "fake"}}}))
    a = load_config(p)
    b = load_config(p)
    b.models["generator"] = dataclasses.replace(b.models["default"], extra={"reasoning_effort": "none"})
    seen = []
    set_llm_factory(lambda m, role: seen.append(m.extra) or object())
    get_llm(a, "generator"), get_llm(b, "generator"), get_llm(a, "generator")
    assert seen == [{}, {"reasoning_effort": "none"}]
