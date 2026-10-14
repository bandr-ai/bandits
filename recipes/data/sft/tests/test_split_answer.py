"""Split generation: the generator writes the world, the agent answers blind; offline fakes per role."""
from __future__ import annotations

import json

import pytest
from test_pipeline import seed

from bandits_sft.analyzer import generate_analyzer, plan_jobs
from bandits_sft.config import load_config
from bandits_sft.llm import set_llm_factory
from bandits_sft.schema import from_sharegpt

WORLD = {"messages": [
    {"role": "user", "content": "cancel #W2"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"order_id": "#W2"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"order_id\": \"#W2\", \"status\": \"shipped\"}"},
    {"role": "assistant", "content": "written by the generator, must be dropped"}]}


def fakes(agent_replies: list[dict], prompts: list):
    def factory(cfg, role):
        def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
            prompts.append((role, messages))
            if role == "generator":
                return {"content": json.dumps(WORLD), "tool_calls": []}
            if role == "agent":
                return agent_replies.pop(0)
            return {"content": json.dumps({"result": "{\"status\": \"shipped\", \"carrier\": \"UPS\"}"}), "tool_calls": []}
        return type("F", (), {"chat": chat})()
    set_llm_factory(factory)


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "features": {"split_answer": True},
                             "answer_samples": 3, "models": {"default": {"model": "fake"}}}))
    yield load_config(p)
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def job():
    return {"job_id": "j1", "seed_id": "s0", "variation": {"name": "shipped", "change": "order is shipped",
                                                           "teaches": "SECRET-OUTCOME", "correct_when": "SECRET-OUTCOME"}}


CALL = {"content": "", "tool_calls": [{"id": "c9", "name": "get_order", "arguments": {"order_id": "#W2"}}]}


def test_agent_acts_with_simulated_results_and_the_outcome_stays_hidden(cfg):
    from bandits_sft.banks import ObsBank
    prompts = []
    fakes([CALL, {"content": "It shipped with UPS, so I cannot cancel it.", "tool_calls": []}], prompts)
    s = from_sharegpt(seed(0), 0)
    t = generate_analyzer(cfg, job(), s, "{seed_trace}{variation}{generation_id}", ObsBank.from_traces([s]))
    roles = [m["role"] for m in t["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"] and not t["meta"]["format_issues"]
    assert t["messages"][-1]["content"] == "It shipped with UPS, so I cannot cancel it." and "UPS" in t["messages"][-2]["content"]
    assert t["meta"]["answer_check"] == {"rounds": 1, "simulated_calls": 1}
    gen_prompt = next(m for r, m in prompts if r == "generator")[0]["content"]
    sim_prompt = next(m for r, m in prompts if r == "tool_sim")[1]["content"]
    assert "SECRET-OUTCOME" not in gen_prompt + sim_prompt and "Split mode" in gen_prompt and "order is shipped" in sim_prompt
    agent_msgs = [m for r, ms in prompts if r == "agent" for m in ms]
    assert agent_msgs and all("must be dropped" not in (m.get("content") or "") for m in agent_msgs)


def test_agent_that_never_answers_drops_the_trace(cfg):
    from bandits_sft.banks import ObsBank
    fakes([CALL] * 5, [])
    s = from_sharegpt(seed(0), 0)
    t = generate_analyzer(cfg, job(), s, "{seed_trace}{variation}{generation_id}", ObsBank.from_traces([s]))
    assert t["meta"]["format_issues"][0].startswith("agent_needs_more_evidence")


def test_family_cap_spreads_jobs():
    s = [from_sharegpt(seed(i), i) for i in range(4)]
    vs = [{"name": f"flip__{x['id']}", "eligible_seeds": [x["id"]], "applies_to": ["all"]} for x in s] + \
         [{"name": f"other{i}", "eligible_seeds": [x["id"]], "applies_to": ["all"]} for i, x in enumerate(s)]
    jobs = plan_jobs("b", s, 8, vs, {}, [], max_per_seed=2, max_per_variation=4, max_per_family=1)
    assert sum(j["variation"]["name"].startswith("flip") for j in jobs) == 1


def test_agents_without_tools_get_real_inputs_from_other_seeds():
    import random

    from bandits_sft.banks import ObsBank, format_input_examples
    a = {"id": "a", "messages": [{"role": "user", "content": "[DOCUMENT] policy A"}, {"role": "assistant", "content": "x"}]}
    b = {"id": "b", "messages": [{"role": "user", "content": "[DOCUMENT] policy B"}, {"role": "assistant", "content": "y"}]}
    ex = ObsBank.from_traces([a, b]).input_examples(3, random.Random(0), exclude_seed="a")
    assert ex == ["[DOCUMENT] policy B"] and "policy B" in format_input_examples(ex)


def test_result_format_learns_the_harness_prefix_from_seeds():
    from bandits_sft.banks import ResultFormat
    seeds = [{"messages": [{"role": "tool", "content": "[r:1] {\"a\": 1}"}]}]
    f = ResultFormat(r"^\[r:(\d+)\] ", seeds)
    assert f.apply('{"b": 2}', 3) == '[r:3] {"b": 2}'
    assert f.apply('[r:9] {"b": 2}', 4) == '[r:4] {"b": 2}'  # a wrong index written by the simulator is replaced
    assert f.strip('[r:2] x') == "x"
    assert ResultFormat(r"^RESULT: ", [{"messages": [{"role": "tool", "content": "RESULT: ok"}]}]).apply("x", 2) == "RESULT: x"
    assert ResultFormat(r"^\[r:(\d+)\] ", [{"messages": [{"role": "tool", "content": "no prefix"}]}]).apply("x", 1) == "x"
    assert ResultFormat(None, seeds).apply("x", 1) == "x"


def test_rollout_prefixes_simulated_results_and_retries_the_final_round(cfg):
    from bandits_sft.rollout import agent_rollout
    prompts = []
    fakes([CALL, CALL, {"content": "done", "tool_calls": []}], prompts)
    prefix = {"system": "s", "tools": [], "messages": [{"role": "user", "content": "q"},
              {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_order", "arguments": {}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "[r:1] {}"}]}
    added, info = agent_rollout(cfg, prefix, "sit", lambda n: "", max_rounds=1,
                                format_result=lambda c, n: f"[r:{n}] {c}")
    assert added[1]["content"].startswith("[r:2] ") and added[-1]["content"] == "done" and info["rounds"] == 2
