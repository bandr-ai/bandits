"""Offline tests: a fake LLM stands in for every role, so CI needs no key and no network."""
from __future__ import annotations

import json

import pytest

from simia_plus import cli
from simia_plus.config import load_config
from simia_plus.llm import extract_json, set_llm_factory
from simia_plus.schema import from_sharegpt, load_trace, to_sharegpt
from simia_plus.simia_text import build_sample_text, parse_simia_text
from simia_plus.state import apply_ops, state_match
from simia_plus.verify import provenance_check, rule_check

TOOLS = [{"name": "get_order", "description": "Look up an order",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
         {"name": "cancel_order", "description": "Cancel a pending order",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}]


def seed(i: int, conversational: bool = True) -> dict:
    oid = f"#W{1000 + i}"
    conv = [{"from": "human", "value": f"Hi, please cancel order {oid}."},
            {"from": "function_call", "value": json.dumps({"name": "get_order", "arguments": {"order_id": oid}})},
            {"from": "observation", "value": json.dumps({"order_id": oid, "status": "pending"})},
            {"from": "gpt", "value": f"Order {oid} is pending. Shall I cancel it?"}]
    if conversational:
        conv += [{"from": "human", "value": "yes"},
                 {"from": "function_call", "value": json.dumps({"name": "cancel_order", "arguments": {"order_id": oid}})},
                 {"from": "observation", "value": json.dumps({"order_id": oid, "status": "cancelled"})},
                 {"from": "gpt", "value": f"Done, {oid} is cancelled."}]
    return {"id": f"s{i}", "system": "You are a store agent. Confirm before cancelling.", "tools": json.dumps(TOOLS),
            "conversations": conv}


SIMIA_OUT = """HUMAN: hey can u cancel #W2222
FUNCTION_CALL: <think>
Need the order first. I will call the function get_order.
</think>
{"name": "get_order", "arguments": {"order_id": "#W2222"}}
OBSERVATION: {"order_id": "#W2222", "status": "pending"}
ASSISTANT: #W2222 is pending. Cancel it?
HUMAN: yes
FUNCTION_CALL: {"name": "cancel_order", "arguments": {"order_id": "#W2222"}}
OBSERVATION: {"order_id": "#W2222", "status": "cancelled"}
ASSISTANT: Cancelled #W2222."""


class FakeLLM:
    """Answers by role and prompt content."""

    def __init__(self, role: str):
        self.role = role
        self.calls = 0

    def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        text = messages[-1]["content"] if messages else ""
        sys = messages[0]["content"] if messages else ""
        if self.role == "generator":
            return {"content": SIMIA_OUT, "tool_calls": []}
        if self.role == "spec":
            n = int(text.split("Write ")[1].split(" scenario")[0])
            specs = [{"strategy": "x", "goal": f"cancel order #W3{k:03d}", "user_facts": {"order_id": f"#W3{k:03d}"},
                      "initial_state": {"orders": {f"#W3{k:03d}": {"status": "pending"}}},
                      "expected_final_state": {"orders": {f"#W3{k:03d}": {"status": "cancelled"}}},
                      "expected_behavior": "look up, confirm, cancel", "difficulty": "easy"} for k in range(n)]
            return {"content": json.dumps({"specs": specs}), "tool_calls": []}
        if self.role == "user_sim":
            n_user = sum(1 for m in messages if m["role"] == "assistant")
            return {"content": ["cancel my order #W3000 pls", "yes go ahead", "[DONE]"][min(n_user, 2)], "tool_calls": []}
        if self.role == "agent":
            convo = [m for m in messages if m["role"] != "system"]
            n_tools = sum(1 for m in convo if m["role"] == "tool")
            last_user = next(m["content"] for m in reversed(convo) if m["role"] == "user")
            if n_tools == 0:
                return {"content": "", "tool_calls": [{"id": "a1", "name": "get_order", "arguments": {"order_id": "#W3000"}}]}
            if "yes" in last_user and n_tools == 1:
                return {"content": "", "tool_calls": [{"id": "a2", "name": "cancel_order", "arguments": {"order_id": "#W3000"}}]}
            if convo[-1]["role"] == "tool" and n_tools == 1:
                return {"content": "#W3000 is pending. Cancel it?", "tool_calls": []}
            return {"content": "Cancelled #W3000.", "tool_calls": []}
        if self.role == "tool_sim" and "You are a simulation environment" in text:  # Simia-RL env
            latest = text.split("RL model's latest response:")[1].split("Requirements:")[0]
            if "<tool_call>" in latest:
                status = "cancelled" if "cancel_order" in latest else "pending"
                return {"content": json.dumps({"order_id": "#W1000", "status": status}), "tool_calls": []}
            return {"content": "yes" if "Cancel it?" in latest else "[TERMINATE]", "tool_calls": []}
        if self.role == "tool_sim":
            if "cancel_order(" in sys + text:
                return {"content": json.dumps({"result": {"order_id": "#W3000", "status": "cancelled"},
                                               "ops": [{"op": "set", "path": "orders.#W3000.status", "value": "cancelled"}]}),
                        "tool_calls": []}
            return {"content": json.dumps({"result": {"order_id": "#W3000", "status": "pending"}, "ops": []}), "tool_calls": []}
        if self.role == "judge":
            return {"content": json.dumps({"task_success": True, "final_state": {"orders": {"#W3000": {"status": "cancelled"}}},
                                           "obs_consistent": True, "user_realistic": True, "policy_followed": True,
                                           "hallucinations": [], "checklist_failed": [], "bad_steps": []}), "tool_calls": []}
        if self.role == "seed_check":
            return {"content": json.dumps({"complete": True, "logical": True, "well_formatted": True}), "tool_calls": []}
        raise AssertionError(self.role)


@pytest.fixture
def fake_llms():
    made: dict[str, FakeLLM] = {}

    def factory(cfg, role):
        made[role] = FakeLLM(role)
        return made[role]

    set_llm_factory(factory)
    yield made
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def write_cfg(tmp_path, features: dict, target: int, n_seeds: int = 3, **extra) -> str:
    seeds_path = tmp_path / "seeds.jsonl"
    seeds_path.write_text("\n".join(json.dumps(seed(i)) for i in range(n_seeds)) + "\n")
    roles = ["default", "spec", "generator", "agent", "user_sim", "tool_sim", "judge", "seed_check"]
    cfg = {"seeds_path": str(seeds_path), "out_dir": str(tmp_path / "out"), "target_count": target,
           "overgen": 1.0, "workers": 2, "features": features, "models": {r: {"model": "fake"} for r in roles}, **extra}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_sharegpt_roundtrip():
    t = from_sharegpt(seed(0), 0)
    assert [m["role"] for m in t["messages"]] == ["user", "assistant", "tool", "assistant",
                                                  "user", "assistant", "tool", "assistant"]
    back = from_sharegpt(to_sharegpt(t), 0)
    assert back["messages"] == t["messages"]


def test_parse_simia_text_matches_simia_roles():
    conv = parse_simia_text(SIMIA_OUT)
    assert [c["from"] for c in conv] == ["human", "function_call", "observation", "gpt",
                                         "human", "function_call", "observation", "gpt"]
    t = from_sharegpt({"id": "g", "tools": TOOLS, "conversations": conv}, 0)
    assert t["messages"][1]["reasoning"].startswith("Need the order")
    assert rule_check(t, 2) == []


def test_sample_text_is_simia_format():
    txt = build_sample_text(load_trace(seed(0), 0))
    assert txt.startswith("SYSTEM: ") and "\n\nFUNCTION_CALL: " in txt and "\n\nOBSERVATION: " in txt


def test_rule_check_flags_bad_traces():
    t = from_sharegpt(seed(0), 0)
    t["messages"][1]["tool_calls"][0]["name"] = "delete_everything"
    t["messages"].pop()  # no final assistant reply
    issues = rule_check(t, 2)
    assert any("unknown tool" in i for i in issues)
    assert any("end with an assistant" in i for i in issues)


def test_provenance_flags_invented_ids():
    t = from_sharegpt(seed(0), 0)
    assert provenance_check(t, None) == []
    t["messages"][5]["tool_calls"][0]["arguments"]["order_id"] = "#W9999"
    assert provenance_check(t, None) == ["cancel_order: #W9999"]


def test_state_ops_and_match():
    s, errs = apply_ops({"orders": {"o1": {"status": "pending"}}},
                        [{"op": "set", "path": "orders.o1.status", "value": "cancelled"},
                         {"op": "append", "path": "log", "value": "x"}, {"op": "bogus"}])
    assert s["orders"]["o1"]["status"] == "cancelled" and s["log"] == ["x"] and len(errs) == 1
    assert state_match({"orders": {"o1": {"status": "cancelled"}}}, s) == 1.0
    assert state_match({"orders": {"o1": {"status": "pending"}}}, s) == 0.0


def test_extract_json_handles_fences_and_prose():
    assert extract_json('sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('result {"a": [1, 2]} trailing') == {"a": [1, 2]}


def test_baseline_is_plain_simia(tmp_path, fake_llms):
    """All features off: one generator call per job, no spec/persona/judge, every seed used, target met."""
    cfg_path = write_cfg(tmp_path, {}, target=6)
    cli.main(["run", "--config", cfg_path])
    cfg = load_config(cfg_path)
    report = json.loads((cfg.out / "final/report.json").read_text())
    assert set(fake_llms) == {"generator"} and fake_llms["generator"].calls == 6
    gen = [json.loads(l) for l in (cfg.out / "generated.jsonl").read_text().splitlines()]
    assert {g["meta"]["seed_id"] for g in gen} == {"s0", "s1", "s2"}
    assert all(g["meta"]["strategy"] == "new_scenario" and g["meta"]["persona"] is None for g in gen)
    # identical fake outputs: Simia's exact dedup collapses them to one
    assert report["final"] == 1 and report["dedup"]["exact_dup"] == 5


def test_full_pipeline_with_loop(tmp_path, fake_llms):
    feats = {"strategies": True, "spec": True, "persona": True, "failure": True, "retrieval": True,
             "loop": True, "provenance": True, "judge": True, "near_dedup": True, "check_seeds": True}
    cfg_path = write_cfg(tmp_path, feats, target=4, loop_frac=1.0, failure_rate=0.0)
    cli.main(["run", "--config", cfg_path])
    cfg = load_config(cfg_path)
    ver = [json.loads(l) for l in (cfg.out / "verified.jsonl").read_text().splitlines()]
    assert ver and all(v["meta"]["mode"] == "split" for v in ver)
    t = ver[0]
    assert t["meta"]["final_state"]["orders"]["#W3000"]["status"] == "cancelled"
    assert t["meta"]["verify"]["kept"], t["meta"]["verify"]
    assert t["meta"]["verify"]["judge"]["state_match"] == 0.0 or t["meta"]["spec"]["expected_final_state"]
    assert {"spec", "agent", "user_sim", "tool_sim", "judge", "seed_check"} <= set(fake_llms)
    assert (cfg.out / "final/seeds_plus_synthetic_sharegpt.jsonl").exists()


def test_resume_skips_done_work(tmp_path, fake_llms):
    cfg_path = write_cfg(tmp_path, {}, target=3)
    cli.main(["run", "--config", cfg_path])
    first = fake_llms["generator"].calls
    cli.main(["generate", "--config", cfg_path])
    assert fake_llms["generator"].calls == first  # nothing regenerated


def test_persona_only_on_conversational_seeds(tmp_path, fake_llms):
    seeds_path = tmp_path / "seeds.jsonl"
    seeds_path.write_text(json.dumps(seed(0, conversational=False)) + "\n" + json.dumps(seed(1)) + "\n")
    cfg = {"seeds_path": str(seeds_path), "out_dir": str(tmp_path / "out"), "target_count": 4, "overgen": 1.0,
           "workers": 1, "features": {"persona": True}, "models": {"default": {"model": "fake"}}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    cli.main(["ingest", "--config", str(p)])
    jobs = cli.cmd_plan(load_config(p))
    assert all("persona" not in j for j in jobs if j["seed_id"] == "s0")
    assert all("persona" in j for j in jobs if j["seed_id"] == "s1")


def test_loop_requires_spec(tmp_path):
    cfg_path = write_cfg(tmp_path, {"loop": True}, target=2)
    with pytest.raises(ValueError, match="needs features.spec"):
        load_config(cfg_path)


def test_simia_env_mode(tmp_path, fake_llms):
    """Simia-RL's simulator (user + tools in one model, seed as reference) driving a teacher agent."""
    cfg_path = write_cfg(tmp_path, {"loop": True}, target=3, loop_frac=1.0, loop_style="simia_env")
    cli.main(["run", "--config", cfg_path])
    cfg = load_config(cfg_path)
    gen = [json.loads(line) for line in (cfg.out / "generated.jsonl").read_text().splitlines()]
    t = next(g for g in gen if g["meta"]["seed_id"] == "s0")
    assert t["meta"]["mode"] == "simia_env" and t["meta"]["ended"] == "terminate"
    assert [m["role"] for m in t["messages"]] == ["user", "assistant", "tool", "assistant", "user",
                                                  "assistant", "tool", "assistant"]
    assert t["messages"][0]["content"] == "Hi, please cancel order #W1000."  # Simia-RL: seed's first message
    assert "spec" not in fake_llms and "user_sim" not in fake_llms


def test_langchain_style_messages_and_named_tools():
    rec = {"session_id": "x1", "tools": ["get_order"],
           "final_messages": [{"role": "system", "content": "be careful"},
                              {"role": "human", "content": "check #W1"},
                              {"role": "ai", "content": [{"type": "thinking", "thinking": "look it up"}],
                               "tool_calls": [{"name": "get_order", "args": {"order_id": "#W1"}, "id": "t1", "type": "tool_call"}]},
                              {"role": "tool", "content": "{\"status\": \"pending\"}"},
                              {"role": "ai", "content": [{"type": "text", "text": "It is pending."}]}]}
    t = load_trace(rec, 0, "auto", defs={"get_order": {"type": "function", "function": TOOLS[0]}})
    assert t["id"] == "x1" and t["system"] == "be careful" and t["tools"][0]["name"] == "get_order"
    assert [m["role"] for m in t["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert t["messages"][1]["reasoning"] == "look it up" and t["messages"][2]["tool_call_id"] == "t1"
    assert rule_check(t, 2) == []
    with pytest.raises(ValueError, match="only a name"):
        load_trace(rec, 0, "auto")


def test_simia_code_is_used():
    """Simia's own argument repair and markup filter run on generated FUNCTION_CALL turns."""
    from simia_plus.simia import fix_arguments_format, should_delete_conversation
    ok, fixed, _ = fix_arguments_format('{"name": "get_order", "arguments": "{\\"order_id\\": \\"#W1\\"}"}')
    assert ok and json.loads(fixed)["arguments"] == {"order_id": "#W1"}
    assert should_delete_conversation([{"from": "gpt", "value": "<tool_call>x"}])


def test_gateway_log_appends_final_response():
    rec = {"session_id": "g1", "tools": [TOOLS[0]],
           "final_messages": [{"role": "human", "content": "check #W1"},
                              {"role": "ai", "content": "", "tool_calls": [{"name": "get_order", "args": {"order_id": "#W1"}, "id": "t1"}]},
                              {"role": "tool", "content": "{}"}],
           "turns": [{"content": "", "tool_calls": [{"name": "get_order", "args": {"order_id": "#W1"}, "id": "t1"}]},
                     {"content": "Order #W1 is pending.", "tool_calls": None}]}
    t = load_trace(rec, 0)
    assert t["messages"][-1] == {"role": "assistant", "content": "Order #W1 is pending."}


def test_parallel_calls_roundtrip_as_consecutive_function_calls():
    t = load_trace({"id": "p", "tools": TOOLS, "messages": [
        {"role": "user", "content": "check #W1 and #W2"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "get_order", "arguments": {"order_id": "#W1"}},
                                                             {"id": "b", "name": "get_order", "arguments": {"order_id": "#W2"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "one"}, {"role": "tool", "tool_call_id": "b", "content": "two"},
        {"role": "assistant", "content": "done"}]}, 0)
    sg = to_sharegpt(t)
    assert [c["from"] for c in sg["conversations"]] == ["human", "function_call", "function_call", "observation", "observation", "gpt"]
    assert not any("[{" in c["value"] for c in sg["conversations"])
    back = from_sharegpt(sg, 0)
    assert len(back["messages"][1]["tool_calls"]) == 2
    assert [m["content"] for m in back["messages"] if m["role"] == "tool"] == ["one", "two"]
    assert rule_check(back, 2) == []


def test_unusable_generation_is_retried(tmp_path, fake_llms):
    outputs = iter(["FUNCTION_CALL: {\"name\": \"get_order\", \"arguments\": {}}", SIMIA_OUT])

    class Flaky(FakeLLM):
        def chat(self, messages, tools=None, json_mode=False):
            self.calls += 1
            return {"content": next(outputs), "tool_calls": []}

    set_llm_factory(lambda cfg, role: fake_llms.setdefault(role, Flaky(role)))
    cfg_path = write_cfg(tmp_path, {}, target=1, n_seeds=1, generation_attempts=3)
    cli.main(["run", "--config", cfg_path])
    gen = [json.loads(line) for line in (load_config(cfg_path).out / "generated.jsonl").read_text().splitlines()]
    assert gen[0]["meta"]["attempts"] == 2 and gen[0]["meta"]["discarded_attempts"][0]["problem"] == "no_leading_user_turn"
    assert fake_llms["generator"].calls == 2


def test_function_call_with_plain_text_reasoning_is_parsed():
    from simia_plus.schema import parse_function_call
    calls, why = parse_function_call('I need the order first. I will call get_order.\n{"name": "get_order", "arguments": {"order_id": "#W1"}}')
    assert calls == [{"name": "get_order", "arguments": {"order_id": "#W1"}}] and why.startswith("I need the order")
    assert parse_function_call("no json here") == ([], "")


def test_orphan_tool_results_are_rejected():
    t = from_sharegpt(seed(0), 0)
    t["messages"][1]["tool_calls"] = []  # call lost, result kept
    issues = rule_check(t, 2)
    assert any("without a matching call" in i for i in issues) or any("empty assistant turn" in i for i in issues)


def test_inserted_prompt_blocks_have_no_template_escapes(tmp_path, fake_llms):
    from simia_plus.banks import ObsBank, UserBank
    from simia_plus.config import load_config
    from simia_plus.generate import simia_prompt
    cfg = load_config(write_cfg(tmp_path, {"retrieval": True, "failure": True}, target=1))
    s = load_trace(seed(0), 0)
    p = simia_prompt(cfg, {"job_id": "j", "seed_id": "s0", "strategy": "new_scenario", "failure": {"at_call": 1, "type": "timeout"}},
                     s, {"s0": s}, ObsBank.from_traces([s]), UserBank.from_traces([s]))
    assert "{{" not in p and "}}" not in p
