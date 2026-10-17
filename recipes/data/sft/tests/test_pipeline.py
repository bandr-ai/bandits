"""Offline tests: a fake LLM stands in for every role, so CI needs no key and no network."""
from __future__ import annotations

import json

import pytest

from bandits_sft import cli
from bandits_sft.config import load_config
from bandits_sft.llm import extract_json, set_llm_factory
from bandits_sft.prompts import JUDGE_CHECKS
from bandits_sft.schema import from_sharegpt, load_trace, render, to_sharegpt
from bandits_sft.state import apply_ops, state_match
from bandits_sft.trace_text import build_sample_text, parse_trace_text
from bandits_sft.verify import provenance_check, rule_check
from bandits_sft.verify import verify as verify_fn

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


BASELINE_OUT = """HUMAN: hey can u cancel #W2222
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


ANALYZER_OUT = json.dumps({"messages": [
    {"role": "user", "content": "cancel #W2222 please"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"order_id": "#W2222"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "{\"order_id\": \"#W2222\", \"status\": \"pending\"}"},
    {"role": "assistant", "content": "{\"status\": \"pending\"}"}]})


def judged_turns(prompt: str) -> list[dict]:
    """A "justified" entry for every assistant message in the judge prompt's <generated> block."""
    import re
    block = prompt.split("<generated", 1)[1].split("</generated>", 1)[0]
    nums = sorted({int(n) for n in re.findall(r"^\[(\d+)\] (?:ASSISTANT|TOOL_CALL)", block, re.MULTILINE)})
    return [{"message": n, "verdict": "justified", "relies_on": [n - 1], "reason": "r"} for n in nums]


class FakeLLM:
    """Answers by role and prompt content."""

    def __init__(self, role: str):
        self.role = role
        self.calls = 0
        self.seen = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        self.seen.append(messages)
        text = messages[-1]["content"] if messages else ""
        sys = messages[0]["content"] if messages else ""
        if self.role == "generator" and "Output format (fixed)" in text:
            return {"content": ANALYZER_OUT, "tool_calls": []}
        if self.role == "generator":
            return {"content": BASELINE_OUT, "tool_calls": []}
        if self.role == "analyzer":
            vs = [{"name": f"var{k}", "applies_to": ["all"] if k else ["S1"], "requires_tools": ["get_order"] if k == 2 else [],
                   "teaches": "t", "change": "c", "preserve": "p", "dependencies": "d", "correct_when": "w"} for k in range(3)]
            ids = __import__("re").findall(r'<seed id="([^"\n]+)"', text)
            for v in vs:
                v["signals"] = ["lookup"]
                v["eligible_seeds"] = ids
                v["basis"] = "seed"
                v["decision_type"] = f"type{vs.index(v)}"
                v["answer_transition"] = "unchanged"
            return {"content": json.dumps({"observed_contracts": "Tool result formats observed in supplied seeds", "variations": vs,
                                           "signals": [{"id": "lookup", "sources": __import__("re").findall(r'<seed id="([^"\n]+)"', text), "teaches": "lookup before answer", "decision": "get_order", "driving_evidence": "user order id"}],
                                           "template": "Seed:\n{seed_trace}\nApply: {variation}\nPolicy: {system_prompt}\nTools: {tools}\n{generation_id}",
                                           "rationale": "seeds are order-support chats"}), "tool_calls": []}
        if self.role == "diagnose":
            return {"content": json.dumps({"notes": "- all fine"}), "tool_calls": []}
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
        if self.role == "tool_sim" and "You are a simulation environment" in text:  # reference simulation env
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
        if self.role == "judge" and "<generated" in text:  # seed-expansion judge: grounded verdict per check
            ev = [{"message": "0", "excerpt": "cancel #W2222 please"}]
            checks = {k: {"verdict": "pass", "evidence": ev, "reason": "r"} for k in JUDGE_CHECKS}
            return {"content": json.dumps({"checks": checks, "checklist": [], "hallucinations": [], "turns": judged_turns(text)}),
                    "tool_calls": []}
        if self.role == "judge":
            return {"content": json.dumps({"task_success": True, "final_state": {"orders": {"#W3000": {"status": "cancelled"}}},
                                           "only_declared_change": True, "dependencies_updated": True,
                                           "tool_contracts_respected": True,
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
    roles = ["default", "spec", "generator", "agent", "user_sim", "tool_sim", "judge", "seed_check", "analyzer", "diagnose"]
    cfg = {"seeds_path": str(seeds_path), "out_dir": str(tmp_path / "out"), "target_count": target,
           "overgen": 1.0, "workers": 2, "features": {"consistency_check": False, **features},  # its own tests: test_consistency
           "models": {r: {"model": "fake"} for r in roles}, **extra}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_sharegpt_roundtrip():
    t = from_sharegpt(seed(0), 0)
    assert [m["role"] for m in t["messages"]] == ["user", "assistant", "tool", "assistant",
                                                  "user", "assistant", "tool", "assistant"]
    back = from_sharegpt(to_sharegpt(t), 0)
    assert back["messages"] == t["messages"]


def test_parse_trace_text_matches_baseline_roles():
    conv = parse_trace_text(BASELINE_OUT)
    assert [c["from"] for c in conv] == ["human", "function_call", "observation", "gpt",
                                         "human", "function_call", "observation", "gpt"]
    t = from_sharegpt({"id": "g", "tools": TOOLS, "conversations": conv}, 0)
    assert t["messages"][1]["reasoning"].startswith("Need the order")
    assert rule_check(t, 2) == []


def test_sample_text_is_baseline_format():
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


def test_baseline_is_plain_baseline(tmp_path, fake_llms):
    """All features off: one generator call per job, no spec/persona/judge, every seed used, target met."""
    cfg_path = write_cfg(tmp_path, {}, target=6)
    cli.main(["run", "--config", cfg_path])
    cfg = load_config(cfg_path)
    report = json.loads((cfg.out / "final/report.json").read_text())
    assert set(fake_llms) == {"generator"} and fake_llms["generator"].calls == 6
    gen = [json.loads(l) for l in (cfg.out / "generated.jsonl").read_text().splitlines()]
    assert {g["meta"]["seed_id"] for g in gen} == {"s0", "s1", "s2"}
    assert all(g["meta"]["strategy"] == "new_scenario" and g["meta"]["persona"] is None for g in gen)
    # identical fake outputs: baseline's exact dedup collapses them to one
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


def test_reference_env_mode(tmp_path, fake_llms):
    """reference simulation's simulator (user + tools in one model, seed as reference) driving a teacher agent."""
    cfg_path = write_cfg(tmp_path, {"loop": True}, target=3, loop_frac=1.0, loop_style="reference_env")
    cli.main(["run", "--config", cfg_path])
    cfg = load_config(cfg_path)
    gen = [json.loads(line) for line in (cfg.out / "generated.jsonl").read_text().splitlines()]
    t = next(g for g in gen if g["meta"]["seed_id"] == "s0")
    assert t["meta"]["mode"] == "reference_env" and t["meta"]["ended"] == "terminate"
    assert [m["role"] for m in t["messages"]] == ["user", "assistant", "tool", "assistant", "user",
                                                  "assistant", "tool", "assistant"]
    assert t["messages"][0]["content"] == "Hi, please cancel order #W1000."  # reference simulation: seed's first message
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


def test_baseline_code_is_used():
    """baseline's own argument repair and markup filter run on generated FUNCTION_CALL turns."""
    from bandits_sft.upstream import fix_arguments_format, should_delete_conversation
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
    outputs = iter(["FUNCTION_CALL: {\"name\": \"get_order\", \"arguments\": {}}", BASELINE_OUT])

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
    from bandits_sft.schema import parse_function_call
    calls, why = parse_function_call('I need the order first. I will call get_order.\n{"name": "get_order", "arguments": {"order_id": "#W1"}}')
    assert calls == [{"name": "get_order", "arguments": {"order_id": "#W1"}}] and why.startswith("I need the order")
    assert parse_function_call("no json here") == ([], "")


def test_orphan_tool_results_are_rejected():
    t = from_sharegpt(seed(0), 0)
    t["messages"][1]["tool_calls"] = []  # call lost, result kept
    issues = rule_check(t, 2)
    assert any("without a matching call" in i for i in issues) or any("empty assistant turn" in i for i in issues)


def test_inserted_prompt_blocks_have_no_template_escapes(tmp_path, fake_llms):
    from bandits_sft.banks import ObsBank, UserBank
    from bandits_sft.config import load_config
    from bandits_sft.generate import baseline_prompt
    cfg = load_config(write_cfg(tmp_path, {"retrieval": True, "failure": True}, target=1))
    s = load_trace(seed(0), 0)
    p = baseline_prompt(cfg, {"job_id": "j", "seed_id": "s0", "strategy": "new_scenario", "failure": {"at_call": 1, "type": "timeout"}},
                     s, {"s0": s}, ObsBank.from_traces([s]), UserBank.from_traces([s]))
    assert '{{"' not in p and "{{..." not in p  # template escapes (nested JSON may legitimately end in "}}")


def test_analyzer_rounds_split_ledger_and_separate_heldout(tmp_path, fake_llms):
    cfg_path = write_cfg(tmp_path, {"judge": True, "judge_gate": True, "provenance": True}, target=2, n_seeds=5, holdout_frac=0.4,
                         analyzer_rounds=2, pilot_jobs=6, heldout_jobs_per_seed=1, outcome_fields=["status"], per_seed_variations=False)
    cli.main(["ingest", "--config", cfg_path])
    cfg = load_config(cfg_path)
    split = json.loads((cfg.out / "split.json").read_text())
    assert len(split["heldout"]) == 2 and len(split["dev"]) == 3
    banked = {e["seed_id"] for e in map(json.loads, (cfg.out / "obs_bank.jsonl").read_text().splitlines())}
    assert banked == set(split["dev"])  # held-out seeds stay out of the banks
    cli.main(["analyze", "--config", cfg_path])
    a = cfg.out / "analyzer"
    for v in ("v1", "v2"):
        for f in ("observed_contracts.md", "signals.json", "variations.json", "prompt.txt", "rationale.md", "jobs.jsonl", "verified.jsonl", "report.json"):
            assert (a / v / f).exists(), (v, f)
    assert (a / "v1" / "diagnosis.md").exists() and not (a / "v2" / "diagnosis.md").exists()
    assert jobs_signal_details(a / "v1" / "jobs.jsonl")
    assert "<previous_version number=\"1\">" in json.loads((a / "v2" / "analyzer_raw.json").read_text())["request"]  # signals are optional now
    assert '<previous_version number="1">' in json.loads((a / "v2" / "analyzer_raw.json").read_text())["request"]
    assert not (a / "heldout").exists()  # held-out runs only on request
    jobs = [json.loads(line) for line in (a / "v1" / "jobs.jsonl").read_text().splitlines()]
    pairs = [(j["seed_id"], j["variation"]["name"]) for j in jobs]
    assert len(set(pairs)) == len(pairs)  # 6 jobs over 3 seeds x 3 variations: no pair repeats
    ledger = [json.loads(line) for line in (a / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 9 and {r["version"] for r in ledger} == {1, 2}  # 3 seeds x 3 variations, no pair twice
    v2_pairs = {(j["seed_id"], j["variation"]["name"]) for j in map(json.loads, (a / "v2" / "jobs.jsonl").read_text().splitlines())}
    untried = {(sid, v) for sid in {p[0] for p in pairs} for v in ("var0", "var1", "var2")} - set(pairs)
    assert untried and untried <= v2_pairs  # the ledger steers v2 to the pairs v1 did not try first
    rep = json.loads((a / "v1" / "report.json").read_text())
    assert rep["format_ok"] == 6 and rep["kept"] == 6 and rep["outcomes_kept"] == {"status=pending": 1}
    assert rep["fidelity_all"]["context_parsed"] == "0/6" and set(rep["by_variation"]) <= {"var0", "var1", "var2"}
    assert json.loads((a / "all_rounds.json").read_text())["kept_all_rounds"] == 9
    seen = " ".join(m["content"] for c in fake_llms["analyzer"].seen for m in c)
    assert all(h not in seen for h in split["heldout"])
    judged = fake_llms["judge"].seen[0][-1]["content"]
    assert "<seed note=" in judged and '"name": "var' in judged and "Tool result formats observed in supplied seeds" in judged
    cli.main(["review", "--config", cfg_path])
    page = (a / "review.html").read_text()
    assert page.count('<details class="trace"') == 9 and "<ins>" in page
    cli.main(["heldout", "--config", cfg_path])
    cmp = json.loads((a / "heldout" / "comparison.json").read_text())
    assert set(cmp) == {"analyzer", "baseline_fixed"} and cmp["baseline_fixed"]["jobs"] == 2


def test_analyzer_resumes_from_saved_answer_without_prompt(tmp_path, fake_llms):
    # a stop after analyzer_raw.json is saved but before prompt.txt: the resume reuses the answer, paying nothing
    cfg_path = write_cfg(tmp_path, {}, target=2, n_seeds=5, holdout_frac=0.4, analyzer_rounds=1, pilot_jobs=2,
                         per_seed_variations=False)
    cli.main(["ingest", "--config", cfg_path])
    cli.main(["analyze", "--config", cfg_path])
    v1 = load_config(cfg_path).out / "analyzer" / "v1"
    template = (v1 / "prompt.txt").read_text()
    assert template
    for f in v1.iterdir():
        if f.is_file() and f.name != "analyzer_raw.json":
            f.unlink()
    paid = len(fake_llms["analyzer"].seen)
    cli.main(["analyze", "--config", cfg_path])
    assert (v1 / "prompt.txt").read_text() == template
    assert len(fake_llms["analyzer"].seen) == paid


def test_decontam_catches_short_eval_inputs_but_not_shared_template():
    from bandits_sft.verify import copied, ngrams
    tpl = "Please read the request below carefully and answer using the tools you have. Request:"
    assert copied("cancel order W1234 right now please, it is urgent", "cancel order W1234 right now")
    assert not copied("yes thanks", "yes")  # too short to call a copy
    assert not copied("the order W1234 now", "cancel order W1234 right now")  # not contained whole
    common = frozenset(ngrams(tpl))
    assert not copied(f"{tpl} where is my parcel", "read the request below carefully", common)  # template text only
    assert copied(f"{tpl} where is my parcel", "where is my parcel", common)


def test_tool_results_must_have_real_shapes_and_prefix():
    from bandits_sft.verify import tool_result_check
    t = from_sharegpt(seed(0), 0)
    shapes = {"get_order": [["order_id", "status"]], "*": [["status", "message"]]}
    assert tool_result_check(t, shapes, None) == []
    t["messages"][2]["content"] = json.dumps({"status": "success", "read": "{}"})
    assert "never returns" in tool_result_check(t, shapes, None)[0]
    t2 = from_sharegpt(seed(0), 0)
    for k, m in enumerate(m for m in t2["messages"] if m["role"] == "tool"):
        m["content"] = f"[r:{k + 1}] {m['content']}"
    assert tool_result_check(t2, shapes, r"^\[r:(\d+)\]\s*") == []
    t2["messages"][6]["content"] = t2["messages"][6]["content"].replace("[r:2]", "[r:7]")
    assert tool_result_check(t2, shapes, r"^\[r:(\d+)\]\s*") == ["tool result 2 (cancel_order) has index 7"]


def test_harness_arg_defaults_satisfy_required_args():
    t = from_sharegpt(seed(0), 0)
    t["messages"][1]["tool_calls"][0]["arguments"] = {}
    assert any("missing required argument order_id" in i for i in rule_check(t, 2))
    assert rule_check(t, 2, {"get_order": {"order_id": ""}}) == []


def test_fidelity_reports_changed_context_and_reuse():
    from bandits_sft.analyzer import decision_cells, fidelity
    s = from_sharegpt(seed(0), 0)
    s["messages"][0]["content"] = 'Failure context: {"platform": "WEB", "element": {"name": "Buy", "role": "button"}}'
    same = json.loads(json.dumps(s))
    assert fidelity(same, s, None)["no_op"] is True
    t = json.loads(json.dumps(s))
    t["messages"][0]["content"] = 'Failure context: {"platform": "WEB", "element": {"name": "Pay", "role": "button"}} hint'
    t["messages"][2]["content"] = '{"order_id": "#W1000", "status": "on_hold"}'
    f = fidelity(t, s, None)
    assert f["context_fields_changed"] == ["element.name"] and f["tool_results_reused_from_seed"] == 1 and not f["no_op"]
    assert len(decision_cells(t)) == 4


def test_analyzer_lint_and_strict_format():
    from bandits_sft.analyzer import lint, parse_output
    full = "x {seed_trace} {variation} {system_prompt} {tools} {generation_id}"
    assert lint(full) == []
    assert lint(full.replace("{generation_id}", "{foo}")) == ["missing required slot {generation_id}", "unknown slot {foo}"]
    assert parse_output('Plan: ...\n{"messages": []}')[1][0].startswith("format: not a single JSON")
    msgs, issues = parse_output(json.dumps({"messages": [{"role": "tool", "tool_call_id": "c1", "content": {"a": 1}}]}))
    assert msgs is None and "string content" in issues[0]


def test_result_before_its_call_is_rejected():
    t = from_sharegpt(seed(0), 0)
    t["messages"][1], t["messages"][2] = t["messages"][2], t["messages"][1]
    assert any("without a matching call" in i for i in rule_check(t, 2))


def test_answer_schema_and_judge_fail_closed(tmp_path, fake_llms):
    from bandits_sft.verify import answer_check, system_key, verify
    t = from_sharegpt(seed(0), 0)
    schema = {"type": "object", "required": ["status"], "properties": {"status": {"enum": ["pending", "cancelled"]}}}
    assert answer_check(t, {system_key(t["system"]): schema}) == ["final answer is not a bare JSON value"]
    t["messages"][-1]["content"] = '{"status": "gone"}'
    assert answer_check(t, {system_key(t["system"]): schema}) == ["answer.status='gone' not in ['pending', 'cancelled']"]
    cfg = load_config(write_cfg(tmp_path, {"judge": True, "judge_gate": True}, target=1, keep_failures=False))
    set_llm_factory(lambda c, role: type("J", (), {"chat": lambda self, *a, **k: {"content": json.dumps({"task_success": True}),
                                                                                  "tool_calls": []}})())
    assert verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]["kept"] is False


def test_judge_flags_are_critique_not_noise():
    from bandits_sft.verify import judge_flags
    assert judge_flags(None) == [] and judge_flags({"error": "token limit"}) == []
    ok = {"checks": {"a": {"verdict": "pass"}, "tool_contracts_respected": {"verdict": "not_applicable"},
                     "case_consistent": {"verdict": "not_applicable"}, "c": {"verdict": "cannot_determine"}},
          "structural_na": ["tool_contracts_respected"],
          "checklist_failed": [], "hallucinations": [], "turn_issues": [], "task_success": True}
    assert judge_flags(ok) == []  # structural not_applicable / cannot_determine are not criticism
    # a judge cannot declare correctness or policy "not applicable" to skip evaluating it
    assert judge_flags({**ok, "checks": {"task_success": {"verdict": "not_applicable"}}}) == ["check:task_success:not_applicable"]
    bad = {**ok, "checks": {"a": {"verdict": "fail"}}, "turn_issues": ["turn 3 unjustified"], "hallucinations": ["x"]}
    assert judge_flags(bad) == ["check:a", "hallucinations", "turn_issues"]


def test_judge_critique_only_never_gates(tmp_path, fake_llms):
    # judge_gate=false: the judge's verdict (even a bad one or a failure) is recorded for the diagnosis, `kept` stays rule-based
    cfg = load_config(write_cfg(tmp_path, {"judge": True, "judge_gate": False}, target=1, keep_failures=False))
    bad = {"task_success": False, "obs_consistent": False, "policy_followed": False, "hallucinations": ["x"], "checklist_failed": ["y"]}
    set_llm_factory(lambda c, role: type("J", (), {"chat": lambda self, *a, **k: {"content": json.dumps(bad), "tool_calls": []}})())
    v = verify_fn(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is True and v["judge"]["hallucinations"] == ["x"]

    def boom(self, *a, **k):
        raise RuntimeError("token limit")
    set_llm_factory(lambda c, role: type("J", (), {"chat": boom})())
    v = verify_fn(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]
    assert v["kept"] is True and "error" in v["judge"] and "eval_error" not in v


def test_variation_prerequisites_and_expansion_judge_fails_closed(tmp_path):
    from bandits_sft.analyzer import applicable
    from bandits_sft.verify import system_key, verify
    s = from_sharegpt(seed(0), 0)
    labels = {"S1": system_key(s["system"])}
    vs = [{"name": "a", "applies_to": ["all"], "requires_tools": []},
          {"name": "b", "applies_to": ["S1"], "requires_tools": ["get_order"]},
          {"name": "c", "applies_to": ["all"], "requires_tools": ["refund_order"]},
          {"name": "d", "applies_to": ["S2"], "requires_tools": []}]
    assert [v["name"] for v in applicable(vs, s, labels)] == ["a", "b"]
    cfg = load_config(write_cfg(tmp_path, {"judge": True, "judge_gate": True}, target=1, keep_failures=False))
    old = {"task_success": True, "obs_consistent": True, "policy_followed": True, "hallucinations": [], "checklist_failed": []}
    set_llm_factory(lambda c, role: type("J", (), {"chat": lambda self, *a, **k: {"content": json.dumps(old), "tool_calls": []}})())
    assert verify(cfg, from_sharegpt(seed(0), 0), set())["meta"]["verify"]["kept"] is True
    ctx = {"seed": s, "variation": vs[0], "harness": "h"}
    assert verify(cfg, from_sharegpt(seed(0), 0), set(), ctx)["meta"]["verify"]["kept"] is False  # fidelity fields missing


def test_expansion_judge_needs_verifiable_excerpts(tmp_path):
    from bandits_sft.verify import verify
    cfg = load_config(write_cfg(tmp_path, {"change_gate": False, "judge": True, "judge_gate": True}, target=1, keep_failures=False))
    s = from_sharegpt(seed(0), 0)
    ctx = {"seed": s, "variation": {"name": "a"}, "harness": "h"}

    def run(excerpt: str, verdict: str = "pass") -> dict:
        checks = {k: {"verdict": verdict, "evidence": [{"message": "0", "excerpt": excerpt}], "reason": "r"} for k in JUDGE_CHECKS}
        set_llm_factory(lambda c, role: type("J", (), {"chat": lambda self, m, *a, **k: {"content": json.dumps(
            {"checks": checks, "checklist": [], "hallucinations": [], "turns": judged_turns(m[-1]["content"])}), "tool_calls": []}})())
        return verify(cfg, from_sharegpt(seed(0), 0), set(), ctx)["meta"]["verify"]

    assert run("please cancel   order #W1000")["kept"] is True  # whitespace/case-insensitive match
    v = run("the order was refunded")  # quote that occurs nowhere
    assert v["kept"] is False and "task_success" in v["judge"]["unverified_excerpts"]
    assert run("please cancel order #W1000", verdict="cannot_determine")["kept"] is False


def test_judge_excerpts_tolerate_quoting_style_not_invention():
    from bandits_sft.verify import _grounded, _norm, _trigrams
    t = from_sharegpt(seed(0), 0)
    t["messages"][2]["content"] = '[r:1] {"status": "success", "read": "{\\"answer\\": \\"Item visible; value fully shown\\"}"}'
    corpus = _norm(render(t, max_obs_chars=10**6))
    grams = _trigrams(corpus)

    def ok(x):
        return _grounded({"evidence": [{"excerpt": x}]}, corpus, grams)[0]

    assert ok('"answer": "Item visible; value fully shown"')  # unescaped JSON, wrapping quotes
    assert ok('Hi, please cancel ... order #W1000')  # elision
    assert not ok("the HAR shows three pending requests")  # invented
    assert not ok("")


def _judge_llm(turns_fn=judged_turns, blind=None):
    """judge: passes every check with a grounded excerpt; turns from turns_fn. blind: the blind answer (dict)."""
    def chat(self, messages, *a, **k):
        text = messages[-1]["content"]
        if self.role == "blind":
            return {"content": json.dumps(blind), "tool_calls": []}
        checks = {c: {"verdict": "pass", "evidence": [{"message": "0", "excerpt": "please cancel order #W1000"}], "reason": "r"}
                  for c in JUDGE_CHECKS}
        return {"content": json.dumps({"checks": checks, "checklist": [], "hallucinations": [], "turns": turns_fn(text)}),
                "tool_calls": []}
    return lambda c, role: type("J", (), {"role": role, "chat": chat})()


def test_per_turn_judging_needs_every_turn_and_no_hindsight(tmp_path):
    from bandits_sft.verify import turn_issues, verify
    t = from_sharegpt(seed(0), 0)
    asst = [i for i, m in enumerate(t["messages"]) if m["role"] == "assistant"]
    ok = [{"message": i, "verdict": "justified", "relies_on": [i - 1]} for i in asst]
    assert turn_issues(t, ok) == []
    assert turn_issues(t, ok[1:]) == [f"turn {asst[0]} not judged"]
    late = [dict(e, relies_on=[e["message"] + 1]) if e["message"] == asst[0] else e for e in ok]
    assert turn_issues(t, late) == [f"turn {asst[0]} cites messages not before it: [{asst[0] + 1}]"]
    assert turn_issues(t, None) == ["turns missing"]
    first = asst[0]
    assert turn_issues(t, [dict(e, relies_on=[]) if e["message"] == first else e for e in ok])[0].startswith(f"turn {first} cites nothing")
    assert turn_issues(t, [dict(e, relies_on=[-1]) if e["message"] == first else e for e in ok]) == [
        f"turn {first} cites messages not before it: [-1]"]
    assert turn_issues(t, [dict(e, relies_on=["system"]) if e["message"] == first else e for e in ok]) == []
    cfg = load_config(write_cfg(tmp_path, {"change_gate": False, "judge": True, "judge_gate": True}, target=1, keep_failures=False))
    ctx = {"seed": t, "variation": {"name": "a"}, "harness": "h"}
    set_llm_factory(_judge_llm())
    assert verify(cfg, from_sharegpt(seed(0), 0), set(), ctx)["meta"]["verify"]["kept"] is True
    bad = lambda text: [dict(e, verdict="unjustified") for e in judged_turns(text)]
    set_llm_factory(_judge_llm(bad))
    v = verify(cfg, from_sharegpt(seed(0), 0), set(), ctx)["meta"]["verify"]
    assert v["kept"] is False and v["judge"]["turn_issues"]


def test_judge_evaluator_errors(tmp_path):
    from bandits_sft.llm import Truncated
    from bandits_sft.verify import system_key, verify
    t = from_sharegpt(seed(0), 0)
    t["messages"][-1]["content"] = '{"status": "cancelled"}'
    schema = {"type": "object", "required": ["status"], "properties": {"status": {"enum": ["pending", "cancelled"]}}}
    (tmp_path / "schemas.json").write_text(json.dumps({system_key(t["system"]): schema}))
    cfg = load_config(write_cfg(tmp_path, {"change_gate": False, "judge": True, "judge_gate": True}, target=1, keep_failures=False,
                                answer_schemas_path=str(tmp_path / "schemas.json")))
    ctx = {"seed": t, "variation": {"name": "a"}, "harness": "h"}
    run = lambda: verify(cfg, json.loads(json.dumps(t)), set(), ctx)["meta"]["verify"]
    calls = []
    good = _judge_llm()
    def flaky(c, role):  # first judge response lacks turns (malformed), the retry is complete
        inner = good(c, role)
        def chat(self, messages, *a, **k):
            out = inner.chat(messages)
            if role == "judge" and not calls:
                calls.append(1)
                body = json.loads(out["content"]); body.pop("turns")
                return {"content": json.dumps(body), "tool_calls": []}
            return out
        return type("J", (), {"role": role, "chat": chat})()
    set_llm_factory(flaky)
    v = run()
    assert v["kept"] is True and v["judge_attempts"] == 2 and not v.get("eval_error")
    always_bad = lambda c, role: type("J", (), {"chat": lambda self, *a, **k: {"content": json.dumps({"checks": {}}),
                                                                                "tool_calls": []}})()
    set_llm_factory(always_bad)
    v = run()
    assert v["kept"] is False and v["judge_attempts"] == 2 and v["eval_error"].startswith("malformed judge output")
    def truncated(c, role):
        def chat(self, *a, **k):
            raise Truncated("response hit the token limit (1)")
        return type("J", (), {"chat": chat})()
    set_llm_factory(truncated)
    v = run()
    assert v["kept"] is False and v["judge_attempts"] == 1 and "Truncated" in v["eval_error"]


def test_case_block_format_and_judge_input(tmp_path):
    from bandits_sft.analyzer import fill, parse_output
    from bandits_sft.prompts import CASE_KEYS
    case = {k: "x" for k in CASE_KEYS}
    msgs = json.loads(ANALYZER_OUT)["messages"]
    c, m, issues = parse_output(json.dumps({"case": case, "messages": msgs}), case=True)
    assert c == case and m and not issues
    c, m, issues = parse_output(json.dumps({"case": {"kept": "x"}, "messages": msgs}), case=True)
    assert c is None and m is None and "case" in issues[0]
    assert '"case"' in fill("{seed_trace}", {}, case=True) and '"case"' not in fill("{seed_trace}", {})


def test_combinations_are_checked_and_planned():
    from bandits_sft.analyzer import check_combinations, plan_jobs
    from bandits_sft.verify import system_key
    s = from_sharegpt(seed(0), 0)
    labels = {"S1": system_key(s["system"])}
    vs = [{"name": n, "applies_to": ["all"], "requires_tools": [], "teaches": n, "change": n, "preserve": "p",
           "dependencies": "d", "correct_when": "w"} for n in ("a", "b", "c")]
    assert check_combinations([["a", "b"]], vs) == [] and check_combinations(None, vs) == []
    assert check_combinations([["a", "a"], ["a", "z"]], vs)
    jobs = plan_jobs("v1", [s], 4, vs, labels, [], [["a", "b"]], 0.5, max_per_seed=10)
    assert len(jobs) == 4  # three singles plus the a+b combination: a tried pair is never scheduled again
    combos = [j["variation"] for j in jobs if j["variation"].get("combines")]
    assert len(combos) == 1 and combos[0]["name"] == "a+b" and "(1) a (2) b" in combos[0]["change"]
    assert all(not j["variation"].get("combines") for j in plan_jobs("v1", [s], 4, vs, labels, [], [["a", "b"]], 0.0))


def test_seed_only_analysis_reads_all_dev_seeds_and_rejects_external_sources():
    from bandits_sft.analyzer import _analyzer_inputs, check_signals, system_labels
    from bandits_sft.banks import ObsBank
    traces = [from_sharegpt(seed(i), i) for i in range(4)]
    from bandits_sft.config import Config
    cfg = Config(seeds_path='unused', analyzer_seeds=1)
    inputs = _analyzer_inputs(cfg, traces, ObsBank.from_traces(traces), system_labels(traces))
    assert inputs['n_seeds'] == 4
    assert 'harness' not in inputs
    good = [{'id': 'lookup', 'sources': [traces[0]['id']], 'teaches': 'lookup',
             'decision': 'call', 'driving_evidence': 'order id'}]
    variations = [{'name': 'hold', 'signals': ['lookup']}]
    assert check_signals(good, variations, inputs['seed_ids']) == []
    good[0]['sources'] = ['raw-unreviewed-trace']
    assert any('supplied dev seeds' in e for e in check_signals(good, variations, inputs['seed_ids']))


def test_seed_only_config_rejects_harness_paths(tmp_path):
    import pytest
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'seeds_path': 'unused', 'harness_paths': ['/external/source']}))
    with pytest.raises(ValueError, match='harness_paths'):
        load_config(path)


def jobs_signal_details(path):
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    return all(j["variation"]["signal_details"][0]["id"] == "lookup" for j in jobs)


# ---- eligibility, refusals, explicit answer contract, structural not_applicable ----

def _vars():
    base = {"teaches": "t", "change": "c", "preserve": "p", "dependencies": "d", "correct_when": "w", "requires_tools": []}
    return [{"name": "a", "applies_to": ["all"], "eligible_seeds": ["s1"], **base},
            {"name": "b", "applies_to": ["all"], "eligible_seeds": ["s2", "s3"], **base},
            {"name": "c", "applies_to": ["all"], **base}]  # no list: label/tool rule only (older outputs)


def test_eligible_seeds_restrict_assignment_and_are_validated():
    from bandits_sft.analyzer import applicable, check_variations
    sd = lambda i: {"id": i, "system": "S", "messages": []}
    assert [v["name"] for v in applicable(_vars(), sd("s1"), {})] == ["a", "c"]
    assert [v["name"] for v in applicable(_vars(), sd("s2"), {})] == ["b", "c"]
    assert check_variations(_vars(), {"S1": "x"}, ["s1", "s2", "s3"])[0].startswith("variation c needs eligible_seeds")
    bad = _vars()[:2] + [{**_vars()[0], "name": "d", "eligible_seeds": ["nope"]}]
    assert any("unknown ids" in e for e in check_variations(bad, {}, ["s1", "s2", "s3"]))
    assert check_variations(_vars(), {})  == []  # seed_ids not given: older callers are unaffected


def test_scheduler_covers_least_used_seeds_first():
    from bandits_sft.analyzer import plan_jobs
    seeds = [{"id": f"s{i}", "system": "S", "messages": []} for i in range(1, 6)]
    vs = [{**_vars()[2], "name": f"v{k}"} for k in range(3)]
    r1 = plan_jobs("v1", seeds, 3, vs, {}, [])
    ledger = [{"seed_id": j["seed_id"], "variation": j["variation"]["name"]} for j in r1]
    r2 = plan_jobs("v2", seeds, 3, vs, {}, ledger)
    assert {j["seed_id"] for j in r1} == {"s1", "s2", "s3"} and {j["seed_id"] for j in r2} == {"s4", "s5", "s1"}


def test_refusal_is_its_own_outcome_not_a_format_failure():
    from bandits_sft.analyzer import parse_refusal, pilot_report
    assert parse_refusal('{"invalid_pairing": true, "reason": "needs a loader seed"}') == "needs a loader seed"
    assert parse_refusal('{"invalid_pairing": true, "messages": [{"role": "user", "content": "x"}]}') is None
    assert parse_refusal("not json") is None
    t = {"id": "j", "system": "", "tools": [], "messages": [], "meta": {"refusal": "does not fit", "format_issues": []}}
    cfg = load_config_for_tests()
    out = verify_fn(cfg, t, set())["meta"]["verify"]
    assert out == {"rule_issues": [], "refused": True, "kept": False}
    rep = pilot_report(cfg, [t], [])
    assert rep["outcome_split"] == {"conversations": 0, "refusals": 1, "execution_errors": 0}
    assert rep["usable_rate"] == 0.0 and rep["usable_of_conversations"] is None and rep["format_ok"] == 0


def load_config_for_tests():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    return load_config(write_cfg(d, {}, target=1))


def test_explicit_json_requirement_is_a_hard_check_only_when_the_prompt_says_so():
    from bandits_sft.verify import answer_soft_flags, explicit_answer_check, requires_json_answer
    assert requires_json_answer("Return a single JSON object with keys a and b.")
    assert requires_json_answer("Respond with JSON only.")
    assert not requires_json_answer("You may quote JSON snippets from the page. Answer politely.")
    mk = lambda sysm, final: {"system": sysm, "messages": [{"role": "assistant", "content": final}]}
    js = mk("", '{"a": 1}')
    assert explicit_answer_check(mk("Return a JSON object.", '{"a": 1}'), js) == []
    assert explicit_answer_check(mk("Return a JSON object.", '{"a": 1'), js) != []
    assert explicit_answer_check(mk("Be brief.", "not json"), js) == []  # no explicit requirement: nothing to enforce
    assert explicit_answer_check(mk("Return a JSON object.", "text"), mk("", "text")) == []  # seed itself is not JSON
    assert explicit_answer_check(mk("Return a JSON object.", "text"), None) == []
    # seed key sets are only a soft signal
    assert answer_soft_flags(mk("", '{"a": 1, "z": 2}'), mk("", '{"a": 1}')) != []
    assert answer_soft_flags(mk("", '{"a": 1}'), mk("", '{"a": 2}')) == []


def test_not_applicable_is_accepted_only_for_structurally_empty_checks(monkeypatch):
    import bandits_sft.verify as vf
    cfg = load_config_for_tests()
    trace = {"system": "s", "tools": [], "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "meta": {}}
    na = {"verdict": "not_applicable", "evidence": [], "reason": "no tools"}
    ok = {"verdict": "pass", "evidence": [], "reason": "fine"}
    checks = {n: ok for n in JUDGE_CHECKS}
    checks.update(tool_contracts_respected=na, observations_consistent=na, task_success=na)
    monkeypatch.setattr(vf, "chat_json", lambda *a, **k: {"checks": checks, "checklist": [], "turns": [], "hallucinations": []})
    monkeypatch.setattr(vf, "_grounded", lambda *a, **k: (True, []))
    j = vf.judge_expansion(cfg, trace, {"seed": trace, "variation": {}, "observed_contracts": "", "case": None})
    assert j["tool_contracts_respected"] is True and j["obs_consistent"] is True  # nothing to check without tool results
    assert j["task_success"] is False  # the judge may not declare correctness not applicable
    assert "check:task_success:not_applicable" in vf.judge_flags(j)


# ---- pilots: reorder / combo / branch builders (no model calls) ----

def _two_call_trace():
    call = lambda i, name, **a: {"id": f"c{i}", "name": name, "arguments": a}
    ms = [{"role": "user", "content": "q"},
          {"role": "assistant", "content": "", "tool_calls": [call(1, "get_order", order_id="#A1"), call(2, "cancel_order", order_id="#B2")]},
          {"role": "tool", "tool_call_id": "c1", "name": "get_order", "content": '[r:1] {"x": 1}'},
          {"role": "tool", "tool_call_id": "c2", "name": "cancel_order", "content": '[r:2] {"y": 2}'},
          {"role": "assistant", "content": "{\"status\": \"ok\"}"}]
    return {"id": "t", "system": "Be brief.", "tools": TOOLS, "messages": ms, "meta": {}}


def test_reorder_keeps_pairs_and_renumbers_results():
    from bandits_sft.experiments import independent_calls, mechanical_ok, reorder_rows, reorder_trace
    t = _two_call_trace()
    assert independent_calls(t)
    r = reorder_trace(t, [1, 0])
    names = [c["name"] for m in r["messages"] if m.get("tool_calls") for c in m["tool_calls"]]
    assert names == ["cancel_order", "get_order"] and mechanical_ok(r) == []
    tools = [m for m in r["messages"] if m["role"] == "tool"]
    assert tools[0]["content"].startswith("[r:1]") and '"y": 2' in tools[0]["content"] and tools[1]["content"].startswith("[r:2]")
    assert r["messages"][-1] == t["messages"][-1]  # the answer is untouched
    rows = reorder_rows([t], per_trace=3)
    assert len(rows) == 1 and rows[0]["meta"]["source_id"] == "t"
    dependent = _two_call_trace()
    dependent["messages"][1]["tool_calls"][1]["arguments"]["order_id"] = "xyz-long-id"
    dependent["messages"][2]["content"] = '[r:1] {"id": "xyz-long-id"}'
    assert not independent_calls(dependent)  # a later call uses an earlier result: not reorderable


def test_combo_and_branch_jobs():
    from bandits_sft.experiments import branch_jobs, combo_jobs, hard_conflict, unused_reads
    t = _two_call_trace()
    seed_ = {**t, "id": "S1"}
    a = {"name": "a", "change": "get_order returns empty", "applies_to": ["all"], "requires_tools": []}
    b = {"name": "b", "change": "cancel_order is refused", "applies_to": ["all"], "requires_tools": []}
    d = {"name": "d", "change": "get_order is unavailable", "applies_to": ["all"], "requires_tools": []}
    n = {"name": "n", "change": "something vague", "applies_to": ["all"], "requires_tools": []}
    assert not hard_conflict(a, b, seed_) and hard_conflict(a, d, seed_) and hard_conflict(a, n, seed_)
    good = [{"seed_id": "S1", "variation": {k: x for k, x in v.items() if k not in ("applies_to", "requires_tools")}} for v in (a, b, d)]
    jobs = combo_jobs([seed_], good, n=5, exclude=set())
    assert [j["variation"]["combines"] for j in jobs] == [["a", "b"], ["b", "d"]]  # a+d: same tool made unavailable
    no = combo_jobs([seed_], good, n=5, exclude=set(), screen=lambda s, x, y: {"compatible": False, "reason": "conflict"})
    assert no == [] and len(combo_jobs.last_log) == 2  # the model screen decides; its verdicts are logged
    assert combo_jobs([seed_], good, n=5, exclude={"S1"}) == []
    assert unused_reads(seed_) == []
    one = {**seed_, "messages": [seed_["messages"][0], {**seed_["messages"][1], "tool_calls": seed_["messages"][1]["tool_calls"][:1]},
                                 seed_["messages"][2], seed_["messages"][4]]}
    assert [x[0] for x in unused_reads(one)] == ["cancel_order"]
    bj = branch_jobs([one], n=3, exclude=set())
    assert len(bj) == 1 and bj[0]["variation"]["name"] == "branch_extra_call_cancel_order"


def test_null_optional_arg_and_seed_types_are_not_schema_errors():
    from bandits_sft.verify import observed_loose_type_args, schema_errors
    spec = {"type": "object", "properties": {"q": {"type": "string"}, "id": {"type": "string"}}, "required": ["id"]}
    assert schema_errors({"q": None, "id": "x"}, spec) == []
    assert schema_errors({"id": None}, spec)  # null for a required argument is still an error
    seeds = [{"tools": [{"name": "t", "parameters": spec}],
              "messages": [{"role": "assistant", "tool_calls": [{"name": "t", "arguments": {"id": 7, "q": None}}]}]}]
    assert observed_loose_type_args(seeds) == {"t": ["id"]}


def test_sharegpt_keeps_text_said_with_tool_calls():
    """Review repro: an assistant turn with text plus a tool call lost its text in the ShareGPT export."""
    t = {"id": "x", "system": "", "tools": [], "messages": [
        {"role": "user", "content": "where is my order?"},
        {"role": "assistant", "content": "Let me check that.", "tool_calls": [{"id": "c1", "name": "get_order", "arguments": {"order_id": "1"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_order", "content": "{}"},
        {"role": "assistant", "content": "It is pending."}]}
    conv = to_sharegpt(t)["conversations"]
    assert [c["from"] for c in conv] == ["human", "gpt", "function_call", "observation", "gpt"]
    assert conv[1]["value"] == "Let me check that." and json.loads(conv[2]["value"])["name"] == "get_order"


def test_state_ops_reject_bad_input_without_crashing():
    """Review repro: a list index equal to its length, a None op and a null path raised instead of reporting."""
    new, errors = apply_ops({"a": [{}]}, [{"op": "set", "path": "a.1.x", "value": 1}, None, {"op": "set", "path": None}])
    assert new == {"a": [{}]} and len(errors) == 3
    new, errors = apply_ops({"a": [{}]}, [{"op": "set", "path": "a.0.x", "value": 1}, {"op": "set", "path": "a.-1.y", "value": 2}])
    assert new == {"a": [{"x": 1, "y": 2}]} and errors == []


def test_selftest_case_skips_a_non_object_answer(tmp_path):
    from bandits_sft.selftest import make_case
    cfg = load_config(write_cfg(tmp_path, {}, target=1))
    set_llm_factory(lambda c, role: type("F", (), {"chat": lambda self, *a, **k: {"content": "[1, 2]", "tool_calls": []}})())
    case = make_case(cfg, from_sharegpt(seed(0), 0), "contradiction")
    assert case["skip"] == "answer is not a JSON object"
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))
