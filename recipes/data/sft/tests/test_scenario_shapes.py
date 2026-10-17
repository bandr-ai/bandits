"""The scenario recipe end to end on different agent shapes (fake LLMs, made-up seeds): what it learns from the
seeds differs per shape, and every shape runs through the worker loop to a dataset."""
import json

import pytest
from test_scenario_pipeline import ScenarioFake, drive

from bandits_sft.io import read_jsonl


@pytest.fixture
def fake(monkeypatch):
    from bandits_sft.llm import configure_call_log, set_budget, set_llm_factory
    monkeypatch.setenv("FAKE_KEY", "x")
    opts: dict = {}
    set_llm_factory(lambda cfg, role: ScenarioFake(role, opts))
    yield opts
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real API in tests")))
    set_budget(None)
    configure_call_log(None)


def tool(name):
    return {"name": name, "description": f"{name} tool", "parameters": {"type": "object", "properties": {}}}


def convo(system, tools, turns):
    return {"system": system, "tools": json.dumps(tools), "conversations": turns}


def call(name, result):
    return [{"from": "function_call", "value": json.dumps({"name": name, "arguments": {}})},
            {"from": "observation", "value": json.dumps(result)}]


def tools_list_context_json(i):
    """Tools, a JSON first input listing which sources exist, a JSON verdict, and a flag that empties results."""
    has = i % 3 != 0
    ctx = {"sources": ["log", "store"] if has else ["store"]}
    log = {"present": True, "lines": [f"line {i}"], "count": 1} if has else {"present": False, "lines": [], "count": 0}
    turns = [{"from": "human", "value": "Check this case. " + json.dumps(ctx)}, *call("read_log", log),
             *call("read_store", {"present": True, "lines": [f"row {i}"], "count": 1}),
             {"from": "gpt", "value": json.dumps({"verdict": ["ok", "bad"][i % 2], "confidence": 0.8, "detail": f"case {i}"})}]
    return {"id": f"a{i}", **convo("Answer with JSON: verdict, confidence, detail.", [tool("read_log"), tool("read_store")], turns)}


def tools_free_text(i):
    turns = [{"from": "human", "value": f"Book a table for {i + 2}."},
             *call("search", {"results": [{"name": f"place {i}", "seats": i + 2}]}),
             {"from": "gpt", "value": f"Place {i} has room for {i + 2}. Shall I book it?"}]
    return {"id": f"b{i}", **convo("You are a helpful booking assistant.", [tool("search")], turns)}


def no_tools_documents(i):
    turns = [{"from": "human", "value": f"Document {i}: the device restarts after update {i}. Question: why?"},
             {"from": "gpt", "value": f"Per document {i}, update {i} triggers the restart."}]
    return {"id": f"c{i}", **convo("Answer only from the documents in the message.", [], turns)}


def run(tmp_path, seeds, target=4):
    _p, rep, stop, out = drive(tmp_path, target=target, seeds=seeds)
    assert stop is None, stop
    assert all(s["status"] == "done" for s in rep["steps"].values())
    assert rep["final"] >= 1 and (out / "final/synthetic_sharegpt.jsonl").exists()
    return rep, json.loads((out / "setup/domain.json").read_text()), read_jsonl(out / "final/synthetic.jsonl")


def test_tools_with_list_context_and_json_labels(tmp_path, fake):
    _rep, setup, final = run(tmp_path, [tools_list_context_json(i) for i in range(9)])
    assert setup["label_fields"] == ["verdict"] and setup["branch_keys"] == ["present"] and not setup["no_op"]
    assert all(isinstance(json.loads(t["messages"][-1]["content"]), dict) for t in final)


def test_tools_with_free_text_answers(tmp_path, fake):
    _rep, setup, final = run(tmp_path, [tools_free_text(i) for i in range(6)])
    assert setup["label_fields"] == [] and setup["branch_keys"] == [] and setup["no_op"]
    assert all(not t["messages"][-1]["content"].startswith("{") for t in final)


def test_no_tools_documents_in_the_message(tmp_path, fake):
    _rep, setup, final = run(tmp_path, [no_tools_documents(i) for i in range(6)])
    assert setup["label_fields"] == [] and setup["gates"] == 0 and setup["shape_rules"] == 0
    assert all(not t["tools"] for t in final)


def test_several_agents_in_one_seed_set(tmp_path, fake):
    seeds = [tools_list_context_json(i) for i in range(6)] + [no_tools_documents(i) for i in range(6)]
    _rep, setup, final = run(tmp_path, seeds, target=6)
    assert setup["label_fields"] == ["verdict"]  # learned from the JSON agent; the free-text agent is unaffected
    assert {t["system"] for t in final} <= {s["system"] for s in seeds}


def test_label_fields_per_agent_are_not_diluted():
    from bandits_sft.scenarios import label_fields

    def agent(system, key, i):
        return {"id": f"{key}{i}", "system": system, "tools": [], "messages": [
            {"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps({key: ["x", "y"][i % 2]})}]}
    seeds = [agent("A", "status", i) for i in range(4)] + [agent("B", "grade", i) for i in range(6)]
    assert label_fields(seeds) == ["grade", "status"]  # status is in 40% of all answers, 100% of agent A's


def test_broad_outcome_uses_its_own_agents_label_fields():
    from bandits_sft.scenario_pipeline import outcome
    from bandits_sft.scenarios import label_fields_by_agent

    def ans(system, i, **a):
        return {"id": f"{system}{i}", "system": system, "tools": [], "messages": [
            {"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps(a)}]}
    seeds = [ans("A", i, status=["x", "y"][i % 2]) for i in range(4)]
    seeds += [ans("B", i, grade=["p", "q"][i % 2], status=f"free {i}") for i in range(6)]
    by = label_fields_by_agent(seeds)
    assert by["A"] == ["status"] and by["B"] == ["grade"]  # B's status is free text: not a label for B
    assert outcome(seeds[-1], by) == {"grade": "q"}
