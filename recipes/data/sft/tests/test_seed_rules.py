"""Seed-learned tool-result rules: generic shapes and availability gates."""
import json

from bandits_sft.seed_rules import SeedRules


def trace(evidence, calls, i=0):
    msgs = [{"role": "user", "content": "Context: " + json.dumps({"evidence": evidence})}]
    for n, (name, args, result) in enumerate(calls):
        cid = f"c{i}_{n}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "name": name, "arguments": args}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": f"[tag:{n + 1}] " + json.dumps(result)})
    msgs.append({"role": "assistant", "content": "{}"})
    return {"id": f"t{i}", "messages": msgs}


ON = {"available": True, "count": 1, "nodes": [{"role": "r", "name": "n"}]}
OFF = {"available": False, "count": 0, "nodes": []}


def seeds():
    out = [trace(["tree", "shot"], [("search", {"side": "live"}, ON)], i) for i in range(3)]
    out += [trace(["shot"], [("search", {"side": "live"}, OFF)], 10 + i) for i in range(2)]
    out += [trace(["shot"], [("history", {"source": "steps"}, {"available": True, "items": [{"step": 1}]})], 20)]
    return out


def test_gate_learned_and_enforced():
    rules = SeedRules(seeds())
    assert not any(rules.errors(s) for s in seeds())
    bad = trace(["shot"], [("search", {"side": "live"}, ON)])
    assert "context has none of" in rules.errors(bad)[0]


def test_unseen_result_shape_rejected():
    rules = SeedRules(seeds())
    invented = trace(["shot"], [("history", {"source": "steps"}, {"available": True, "items": [{"step": 1, "note": "x"}]})])
    assert "never returned" in rules.errors(invented)[0]


def test_omitted_optional_key_is_fine():
    rules = SeedRules(seeds())
    fewer = trace(["shot"], [("history", {"source": "steps"}, {"available": True, "items": [{}]})])
    assert not rules.errors(fewer)


def test_too_little_support_learns_no_gate():
    rules = SeedRules(seeds()[:4])  # 3 available, 1 unavailable: below min_unavail
    assert not rules.gates
