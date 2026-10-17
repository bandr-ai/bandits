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
    out += [trace(["shot"], [("history", {"source": "steps"}, {"available": True, "items": [{"step": 1}]})], 20 + i)
            for i in range(3)]
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


def test_branch_key_is_learned_not_named():
    """Any field can be the branch flag; the empty branch is the one whose other fields are empty, even when it
    carries an extra explanation field."""
    full = {"state": "found", "rows": [{"k": "v"}], "total": 2}
    empty = {"state": "missing", "rows": [], "total": 0, "why": "nothing matched"}
    seeds = [trace(["a"], [("lookup", {"q": f"x{i}"}, full)], i) for i in range(3)]
    seeds += [trace(["b"], [("lookup", {"q": f"y{i}"}, empty)], 10 + i) for i in range(2)]
    seeds += [trace(["b"], [("archive", {"q": f"z{i}"}, empty)], 20 + i) for i in range(2)]
    rules = SeedRules(seeds)
    assert rules.branch_keys == {"state": ["missing", "found"]}
    assert not any(rules.errors(s) for s in seeds)
    invented = trace(["b"], [("archive", {"q": "z9"}, full)])
    assert "unknown branch" in rules.errors(invented)[0]  # archive only ever came back empty
    assert "archive" in rules.describe() and "never returned data" in rules.describe()


def test_echoed_argument_is_not_a_branch_key():
    def call(side, n):
        return trace(["a"], [("probe", {"side": side}, {"side": side, "count": n, "items": [{"x": 1}] * n})], n)
    seeds = [call("left", 0), call("left", 0), call("right", 2), call("right", 3)]
    assert "side" not in SeedRules(seeds).branch_keys


def test_rarely_seen_signature_allows_unseen_keys():
    """With fewer calls than min_support an unseen key is as likely an optional field: not enforced."""
    seeds = [trace(["a"], [("mail", {}, {"id": "1", "subject": "s"})], i) for i in range(2)]
    extra = trace(["a"], [("mail", {}, {"id": "2", "subject": "s", "cc": "x"})])
    assert not SeedRules(seeds).errors(extra)
    assert "never returned" in SeedRules(seeds * 2).errors(extra)[0]


def test_no_branch_flag_means_no_branch_rules():
    seeds = [trace(["a"], [("mail", {}, {"id": str(i), "subject": "s"})], i) for i in range(4)]
    rules = SeedRules(seeds)
    assert rules.branch_keys == {} and rules.gates == {} and rules.empty_only() == []
