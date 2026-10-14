"""Self-test pieces that need no model: edits applied exactly, scoring counts errors apart."""
from bandits_sft.selftest import apply_edits, score

SEED = {"id": "s", "messages": [{"role": "user", "content": "doc: the window is 5 days"},
                                {"role": "assistant", "content": "It takes 5 days."}]}


def test_apply_edits_exact_and_unique_only():
    t = apply_edits(SEED, [{"message": 0, "find": "5 days", "replace": "2 days"}])
    assert t["messages"][0]["content"] == "doc: the window is 2 days" and SEED["messages"][0]["content"].endswith("5 days")
    assert apply_edits(SEED, [{"message": 0, "find": "nowhere", "replace": "x"}]) is None
    assert apply_edits(SEED, [{"message": 9, "find": "5", "replace": "x"}]) is None
    assert apply_edits(SEED, []) is None


def test_score_counts_errors_apart():
    v = [{"kind": "contradiction", "expected": "bad", "label": "bad"},
         {"kind": "wrong_answer", "expected": "bad", "label": "good"},
         {"kind": "harmless", "expected": "good", "label": "good"},
         {"kind": "unchanged", "expected": "good", "label": "bad"},
         {"kind": "contradiction", "expected": "bad", "label": "error"}]
    s = score(v)
    assert s["bad_caught"] == "1/2" and s["good_rejected"] == "1/2" and s["errors"] == 1 and s["balanced"] == 0.5
