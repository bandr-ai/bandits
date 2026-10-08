from __future__ import annotations

import json

from recipes.jev.scripts.outcome_labels import build

from bandits_jev.outcome_judge import OPTIONS, judge_outcome, outcome_state
from bandits_jev.scorer import LogitPrediction

EVENTS = [{"tool": "browser.search", "action": '{"query": "x"}', "observation": "result " * 400}] * 5


def test_state_shows_actions_last_results_and_final_but_no_gold():
    state = outcome_state("Q?", EVENTS, "Answer: Paris")
    assert state.startswith("Research question: Q?") and state.endswith("Final answer: Answer: Paris")
    results = json.loads(state.split("Last tool results: ")[1].split("\nFinal answer")[0])
    assert len(results) == 3 and all(len(r) == 1200 for r in results)
    assert "(no answer)" in outcome_state("Q?", [], None)


class Fake:
    def __init__(self, favour, tokens=10):
        self.favour, self.tokens = favour, tokens

    def token_count(self, prompt):
        return self.tokens

    def predict(self, prompt, letters):
        return LogitPrediction(letter_logits={x: (3.0 if x == self.favour else 0.0) for x in letters}, full_vocab_logsumexp=4.0)


def test_score_is_probability_of_correct_and_overlong_is_unscored():
    assert list(OPTIONS) == ["correct", "incorrect"]
    assert judge_outcome(Fake("A"), "Q?", EVENTS, "Answer: x") > 0.9
    assert judge_outcome(Fake("B"), "Q?", EVENTS, "Answer: x") < 0.1
    assert judge_outcome(Fake("A", tokens=10_000), "Q?", EVENTS, "x") is None


def test_build_labels_from_correctness_and_balances(tmp_path):
    rollouts = [{"query_id": "1", "sample": s, "correct": s == 0, "final": f"Answer: {s}", "events": []} for s in range(4)]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(rollouts))
    rows = build([path], {"1": "Q?"}, per_class=1, splits=True)
    assert sorted(r["target"] for r in rows) == ["correct", "incorrect"]
    assert all(r["split"] == rows[0]["split"] for r in rows)
