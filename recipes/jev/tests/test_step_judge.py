"""The judge input must match what the judge was trained and audited on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recipes.jev.scripts.search_reward_data import OPTIONS as TRAINED_OPTIONS

from bandits_jev.scorer import LogitPrediction
from bandits_jev.step_judge import MAX_PROMPT_TOKENS, OPTIONS, QUESTION, judge_step, step_state

WORK = Path(__file__).resolve().parents[3] / "work/step-rl"
CANDIDATES = WORK / "openresearcher-v3-audit-candidates.jsonl"
LABELED = WORK / "openresearcher-v3-audit-labeled.jsonl"


class FakePredictor:
    """Favours the letter named by `favour`; letters are A, B, C for pos/neu/neg."""

    model_id, revision, dtype, device = "fake", "0", "float32", "cpu"
    max_context_tokens = None

    def __init__(self, favour="A", tokens=100):
        self.favour, self.tokens, self.prompts = favour, tokens, []

    def token_count(self, prompt):
        return self.tokens

    def predict(self, prompt, letters):
        self.prompts.append(prompt)
        logits = {letter: (4.0 if letter == self.favour else 0.0) for letter in letters}
        return LogitPrediction(letter_logits=logits, full_vocab_logsumexp=5.0)


def test_options_and_question_match_the_trained_data():
    assert OPTIONS == TRAINED_OPTIONS
    assert list(OPTIONS) == ["positive", "neutral", "negative"]
    assert QUESTION == "What did this observed search action accomplish?"


@pytest.mark.skipif(not (CANDIDATES.exists() and LABELED.exists()), reason="needs ignored work/step-rl audit files")
def test_step_state_reproduces_every_audited_state_exactly():
    raw = {r["id"]: r for r in map(json.loads, CANDIDATES.read_text().splitlines())}
    rows = [json.loads(line) for line in LABELED.read_text().splitlines()]
    assert len(rows) == 100
    for row in rows:
        step = raw[row["id"]]
        previous = [(p["tool"], p["action"], p["observation"]) for p in step["previous"]]
        state = step_state(step["question"], previous, step["tool"], step["action"], step["observation"])
        assert state == row["state"], row["id"]


def test_only_the_last_two_previous_steps_are_shown_and_long_text_is_cut():
    previous = [("browser.search", f"a{i}" + "x" * 900, f"o{i}" + "y" * 2000) for i in range(3)]
    state = step_state("Q?", previous, "browser.open", "z" * 4000, "§" * 9000)
    history = json.loads(state.split("Previous steps: ")[1].split("\nCurrent action")[0])
    assert [h["action"][:2] for h in history] == ["a1", "a2"]
    assert all(len(h["action"]) == 700 and len(h["observation"]) == 900 for h in history)
    assert state.count("z") == 3000 and state.count("§") == 4500


def test_score_is_positive_minus_negative_and_probabilities_sum_to_one():
    result = judge_step(FakePredictor("A"), "Q?", [], "browser.search", "{}", "reply")
    assert result.reason is None
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.score == pytest.approx(result.probabilities["positive"] - result.probabilities["negative"])
    assert result.score > 0.5
    assert judge_step(FakePredictor("C"), "Q?", [], "browser.search", "{}", "reply").score < -0.5
    assert abs(judge_step(FakePredictor("B"), "Q?", [], "browser.search", "{}", "reply").score) < 0.1


def test_an_overlong_prompt_is_rejected_not_truncated():
    predictor = FakePredictor(tokens=MAX_PROMPT_TOKENS + 1)
    result = judge_step(predictor, "Q?", [], "browser.search", "{}", "reply")
    assert result.score is None and "over the" in result.reason
    assert predictor.prompts == []
