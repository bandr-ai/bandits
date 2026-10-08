"""Jev as a final-answer verifier for domains without a ground-truth checker.

The judge reads the question, the agent's actions, its last tool results and its
final answer, and says whether the answer is correct. Its probability of
"correct" is the outcome reward, so training needs no gold answers; gold is
used only to make the judge's training labels and to evaluate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from bandits_jev.prompt import build_prompt
from bandits_jev.scorer import LogitPredictor, TokenizationError, _softmax

QUESTION = "Is the agent's final answer to the research question correct?"
OPTIONS = {
    "correct": "The final answer is the correct answer to the question, supported by what the agent found.",
    "incorrect": "The final answer is wrong or unsupported, or the agent gave no answer.",
}
ACTION_CHARS = 200
RESULT_CHARS = 1200
RESULTS_SHOWN = 3
FINAL_CHARS = 1000
MAX_PROMPT_TOKENS = 6000


def outcome_state(question: str, events: Sequence[dict], final: str | None) -> str:
    actions = [f"{e['tool']} {e['action'][:ACTION_CHARS]}" for e in events]
    results = [e["observation"][:RESULT_CHARS] for e in events[-RESULTS_SHOWN:]]
    return (
        f"Research question: {question}\n"
        f"Agent actions: {json.dumps(actions, ensure_ascii=False)}\n"
        f"Last tool results: {json.dumps(results, ensure_ascii=False)}\n"
        f"Final answer: {(final or '(no answer)')[:FINAL_CHARS]}"
    )


def judge_outcome(predictor: LogitPredictor, question: str, events: Sequence[dict], final: str | None) -> float | None:
    """P(correct), or None when the prompt cannot be scored."""
    prompt, letters = build_prompt(outcome_state(question, events, final), QUESTION, dict(OPTIONS))
    if predictor.token_count(prompt) > MAX_PROMPT_TOKENS:
        return None
    try:
        prediction = predictor.predict(prompt, list(letters.values()))
    except TokenizationError:
        return None
    return _softmax(prediction.letter_logits)[letters["correct"]]
