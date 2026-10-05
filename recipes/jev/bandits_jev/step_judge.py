"""Score one observed search step with the frozen search judge.

The judge was trained on the string ``step_state`` builds, with the same
truncation, options and question, so serving and the audit must build it the
same way. ``judge_step`` runs one forward pass through any ``LogitPredictor``
and returns ``P(positive) - P(negative)`` for ``step_shaping``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from bandits_jev.prompt import build_prompt
from bandits_jev.scorer import LogitPredictor, TokenizationError, _softmax

QUESTION = "What did this observed search action accomplish?"
OPTIONS = {
    "positive": "The action found answer evidence or a specific credible source to inspect next that advances the question.",
    "neutral": "The action was plausible, but the result gave only broad topical material or no useful next source.",
    "negative": "The action was clearly off task, repeated the same evidence, failed through misuse, or made a claim contradicted by the result.",
}
ACTION_CHARS = 3000
OBSERVATION_CHARS = 4500
HISTORY_ACTION_CHARS = 700
HISTORY_OBSERVATION_CHARS = 900
HISTORY_STEPS = 2
MAX_PROMPT_TOKENS = 8000

Step = tuple[str, str, str]
"""(tool, action, observation) exactly as the agent made and received it."""


def step_state(question: str, previous: Sequence[Step], tool: str, action: str, observation: str) -> str:
    history = [
        {
            "tool": t,
            "action": a[:HISTORY_ACTION_CHARS],
            "observation": o[:HISTORY_OBSERVATION_CHARS],
        }
        for t, a, o in previous[-HISTORY_STEPS:]
    ]
    return (
        f"Research question: {question}\n"
        f"Previous steps: {json.dumps(history, ensure_ascii=False)}\n"
        f"Current action: {tool} {action[:ACTION_CHARS]}\n"
        f"Observed tool result: {observation[:OBSERVATION_CHARS]}"
    )


@dataclass(frozen=True)
class JudgeResult:
    probabilities: dict[str, float] | None
    score: float | None
    """P(positive) - P(negative), or None when the step was rejected."""
    reason: str | None
    prompt_tokens: int


def judge_state(predictor: LogitPredictor, state: str) -> JudgeResult:
    prompt, letters = build_prompt(state, QUESTION, dict(OPTIONS))
    tokens = predictor.token_count(prompt)
    context = getattr(predictor, "max_context_tokens", None)
    limit = min(MAX_PROMPT_TOKENS, context) if context else MAX_PROMPT_TOKENS
    if tokens > limit:
        return JudgeResult(None, None, f"prompt is {tokens} tokens, over the {limit} limit", tokens)
    try:
        prediction = predictor.predict(prompt, list(letters.values()))
    except TokenizationError as exc:
        return JudgeResult(None, None, str(exc), tokens)
    by_letter = _softmax(prediction.letter_logits)
    probabilities = {option: by_letter[letter] for option, letter in letters.items()}
    return JudgeResult(probabilities, probabilities["positive"] - probabilities["negative"], None, tokens)


def judge_step(
    predictor: LogitPredictor,
    question: str,
    previous: Sequence[Step],
    tool: str,
    action: str,
    observation: str,
) -> JudgeResult:
    return judge_state(predictor, step_state(question, previous, tool, action, observation))
