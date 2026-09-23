"""Frozen (untrained) decision scorer: one forward pass, softmax over the
allowed option-letter logits. Never calls ``generate()``.

The heavy model lives behind a small ``LogitPredictor`` protocol so tests run
against a fake tokenizer/model with no download and no GPU; a real HF/torch
predictor is a thin adapter over the same protocol, implemented in
``bandits.decide.hf_predictor`` (the only module in this package allowed to
import torch, and imported lazily from there).
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Literal, Protocol

from bandits.decide.dataset import DecisionExample
from bandits.decide.prompt import (
    PROMPT_VERSION,
    build_prompt,
    prompt_digest,
    template_digest,
)
from bandits.store import Contract, DerivedEnvelope, DerivedStore

DEFAULT_MAX_PROMPT_TOKENS = 8_000
ScoreMode = Literal["single_order", "two_order_average"]


class TokenizationError(ValueError):
    """An option letter is not a single, distinct token in the answer
    context, or the model's vocabulary does not distinguish the requested
    letters. Raised at prediction time, never silently worked around."""


class LogitPrediction(Contract):
    letter_logits: dict[str, float]
    """{letter: raw logit} at the last non-padding position, for exactly the
    requested letters."""
    full_vocab_logsumexp: float
    """log(sum(exp(logit))) over the *entire* vocabulary at that same
    position. Together with ``letter_logits`` this gives the true fraction
    of the model's full-vocabulary probability mass that landed on the
    allowed letters (``candidate_token_mass``), not just how that mass is
    split among them."""


class LogitPredictor(Protocol):
    """The one call a scorer makes, so tests need no real model.

    ``letters`` is the list of single-character option codes to read logits
    for, in no particular order. Implementations must raise
    ``TokenizationError`` if any letter is not exactly one distinct token in
    this model's vocabulary -- see the module docstring for why that check
    belongs to the predictor, which is the only thing that knows the
    tokenizer, not the scorer.
    """

    model_id: str
    revision: str
    dtype: str
    device: str

    def token_count(self, prompt: str) -> int: ...

    def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
        """Never calls generate()."""
        ...


class OptionScore(Contract):
    option_id: str
    probability: float
    raw_logit: float
    """From the single, non-reversed pass -- see ``DecisionScoreResult.mode``."""


class DecisionScoreResult(Contract):
    decision_id: str
    scores: tuple[OptionScore, ...]
    chosen_option_id: str
    candidate_token_mass: float
    """The fraction of the model's *full-vocabulary* softmax at this
    position that landed on the allowed letters -- not the allowed-letters
    softmax itself, which always sums to 1 by construction. Low mass means
    the model didn't want to answer with a letter code at all; see the
    cross-repo notes in dataset.py's module docstring. A diagnostic, never a
    training signal. From the single (non-reversed) pass only."""
    mode: ScoreMode
    prompt_digest: str
    latency_seconds: float


class RejectedScore(Contract):
    decision_id: str
    reasons: tuple[str, ...]


class ScorerRun(Contract):
    model_id: str
    revision: str
    dtype: str
    device: str
    dataset_id: str | None = None
    split: str | None = None
    prompt_version: int
    template_digest: str
    mode: ScoreMode
    max_prompt_tokens: int
    results: tuple[DecisionScoreResult, ...]
    rejections: tuple[RejectedScore, ...]


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    """Softmax in ordinary Python floats (64-bit), over exactly the given
    letters -- never the full vocabulary. Numerically stable via max-shift."""
    if not logits:
        raise ValueError("cannot softmax an empty logit set")
    top = max(logits.values())
    exps = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _candidate_token_mass(letter_logits: dict[str, float], full_vocab_logsumexp: float) -> float:
    """sum(exp(letter_logit - full_vocab_logsumexp)) -- the true share of
    the full-vocabulary distribution on these letters, per LSE = logsumexp
    identity: softmax_full(letter) = exp(letter_logit - lse)."""
    return sum(math.exp(v - full_vocab_logsumexp) for v in letter_logits.values())


def _score_once(
    predictor: LogitPredictor, example: DecisionExample, *, reverse: bool = False
) -> tuple[dict[str, float], dict[str, float], float, str]:
    """One forward pass. Returns (option_id -> probability, option_id -> raw
    logit, candidate_token_mass, the rendered prompt's digest)."""
    options = dict(reversed(example.options.items())) if reverse else dict(example.options)
    prompt, letters = build_prompt(example.state, example.question, options)
    option_by_letter = {letter: option_id for option_id, letter in letters.items()}
    prediction = predictor.predict(prompt, list(letters.values()))
    missing = set(letters.values()) - set(prediction.letter_logits)
    if missing:
        raise TokenizationError(
            f"predictor did not return logits for letter(s) {sorted(missing)}"
        )
    probs_by_letter = _softmax(prediction.letter_logits)
    probs_by_option = {option_by_letter[letter]: p for letter, p in probs_by_letter.items()}
    raw_by_option = {option_by_letter[letter]: v for letter, v in prediction.letter_logits.items()}
    mass = _candidate_token_mass(prediction.letter_logits, prediction.full_vocab_logsumexp)
    return probs_by_option, raw_by_option, mass, prompt_digest(prompt)


def score_example(
    predictor: LogitPredictor,
    example: DecisionExample,
    *,
    mode: ScoreMode = "single_order",
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> DecisionScoreResult | RejectedScore:
    """Score one example. Overlength prompts are rejected, never truncated --
    truncation would silently change what the model is being asked."""
    prompt_preview, _ = build_prompt(example.state, example.question, dict(example.options))
    token_count = predictor.token_count(prompt_preview)
    if token_count > max_prompt_tokens:
        return RejectedScore(
            decision_id=example.decision_id,
            reasons=(f"prompt is {token_count} tokens, over the {max_prompt_tokens} limit",),
        )

    start = time.perf_counter()
    try:
        probs, raw_by_option, mass, digest = _score_once(predictor, example)
        if mode == "two_order_average":
            second_probs, _, _, _ = _score_once(predictor, example, reverse=True)
            probs = {
                option_id: (probs[option_id] + second_probs.get(option_id, 0.0)) / 2
                for option_id in probs
            }
    except TokenizationError as exc:
        return RejectedScore(decision_id=example.decision_id, reasons=(str(exc),))
    latency = time.perf_counter() - start
    # latency covers both passes when mode == "two_order_average".

    scores = tuple(
        OptionScore(option_id=option_id, probability=p, raw_logit=raw_by_option[option_id])
        for option_id, p in probs.items()
    )
    chosen = max(scores, key=lambda s: s.probability)
    return DecisionScoreResult(
        decision_id=example.decision_id,
        scores=scores,
        chosen_option_id=chosen.option_id,
        candidate_token_mass=mass,
        mode=mode,
        prompt_digest=digest,
        latency_seconds=latency,
    )


def score_dataset(
    predictor: LogitPredictor,
    examples: list[DecisionExample],
    *,
    mode: ScoreMode = "single_order",
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    dataset_id: str | None = None,
    split: str | None = None,
) -> ScorerRun:
    results: list[DecisionScoreResult] = []
    rejections: list[RejectedScore] = []
    for example in examples:
        outcome = score_example(predictor, example, mode=mode, max_prompt_tokens=max_prompt_tokens)
        if isinstance(outcome, RejectedScore):
            rejections.append(outcome)
        else:
            results.append(outcome)
    return ScorerRun(
        model_id=predictor.model_id,
        revision=predictor.revision,
        dtype=predictor.dtype,
        device=predictor.device,
        dataset_id=dataset_id,
        split=split,
        prompt_version=PROMPT_VERSION,
        template_digest=template_digest(),
        mode=mode,
        max_prompt_tokens=max_prompt_tokens,
        results=tuple(results),
        rejections=tuple(rejections),
    )


def compute_scorer_run_id(run: ScorerRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode()).hexdigest()
    return f"decision-scorer-run-{digest[:16]}"


def save_scorer_run(run: ScorerRun, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_scorer_run_id(run),
        kind="decision_scorer_run",
        parent_artifact_id=run.dataset_id or run.model_id,
        payload=run.model_dump_json().encode(),
        summary={
            "results": len(run.results),
            "rejections": len(run.rejections),
        },
    )


def load_scorer_run(scorer_run_id: str, store: DerivedStore) -> ScorerRun:
    return ScorerRun.model_validate_json(store.read_payload(scorer_run_id))
