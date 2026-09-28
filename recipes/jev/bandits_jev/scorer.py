"""Frozen (untrained) decision scorer: one forward pass, softmax over the
allowed option-letter logits. Never calls ``generate()``.

The heavy model lives behind a small ``LogitPredictor`` protocol so tests run
against a fake tokenizer/model with no download and no GPU; a real HF/torch
predictor is a thin adapter over the same protocol, implemented in
``bandits_jev.hf_predictor`` (the only module in this package allowed to
import torch, and imported lazily from there).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Literal, Protocol

from bandits.store import ArtifactConflict, Contract, DerivedEnvelope, DerivedStore
from bandits_jev.dataset import DecisionExample
from bandits_jev.prompt import (
    PROMPT_VERSION,
    build_prompt,
    prompt_digest,
    template_digest,
)

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
    """For "single_order": this option's softmax over the one pass's allowed
    letters. For "two_order_average": the mean of both passes' softmax --
    the actual number a caller acts on."""
    raw_logit_pass1: float
    """The first (non-reversed) pass's raw logit -- always present."""
    raw_logit_pass2: float | None = None
    """The second (reversed-order) pass's raw logit. Set only in
    "two_order_average" mode. Recording both passes' raw logits (not just
    pass 1) is what lets a later temperature-fit reconstruct the exact
    averaged distribution this result reported, instead of only the
    single-order one."""


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
    training signal. In "two_order_average" mode this is the mean of both
    passes' mass, matching how ``scores[*].probability`` is also averaged."""
    mode: ScoreMode
    prompt_digest: str
    """The first (non-reversed) pass's prompt digest. The reversed pass's
    prompt differs only in option order, under the same template version."""
    latency_seconds: float | None = None
    """Wall-clock latency when reported. Imported predictions may omit it."""


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
    adapter_digest: str | None = None
    """Digest of the LoRA adapter files scored on top of the base model
    (``adapter_digest``); None for the untrained base. Without it a trained
    run and an untrained run of the same base are indistinguishable."""
    trained_on_dataset_id: str | None = None
    """Dataset recorded by the adapter checkpoint's training progress."""
    predictions_source: str | None = None
    """Set only for a run imported from someone else's predictions (e.g.
    "imported:jev"), never for a run this scorer produced. Such a run did
    not use this repo's prompt, so ``prompt_version``/``template_digest`` are
    0/"" and ``raw_logit_pass1`` holds log(probability)."""
    cost_usd: float | None = None
    """The actual bill for producing an imported run's predictions, as
    recorded by whoever ran it. None when unknown or not applicable."""


_OPTIONAL_IDENTITY_FIELDS = (
    "adapter_digest",
    "trained_on_dataset_id",
    "predictions_source",
    "cost_usd",
)
"""Added after scorer runs were already being saved. Left out of a run's
identity while unset, so every run saved before they existed keeps its id."""


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


def token_limit(model, max_prompt_tokens: int) -> tuple[int, str]:
    """The limit a prompt must fit: the configured ``max_prompt_tokens`` or,
    when smaller, the model's own context length (``max_context_tokens``,
    set by the Hugging Face adapters from the model config). A prompt over
    the model's context does not merely score worse -- the forward pass
    fails -- so it must be rejected like any other overlength prompt.
    Returns the limit and how to name it in a rejection reason."""
    context = getattr(model, "max_context_tokens", None)
    if context is not None and context < max_prompt_tokens:
        return context, f"model's {context}-token context"
    return max_prompt_tokens, f"{max_prompt_tokens} limit"


def _check_prompt_length(
    predictor: LogitPredictor, state: str, question: str, options: dict[str, str], max_prompt_tokens: int
) -> int | None:
    """Returns the token count if it is over the limit, else None. Checked
    separately for each option order actually scored -- reversing the
    option list can change the rendered prompt's length (different option
    text lengths land in a different position relative to any per-token
    overhead), so a two-order run must not check only the first order and
    assume the second is also within budget."""
    prompt, _ = build_prompt(state, question, options)
    token_count = predictor.token_count(prompt)
    return token_count if token_count > max_prompt_tokens else None


def score_example(
    predictor: LogitPredictor,
    example: DecisionExample,
    *,
    mode: ScoreMode = "single_order",
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> DecisionScoreResult | RejectedScore:
    """Score one example. Overlength prompts are rejected, never truncated --
    truncation would silently change what the model is being asked."""
    limit, limit_name = token_limit(predictor, max_prompt_tokens)
    over_limit = _check_prompt_length(predictor, example.state, example.question, dict(example.options), limit)
    if over_limit is not None:
        return RejectedScore(
            decision_id=example.decision_id,
            reasons=(f"prompt is {over_limit} tokens, over the {limit_name}",),
        )
    if mode == "two_order_average":
        reversed_options = dict(reversed(example.options.items()))
        reversed_over_limit = _check_prompt_length(
            predictor, example.state, example.question, reversed_options, limit
        )
        if reversed_over_limit is not None:
            return RejectedScore(
                decision_id=example.decision_id,
                reasons=(
                    f"reversed-order prompt is {reversed_over_limit} tokens, over the "
                    f"{limit_name} (first-order prompt was within budget)",
                ),
            )

    start = time.perf_counter()
    try:
        probs, raw_by_option, mass, digest = _score_once(predictor, example)
        second_raw_by_option: dict[str, float] | None = None
        second_mass: float | None = None
        if mode == "two_order_average":
            second_probs, second_raw_by_option, second_mass, _ = _score_once(predictor, example, reverse=True)
            probs = {
                option_id: (probs[option_id] + second_probs.get(option_id, 0.0)) / 2
                for option_id in probs
            }
            mass = (mass + second_mass) / 2
    except TokenizationError as exc:
        return RejectedScore(decision_id=example.decision_id, reasons=(str(exc),))
    latency = time.perf_counter() - start
    # latency covers both passes when mode == "two_order_average".

    scores = tuple(
        OptionScore(
            option_id=option_id,
            probability=p,
            raw_logit_pass1=raw_by_option[option_id],
            raw_logit_pass2=second_raw_by_option[option_id] if second_raw_by_option is not None else None,
        )
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
    adapter_digest: str | None = None,
    trained_on_dataset_id: str | None = None,
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
        adapter_digest=adapter_digest,
        trained_on_dataset_id=trained_on_dataset_id,
    )


def adapter_digest(adapter_path: str | Path) -> str:
    """Digest of a saved LoRA adapter: its config and weight files, by name
    and content. Optimizer state and training progress saved beside them in
    a checkpoint directory are not part of the model and are ignored."""
    root = Path(adapter_path)
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.name.startswith(("adapter_config", "adapter_model"))
    )
    if not any(p.name.startswith("adapter_model") for p in files):
        raise FileNotFoundError(f"no adapter_model file in {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def adapter_training_dataset_id(adapter_path: str | Path) -> str:
    """Read the training dataset provenance saved beside an adapter."""
    progress_path = Path(adapter_path) / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError(f"no progress.json in {Path(adapter_path)}")
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        dataset_id = payload["config"]["dataset_id"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{progress_path} has no valid config.dataset_id") from exc
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"{progress_path} has no valid config.dataset_id")
    return dataset_id


def compute_scorer_run_id(run: ScorerRun) -> str:
    """Content-addressed: two runs over the same model+revision+dataset+
    split+prompt+predictions must get the same id even if their wall-clock
    latency differs, so an exact rerun is recognized as the same logical
    result rather than minted as a new artifact every time. ``latency_seconds``
    is excluded from the hashed payload for exactly that reason -- it stays
    in the stored payload (``save_scorer_run``), just not in identity."""
    digest = hashlib.sha256(_identity_json(run).encode()).hexdigest()
    return f"decision-scorer-run-{digest[:16]}"


def _identity_json(run: ScorerRun) -> str:
    payload = run.model_dump(mode="json")
    for field in _OPTIONAL_IDENTITY_FIELDS:
        if payload.get(field) is None:
            payload.pop(field, None)
    for result in payload["results"]:
        result.pop("latency_seconds", None)
    return json.dumps(payload, sort_keys=True)


def save_scorer_run(run: ScorerRun, store: DerivedStore) -> DerivedEnvelope:
    """An exact rerun has the same id but a different stored latency, so the
    store sees different bytes. That is the same logical result: keep the
    first run's payload and return its envelope. Any other difference under
    the same id is a real conflict and still raises."""
    run_id = compute_scorer_run_id(run)
    try:
        return store.write(
            run_id,
            kind="decision_scorer_run",
            parent_artifact_id=run.dataset_id or run.model_id,
            payload=run.model_dump_json().encode(),
            summary={
                "results": len(run.results),
                "rejections": len(run.rejections),
            },
        )
    except ArtifactConflict:
        if _identity_json(load_scorer_run(run_id, store)) != _identity_json(run):
            raise
        return store.read_envelope(run_id)


def load_scorer_run(scorer_run_id: str, store: DerivedStore) -> ScorerRun:
    return ScorerRun.model_validate_json(store.read_payload(scorer_run_id))


def import_predictions(
    text: str,
    examples: list[DecisionExample],
    *,
    name: str,
    model_id: str,
    revision: str,
    dataset_id: str,
    split: str,
    cost_usd: float | None = None,
) -> ScorerRun:
    """Turn another system's predictions into a ``ScorerRun`` so the report
    scores them with exactly the metrics it applies to local runs.

    One JSON object per line: ``{"decision_id", "probabilities": {option_id:
    p}, "latency_seconds"?}``. Every example in ``examples`` either gets a
    result or a rejection -- a missing, malformed or mismatched line is
    rejected with its reason, never dropped, so a system that skipped hard
    rows can't look better by having answered only easy ones. Probabilities
    must be finite, non-negative and sum to 1 within 1e-3; they are then
    renormalized exactly.
    """
    by_id = {e.decision_id: e for e in examples}
    predicted: dict[str, dict] = {}
    rejections: dict[str, list[str]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: not valid JSON ({exc})") from exc
        decision_id = row.get("decision_id") if isinstance(row, dict) else None
        if decision_id not in by_id:
            raise ValueError(f"line {line_number}: decision_id {decision_id!r} is not in this split")
        if decision_id in predicted or decision_id in rejections:
            raise ValueError(f"line {line_number}: duplicate prediction for {decision_id!r}")
        problem = _prediction_problem(row, by_id[decision_id])
        if problem is not None:
            rejections[decision_id] = [f"line {line_number}: {problem}"]
            continue
        predicted[decision_id] = row

    results: list[DecisionScoreResult] = []
    rejected: list[RejectedScore] = []
    for example in examples:
        row = predicted.get(example.decision_id)
        if row is None:
            reasons = rejections.get(example.decision_id, [f"no prediction from {name}"])
            rejected.append(RejectedScore(decision_id=example.decision_id, reasons=tuple(reasons)))
            continue
        raw = {o: float(row["probabilities"][o]) for o in example.options}
        total = sum(raw.values())
        probs = {o: p / total for o, p in raw.items()}
        scores = tuple(
            OptionScore(
                option_id=o, probability=p, raw_logit_pass1=math.log(max(p, _IMPORTED_LOG_FLOOR))
            )
            for o, p in probs.items()
        )
        results.append(
            DecisionScoreResult(
                decision_id=example.decision_id,
                scores=scores,
                chosen_option_id=max(scores, key=lambda s: s.probability).option_id,
                candidate_token_mass=1.0,
                mode="single_order",
                prompt_digest="",
                latency_seconds=(
                    float(row["latency_seconds"])
                    if row.get("latency_seconds") is not None
                    else None
                ),
            )
        )
    return ScorerRun(
        model_id=model_id,
        revision=revision,
        dtype="n/a",
        device="n/a",
        dataset_id=dataset_id,
        split=split,
        prompt_version=0,
        template_digest="",
        mode="single_order",
        max_prompt_tokens=0,
        results=tuple(results),
        rejections=tuple(rejected),
        predictions_source=f"imported:{name}",
        cost_usd=cost_usd,
    )


_IMPORTED_LOG_FLOOR = 1e-15


def _prediction_problem(row: dict, example: DecisionExample) -> str | None:
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, dict):
        return "'probabilities' must be an object of option id -> probability"
    if set(probabilities) != set(example.options):
        return (
            f"probabilities cover {sorted(probabilities)}, but this example's options are "
            f"{sorted(example.options)}"
        )
    try:
        values = [float(v) for v in probabilities.values()]
    except (TypeError, ValueError, OverflowError):
        return "probabilities must be numbers"
    if any(not math.isfinite(v) or v < 0 for v in values):
        return "probabilities must be finite and non-negative"
    if abs(sum(values) - 1.0) > 1e-3:
        return f"probabilities sum to {sum(values)!r}, not 1"
    latency = row.get("latency_seconds")
    if latency is not None:
        try:
            latency_value = float(latency)
        except (TypeError, ValueError, OverflowError):
            return "latency_seconds must be a number when provided"
        if not math.isfinite(latency_value) or latency_value < 0:
            return "latency_seconds must be finite and non-negative"
    return None
