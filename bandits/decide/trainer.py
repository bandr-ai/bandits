"""LoRA SFT orchestration: data order, option shuffling, dev-interval
evaluation, checkpoint-by-dev selection, and the training-run artifact.

Pure logic, no torch import -- the actual forward/backward pass lives behind
a small ``Trainable`` protocol (mirroring ``scorer.LogitPredictor``) so this
module's data-flow, resume-determinism and checkpoint-selection logic is
testable without a real model. ``bandits.decide.hf_trainer.HFTrainer``
implements the protocol for real; there is no fake/mock implementation --
tests exercise this module against ``HFTrainer`` wired to a tiny public
checkpoint (see ``tests/decide/test_trainer.py``), never a hand-rolled
stand-in, so this module's data-flow claims are always checked against a
real forward/backward pass.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Protocol

from bandits.decide.dataset import DecisionExample
from bandits.decide.prompt import PROMPT_VERSION, template_digest
from bandits.decide.scorer import LogitPredictor, ScoreMode, score_dataset
from bandits.store import Contract, DerivedEnvelope, DerivedStore

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EFFECTIVE_BATCH = 8


class Trainable(Protocol):
    """The one thing a trainer must do, so this module's orchestration logic
    needs no real model to test. ``HFTrainer`` is the only implementation."""

    model_id: str
    revision: str
    lora_config: dict

    def train_step(
        self, examples: list[DecisionExample], *, option_orders: list[dict[str, str]] | None = None
    ) -> float: ...

    def save_adapter(self, path: str) -> None: ...


class LoRAConfig(Contract):
    rank: int = DEFAULT_LORA_RANK
    alpha: int = DEFAULT_LORA_ALPHA
    dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: str = "all-linear"


class TrainingRunConfig(Contract):
    base_model_id: str
    base_revision: str
    dataset_id: str
    prompt_version: int
    template_digest: str
    lora: LoRAConfig
    learning_rate: float = DEFAULT_LEARNING_RATE
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH
    epochs: int = 1
    seed: int
    eval_every_steps: int
    max_prompt_tokens: int = 8_000
    dtype: str = "bfloat16"
    device: str = "cuda"


class CheckpointRecord(Contract):
    step: int
    adapter_path: str
    dev_accuracy: float
    train_loss: float


class TrainingRun(Contract):
    config: TrainingRunConfig
    checkpoints: tuple[CheckpointRecord, ...]
    best_checkpoint_step: int | None
    """None only if no dev evaluation ever ran (e.g. a dev split of zero
    rows) -- distinct from "no checkpoint was ever good enough", which is
    still an int pointing at whichever checkpoint scored highest."""
    final_train_loss: float
    steps_completed: int


def shuffled_options(options: dict[str, str], rng: random.Random) -> dict[str, str]:
    """A fresh random order of ``options`` for one training example. Called
    once per example, per epoch -- the target is never in this dict, so
    shuffling it teaches the model to read the letter from the option text,
    never from position (see ``hf_trainer.HFTrainer.compute_loss``, which
    remaps the target against whatever order this produces)."""
    items = list(options.items())
    rng.shuffle(items)
    return dict(items)


def build_data_order(
    examples: list[DecisionExample], *, epochs: int, seed: int
) -> list[tuple[DecisionExample, dict[str, str]]]:
    """The full, deterministic sequence of (example, option order) pairs
    this run will train on, computed once up front from ``seed`` alone --
    never from wall-clock time, iteration order of a set, or any other
    non-reproducible source. Resuming a run re-derives this exact sequence
    and skips forward to the resume point, rather than reshuffling; the
    remaining, unconsumed suffix of this same list is what a resumed run
    trains on next (see ``TrainingRun.steps_completed``)."""
    order: list[tuple[DecisionExample, dict[str, str]]] = []
    rng = random.Random(seed)
    for _epoch in range(epochs):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)
        for example in epoch_examples:
            order.append((example, shuffled_options(dict(example.options), rng)))
    return order


def _batches(
    order: list[tuple[DecisionExample, dict[str, str]]], batch_size: int
) -> list[list[tuple[DecisionExample, dict[str, str]]]]:
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def evaluate_dev_accuracy(
    predictor: LogitPredictor,
    dev_examples: list[DecisionExample],
    *,
    mode: ScoreMode = "single_order",
    max_prompt_tokens: int = 8_000,
) -> float:
    """Fraction of dev examples where the argmax option matches the target's
    highest-probability option (a hard target's single correct option, or a
    soft target's plurality). Rejected examples (overlength, tokenizer
    contract failure) do not count as correct or incorrect and are excluded
    from the denominator -- they are a scoring failure, not a wrong
    prediction, and folding them into the accuracy would hide how many rows
    were actually judged."""
    if not dev_examples:
        return 0.0
    run = score_dataset(predictor, dev_examples, mode=mode, max_prompt_tokens=max_prompt_tokens)
    if not run.results:
        return 0.0
    correct = 0
    for result in run.results:
        example = next(e for e in dev_examples if e.decision_id == result.decision_id)
        gold = max(example.target.probabilities.items(), key=lambda kv: kv[1])[0]
        if result.chosen_option_id == gold:
            correct += 1
    return correct / len(run.results)


def compute_training_run_id(run: TrainingRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode()).hexdigest()
    return f"decision-training-run-{digest[:16]}"


def save_training_run(run: TrainingRun, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_training_run_id(run),
        kind="decision_training_run",
        parent_artifact_id=run.config.dataset_id,
        payload=run.model_dump_json().encode(),
        summary={
            "checkpoints": len(run.checkpoints),
            "steps_completed": run.steps_completed,
        },
    )


def load_training_run(training_run_id: str, store: DerivedStore) -> TrainingRun:
    return TrainingRun.model_validate_json(store.read_payload(training_run_id))


def build_training_config(
    *,
    base_model_id: str,
    base_revision: str,
    dataset_id: str,
    seed: int,
    eval_every_steps: int,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    effective_batch: int = DEFAULT_EFFECTIVE_BATCH,
    epochs: int = 1,
    max_prompt_tokens: int = 8_000,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> TrainingRunConfig:
    return TrainingRunConfig(
        base_model_id=base_model_id,
        base_revision=base_revision,
        dataset_id=dataset_id,
        prompt_version=PROMPT_VERSION,
        template_digest=template_digest(),
        lora=LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout),
        learning_rate=learning_rate,
        effective_batch=effective_batch,
        epochs=epochs,
        seed=seed,
        eval_every_steps=eval_every_steps,
        max_prompt_tokens=max_prompt_tokens,
        dtype=dtype,
        device=device,
    )


def train(
    trainable: Trainable,
    config: TrainingRunConfig,
    *,
    train_examples: list[DecisionExample],
    dev_examples: list[DecisionExample],
    dev_predictor_factory: Callable[[str], LogitPredictor],
    checkpoint_dir: str,
    resume_from_step: int = 0,
) -> TrainingRun:
    """Run SFT to completion. ``dev_predictor_factory(adapter_path) ->
    LogitPredictor`` builds a scorer for a saved checkpoint -- passed in
    rather than constructed here, so this module still never imports torch
    (the caller supplies an ``HFPredictor`` factory; a test can supply
    whatever it likes as long as it satisfies ``LogitPredictor``).

    ``resume_from_step`` skips that many already-completed steps of the
    deterministic data order built from ``config.seed`` -- the same seed
    reproduces the same full order, so resuming trains on exactly the
    remaining suffix, never a reshuffled one (see ``build_data_order``).
    """
    if not train_examples:
        raise ValueError("cannot train on an empty train split")

    order = build_data_order(train_examples, epochs=config.epochs, seed=config.seed)
    batches = _batches(order, config.effective_batch)
    resume_from_batch = resume_from_step
    if resume_from_batch >= len(batches):
        raise ValueError(
            f"resume_from_step {resume_from_step} is at or past this run's {len(batches)} total "
            "steps -- nothing left to train"
        )

    checkpoints: list[CheckpointRecord] = []
    best_step: int | None = None
    best_accuracy = -1.0
    train_loss = 0.0

    for step, batch in enumerate(batches[resume_from_batch:], start=resume_from_batch):
        examples = [e for e, _ in batch]
        option_orders = [o for _, o in batch]
        train_loss = trainable.train_step(examples, option_orders=option_orders)

        if config.eval_every_steps > 0 and (step + 1) % config.eval_every_steps == 0:
            adapter_path = f"{checkpoint_dir}/step-{step + 1}"
            trainable.save_adapter(adapter_path)
            predictor = dev_predictor_factory(adapter_path)
            dev_accuracy = evaluate_dev_accuracy(
                predictor, dev_examples, max_prompt_tokens=config.max_prompt_tokens
            )
            checkpoints.append(
                CheckpointRecord(
                    step=step + 1, adapter_path=adapter_path, dev_accuracy=dev_accuracy, train_loss=train_loss
                )
            )
            if dev_accuracy > best_accuracy:
                best_accuracy = dev_accuracy
                best_step = step + 1

    if not checkpoints:
        # No interval fell on a step boundary (eval_every_steps <= 0, or a
        # very short run) -- still save and evaluate a final checkpoint, so
        # a run never completes with nothing to score against.
        final_path = f"{checkpoint_dir}/step-{len(batches)}"
        trainable.save_adapter(final_path)
        predictor = dev_predictor_factory(final_path)
        dev_accuracy = evaluate_dev_accuracy(predictor, dev_examples, max_prompt_tokens=config.max_prompt_tokens)
        checkpoints.append(
            CheckpointRecord(step=len(batches), adapter_path=final_path, dev_accuracy=dev_accuracy, train_loss=train_loss)
        )
        best_step = len(batches)

    return TrainingRun(
        config=config,
        checkpoints=tuple(checkpoints),
        best_checkpoint_step=best_step,
        final_train_loss=train_loss,
        steps_completed=len(batches),
    )
