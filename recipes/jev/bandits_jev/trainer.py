"""LoRA SFT orchestration: data order, option shuffling, overlength
rejection, dev-interval evaluation, checkpoint-by-dev selection, resume, and
the training-run artifact.

Pure logic, no torch import -- the actual forward/backward pass lives behind
a small ``Trainable`` protocol (mirroring ``scorer.LogitPredictor``) so this
module's data-flow, resume and checkpoint-selection logic stays separate
from the model code. ``bandits_jev.hf_trainer.HFTrainer`` implements the
protocol for real; there is no fake/mock implementation -- tests exercise
this module against ``HFTrainer`` wired to a tiny public checkpoint (see
``recipes/jev/tests/test_trainer.py``), never a hand-rolled stand-in.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Protocol

from pydantic import Field

from bandits.store import Contract, DerivedEnvelope, DerivedStore
from bandits_jev.dataset import DecisionExample
from bandits_jev.metrics import is_correct
from bandits_jev.prompt import PROMPT_VERSION, build_prompt, template_digest
from bandits_jev.scorer import LogitPredictor, ScoreMode, TokenizationError, score_dataset

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_EFFECTIVE_BATCH = 8

_PROGRESS_FILE = "progress.json"


class Trainable(Protocol):
    """What the orchestration needs from a trainer. ``HFTrainer`` is the
    only implementation."""

    model_id: str
    revision: str
    lora_config: dict

    def configure_schedule(self, *, total_steps: int, warmup_steps: int) -> None: ...

    def token_count(self, prompt: str) -> int: ...

    def check_letters(self, prompt: str, letters) -> None: ...

    def train_step(
        self, examples: list[DecisionExample], *, option_orders: list[dict[str, str]] | None = None
    ) -> float: ...

    def predictor(self, adapter_path: str | None = None) -> LogitPredictor: ...

    def save_checkpoint(self, path: str) -> None: ...

    def load_checkpoint(self, path: str) -> None: ...

    def environment(self) -> dict[str, str]: ...


class LoRAConfig(Contract):
    rank: int = Field(DEFAULT_LORA_RANK, ge=1)
    alpha: int = Field(DEFAULT_LORA_ALPHA, ge=1)
    dropout: float = Field(DEFAULT_LORA_DROPOUT, ge=0.0, lt=1.0)
    target_modules: str = "all-linear"


class TrainingRunConfig(Contract):
    base_model_id: str
    base_revision: str
    dataset_id: str
    prompt_version: int
    template_digest: str
    lora: LoRAConfig
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, gt=0.0)
    warmup_ratio: float = Field(DEFAULT_WARMUP_RATIO, ge=0.0, lt=1.0)
    """Fraction of total steps spent warming up linearly to ``learning_rate``;
    linear decay to zero after that."""
    effective_batch: int = Field(DEFAULT_EFFECTIVE_BATCH, ge=1)
    epochs: int = Field(1, ge=1)
    seed: int
    eval_every_steps: int = Field(ge=0)
    """0 = score only the final checkpoint."""
    max_prompt_tokens: int = Field(8_000, ge=1)
    dtype: str = "bfloat16"
    device: str = "cuda"


class DevEvaluation(Contract):
    accuracy: float
    """Over scored rows only; rejected rows are a scoring failure, not a
    wrong answer, and are counted separately."""
    scored: int
    rejected: int


class CheckpointRecord(Contract):
    step: int
    adapter_path: str
    dev: DevEvaluation
    train_loss: float
    """Mean train loss over the steps since the previous checkpoint."""


class RejectedTrainExample(Contract):
    decision_id: str
    reasons: tuple[str, ...]


class TrainingRun(Contract):
    config: TrainingRunConfig
    environment: dict[str, str]
    checkpoints: tuple[CheckpointRecord, ...]
    best_checkpoint_step: int
    """The checkpoint with the highest dev accuracy (earliest on ties), not
    the last one."""
    final_train_loss: float
    steps_completed: int
    rejected_train: tuple[RejectedTrainExample, ...]
    """Train rows never trained on: over ``max_prompt_tokens`` (rejected,
    never truncated) or failing the letter-token contract."""


class _Progress(Contract):
    """Written next to each checkpoint so a resume restores the run's
    history, not only its weights."""

    config: TrainingRunConfig
    step: int
    checkpoints: tuple[CheckpointRecord, ...]


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
    and skips forward to the resume point, rather than reshuffling."""
    order: list[tuple[DecisionExample, dict[str, str]]] = []
    rng = random.Random(seed)
    for _epoch in range(epochs):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)
        for example in epoch_examples:
            order.append((example, shuffled_options(dict(example.options), rng)))
    return order


def reject_unscorable(
    trainable: Trainable,
    order: list[tuple[DecisionExample, dict[str, str]]],
    *,
    max_prompt_tokens: int,
) -> tuple[list[tuple[DecisionExample, dict[str, str]]], list[RejectedTrainExample]]:
    """Drop every data-order entry the scorer would reject: prompt over
    ``max_prompt_tokens`` (same count and limit), or option letters that
    are not distinct single tokens. Each rejected row is recorded once.
    Deterministic, so a resume re-derives the same filtered order."""
    kept: list[tuple[DecisionExample, dict[str, str]]] = []
    rejected: dict[str, RejectedTrainExample] = {}
    for example, options in order:
        prompt, letters = build_prompt(example.state, example.question, options)
        reason = None
        tokens = trainable.token_count(prompt)
        if tokens > max_prompt_tokens:
            reason = f"prompt is {tokens} tokens, over the {max_prompt_tokens}-token limit"
        else:
            try:
                trainable.check_letters(prompt, list(letters.values()))
            except TokenizationError as exc:
                reason = str(exc)
        if reason is not None:
            rejected.setdefault(
                example.decision_id, RejectedTrainExample(decision_id=example.decision_id, reasons=(reason,))
            )
            continue
        kept.append((example, options))
    return kept, list(rejected.values())


def _batches(
    order: list[tuple[DecisionExample, dict[str, str]]], batch_size: int
) -> list[list[tuple[DecisionExample, dict[str, str]]]]:
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def evaluate_dev(
    predictor: LogitPredictor,
    dev_examples: list[DecisionExample],
    *,
    mode: ScoreMode = "single_order",
    max_prompt_tokens: int = 8_000,
) -> DevEvaluation:
    """Accuracy of the argmax option against any target option tied for the
    highest probability, over scored rows only. Raises if no dev row could
    be scored: a checkpoint cannot be judged against nothing."""
    run = score_dataset(predictor, dev_examples, mode=mode, max_prompt_tokens=max_prompt_tokens)
    if not run.results:
        raise ValueError(
            f"no dev example could be scored ({len(run.rejections)} rejected); "
            "cannot select a checkpoint by dev"
        )
    by_id = {e.decision_id: e for e in dev_examples}
    correct = 0
    for result in run.results:
        target = by_id[result.decision_id].target.probabilities
        probabilities = {score.option_id: score.probability for score in result.scores}
        if is_correct(probabilities, target):
            correct += 1
    return DevEvaluation(
        accuracy=correct / len(run.results), scored=len(run.results), rejected=len(run.rejections)
    )


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
            "rejected_train": len(run.rejected_train),
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
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
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
        warmup_ratio=warmup_ratio,
        effective_batch=effective_batch,
        epochs=epochs,
        seed=seed,
        eval_every_steps=eval_every_steps,
        max_prompt_tokens=max_prompt_tokens,
        dtype=dtype,
        device=device,
    )


def _checkpoint_path(checkpoint_dir: str, step: int) -> str:
    return f"{checkpoint_dir}/step-{step}"


def _load_progress(checkpoint_dir: str, step: int, config: TrainingRunConfig) -> _Progress:
    progress_path = Path(_checkpoint_path(checkpoint_dir, step)) / _PROGRESS_FILE
    if not progress_path.exists():
        raise ValueError(
            f"no checkpoint at step {step} in {checkpoint_dir}; a run can only resume "
            "from a step it saved a checkpoint at"
        )
    progress = _Progress.model_validate_json(progress_path.read_bytes())
    if progress.config != config:
        raise ValueError(
            f"checkpoint at step {step} was written by a different training config; "
            "resuming it under this config would not continue the same run"
        )
    return progress


def check_checkpoint_dir(checkpoint_dir: str, resume_from_step: int) -> None:
    """Cheap pre-flight checks, callable before any model is loaded. A fresh
    run must not write into a directory holding another run's checkpoints:
    saving would silently overwrite their adapters and training state."""
    if resume_from_step < 0:
        raise ValueError(f"resume_from_step must be >= 0, got {resume_from_step}")
    if resume_from_step == 0 and any(Path(checkpoint_dir).glob("step-*")):
        raise ValueError(
            f"{checkpoint_dir} already holds checkpoints; a fresh run would overwrite them. "
            "Use a new --checkpoint-dir, or --resume-from-step to continue that run"
        )


def train(
    trainable: Trainable,
    config: TrainingRunConfig,
    *,
    train_examples: list[DecisionExample],
    dev_examples: list[DecisionExample],
    checkpoint_dir: str,
    resume_from_step: int = 0,
) -> TrainingRun:
    """Run SFT to completion. A checkpoint (adapter, optimizer, scheduler,
    RNG state, and run history) is saved and scored on dev every
    ``config.eval_every_steps`` steps and always at the final step.

    ``resume_from_step`` continues from the checkpoint saved at that step:
    the same seed re-derives the same full data order, the run trains on
    exactly the remaining suffix, and weights, optimizer and scheduler pick
    up where they stopped.
    """
    check_checkpoint_dir(checkpoint_dir, resume_from_step)
    if not train_examples:
        raise ValueError("cannot train on an empty train split")
    if not dev_examples:
        raise ValueError("cannot train without a dev split: checkpoints are selected by dev")

    order = build_data_order(train_examples, epochs=config.epochs, seed=config.seed)
    order, rejected_train = reject_unscorable(trainable, order, max_prompt_tokens=config.max_prompt_tokens)
    if not order:
        raise ValueError("every train example was rejected; nothing to train on")
    batches = _batches(order, config.effective_batch)
    if resume_from_step >= len(batches):
        raise ValueError(
            f"resume_from_step {resume_from_step} is at or past this run's {len(batches)} total "
            "steps -- nothing left to train"
        )

    trainable.configure_schedule(
        total_steps=len(batches), warmup_steps=round(config.warmup_ratio * len(batches))
    )
    checkpoints: list[CheckpointRecord] = []
    if resume_from_step > 0:
        progress = _load_progress(checkpoint_dir, resume_from_step, config)
        checkpoints = list(progress.checkpoints)
        trainable.load_checkpoint(_checkpoint_path(checkpoint_dir, resume_from_step))

    window_losses: list[float] = []
    train_loss = 0.0
    for step, batch in enumerate(batches[resume_from_step:], start=resume_from_step + 1):
        train_loss = trainable.train_step([e for e, _ in batch], option_orders=[o for _, o in batch])
        window_losses.append(train_loss)

        at_interval = config.eval_every_steps > 0 and step % config.eval_every_steps == 0
        if not (at_interval or step == len(batches)):
            continue
        path = _checkpoint_path(checkpoint_dir, step)
        trainable.save_checkpoint(path)
        dev = evaluate_dev(
            trainable.predictor(path), dev_examples, max_prompt_tokens=config.max_prompt_tokens
        )
        checkpoints.append(
            CheckpointRecord(
                step=step, adapter_path=path, dev=dev, train_loss=sum(window_losses) / len(window_losses)
            )
        )
        window_losses = []
        progress = _Progress(config=config, step=step, checkpoints=tuple(checkpoints))
        (Path(path) / _PROGRESS_FILE).write_text(progress.model_dump_json())

    best = max(checkpoints, key=lambda c: (c.dev.accuracy, -c.step))
    return TrainingRun(
        config=config,
        environment=trainable.environment(),
        checkpoints=tuple(checkpoints),
        best_checkpoint_step=best.step,
        final_train_loss=train_loss,
        steps_completed=len(batches),
        rejected_train=tuple(rejected_train),
    )
