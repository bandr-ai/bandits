"""Trainer tests against a real tiny public model (hf-internal-testing/tiny-random-gpt2)
on CPU -- no mock/fake trainable. These download a small checkpoint on first
run (cached by huggingface_hub afterward) and require the `decide` extra
(torch/transformers/peft); skipped automatically if it is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from bandits.decide.dataset import DecisionExample, DecisionLineage, DecisionTarget
from bandits.decide.hf_predictor import HFPredictor
from bandits.decide.hf_trainer import HFTrainer
from bandits.decide.scorer import score_example
from bandits.decide.trainer import (
    build_data_order,
    build_training_config,
    evaluate_dev,
    load_training_run,
    save_training_run,
    shuffled_options,
    train,
)
from bandits.store import DerivedStore

_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
_REVISION = "main"


def _example(decision_id: str, state: str, target: str, *, split: str = "train") -> DecisionExample:
    options = {"a": "apple", "b": "banana"}
    return DecisionExample(
        decision_id=decision_id,
        family_id=decision_id,
        state=state,
        question="which fruit is mentioned",
        primitive="choice",
        options=options,
        target=DecisionTarget(
            kind="hard", probabilities={"a": 1.0 if target == "a" else 0.0, "b": 1.0 if target == "b" else 0.0}
        ),
        label_source="test",
        split=split,
        lineage=DecisionLineage(source_kind="test", record_id=decision_id),
    )


def _train_examples(n: int = 6) -> list[DecisionExample]:
    return [_example(f"train-{i}", f"state {i}", "a" if i % 2 == 0 else "b") for i in range(n)]


def _dev_examples(n: int = 4) -> list[DecisionExample]:
    return [_example(f"dev-{i}", f"dev state {i}", "a" if i % 2 == 0 else "b", split="dev") for i in range(n)]


def _config(**overrides):
    kwargs = {
        "base_model_id": _MODEL_ID,
        "base_revision": _REVISION,
        "dataset_id": "dataset-test",
        "seed": 7,
        "eval_every_steps": 0,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "effective_batch": 2,
        "dtype": "float32",
        "device": "cpu",
    }
    kwargs.update(overrides)
    return build_training_config(**kwargs)


def _trainer(config=None) -> HFTrainer:
    return HFTrainer.from_config(config or _config())


def _run(config, checkpoint_dir, *, trainer=None, train_examples=None, dev_examples=None, resume_from_step=0):
    return train(
        trainer or _trainer(config),
        config,
        train_examples=_train_examples() if train_examples is None else train_examples,
        dev_examples=_dev_examples() if dev_examples is None else dev_examples,
        checkpoint_dir=str(checkpoint_dir),
        resume_from_step=resume_from_step,
    )


def _adapter_weights(path) -> dict:
    from safetensors.torch import load_file

    return load_file(f"{path}/adapter_model.safetensors")


def _same_weights(a: dict, b: dict) -> bool:
    import torch

    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _disk_predictor(adapter_path: str) -> HFPredictor:
    return HFPredictor(_MODEL_ID, revision=_REVISION, device="cpu", dtype="float32", adapter_path=adapter_path)


def test_option_order_is_shuffled_and_target_stays_attached_to_its_option() -> None:
    import random

    options = {"a": "apple", "b": "banana", "c": "cherry"}
    rng = random.Random(1)
    shuffled = shuffled_options(options, rng)
    assert set(shuffled) == set(options)
    assert shuffled["a"] == "apple"  # description stays attached to its own id


def test_data_order_is_deterministic_for_a_fixed_seed() -> None:
    examples = _train_examples()
    order_a = build_data_order(examples, epochs=1, seed=42)
    order_b = build_data_order(examples, epochs=1, seed=42)

    assert [e.decision_id for e, _ in order_a] == [e.decision_id for e, _ in order_b]
    assert [tuple(o.items()) for _, o in order_a] == [tuple(o.items()) for _, o in order_b]


def test_different_seeds_produce_different_orders() -> None:
    examples = _train_examples()
    order_a = build_data_order(examples, epochs=1, seed=1)
    order_b = build_data_order(examples, epochs=1, seed=2)
    assert [e.decision_id for e, _ in order_a] != [e.decision_id for e, _ in order_b]


def test_tiny_model_training_integration_end_to_end(tmp_path) -> None:
    run = _run(_config(eval_every_steps=2), tmp_path / "checkpoints")

    assert run.steps_completed == 3  # 6 train examples / effective_batch 2
    assert all(0.0 <= c.dev.accuracy <= 1.0 and c.dev.scored == 4 for c in run.checkpoints)
    assert run.environment["torch"] and run.environment["peft"]


def test_same_seed_reproduces_weights_and_losses_not_only_data_order(tmp_path) -> None:
    """Reproducible with a fixed seed means the LoRA init and dropout too,
    not only the data order. Dropout is on so its RNG is exercised."""
    config = _config(lora_dropout=0.1)
    run_a = _run(config, tmp_path / "a")
    run_b = _run(config, tmp_path / "b")

    assert [c.train_loss for c in run_a.checkpoints] == [c.train_loss for c in run_b.checkpoints]
    assert _same_weights(_adapter_weights(tmp_path / "a/step-3"), _adapter_weights(tmp_path / "b/step-3"))


def test_different_seed_gives_different_lora_init() -> None:
    a = _trainer(_config(seed=1))._model.state_dict()
    b = _trainer(_config(seed=2))._model.state_dict()
    lora_a = {k: v for k, v in a.items() if "lora_A" in k}
    lora_b = {k: v for k, v in b.items() if "lora_A" in k}
    assert lora_a and not _same_weights(lora_a, lora_b)


def test_best_checkpoint_is_selected_by_dev_not_the_last_step(tmp_path) -> None:
    run = _run(_config(seed=3, eval_every_steps=1), tmp_path / "checkpoints")

    best = next(c for c in run.checkpoints if c.step == run.best_checkpoint_step)
    assert all(best.dev.accuracy >= c.dev.accuracy for c in run.checkpoints)


def test_final_step_is_always_checkpointed(tmp_path) -> None:
    """3 steps with eval every 2: step 3 would otherwise be trained but
    never saved or scored."""
    run = _run(_config(eval_every_steps=2), tmp_path / "checkpoints")
    assert [c.step for c in run.checkpoints] == [2, 3]


def test_resume_continues_the_same_run(tmp_path) -> None:
    """Resuming from a checkpoint trains on exactly the remaining suffix of
    the data order (recorded from actual train_step calls) and ends with
    the same weights and history as the uninterrupted run -- so weights,
    optimizer, LR schedule and dropout RNG were all restored."""
    import shutil

    class RecordingTrainer(HFTrainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.batches_seen: list[list[str]] = []

        def train_step(self, batch_examples, *, option_orders=None) -> float:
            self.batches_seen.append([e.decision_id for e in batch_examples])
            return super().train_step(batch_examples, option_orders=option_orders)

    config = _config(seed=11, eval_every_steps=1, lora_dropout=0.1, warmup_ratio=0.5)
    full = RecordingTrainer.from_config(config)
    full_run = _run(config, tmp_path / "full", trainer=full)

    shutil.copytree(tmp_path / "full/step-1", tmp_path / "resumed/step-1")
    resumed = RecordingTrainer.from_config(config)
    resumed_run = _run(config, tmp_path / "resumed", trainer=resumed, resume_from_step=1)

    assert resumed.batches_seen == full.batches_seen[1:]
    assert _same_weights(_adapter_weights(tmp_path / "full/step-3"), _adapter_weights(tmp_path / "resumed/step-3"))
    assert [c.step for c in resumed_run.checkpoints] == [1, 2, 3]
    assert [c.train_loss for c in resumed_run.checkpoints] == [c.train_loss for c in full_run.checkpoints]
    assert resumed_run.best_checkpoint_step == full_run.best_checkpoint_step


def test_resume_without_a_checkpoint_at_that_step_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="no checkpoint at step 1"):
        _run(_config(), tmp_path / "empty", resume_from_step=1)


def test_resume_under_a_different_config_raises(tmp_path) -> None:
    _run(_config(eval_every_steps=1), tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="different training config"):
        _run(_config(eval_every_steps=1, learning_rate=1e-3), tmp_path / "checkpoints", resume_from_step=1)


def test_resuming_past_the_end_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="nothing left to train"):
        _run(_config(), tmp_path / "unused", resume_from_step=999)


def test_learning_rate_warms_up_then_decays_linearly_to_zero() -> None:
    trainer = _trainer(_config(learning_rate=1e-3))
    trainer.configure_schedule(total_steps=4, warmup_steps=2)
    example = _train_examples(1)[0]
    lrs = [trainer._scheduler.get_last_lr()[0]]
    for _ in range(4):
        trainer.train_step([example])
        lrs.append(trainer._scheduler.get_last_lr()[0])
    assert lrs == pytest.approx([0.0, 5e-4, 1e-3, 5e-4, 0.0])


def test_overlength_train_examples_are_rejected_not_truncated(tmp_path) -> None:
    config = _config(max_prompt_tokens=200)
    long_example = _example("train-long", "word " * 500, "a")
    run = _run(config, tmp_path / "checkpoints", train_examples=[*_train_examples(), long_example])

    assert [r.decision_id for r in run.rejected_train] == ["train-long"]
    assert run.steps_completed == 3  # only the 6 short rows were batched


def test_training_without_a_dev_split_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="without a dev split"):
        _run(_config(), tmp_path / "checkpoints", dev_examples=[])


def test_evaluate_dev_raises_when_no_row_can_be_scored() -> None:
    predictor = HFPredictor(_MODEL_ID, revision=_REVISION, device="cpu", dtype="float32")
    with pytest.raises(ValueError, match="no dev example could be scored"):
        evaluate_dev(predictor, _dev_examples(), max_prompt_tokens=1)


def test_evaluate_dev_is_a_valid_fraction() -> None:
    predictor = HFPredictor(_MODEL_ID, revision=_REVISION, device="cpu", dtype="float32")
    dev = evaluate_dev(predictor, _dev_examples())
    assert 0.0 <= dev.accuracy <= 1.0
    assert dev.scored == 4 and dev.rejected == 0


def test_live_dev_scoring_matches_scoring_the_saved_checkpoint(tmp_path) -> None:
    """Dev is scored on the trainer's live model (no second model copy);
    it must equal scoring the saved adapter loaded from disk."""
    trainer = _trainer(_config(lora_dropout=0.1))
    example = _train_examples(1)[0]
    trainer.train_step([example])
    adapter_path = str(tmp_path / "adapter")
    trainer.save_adapter(adapter_path)

    live = score_example(trainer.predictor(adapter_path), example)
    disk = score_example(_disk_predictor(adapter_path), example)
    assert {s.option_id: s.probability for s in live.scores} == pytest.approx(
        {s.option_id: s.probability for s in disk.scores}, abs=1e-6
    )


def test_training_and_scoring_produce_identical_probabilities_for_the_same_checkpoint(tmp_path) -> None:
    trainer = _trainer()
    example = _train_examples(1)[0]
    trainer.train_step([example])
    adapter_path = str(tmp_path / "adapter")
    trainer.save_adapter(adapter_path)

    result_1 = score_example(_disk_predictor(adapter_path), example)
    result_2 = score_example(_disk_predictor(adapter_path), example)
    assert {s.option_id: s.probability for s in result_1.scores} == {
        s.option_id: s.probability for s in result_2.scores
    }


def test_only_lora_weights_update_base_model_stays_frozen() -> None:
    trainer = _trainer()
    base_params = [p for n, p in trainer._model.named_parameters() if "lora_" not in n]
    lora_params = [p for n, p in trainer._model.named_parameters() if "lora_" in n]
    assert base_params and lora_params
    assert all(not p.requires_grad for p in base_params)
    assert all(p.requires_grad for p in lora_params)


def test_dataset_id_and_prompt_digest_recorded_in_config() -> None:
    config = _config(dataset_id="my-dataset-id")
    assert config.dataset_id == "my-dataset-id"
    assert config.template_digest
    assert config.base_revision == _REVISION


def test_training_run_round_trips_through_store(tmp_path) -> None:
    run = _run(_config(seed=5), tmp_path / "checkpoints")

    store = DerivedStore(tmp_path / "store")
    envelope = save_training_run(run, store)
    assert load_training_run(envelope.artifact_id, store) == run
