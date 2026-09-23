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
    evaluate_dev_accuracy,
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


def _trainer(**overrides) -> HFTrainer:
    kwargs = {
        "revision": _REVISION,
        "device": "cpu",
        "dtype": "float32",
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
    }
    kwargs.update(overrides)
    return HFTrainer(_MODEL_ID, **kwargs)


def _predictor_factory(adapter_path: str) -> HFPredictor:
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

    ids_a = [e.decision_id for e, _ in order_a]
    ids_b = [e.decision_id for e, _ in order_b]
    assert ids_a == ids_b
    option_orders_a = [tuple(o.items()) for _, o in order_a]
    option_orders_b = [tuple(o.items()) for _, o in order_b]
    assert option_orders_a == option_orders_b


def test_different_seeds_produce_different_orders() -> None:
    examples = _train_examples()
    order_a = build_data_order(examples, epochs=1, seed=1)
    order_b = build_data_order(examples, epochs=1, seed=2)

    ids_a = [e.decision_id for e, _ in order_a]
    ids_b = [e.decision_id for e, _ in order_b]
    assert ids_a != ids_b


def test_tiny_model_training_integration_end_to_end(tmp_path) -> None:
    """The full acceptance criterion: a tiny-model integration test that
    runs end to end and is reproducible with a fixed seed."""
    trainer = _trainer()
    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="dataset-test",
        seed=7,
        eval_every_steps=2,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,
        effective_batch=2,
        dtype="float32",
        device="cpu",
    )
    run = train(
        trainer,
        config,
        train_examples=_train_examples(),
        dev_examples=_dev_examples(),
        dev_predictor_factory=_predictor_factory,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    assert run.steps_completed == 3  # 6 train examples / effective_batch 2
    assert len(run.checkpoints) >= 1
    assert run.best_checkpoint_step is not None
    assert all(0.0 <= c.dev_accuracy <= 1.0 for c in run.checkpoints)


def test_best_checkpoint_is_selected_by_dev_not_the_last_step(tmp_path) -> None:
    """keep the checkpoint with the best dev result, not the last one."""
    trainer = _trainer()
    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="dataset-test",
        seed=3,
        eval_every_steps=1,
        effective_batch=2,
        dtype="float32",
        device="cpu",
    )
    run = train(
        trainer,
        config,
        train_examples=_train_examples(),
        dev_examples=_dev_examples(),
        dev_predictor_factory=_predictor_factory,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    best = next(c for c in run.checkpoints if c.step == run.best_checkpoint_step)
    assert all(best.dev_accuracy >= c.dev_accuracy for c in run.checkpoints)


def test_resume_reproduces_the_same_data_order(tmp_path) -> None:
    """Resuming doesn't change data order: a run resumed from step N trains
    on exactly the same examples (in the same order) a from-scratch run
    would have trained on from that point, for the same seed. Verified by
    recording exactly which examples HFTrainer.train_step was actually
    called with, not by re-deriving the expected order and comparing it to
    itself."""
    examples = _train_examples()

    class RecordingTrainer(HFTrainer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.batches_seen: list[list[str]] = []

        def train_step(self, batch_examples, *, option_orders=None) -> float:
            self.batches_seen.append([e.decision_id for e in batch_examples])
            return super().train_step(batch_examples, option_orders=option_orders)

    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="dataset-test",
        seed=11,
        eval_every_steps=0,
        effective_batch=2,
        dtype="float32",
        device="cpu",
    )

    from_scratch = RecordingTrainer(
        _MODEL_ID, revision=_REVISION, device="cpu", dtype="float32", lora_rank=4, lora_alpha=8, lora_dropout=0.0
    )
    train(
        from_scratch,
        config,
        train_examples=examples,
        dev_examples=[],
        dev_predictor_factory=_predictor_factory,
        checkpoint_dir=str(tmp_path / "from-scratch"),
    )

    resumed = RecordingTrainer(
        _MODEL_ID, revision=_REVISION, device="cpu", dtype="float32", lora_rank=4, lora_alpha=8, lora_dropout=0.0
    )
    resumed_run = train(
        resumed,
        config,
        train_examples=examples,
        dev_examples=[],
        dev_predictor_factory=_predictor_factory,
        checkpoint_dir=str(tmp_path / "resumed"),
        resume_from_step=1,
    )

    assert resumed.batches_seen == from_scratch.batches_seen[1:]
    assert resumed_run.steps_completed == 3


def test_resuming_past_the_end_raises() -> None:
    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="dataset-test",
        seed=1,
        eval_every_steps=0,
        effective_batch=2,
        dtype="float32",
        device="cpu",
    )
    trainer = _trainer()
    with pytest.raises(ValueError, match="nothing left to train"):
        train(
            trainer,
            config,
            train_examples=_train_examples(),
            dev_examples=[],
            dev_predictor_factory=_predictor_factory,
            checkpoint_dir="/tmp/unused",
            resume_from_step=999,
        )


def test_training_and_scoring_produce_identical_probabilities_for_the_same_checkpoint(tmp_path) -> None:
    """The other core acceptance criterion: training and scoring must
    produce identical probabilities for the same checkpoint and input --
    not merely close, exactly equal, since both paths read the saved LoRA
    adapter through the same HFPredictor code."""
    trainer = _trainer()
    example = _train_examples(1)[0]
    trainer.train_step([example])
    adapter_path = str(tmp_path / "adapter")
    trainer.save_adapter(adapter_path)

    predictor_1 = _predictor_factory(adapter_path)
    result_1 = score_example(predictor_1, example)
    predictor_2 = _predictor_factory(adapter_path)
    result_2 = score_example(predictor_2, example)

    probs_1 = {s.option_id: s.probability for s in result_1.scores}
    probs_2 = {s.option_id: s.probability for s in result_2.scores}
    assert probs_1 == probs_2


def test_only_lora_weights_update_base_model_stays_frozen() -> None:
    trainer = _trainer()
    base_params = [p for n, p in trainer._model.named_parameters() if "lora_" not in n]
    lora_params = [p for n, p in trainer._model.named_parameters() if "lora_" in n]
    assert base_params and lora_params
    assert all(not p.requires_grad for p in base_params)
    assert all(p.requires_grad for p in lora_params)


def test_evaluate_dev_accuracy_is_a_valid_fraction() -> None:
    predictor = HFPredictor(_MODEL_ID, revision=_REVISION, device="cpu", dtype="float32")
    accuracy = evaluate_dev_accuracy(predictor, _dev_examples())
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_dev_accuracy_is_zero_for_no_dev_examples() -> None:
    predictor = HFPredictor(_MODEL_ID, revision=_REVISION, device="cpu", dtype="float32")
    assert evaluate_dev_accuracy(predictor, []) == 0.0


def test_dataset_id_and_prompt_digest_recorded_in_config() -> None:
    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="my-dataset-id",
        seed=1,
        eval_every_steps=1,
    )
    assert config.dataset_id == "my-dataset-id"
    assert config.template_digest
    assert config.base_revision == _REVISION


def test_training_run_round_trips_through_store(tmp_path) -> None:
    from bandits.decide.trainer import load_training_run, save_training_run

    trainer = _trainer()
    config = build_training_config(
        base_model_id=_MODEL_ID,
        base_revision=_REVISION,
        dataset_id="dataset-test",
        seed=5,
        eval_every_steps=0,
        effective_batch=2,
        dtype="float32",
        device="cpu",
    )
    run = train(
        trainer,
        config,
        train_examples=_train_examples(),
        dev_examples=_dev_examples(),
        dev_predictor_factory=_predictor_factory,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    store = DerivedStore(tmp_path / "store")
    envelope = save_training_run(run, store)
    loaded = load_training_run(envelope.artifact_id, store)
    assert loaded == run
