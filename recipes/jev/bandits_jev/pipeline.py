"""One call from a decision dataset to a finished report.

``run_pipeline`` chains what the separate ``jev`` commands do -- score the
untrained base on test, train, score the trained adapter on calibration,
fit the temperature, score it on test, build the report -- and saves every
step's result as its own artifact exactly as those commands would. Nothing
here is new logic; it is the order, plus reuse: a step whose result is
already in the store (same dataset, split, model, adapter, mode, prompt) is
picked up instead of recomputed, so a run that stopped halfway resumes from
the last finished step, and a finished run repeated scores nothing.

No torch here: the model code comes in through ``make_predictor`` and
``make_trainable``, so the same pipeline drives the real Hugging Face model
on a GPU and the tiny model in tests.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from pathlib import Path

from bandits.store import Contract, DerivedStore
from bandits_jev.calibration import calibrate, compute_calibration_id, save_calibration
from bandits_jev.dataset import DecisionExample, load_decision_dataset
from bandits_jev.prompt import template_digest
from bandits_jev.report import build_report, compute_report_id, save_report, write_report
from bandits_jev.scorer import (
    LogitPredictor,
    ScoreMode,
    ScorerRun,
    adapter_digest,
    adapter_training_dataset_id,
    load_scorer_run,
    save_scorer_run,
    score_dataset,
)
from bandits_jev.trainer import (
    Trainable,
    TrainingRun,
    TrainingRunConfig,
    load_training_run,
    save_training_run,
    train,
)


class PipelineStep(Contract):
    name: str
    artifact_id: str
    reused: bool
    """True when the result was already in the store and nothing was run."""


class PipelineResult(Contract):
    dataset_id: str
    report_id: str
    best_adapter_path: str
    steps: tuple[PipelineStep, ...]
    written: tuple[str, ...]
    """The report files written to the output directory."""


def _find_scorer_run(
    store: DerivedStore,
    *,
    dataset_id: str,
    split: str,
    model_id: str,
    revision: str,
    adapter: str | None,
    trained_on: str | None,
    mode: ScoreMode,
    max_prompt_tokens: int,
) -> str | None:
    """An already-saved run of exactly this scoring, if there is one --
    including the adapter's recorded training dataset, so a reused trained
    run carries the provenance the report checks."""
    wanted = (
        dataset_id, split, model_id, revision, adapter, trained_on, mode, template_digest(), max_prompt_tokens
    )
    for envelope in sorted(store.list(kind="decision_scorer_run"), key=lambda e: e.artifact_id):
        run = load_scorer_run(envelope.artifact_id, store)
        have = (
            run.dataset_id,
            run.split,
            run.model_id,
            run.revision,
            run.adapter_digest,
            run.trained_on_dataset_id,
            run.mode,
            run.template_digest,
            run.max_prompt_tokens,
        )
        if have == wanted and run.predictions_source is None:
            return envelope.artifact_id
    return None


def _exists(store: DerivedStore, artifact_id: str) -> bool:
    try:
        store.read_envelope(artifact_id)
    except FileNotFoundError:
        return False
    return True


def _find_training_run(store: DerivedStore, config: TrainingRunConfig) -> tuple[str, TrainingRun] | None:
    for envelope in sorted(store.list(kind="decision_training_run"), key=lambda e: e.artifact_id):
        run = load_training_run(envelope.artifact_id, store)
        if run.config == config:
            best = next(c for c in run.checkpoints if c.step == run.best_checkpoint_step)
            if Path(best.adapter_path).is_dir():
                return envelope.artifact_id, run
    return None


def run_pipeline(
    store: DerivedStore,
    dataset_id: str,
    *,
    config: TrainingRunConfig,
    checkpoint_dir: str,
    output: Path,
    make_predictor: Callable[[str | None], LogitPredictor],
    make_trainable: Callable[[TrainingRunConfig], Trainable],
    two_order: bool = False,
    verifier_cost_id: str | None = None,
    gpu_usd_per_hour: float | None = None,
    draws: int = 2000,
    seed: int = 0,
    n_bins: int = 10,
    resume_from_step: int = 0,
    allow_test: bool = False,
    held_out_dataset_ids: tuple[str, ...] = (),
    log: Callable[[str], None] = lambda _message: None,
) -> PipelineResult:
    """``make_predictor(None)`` must load the untrained base named in
    ``config``; ``make_predictor(path)`` the base plus the adapter saved at
    ``path``. The test split is scored twice (untrained, trained), and each
    only if that exact scoring is not already saved. ``resume_from_step``
    continues an interrupted training run from the checkpoint it saved at
    that step (see ``trainer.train``). ``allow_test`` must be set: every run
    scores the locked test split, and that is a deliberate choice, as with
    ``jev score --allow-test``. ``held_out_dataset_ids`` are test-only
    datasets from sources the model never trained on (see
    ``dataset.as_test_only``); each is scored untrained and trained and
    becomes its own report section."""
    if not allow_test:
        raise ValueError(
            "the pipeline scores the locked test split; pass allow_test (--allow-test) to do so deliberately"
        )
    if config.dataset_id != dataset_id:
        raise ValueError(f"the training config names dataset {config.dataset_id!r}, not {dataset_id!r}")
    dataset = load_decision_dataset(dataset_id, store)
    by_split: dict[str, list[DecisionExample]] = {}
    for example in dataset.examples:
        by_split.setdefault(example.split, []).append(example)
    held_out: dict[str, list[DecisionExample]] = {}
    training_ids = {e.decision_id for e in dataset.examples if e.split != "test"}
    for held_out_id in held_out_dataset_ids:
        if held_out_id == dataset_id:
            raise ValueError("a held-out dataset must not be the training dataset")
        rows = [e for e in load_decision_dataset(held_out_id, store).examples if e.split == "test"]
        leaked = [e.decision_id for e in rows if e.decision_id in training_ids]
        if leaked:
            raise ValueError(f"{len(leaked)} held-out row(s) of {held_out_id} are also in training: {leaked[:3]}")
        if not rows:
            raise ValueError(f"held-out dataset {held_out_id} has no test rows")
        held_out[held_out_id] = rows
    missing = [s for s in ("train", "dev", "calibration", "test") if not by_split.get(s)]
    if missing:
        raise ValueError(f"dataset {dataset_id} has no {', '.join(missing)} rows; the pipeline needs all four splits")

    steps: list[PipelineStep] = []
    model_id, revision = config.base_model_id, config.base_revision

    def score(
        name: str,
        split: str,
        adapter_path: str | None,
        mode: ScoreMode,
        *,
        on_dataset: str | None = None,
        rows: list[DecisionExample] | None = None,
    ) -> str:
        target_id = on_dataset or dataset_id
        rows = rows if rows is not None else by_split[split]
        adapter = adapter_digest(adapter_path) if adapter_path is not None else None
        trained_on = adapter_training_dataset_id(adapter_path) if adapter_path is not None else None
        found = _find_scorer_run(
            store,
            dataset_id=target_id,
            split=split,
            model_id=model_id,
            revision=revision,
            adapter=adapter,
            trained_on=trained_on,
            mode=mode,
            max_prompt_tokens=config.max_prompt_tokens,
        )
        if found is not None:
            log(f"{name}: reusing {found}")
            steps.append(PipelineStep(name=name, artifact_id=found, reused=True))
            return found
        log(f"{name}: scoring {len(rows)} {split} rows")
        predictor = make_predictor(adapter_path)
        run: ScorerRun = score_dataset(
            predictor,
            rows,
            mode=mode,
            max_prompt_tokens=config.max_prompt_tokens,
            dataset_id=target_id,
            split=split,
            adapter_digest=adapter,
            trained_on_dataset_id=trained_on,
        )
        del predictor
        gc.collect()
        run_id = save_scorer_run(run, store).artifact_id
        steps.append(PipelineStep(name=name, artifact_id=run_id, reused=False))
        return run_id

    untrained_id = score("untrained on test", "test", None, "single_order")
    two_order_id = score("untrained on test (two orders)", "test", None, "two_order_average") if two_order else None

    found_training = _find_training_run(store, config)
    if found_training is not None:
        training_id, training = found_training
        log(f"training: reusing {training_id}")
        steps.append(PipelineStep(name="training", artifact_id=training_id, reused=True))
    else:
        log(f"training: {len(by_split['train'])} train rows, checkpoints in {checkpoint_dir}")
        trainable = make_trainable(config)
        training = train(
            trainable,
            config,
            train_examples=by_split["train"],
            dev_examples=by_split["dev"],
            checkpoint_dir=checkpoint_dir,
            resume_from_step=resume_from_step,
        )
        del trainable
        gc.collect()
        training_id = save_training_run(training, store).artifact_id
        steps.append(PipelineStep(name="training", artifact_id=training_id, reused=False))
    best = next(c for c in training.checkpoints if c.step == training.best_checkpoint_step)
    log(f"best checkpoint: step {best.step}, dev accuracy {best.dev.accuracy:.4f}")

    calibration_run_id = score("trained on calibration", "calibration", best.adapter_path, "single_order")
    calibration = calibrate(
        load_scorer_run(calibration_run_id, store), calibration_run_id, by_split["calibration"], n_bins=n_bins
    )
    calibration_id = compute_calibration_id(calibration)
    existed = _exists(store, calibration_id)
    save_calibration(calibration, store)
    steps.append(PipelineStep(name="calibration", artifact_id=calibration_id, reused=existed))
    log(f"temperature: {calibration.temperature:.4f}")

    trained_id = score("trained on test", "test", best.adapter_path, "single_order")
    external = []
    for held_out_id, rows in held_out.items():
        external.append(
            (
                score(f"untrained on held-out {held_out_id}", "test", None, "single_order",
                      on_dataset=held_out_id, rows=rows),
                score(f"trained on held-out {held_out_id}", "test", best.adapter_path, "single_order",
                      on_dataset=held_out_id, rows=rows),
            )
        )

    report = build_report(
        store,
        untrained_run_id=untrained_id,
        trained_run_id=trained_id,
        untrained_two_order_run_id=two_order_id,
        calibration_id=calibration_id,
        external=external,
        verifier_cost_id=verifier_cost_id,
        gpu_usd_per_hour=gpu_usd_per_hour,
        draws=draws,
        seed=seed,
        n_bins=n_bins,
    )
    report_id = compute_report_id(report)
    existed = _exists(store, report_id)
    save_report(report, store)
    steps.append(PipelineStep(name="report", artifact_id=report_id, reused=existed))
    written = write_report(report, output)
    return PipelineResult(
        dataset_id=dataset_id,
        report_id=report_id,
        best_adapter_path=best.adapter_path,
        steps=tuple(steps),
        written=tuple(str(p) for p in written),
    )
