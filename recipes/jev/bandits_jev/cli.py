"""The jev command: train your own Jev-style decision model on top of
Bandits.

A recipe, not part of Bandits core. It reads what core produces -- traces,
turn-judge runs, task sets, the artifact store -- and writes its own artifacts
beside them in the same .bandits project. Core never imports this package,
so removing recipes/jev leaves Bandits exactly as it was.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bandits.analyze import load_task_set
from bandits.store import ArtifactStore, DerivedStore

app = typer.Typer(no_args_is_help=True, help="Train your own Jev-style decision model on Bandits traces.")
console = Console()

_DEFAULT_PROJECT = Path(".")


def _derived(project: Path) -> DerivedStore:
    return DerivedStore(project / ".bandits")


def _load_task_set(task_set_id: str, project: Path):
    try:
        return load_task_set(task_set_id, _derived(project))
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no task set {task_set_id!r}")
        raise typer.Exit(code=1) from exc


def _corpus_traces(corpus_id: str, project: Path, trace_ids: tuple[str, ...] | None = None):
    try:
        corpus = ArtifactStore(project / ".bandits").read(corpus_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no corpus {corpus_id!r}")
        raise typer.Exit(code=1) from exc
    traces = corpus.traces
    if trace_ids is not None:
        wanted = set(trace_ids)
        traces = tuple(t for t in traces if t.trace_id in wanted)
    return traces


@app.command(name="dataset")
def dataset_command(
    judge_run_id: str,
    task_set_id: str = typer.Option(
        None,
        "--task-set",
        help="Split along this task set's own within-family fit/held-out membership "
        "(train/dev only). The task set must come from the same corpus as the judge run, "
        "and every judged trace must resolve to one of its families or it is quarantined. "
        "Omit to split by trace into train/dev/calibration/test (~70/10/10/10), keeping "
        "every trace's steps in one split.",
    ),
    minimum_valid_votes: int = typer.Option(
        1, "--minimum-valid-votes", min=1, help="Quarantine a turn with fewer successful votes."
    ),
    output: Path = typer.Option(None, "--output", help="Write examples and quarantine JSONL."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Compile a turn-judge run's verdicts into a decision dataset: every
    judged step becomes a question, the verifier's vote shares its label."""
    from bandits.verify.nextstate import load_turn_judge_run
    from bandits_jev.dataset import (
        LONG_STATE_CHARS,
        build_decision_dataset_from_corpus,
        save_decision_dataset,
        summarize_dataset,
        write_decision_dataset,
    )

    store = _derived(project)
    try:
        run = load_turn_judge_run(judge_run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no turn judge run {judge_run_id!r}")
        raise typer.Exit(code=1) from exc
    task_set = _load_task_set(task_set_id, project) if task_set_id else None
    traces = _corpus_traces(run.corpus_id, project, run.trace_ids)

    try:
        dataset = build_decision_dataset_from_corpus(
            traces,
            run,
            judge_run_id,
            task_set=task_set,
            task_set_id=task_set_id,
            minimum_valid_votes=minimum_valid_votes,
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_decision_dataset(dataset, store)
    console.print(f"decision_dataset_id: {envelope.artifact_id}")
    console.print(f"examples:            {dataset.counts.examples}")
    console.print(f"quarantined:         {dataset.counts.quarantined}")
    console.print(f"split by:            {'task set ' + task_set_id if task_set_id else 'trace'}")
    table = Table("split", "rows", "majority label (rows)", "state chars p50 / p99", f"> {LONG_STATE_CHARS:,} chars")
    for split, row in summarize_dataset(dataset).items():
        labels = ", ".join(f"{label} {n}" for label, n in row.majority_label.items()) or "—"
        table.add_row(split, str(row.rows), labels, f"{row.state_chars_p50:,} / {row.state_chars_p99:,}", str(row.long_states))
    console.print(table)
    if output is not None:
        rows_path, quarantine_path = write_decision_dataset(dataset, output)
        console.print(f"output:              {rows_path}")
        console.print(f"quarantine:          {quarantine_path}")


@app.command(name="import")
def import_command(
    path: Path = typer.Argument(..., help="JSONL file: one labeled decision per line."),
    source: str = typer.Option(None, "--source", help="Dataset-level source name/URL, used when a row omits its own."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
    output: Path = typer.Option(None, "--output", help="Write examples and quarantine JSONL."),
) -> None:
    """Import a JSONL file of user-labeled decisions into a DecisionDataset."""
    from bandits_jev.dataset import write_decision_dataset
    from bandits_jev.importer import import_jsonl, save_imported_dataset

    text = path.read_text(encoding="utf-8")
    dataset = import_jsonl(text, source_file=str(path), dataset_source=source)
    store = _derived(project)
    envelope = save_imported_dataset(dataset, store, source_file=str(path))
    console.print(f"decision_dataset_id: {envelope.artifact_id}")
    console.print(f"examples:            {dataset.counts.examples}")
    console.print(f"train:               {dataset.counts.train}")
    console.print(f"dev:                 {dataset.counts.dev}")
    console.print(f"calibration:         {dataset.counts.calibration}")
    console.print(f"test:                {dataset.counts.test}")
    console.print(f"quarantined:         {dataset.counts.quarantined}")
    if output is not None:
        rows_path, quarantine_path = write_decision_dataset(dataset, output)
        console.print(f"output:              {rows_path}")
        console.print(f"quarantine:          {quarantine_path}")


_DECISION_SPLITS = ("train", "dev", "calibration", "test")


@app.command(name="score")
def score_command(
    dataset_id: str,
    model: str = typer.Option(..., "--model", help="Hugging Face model id, e.g. Qwen/Qwen3.5-4B."),
    revision: str = typer.Option(..., "--revision", help="Pinned model revision (commit SHA)."),
    split: str = typer.Option("dev", "--split", help="Which split to score: train/dev/calibration/test."),
    allow_test: bool = typer.Option(
        False,
        "--allow-test",
        help="Required to score the locked test split. The test split is meant to be "
        "scored once, for the final report -- this flag exists so that never happens by accident.",
    ),
    two_order: bool = typer.Option(
        False, "--two-order", help="Average two option orders per example (two forward passes)."
    ),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    max_prompt_tokens: int = typer.Option(8_000, "--max-prompt-tokens"),
    adapter: Path = typer.Option(
        None,
        "--adapter",
        help="A LoRA adapter directory (a decision-train checkpoint) to score on top of the base "
        "model. Its digest is recorded in the run, so trained and untrained runs stay distinguishable.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Score a decision dataset's split with a frozen model -- the untrained
    base, or the base plus a trained adapter: one forward pass per example,
    softmax over the option-letter logits."""
    from bandits_jev.dataset import load_decision_dataset
    from bandits_jev.hf_predictor import HFPredictor
    from bandits_jev.scorer import (
        adapter_digest,
        adapter_training_dataset_id,
        save_scorer_run,
        score_dataset,
    )

    if split not in _DECISION_SPLITS:
        console.print(f"[red]error:[/red] --split must be one of {_DECISION_SPLITS}, got {split!r}")
        raise typer.Exit(code=1)
    if split == "test" and not allow_test:
        console.print(
            "[red]error:[/red] scoring the test split requires --allow-test; "
            "it is meant to be scored once, for the final report"
        )
        raise typer.Exit(code=1)

    store = _derived(project)
    try:
        dataset = load_decision_dataset(dataset_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no decision dataset {dataset_id!r}")
        raise typer.Exit(code=1) from exc

    examples = [e for e in dataset.examples if e.split == split]
    if not examples:
        console.print(f"[yellow]no examples in split {split!r}[/yellow]")
        raise typer.Exit(code=1)

    digest = None
    trained_on_dataset_id = None
    if adapter is not None:
        try:
            digest = adapter_digest(adapter)
            trained_on_dataset_id = adapter_training_dataset_id(adapter)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            console.print(f"[red]error:[/red] {adapter} is not a saved adapter: {exc}")
            raise typer.Exit(code=1) from exc
    predictor = HFPredictor(
        model,
        revision=revision,
        device=device,
        dtype=dtype,
        adapter_path=str(adapter) if adapter is not None else None,
    )
    mode = "two_order_average" if two_order else "single_order"
    run = score_dataset(
        predictor,
        examples,
        mode=mode,
        max_prompt_tokens=max_prompt_tokens,
        dataset_id=dataset_id,
        split=split,
        adapter_digest=digest,
        trained_on_dataset_id=trained_on_dataset_id,
    )
    envelope = save_scorer_run(run, store)
    console.print(f"scorer_run_id: {envelope.artifact_id}")
    console.print(f"scored:        {len(run.results)}")
    console.print(f"rejected:      {len(run.rejections)}")


@app.command(name="import-predictions")
def import_predictions_command(
    path: Path = typer.Argument(..., help="JSONL: {decision_id, probabilities: {option_id: p}, latency_seconds?}."),
    dataset_id: str = typer.Option(..., "--dataset", help="The decision dataset the predictions answer."),
    name: str = typer.Option(..., "--name", help="Who produced them, e.g. jev."),
    model: str = typer.Option(..., "--model", help="The producing model or API, as its provider names it."),
    revision: str = typer.Option(..., "--revision", help="Model/API version, or the date it was called."),
    split: str = typer.Option("test", "--split"),
    allow_test: bool = typer.Option(
        False,
        "--allow-test",
        help="Required to import predictions for the locked test split.",
    ),
    cost_usd: float = typer.Option(None, "--cost-usd", help="The actual bill for producing these predictions."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Import another system's predictions (e.g. real Jev) as a scorer run,
    so the evaluation report scores them with the same metrics. Items with no
    valid prediction are recorded as rejections, never dropped."""
    from bandits_jev.dataset import load_decision_dataset
    from bandits_jev.scorer import import_predictions, save_scorer_run

    if split not in _DECISION_SPLITS:
        console.print(f"[red]error:[/red] --split must be one of {_DECISION_SPLITS}, got {split!r}")
        raise typer.Exit(code=1)
    if split == "test" and not allow_test:
        console.print(
            "[red]error:[/red] importing predictions for the test split requires --allow-test; "
            "it is meant to be scored once, for the final report"
        )
        raise typer.Exit(code=1)
    store = _derived(project)
    try:
        dataset = load_decision_dataset(dataset_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no decision dataset {dataset_id!r}")
        raise typer.Exit(code=1) from exc
    examples = [e for e in dataset.examples if e.split == split]
    try:
        run = import_predictions(
            path.read_text(encoding="utf-8"),
            examples,
            name=name,
            model_id=model,
            revision=revision,
            dataset_id=dataset_id,
            split=split,
            cost_usd=cost_usd,
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_scorer_run(run, store)
    console.print(f"scorer_run_id: {envelope.artifact_id}")
    console.print(f"scored:        {len(run.results)}")
    console.print(f"rejected:      {len(run.rejections)}")


@app.command(name="calibrate")
def calibrate_command(
    scorer_run_id: str = typer.Argument(..., help="A scorer run over the calibration split (trained model)."),
    bins: int = typer.Option(10, "--bins", help="Fixed-width ECE bins."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Fit one temperature on a calibration-split scorer run's stored raw
    logits. Refuses any other split."""
    from bandits_jev.calibration import calibrate, save_calibration
    from bandits_jev.dataset import load_decision_dataset
    from bandits_jev.scorer import load_scorer_run

    store = _derived(project)
    try:
        run = load_scorer_run(scorer_run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no scorer run {scorer_run_id!r}")
        raise typer.Exit(code=1) from exc
    try:
        if run.dataset_id is None:
            raise ValueError("this scorer run does not record its dataset")
        dataset = load_decision_dataset(run.dataset_id, store)
        calibration = calibrate(
            run,
            scorer_run_id,
            [e for e in dataset.examples if e.split == run.split],
            n_bins=bins,
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_calibration(calibration, store)
    console.print(f"calibration_id: {envelope.artifact_id}")
    console.print(f"temperature:    {calibration.temperature:.4f}")
    if calibration.at_bound:
        console.print("[yellow]warning:[/yellow] the fit hit the search bound; the range decided it, not the data")
    console.print(f"rows:           {calibration.rows}")
    console.print(f"nll:            {calibration.nll_before:.4f} -> {calibration.nll_after:.4f}")
    console.print(f"brier:          {calibration.brier_before:.4f} -> {calibration.brier_after:.4f}")
    console.print(f"ece:            {calibration.ece_before:.4f} -> {calibration.ece_after:.4f}")


@app.command(name="report")
def report_command(
    untrained: str = typer.Option(..., "--untrained", help="Scorer run: base model, one option order."),
    trained: str = typer.Option(..., "--trained", help="Scorer run: base + trained adapter."),
    untrained_two_order: str = typer.Option(None, "--untrained-two-order", help="Scorer run: base model, two orders."),
    calibration: str = typer.Option(None, "--calibration", help="`jev calibrate` result for the trained model."),
    jev: str = typer.Option(None, "--jev", help="Imported Jev predictions run; the column shows 'not run' without it."),
    external: list[str] = typer.Option(
        [],
        "--external",
        help="UNTRAINED_RUN:TRAINED_RUN scored on a dataset the model never trained on. Repeatable.",
    ),
    draws: int = typer.Option(2000, "--draws", help="Bootstrap draws."),
    seed: int = typer.Option(0, "--seed", help="Bootstrap seed."),
    bins: int = typer.Option(10, "--bins", help="Fixed-width ECE / reliability bins."),
    output: Path = typer.Option(..., "--output", help="Directory for report.json, report.md and charts."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Build the evaluation report from saved scorer runs and a calibration.
    Scores nothing; the same artifacts always regenerate the same bytes."""
    from bandits_jev.report import build_report, save_report, write_report

    pairs = []
    for item in external:
        if item.count(":") != 1:
            console.print(f"[red]error:[/red] --external must be UNTRAINED_RUN:TRAINED_RUN, got {item!r}")
            raise typer.Exit(code=1)
        pairs.append(tuple(item.split(":")))
    store = _derived(project)
    try:
        report = build_report(
            store,
            untrained_run_id=untrained,
            trained_run_id=trained,
            untrained_two_order_run_id=untrained_two_order,
            calibration_id=calibration,
            jev_run_id=jev,
            external=pairs,
            draws=draws,
            seed=seed,
            n_bins=bins,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] missing artifact: {exc.filename or exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_report(report, store)
    written = write_report(report, output)
    console.print(f"decision_report_id: {envelope.artifact_id}")
    for path in written:
        console.print(f"wrote:              {path}")
    if any(not r.used_in_report for usage in report.test_usage for r in usage.runs):
        console.print("[yellow]warning:[/yellow] the test split was scored by runs this report does not use")


@app.command(name="train")
def train_command(
    dataset_id: str,
    model: str = typer.Option(..., "--model", help="Hugging Face base model id, e.g. Qwen/Qwen3.5-4B."),
    revision: str = typer.Option(..., "--revision", help="Pinned base model revision (commit SHA)."),
    seed: int = typer.Option(..., "--seed", help="Data-order and option-shuffle seed. Required, not defaulted, "
    "so a reproducible run is a deliberate choice, not an accident."),
    eval_every_steps: int = typer.Option(50, "--eval-every-steps", help="Score dev every N steps; 0 disables "
    "interval evals and scores only a single final checkpoint."),
    resume_from_step: int = typer.Option(0, "--resume-from-step", help="Continue from the checkpoint this run "
    "saved at that step in --checkpoint-dir (weights, optimizer, schedule, history); same flags required."),
    lora_rank: int = typer.Option(16, "--lora-rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate"),
    warmup_ratio: float = typer.Option(0.1, "--warmup-ratio", help="Fraction of steps warming up linearly "
    "to --learning-rate; linear decay to zero after."),
    effective_batch: int = typer.Option(8, "--effective-batch"),
    epochs: int = typer.Option(1, "--epochs"),
    max_prompt_tokens: int = typer.Option(8_000, "--max-prompt-tokens"),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    checkpoint_dir: Path = typer.Option(..., "--checkpoint-dir", help="Where LoRA adapter checkpoints are saved."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """LoRA SFT on a decision dataset's train split, evaluated against dev.
    Never called sft -- in Bandits that word already means exporting SFT
    rows (build-sft/export-nextstate), not training a model."""
    from pydantic import ValidationError

    from bandits_jev.dataset import load_decision_dataset
    from bandits_jev.hf_trainer import HFTrainer
    from bandits_jev.trainer import (
        build_training_config,
        check_checkpoint_dir,
        save_training_run,
        train,
    )

    store = _derived(project)
    try:
        dataset = load_decision_dataset(dataset_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no decision dataset {dataset_id!r}")
        raise typer.Exit(code=1) from exc

    train_examples = [e for e in dataset.examples if e.split == "train"]
    dev_examples = [e for e in dataset.examples if e.split == "dev"]
    if not train_examples:
        console.print("[red]error:[/red] no examples in the train split")
        raise typer.Exit(code=1)
    if not dev_examples:
        console.print("[red]error:[/red] no examples in the dev split; checkpoints are selected by dev")
        raise typer.Exit(code=1)

    try:
        config = build_training_config(
            base_model_id=model,
            base_revision=revision,
            dataset_id=dataset_id,
            seed=seed,
            eval_every_steps=eval_every_steps,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            effective_batch=effective_batch,
            epochs=epochs,
            max_prompt_tokens=max_prompt_tokens,
            dtype=dtype,
            device=device,
        )
    except ValidationError as exc:
        console.print(f"[red]error:[/red] invalid training settings:\n{exc}")
        raise typer.Exit(code=1) from exc
    try:
        check_checkpoint_dir(str(checkpoint_dir), resume_from_step)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    trainer = HFTrainer.from_config(config)
    try:
        run = train(
            trainer,
            config,
            train_examples=train_examples,
            dev_examples=dev_examples,
            checkpoint_dir=str(checkpoint_dir),
            resume_from_step=resume_from_step,
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_training_run(run, store)
    best = next(c for c in run.checkpoints if c.step == run.best_checkpoint_step)
    console.print(f"training_run_id:      {envelope.artifact_id}")
    console.print(f"steps_completed:      {run.steps_completed}")
    console.print(f"rejected_train:       {len(run.rejected_train)}")
    console.print(f"checkpoints:          {len(run.checkpoints)}")
    console.print(f"best_checkpoint_step: {best.step}")
    console.print(f"best_dev_accuracy:    {best.dev.accuracy:.4f} ({best.dev.scored} scored, {best.dev.rejected} rejected)")
    console.print(f"best_adapter_path:    {best.adapter_path}")


if __name__ == "__main__":
    app()
