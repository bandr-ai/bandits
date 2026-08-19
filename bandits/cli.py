"""``bandits`` -- the command line that runs the whole loop.

    bandits ingest   traces.otlp.jsonl --source otlp --tools tools.json -o corpus.json
    bandits surface  corpus.json                                        -o surface.json
    bandits schema   corpus.json surface.json                           -o schema.json
    bandits tasks    corpus.json schema.json                            -o tasks.json
    bandits verify   tasks.json corpus.json schema.json                 -o verifiers.json
    bandits fidelity corpus.json schema.json tasks.json                 -o fidelity.json
    bandits run      traces.otlp.jsonl --source otlp --tools tools.json -o out/

Three rules this file exists to enforce:

**``--source`` is required and never sniffed.** Format detection is a silent
failure waiting to happen: a chat export misread as OTLP yields zero invocation
points and looks like a tool-free episode rather than an error.

**Nothing is dropped quietly.** Ingest issues, skipped traces, solvability
warnings and unresolved tools are printed at every stage. A silently dropped
trace is exactly what this project exists to avoid.

**``fidelity`` exits nonzero when the gate rejects.** It is a CI gate, not a
report. ``run`` behaves the same way, so a pipeline that rebuilds an environment
fails the build when the environment stops reproducing its own traces.

Artifacts are pydantic JSON and round-trip losslessly. The bundle wrappers below
exist because some stages produce a *list* of contract objects and
``contracts.py`` is frozen -- they are transport for this CLI, never new
contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from bandits.contracts import (
    FidelityReport,
    JsonObject,
    StateSchema,
    TaskCase,
    ToolSurface,
    TraceCorpus,
    Verifier,
)
from bandits.fidelity import (
    DEFAULT_PER_TOOL_FLOOR,
    DEFAULT_THRESHOLD,
    GateCriteria,
    build_report,
    render,
    replay_corpus,
    to_json,
    tool_classes_from_surface,
)
from bandits.ingest import CANONICAL_SOURCES, load_corpus_and_registry
from bandits.state import infer_schema
from bandits.surface import build_surface
from bandits.task import mine_tasks
from bandits.triage import Verdict, render_report, triage_corpus
from bandits.verify import UnlabeledTraceError, synthesize_verifier

app = typer.Typer(
    name="bandits",
    help="Turn production agent traces into executable, verifiable RL environments.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

__all__ = ["TaskBundle", "VerifierBundle", "FidelityBundle", "app", "main"]


# -- artifact envelopes ----------------------------------------------------


class TaskBundle(BaseModel):
    """``tasks.json``: the mined tasks plus everything mining refused, with reasons."""

    tasks: tuple[TaskCase, ...] = ()
    skipped: tuple[JsonObject, ...] = ()
    warned: tuple[str, ...] = ()
    """task_ids carrying solvability warnings. Do not train on these unreviewed."""


class VerifierBundle(BaseModel):
    """``verifiers.json``: synthesized reward functions plus the tasks we refused one for."""

    verifiers: tuple[Verifier, ...] = ()
    skipped: tuple[JsonObject, ...] = ()


class FidelityBundle(BaseModel):
    """``fidelity.json``: the corpus verdict plus one report per environment."""

    overall: FidelityReport
    per_trace: tuple[FidelityReport, ...] = ()


# -- io --------------------------------------------------------------------


def _write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[dim]wrote[/dim] [cyan]{path}[/cyan]")


def _read(path: Path, model: type[BaseModel]) -> Any:
    if not path.exists():
        console.print(f"[bold red]missing artifact:[/bold red] {path}")
        raise typer.Exit(code=2)
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _check_source(source: str) -> None:
    if source not in CANONICAL_SOURCES:
        console.print(
            f"[bold red]unknown --source {source!r}[/bold red]; declare one of "
            f"{list(CANONICAL_SOURCES)}. Formats are never sniffed."
        )
        raise typer.Exit(code=2)


# -- reporting helpers -----------------------------------------------------


def _report_issues(corpus: TraceCorpus) -> None:
    console.print(
        f"[bold]ingest[/bold]: {len(corpus.traces)} traces, "
        f"{sum(len(t.invocations) for t in corpus.traces)} invocation points, "
        f"{len(corpus.issues)} issue(s)"
    )
    for issue in corpus.issues:
        console.print(f"  [yellow]issue[/yellow] {issue.kind}: {issue.detail} ({issue.location})")


def _report_surface(surface: ToolSurface) -> None:
    table = Table(title="action space", title_justify="left", show_edge=False)
    table.add_column("tool", style="cyan")
    table.add_column("class")
    table.add_column("calls", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("flags", style="dim")
    for profile in surface.tools:
        flags = []
        if profile.observed_only:
            flags.append("observed-only")
        if profile.declared_only:
            flags.append("declared-only (probing candidate)")
        table.add_row(
            profile.name,
            profile.tool_class.value,
            str(profile.call_count),
            str(len(profile.error_modes)),
            ", ".join(flags),
        )
    console.print(table)


def _report_schema(schema: StateSchema) -> None:
    table = Table(title="inferred state", title_justify="left", show_edge=False)
    table.add_column("entity", style="cyan")
    table.add_column("pk")
    table.add_column("fields", justify="right")
    table.add_column("written by")
    table.add_column("read by")
    table.add_column("note", style="dim")
    for entity in schema.entities:
        table.add_row(
            entity.name,
            entity.primary_key or "-",
            str(len(entity.fields)),
            ", ".join(entity.written_by) or "-",
            ", ".join(entity.read_by) or "-",
            "static snapshot (structure undetermined)" if entity.static_snapshot else "",
        )
    console.print(table)
    if schema.unresolved:
        console.print(
            "[yellow]unresolved tools[/yellow] (responses attributed to no entity): "
            + ", ".join(schema.unresolved)
        )


def _report_tasks(bundle: TaskBundle) -> None:
    console.print(f"[bold]tasks[/bold]: {len(bundle.tasks)} mined, {len(bundle.skipped)} skipped")
    for skip in bundle.skipped:
        console.print(f"  [yellow]skipped[/yellow] {skip.get('trace_id')}: {skip.get('reason')}")
    for task in bundle.tasks:
        warnings = task.provenance.get("solvability_warnings") or ()
        for warning in warnings:
            console.print(f"  [yellow]warning[/yellow] {task.task_id}: {warning}")
        blocked = task.provenance.get("blocked_post_state_reads") or ()
        if blocked:
            console.print(
                f"  [dim]note[/dim] {task.task_id}: {len(blocked)} post-write read(s) "
                f"excluded from the pre-state"
            )


# -- shared pipeline steps -------------------------------------------------


def _build_tasks(corpus: TraceCorpus, schema: StateSchema) -> TaskBundle:
    mining = mine_tasks(corpus, schema)
    return TaskBundle(
        tasks=tuple(mining.tasks),
        skipped=tuple(mining.skipped),
        warned=tuple(t.task_id for t in mining.warned),
    )


def _build_verifiers(
    corpus: TraceCorpus, schema: StateSchema, tasks: tuple[TaskCase, ...]
) -> VerifierBundle:
    by_id = {t.trace_id: t for t in corpus.traces}
    verifiers: list[Verifier] = []
    skipped: list[JsonObject] = []
    for task in tasks:
        trace = by_id.get(task.trace_id)
        if trace is None:
            skipped.append({"task_id": task.task_id, "reason": "source trace not in this corpus"})
            continue
        try:
            verifiers.append(synthesize_verifier(task, trace, schema))
        except UnlabeledTraceError as exc:
            skipped.append({"task_id": task.task_id, "reason": str(exc)})
        except ValueError as exc:
            skipped.append({"task_id": task.task_id, "reason": f"{type(exc).__name__}: {exc}"})
    return VerifierBundle(verifiers=tuple(verifiers), skipped=tuple(skipped))


def _run_gate(
    corpus: TraceCorpus,
    schema: StateSchema,
    tasks: tuple[TaskCase, ...],
    surface: ToolSurface,
    criteria: GateCriteria,
) -> FidelityBundle:
    tool_classes = tool_classes_from_surface(surface)
    results = replay_corpus(corpus, schema, tasks, tool_classes, surface=surface)
    per_trace = tuple(build_report([r], criteria=criteria) for r in results)
    overall = build_report(results, criteria=criteria)
    return FidelityBundle(overall=overall, per_trace=per_trace)


# -- commands --------------------------------------------------------------

SourceOpt = Annotated[
    str,
    typer.Option(
        "--source",
        help=f"Declared adapter, one of {list(CANONICAL_SOURCES)}. Required; never sniffed.",
    ),
]
ToolsOpt = Annotated[
    Path | None,
    typer.Option("--tools", help="Declared tool registry JSON. Optional but strongly advised."),
]


@app.command()
def triage(
    traces: Annotated[Path, typer.Argument(help="Raw export, JSONL or a JSON array.")],
    source: SourceOpt,
    tools: ToolsOpt = None,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write the report as JSON.")] = None,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit nonzero on PARTIAL as well as NO_GO."),
    ] = False,
) -> None:
    """Can an environment be built from this telemetry at all? Run this first.

    EXITS NONZERO ON NO_GO. This is the cheap upstream check, not the fidelity
    gate: a GO says reconstruction is worth attempting, never that it will pass.
    """
    _check_source(source)
    corpus, _registry = load_corpus_and_registry(traces, source, tools_path=tools)
    report = triage_corpus(corpus)
    render_report(report, console)
    if out is not None:
        out.write_text(json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8")
    if report.verdict is Verdict.NO_GO or (strict and report.verdict is not Verdict.GO):
        raise typer.Exit(code=1)


@app.command()
def ingest(
    traces: Annotated[Path, typer.Argument(help="Raw export, JSONL or a JSON array.")],
    source: SourceOpt,
    tools: ToolsOpt = None,
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("corpus.json"),
) -> None:
    """Normalize a raw vendor export into a TraceCorpus."""
    _check_source(source)
    corpus, _registry = load_corpus_and_registry(traces, source, tools_path=tools)
    _report_issues(corpus)
    _write(out, corpus)


@app.command()
def surface(
    corpus_path: Annotated[Path, typer.Argument(metavar="CORPUS")],
    tools: ToolsOpt = None,
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("surface.json"),
) -> None:
    """Profile the action space: arguments, responses, error modes, read/write/external."""
    corpus = _read(corpus_path, TraceCorpus)
    registry = json.loads(tools.read_text(encoding="utf-8")) if tools else None
    result = build_surface(corpus, declared_tools=registry)
    _report_surface(result)
    _write(out, result)


@app.command(name="schema")
def schema_cmd(
    corpus_path: Annotated[Path, typer.Argument(metavar="CORPUS")],
    surface_path: Annotated[Path, typer.Argument(metavar="SURFACE")],
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("schema.json"),
) -> None:
    """Infer the database behind the tool responses."""
    corpus = _read(corpus_path, TraceCorpus)
    tool_surface = _read(surface_path, ToolSurface)
    result = infer_schema(corpus, tool_surface)
    _report_schema(result)
    _write(out, result)


@app.command()
def tasks(
    corpus_path: Annotated[Path, typer.Argument(metavar="CORPUS")],
    schema_path: Annotated[Path, typer.Argument(metavar="SCHEMA")],
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("tasks.json"),
) -> None:
    """Mine TaskCases: instruction plus the reconstructed starting state."""
    corpus = _read(corpus_path, TraceCorpus)
    schema = _read(schema_path, StateSchema)
    bundle = _build_tasks(corpus, schema)
    _report_tasks(bundle)
    _write(out, bundle)


@app.command()
def verify(
    tasks_path: Annotated[Path, typer.Argument(metavar="TASKS")],
    corpus_path: Annotated[Path, typer.Argument(metavar="CORPUS")],
    schema_path: Annotated[Path, typer.Argument(metavar="SCHEMA")],
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("verifiers.json"),
) -> None:
    """Synthesize the reward functions. Code, never a judge."""
    bundle_in: TaskBundle = _read(tasks_path, TaskBundle)
    corpus = _read(corpus_path, TraceCorpus)
    schema = _read(schema_path, StateSchema)
    bundle = _build_verifiers(corpus, schema, bundle_in.tasks)
    console.print(
        f"[bold]verify[/bold]: {len(bundle.verifiers)} verifier(s), {len(bundle.skipped)} refused"
    )
    for skip in bundle.skipped:
        console.print(f"  [yellow]no verifier[/yellow] {skip.get('task_id')}: {skip.get('reason')}")
    if bundle.verifiers:
        console.print(
            "[yellow]reviewed_by is unset on every synthesized verifier.[/yellow] "
            "A generated reward function nobody has read must not grade anything."
        )
    _write(out, bundle)


@app.command()
def fidelity(
    corpus_path: Annotated[Path, typer.Argument(metavar="CORPUS")],
    schema_path: Annotated[Path, typer.Argument(metavar="SCHEMA")],
    tasks_path: Annotated[Path, typer.Argument(metavar="TASKS")],
    surface_path: Annotated[
        Path | None,
        typer.Option("--surface", help="surface.json. Rebuilt from the corpus when omitted."),
    ] = None,
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("fidelity.json"),
    threshold: Annotated[float, typer.Option(help="Overall rate required to accept.")] = DEFAULT_THRESHOLD,
    per_tool_floor: Annotated[
        float, typer.Option(help="No single tool may fall below this.")
    ] = DEFAULT_PER_TOOL_FLOOR,
    min_calls: Annotated[
        int, typer.Option(help="Calls a tool needs before the per-tool floor is enforced.")
    ] = 1,
) -> None:
    """Replay the corpus against the rebuilt environments. EXITS NONZERO ON REJECT."""
    corpus = _read(corpus_path, TraceCorpus)
    schema = _read(schema_path, StateSchema)
    bundle_in: TaskBundle = _read(tasks_path, TaskBundle)
    tool_surface = (
        _read(surface_path, ToolSurface) if surface_path else build_surface(corpus)
    )
    criteria = GateCriteria(threshold, per_tool_floor, min_calls)
    bundle = _run_gate(corpus, schema, bundle_in.tasks, tool_surface, criteria)
    render(bundle.overall, console, floor=per_tool_floor)
    _write(out, bundle)
    if not bundle.overall.accepted:
        raise typer.Exit(code=1)


@app.command()
def run(
    traces: Annotated[Path, typer.Argument(help="Raw export, JSONL or a JSON array.")],
    source: SourceOpt,
    tools: ToolsOpt = None,
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("out"),
    threshold: Annotated[float, typer.Option()] = DEFAULT_THRESHOLD,
    per_tool_floor: Annotated[float, typer.Option()] = DEFAULT_PER_TOOL_FLOOR,
    min_calls: Annotated[int, typer.Option()] = 1,
) -> None:
    """Run the whole loop and print the fidelity table last. Exits nonzero on reject."""
    _check_source(source)
    out.mkdir(parents=True, exist_ok=True)

    corpus, registry = load_corpus_and_registry(traces, source, tools_path=tools)
    _report_issues(corpus)
    _write(out / "corpus.json", corpus)

    tool_surface = build_surface(corpus, declared_tools=registry or None)
    _report_surface(tool_surface)
    _write(out / "surface.json", tool_surface)

    schema = infer_schema(corpus, tool_surface)
    _report_schema(schema)
    _write(out / "schema.json", schema)

    task_bundle = _build_tasks(corpus, schema)
    _report_tasks(task_bundle)
    _write(out / "tasks.json", task_bundle)

    verifier_bundle = _build_verifiers(corpus, schema, task_bundle.tasks)
    console.print(
        f"[bold]verify[/bold]: {len(verifier_bundle.verifiers)} verifier(s), "
        f"{len(verifier_bundle.skipped)} refused"
    )
    for skip in verifier_bundle.skipped:
        console.print(f"  [yellow]no verifier[/yellow] {skip.get('task_id')}: {skip.get('reason')}")
    _write(out / "verifiers.json", verifier_bundle)

    criteria = GateCriteria(threshold, per_tool_floor, min_calls)
    fidelity_bundle = _run_gate(corpus, schema, task_bundle.tasks, tool_surface, criteria)
    _write(out / "fidelity.json", fidelity_bundle)
    console.print()
    render(fidelity_bundle.overall, console, floor=per_tool_floor)
    if not fidelity_bundle.overall.accepted:
        raise typer.Exit(code=1)


@app.command(name="show")
def show(
    fidelity_path: Annotated[Path, typer.Argument(metavar="FIDELITY")],
    per_trace: Annotated[bool, typer.Option("--per-trace")] = False,
) -> None:
    """Re-render a saved fidelity.json. Exits nonzero when it records a rejection."""
    bundle: FidelityBundle = _read(fidelity_path, FidelityBundle)
    if per_trace:
        for report in bundle.per_trace:
            render(report, console)
            console.print()
    render(bundle.overall, console)
    console.print_json(json.dumps(to_json(bundle.overall)["totals"]))
    if not bundle.overall.accepted:
        raise typer.Exit(code=1)


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
