"""Command-line interface: ingest a trace export, then inspect what's stored."""

from __future__ import annotations

import functools
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from bandits import ledger
from bandits.analyze import (
    DEFAULT_HELD_OUT,
    analyze_corpus,
    load_analysis,
    load_task_set,
    save_analysis,
    save_task_set,
)
from bandits.analyze.rlm_audit import (
    DEFAULT_MAX_TOKENS as RLM_AUDIT_MAX_TOKENS,
)
from bandits.analyze.rlm_audit import (
    DEFAULT_MODEL as RLM_AUDIT_MODEL,
)
from bandits.analyze.rlm_audit import (
    ClusteringAuditError,
    _run_members,
    audit_clustering,
)
from bandits.analyze.rlm_audit import (
    build_predictor as build_taxonomy_audit_predictor,
)
from bandits.analyze.rlm_audit import (
    load_audit as load_rlm_audit,
)
from bandits.analyze.rlm_audit import (
    prompt_digest as rlm_audit_prompt_digest,
)
from bandits.analyze.rlm_audit import (
    save_audit as save_rlm_audit,
)
from bandits.analyze.rlm_audit_session import (
    AuditSessionRecorder,
    AuditSessionStore,
    new_audit_session_id,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_mine import (
    DEFAULT_CHUNK_SIZE as RLM_CHUNK_SIZE,
)
from bandits.analyze.rlm_mine import (
    DEFAULT_MAX_TOKENS as RLM_MAX_TOKENS,
)
from bandits.analyze.rlm_mine import (
    DEFAULT_MODEL as RLM_MODEL,
)
from bandits.analyze.rlm_mine import (
    DEFAULT_SEED as RLM_SEED,
)
from bandits.analyze.rlm_mine import (
    MiningError,
    load_clustering_run,
    mine_taxonomy,
    save_clustering_run,
)
from bandits.analyze.rlm_mine import (
    build_predictor as build_rlm_predictor,
)
from bandits.analyze.rlm_models import (
    DEFAULT_PASSES as RLM_PASSES,
)
from bandits.analyze.rlm_models import (
    AuditBudget,
    Budget,
    TraceView,
)
from bandits.analyze.rlm_session import SessionRecorder, SessionStore, new_session_id
from bandits.analyze.rlm_taskset import MaterializationError, materialize_task_set
from bandits.analyze.rlm_view import (
    audit_live_panel,
    family_card,
    live_panel,
    print_pass_history,
    taxonomy_overview,
)
from bandits.export import (
    build_direct_sft,
    save_direct_sft,
    write_direct_sft,
)
from bandits.ingest import CANONICAL_SOURCES, UnknownSourceError, load_corpus
from bandits.redact import DEFAULT_RULESET, ruleset_by_name
from bandits.store import ArtifactStore, DerivedStore
from bandits.verify.judge import DEFAULT_MODEL, JudgeError

app = typer.Typer(add_completion=False)
console = Console()

_MAX_INLINE_ISSUES = 3
_DEFAULT_PROJECT = Path(".")
_SINGLETON_WARNING = 0.8
"""Fraction of one-trace families above which grouping is reported as inert."""


@app.command()
def ingest(
    path: Path,
    source: str = typer.Option(..., "--source", help=f"One of: {', '.join(CANONICAL_SOURCES)}"),
    redaction: str = typer.Option(
        DEFAULT_RULESET.name,
        "--redaction",
        help="Redaction ruleset. 'secrets-only-v1' keeps email addresses, which are "
        "often the task's own identifier.",
    ),
    control_marker: list[str] = typer.Option(
        [],
        "--control-marker",
        help="Literal token this export writes into a message's own text that is "
        "not part of the user's request (tau2's '###TRANSFER###', for one). "
        "Repeatable. Declared once here rather than left for every downstream "
        "RLM command to remember: every mining, audit and assignment run "
        "reading this corpus strips it before a model ever sees it.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Load a trace export into the local artifact store."""
    try:
        corpus = load_corpus(path, source, ruleset_by_name(redaction))
    except (UnknownSourceError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if control_marker:
        corpus = corpus.replace(control_markers=tuple(control_marker))

    store = ArtifactStore(project / ".bandits")
    envelope = store.write(corpus, source_path=str(path))

    console.print(f"artifact_id: {envelope.artifact_id}")
    console.print(f"source:      {envelope.source}")
    console.print(f"traces:      {envelope.trace_count}")
    console.print(f"spans:       {envelope.span_count}")
    console.print(f"issues:      {envelope.issue_count}")
    console.print(f"redaction:   {corpus.redaction_ruleset}")
    for issue in corpus.issues[:_MAX_INLINE_ISSUES]:
        location = f" at {issue.location}" if issue.location else ""
        console.print(f"  - {issue.kind}{location}: {issue.detail}")
    remaining = envelope.issue_count - _MAX_INLINE_ISSUES
    if remaining > 0:
        console.print(f"  (+{remaining} more — see `bandits show {envelope.artifact_id} --issues`)")


@app.command(name="list")
def list_artifacts(project: Path = typer.Option(_DEFAULT_PROJECT, "--project")) -> None:
    """List every artifact in the local store."""
    store = ArtifactStore(project / ".bandits")
    table = Table("artifact_id", "source", "traces", "spans", "issues", "created_at")
    for envelope in store.list():
        table.add_row(
            envelope.artifact_id[:19],
            envelope.source,
            str(envelope.trace_count),
            str(envelope.span_count),
            str(envelope.issue_count),
            envelope.created_at,
        )
    console.print(table)


@app.command()
def show(
    artifact_id: str,
    trace: str = typer.Option(None, "--trace"),
    issues: bool = typer.Option(False, "--issues"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Inspect one stored artifact."""
    store = ArtifactStore(project / ".bandits")
    corpus = store.read(artifact_id)

    if issues:
        table = Table("kind", "location", "detail")
        for issue in corpus.issues:
            table.add_row(issue.kind, issue.location or "", issue.detail)
        console.print(table)
        return

    if trace is not None:
        traced = next((t for t in corpus.traces if t.trace_id == trace), None)
        if traced is None:
            console.print(f"[red]error:[/red] no trace {trace!r} in {artifact_id}")
            raise typer.Exit(code=1)
        table = Table("span_id", "parent_span_id", "kind", "name", "status", "output")
        for span in traced.spans:
            output = str(span.output)
            table.add_row(
                span.span_id,
                span.parent_span_id or "",
                span.kind.value,
                span.name,
                span.status.value,
                output[:80],
            )
        console.print(table)
        return

    table = Table("trace_id", "task", "spans")
    for traced in corpus.traces:
        task = (traced.task or "")[:60]
        table.add_row(traced.trace_id, task, str(len(traced.spans)))
    console.print(table)


@app.command()
def analyze(
    artifact_id: str,
    tasks: bool = typer.Option(False, "--tasks", help="List every extracted task candidate."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Extract task candidates and outcome evidence from a stored corpus."""
    store = ArtifactStore(project / ".bandits")
    try:
        corpus = store.read(artifact_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no artifact {artifact_id!r}")
        raise typer.Exit(code=1) from exc

    analysis = analyze_corpus(corpus)
    envelope = save_analysis(analysis, DerivedStore(project / ".bandits"))

    console.print(f"analysis_id: {envelope.artifact_id}")
    console.print(f"corpus:      {analysis.corpus_id}")
    console.print(f"tasks:       {len(analysis.tasks)}")
    console.print(f"evidence:    {len(analysis.evidence)}")

    if tasks:
        table = Table("task_id", "instruction", "outcome evidence", "limitations")
        for task in analysis.tasks:
            table.add_row(
                task.task_id,
                (task.instruction or "[dim]none declared[/dim]")[:40],
                str(len(task.outcome_evidence_ids)),
                str(len(task.limitations)),
            )
        console.print(table)

    # Printed last and never suppressed: what could not be read matters as much
    # as what could, and burying it under a summary is how a corpus gets trusted
    # further than its evidence supports.
    for limitation in analysis.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


def _derived(project: Path) -> DerivedStore:
    return DerivedStore(project / ".bandits")


def _load_task_set(task_set_id: str, project: Path):
    try:
        return load_task_set(task_set_id, _derived(project))
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no task set {task_set_id!r}")
        raise typer.Exit(code=1) from exc


def _report(task_set, envelope_id: str) -> None:
    """Coverage first, then what the selection could not reach. Both, always."""
    console.print(f"taskset_id:  {envelope_id}")
    console.print(f"families:    {len(task_set.families)}")
    console.print(f"selected:    {len(task_set.selected)}")
    console.print(
        f"coverage:    {task_set.workload_coverage:.1%} "
        f"of {task_set.total_workload_mass} production run(s)"
    )
    if task_set.underfilled:
        console.print("[yellow]underfilled:[/yellow] eligibility ran out before the budget did")

    # Printed with the families rather than buried in the artifact: two task sets
    # from one analysis differ only by this, and the summary above them reads the
    # same either way.
    clustering = task_set.clustering
    if clustering is None:
        console.print(
            "[yellow]clustering:[/yellow] this task set records nothing about how it "
            "was grouped and cannot be reproduced from the artifact alone"
        )
    else:
        pinned = (
            f", {clustering.embedding_model} ({clustering.embedding_cache_id})"
            if clustering.embedding_model
            else ""
        )
        console.print(
            f"clustering:  {clustering.backend} at similarity {clustering.similarity:g}, "
            f"{clustering.neighbors} neighbor(s){pinned}"
        )

    # A grouping stage that grouped nothing is not obviously broken from the
    # summary above: coverage still reads high when every trace is its own
    # family. Saying so is what makes an inert backend visible.
    singletons = sum(1 for f in task_set.families if f.workload_mass == 1)
    if task_set.families and singletons > _SINGLETON_WARNING * len(task_set.families):
        console.print(
            f"[yellow]warning:[/yellow] {singletons} of {len(task_set.families)} families "
            "contain one trace; grouping found almost no structure — the corpus may be "
            "genuinely diverse, or --similarity may be too high for this backend"
        )

    table = Table("family_id", "descriptor", "mass", "medoid", "fit", "held out", "status")
    for family in sorted(task_set.families, key=lambda f: -f.workload_mass):
        table.add_row(
            family.family_id,
            family.descriptor[:44],
            str(family.workload_mass),
            family.medoid_trace_id,
            str(len(family.fit_trace_ids)),
            str(len(family.held_out_trace_ids)),
            family.review_status,
        )
    console.print(table)

    for slot in task_set.missing_slots:
        console.print(f"[yellow]missing slot[/yellow] {slot.slot}: {slot.reason}")

    # An over-merged family is invisible in the table above: it reads as one
    # large healthy group. Verifiers are drafted per family, so it has to be
    # said here rather than only under `families --family`.
    for family in task_set.families:
        # Read off the measurement, not off the limitation prose: a family also
        # carries limitations about lineage and splits, and printing those under
        # an over-merged heading would attribute them to the wrong finding.
        coherence = family.coherence
        if coherence is None:
            for limitation in family.limitations:
                if limitation.startswith("coherence was not recomputed"):
                    console.print(
                        f"[yellow]coherence unknown[/yellow] {family.family_id}: {limitation}"
                    )
            continue
        if not coherence.over_merged:
            continue
        left, right = coherence.widest_pair
        console.print(
            f"[yellow]over-merged[/yellow] {family.family_id}: widest pair is "
            f"{coherence.diameter:.2f} apart, over {coherence.diameter_factor:g}x the "
            f"{coherence.link_threshold:.2f} that admitted any single link — "
            f"{left!r} vs {right!r}"
        )

    for limitation in task_set.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command()
def families(
    task_set_id: str,
    family: str = typer.Option(None, "--family", help="Show one family's members in full."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Inspect mined families and the selection drawn from them."""
    task_set = _load_task_set(task_set_id, project)

    if family is None:
        _report(task_set, task_set_id)
        table = Table("slot", "trace_id", "family_id")
        for selection in task_set.selected:
            table.add_row(selection.slot.value, selection.trace_id, selection.family_id)
        console.print(table)
        return

    found = task_set.family_by_id().get(family)
    if found is None:
        console.print(f"[red]error:[/red] no family {family!r} in {task_set_id}")
        raise typer.Exit(code=1)

    console.print(f"descriptor:  {found.descriptor}")
    console.print(f"mass:        {found.workload_mass}")
    console.print(f"medoid:      {found.medoid_trace_id}")
    console.print(f"proposed by: {found.proposed_by} ({found.review_status})")
    table = Table("trace_id", "split")
    held = set(found.held_out_trace_ids)
    for trace_id in found.trace_ids:
        table.add_row(trace_id, "held_out" if trace_id in held else "fit")
    console.print(table)
    for limitation in found.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="build-sft")
def build_sft_command(
    corpus_id: str,
    trace_ids: list[str] = typer.Option(
        None, "--trace", help="Trace to consider. Repeat to select several; omit for all."
    ),
    output: Path = typer.Option(..., "--output", help="Directory for the three review buckets."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Fireworks review model."),
    samples: int = typer.Option(3, "--samples", min=1, help="Independent LLM reviews per trace."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Build an LLM-reviewed SFT dataset directly from normalized traces."""
    try:
        corpus = ArtifactStore(project / ".bandits").read(corpus_id)
        bundle = build_direct_sft(
            corpus,
            corpus_id,
            trace_ids=trace_ids or (),
            model=model,
            samples=samples,
        )
        envelope = save_direct_sft(bundle, _derived(project))
        paths = write_direct_sft(bundle, output)
    except (FileNotFoundError, ValueError, JudgeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"dataset_id: {envelope.artifact_id}")
    console.print(f"model:      {bundle.review_model}")
    console.print(f"reviewed:   {len(bundle.candidates)}")
    console.print(f"accepted:   {len(bundle.accepted)} -> {paths['accepted']}")
    console.print(f"review:     {len(bundle.review)} -> {paths['review']}")
    console.print(f"rejected:   {len(bundle.rejected)} -> {paths['rejected']}")
    console.print(f"report:     {paths['report']}")


# ---------------------------------------------------------------- next-state


def _archetype(value: str):
    from bandits.verify.nextstate import Archetype

    try:
        return Archetype(value)
    except ValueError:
        console.print(
            f"[red]error:[/red] --archetype must be one of: {', '.join(a.value for a in Archetype)}"
        )
        raise typer.Exit(code=1) from None


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


@app.command(name="judge-turns")
def judge_turns_command(
    corpus_id: str,
    archetype: str = typer.Option(
        ..., "--archetype", help="support, coding, computer-use or generic."
    ),
    task_set_id: str = typer.Option(
        None, "--task-set", help="Restrict to one family of this task set."
    ),
    family_id: str = typer.Option(None, "--family"),
    limit: int = typer.Option(None, "--limit", help="Only the first N traces, for a cheap look."),
    votes: int = typer.Option(1, "--votes", min=1, help="Samples per turn; majority wins."),
    temperature: float = typer.Option(0.0, "--temperature"),
    workers: int = typer.Option(8, "--workers", min=1),
    model: str = typer.Option(None, "--model"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Score every turn of every trace by what happened next."""
    from bandits.verify.judge import fireworks_completion
    from bandits.verify.nextstate import DEFAULT_MODEL, judge_turns, save_turn_judge_run

    kind = _archetype(archetype)
    trace_ids = None
    if family_id:
        if not task_set_id:
            console.print("[red]error:[/red] --family needs --task-set")
            raise typer.Exit(code=1)
        task_set = _load_task_set(task_set_id, project)
        family = task_set.family_by_id().get(family_id)
        if family is None:
            console.print(f"[red]error:[/red] no family {family_id!r} in {task_set_id}")
            raise typer.Exit(code=1)
        trace_ids = family.trace_ids
    traces = _corpus_traces(corpus_id, project, trace_ids)
    if limit:
        traces = traces[:limit]
    if not traces:
        console.print("[red]error:[/red] no traces to judge")
        raise typer.Exit(code=1)

    def progress(done: int, total: int) -> None:
        if done == total or done % 50 == 0:
            console.print(f"  judged {done}/{total} turn(s)")

    with ledger.stage("judge_turns", corpus_id=corpus_id, archetype=kind.value):
        run = judge_turns(
            traces,
            corpus_id,
            kind,
            # The judge is told to think first; at the default budget one reply
            # in fifteen was cut off before the score.
            predict=functools.partial(fireworks_completion, max_tokens=6000),
            model=model or DEFAULT_MODEL,
            votes=votes,
            temperature=temperature,
            workers=workers,
            on_progress=progress,
        )
    envelope = save_turn_judge_run(run, _derived(project))
    console.print(f"turn_judge_run_id: {envelope.artifact_id}")
    console.print(f"archetype:         {kind.value}")
    console.print(f"traces:            {len(run.trace_ids)}")
    for key in ("turns", "scored", "negative", "failed"):
        console.print(f"{key + ':':<19}{envelope.summary[key]}")
    passing = sum(1 for s in run.signals if s.passes)
    console.print(f"passing traces:    {passing} of {len(run.signals)} (no negative turn)")


@app.command(name="propose-verifier")
def propose_verifier_command(
    judge_run_id: str,
    family_id: str = typer.Option(
        None, "--family", help="Label for the family. Defaults to the corpus."
    ),
    rounds: int = typer.Option(2, "--rounds", min=1),
    sample: int = typer.Option(120, "--sample", help="Turns shown to the model per round."),
    min_fired: int = typer.Option(3, "--min-fired"),
    min_precision: float = typer.Option(0.6, "--min-precision"),
    max_iterations: int = typer.Option(20, "--max-iterations"),
    max_llm_calls: int = typer.Option(40, "--max-llm-calls"),
    model: str = typer.Option(None, "--model"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Have the RLM propose checks over a family's turns, re-execute them, keep survivors."""
    from bandits.verify.nextstate import load_turn_judge_run
    from bandits.verify.propose import (
        DEFAULT_MODEL,
        ProposalError,
        build_proposer,
        propose_verifier,
        save_family_verifier,
    )
    from bandits.verify.turns import extract_turns

    store = _derived(project)
    try:
        run = load_turn_judge_run(judge_run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no turn judge run {judge_run_id!r}")
        raise typer.Exit(code=1) from exc
    traces = _corpus_traces(run.corpus_id, project, run.trace_ids)
    turns = [turn for trace in traces for turn in extract_turns(trace)]
    tasks = {trace.trace_id: trace.task for trace in traces}
    chosen_model = model or DEFAULT_MODEL
    # The REPL trace goes to stderr. When the sandbox dies, the code the model
    # ran just before is the only thing that explains it, and it is not
    # recorded anywhere else.
    import logging

    logging.getLogger("dspy.predict.rlm").setLevel(logging.INFO)
    propose = build_proposer(
        archetype=run.archetype,
        model=chosen_model,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
    )
    try:
        with ledger.stage("propose_verifier", judge_run_id=judge_run_id):
            verifier = propose_verifier(
                turns,
                tasks,
                run,
                judge_run_id,
                family_id=family_id or f"corpus:{run.corpus_id}",
                propose=propose,
                model=chosen_model,
                rounds=rounds,
                sample=sample,
                min_fired=min_fired,
                min_precision=min_precision,
            )
    except ProposalError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_family_verifier(verifier, store)
    console.print(f"family_verifier_id: {envelope.artifact_id}")
    console.print(f"proposed: {verifier.proposed}  survived: {envelope.summary['survived']}")
    _show_checks(verifier)


def _show_checks(verifier) -> None:
    table = Table("check", "survived", "fired", "precision", "recall", "decision", "reason")
    for check in verifier.checks:
        stats = check.stats
        table.add_row(
            check.name[:28],
            "[green]yes[/green]" if check.survived else "[red]no[/red]",
            str(stats.fired),
            "n/a" if stats.precision is None else f"{stats.precision:.0%}",
            "n/a" if stats.recall is None else f"{stats.recall:.0%}",
            check.decision,
            check.reason[:48],
        )
    console.print(table)


_CHECK_KEYS = {"a": "accepted", "r": "rejected"}


@app.command(name="review-checks")
def review_checks_command(
    verifier_id: str,
    all_checks: bool = typer.Option(
        False, "--all", help="Also review checks that did not survive."
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Accept or reject each proposed check. One prompt per check; skip and resume freely.

    Revising sends a check back with feedback: pick which shown examples it got
    wrong, say why, and a revised check is proposed, re-scored, and queued as a
    new pending check for a later pass.
    """
    from bandits.verify.nextstate import load_turn_judge_run
    from bandits.verify.propose import (
        ProposalError,
        build_reviser,
        decide_check,
        load_family_verifier,
        revise_check,
        save_family_verifier,
    )
    from bandits.verify.turns import extract_turns

    store = _derived(project)
    try:
        verifier = load_family_verifier(verifier_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no family verifier {verifier_id!r}")
        raise typer.Exit(code=1) from exc
    try:
        judge_run = load_turn_judge_run(verifier.judge_run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no turn judge run {verifier.judge_run_id!r}")
        raise typer.Exit(code=1) from exc
    traces = _corpus_traces(verifier.corpus_id, project)
    all_turns = [t for trace in traces for t in extract_turns(trace)]
    tasks = {trace.trace_id: trace.task for trace in traces}
    turns = {(t.trace_id, t.index): t for t in all_turns}

    queue = [c for c in verifier.pending() if c.survived or all_checks]
    console.print(f"family: {verifier.family_id}  archetype: {verifier.archetype.value}")
    console.print(f"to review: {len(queue)} check(s)\n")
    envelope = None
    for check in queue:
        stats = check.stats
        console.print(f"[bold]{check.name}[/bold] — {check.hypothesis}")
        console.print(Text(check.code.rstrip(), style="dim"))
        precision = "n/a" if stats.precision is None else f"{stats.precision:.0%}"
        recall = "n/a" if stats.recall is None else f"{stats.recall:.0%}"
        console.print(
            f"  fired on {stats.fired} of {stats.turns} turn(s); agrees with the judge's -1 on "
            f"{precision} of the {stats.fired_scored} it fired on that were judged; catches {recall} of "
            f"the judge's {stats.negatives} negatives"
        )
        console.print(
            f"  {'[green]survived[/green]' if check.survived else '[red]did not survive[/red]'}: {check.reason}"
        )
        shown = [(key, "fired") for key in stats.examples[:3]] + [
            (key, "missed") for key in stats.missed[:3]
        ]
        for i, (turn_key, label) in enumerate(shown, start=1):
            trace_id, index = turn_key
            turn = turns.get(turn_key)
            if turn is None:
                continue
            state = (turn.next_state() or "").replace("\n", " ")[:160]
            tag = "fired" if label == "fired" else "[yellow]missed[/yellow]"
            console.print(f"  {i}. [dim]{trace_id}:{index}[/dim] ({tag}) {state}")
        raw = typer.prompt("  [a]ccept/[r]eject/[v]revise/[s]kip/[q]uit", default="s")
        key = raw.strip().lower()[:1]
        if key == "q":
            break
        if key == "v":
            if not shown:
                console.print("  [red]no examples to revise against; skipping[/red]\n")
                continue
            feedback = typer.prompt("  what's wrong with it")
            picks = typer.prompt(
                f"  which numbered example(s) are wrong (1-{len(shown)}, comma-separated; "
                "'fired' means a false positive, 'missed' a false negative)",
                default=",".join(str(i + 1) for i in range(len(shown))),
            )
            try:
                counterexamples = [
                    shown[int(p.strip()) - 1][0]
                    for p in picks.split(",")
                    if p.strip() and 1 <= int(p.strip()) <= len(shown)
                ]
            except ValueError:
                counterexamples = []
            if not counterexamples:
                console.print("  [red]no valid examples selected; skipping[/red]\n")
                continue
            reviser = build_reviser(
                archetype=verifier.archetype,
                name=check.name,
                hypothesis=check.hypothesis,
                code=check.code,
                model=verifier.model,
            )
            try:
                verifier = revise_check(
                    verifier,
                    check.check_id,
                    feedback,
                    counterexamples,
                    all_turns,
                    tasks,
                    judge_run,
                    reviser=reviser,
                )
            except ProposalError as exc:
                console.print(f"  [red]revision failed:[/red] {exc}\n")
                continue
            envelope = save_family_verifier(verifier, store)
            child = verifier.checks[-1]
            ledger.record(
                {
                    "event_type": "check_revised",
                    "verifier_id": envelope.artifact_id,
                    "check": check.name,
                    "check_id": check.check_id,
                    "revised_check_id": child.check_id,
                    "feedback": feedback,
                }
            )
            console.print(
                f"  [green]revised as {child.name!r}[/green] "
                f"({'survived' if child.survived else 'did not survive'}: {child.reason}); "
                "queued for a later review pass\n"
            )
            continue
        if key not in _CHECK_KEYS:
            continue
        note = typer.prompt("  why (optional)", default="", show_default=False)
        verifier = decide_check(verifier, check.check_id, _CHECK_KEYS[key], note)
        envelope = save_family_verifier(verifier, store)
        ledger.record(
            {
                "event_type": "check_review",
                "verifier_id": envelope.artifact_id,
                "check": check.name,
                "check_id": check.check_id,
                "decision": _CHECK_KEYS[key],
                "note": note,
            }
        )
        console.print()

    console.print(f"\nfamily_verifier_id: {envelope.artifact_id if envelope else verifier_id}")
    console.print(f"accepted: {len(verifier.accepted())}  pending: {len(verifier.pending())}")


@app.command(name="score-traces")
def score_traces_command(
    verifier_id: str,
    include_judge: bool = typer.Option(
        True, "--judge/--no-judge", help="Also flag turns the judge scored -1."
    ),
    survivors: bool = typer.Option(
        False,
        "--survivors",
        help="Apply every check that cleared the automatic bar and was not "
        "human-rejected or revised, not only accepted ones.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Score every trace of the family with the accepted checks and the judge."""
    from bandits.verify.nextstate import load_turn_judge_run
    from bandits.verify.propose import apply_verifier, load_family_verifier, save_verifier_scores
    from bandits.verify.turns import extract_turns

    store = _derived(project)
    try:
        verifier = load_family_verifier(verifier_id, store)
        run = load_turn_judge_run(verifier.judge_run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    traces = _corpus_traces(verifier.corpus_id, project, run.trace_ids)
    turns = [turn for trace in traces for turn in extract_turns(trace)]
    tasks = {trace.trace_id: trace.task for trace in traces}
    checks = (
        tuple(
            c
            for c in verifier.checks
            if c.survived and c.decision not in ("rejected", "revised")
        )
        if survivors
        else None
    )
    scores = apply_verifier(
        verifier,
        turns,
        tasks,
        run,
        verifier_id=verifier_id,
        judge_run_id=verifier.judge_run_id,
        include_judge=include_judge,
        checks=checks,
        via_survivors=survivors,
    )
    envelope = save_verifier_scores(scores, store)
    names_by_id = {c.check_id: c.name for c in verifier.checks}
    applied_display = ", ".join(names_by_id.get(cid, cid) for cid in scores.checks_applied)
    console.print(f"verifier_scores_id: {envelope.artifact_id}")
    console.print(
        f"checks applied:     {applied_display or '(none)'}{' + judge' if include_judge else ''}"
    )
    console.print(f"passing:            {envelope.summary['passing']} of {len(scores.scores)}")
    if envelope.summary["unresolved_traces"]:
        console.print(
            f"[yellow]unresolved:[/yellow]         {envelope.summary['unresolved_traces']} "
            f"trace(s), {envelope.summary['unresolved_turns']} turn(s) with no confirmed-clean "
            "signal from any source — not a pass, not a flag"
        )
    table = Table("trace", "turns", "observed", "flagged", "unresolved", "score", "passes")
    for item in sorted(scores.scores, key=lambda s: (s.score is None, -(s.score or 0)))[:40]:
        table.add_row(
            item.trace_id[:20],
            str(item.turns),
            str(item.observed),
            str(len(item.flagged)),
            str(len(item.unresolved)),
            "n/a" if item.score is None else f"{item.score:.2f}",
            "yes" if item.passes else "no",
        )
    console.print(table)


@app.command(name="export-nextstate")
def export_nextstate_command(
    scores_id: str,
    output: Path = typer.Option(..., "--output"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Write a family-verifier's scored traces as labeled SFT rows: one full
    trajectory per row, positive or negative, with the flagged turns kept as
    metadata rather than used to cut the trajectory apart."""
    from bandits.export.nextstate_sft import (
        build_nextstate_sft_export,
        save_nextstate_sft,
        write_nextstate_sft,
    )
    from bandits.verify.propose import load_family_verifier, load_verifier_scores

    store = _derived(project)
    try:
        scores = load_verifier_scores(scores_id, store)
        verifier = load_family_verifier(scores.verifier_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    traces = _corpus_traces(verifier.corpus_id, project)
    bundle = build_nextstate_sft_export(traces, verifier, scores.verifier_id, scores, scores_id)
    envelope = save_nextstate_sft(bundle, store)
    rows_path, unresolved_path = write_nextstate_sft(bundle, output)
    console.print(f"export_id:  {envelope.artifact_id}")
    console.print(f"positive:   {bundle.positive}")
    console.print(f"negative:   {bundle.negative}")
    console.print(f"unresolved: {len(bundle.unresolved)}")
    reviewed_label = "yes" if bundle.all_checks_reviewed else "no (--survivors or unreviewed checks)"
    console.print(f"reviewed:   {reviewed_label}")
    console.print(f"output:     {rows_path}")
    console.print(f"quarantine: {unresolved_path}")


@app.command(name="decision-dataset")
def decision_dataset_command(
    judge_run_id: str,
    task_set_id: str = typer.Option(
        None,
        "--task-set",
        help="Split examples along this task set's own within-family fit/held-out "
        "membership, so no family is split across the boundary. The task set must "
        "have been built from the same corpus as the judge run, and every judged "
        "trace must resolve to one of its families or it is quarantined. Omit to "
        "put every example in within_family_fit.",
    ),
    minimum_valid_votes: int = typer.Option(
        1, "--minimum-valid-votes", min=1, help="Quarantine a turn with fewer successful votes."
    ),
    output: Path = typer.Option(None, "--output", help="Write fit+held-out and quarantine JSONL."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Compile a turn-judge run's verdicts into a generic decision dataset."""
    from bandits.decide.dataset import (
        build_decision_dataset_from_corpus,
        save_decision_dataset,
        write_decision_dataset,
    )
    from bandits.verify.nextstate import load_turn_judge_run

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
    console.print(f"decision_dataset_id:     {envelope.artifact_id}")
    console.print(f"examples:                {dataset.counts.examples}")
    console.print(f"train:                   {dataset.counts.train}")
    console.print(f"dev:                     {dataset.counts.dev}")
    console.print(f"quarantined:             {dataset.counts.quarantined}")
    if not task_set_id:
        console.print(
            "[yellow]no --task-set given:[/yellow] every example was put in train; "
            "there is no dev split to certify against"
        )
    if output is not None:
        rows_path, quarantine_path = write_decision_dataset(dataset, output)
        console.print(f"output:              {rows_path}")
        console.print(f"quarantine:          {quarantine_path}")


@app.command(name="decision-import")
def decision_import_command(
    path: Path = typer.Argument(..., help="JSONL file: one labeled decision per line."),
    source: str = typer.Option(None, "--source", help="Dataset-level source name/URL, used when a row omits its own."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
    output: Path = typer.Option(None, "--output", help="Write examples and quarantine JSONL."),
) -> None:
    """Import a JSONL file of user-labeled decisions into a DecisionDataset."""
    from bandits.decide.dataset import write_decision_dataset
    from bandits.decide.importer import import_jsonl, save_imported_dataset

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


@app.command(name="decision-score")
def decision_score_command(
    dataset_id: str,
    model: str = typer.Option(..., "--model", help="Hugging Face model id, e.g. Qwen/Qwen3.5-4B."),
    revision: str = typer.Option(..., "--revision", help="Pinned model revision (commit SHA)."),
    split: str = typer.Option("dev", "--split", help="Which split to score: train/dev/calibration/test."),
    two_order: bool = typer.Option(
        False, "--two-order", help="Average two option orders per example (two forward passes)."
    ),
    device: str = typer.Option("cuda", "--device"),
    dtype: str = typer.Option("bfloat16", "--dtype"),
    max_prompt_tokens: int = typer.Option(8_000, "--max-prompt-tokens"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Score a decision dataset's split with an untrained (frozen) model: one
    forward pass per example, softmax over the option-letter logits."""
    from bandits.decide.dataset import load_decision_dataset
    from bandits.decide.hf_predictor import HFPredictor
    from bandits.decide.scorer import save_scorer_run, score_dataset

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

    predictor = HFPredictor(model, revision=revision, device=device, dtype=dtype)
    mode = "two_order_average" if two_order else "single_order"
    run = score_dataset(
        predictor,
        examples,
        mode=mode,
        max_prompt_tokens=max_prompt_tokens,
        dataset_id=dataset_id,
        split=split,
    )
    envelope = save_scorer_run(run, store)
    console.print(f"scorer_run_id: {envelope.artifact_id}")
    console.print(f"scored:        {len(run.results)}")
    console.print(f"rejected:      {len(run.rejections)}")


if __name__ == "__main__":
    app()


# --- RLM task-family mining --------------------------------------------------
#
# Experimental, and kept beside the embedding miner rather than replacing it.
# The two answer the same question by different means, and the embedding path
# stays the baseline until this one demonstrates better semantic coherence,
# stability, and downstream verifier transfer. Nothing here feeds `mine`.


def _rlm_corpus(analysis_id: str, project: Path, view: str):
    """The read-only user-message view of the corpus behind an analysis.

    The miner is handed this and never the corpus, so there is no path from a
    mining command to an assistant message, a tool call, or an outcome.
    """
    store = _derived(project)
    try:
        analysis = load_analysis(analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no analysis {analysis_id!r}")
        raise typer.Exit(code=1) from exc
    try:
        corpus = ArtifactStore(project / ".bandits").read(analysis.corpus_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no corpus {analysis.corpus_id!r} behind this analysis")
        raise typer.Exit(code=1) from exc
    return (
        analysis,
        ReadOnlyCorpus(corpus, view=TraceView(view), control_markers=corpus.control_markers),
        store,
    )


def _report_unresolved(ambiguous: int, uncovered: int, unreadable: int) -> None:
    """What the taxonomy could not reach. Never suppressed, never rolled into a total."""
    for count, label, note in (
        (ambiguous, "ambiguous", "matched several contracts and were left unplaced"),
        (uncovered, "uncovered", "matched no contract at all"),
        (unreadable, "unreadable", "recorded no user messages to read"),
    ):
        if count:
            console.print(f"[yellow]{label}:[/yellow]   {count} trace(s) {note}")


@app.command(name="mine-rlm")
def mine_rlm_command(
    analysis_id: str,
    view: str = typer.Option(
        TraceView.USER_MESSAGES.value,
        "--view",
        help=(
            "user-messages (Path U), full-trajectory (Path F: adds assistant turns and "
            "tool activity, rewards withheld), or first-user-message."
        ),
    ),
    chunk_size: int = typer.Option(RLM_CHUNK_SIZE, "--chunk-size"),
    passes: int = typer.Option(
        RLM_PASSES,
        "--passes",
        help="Complete corpus passes before pausing for review. Each reads every trace once.",
    ),
    max_iterations: int = typer.Option(
        200, "--max-iterations", help="Emergency guard on chunk count, not the stopping rule."
    ),
    max_llm_calls: int = typer.Option(400, "--max-llm-calls"),
    max_tokens: int = typer.Option(
        RLM_MAX_TOKENS,
        "--max-tokens",
        help="Maximum completion tokens for each model call.",
    ),
    max_seconds: float = typer.Option(3600.0, "--max-seconds"),
    max_usd: float = typer.Option(None, "--max-usd", help="Monetary ceiling. Unset means none."),
    seed: int = typer.Option(RLM_SEED, "--seed"),
    model: str = typer.Option(RLM_MODEL, "--model"),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Continue a session that stopped, from the chunk it reached.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Discover task families from raw user requests with an iterative RLM loop."""
    try:
        trace_view = TraceView(view)
    except ValueError as exc:
        console.print(f"[red]error:[/red] unknown view {view!r}")
        raise typer.Exit(code=1) from exc

    analysis, corpus, store = _rlm_corpus(analysis_id, project, trace_view.value)
    budget = Budget(
        passes=passes,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_seconds=max_seconds,
        max_usd=max_usd,
    )
    try:
        predict = build_rlm_predictor(model=model, view=trace_view, max_tokens=max_tokens)
    except MiningError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    session_store = SessionStore(project / ".bandits")
    resumed_state = None
    if resume:
        try:
            resumed_state = session_store.read(resume)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no session {resume!r}")
            raise typer.Exit(code=1) from exc
        if resumed_state.analysis_id != analysis_id:
            # Resuming onto a different corpus would carry a taxonomy built from
            # one set of requests onto another and call the result one run.
            console.print(
                f"[red]error:[/red] session {resume!r} was mining "
                f"{resumed_state.analysis_id!r}, not {analysis_id!r}"
            )
            raise typer.Exit(code=1)
        if resumed_state.view is not trace_view:
            console.print(
                f"[red]error:[/red] session {resume!r} used the "
                f"{resumed_state.view.value} view; the two arms are different experiments"
            )
            raise typer.Exit(code=1)
        # The same seed, or the reshuffle of a later pass would differ from what
        # the interrupted run would have done.
        seed = resumed_state.seed
        console.print(
            f"resuming:    {resume} at pass {resumed_state.pass_index + 1}, "
            f"{resumed_state.traces_seen_this_pass}/{resumed_state.traces_total} read, "
            f"{len(resumed_state.contracts)} contract(s) restored"
        )

    recorder = SessionRecorder(
        session_store,
        # A resume continues writing to the same session, so one interrupted run
        # stays one row in the listing rather than fragmenting across restarts.
        session_id=resume or new_session_id(analysis_id, trace_view, seed),
        analysis_id=analysis_id,
        view=trace_view,
        model=model,
        resumed_from=resume,
    )
    console.print(f"session:     {recorder.session_id}")
    console.print(f"[dim]watch: bandits rlm-session {recorder.session_id}[/dim]\n")

    with ledger.stage("rlm_mining_run", analysis_id=analysis_id, view=trace_view.value, seed=seed):
        try:
            draft = mine_taxonomy(
                corpus,
                analysis_id,
                predict=predict,
                analysis=analysis,
                model=model,
                chunk_size=chunk_size,
                seed=seed,
                budget=budget,
                session=recorder,
                resume=resumed_state,
                on_chunk=lambda c: console.print(
                    f"[dim]pass {c.pass_index + 1} chunk {c.index}: "
                    f"{len(c.trace_ids)} trace(s), {len(c.operations)} operation(s)"
                    f"{' [failed: ' + c.error[:40] + ']' if c.status == 'error' else ''}[/dim]"
                ),
            )
        except MiningError as exc:
            recorder.fail(str(exc))
            console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            # The session file is the only record of a run that died partway,
            # so it must say so rather than being left reading as still running.
            recorder.fail(str(exc))
            raise
        envelope = save_clustering_run(draft, store)
        recorder.finish(
            status="awaiting_review" if draft.complete else "incomplete",
            stop_reason=draft.stop_reason.value,
            # run_id, not draft_id: the 6a822c2 rename ("taxonomy draft" ->
            # "clustering run") missed this call site, so every mine-rlm run
            # crashed here after mining actually completed.
            run_id=envelope.artifact_id,
            completed_passes=draft.completed_passes,
        )
        ledger.record(
            {
                "event_type": "stage_complete",
                "stage_name": "rlm_mining_run",
                "input_artifact_id": analysis_id,
                "output_artifact_id": envelope.artifact_id,
                "contracts": len(draft.contracts),
                "stop_reason": draft.stop_reason.value,
            }
        )

    console.print(f"\ndraft_id:    {envelope.artifact_id}")
    console.print(f"view:        {draft.view.value} (seed {draft.seed})")
    console.print(f"contracts:   {len(draft.contracts)}")
    console.print(f"chunks:      {len(draft.chunks)}")
    console.print(f"passes:      {draft.completed_passes}/{draft.requested_passes} complete")
    # The distinction the whole artifact turns on: finishing the schedule is not
    # convergence, and a run that hit a guard did not even finish the schedule.
    if draft.complete:
        console.print(
            "[green]awaiting review[/green]  every requested pass read every trace; "
            "this is a checkpoint, not a converged taxonomy"
        )
    else:
        console.print(
            f"[yellow]incomplete:[/yellow]  stopped on {draft.stop_reason.value} "
            "before finishing its passes"
        )
    console.print("")
    console.print(taxonomy_overview(draft.contracts, members=_run_members(draft)))
    # What the second look changed. The question a reviewer has at a pause, and
    # one no total over the whole run answers.
    console.print("")
    print_pass_history(draft, console)
    console.print(f"\n[dim]review the families: bandits rlm-families {envelope.artifact_id}[/dim]")
    _report_unresolved(
        len(draft.ambiguous_trace_ids),
        len(draft.uncovered_trace_ids),
        len(draft.unreadable_trace_ids),
    )
    for limitation in draft.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="audit-rlm")
def audit_rlm_command(
    run_id: str,
    model: str = typer.Option(RLM_AUDIT_MODEL, "--model"),
    max_tokens: int = typer.Option(
        RLM_AUDIT_MAX_TOKENS,
        "--max-tokens",
        help="Maximum completion tokens for each model call.",
    ),
    max_llm_calls: int = typer.Option(400, "--max-llm-calls"),
    max_seconds: float = typer.Option(3600.0, "--max-seconds"),
    max_usd: float = typer.Option(None, "--max-usd", help="Monetary ceiling. Unset means none."),
    max_contracts: int = typer.Option(
        None, "--max-contracts", help="Audit at most this many contracts this session."
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Continue an audit session that stopped, skipping contracts it already covered.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Challenge every contract in a clustering run with a fresh, adversarial context.

    Advisory only. This saves findings and changes nothing: no placement moves,
    and materializing the run neither requires this nor consults it.

    Checkpointed after every contract to .bandits/sessions/audits/<session_id>,
    so Ctrl+C or a crash loses at most the one contract in flight; resume with
    --resume <session_id>. Every model call is also recorded to the path in the
    BANDITS_LEDGER env var, if set, one row per call with cost and duration.
    """
    store = _derived(project)
    try:
        run = load_clustering_run(run_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no clustering run {run_id!r}")
        raise typer.Exit(code=1) from exc

    _, corpus, _ = _rlm_corpus(run.analysis_id, project, run.view.value)
    try:
        predict = build_taxonomy_audit_predictor(model=model, view=run.view, max_tokens=max_tokens)
    except ClusteringAuditError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    budget = AuditBudget(
        max_llm_calls=max_llm_calls,
        max_seconds=max_seconds,
        max_usd=max_usd,
        max_contracts=max_contracts,
    )
    digest = rlm_audit_prompt_digest(model)

    session_store = AuditSessionStore(project / ".bandits")
    resumed_state = None
    if resume:
        try:
            resumed_state = session_store.read(resume)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no audit session {resume!r}")
            raise typer.Exit(code=1) from exc
        if resumed_state.run_id != run_id:
            console.print(
                f"[red]error:[/red] session {resume!r} was auditing "
                f"{resumed_state.run_id!r}, not {run_id!r}"
            )
            raise typer.Exit(code=1)
        if resumed_state.model != model:
            console.print(
                f"[red]error:[/red] session {resume!r} used model "
                f"{resumed_state.model!r}, not {model!r}"
            )
            raise typer.Exit(code=1)
        if resumed_state.prompt_digest != digest:
            console.print(
                f"[red]error:[/red] session {resume!r} audited under a different prompt; "
                "its existing findings do not answer the same question a fresh contract would be asked"
            )
            raise typer.Exit(code=1)
        console.print(
            f"resuming:    {resume} at {len(resumed_state.completed_contract_ids)}/"
            f"{len(resumed_state.contract_order)} contracts, "
            f"${resumed_state.cost_usd:.4f} spent so far"
        )

    recorder = AuditSessionRecorder(
        session_store,
        session_id=resume or new_audit_session_id(run_id),
        run_id=run_id,
        model=model,
        view=run.view,
        prompt_digest=digest,
        resumed_from=resume,
        seed_state=resumed_state,
    )
    console.print(f"session:     {recorder.session_id}")
    console.print(f"[dim]watch: bandits rlm-session {recorder.session_id} --watch[/dim]\n")

    interrupted = {"flag": False}

    def _on_interrupt_signal(_signum: int, _frame: object) -> None:
        # Set-and-return rather than raising: the loop in audit_clustering
        # polls this between contracts and stops cleanly, saving the session
        # as interrupted with everything checkpointed so far intact. Raising
        # here instead could land mid-write to the session file. Handled for
        # both SIGINT (Ctrl+C) and SIGTERM (`kill`, `timeout`, a supervisor
        # stopping the process) — a session killed either way must end up
        # marked interrupted, not stuck reading "running" forever because
        # nothing ever told it otherwise.
        interrupted["flag"] = True
        console.print(
            "\n[yellow]stopping after the contract in flight; "
            "progress so far is saved[/yellow]"
        )

    import signal

    previous_sigint = signal.signal(signal.SIGINT, _on_interrupt_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _on_interrupt_signal)
    try:
        with ledger.stage("rlm_taxonomy_audit_run", run_id=run_id, model=model):
            try:
                audit = audit_clustering(
                    run,
                    run_id,
                    corpus,
                    predict=predict,
                    model=model,
                    budget=budget,
                    session=recorder,
                    resume=resumed_state,
                    should_stop=lambda: interrupted["flag"],
                )
            except ClusteringAuditError as exc:
                recorder.fail(str(exc))
                console.print(f"[red]error:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            except Exception as exc:
                recorder.fail(str(exc))
                raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    audit_envelope = save_rlm_audit(audit, store)
    ledger.record(
        {
            "event_type": "stage_complete",
            "stage_name": "rlm_taxonomy_audit_run",
            "input_artifact_id": run_id,
            "output_artifact_id": audit_envelope.artifact_id,
            "findings": len(audit.findings),
            "status": audit.status,
            "stop_reason": audit.stop_reason,
        }
    )
    console.print(f"audit_id:    {audit_envelope.artifact_id} ({audit.model})")
    if audit.status != "complete":
        console.print(
            f"[yellow]incomplete:[/yellow] stopped on {audit.stop_reason} — "
            f"resume with: bandits audit-rlm {run_id} --resume {recorder.session_id}"
        )

    for finding in audit.findings:
        colour = "yellow" if finding.demands_action else "dim"
        topical = " [topical grouping]" if finding.topical_only else ""
        console.print(
            f"  [{colour}]{finding.recommendation}[/{colour}] {finding.contract_id}"
            f"{topical}: {finding.rationale}"
        )
        if finding.least_compatible_pair:
            left, right = finding.least_compatible_pair
            console.print(f"    [dim]least compatible: {left} vs {right}[/dim]")
        if finding.strongest_outsider_trace_id:
            console.print(
                f"    [dim]strongest outsider: {finding.strongest_outsider_trace_id}[/dim]"
            )
        if finding.merge_with_contract_id:
            console.print(f"    [dim]merge with: {finding.merge_with_contract_id}[/dim]")

    unresolved = audit.unresolved()
    if unresolved:
        console.print(
            f"\n[yellow]{len(unresolved)} finding(s) recommend revising, splitting, or merging a "
            "contract and are unresolved[/yellow]"
        )
        console.print("[dim]advisory: nothing here changed the run or its placements[/dim]")


@app.command(name="materialize-rlm-taskset")
def materialize_rlm_taskset_command(
    run_id: str,
    held_out: float = typer.Option(DEFAULT_HELD_OUT, "--held-out"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Turn the traces a clustering run placed into a TaskSet, with honest provenance."""
    store = _derived(project)
    try:
        run = load_clustering_run(run_id, store)
        analysis = load_analysis(run.analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        task_set = materialize_task_set(run, analysis, held_out=held_out, run_id=run_id)
    except MaterializationError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_task_set(task_set, store)
    _report(task_set, envelope.artifact_id)
    # Said plainly, because a TaskSet from this path carries no measured
    # geometry and every reader downstream is used to one that does.
    console.print(
        "[dim]families here were proposed by a model reading user requests; no "
        "embedding distance was computed, so none carries a coherence figure, and "
        "membership is the miner's own placement rather than an independent pass[/dim]"
    )


def _find_session(
    mining_store: SessionStore, audit_store: AuditSessionStore, session_id: str
):
    """Locate a session by id in whichever store actually has it.

    Session ids are self-describing (``rlm-audit-...`` vs ``rlm-...``) but
    this checks both stores rather than trusting the prefix, so a renamed or
    hand-typed id still resolves instead of failing on a naming assumption.
    """
    if audit_store.exists(session_id):
        return "audit", audit_store.read(session_id)
    if mining_store.exists(session_id):
        return "mining", mining_store.read(session_id)
    return None, None


@app.command(name="rlm-session")
def rlm_session_command(
    session_id: str = typer.Argument(None, help="Omit to list every session."),
    events: int = typer.Option(0, "--events", help="Show the last N progress events."),
    watch: bool = typer.Option(
        False, "--watch", help="Redraw as the run progresses. Exits when it finishes."
    ),
    interval: float = typer.Option(1.0, "--interval", help="Seconds between redraws."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Inspect a mining or audit session, including one still running.

    Lists and watches both kinds, labeled by kind: mine-rlm's discovery
    sessions and audit-rlm's per-contract audit sessions live in separate
    stores on disk but are one surface here.
    """
    mining_store = SessionStore(project / ".bandits")
    audit_store = AuditSessionStore(project / ".bandits")

    if watch:
        if session_id is None:
            candidates = [("mining", s) for s in mining_store.list()] + [
                ("audit", s) for s in audit_store.list()
            ]
            if not candidates:
                console.print("[dim]no sessions[/dim]")
                return
            kind, latest = max(candidates, key=lambda pair: pair[1].updated_at)
            session_id = latest.session_id
            _watch_session(kind, mining_store if kind == "mining" else audit_store, session_id, interval)
            return
        kind, _ = _find_session(mining_store, audit_store, session_id)
        if kind is None:
            console.print(f"[red]error:[/red] no session {session_id!r}")
            raise typer.Exit(code=1)
        _watch_session(kind, mining_store if kind == "mining" else audit_store, session_id, interval)
        return

    if session_id is None:
        sessions = [("mining", s) for s in mining_store.list()] + [
            ("audit", s) for s in audit_store.list()
        ]
        if not sessions:
            console.print("[dim]no sessions[/dim]")
            return
        sessions.sort(key=lambda pair: pair[1].updated_at, reverse=True)
        table = Table("kind", "session", "status", "progress", "updated")
        for kind, state in sessions:
            colour = {
                "running": "cyan",
                "awaiting_review": "green",
                "failed": "red",
                "interrupted": "yellow",
            }.get(state.status, "yellow")
            table.add_row(
                kind,
                state.session_id,
                f"[{colour}]{state.status}[/{colour}]",
                state.progress,
                state.updated_at[:19],
            )
        console.print(table)
        return

    kind, state = _find_session(mining_store, audit_store, session_id)
    if kind is None:
        console.print(f"[red]error:[/red] no session {session_id!r}")
        raise typer.Exit(code=1)

    store = mining_store if kind == "mining" else audit_store

    if kind == "audit":
        console.print(f"session:     {state.session_id}")
        console.print("kind:        audit")
        console.print(f"status:      {state.status}")
        console.print(f"run:         {state.run_id}")
        console.print(f"view:        {state.view.value}")
        console.print(f"progress:    {state.progress}")
        if state.resumed_from:
            console.print(f"resumed from: {state.resumed_from}")
        for finding in state.findings:
            colour = "yellow" if finding.demands_action else "dim"
            console.print(
                f"  [{colour}]{finding.recommendation}[/{colour}] {finding.contract_id}"
            )
        if state.last_error:
            console.print(f"[red]last error:[/red] {state.last_error}")
    else:
        console.print(f"session:     {state.session_id}")
        console.print("kind:        mining")
        console.print(f"status:      {state.status}")
        console.print(f"view:        {state.view.value} (seed {state.seed})")
        console.print(f"progress:    {state.progress}")
        console.print(
            f"passes:      {state.completed_passes}/{state.requested_passes} complete, "
            f"pass {state.pass_index + 1} in flight"
        )
        console.print(f"assigned:    {len(state.assignments)}")
        _report_unresolved(len(state.ambiguous_trace_ids), len(state.uncovered_trace_ids), 0)
        for contract in state.contracts:
            console.print(f"  {contract.contract_id}  {contract.name}")
        if state.last_error:
            console.print(f"[red]last error:[/red] {state.last_error}")

    if events:
        console.print("")
        for event in store.read_events(session_id, limit=events):
            name = event.get("event", "?")
            detail = " ".join(
                f"{k}={v}"
                for k, v in event.items()
                if k not in ("at", "event") and v not in ("", [], None)
            )
            console.print(f"[dim]{event.get('at', '')[:19]}  {name}  {detail}[/dim]")


def _watch_session(kind: str, store, session_id: str, interval: float) -> None:
    """Redraw one session until it stops running.

    Reads the session file rather than hooking into the run, so this works on a
    run started in another terminal, in CI, or by someone else — and cannot
    slow the run down or lose it if the viewer dies.
    """
    from rich.live import Live

    try:
        state = store.read(session_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no session {session_id!r}")
        raise typer.Exit(code=1) from exc

    panel = audit_live_panel if kind == "audit" else live_panel
    terminal_statuses = {"awaiting_review", "failed", "interrupted", "incomplete"}

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            live.update(panel(state, recent=store.read_events(session_id, limit=6)))
            if state.status in terminal_statuses:
                break
            time.sleep(interval)
            try:
                state = store.read(session_id)
            except (FileNotFoundError, ValueError):
                # A read landing mid-write is expected: the writer replaces the
                # file atomically, so the next poll gets a whole one.
                continue

    if state.status == "awaiting_review":
        if kind == "audit":
            console.print(
                "\n[green]audit complete.[/green] "
                "[dim]bandits rlm-families <draft_id> --audit <audit_id>[/dim]"
            )
        else:
            console.print(
                "\n[green]paused for review.[/green] "
                "[dim]families: bandits rlm-families <draft_id>[/dim]"
            )
    elif state.status in ("interrupted", "incomplete"):
        if kind == "audit":
            resume_command = f"bandits audit-rlm {state.run_id} --resume {session_id}"
        else:
            resume_command = f"bandits mine-rlm {state.analysis_id} --resume {session_id}"
        console.print(f"\n[yellow]{state.status}.[/yellow] [dim]resume with: {resume_command}[/dim]")


@app.command(name="rlm-families")
def rlm_families_command(
    draft_id: str,
    family: str = typer.Option(None, "--family", help="Show one family's card in full."),
    audit_id: str = typer.Option(None, "--audit", help="Overlay an audit's verdicts."),
    overview: bool = typer.Option(
        False, "--overview", help="Table only, without the per-family cards."
    ),
    examples: int = typer.Option(4, "--examples", help="Member requests to show per family."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Read a mined taxonomy as reviewable family cards, not id lists."""
    store = _derived(project)
    try:
        draft = load_clustering_run(draft_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no draft {draft_id!r}")
        raise typer.Exit(code=1) from exc

    audit = None
    if audit_id:
        try:
            audit = load_rlm_audit(audit_id, store)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no audit {audit_id!r}")
            raise typer.Exit(code=1) from exc

    # Best effort: the cards are far more useful with the requests beside the
    # ids, but a draft whose corpus has moved must still be readable.
    corpus = None
    try:
        _, corpus, _ = _rlm_corpus(draft.analysis_id, project, draft.view.value)
    except typer.Exit:
        console.print("[dim]corpus unavailable; showing ids without requests[/dim]")

    members = _run_members(draft)
    contracts = draft.contracts
    if family is not None:
        contracts = tuple(c for c in contracts if c.contract_id == family)
        if not contracts:
            console.print(f"[red]error:[/red] no family {family!r} in this draft")
            raise typer.Exit(code=1)

    console.print(taxonomy_overview(contracts, members=members, audit=audit))
    if not overview:
        for contract in contracts:
            console.print("")
            console.print(
                family_card(
                    contract,
                    members=members.get(contract.contract_id, ()),
                    corpus=corpus,
                    audit=audit,
                    max_examples=examples,
                )
            )

    console.print("")
    print_pass_history(draft, console)
    _report_unresolved(
        len(draft.ambiguous_trace_ids),
        len(draft.uncovered_trace_ids),
        len(draft.unreadable_trace_ids),
    )
    for limitation in draft.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")
