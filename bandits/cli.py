"""Command-line interface: ingest a trace export, then inspect what's stored."""

from __future__ import annotations

import functools
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from bandits import ledger
from bandits.analyze import (
    DEFAULT_BUDGET,
    DEFAULT_DUPLICATE_SIMILARITY,
    DEFAULT_HELD_OUT,
    DEFAULT_NEIGHBORS,
    analyze_corpus,
    load_analysis,
    load_task_set,
    merge_families,
    mine_task_set,
    save_analysis,
    save_task_set,
    split_family,
)
from bandits.analyze.audit import (
    DEFAULT_MODEL as AUDIT_MODEL,
)
from bandits.analyze.audit import (
    AuditError,
    audit_task_set,
    build_predictor,
    load_audit_run,
    save_audit_run,
)
from bandits.analyze.embed import (
    DEFAULT_MODEL as EMBEDDING_MODEL,
)
from bandits.analyze.embed import (
    DEFAULT_SIMILARITY as EMBEDDING_SIMILARITY,
)
from bandits.analyze.embed import (
    EmbeddingCache,
    EmbeddingError,
    build_cache,
    descriptors,
    embedding_distance,
    load_cache,
    requests,
    save_cache,
)
from bandits.analyze.rlm_assign import (
    DEFAULT_MODEL as RLM_ASSIGN_MODEL,
)
from bandits.analyze.rlm_assign import (
    AssignmentError,
    assign_traces,
    load_assignment_run,
    save_assignment_run,
)
from bandits.analyze.rlm_assign import (
    build_predictor as build_assignment_predictor,
)
from bandits.analyze.rlm_audit import (
    DEFAULT_MODEL as RLM_AUDIT_MODEL,
)
from bandits.analyze.rlm_audit import (
    FreezeRefused,
    TaxonomyAuditError,
    _draft_members,
    audit_taxonomy,
    freeze_taxonomy,
    load_taxonomy,
    save_taxonomy,
)
from bandits.analyze.rlm_audit import (
    build_predictor as build_taxonomy_audit_predictor,
)
from bandits.analyze.rlm_audit import (
    load_audit as load_rlm_audit,
)
from bandits.analyze.rlm_audit import (
    save_audit as save_rlm_audit,
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
    load_draft,
    mine_taxonomy,
    save_draft,
)
from bandits.analyze.rlm_mine import (
    build_predictor as build_rlm_predictor,
)
from bandits.analyze.rlm_models import (
    DEFAULT_PASSES as RLM_PASSES,
)
from bandits.analyze.rlm_models import (
    AssignmentStatus,
    Budget,
    TraceView,
)
from bandits.analyze.rlm_session import SessionRecorder, SessionStore, new_session_id
from bandits.analyze.rlm_stability import compare_runs, save_stability_report
from bandits.analyze.rlm_taskset import MaterializationError, materialize_task_set
from bandits.analyze.rlm_view import (
    family_card,
    live_panel,
    print_pass_history,
    taxonomy_overview,
)
from bandits.export import (
    CompositionReport,
    Partition,
    SamplingCaps,
    build_direct_sft,
    build_eval_export,
    build_sft_export,
    save_direct_sft,
    save_export,
    write_direct_sft,
    write_jsonl,
)
from bandits.ingest import CANONICAL_SOURCES, UnknownSourceError, load_corpus
from bandits.labels import (
    LabelSet,
    Verdict,
    load_label_set,
    make_label,
    save_label_set,
)
from bandits.redact import DEFAULT_RULESET, ruleset_by_name
from bandits.store import ArtifactStore, DerivedStore
from bandits.verify import (
    answer_question,
    apply_decision,
    build_check_summary,
    draft_verifiers,
    find_check,
    load_interview,
    load_reviewed_verifier,
    load_verifier_draft,
    next_check,
    next_question,
    prior_decisions,
    review_verifier,
    run_draft,
    save_draft_run,
    save_interview,
    save_reviewed_verifier,
    save_verifier_draft,
    start_interview,
    start_review,
)
from bandits.verify.interpret import (
    DEFAULT_MODEL as INTERPRETER_MODEL,
)
from bandits.verify.interpret import (
    InterpretationFailure,
    interpret_reply,
    parse_expected,
)
from bandits.verify.judge import (
    DEFAULT_MODEL,
    JudgeError,
    Rubric,
    judge_traces,
    save_judge_run,
)
from bandits.verify.models import (
    CheckOperator,
    CheckReview,
    CheckSpec,
    Interpretation,
    InterviewDecision,
)
from bandits.verify.validate import (
    load_validation,
    probe_gameability,
    save_validation,
    validate_draft,
)

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


def _embedding_cache(analysis, store: DerivedStore, model: str) -> tuple[EmbeddingCache, str]:
    """Vectors for every descriptor this analysis will be grouped on.

    Reuses a saved cache when one covers the corpus, and embeds only what it is
    missing, so re-mining the same analysis at a different threshold costs
    nothing. Vectors from two models are never mixed — a cache pinned to another
    model is passed over rather than extended.
    """
    # Both halves of what mining compares. Values remain visible in descriptors
    # and requests, so task-defining identifiers reach both distance backends. Building
    # only the first leaves every duplicate comparison reading maximally far.
    wanted = descriptors(analysis) + requests(analysis)
    existing: EmbeddingCache | None = None
    reused_id = ""
    for envelope in store.list(kind="embeddings"):
        if envelope.parent_artifact_id != analysis.corpus_id:
            continue
        candidate = load_cache(envelope.artifact_id, store)
        if candidate.model == model:
            existing, reused_id = candidate, envelope.artifact_id
            break

    cache = build_cache(wanted, model=model, existing=existing)
    if existing is not None and cache.vectors == existing.vectors:
        return cache, reused_id
    return cache, save_cache(cache, store, analysis.corpus_id).artifact_id


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


def _report_audit_spend(run) -> None:
    """What the pass cost, and which family cost the most of it.

    A total on its own is not actionable: the reason to report a slowest family
    is to go and look at it, and a bare duration named nothing to look at. Every
    figure here is reported only over the families that actually carry it, so a
    partial total is never printed as if it were the whole.
    """
    timed = [
        (a.duration_seconds, a.family_id) for a in run.audits if a.duration_seconds is not None
    ]
    if timed:
        slowest_seconds, slowest_family = max(timed)
        console.print(
            f"[dim]{len(timed)} family audit(s) took {sum(t for t, _ in timed):.1f}s, "
            f"slowest {slowest_family} at {slowest_seconds:.1f}s[/dim]"
        )

    # The per-family budget is thirty calls, so what matters is whether a family
    # is approaching it — a total alone would hide one family spending thirty
    # among nine spending two.
    counted = [(a.llm_calls, a.family_id) for a in run.audits if a.llm_calls is not None]
    if counted:
        most_calls, priciest = max(counted)
        console.print(
            f"[dim]{sum(c for c, _ in counted)} model call(s) over {len(counted)} family(ies), "
            f"most {priciest} at {most_calls}[/dim]"
        )
    unreported = [a for a in run.audits if a.llm_calls is None]
    if unreported:
        # Said out loud rather than left as a dash in a column: a total that
        # silently omits families reads as the whole bill.
        console.print(
            f"[dim]{len(unreported)} family(ies) reported no call count; "
            "the totals above exclude them[/dim]"
        )

    totals: dict[str, int] = {}
    for audit in run.audits:
        for field, value in audit.tokens.items():
            totals[field] = totals.get(field, 0) + value
    if totals:
        console.print(
            "[dim]tokens: " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())) + "[/dim]"
        )
    else:
        # Never estimated. A token count computed here from characters would be
        # wrong in a way that looks authoritative on a bill.
        console.print("[dim]no token usage was reported by the backend[/dim]")

    failures = run.failed()
    if failures:
        console.print(
            f"\n[yellow]{len(failures)} family audit(s) failed and reached no verdict[/yellow]"
        )
        for audit in failures:
            spent = "" if audit.llm_calls is None else f" after {audit.llm_calls} call(s)"
            console.print(f"  [yellow]failed[/yellow] {audit.family_id}{spent}: {audit.error}")


def _report_audit(run, run_id: str, task_set) -> None:
    """Advisory findings, kept visibly separate from what mining decided.

    The audit never changed the grouping printed above it, so it reads as a
    second opinion pointing at `split-family`, not as a result.
    """
    console.print(f"\naudit_id:    {run_id} ({run.model})")
    if not run.audits and not run.skipped:
        console.print("[dim]no families were eligible for audit[/dim]")
        return

    coherence_of = {f.family_id: f.coherence for f in task_set.families}
    concluded = run.concluded()
    table = Table(
        "family_id", "semantic", "geometric", "outliers", "proposed split", "calls", "took"
    )
    for audit in sorted(concluded, key=lambda a: (a.coherent, a.family_id)):
        measured = coherence_of.get(audit.family_id)
        # Printed beside each other and never reconciled: one read the
        # instructions, the other measured embedding distance. Where they
        # disagree is the finding, not a conflict to resolve here.
        geometric = (
            "[dim]not measured[/dim]"
            if measured is None
            else ("[yellow]over-merged[/yellow]" if measured.over_merged else "within threshold")
        )
        table.add_row(
            audit.family_id,
            "coherent" if audit.coherent else "[yellow]incoherent[/yellow]",
            geometric,
            str(len(audit.outlier_trace_ids)),
            " | ".join(str(len(g)) for g in audit.proposed_subgroups) or "-",
            "-" if audit.llm_calls is None else str(audit.llm_calls),
            "-" if audit.duration_seconds is None else f"{audit.duration_seconds:.1f}s",
        )
    if concluded:
        console.print(table)
    # Not gated on the table: a pass where every family failed has no rows and
    # is exactly the run whose cost and failures most need reporting.
    if run.audits:
        _report_audit_spend(run)

    # Names go under the table rather than in it: a generated name is prose and
    # a column narrow enough to fit beside five others would truncate the one
    # thing that made it worth generating.
    for audit in sorted(concluded, key=lambda a: a.family_id):
        if audit.generated_name:
            console.print(
                f"[dim]name[/dim] {audit.family_id}: {audit.generated_name}",
                overflow="ignore",
                crop=False,
                soft_wrap=True,
            )

    for audit in run.incoherent():
        console.print(f"\n[yellow]incoherent[/yellow] {audit.family_id}: {audit.rationale}")
        if audit.outlier_trace_ids:
            console.print(f"  outliers: {', '.join(audit.outlier_trace_ids)}")
        if audit.proposed_subgroups:
            # The audit proposes; `split-family` is what actually splits, and it
            # splits deterministically by exact instruction rather than by this.
            console.print(
                f"  to act on this: bandits split-family {run.task_set_id} {audit.family_id}",
                # A wrapped command cannot be copied and run; ids are long
                # enough that a narrow terminal would break every one of them.
                overflow="ignore",
                crop=False,
                soft_wrap=True,
            )

    for skip in run.skipped:
        # Failures are reported above with what they spent; repeating them here
        # as skips would read as two families lost rather than one.
        if skip.failed:
            continue
        console.print(f"[dim]skipped[/dim] {skip.family_id}: {skip.reason}")
    for limitation in run.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


def _run_audit(task_set, task_set_id: str, analysis, store, *, model: str, family_ids=None):
    """Audit a task set and persist the result beside it. Never rewrites it."""
    predict = build_predictor(model=model)
    with ledger.stage("audit_run", task_set_id=task_set_id, model=model):
        run = audit_task_set(
            task_set,
            task_set_id,
            analysis,
            predict=predict,
            model=model,
            family_ids=family_ids,
            on_error=lambda fid, msg: console.print(f"[yellow]audit failed[/yellow] {fid}: {msg}"),
        )
        envelope = save_audit_run(run, store)
        # Lineage rather than contents: the run is already an immutable
        # artifact, so the ledger records which one this pass produced and lets
        # the store hold what is in it.
        ledger.record(
            {
                "event_type": "stage_complete",
                "stage_name": "audit_run",
                "input_artifact_id": task_set_id,
                "output_artifact_id": envelope.artifact_id,
                "audited": len(run.concluded()),
                "failed": len(run.failed()),
                "skipped": len(run.skipped),
            }
        )
    _report_audit(run, envelope.artifact_id, task_set)
    return run


@app.command()
def mine(
    analysis_id: str,
    budget: int = typer.Option(DEFAULT_BUDGET, "--budget", help="How many tasks to select."),
    held_out: float = typer.Option(DEFAULT_HELD_OUT, "--held-out"),
    similarity: float = typer.Option(
        EMBEDDING_SIMILARITY,
        "--similarity",
        help="Higher groups more conservatively. Tuned for cosine similarity.",
    ),
    neighbors: int = typer.Option(
        DEFAULT_NEIGHBORS, "--neighbors", help="Maximum mutual neighbors per descriptor."
    ),
    duplicate_similarity: float = typer.Option(
        DEFAULT_DUPLICATE_SIMILARITY,
        "--duplicate-similarity",
        help="Above this two requests are the same one, and never straddle the split.",
    ),
    embedding_model: str = typer.Option(
        EMBEDDING_MODEL, "--embedding-model", help="Fireworks embedding model."
    ),
    audit: bool = typer.Option(
        True,
        "--audit/--no-audit",
        help="Run the advisory family coherence audit. Never changes grouping.",
    ),
    audit_model: str = typer.Option(AUDIT_MODEL, "--audit-model", help="Model for the audit."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Group an analysis into task families and select a representative set."""
    store = _derived(project)
    try:
        analysis = load_analysis(analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no analysis {analysis_id!r}")
        raise typer.Exit(code=1) from exc

    try:
        cache, cache_id = _embedding_cache(analysis, store, embedding_model)
    except EmbeddingError as exc:
        # Embedding failures must stop the run rather than produce a task set
        # whose requested clustering operation never completed.
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    task_set = mine_task_set(
        analysis,
        analysis_id,
        budget=budget,
        held_out=held_out,
        similarity=similarity,
        neighbors=neighbors,
        distance=embedding_distance(cache),
        duplicate_distance=embedding_distance(cache),
        duplicate_similarity=duplicate_similarity,
        backend="embedding",
        embedding_model=embedding_model,
        embedding_cache_id=cache_id,
        proposed_by="model",
    )
    envelope = save_task_set(task_set, store)
    _report(task_set, envelope.artifact_id)
    console.print(f"embeddings:  {cache_id} ({len(cache.vectors)} vectors, {embedding_model})")

    if not audit:
        return
    try:
        # Written as its own artifact parented to the task set just saved. The
        # task set is already persisted and is not touched again, so mining is
        # byte-identical whether or not this pass runs.
        _run_audit(task_set, envelope.artifact_id, analysis, store, model=audit_model)
    except AuditError as exc:
        # The grouping above is complete and saved. An audit that could not run
        # is a missing second opinion, not a failed mine.
        console.print(f"[yellow]audit skipped:[/yellow] {exc}")


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


@app.command(name="audit-families")
def audit_families_command(
    task_set_id: str,
    family: list[str] = typer.Option(
        None, "--family", help="Audit only these families. Repeatable."
    ),
    audit_model: str = typer.Option(AUDIT_MODEL, "--audit-model", help="Model for the audit."),
    show: str = typer.Option(
        None, "--show", help="Print a saved audit by id instead of running a new one."
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Read each family with a model and report whether it holds together.

    Advisory only: proposes splits for a reviewer to apply and never changes
    grouping, merges families, or touches clustering parameters.
    """
    store = _derived(project)
    task_set = _load_task_set(task_set_id, project)

    if show is not None:
        try:
            run = load_audit_run(show, store)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no audit {show!r}")
            raise typer.Exit(code=1) from exc
        if run.task_set_id != task_set_id:
            # The two are read independently, and nothing downstream notices the
            # mismatch: geometric coherence is looked up by family id, so a
            # family this task set never had reads as "not measured" rather than
            # as wrong, and the `split-family` line would name the audit's task
            # set beside a family judged against another one.
            console.print(
                f"[red]error:[/red] audit {show} is for task set {run.task_set_id}, "
                f"not {task_set_id}"
            )
            raise typer.Exit(code=1)
        _report_audit(run, show, task_set)
        return

    try:
        analysis = load_analysis(task_set.analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no analysis {task_set.analysis_id!r}")
        raise typer.Exit(code=1) from exc

    try:
        _run_audit(
            task_set,
            task_set_id,
            analysis,
            store,
            model=audit_model,
            family_ids=tuple(family) if family else None,
        )
    except (AuditError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="merge-families")
def merge_families_command(
    task_set_id: str,
    family_ids: list[str] = typer.Argument(..., help="Two or more families that are one task."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Record a reviewer's decision that several families are the same task."""
    store = _derived(project)
    task_set = _load_task_set(task_set_id, project)
    try:
        corrected = merge_families(
            task_set, tuple(family_ids), load_analysis(task_set.analysis_id, store)
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_task_set(corrected, store)
    _report(corrected, envelope.artifact_id)


@app.command(name="split-family")
def split_family_command(
    task_set_id: str,
    family_id: str,
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Split a family back into its exact-instruction groups."""
    store = _derived(project)
    task_set = _load_task_set(task_set_id, project)
    try:
        corrected = split_family(task_set, family_id, load_analysis(task_set.analysis_id, store))
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_task_set(corrected, store)
    _report(corrected, envelope.artifact_id)


@app.command(name="draft-verifier")
def draft_verifier_command(
    task_set_id: str,
    family_id: str = typer.Option(..., "--family", help="Family to draft checks for."),
    limit: int = typer.Option(3, "--limit", help="Maximum independent verifier drafts."),
    labels_id: str = typer.Option(
        None,
        "--labels",
        help="Adjudicated labels to rank candidates against, and to compose from.",
    ),
    interview: bool = typer.Option(
        False, "--interview", help="Immediately run the bounded owner-review interview."
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Propose deterministic replay verifiers from recorded terminal evidence."""
    store = _derived(project)
    task_set = _load_task_set(task_set_id, project)
    try:
        analysis = load_analysis(task_set.analysis_id, store)
        labels = load_label_set(labels_id, store) if labels_id else None
        draft = draft_verifiers(
            task_set, task_set_id, analysis, family_id, limit=limit, labels=labels
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_verifier_draft(draft, store)
    console.print(f"verifier_draft_id: {envelope.artifact_id}")
    console.print(f"family:            {draft.family_id}")
    console.print(f"verifiers:         {len(draft.verifiers)}")

    table = Table("verifier_id", "mode", "status", "check", "expected", "evidence")
    for spec in draft.verifiers:
        for index, check in enumerate(spec.checks):
            table.add_row(
                spec.verifier_id if index == 0 else "",
                spec.mode.value if index == 0 else "",
                spec.status.value if index == 0 else "",
                check.claim,
                repr(check.expected),
                check.evidence_kind.value,
            )
    console.print(table)
    _show_candidates(draft)
    for unresolved in draft.unresolved:
        console.print(f"[yellow]unresolved:[/yellow] {unresolved}")

    # A drafted check is a hypothesis. Run it before anyone is asked about it.
    run = run_draft(draft, analysis, task_set)
    save_draft_run(run, store)
    _show_draft_run(run)

    if interview:
        _run_verifier_interview(draft, envelope.artifact_id, store)


def _show_candidates(draft) -> None:
    """How each proposal behaved, next to what it proposes.

    A ranked list with nothing behind it invites the top row to be taken as the
    answer. The numbers are what let an owner disagree with the ranking.
    """
    if not draft.candidates:
        return
    table = Table("verifier_id", "from", "successes", "failures", "false pos", "coverage", "why")
    for stats in draft.candidates:
        table.add_row(
            stats.verifier_id,
            stats.derivation,
            f"{stats.success_support}/{stats.labeled_successes}" if stats.calibrated else "—",
            f"{stats.failure_rejection}/{stats.labeled_failures}" if stats.calibrated else "—",
            str(stats.false_positives) if stats.calibrated else "—",
            f"{stats.coverage:.0%} ({stats.unknown} unknown)",
            stats.rationale,
        )
    console.print(table)


def _show_draft_run(run) -> None:
    """Put results in front of the owner before asking them to review the check."""
    console.print(
        f"\nscored {len({o.trace_id for o in run.outcomes})} historical run(s) "
        f"with {len({o.verifier_id for o in run.outcomes})} verifier(s)"
    )

    if run.disagreements:
        table = Table("trace_id", "kind", "scores")
        for item in run.disagreements:
            scores = ", ".join(
                f"{vid[:20]}={'unknown' if score is None else score}"
                for vid, score in sorted(item.scores.items())
            )
            table.add_row(item.trace_id, item.kind, scores)
        console.print(table)
        console.print(
            "[yellow]these runs are where labeling pays[/yellow]: the verifiers "
            "split on them, so one label resolves all of them at once"
        )
    else:
        console.print("no verifier disagreed with another on any scored run")

    if run.unscorable_trace_ids:
        console.print(
            f"[yellow]unscorable:[/yellow] {len(run.unscorable_trace_ids)} run(s) recorded no "
            "evidence any check could read — reported, never counted as failures"
        )


def _run_verifier_interview(draft, verifier_draft_id: str, store: DerivedStore, run=None) -> None:
    if run is not None:
        _show_draft_run(run)
    interview = start_interview(draft, verifier_draft_id)
    while (question := next_question(interview)) is not None:
        console.print(f"\n[bold]{question.prompt}[/bold]")
        answer = typer.prompt("Answer", default="", show_default=False)
        interview = answer_question(interview, answer)

    envelope = save_interview(interview, store)
    console.print(f"interview_id: {envelope.artifact_id}")
    console.print(f"questions:    {len(interview.questions)}")
    console.print("status:       complete")
    console.print(
        "[yellow]note:[/yellow] review refined the hypothesis; validation is still required "
        "before calibrated or reviewed status"
    )


@app.command(name="interview-verifier")
def interview_verifier_command(
    verifier_draft_id: str,
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Review a verifier draft through a bounded, one-question-at-a-time interview."""
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no verifier draft {verifier_draft_id!r}")
        raise typer.Exit(code=1) from exc

    _run_verifier_interview(draft, verifier_draft_id, store)


_VERDICTS = {"s": Verdict.SUCCESS, "f": Verdict.FAILURE, "u": Verdict.UNCLEAR}


def _brief(value: object, limit: int = 180) -> str:
    rendered = str(value).replace("\n", " ").strip()
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _show_label_card(trace_id: str, task, evidence: list, position: int, total: int) -> None:
    """Show only the facts a reviewer needs; raw evidence remains in the artifact."""
    console.rule(Text(f"Review {position}/{total} · {trace_id}"))
    console.print("[bold cyan]Request[/bold cyan]")
    console.print(
        _brief(task.instruction, 500) if task is not None else "[not recorded]",
        markup=False,
    )

    states = [item for item in evidence if item.claim == "final_state_field"]
    errors = [item for item in evidence if item.claim in {"span_error", "missing_tool_result"}]
    finals = [item for item in evidence if item.claim == "final_output"]
    scores = [item for item in evidence if item.claim in {"command_exit_code", "recorded_score"}]

    console.print("\n[bold cyan]What happened[/bold cyan]")
    if states:
        by_tool: dict[str, list[str]] = {}
        for item in states:
            value = item.value if isinstance(item.value, dict) else {}
            tool = str(value.get("tool", "result"))
            key = str(value.get("key", "value"))
            by_tool.setdefault(tool, []).append(f"{key}={_brief(value.get('value'), 80)}")
        for tool, facts in list(by_tool.items())[-3:]:
            visible = facts[:6]
            suffix = f" (+{len(facts) - 6} more)" if len(facts) > 6 else ""
            console.print(f"  {tool}: {', '.join(visible)}{suffix}", markup=False)
    elif not errors and not scores:
        console.print("  [yellow]No structured outcome was recorded.[/yellow]")
    for item in errors:
        console.print(f"  {item.claim}: {_brief(item.value)}", markup=False)
    for item in scores:
        console.print(f"  {item.claim}: {_brief(item.value)}", markup=False)

    console.print("\n[bold cyan]Agent's final response[/bold cyan]")
    if finals:
        value = finals[-1].value
        output = value.get("output") if isinstance(value, dict) else value
        console.print(_brief(output, 600), markup=False)
    else:
        console.print("[yellow]None recorded.[/yellow]")


@app.command()
def label(
    verifier_draft_id: str,
    labeler: str = typer.Option(..., "--labeler", help="Who is answering."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Label the runs a family's verifiers disagree about.

    Disagreements first, because that is where one label buys the most: it
    resolves an ambiguity every verifier in the family shares.
    """
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
        task_set = load_task_set(draft.task_set_id, store)
        analysis = load_analysis(draft.analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    run = run_draft(draft, analysis, task_set)
    family = task_set.family_by_id()[draft.family_id]
    queue = [item.trace_id for item in run.disagreements]
    queue += [t for t in family.trace_ids if t not in set(queue)]

    console.print(f"family:  {family.descriptor}")
    console.print(f"to label: {len(queue)} run(s), {len(run.disagreements)} disputed first\n")

    labels = []
    tasks = {task.trace_id: task for task in analysis.tasks}
    evidence_by_trace = {}
    for item in analysis.evidence:
        evidence_by_trace.setdefault(item.trace_id, []).append(item)
    for position, trace_id in enumerate(queue, 1):
        scores = run.scores_for(trace_id)
        rendered = ", ".join(
            f"{vid[:18]}={'unknown' if s is None else s}" for vid, s in sorted(scores.items())
        )
        task = tasks.get(trace_id)
        _show_label_card(trace_id, task, evidence_by_trace.get(trace_id, []), position, len(queue))
        console.print(f"\n[dim]Verifier scores: {rendered or 'not scored'}[/dim]")
        answer = typer.prompt("Decision [s]uccess / [f]ailure / [u]nclear / [q]uit", default="u")
        if answer.strip().lower().startswith("q"):
            break
        verdict = _VERDICTS.get(answer.strip().lower()[:1], Verdict.UNCLEAR)
        rationale = typer.prompt("  why (optional)", default="", show_default=False)
        labels.append(
            make_label(
                trace_id=trace_id,
                family_id=family.family_id,
                verdict=verdict,
                labeler=labeler,
                rationale=rationale,
                prompted_by=verifier_draft_id
                if trace_id in set(queue[: len(run.disagreements)])
                else None,
            )
        )

    label_set = LabelSet(
        task_set_id=draft.task_set_id, family_id=family.family_id, labels=tuple(labels)
    )
    envelope = save_label_set(label_set, store)
    console.print(f"\nlabel_set_id: {envelope.artifact_id}")
    console.print(f"labels:       {len(label_set.labels)}")
    console.print(f"adjudicated:  {len(label_set.adjudicated())}")


@app.command(name="validate-verifier")
def validate_verifier_command(
    verifier_draft_id: str,
    label_set_id: str = typer.Option(..., "--labels"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Measure a draft against labels, then try to satisfy it without doing the task."""
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
        task_set = load_task_set(draft.task_set_id, store)
        analysis = load_analysis(draft.analysis_id, store)
        label_set = load_label_set(label_set_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        validation = validate_draft(
            draft, verifier_draft_id, task_set, analysis, label_set, label_set_id
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_validation(validation, store)
    console.print(f"validation_id: {envelope.artifact_id}")
    console.print(f"labels used:   {validation.labels_used}")

    expected_by_id = {spec.verifier_id: spec.checks[0].claim for spec in draft.verifiers}
    table = Table("verifier", "check", "split", "agree", "disagree", "unscored", "rate")
    for item in validation.agreements:
        if not item.labeled:
            continue
        table.add_row(
            item.verifier_id[:18],
            expected_by_id.get(item.verifier_id, "")[:30],
            item.split,
            str(item.agreed),
            str(item.disagreed),
            str(item.unscored),
            "n/a" if item.agreement is None else f"{item.agreement:.0%}",
        )
    console.print(table)

    # The counterexamples matter more than the rate: they show how a check would
    # reward the wrong behaviour.
    for item in validation.agreements:
        for counter in item.counterexamples:
            console.print(
                f"[red]{counter.kind}[/red] {counter.trace_id} ({item.split}): "
                f"verifier={counter.verifier_score} human={counter.human_verdict}"
            )

    if validation.gameability:
        table = Table("verifier", "forged facts", "result", "hypothesis")
        for result in validation.gameability:
            table.add_row(
                result.verifier_id[:18],
                str(result.forged_facts),
                "[red]gamed[/red]" if result.passed else "held",
                result.hypothesis[:52],
            )
        console.print(table)

    for limitation in validation.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="review-verifier")
def review_verifier_command(
    verifier_draft_id: str,
    validation_id: str = typer.Option(..., "--validation"),
    verifier_id: str = typer.Option(..., "--verifier"),
    interview_id: str = typer.Option(
        ...,
        "--interview",
        help="The confirmed review round that accepted this verifier.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Promote one calibrated verifier a review round accepted.

    Takes the review rather than a sign-off string: what authorises a promotion
    is a person having seen the measurements and said yes, and only the review
    artifact records that happening.
    """
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
        validation = load_validation(validation_id, store)
        interview = load_interview(interview_id, store)
        reviewed = review_verifier(
            draft,
            verifier_draft_id,
            validation,
            validation_id,
            verifier_id,
            interview,
            interview_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_reviewed_verifier(reviewed, store)
    console.print(f"reviewed_verifier_id: {envelope.artifact_id}")
    console.print(f"verifier:             {reviewed.spec.verifier_id}")
    console.print(f"status:               {reviewed.spec.status.value}")
    console.print(f"validation:           {reviewed.validation_id}")
    console.print(f"review:               {reviewed.interview_id}")
    console.print(f"threshold:            {reviewed.success_threshold}")


_SKEW_WARNING = 0.5
"""Share of a dataset one family, source, model or tool may hold before it is
worth saying out loud. Not a limit — nothing is refused for crossing it — but a
dataset half made of one thing is rarely the dataset someone believes they
asked for, and the summary above it reads the same either way."""


def _report_composition(report: CompositionReport | None) -> None:
    """Say what the selected dataset is mostly made of, when it is mostly one thing."""
    if report is None or not report.selected.rows:
        return
    rows = report.selected.rows
    console.print(
        f"composed of: {report.selected.lineages} lineage(s) over "
        f"{rows} row(s), median {report.selected.messages_per_row.median:g} message(s) "
        f"and {report.selected.characters_per_row.median:g} character(s) per row"
    )
    for dimension, counts in (
        ("family", report.selected.rows_by_family),
        ("source", report.selected.rows_by_source),
        ("model", report.selected.rows_by_model),
        ("tool", report.selected.rows_by_tool),
    ):
        if len(counts) < 2:
            continue
        name, count = next(iter(counts.items()))
        if count > _SKEW_WARNING * rows:
            console.print(
                f"[yellow]skew:[/yellow] {count} of {rows} row(s) share one {dimension} "
                f"({name}); see the composition report"
            )
    repeated = report.selected.repeated_lineages
    if repeated:
        console.print(
            f"[yellow]repeated lineage:[/yellow] {len(repeated)} lineage(s) contribute "
            f"more than one row, the largest {max(repeated.values())}"
        )


@app.command(name="export")
def export_command(
    task_set_id: str,
    format_: str = typer.Option(..., "--format", help="One of: eval, sft"),
    reviewed_verifier_id: str = typer.Option(..., "--verifier"),
    output: Path = typer.Option(..., "--output"),
    split: str = typer.Option(
        None,
        "--split",
        help="fit, held_out or all. Defaults to fit for sft and held_out for eval.",
    ),
    max_rows_per_family: int = typer.Option(
        None, "--max-rows-per-family", help="SFT only. Unset means no limit."
    ),
    max_rows_per_lineage: int = typer.Option(
        None, "--max-rows-per-lineage", help="SFT only. Caps one retry chain or session."
    ),
    max_messages_per_row: int = typer.Option(None, "--max-messages-per-row", help="SFT only."),
    max_characters_per_row: int = typer.Option(
        None, "--max-characters-per-row", help="SFT only. Characters, never tokens."
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Write reviewed eval or SFT rows plus an unresolved quarantine file."""
    if format_ not in {"eval", "sft"}:
        console.print("[red]error:[/red] --format must be one of: eval, sft")
        raise typer.Exit(code=1)
    try:
        caps = SamplingCaps(
            max_rows_per_family=max_rows_per_family,
            max_rows_per_lineage=max_rows_per_lineage,
            max_messages_per_row=max_messages_per_row,
            max_characters_per_row=max_characters_per_row,
        )
    except ValidationError as exc:
        console.print(f"[red]error:[/red] {exc.errors()[0]['msg']}")
        raise typer.Exit(code=1) from exc
    if format_ == "eval" and caps.configured:
        # Silently ignoring them would produce an eval set that looks capped.
        console.print("[red]error:[/red] sampling caps apply to --format sft only")
        raise typer.Exit(code=1)
    default_split = Partition.FIT if format_ == "sft" else Partition.HELD_OUT
    try:
        partition = Partition(split) if split else default_split
    except ValueError:
        console.print("[red]error:[/red] --split must be one of: fit, held_out, all")
        raise typer.Exit(code=1) from None
    store = _derived(project)
    try:
        task_set = load_task_set(task_set_id, store)
        analysis = load_analysis(task_set.analysis_id, store)
        corpus = ArtifactStore(project / ".bandits").read(task_set.corpus_id)
        reviewed = load_reviewed_verifier(reviewed_verifier_id, store)
        arguments = {"partition": partition}
        if format_ == "sft":
            arguments["caps"] = caps
        builder = build_eval_export if format_ == "eval" else build_sft_export
        bundle = builder(
            corpus,
            task_set,
            task_set_id,
            analysis,
            reviewed,
            reviewed_verifier_id,
            **arguments,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_export(bundle, store)
    accepted_path, unresolved_path, composition_path = write_jsonl(bundle, output)
    console.print(f"export_id:   {envelope.artifact_id}")
    console.print(f"format:      {format_}")
    console.print(
        f"split:       {partition.value} ({bundle.manifest.partition_trace_count} trace(s))"
    )
    console.print(f"threshold:   {bundle.manifest.success_threshold}")
    console.print(f"authorized:  {bundle.manifest.verifier_status}")
    console.print(f"rows:        {len(bundle.rows)}")
    console.print(f"unresolved:  {len(bundle.unresolved)}")
    console.print(f"output:      {accepted_path}")
    console.print(f"quarantine:  {unresolved_path}")
    if composition_path is not None:
        console.print(f"composition: {composition_path}")
    _report_composition(bundle.composition)
    for code in bundle.manifest.accepted_risks:
        console.print(f"[yellow]accepted risk:[/yellow] {code}")
    for warning in bundle.manifest.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


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


@app.command()
def judge(
    task_set_id: str,
    family_id: str = typer.Option(..., "--family"),
    criterion: str = typer.Option(..., "--criterion", help="What success means, in one line."),
    rubric_id: str = typer.Option("rubric-v1", "--rubric-id"),
    samples: int = typer.Option(5, "--samples", help="Higher separates confident from contested."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Score a family with a model judge, for tasks no deterministic check reaches."""
    store = _derived(project)
    task_set = _load_task_set(task_set_id, project)
    family = task_set.family_by_id().get(family_id)
    if family is None:
        console.print(f"[red]error:[/red] no family {family_id!r} in {task_set_id}")
        raise typer.Exit(code=1)

    corpus = ArtifactStore(project / ".bandits").read(task_set.corpus_id)
    traces = [t for t in corpus.traces if t.trace_id in set(family.trace_ids)]
    rubric = Rubric(rubric_id=rubric_id, family_id=family_id, criterion=criterion, samples=samples)

    try:
        run = judge_traces(traces, rubric, task_set_id)
    except JudgeError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_judge_run(run, store)
    console.print(f"judge_run_id:  {envelope.artifact_id}")
    console.print(f"prompt digest: {rubric.prompt_digest}")
    console.print(f"model:         {rubric.model}")

    table = Table("trace_id", "samples", "score", "agreement")
    for verdict in run.verdicts:
        table.add_row(
            verdict.trace_id,
            str(list(verdict.samples)),
            "unknown" if verdict.score is None else f"{verdict.score:.2f}",
            f"{verdict.agreement:.0%}",
        )
    console.print(table)

    # A run the judge argued with itself about is worth a label, not a score.
    contested = run.contested_trace_ids()
    if contested:
        console.print(
            f"[yellow]contested:[/yellow] {', '.join(contested)} — the judge disagreed "
            "with itself; these score unknown until a human settles them"
        )


_INTERPRETER: object | None = None
"""Overridden by tests so the interview never reaches the network.

None means ``interpret_reply`` resolves its own default client. Injecting here
rather than threading a parameter through the command keeps the CLI signature
about the review and not about which model client is in use.
"""


def _interpreter():
    return _INTERPRETER


_DECISION_KEYS = {
    "a": InterviewDecision.ACCEPT,
    "r": InterviewDecision.REJECT,
    "v": InterviewDecision.REVISE,
    "c": InterviewDecision.COMBINE,
}


def _interpretation_record(interpretation) -> dict[str, object] | None:
    """A reading in full, not just the decision it reached.

    ``decision`` and ``rationale`` say what was proposed and not what it would
    have done. The revised value, the operator, the combine target and the
    target that could not be found are the proposal; a record holding only the
    first two cannot say what a reviewer accepted or refused.
    """
    if interpretation is None:
        return None
    return {
        "decision": interpretation.decision.value,
        "rationale": interpretation.rationale,
        "source": getattr(interpretation, "source", None),
        "revised_expected": getattr(interpretation, "revised_expected", None),
        "revised_operator": getattr(
            getattr(interpretation, "revised_operator", None), "value", None
        ),
        "combine_with": getattr(interpretation, "combine_with", None),
        "dropped_combine_target": getattr(interpretation, "dropped_combine_target", None),
        "blind_spots": list(getattr(interpretation, "blind_spots", ()) or ()),
        "gaming_hypotheses": list(getattr(interpretation, "gaming_hypotheses", ()) or ()),
    }


def _record_turn(
    outcome: str,
    *,
    verifier_id: str,
    check_id: str,
    round_number: int,
    shown_at: str,
    answered_seconds: float,
    shown: dict,
    reply: str,
    authoritative: bool,
    authoritative_why: str,
    interpretation,
    applied,
    manual: bool,
    failure: str | None,
    **extra: object,
) -> None:
    """One interview turn, however it ended.

    ``outcome`` is what happened to the answer: applied, stopped, or skipped
    because the decision named nothing that could be acted on. A turn that
    ended in a skip still cost the reviewer the reading and the reply, and
    dropping it would make the review look shorter than it was.
    """
    ledger.record(
        {
            "event_type": "interview_turn",
            "outcome": outcome,
            "verifier_id": verifier_id,
            "check_id": check_id,
            "round_number": round_number,
            "shown_at": shown_at,
            "answered_seconds": answered_seconds,
            "shown": shown,
            "reply": reply,
            "authoritative": authoritative,
            "authoritative_why": authoritative_why,
            # Both readings, whole. A decision and a rationale describe what was
            # proposed and not what it would have done: the revised value, the
            # operator and the combine target are the proposal. Where a reviewer
            # overruled the model, `CheckReview.interpretation` holds their
            # replacement, so keeping only one of these loses whichever was not
            # taken — and the pair is the whole point of recording an overrule.
            "proposed_interpretation": _interpretation_record(interpretation),
            "applied_interpretation": _interpretation_record(applied),
            "proposed_decision": (
                interpretation.decision.value if interpretation is not None else None
            ),
            "proposed_rationale": (
                interpretation.rationale if interpretation is not None else None
            ),
            "decision_source": _decision_source(interpretation, manual, failure),
            # True only where a reading existed and was refused. A manual
            # decision after an interpretation failure overrules nothing.
            "model_overruled": interpretation is not None and manual,
            "failure": failure,
            **extra,
        }
    )


def _decision_source(interpretation, manual: bool, failure: str | None) -> str:
    """How the decision was arrived at, which is not a single boolean.

    Three cases, and flattening them misreports the evidence: a reading the
    reviewer accepted, a reading they refused and replaced, and a decision they
    entered because no reading existed. Only the middle one is an override —
    calling the third an override invents a model opinion that was never given.
    """
    if failure is not None or interpretation is None:
        return "manual_after_failure"
    return "model_overruled" if manual else "model_accepted"


def _displayed_context(summary, check, spec, interview, check_id: str) -> dict[str, object]:
    """Everything the reviewer had in front of them when they answered.

    The prompt lines alone were not it. The terminal also showed the check's
    own wording, the operator and expected value being asserted, the evidence
    kind, the agreement and error counts, the gameability findings, the blind
    spots and every earlier decision on this check. All of it is input to a
    human judgement, so a record that kept only part of it cannot explain the
    answer that came back.

    A structured snapshot rather than the rendered text: the console output is
    styled for a terminal, and re-reading markup is not the same as reading
    what it said.
    """
    return {
        "prompt_lines": list(summary.prompt_lines()),
        "check": {
            "check_id": check.check_id,
            "claim": check.claim,
            "description": check.description,
            "operator": getattr(check, "operator", None),
            "expected": getattr(check, "expected", None),
        },
        "verifier_id": spec.verifier_id,
        "scored": {
            "passed": summary.passed,
            "failed": summary.failed,
            "unscorable": summary.unscorable,
            "example_trace_ids": list(summary.example_trace_ids),
            "evidence_kind": str(summary.evidence_kind),
        },
        "agreements": [
            {
                "split": a.split,
                "agreement": a.agreement,
                "labeled": a.labeled,
                "scored": a.scored,
                "false_positives": a.false_positives,
                "false_negatives": a.false_negatives,
                "coverage": a.coverage,
            }
            for a in summary.agreements
        ],
        "gameability": [
            {"hypothesis": g.hypothesis, "passed": g.passed, "forged_facts": g.forged_facts}
            for g in summary.gameability
        ],
        # Coverage is shown beside the attacks and says something they cannot:
        # that checks no template could attack were never tried, so a clean
        # sheet above is not evidence they resist one.
        "gameability_assessment": (
            None
            if summary.assessment is None
            else {
                "coverage": summary.assessment.coverage,
                "checks_attacked": summary.assessment.checks_attacked,
                "checks_total": summary.assessment.checks_total,
            }
        ),
        "blind_spots": list(summary.blind_spots),
        "gaming_hypotheses": list(summary.gaming_hypotheses),
        "prior_decisions": list(prior_decisions(interview, check_id)),
    }


def _show_check_summary(summary, check, spec) -> None:
    console.print(f"\n[bold]{check.claim}[/bold]  [dim]{check.check_id}[/dim]")
    console.print(f"  {check.description}")
    console.print(
        f"  scored: [green]{summary.passed} passed[/green], "
        f"[red]{summary.failed} failed[/red], {summary.unscorable} unscorable"
    )
    if summary.passed and not summary.failed:
        console.print(
            "  [yellow]passed every run it could score[/yellow]: nothing here shows it "
            "telling success from failure"
        )
    if summary.example_trace_ids:
        console.print(f"  examples: {', '.join(summary.example_trace_ids)}")
    console.print(f"  evidence: {summary.evidence_kind}")
    if not summary.agreements:
        console.print(
            "  [yellow]agreement: unavailable[/yellow] — no labeled measurement covers "
            "this verifier yet"
        )
    for agreement in summary.agreements:
        rate = "unmeasured" if agreement.agreement is None else f"{agreement.agreement:.0%}"
        console.print(
            f"  [cyan]agreement ({agreement.split})[/cyan]: {rate} of "
            f"{agreement.labeled} labeled run(s)"
        )
        if agreement.scored and agreement.false_positives is not None:
            console.print(
                f"    errors: [red]{agreement.false_positives} false positive(s)[/red], "
                f"{agreement.false_negatives} false negative(s)"
            )
            caught = agreement.failure_catch_rate
            if caught is not None:
                console.print(
                    f"    failures caught: {caught:.0%} ({agreement.caught_failures} of "
                    f"{agreement.caught_failures + agreement.false_positives})"
                )
        if agreement.coverage is not None and agreement.unscored:
            console.print(
                f"    coverage: {agreement.coverage:.0%} "
                f"({agreement.unscored} of {agreement.labeled} unscorable)"
            )
    if summary.assessment is not None:
        assessment = summary.assessment
        colour = "red" if assessment.coverage == "none" else "cyan"
        console.print(
            f"  [{colour}]gameability coverage[/{colour}]: {assessment.coverage} "
            f"({assessment.checks_attacked} of {assessment.checks_total} check(s) attackable)"
        )
        if assessment.coverage != "complete":
            console.print(
                "    [yellow]checks no template could attack were never tried[/yellow]: "
                "a failed attack here is not evidence they resist one"
            )
    for attack in summary.gameability:
        if attack.passed:
            console.print(
                f"  [red]gameable[/red]: {attack.hypothesis} ({attack.forged_facts} forged fact(s))"
            )
        else:
            console.print(
                f"  [dim]resisted[/dim]: {attack.hypothesis} ({attack.forged_facts} forged fact(s))"
            )
    for blind in summary.blind_spots:
        console.print(f"  [dim]blind spot:[/dim] {blind}")
    for gaming in summary.gaming_hypotheses:
        console.print(f"  [dim]gaming:[/dim] {gaming}")


def _probe_hypotheses(spec, check, interpretation) -> None:
    """Run any named gaming hypothesis through the real attack machinery.

    A hypothesis an owner names is worth only as much as what tests it. Where a
    template exists for the operator, the attack is constructed and scored;
    where none does, that is said plainly rather than left looking tested.
    """
    if interpretation is None or not interpretation.gaming_hypotheses:
        return
    # Out of scope for #14: synthesising forged evidence for a hypothesis no
    # template matches. ``_attack()`` dispatches on the operator, so a novel
    # attack against an operator that already has a template gets that
    # template's canned attack rather than the one the reviewer described. The
    # gap is reported below rather than hidden; closing it means teaching
    # ``_attack`` to build evidence from an interpreted hypothesis, which is a
    # change to the attack machinery and deserves its own issue.
    single = spec.replace(checks=(check,))
    results = probe_gameability(single)
    for hypothesis in interpretation.gaming_hypotheses:
        console.print(f"\n  [dim]probing:[/dim] {hypothesis}")
        if not results:
            console.print(
                f"  [yellow]no attack template for {check.operator.value}[/yellow]: "
                "recorded, but nothing here tests it"
            )
            continue
        for result in results:
            verdict = "[red]passed[/red]" if result.passed else "[green]held[/green]"
            console.print(f"  {verdict}: {result.hypothesis}")


def _manual_decision(reason: str) -> InterviewDecision | None:
    console.print(f"  [yellow]{reason}[/yellow]")
    raw = typer.prompt("  decide directly [a]ccept/[r]eject/re[v]ise/[c]ombine", default="")
    return _DECISION_KEYS.get(raw.strip().lower()[:1])


def _manual_interpretation(
    decision: InterviewDecision,
    check: CheckSpec,
    known_check_ids: tuple[str, ...],
) -> Interpretation | None:
    """Collect what a revise or combine needs when the model did not supply it.

    A decision enum alone cannot carry either action: a revise needs the value
    or operator the check becomes, and a combine needs a target that resolves.
    Without them the decision is refused downstream and the check stays pending,
    so the reviewer would be re-asked forever with no way to answer.

    ``accept`` and ``reject`` need nothing beyond the enum and return ``None``,
    which is what ``apply_decision`` already expects for them.
    """
    if decision is InterviewDecision.REVISE:
        rationale = typer.prompt("  why this revision", default="entered by the reviewer")
        raw_value = typer.prompt("  new expected value (blank to keep)", default="")
        raw_operator = typer.prompt(
            f"  new operator (blank to keep {check.operator.value})", default=""
        )
        operator = None
        if raw_operator.strip():
            try:
                operator = CheckOperator(raw_operator.strip().lower())
            except ValueError:
                console.print(f"  [yellow]unknown operator {raw_operator!r}[/yellow]")
                return None
        # Parsed the way the model path parses it, so a value entered by hand and
        # the same value proposed by the model resolve to one check identity.
        expected = parse_expected(raw_value) if raw_value.strip() else None
        if expected is None and operator is None:
            console.print("  [yellow]a revision needs a new value or operator[/yellow]")
            return None
        return Interpretation(
            source="human",
            decision=decision,
            rationale=rationale,
            revised_expected=expected,
            revised_operator=operator,
        )

    if decision is InterviewDecision.COMBINE:
        targets = tuple(c for c in known_check_ids if c != check.check_id)
        if not targets:
            console.print("  [yellow]no other check to combine with[/yellow]")
            return None
        console.print(f"  other checks: {', '.join(targets)}")
        rationale = typer.prompt("  why this combination", default="entered by the reviewer")
        target = typer.prompt("  combine with which check_id", default="").strip()
        if target not in targets:
            console.print(f"  [yellow]no check named {target!r}[/yellow]")
            return None
        return Interpretation(
            source="human", decision=decision, rationale=rationale, combine_with=target
        )

    return None


# Named ``interview-review`` rather than ``review-verifier``: that name is
# already the acceptance command above, which promotes a calibrated verifier to
# reviewed. Two commands whose names differ only by word order, one refining a
# hypothesis and one promoting it past validation, is a mistake waiting to be
# typed. ``interview-verifier`` keeps the older fixed-question flow, which still
# works and is still tested.
@app.command(name="interview-review")
def interview_review_command(
    verifier_draft_id: str,
    validation_id: str = typer.Option(None, "--validation", help="Results of an earlier round."),
    prior_interview_id: str = typer.Option(None, "--prior", help="The round before this one."),
    round_number: int = typer.Option(
        1, "--round", min=1, help="Ignored with --prior, which derives the round from the chain."
    ),
    model: str = typer.Option(INTERPRETER_MODEL, "--model"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Review a verifier draft by saying what you think, in your own words.

    One open question per check. A model reads the reply and proposes a
    decision; you confirm it before anything is applied.
    """
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
        task_set = load_task_set(draft.task_set_id, store)
        analysis = load_analysis(draft.analysis_id, store)
        validation = load_validation(validation_id, store) if validation_id else None
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Each invocation opens a new round rather than extending an existing
    # interview. ``DerivedStore.write`` is content-addressed, so appending to one
    # record would mint a fresh id on every save regardless — the chain exists
    # either way, and ``prior_interview_id`` makes it explicit instead of
    # leaving a series of ids that each claim to be the whole history. It also
    # keeps a round's payload from re-serialising every earlier round on each
    # per-decision save.
    prior = None
    if prior_interview_id:
        try:
            prior = load_interview(prior_interview_id, store)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no interview {prior_interview_id!r}")
            raise typer.Exit(code=1) from exc
        if prior.source_draft_id != verifier_draft_id:
            console.print(
                f"[red]error:[/red] interview {prior_interview_id} reviewed draft "
                f"{prior.source_draft_id}, not {verifier_draft_id}"
            )
            raise typer.Exit(code=1)
        if not prior.complete:
            console.print(
                f"[red]error:[/red] interview {prior_interview_id} has "
                f"{len(prior.pending)} check(s) still undecided"
            )
            raise typer.Exit(code=1)
        # Derived, not taken on trust: a round number that disagreed with the
        # chain would misorder the decisions a later reader walks back through.
        if round_number != 1 and round_number != prior.round_number + 1:
            console.print(
                f"[yellow]note:[/yellow] --round {round_number} ignored; "
                f"{prior_interview_id} is round {prior.round_number}"
            )
        round_number = prior.round_number + 1

    interview = start_review(
        draft,
        verifier_draft_id,
        validation_id=validation_id,
        prior_interview_id=prior_interview_id,
        round_number=round_number,
        prior=prior,
    )
    # Executed after the round opens, over the draft the round actually holds.
    # A revision or combination in an earlier round mints a new verifier id, and
    # a run built from the originally loaded draft carries outcomes for the ids
    # that round replaced. ``build_check_summary`` matches on ``verifier_id``,
    # so those summaries came out zero-passed, zero-failed and zero-unscorable —
    # a check that looks unscored rather than one that was never run.
    run = run_draft(interview.draft, analysis, task_set)
    if prior is not None:
        console.print(
            f"[dim]round {round_number}, continuing {prior_interview_id} "
            f"({len(prior.reviews)} earlier decision(s))[/dim]"
        )
    envelope = save_interview(interview, store)

    while (target := next_check(interview)) is not None:
        verifier_id, check_id = target
        spec, check = find_check(interview.draft, verifier_id, check_id)
        summary = build_check_summary(spec, check, run, validation=validation)
        _show_check_summary(summary, check, spec)

        for line in prior_decisions(interview, check_id):
            console.print(f"  [dim]earlier:[/dim] {line}")

        # Marked before the prompt goes up: the interval a reviewer spent on a
        # check is part of how much its answer is worth, and it is unrecoverable
        # once the answer is stored on its own.
        shown_at = datetime.now(UTC).isoformat()
        asked = time.monotonic()
        reply = typer.prompt("\n  what do you think?", default="", show_default=False)
        authoritative = typer.confirm(
            "  is this evidence source authoritative for the claim?", default=True
        )
        why = typer.prompt("  why", default="", show_default=False)
        answered_seconds = round(time.monotonic() - asked, 3)

        known = tuple(c.check_id for s in interview.draft.verifiers for c in s.checks)
        interpretation = prompt_text = response = None
        failure = None
        manual = False
        try:
            with ledger.stage(
                "interview_interpret",
                verifier_id=verifier_id,
                check_id=check_id,
                round_number=interview.round_number,
            ):
                interpretation, prompt_text, response = interpret_reply(
                    check,
                    spec,
                    reply,
                    predict=_interpreter(),
                    model=model,
                    summary_lines=summary.prompt_lines(),
                    prior_reviews=prior_decisions(interview, check_id),
                    known_check_ids=known,
                )
        except InterpretationFailure as exc:
            failure = f"{exc.kind}: {exc}"
            decision = _manual_decision(f"could not read that reply — {failure}")
            manual = decision is not None
        else:
            console.print(f"\n  [bold]read as:[/bold] {interpretation.decision.value}")
            console.print(f"  rationale: {interpretation.rationale}")
            if interpretation.revised_expected is not None:
                console.print(f"  new expected: {interpretation.revised_expected!r}")
            if interpretation.combine_with:
                console.print(f"  combine with: {interpretation.combine_with}")
            if interpretation.dropped_combine_target:
                console.print(
                    f"  [yellow]no check named {interpretation.dropped_combine_target!r}[/yellow]"
                )
            _probe_hypotheses(spec, check, interpretation)
            if typer.confirm("\n  apply this?", default=True):
                decision = interpretation.decision
            else:
                decision = _manual_decision("overruled")
                # The model's reading was refused, so its revise or combine
                # payload is not the reviewer's either; it is collected again
                # below rather than carried over.
                manual = decision is not None

        # Every exit below records an outcome. A reviewer who stopped, or whose
        # answer could not be applied, interacted with the system just as much
        # as one whose decision landed; a record that kept only the successes
        # would overstate how much of the review actually concluded. Bound
        # explicitly rather than closed over, so the record cannot drift from
        # the iteration that produced it.
        turn = functools.partial(
            _record_turn,
            verifier_id=verifier_id,
            check_id=check_id,
            round_number=interview.round_number,
            shown_at=shown_at,
            answered_seconds=answered_seconds,
            shown=_displayed_context(summary, check, spec, interview, check_id),
            reply=reply,
            authoritative=authoritative,
            authoritative_why=why,
            interpretation=interpretation,
            applied=None,
            manual=manual,
            failure=failure,
        )

        if decision is None:
            console.print("[yellow]stopped[/yellow] — nothing applied for this check")
            turn("stopped", applied_decision=None)
            break

        # What the model proposed, kept whether or not it was followed: an
        # overruled reading is exactly what a later reader needs to see.
        proposed = interpretation
        applied = interpretation
        if manual:
            # A decision the reviewer entered carries no payload of its own. For
            # revise and combine that payload is the decision, so it is asked for
            # here; without it the guards below would refuse the action and leave
            # the check pending, re-asking a question the reviewer cannot answer.
            applied = _manual_interpretation(decision, check, known)

        # Rebound once `applied` exists: a turn recorded before this point had
        # no replacement to name, and one recorded after has to carry both
        # readings or an overrule loses whichever the reviewer did not take.
        turn = functools.partial(turn, applied=applied)

        if decision is InterviewDecision.COMBINE and (applied is None or not applied.combine_with):
            console.print("  [yellow]no resolved target to combine with[/yellow]; skipped")
            turn("skipped_invalid_combine", applied_decision=decision.value)
            continue

        if decision is InterviewDecision.REVISE and (
            applied is None
            or (applied.revised_expected is None and applied.revised_operator is None)
        ):
            # Reachable by overruling some other reading into a revise: the
            # interpretation on hand names nothing to revise, and applying it
            # would strip the check's evidence without changing the check.
            console.print("  [yellow]nothing named to revise[/yellow]; skipped")
            turn("skipped_empty_revision", applied_decision=decision.value)
            continue

        review = CheckReview(
            review_id=f"review-{len(interview.reviews) + 1:03d}-{check_id}",
            verifier_id=verifier_id,
            check_id=check_id,
            reply=reply,
            decision=decision,
            authoritative=authoritative,
            authoritative_why=why,
            # `apply_decision` acts on this, so a manual revise or combine records
            # the payload it acted on; every other case records what the model
            # proposed, followed or not. A failure carries neither: the validator
            # refuses an interpretation beside one, and there was no reading to keep.
            interpretation=applied
            if applied is not None
            else (proposed if failure is None else None),
            model=model,
            prompt=prompt_text or "",
            response=response or "",
            failure=failure,
        )
        previous_interview_id = envelope.artifact_id
        interview = apply_decision(interview, review)
        # Saved after every decision: the store is content-addressed, so each
        # save is its own artifact and the latest id is where a resume starts.
        envelope = save_interview(interview, store)

        # The chain, in one event: what was shown, what the reviewer said, what
        # the model read it as, whether that reading survived, and which
        # artifact the answer produced. The pieces exist scattered across the
        # draft, the interpretation and the interview; what was missing is the
        # order they happened in and whether the human agreed.
        turn(
            "applied",
            applied_decision=decision.value,
            review_id=review.review_id,
            input_artifact_id=previous_interview_id,
            output_artifact_id=envelope.artifact_id,
        )

    console.print(f"\ninterview_id: {envelope.artifact_id}")
    console.print(f"round:        {interview.round_number}")
    console.print(
        f"reviewed:     {len(interview.reviews)} of {len(interview.pending) + len(interview.reviews)}"
    )
    console.print(
        "[yellow]note:[/yellow] review refined the hypothesis; validation is still required "
        "before calibrated or reviewed status"
    )


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
        envelope = save_draft(draft, store)
        recorder.finish(
            status="awaiting_review" if draft.complete else "incomplete",
            stop_reason=draft.stop_reason.value,
            draft_id=envelope.artifact_id,
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
    console.print(taxonomy_overview(draft.contracts, members=_draft_members(draft)))
    # What the second look changed. The question a reviewer has at a pause, and
    # one no total over the whole run answers.
    console.print("")
    print_pass_history(draft, console)
    console.print(
        f"\n[dim]review the families: bandits rlm-families {envelope.artifact_id}[/dim]"
    )
    _report_unresolved(
        len(draft.ambiguous_trace_ids),
        len(draft.uncovered_trace_ids),
        len(draft.unreadable_trace_ids),
    )
    for limitation in draft.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="audit-rlm-taxonomy")
def audit_rlm_taxonomy_command(
    draft_id: str,
    model: str = typer.Option(RLM_AUDIT_MODEL, "--model"),
    freeze: bool = typer.Option(
        True, "--freeze/--no-freeze", help="Freeze the taxonomy when nothing is unresolved."
    ),
    force: bool = typer.Option(
        False, "--force", help="Freeze despite unresolved findings, recording that it was forced."
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Challenge every contract in a draft with a fresh, adversarial context."""
    store = _derived(project)
    try:
        draft = load_draft(draft_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no draft {draft_id!r}")
        raise typer.Exit(code=1) from exc

    _, corpus, _ = _rlm_corpus(draft.analysis_id, project, draft.view.value)
    try:
        predict = build_taxonomy_audit_predictor(model=model, view=draft.view)
    except TaxonomyAuditError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    audit = audit_taxonomy(draft, draft_id, corpus, predict=predict, model=model)
    audit_envelope = save_rlm_audit(audit, store)
    console.print(f"audit_id:    {audit_envelope.artifact_id} ({audit.model})")

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
    if not freeze:
        return

    try:
        taxonomy = freeze_taxonomy(
            draft, draft_id, audit=audit, audit_id=audit_envelope.artifact_id, force=force
        )
    except FreezeRefused as exc:
        # Not an error in the run: the audit did its job. Re-mine to address the
        # findings, or freeze over them deliberately with --force.
        console.print(f"\n[yellow]not frozen:[/yellow] {exc}")
        console.print("[dim]re-mine to address them, or pass --force to freeze anyway[/dim]")
        return

    envelope = save_taxonomy(taxonomy, store)
    console.print(f"\ntaxonomy_id: {envelope.artifact_id}")
    for limitation in taxonomy.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="assign-rlm-taxonomy")
def assign_rlm_taxonomy_command(
    taxonomy_id: str,
    batch_size: int = typer.Option(20, "--batch-size"),
    model: str = typer.Option(RLM_ASSIGN_MODEL, "--model"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Classify every trace against a frozen taxonomy in a fresh context."""
    store = _derived(project)
    try:
        taxonomy = load_taxonomy(taxonomy_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no taxonomy {taxonomy_id!r}")
        raise typer.Exit(code=1) from exc

    _, corpus, _ = _rlm_corpus(taxonomy.analysis_id, project, taxonomy.view.value)
    try:
        predict = build_assignment_predictor(model=model, view=taxonomy.view)
    except AssignmentError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    with ledger.stage("rlm_assignment_run", taxonomy_id=taxonomy_id, model=model):
        run = assign_traces(
            taxonomy,
            taxonomy_id,
            corpus,
            predict=predict,
            model=model,
            batch_size=batch_size,
        )
        envelope = save_assignment_run(run, store)
        ledger.record(
            {
                "event_type": "stage_complete",
                "stage_name": "rlm_assignment_run",
                "input_artifact_id": taxonomy_id,
                "output_artifact_id": envelope.artifact_id,
                "assigned": len(run.by_status(AssignmentStatus.ASSIGNED)),
            }
        )

    console.print(f"run_id:      {envelope.artifact_id}")
    console.print(f"assigned:    {len(run.by_status(AssignmentStatus.ASSIGNED))}")
    for contract_id, traces in run.members().items():
        console.print(f"  {contract_id}  {len(traces)} trace(s)")
    _report_unresolved(
        len(run.by_status(AssignmentStatus.AMBIGUOUS)),
        len(run.by_status(AssignmentStatus.UNCOVERED)),
        len(run.by_status(AssignmentStatus.UNREADABLE)),
    )
    for limitation in run.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="validate-rlm-mining")
def validate_rlm_mining_command(
    run_ids: list[str] = typer.Argument(..., help="Two or more assignment run ids."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Compare independent runs by trace co-assignment, never by family name."""
    store = _derived(project)
    runs = []
    taxonomies = []
    for run_id in run_ids:
        try:
            run = load_assignment_run(run_id, store)
        except FileNotFoundError as exc:
            console.print(f"[red]error:[/red] no assignment run {run_id!r}")
            raise typer.Exit(code=1) from exc
        runs.append(run)
        try:
            taxonomies.append(load_taxonomy(run.taxonomy_id, store))
        except FileNotFoundError:
            # Recurrence is measured over whatever taxonomies are present, and
            # the report says when it saw fewer than there are runs.
            pass

    try:
        report = compare_runs(
            runs,
            run_ids,
            analysis_id=runs[0].analysis_id,
            taxonomies=taxonomies,
        )
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_stability_report(report, store)
    console.print(f"report_id:   {envelope.artifact_id}")
    console.print(f"runs:        {report.runs}")
    console.print(f"stable:      {report.stable_assignment_fraction:.1%} of classified traces")
    console.print(f"agreement:   {report.pairwise_agreement:.1%} mean pairwise co-assignment")
    console.print(f"recurring:   {len(report.recurring_contracts)} contract(s) in >1 run")

    if report.disagreements:
        console.print(f"\n[yellow]{len(report.disagreements)} contested pair(s)[/yellow]")
        for pair in report.disagreements[:10]:
            left, right = pair.trace_ids
            console.print(
                f"  {left} / {right}: together in {pair.together}, apart in {pair.apart}"
            )
    always = [u for u in report.unplaced if u.always_unplaced]
    if always:
        console.print(
            f"\n[yellow]{len(always)} trace(s) no run could place[/yellow] "
            "[dim](a gap in the taxonomy, or a request that is not one task)[/dim]"
        )
    for limitation in report.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


@app.command(name="materialize-rlm-taskset")
def materialize_rlm_taskset_command(
    run_id: str,
    held_out: float = typer.Option(DEFAULT_HELD_OUT, "--held-out"),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Turn confidently assigned traces into a TaskSet, with honest provenance."""
    store = _derived(project)
    try:
        run = load_assignment_run(run_id, store)
        taxonomy = load_taxonomy(run.taxonomy_id, store)
        analysis = load_analysis(run.analysis_id, store)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        task_set = materialize_task_set(
            run, taxonomy, corpus_id=analysis.corpus_id, held_out=held_out
        )
    except MaterializationError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_task_set(task_set, store)
    _report(task_set, envelope.artifact_id)
    # Said plainly, because a TaskSet from this path carries no measured
    # geometry and every reader downstream is used to one that does.
    console.print(
        "[dim]families here were proposed by a model reading user requests; no "
        "embedding distance was computed, so none carries a coherence figure[/dim]"
    )


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
    """Inspect a mining session, including one still running."""
    store = SessionStore(project / ".bandits")
    if watch:
        if session_id is None:
            # Watching means watching one run. Defaulting to the newest is what
            # a person means by "watch it" right after starting a run.
            latest = store.list()
            if not latest:
                console.print("[dim]no mining sessions[/dim]")
                return
            session_id = latest[0].session_id
        _watch_session(store, session_id, interval)
        return
    if session_id is None:
        sessions = store.list()
        if not sessions:
            console.print("[dim]no mining sessions[/dim]")
            return
        table = Table("session", "status", "progress", "updated")
        for state in sessions:
            colour = {"running": "cyan", "awaiting_review": "green", "failed": "red"}.get(
                state.status, "yellow"
            )
            table.add_row(
                state.session_id,
                f"[{colour}]{state.status}[/{colour}]",
                state.progress,
                state.updated_at[:19],
            )
        console.print(table)
        return

    try:
        state = store.read(session_id)
    except FileNotFoundError as exc:
        console.print(f"[red]error:[/red] no session {session_id!r}")
        raise typer.Exit(code=1) from exc

    console.print(f"session:     {state.session_id}")
    console.print(f"status:      {state.status}")
    console.print(f"view:        {state.view.value} (seed {state.seed})")
    console.print(f"progress:    {state.progress}")
    console.print(
        f"passes:      {state.completed_passes}/{state.requested_passes} complete, "
        f"pass {state.pass_index + 1} in flight"
    )
    console.print(f"assigned:    {len(state.assignments)}")
    _report_unresolved(
        len(state.ambiguous_trace_ids), len(state.uncovered_trace_ids), 0
    )
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


def _watch_session(store, session_id: str, interval: float) -> None:
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

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            live.update(live_panel(state, recent=store.read_events(session_id, limit=6)))
            if state.status != "running":
                break
            time.sleep(interval)
            try:
                state = store.read(session_id)
            except (FileNotFoundError, ValueError):
                # A read landing mid-write is expected: the writer replaces the
                # file atomically, so the next poll gets a whole one.
                continue

    if state.status == "awaiting_review":
        console.print(
            "\n[green]paused for review.[/green] "
            "[dim]families: bandits rlm-families <draft_id>[/dim]"
        )


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
        draft = load_draft(draft_id, store)
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

    members = _draft_members(draft)
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
