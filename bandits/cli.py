"""Command-line interface: ingest a trace export, then inspect what's stored."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

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
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Load a trace export into the local artifact store."""
    try:
        corpus = load_corpus(path, source, ruleset_by_name(redaction))
    except (UnknownSourceError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

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
    table = Table("family_id", "semantic", "geometric", "outliers", "proposed split")
    for audit in sorted(run.audits, key=lambda a: (a.coherent, a.family_id)):
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
        )
    if run.audits:
        console.print(table)

    # Names go under the table rather than in it: a generated name is prose and
    # a column narrow enough to fit beside five others would truncate the one
    # thing that made it worth generating.
    for audit in sorted(run.audits, key=lambda a: a.family_id):
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
        console.print(f"[dim]skipped[/dim] {skip.family_id}: {skip.reason}")
    for limitation in run.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


def _run_audit(task_set, task_set_id: str, analysis, store, *, model: str, family_ids=None):
    """Audit a task set and persist the result beside it. Never rewrites it."""
    predict = build_predictor(model=model)
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
    for trace_id in queue:
        scores = run.scores_for(trace_id)
        rendered = ", ".join(
            f"{vid[:18]}={'unknown' if s is None else s}" for vid, s in sorted(scores.items())
        )
        console.print(f"[bold]{trace_id}[/bold]  verifiers: {rendered or 'not scored'}")
        answer = typer.prompt("  succeeded? [s]uccess/[f]ailure/[u]nclear/[q]uit", default="u")
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

        reply = typer.prompt("\n  what do you think?", default="", show_default=False)
        authoritative = typer.confirm(
            "  is this evidence source authoritative for the claim?", default=True
        )
        why = typer.prompt("  why", default="", show_default=False)

        known = tuple(c.check_id for s in interview.draft.verifiers for c in s.checks)
        interpretation = prompt_text = response = None
        failure = None
        manual = False
        try:
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

        if decision is None:
            console.print("[yellow]stopped[/yellow] — nothing applied for this check")
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

        if decision is InterviewDecision.COMBINE and (applied is None or not applied.combine_with):
            console.print("  [yellow]no resolved target to combine with[/yellow]; skipped")
            continue

        if decision is InterviewDecision.REVISE and (
            applied is None
            or (applied.revised_expected is None and applied.revised_operator is None)
        ):
            # Reachable by overruling some other reading into a revise: the
            # interpretation on hand names nothing to revise, and applying it
            # would strip the check's evidence without changing the check.
            console.print("  [yellow]nothing named to revise[/yellow]; skipped")
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
        interview = apply_decision(interview, review)
        # Saved after every decision: the store is content-addressed, so each
        # save is its own artifact and the latest id is where a resume starts.
        envelope = save_interview(interview, store)

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
