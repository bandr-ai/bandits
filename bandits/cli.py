"""Command-line interface: ingest a trace export, then inspect what's stored."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bandits.analyze import (
    DEFAULT_BUDGET,
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
    save_cache,
)
from bandits.export import (
    Partition,
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
    draft_verifiers,
    load_reviewed_verifier,
    load_verifier_draft,
    next_question,
    review_verifier,
    run_draft,
    save_draft_run,
    save_interview,
    save_reviewed_verifier,
    save_verifier_draft,
    start_interview,
)
from bandits.verify.judge import (
    DEFAULT_MODEL,
    JudgeError,
    Rubric,
    judge_traces,
    save_judge_run,
)
from bandits.verify.validate import (
    load_validation,
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
    wanted = descriptors(analysis)
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
    for limitation in task_set.limitations:
        console.print(f"[yellow]limitation:[/yellow] {limitation}")


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
    embedding_model: str = typer.Option(
        EMBEDDING_MODEL, "--embedding-model", help="Fireworks embedding model."
    ),
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
        proposed_by="model",
    )
    envelope = save_task_set(task_set, store)
    _report(task_set, envelope.artifact_id)
    console.print(f"embeddings:  {cache_id} ({len(cache.vectors)} vectors, {embedding_model})")


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


@app.command(name="merge-families")
def merge_families_command(
    task_set_id: str,
    family_ids: list[str] = typer.Argument(..., help="Two or more families that are one task."),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Record a reviewer's decision that several families are the same task."""
    task_set = _load_task_set(task_set_id, project)
    try:
        corrected = merge_families(task_set, tuple(family_ids))
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_task_set(corrected, _derived(project))
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
        draft = draft_verifiers(task_set, task_set_id, analysis, family_id, limit=limit)
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
    for unresolved in draft.unresolved:
        console.print(f"[yellow]unresolved:[/yellow] {unresolved}")

    # A drafted check is a hypothesis. Run it before anyone is asked about it.
    run = run_draft(draft, analysis, task_set)
    save_draft_run(run, store)
    _show_draft_run(run)

    if interview:
        _run_verifier_interview(draft, envelope.artifact_id, store)


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
            draft, verifier_draft_id, task_set, analysis.evidence, label_set, label_set_id
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
    acceptance_id: str = typer.Option(
        ..., "--acceptance-id", help="Ticket, review record, or owner decision id."
    ),
    accept_risks: bool = typer.Option(
        False,
        "--accept-risks",
        help="Promote despite blockers, recording them on the artifact.",
    ),
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Record explicit human acceptance of one calibrated verifier."""
    store = _derived(project)
    try:
        draft = load_verifier_draft(verifier_draft_id, store)
        validation = load_validation(validation_id, store)
        reviewed = review_verifier(
            draft,
            verifier_draft_id,
            validation,
            validation_id,
            verifier_id,
            acceptance_id,
            accept_risks=accept_risks,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    envelope = save_reviewed_verifier(reviewed, store)
    console.print(f"reviewed_verifier_id: {envelope.artifact_id}")
    console.print(f"verifier:             {reviewed.spec.verifier_id}")
    console.print(f"status:               {reviewed.spec.status.value}")
    console.print(f"validation:           {reviewed.validation_id}")
    console.print(f"acceptance:           {reviewed.human_acceptance_id}")
    console.print(f"threshold:            {reviewed.success_threshold}")
    for risk in reviewed.accepted_risks:
        console.print(f"[yellow]accepted risk:[/yellow] {risk.code} — {risk.detail}")


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
    project: Path = typer.Option(_DEFAULT_PROJECT, "--project"),
) -> None:
    """Write reviewed eval or SFT rows plus an unresolved quarantine file."""
    if format_ not in {"eval", "sft"}:
        console.print("[red]error:[/red] --format must be one of: eval, sft")
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
        builder = build_eval_export if format_ == "eval" else build_sft_export
        bundle = builder(
            corpus,
            task_set,
            task_set_id,
            analysis,
            reviewed,
            reviewed_verifier_id,
            partition=partition,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    envelope = save_export(bundle, store)
    accepted_path, unresolved_path = write_jsonl(bundle, output)
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


if __name__ == "__main__":
    app()
