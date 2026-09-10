#!/usr/bin/env python3
"""A tiny paid smoke test of the RLM miner against a real model.

Why this exists. The unit suite injects predictors, so it proves the plumbing
around a model call and nothing about the call itself: whether DSPy's RLM
returns the fields the signature declares, whether the parsers survive what a
real model actually writes, whether the prompts produce contracts with outcome
shapes, and whether a provider reports a cost the budget can be enforced against.
None of that is knowable without spending money, and all of it has to work
before the 160-trace experiment is worth running.

Deliberately tiny: six synthetic traces spanning two obviously different task
families, one chunk of three, a hard call ceiling and a hard dollar ceiling. It
costs cents and answers the only question worth asking first — does this path
work at all against a real model.

    export FIREWORKS_API_KEY=...
    uv sync --extra audit
    uv run scripts/rlm_smoke_test.py            # Path U
    uv run scripts/rlm_smoke_test.py --view full-trajectory

Exits non-zero when a check fails, so it can gate the real run.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from bandits.analyze.analysis import save_analysis
from bandits.analyze.rlm_audit import (
    audit_clustering,
    save_audit,
)
from bandits.analyze.rlm_audit import (
    build_predictor as build_auditor,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_mine import (
    DEFAULT_MODEL,
    build_predictor,
    mine_taxonomy,
    save_clustering_run,
)
from bandits.analyze.rlm_models import Budget, TraceView
from bandits.analyze.rlm_session import SessionRecorder, SessionStore, new_session_id
from bandits.analyze.rlm_taskset import materialize_task_set
from bandits.analyze.tasksets import save_task_set
from bandits.store import ArtifactStore, DerivedStore


def _load_corpus(project: Path, corpus_id: str | None, lineages: int, seed: int):
    """A few real lineages from an ingested corpus, chosen reproducibly.

    Real traces rather than invented ones, because the invented ones only ever
    tested whether a model can tell "refund" from "book a flight" — which it
    can, and which says nothing about the corpus this is meant to run on. Real
    requests are long, mix several asks in one message, and carry the surface
    detail that makes two superficially similar requests need different
    verifiers. That is the case worth paying to test.

    Sampled by *lineage*, never by trace: four trials of one task are near
    duplicates, and a sample that split them would let the miner look good by
    grouping copies of the same request.
    """
    store = ArtifactStore(project / ".bandits")
    if corpus_id is None:
        corpora = store.list()
        if not corpora:
            raise SystemExit(f"no ingested corpus under {project}/.bandits")
        corpus_id = corpora[0].artifact_id
    corpus = store.read(corpus_id)

    by_lineage: dict[str, list] = {}
    for trace in corpus.traces:
        by_lineage.setdefault(trace.lineage_id or trace.trace_id, []).append(trace)

    chosen = sorted(by_lineage)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:lineages]
    traces = tuple(t for name in sorted(chosen) for t in by_lineage[name])
    return corpus.replace(traces=traces), corpus_id, sorted(chosen)


def _finalize(draft, args, recorder, derived_store) -> str:
    """Persist the draft, close the session, and report spend and limitations.

    Called before any early exit. A run that cost money and left nothing on disk
    cannot be re-read, compared against another seed, or assigned against, and a
    session left marked ``running`` reads as a run still in flight.
    """

    envelope = save_clustering_run(draft, derived_store)
    print(f"\n  draft saved: {envelope.artifact_id}")
    print(f"  read it: uv run bandits rlm-families {envelope.artifact_id} --project {args.project}")
    recorder.finish(
        status="awaiting_review" if draft.complete else "incomplete",
        stop_reason=draft.stop_reason.value,
        draft_id=envelope.artifact_id,
        completed_passes=draft.completed_passes,
    )

    spent = sum(c.cost_usd or 0.0 for c in draft.chunks)
    calls = sum(c.llm_calls or 0 for c in draft.chunks)
    print(f"  spent on discovery: ${spent:.4f} over {calls} call(s)")
    # Printed before any exit, because these are what distinguish a taxonomy
    # that came back empty because the model proposed nothing from one whose
    # proposals were rejected for missing a definition or an outcome shape.
    for limitation in draft.limitations:
        print(f"  limitation: {limitation}")
    return envelope.artifact_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("work/tau/run/proj"),
        help="Project holding an ingested corpus.",
    )
    parser.add_argument("--corpus", default=None, help="Defaults to the newest corpus.")
    parser.add_argument(
        "--lineages",
        type=int,
        default=1,
        help="Task lineages to sample (all trials of each). One answers whether a "
        "contract comes out at all; raise it to judge the taxonomy.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        default=True,
        help="Stop after mining. The question is whether a contract comes out at all.",
    )
    parser.add_argument(
        "--full",
        dest="discovery_only",
        action="store_false",
        help="Continue into the advisory audit and materialization.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Complete corpus passes. One is enough to see whether this works at all.",
    )
    parser.add_argument("--view", default=TraceView.USER_MESSAGES.value)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--max-llm-calls", type=int, default=400)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-call completion token ceiling. What counts as safe is "
        "model-specific and measured per experiment, not a harness default — "
        "omit to use build_predictor's own default.",
    )
    args = parser.parse_args()

    # Every physical model call, with its prompt, reply, tokens and the
    # provider's own cost, appended as it happens. On by default here rather
    # than opt-in: this script exists to diagnose runs, and the one artifact
    # that survives a process dying mid-chunk is the one written per call.
    if not os.environ.get("BANDITS_LEDGER"):
        ledger_path = args.project / ".bandits" / "rlm-ledger.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["BANDITS_LEDGER"] = str(ledger_path)
    print(f"  ledger: {os.environ['BANDITS_LEDGER']}")

    view = TraceView(args.view)
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj, corpus_id, lineages = _load_corpus(
        args.project, args.corpus, args.lineages, args.seed
    )
    analysis = analyze_corpus(corpus_obj)
    # control_markers belongs on the TraceCorpus artifact, declared once at
    # `bandits ingest --control-marker`, and every RLM command reads it from
    # there — this corpus predates that field, so the tau2-specific fallback
    # stays here rather than silently mining an ungoverned view. Re-ingest
    # with --control-marker '###TRANSFER###' to carry it on the artifact
    # instead and drop this override.
    control_markers = corpus_obj.control_markers or ("###TRANSFER###",)
    corpus = ReadOnlyCorpus(corpus_obj, view=view, control_markers=control_markers)
    print(f"\ncorpus: {corpus_id}")
    print(
        f"  {len(corpus_obj.traces)} trace(s) from {len(lineages)} lineage(s): "
        f"{', '.join(lineages)}"
    )
    for trace in corpus_obj.traces[:3]:
        first = trace.user_turns[0].text if trace.user_turns else ""
        print(f"  {trace.trace_id}: {' '.join(first.split())[:88]}")

    # The view travels with every predictor: each arm's instructions differ,
    # and a Path F run built with Path U wording would be neither experiment.
    kwargs: dict[str, object] = {"view": view}
    if args.model:
        kwargs["model"] = args.model
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"\n== view: {view.value} ==")
    if view.reads_agent_behavior:
        leaked = corpus.leakage_report(analysis)
        check("planted score is redacted from the view", not leaked, "; ".join(leaked[:2]))
        check("redaction recorded what it removed", "score" in corpus.withheld_fields())

    print("\n== discovery ==")
    # A session recorder even here, so a run in flight is watchable rather than a
    # terminal that has gone quiet. Without it the only sign of progress is a
    # line printed once a chunk finishes, which is a long silence when one chunk
    # is a dozen model calls.
    derived_store = DerivedStore(args.project / ".bandits")
    session_store = SessionStore(args.project / ".bandits")
    # Saved, not stamped with a made-up parent. A run written with an analysis id
    # that was never persisted cannot be audited or materialized afterwards: every
    # command downstream loads the analysis to reach the traces, and a fabricated
    # parent leaves the artifact readable but permanently orphaned.
    analysis_envelope = save_analysis(analysis, derived_store)
    analysis_id = analysis_envelope.artifact_id
    print(f"  analysis saved: {analysis_id}")
    recorder = SessionRecorder(
        session_store,
        session_id=new_session_id("smoke", view, args.seed),
        analysis_id=analysis_id,
        view=view,
        model=args.model or "default",
    )
    print(f"\n  session: {recorder.session_id}")
    # `uv run`, because the console script is only on PATH inside the project's
    # environment and a printed command that does not run is worse than none.
    print(
        f"  watch it: uv run bandits rlm-session {recorder.session_id} "
        f"--watch --project {args.project}"
    )

    draft = mine_taxonomy(
        corpus,
        analysis_id,
        predict=build_predictor(**kwargs),
        analysis=analysis,
        model=args.model or DEFAULT_MODEL,
        seed=args.seed,
        session=recorder,
        chunk_size=3,
        budget=Budget(
            passes=args.passes,
            max_iterations=40,
            max_llm_calls=args.max_llm_calls,
            max_usd=args.max_usd,
        ),
        on_chunk=lambda c: print(
            f"  chunk {c.index}: {len(c.trace_ids)} traces, {len(c.operations)} ops, "
            f"{c.llm_calls} calls, ${c.cost_usd if c.cost_usd is not None else 0:.4f}"
            f"{' FAILED: ' + c.error if c.status == 'error' else ''}"
        ),
    )
    check("the model returned at least one contract", bool(draft.contracts))
    check(
        "every contract states a required outcome shape",
        all(c.required_outcome_shape for c in draft.contracts),
    )
    check("chunks reported a real call count", any(c.llm_calls for c in draft.chunks))
    # The one that decides whether --max-usd is enforceable in the real run.
    priced = any(c.cost_usd is not None for c in draft.chunks)
    check(
        "the provider reported a cost",
        priced,
        "" if priced else "--max-usd cannot be enforced against this backend",
    )
    for contract in draft.contracts:
        print(f"\n  [kept] {contract.contract_id}  {contract.name}")
        print(f"    definition: {contract.definition}")
        for line in contract.required_outcome_shape:
            print(f"    outcome:    {line}")

    # Printed, not just stored. A contract the parser refused is the most
    # informative thing a failed run produces — it says what the model actually
    # proposed — and having to open the artifact to read it is how three runs
    # went by without anyone seeing the real output.
    rejected = [raw for chunk in draft.chunks for raw in chunk.dropped_contracts]
    if rejected:
        print(f"\n  {len(rejected)} contract(s) the parser refused:")
        for raw in rejected[:6]:
            print(f"    {raw[:220]}")
        if len(rejected) > 6:
            print(f"    … {len(rejected) - 6} more (all kept in the saved draft)")
    print(
        f"  stop_reason: {draft.stop_reason.value} "
        f"(passes {draft.completed_passes}/{draft.requested_passes})"
    )

    # Finalized here, above the no-contract exit rather than below it. The exact
    # run this script exists to catch is the one that produces no contracts, and
    # that run used to return before saving anything: no draft to re-read, a
    # session still marked running, and the limitations that would say *why* it
    # came back empty never printed. A failed run is the one most worth keeping.
    draft_id = _finalize(draft, args, recorder, derived_store)

    if not draft.contracts:
        print("\nno contracts: the stages below need a taxonomy and were skipped")
        return 1

    if args.discovery_only:
        # The whole question this run exists to answer is above. The audit and
        # materialization cost money to re-answer things the discovery output
        # already settles, so they are opt-in.
        print("\n[discovery only] pass --full to continue into audit and materialization")
        print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
        return 1 if failures else 0

    print("\n== adversarial audit ==")
    audit = audit_clustering(draft, draft_id, corpus, predict=build_auditor(**kwargs))
    # Saved, like everything else this run produces. Each finding carries the
    # auditor's raw reply, which is the only record of why a contract was told
    # to keep or split.
    audit_envelope = save_audit(audit, derived_store)
    print(f"  audit saved: {audit_envelope.artifact_id}")
    check("every contract was challenged", len(audit.findings) == len(draft.contracts))
    for finding in audit.findings:
        print(f"  {finding.contract_id}: {finding.recommendation} — {finding.rationale[:90]}")

    print("\n== materialization ==")
    # Straight from the run. Its own final assignments are the placement, so
    # there is no freeze and no second classification pass between here and a
    # task set.
    task_set = materialize_task_set(draft, analysis, held_out=0.3)
    task_set_id = save_task_set(task_set, derived_store).artifact_id
    print(f"  task set saved: {task_set_id}")
    for family in task_set.families:
        print(f"  {family.family_id}: {list(family.trace_ids)}")

    assert task_set.clustering is not None
    check(
        "the task set names the arm that produced it",
        task_set.clustering.backend == f"rlm-{view.value}",
        task_set.clustering.backend,
    )
    check(
        "coverage is not overstated",
        task_set.workload_coverage <= 1.0,
        f"{task_set.workload_coverage:.1%}",
    )
    check(
        "no family claims a measured coherence",
        all(family.coherence is None for family in task_set.families),
    )

    # The substantive check, and the one worth paying for: the four trials of a
    # single task are the same request, so a grouping that scatters them across
    # families is not finding tasks. This is the real corpus's own ground truth,
    # and it is the only labelled signal used anywhere in this script.
    groups = [set(family.trace_ids) for family in task_set.families]
    split_lineages = []
    for name in lineages:
        trials = {t.trace_id for t in corpus_obj.traces if (t.lineage_id or t.trace_id) == name}
        placed = [g & trials for g in groups if g & trials]
        if len(placed) > 1:
            split_lineages.append(f"{name} across {len(placed)} families")
    check(
        "trials of one task landed together",
        not split_lineages,
        "; ".join(split_lineages),
    )

    # A lineage on both sides of the split makes every held-out measurement a
    # measurement against a rerun of what the verifier was drafted from.
    straddling = []
    for family in task_set.families:
        fit, held = set(family.fit_trace_ids), set(family.held_out_trace_ids)
        for name in lineages:
            trials = {t.trace_id for t in corpus_obj.traces if (t.lineage_id or t.trace_id) == name}
            if trials & fit and trials & held:
                straddling.append(f"{name} in {family.family_id}")
    check("no lineage straddles the split", not straddling, "; ".join(straddling))

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
