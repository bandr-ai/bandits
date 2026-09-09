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
import random
import sys
from pathlib import Path

from bandits.analyze.rlm_assign import assign_traces
from bandits.analyze.rlm_assign import build_predictor as build_assigner
from bandits.analyze.rlm_audit import (
    audit_taxonomy,
    compute_taxonomy_id,
    freeze_taxonomy,
)
from bandits.analyze.rlm_audit import (
    build_predictor as build_auditor,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_mine import build_predictor, mine_taxonomy
from bandits.analyze.rlm_models import AssignmentStatus, Budget, TraceView
from bandits.analyze.rlm_taskset import materialize_task_set
from bandits.store import ArtifactStore


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
        "--lineages", type=int, default=3, help="Task lineages to sample (all trials of each)."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--view", default=TraceView.USER_MESSAGES.value)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--max-llm-calls", type=int, default=40)
    args = parser.parse_args()

    view = TraceView(args.view)
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj, corpus_id, lineages = _load_corpus(
        args.project, args.corpus, args.lineages, args.seed
    )
    analysis = analyze_corpus(corpus_obj)
    corpus = ReadOnlyCorpus(corpus_obj, view=view)
    print(f"\ncorpus: {corpus_id}")
    print(f"  {len(corpus_obj.traces)} trace(s) from {len(lineages)} lineage(s): "
          f"{', '.join(lineages)}")
    for trace in corpus_obj.traces[:3]:
        first = trace.user_turns[0].text if trace.user_turns else ""
        print(f"  {trace.trace_id}: {' '.join(first.split())[:88]}")

    kwargs = {"model": args.model} if args.model else {}
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
    draft = mine_taxonomy(
        corpus,
        "smoke-analysis",
        predict=build_predictor(**kwargs),
        analysis=analysis,
        chunk_size=3,
        budget=Budget(
            passes=2, max_iterations=20, max_llm_calls=args.max_llm_calls, max_usd=args.max_usd
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
    print(f"  contracts: {[(c.contract_id, c.name) for c in draft.contracts]}")
    print(
        f"  stop_reason: {draft.stop_reason.value} "
        f"(passes {draft.completed_passes}/{draft.requested_passes})"
    )

    if not draft.contracts:
        print("\nno contracts: skipping the remaining stages")
        return 1

    print("\n== adversarial audit ==")
    audit = audit_taxonomy(draft, "smoke-draft", corpus, predict=build_auditor(**kwargs))
    check("every contract was challenged", len(audit.findings) == len(draft.contracts))
    for finding in audit.findings:
        print(f"  {finding.contract_id}: {finding.recommendation} — {finding.rationale[:90]}")

    taxonomy = freeze_taxonomy(draft, "smoke-draft", audit=audit, audit_id="smoke-audit", force=True)
    taxonomy_id = compute_taxonomy_id(taxonomy)

    print("\n== fresh assignment ==")
    run = assign_traces(
        taxonomy, taxonomy_id, corpus, predict=build_assigner(**kwargs), batch_size=6
    )
    assigned = run.by_status(AssignmentStatus.ASSIGNED)
    check("something was assigned", bool(assigned))
    check(
        "no trace was silently dropped",
        len(run.assignments) == corpus.count_traces(),
        f"{len(run.assignments)} of {corpus.count_traces()}",
    )
    for contract_id, traces in run.members().items():
        print(f"  {contract_id}: {list(traces)}")

    # The substantive check, and the one worth paying for: the four trials of a
    # single task are the same request, so a taxonomy that scatters them across
    # families is not finding tasks. This is the real corpus's own ground truth,
    # and it is the only labelled signal used anywhere in this script.
    groups = [set(v) for v in run.members().values()]
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

    if assigned:
        print("\n== materialization ==")
        task_set = materialize_task_set(run, taxonomy, corpus_id=analysis.corpus_id)
        assert task_set.clustering is not None
        check(
            "the task set names the arm that produced it",
            task_set.clustering.backend == f"rlm-{view.value}",
            task_set.clustering.backend,
        )
        check(
            "coverage is not overstated",
            task_set.workload_coverage <= 1.0
            and task_set.workload_coverage
            == len(assigned) / max(len([a for a in run.assignments
                                        if a.status is not AssignmentStatus.UNREADABLE]), 1),
            f"{task_set.workload_coverage:.1%}",
        )

    spent = sum(c.cost_usd or 0.0 for c in draft.chunks)
    print(f"\n== spent on discovery: ${spent:.4f} ==")
    for limitation in draft.limitations:
        print(f"  limitation: {limitation}")

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
