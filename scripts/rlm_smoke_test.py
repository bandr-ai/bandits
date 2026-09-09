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
import sys
from datetime import UTC, datetime

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
from bandits.traces import Span, SpanKind, Trace, TraceCorpus, UserTurn

_MOMENT = datetime(2024, 1, 1, tzinfo=UTC)

# Two families a competent reader cannot confuse, so a failure here is a failure
# of the machinery rather than a hard judgement call. Both carry tool spans so
# the full-trajectory arm has something to read, and both carry a planted score
# so the redaction and leakage checks have something to catch.
_REQUESTS = [
    ("refund-1", "I want a refund for order A-1001, it arrived broken.", "refund"),
    ("refund-2", "Please refund order B-2002. Wrong item shipped.", "refund"),
    ("refund-3", "Can you give me my money back for order C-3003?", "refund"),
    ("flight-1", "Book me a flight from SFO to Tokyo on the 3rd.", "book_flight"),
    ("flight-2", "I need a one-way ticket to Berlin next Tuesday.", "book_flight"),
    ("flight-3", "Reserve a seat on the morning flight to Delhi.", "book_flight"),
]


def _corpus() -> TraceCorpus:
    traces = []
    for trace_id, message, tool in _REQUESTS:
        traces.append(
            Trace(
                trace_id=trace_id,
                source="chat-json",
                source_digest="0" * 64,
                task=message,
                user_turns=(UserTurn(text=message),),
                spans=(
                    Span(
                        span_id=f"{trace_id}:s1",
                        kind=SpanKind.TOOL,
                        name=tool,
                        started_at=_MOMENT,
                        ended_at=_MOMENT,
                        arguments={"request": message},
                        # The planted outcome. Redaction must remove it and the
                        # leakage audit must confirm it never reached the miner.
                        output={"confirmation": f"{trace_id}-ok", "score": 0.9931},
                    ),
                ),
            )
        )
    return TraceCorpus(source="chat-json", traces=tuple(traces))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", default=TraceView.USER_MESSAGES.value)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-usd", type=float, default=2.0)
    parser.add_argument("--max-llm-calls", type=int, default=40)
    args = parser.parse_args()

    view = TraceView(args.view)
    corpus_obj = _corpus()
    from bandits.analyze.analysis import analyze_corpus

    analysis = analyze_corpus(corpus_obj)
    corpus = ReadOnlyCorpus(corpus_obj, view=view)

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
        budget=Budget(max_iterations=8, max_llm_calls=args.max_llm_calls, max_usd=args.max_usd),
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
    print(f"  stop_reason: {draft.stop_reason.value} (complete={draft.complete})")

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

    # The substantive check: refunds and flights are different families, and any
    # taxonomy worth running the real experiment on must separate them.
    refunds = {t for t, *_ in [(r[0],) for r in _REQUESTS] if t.startswith("refund")}
    flights = {t for t in (r[0] for r in _REQUESTS) if t.startswith("flight")}
    groups = [set(v) for v in run.members().values()]
    check(
        "refunds and flights did not land in one family",
        not any(g & refunds and g & flights for g in groups),
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
