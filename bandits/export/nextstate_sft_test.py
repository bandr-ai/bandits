from __future__ import annotations

from datetime import UTC, datetime

from bandits.export.nextstate_sft import build_nextstate_sft_export
from bandits.traces import Span, SpanKind, SpanStatus, Trace
from bandits.verify.nextstate import Archetype
from bandits.verify.propose import (
    CheckStats,
    FamilyCheck,
    FamilyVerifier,
    FlaggedTurn,
    TraceScore,
    VerifierScores,
)

_T = datetime(2025, 1, 1, tzinfo=UTC)


def _span(span_id: str, kind: SpanKind, name: str, output, status=SpanStatus.OK) -> Span:
    return Span(
        span_id=span_id, kind=kind, name=name, started_at=_T, ended_at=_T, output=output, status=status
    )


def _trace(trace_id: str, task: str, spans) -> Trace:
    return Trace(
        trace_id=trace_id, source="test", source_digest="d", task=task, spans=tuple(spans)
    )


def _verifier(checks=()) -> FamilyVerifier:
    return FamilyVerifier(
        family_id="fam-1",
        archetype=Archetype.COMPUTER_USE,
        corpus_id="corpus-1",
        judge_run_id="judge-1",
        model="m",
        prompt_digest="d",
        checks=tuple(checks),
    )


def _check(name: str, decision: str) -> FamilyCheck:
    stats = CheckStats(turns=10, fired=3, fired_scored=3, fired_negative=3, fired_positive=0, negatives=3)
    return FamilyCheck(
        check_id=f"{name}-000",
        name=name,
        hypothesis="h",
        code="def check(turn): return False",
        code_digest="d",
        stats=stats,
        survived=True,
        reason="ok",
        decision=decision,
    )


def test_passing_trace_becomes_a_positive_row() -> None:
    trace = _trace(
        "t1", "do the thing", [_span("m1", SpanKind.MODEL, "gpt", "done"), _span("m2", SpanKind.MODEL, "gpt", "ok")]
    )
    verifier = _verifier()
    scores = VerifierScores(
        verifier_id="fv-1",
        judge_run_id="judge-1",
        include_judge=True,
        checks_applied=(),
        scores=(TraceScore(trace_id="t1", turns=1, observed=1, flagged=()),),
    )
    bundle = build_nextstate_sft_export([trace], verifier, "fv-1", scores, "scores-1")
    assert bundle.positive == 1
    assert bundle.negative == 0
    assert bundle.rows[0].label == "positive"
    assert bundle.rows[0].flagged_turns == ()


def test_flagged_trace_becomes_a_negative_row_with_reasons() -> None:
    trace = _trace(
        "t1",
        "do the thing",
        [_span("m1", SpanKind.MODEL, "gpt", "boom"), _span("m2", SpanKind.MODEL, "gpt", "ok")],
    )
    verifier = _verifier([_check("has-error", "accepted")])
    scores = VerifierScores(
        verifier_id="fv-1",
        judge_run_id="judge-1",
        include_judge=True,
        checks_applied=("has-error",),
        scores=(
            TraceScore(
                trace_id="t1",
                turns=1,
                observed=1,
                flagged=(FlaggedTurn(index=0, by=("has-error",)),),
            ),
        ),
    )
    bundle = build_nextstate_sft_export([trace], verifier, "fv-1", scores, "scores-1")
    assert bundle.negative == 1
    row = bundle.rows[0]
    assert row.label == "negative"
    assert row.flagged_turns == (0,)
    assert row.flagged_by == {"0": ("has-error",)}
    assert row.all_checks_reviewed is True


def test_unreviewed_check_is_flagged_on_every_row() -> None:
    trace = _trace("t1", "do the thing", [_span("m1", SpanKind.MODEL, "gpt", "done")])
    verifier = _verifier([_check("survivor-only", "pending")])
    scores = VerifierScores(
        verifier_id="fv-1",
        judge_run_id="judge-1",
        include_judge=False,
        checks_applied=("survivor-only",),
        scores=(TraceScore(trace_id="t1", turns=1, observed=1, flagged=()),),
    )
    bundle = build_nextstate_sft_export([trace], verifier, "fv-1", scores, "scores-1")
    assert bundle.all_checks_reviewed is False
    assert bundle.rows[0].all_checks_reviewed is False


def test_trace_missing_from_the_corpus_is_quarantined() -> None:
    verifier = _verifier()
    scores = VerifierScores(
        verifier_id="fv-1",
        judge_run_id="judge-1",
        include_judge=True,
        checks_applied=(),
        scores=(TraceScore(trace_id="gone", turns=1, observed=1, flagged=()),),
    )
    bundle = build_nextstate_sft_export([], verifier, "fv-1", scores, "scores-1")
    assert bundle.rows == ()
    assert len(bundle.unresolved) == 1
    assert bundle.unresolved[0].trace_id == "gone"


def test_no_observed_turns_is_quarantined_not_scored() -> None:
    trace = _trace("t1", "do the thing", [_span("m1", SpanKind.MODEL, "gpt", "done")])
    verifier = _verifier()
    scores = VerifierScores(
        verifier_id="fv-1",
        judge_run_id="judge-1",
        include_judge=True,
        checks_applied=(),
        scores=(TraceScore(trace_id="t1", turns=1, observed=0, flagged=()),),
    )
    bundle = build_nextstate_sft_export([trace], verifier, "fv-1", scores, "scores-1")
    assert bundle.rows == ()
    assert bundle.unresolved[0].reasons == ("no observed turns; the verifier scored nothing",)
