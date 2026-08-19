"""Triage must be right about the one question it exists to answer.

Two failure directions, and they are not symmetric. A false NO_GO costs a deal.
A false GO costs six weeks of a deployment and the customer's trust, because we
will have promised a reward that the data could never support. So the NO_GO
cases here are the load-bearing ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.contracts import (
    CallStatus,
    InvocationPoint,
    Message,
    Trace,
    TraceCorpus,
)
from bandits.ingest import load_corpus
from bandits.triage import Verdict, triage_corpus

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@pytest.fixture(scope="module")
def golden() -> TraceCorpus:
    return load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp")


def _trace(trace_id: str, invocations: tuple[InvocationPoint, ...], *, instruction: str = "do the thing") -> Trace:
    return Trace(
        trace_id=trace_id,
        source="test",
        source_digest="0" * 64,
        messages=(Message(role="user", content=instruction),),
        invocations=invocations,
    )


def _call(trace_id: str, step: int, tool: str, arguments: dict, response: object) -> InvocationPoint:
    return InvocationPoint(
        call_id=f"{trace_id}:{step}",
        trace_id=trace_id,
        step=step,
        tool=tool,
        arguments=arguments,
        response=response,
    )


# -- the golden corpus is buildable ----------------------------------------


def test_golden_corpus_is_go(golden: TraceCorpus) -> None:
    """The fixture corpus is the shape we claim an environment can be built from."""
    report = triage_corpus(golden)
    assert report.verdict is Verdict.GO
    assert report.blocking_failures == ()


def test_golden_corpus_finds_state_changes(golden: TraceCorpus) -> None:
    """A refund changes an order's status, and that is what a verifier asserts on."""
    signal = triage_corpus(golden).signal("state_changes")
    assert signal is not None
    assert signal.present
    assert signal.observed > 0


def test_every_golden_tool_is_reconstructible(golden: TraceCorpus) -> None:
    report = triage_corpus(golden)
    assert report.tools
    assert [t.tool for t in report.tools if not t.reconstructible] == []


def test_identifiers_are_attributed_to_the_tools_that_return_them(golden: TraceCorpus) -> None:
    """Response-side attribution: get_order returns order_id, so it is entity evidence."""
    report = triage_corpus(golden)
    get_order = next(t for t in report.tools if t.tool == "get_order")
    assert "order_id" in get_order.identifier_fields


# -- the cases that must be refused ----------------------------------------


def test_transcript_only_corpus_is_no_go() -> None:
    """The whole product thesis: no invocation points, no environment. Say so."""
    corpus = TraceCorpus(
        source="chat-json",
        traces=tuple(_trace(f"t{i}", ()) for i in range(5)),
    )
    report = triage_corpus(corpus)
    assert report.verdict is Verdict.NO_GO
    assert "tool call" in report.reasons[0]


def test_no_go_on_transcripts_names_only_the_real_problem() -> None:
    """Downstream signals are noise when there is nothing to have arguments of."""
    corpus = TraceCorpus(source="chat-json", traces=tuple(_trace(f"t{i}", ()) for i in range(5)))
    assert len(triage_corpus(corpus).reasons) == 1


def test_prose_responses_are_not_reconstructible() -> None:
    """A tool that only ever returned text has no fields to build a table from."""
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", tuple(
                _call("t1", i, "ask_docs", {"q": "refund policy"}, "Refunds take 5 days.")
                for i in range(4)
            )),
        ),
    )
    report = triage_corpus(corpus)
    assert report.verdict is Verdict.NO_GO
    assert not report.tools[0].reconstructible
    assert report.tools[0].note == "no structured response"


def test_responses_without_recurring_ids_are_no_go() -> None:
    """Structured but unlinkable: every response is a fresh blob, so there are no rows."""
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", tuple(
                _call("t1", i, "summarize", {"text": f"doc {i}"}, {"summary": f"a summary {i}"})
                for i in range(6)
            )),
        ),
    )
    report = triage_corpus(corpus)
    assert report.verdict is Verdict.NO_GO
    assert not report.signal("identifiers").present


def test_read_only_corpus_is_partial_not_go() -> None:
    """Reconstructible, but nothing ever changes -- so there is nothing to reward."""
    rows = [{"order_id": 1, "status": "shipped"}, {"order_id": 2, "status": "shipped"}]
    corpus = TraceCorpus(
        source="test",
        traces=tuple(
            _trace(f"t{i}", tuple(
                _call(f"t{i}", step, "get_order", {"order_id": row["order_id"]}, dict(row))
                for step, row in enumerate(rows)
            ))
            for i in range(3)
        ),
    )
    report = triage_corpus(corpus)
    assert report.verdict is Verdict.PARTIAL
    assert not report.signal("state_changes").present
    assert report.signal("identifiers").present


def test_state_change_denominator_excludes_rows_seen_once() -> None:
    """A short corpus is not a read-only corpus. Don't count rows that cannot testify."""
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", (
                _call("t1", 0, "get_order", {"order_id": 1}, {"order_id": 1, "status": "shipped"}),
                _call("t1", 1, "get_order", {"order_id": 2}, {"order_id": 2, "status": "shipped"}),
                _call("t1", 2, "get_order", {"order_id": 3}, {"order_id": 3, "status": "shipped"}),
                _call("t1", 3, "get_order", {"order_id": 1}, {"order_id": 1, "status": "refunded"}),
            )),
        ),
    )
    signal = triage_corpus(corpus).signal("state_changes")
    assert (signal.observed, signal.population) == (1, 1)


def test_ids_only_ever_nested_are_not_nominated() -> None:
    """Triage inherits stage 3's top-level nomination rule, and must not outrun it.

    A key that never appears as a top-level response scalar is not nominated by
    :func:`find_identifiers`, so reconstruction will find no entity for it. Triage
    saying GO here would be the expensive error: a promise the pipeline cannot keep.
    """
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", (
                _call("t1", 0, "get_order", {"order_id": 9}, {"order": {"order_id": 9, "status": "open"}}),
                _call("t1", 1, "get_order", {"order_id": 9}, {"order": {"order_id": 9, "status": "closed"}}),
                _call("t1", 2, "get_order", {"order_id": 9}, {"order": {"order_id": 9, "status": "closed"}}),
            )),
        ),
    )
    report = triage_corpus(corpus)
    assert not report.signal("identifiers").present
    assert report.verdict is Verdict.NO_GO


def test_nested_rows_count_once_the_key_is_established() -> None:
    """Mixed wrappers are normal, and must not hide a state change.

    Once ``order_id`` is nominated from a top-level response scalar, the same row
    seen inside an envelope is the same row. Requiring one consistent wrapper would
    mean missing changes for no reason other than a tool's return style.
    """
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", (
                _call("t1", 0, "search", {"order_id": 9}, {"order_id": 9, "status": "open"}),
                _call("t1", 1, "get_order", {"order_id": 9}, {"order": {"order_id": 9, "status": "closed"}}),
                _call("t1", 2, "get_order", {"order_id": 9}, {"order": {"order_id": 9, "status": "closed"}}),
            )),
        ),
    )
    assert triage_corpus(corpus).signal("state_changes").present


# -- report mechanics ------------------------------------------------------


def test_non_blocking_signals_never_move_the_verdict() -> None:
    """Missing instructions are a nuisance, not a disqualification: write them by hand."""
    rows = [
        {"order_id": 1, "status": "shipped"},
        {"order_id": 1, "status": "refunded"},
        {"order_id": 2, "status": "open"},
    ]
    traces = tuple(
        Trace(
            trace_id=f"t{i}",
            source="test",
            source_digest="0" * 64,
            messages=(),
            invocations=tuple(
                _call(f"t{i}", step, "get_order", {"order_id": row["order_id"]}, dict(row))
                for step, row in enumerate(rows)
            ),
        )
        for i in range(3)
    )
    report = triage_corpus(TraceCorpus(source="test", traces=traces))
    assert not report.signal("instructions").present
    assert not report.signal("outcome_labels").present
    assert report.verdict is not Verdict.NO_GO


def test_error_modes_signal_reads_recorded_failures() -> None:
    corpus = TraceCorpus(
        source="test",
        traces=(
            _trace("t1", (
                InvocationPoint(
                    call_id="c0", trace_id="t1", step=0, tool="get_order",
                    arguments={"order_id": 99}, response={"error": "not_found"},
                    status=CallStatus.ERROR, error_kind="not_found",
                ),
            )),
        ),
    )
    assert triage_corpus(corpus).signal("error_modes").present


def test_tools_are_ordered_by_call_volume(golden: TraceCorpus) -> None:
    """The tool worth fixing first is the one the agent actually leans on."""
    counts = [t.calls for t in triage_corpus(golden).tools]
    assert counts == sorted(counts, reverse=True)


def test_report_serializes(golden: TraceCorpus) -> None:
    payload = triage_corpus(golden).to_json()
    assert payload["verdict"] == "GO"
    assert payload["signals"] and payload["tools"]
    assert all("detail" in signal for signal in payload["signals"])


def test_empty_corpus_is_no_go_without_crashing() -> None:
    report = triage_corpus(TraceCorpus(source="test", traces=()))
    assert report.verdict is Verdict.NO_GO
