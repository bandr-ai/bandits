from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, TraceIssue


def _span(**overrides: object) -> Span:
    defaults = dict(
        span_id="span-1",
        parent_span_id=None,
        kind=SpanKind.MODEL,
        name="gpt-5",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Span(**defaults)


def test_round_trips_through_json() -> None:
    model_span = _span()
    tool_span = _span(
        span_id="span-2",
        parent_span_id="span-1",
        kind=SpanKind.TOOL,
        name="lookup_order",
        arguments={"order_id": "7741"},
        output={"status": "refunded"},
        status=SpanStatus.OK,
    )
    trace = Trace(
        trace_id="trace-1",
        source="otlp",
        source_digest="a" * 64,
        task="Refund order 7741",
        spans=(model_span, tool_span),
    )
    corpus = TraceCorpus(source="otlp", traces=(trace,), issues=(TraceIssue(kind="x", detail="y"),))

    restored = TraceCorpus.model_validate_json(corpus.model_dump_json())

    assert restored == corpus
    assert restored.traces[0].spans[1].parent_span_id == "span-1"


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Span.model_validate(
            {
                "span_id": "span-1",
                "kind": "model",
                "name": "gpt-5",
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:00:01Z",
                "unexpected_field": True,
            }
        )
