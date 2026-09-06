from __future__ import annotations

from pathlib import Path

from bandits.ingest.otlp import load_otlp
from bandits.traces import SpanKind, SpanStatus

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces.otlp.jsonl"


def test_groups_spans_into_traces() -> None:
    corpus = load_otlp(FIXTURE)

    assert corpus.source == "otlp"
    assert {t.trace_id for t in corpus.traces} == {"trace-1", "trace-2"}

    trace_1 = next(t for t in corpus.traces if t.trace_id == "trace-1")
    assert trace_1.task == "Refund order 7741"
    assert [s.span_id for s in trace_1.spans] == ["span-1", "span-2"]
    assert trace_1.spans[1].parent_span_id == "span-1"
    assert trace_1.spans[1].kind is SpanKind.TOOL
    assert trace_1.spans[1].arguments == {"order_id": "7741"}
    assert trace_1.spans[1].output == {"status": "refunded"}


def test_error_status_is_captured() -> None:
    corpus = load_otlp(FIXTURE)
    trace_2 = next(t for t in corpus.traces if t.trace_id == "trace-2")
    tool_span = next(s for s in trace_2.spans if s.kind is SpanKind.TOOL)
    assert tool_span.status is SpanStatus.ERROR


def test_malformed_line_becomes_an_issue_not_a_failure() -> None:
    corpus = load_otlp(FIXTURE)
    assert len(corpus.issues) == 1
    assert corpus.issues[0].kind == "malformed_json"
    # the well-formed record on the line after it still loads
    assert any(t.trace_id == "trace-2" and len(t.spans) == 2 for t in corpus.traces)


def test_all_traces_share_the_file_digest() -> None:
    corpus = load_otlp(FIXTURE)
    digests = {t.source_digest for t in corpus.traces}
    assert len(digests) == 1


def _root(tools: str, extra: str = "") -> str:
    return (
        '{"trace_id": "t-1", "span_id": "s-1", "parent_span_id": null, "name": "gpt-5",'
        ' "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:00:01Z",'
        ' "attributes": {"gen_ai.operation.name": "chat", "task": "Refund order 7741",'
        ' "gen_ai.request.model": "gpt-5", ' + tools + extra + "}}"
    )


def test_the_root_span_declares_the_offered_toolset(tmp_path) -> None:
    path = tmp_path / "tools.jsonl"
    path.write_text(
        _root(
            '"gen_ai.request.tools": [{"name": "refund_order", "parameters": {"type": "object"}}],',
            ' "gen_ai.system_instructions": "You are a support agent."',
        )
        + "\n"
    )

    trace = load_otlp(path).traces[0]

    assert trace.tools_available is not None
    assert [tool.name for tool in trace.tools_available] == ["refund_order"]
    assert trace.system_prompt == "You are a support agent."
    assert trace.runtime_context == {"gen_ai.request.model": "gpt-5"}


def test_a_toolset_declared_mid_episode_is_not_read_as_the_offered_one(tmp_path) -> None:
    """By then the toolset has already been narrowed by what the agent learned."""
    path = tmp_path / "late.jsonl"
    path.write_text(
        _root('"note": "no tools here",')
        + "\n"
        + '{"trace_id": "t-1", "span_id": "s-2", "parent_span_id": "s-1", "name": "refund_order",'
        ' "start_time": "2026-01-01T00:00:02Z", "end_time": "2026-01-01T00:00:03Z",'
        ' "attributes": {"gen_ai.operation.name": "execute_tool",'
        ' "gen_ai.request.tools": [{"name": "refund_order"}]}}\n'
    )

    assert load_otlp(path).traces[0].tools_available is None
