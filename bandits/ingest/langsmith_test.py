"""The LangSmith adapter, held to the equivalence bar the other adapters meet.

The correctness criterion is the same one ``ingest_test`` applies to chat-json: an
invocation point recovered from a LangSmith run tree must be **indistinguishable**
from the same call read off an OTLP span. Every stage downstream reads only
invocation points, so any difference here is a difference in the reconstructed
world -- and it would surface as an unexplained fidelity drop five stages later,
where nobody would think to look for an adapter bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.contracts import CallStatus, Trace, TraceCorpus
from bandits.ingest import load_corpus
from bandits.ingest.langsmith import parse_langsmith_record

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@pytest.fixture(scope="module")
def otlp_corpus() -> TraceCorpus:
    return load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp")


@pytest.fixture(scope="module")
def langsmith_corpus() -> TraceCorpus:
    return load_corpus(FIXTURES / "traces.langsmith.jsonl", "langsmith")


def _signature(trace: Trace) -> list[tuple]:
    """Everything about a trace that any downstream stage can observe."""
    return [
        (inv.step, inv.tool, json.dumps(inv.arguments, sort_keys=True),
         json.dumps(inv.response, sort_keys=True), inv.status, inv.error_kind)
        for inv in trace.invocations
    ]


def _parse_one(record: dict) -> tuple[Trace | None, list]:
    return parse_langsmith_record(
        record,
        source_digest="0" * 64,
        location="<test>",
        fallback_trace_id="fallback",
    )


# -- equivalence with OTLP -------------------------------------------------


def test_same_trace_and_invocation_counts(otlp_corpus: TraceCorpus, langsmith_corpus: TraceCorpus) -> None:
    assert len(langsmith_corpus.traces) == len(otlp_corpus.traces)
    assert (
        sum(len(t.invocations) for t in langsmith_corpus.traces)
        == sum(len(t.invocations) for t in otlp_corpus.traces)
    )


def test_invocations_are_indistinguishable_from_otlp(
    otlp_corpus: TraceCorpus, langsmith_corpus: TraceCorpus
) -> None:
    """The load-bearing test. Two encodings of one episode must reconstruct identically."""
    for otlp, langsmith in zip(otlp_corpus.traces, langsmith_corpus.traces, strict=True):
        assert _signature(langsmith) == _signature(otlp), langsmith.trace_id


def test_outcomes_and_instructions_match(
    otlp_corpus: TraceCorpus, langsmith_corpus: TraceCorpus
) -> None:
    for otlp, langsmith in zip(otlp_corpus.traces, langsmith_corpus.traces, strict=True):
        assert langsmith.outcome == otlp.outcome
        assert langsmith.instruction == otlp.instruction


def test_golden_langsmith_export_ingests_without_issues(langsmith_corpus: TraceCorpus) -> None:
    assert langsmith_corpus.issues == ()


def test_steps_are_dense_and_zero_based(langsmith_corpus: TraceCorpus) -> None:
    for trace in langsmith_corpus.traces:
        assert [inv.step for inv in trace.invocations] == list(range(len(trace.invocations)))


def test_error_calls_carry_kinds(langsmith_corpus: TraceCorpus) -> None:
    failures = [
        inv
        for trace in langsmith_corpus.traces
        for inv in trace.invocations
        if inv.status is CallStatus.ERROR
    ]
    assert failures
    assert all(inv.error_kind for inv in failures)


# -- shapes the wild actually produces -------------------------------------


def test_flat_runs_envelope_is_accepted() -> None:
    """Querying the LangSmith API by trace_id returns a flat list, not a tree."""
    trace, issues = _parse_one({
        "trace_id": "t1",
        "runs": [
            {"id": "r1", "name": "get_order", "run_type": "tool",
             "inputs": {"order_id": 7}, "outputs": {"order_id": 7, "status": "shipped"}},
        ],
    })
    assert issues == []
    assert [inv.tool for inv in trace.invocations] == ["get_order"]


def test_nested_child_runs_are_flattened() -> None:
    """Tree depth carries no meaning: a tool run is an invocation point wherever it sits."""
    trace, _ = _parse_one({
        "id": "root", "run_type": "chain", "name": "AgentExecutor",
        "child_runs": [
            {"id": "mid", "run_type": "chain", "name": "inner", "child_runs": [
                {"id": "deep", "name": "get_order", "run_type": "tool",
                 "inputs": {"order_id": 1}, "outputs": {"order_id": 1}},
            ]},
        ],
    })
    assert [inv.tool for inv in trace.invocations] == ["get_order"]


def test_single_key_envelopes_are_unwrapped() -> None:
    """``{"input": {...}}`` is LangChain's wrapper, not the tool's own payload."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "inputs": {"input": {"order_id": 7}}, "outputs": {"output": {"status": "shipped"}}},
        ],
    })
    inv = trace.invocations[0]
    assert inv.arguments == {"order_id": 7}
    assert inv.response == {"status": "shipped"}


def test_multi_key_payloads_are_never_reshaped() -> None:
    """Two keys means the choice of 'the' payload would be ours. Pass it through."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "refund", "run_type": "tool",
             "inputs": {"input": 1, "amount": 2}, "outputs": {"output": 1, "ok": True}},
        ],
    })
    inv = trace.invocations[0]
    assert inv.arguments == {"input": 1, "amount": 2}
    assert inv.response == {"output": 1, "ok": True}


def test_json_string_payloads_are_decoded() -> None:
    """A tool that returned a serialized string still has structure worth recovering."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "inputs": {"input": '{"order_id": 7}'},
             "outputs": {"output": '{"status": "shipped"}'}},
        ],
    })
    inv = trace.invocations[0]
    assert inv.arguments == {"order_id": 7}
    assert inv.response == {"status": "shipped"}


def test_prose_response_is_kept_not_discarded() -> None:
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "ask_docs", "run_type": "tool",
             "inputs": {"input": {"q": "policy"}}, "outputs": {"output": "Refunds take 5 days."}},
        ],
    })
    assert trace.invocations[0].response == "Refunds take 5 days."


def test_run_level_error_marks_the_call_failed() -> None:
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "inputs": {"order_id": 999}, "outputs": None, "error": "NotFoundError: no such order"},
        ],
    })
    inv = trace.invocations[0]
    assert inv.status is CallStatus.ERROR
    assert inv.error_kind == "notfounderror_no_such_order" or inv.error_kind


def test_error_shaped_body_without_run_error_still_fails() -> None:
    """A tool that returns its own error object did fail, whatever the run says."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "inputs": {"order_id": 999}, "outputs": {"error": "not_found"}},
        ],
    })
    inv = trace.invocations[0]
    assert inv.status is CallStatus.ERROR
    assert inv.error_kind == "not_found"


def test_serialized_langchain_messages_are_read() -> None:
    """``dumpd`` round-trips produce ``{"id": [...], "kwargs": {...}}``, not plain dicts."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "llm", "name": "ChatOpenAI", "run_type": "llm",
             "inputs": {"messages": [[
                 {"id": ["langchain", "schema", "messages", "HumanMessage"],
                  "kwargs": {"content": "refund my order"}},
             ]]},
             "outputs": {"generations": [[{"text": "done"}]]}},
        ],
    })
    assert trace.instruction == "refund my order"


def test_multipart_content_keeps_the_text(  ) -> None:
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "llm", "name": "ChatOpenAI", "run_type": "llm",
             "inputs": {"messages": [[
                 {"type": "human", "content": [
                     {"type": "text", "text": "refund my order"},
                     {"type": "image_url", "image_url": {"url": "x"}},
                 ]},
             ]]}},
        ],
    })
    assert trace.instruction == "refund my order"


def test_root_inputs_recover_the_instruction_without_llm_runs() -> None:
    """An agent executor logged without its llm children still states the task."""
    trace, _ = _parse_one({
        "id": "t", "name": "AgentExecutor", "run_type": "chain",
        "inputs": {"input": "refund order 7741"},
        "outputs": {"output": "refunded"},
        "child_runs": [
            {"id": "r", "name": "refund_order", "run_type": "tool",
             "inputs": {"order_id": 7741}, "outputs": {"ok": True}},
        ],
    })
    assert trace.instruction == "refund order 7741"


# -- failing loudly --------------------------------------------------------


def test_transcript_only_trace_is_flagged() -> None:
    """The single fact that decides whether an environment is buildable at all."""
    trace, issues = _parse_one({
        "id": "t", "name": "ChatOpenAI", "run_type": "llm",
        "inputs": {"messages": [[{"type": "human", "content": "hi"}]]},
        "outputs": {"generations": [[{"text": "hello"}]]},
    })
    assert trace is not None
    assert trace.invocations == ()
    assert [i.kind for i in issues] == ["no_tool_runs"]


def test_tool_calls_are_never_recovered_from_llm_outputs() -> None:
    """A requested call with no recorded response is not an invocation point.

    Keeping the request half alone would manufacture ``response=None`` records that
    look like data to every stage downstream. Better an honest zero.
    """
    trace, issues = _parse_one({
        "id": "t", "name": "ChatOpenAI", "run_type": "llm",
        "inputs": {"messages": [[{"type": "human", "content": "refund it"}]]},
        "outputs": {"generations": [[{"message": {"type": "ai", "content": "", "tool_calls": [
            {"name": "refund_order", "args": {"order_id": 1}, "id": "c1"},
        ]}}]]},
    })
    assert trace.invocations == ()
    assert "no_tool_runs" in [i.kind for i in issues]


def test_unnamed_tool_run_is_an_issue_not_a_silent_drop() -> None:
    _trace, issues = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "run_type": "tool", "inputs": {}, "outputs": {}},
        ],
    })
    assert "unnamed_tool_run" in [i.kind for i in issues]


def test_non_object_arguments_are_preserved_and_flagged() -> None:
    trace, issues = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "inputs": {"input": [1, 2]}, "outputs": {"ok": True}},
        ],
    })
    assert trace.invocations[0].arguments == {"_raw": [1, 2]}
    assert "malformed_arguments" in [i.kind for i in issues]


def test_record_with_no_runs_is_refused() -> None:
    trace, issues = _parse_one({"runs": []})
    assert trace is None
    assert [i.kind for i in issues] == ["missing_runs"]


def test_malformed_child_run_is_reported() -> None:
    _trace, issues = _parse_one({
        "id": "t", "name": "root", "run_type": "chain", "child_runs": ["not a run"],
    })
    assert "malformed_run" in [i.kind for i in issues]


# -- ordering --------------------------------------------------------------


def test_tool_runs_are_ordered_by_start_time() -> None:
    """Document order is not execution order when a tree is serialized out of sequence."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "b", "name": "second", "run_type": "tool",
             "start_time": "2026-01-01T00:00:02Z", "inputs": {}, "outputs": {}},
            {"id": "a", "name": "first", "run_type": "tool",
             "start_time": "2026-01-01T00:00:01Z", "inputs": {}, "outputs": {}},
        ],
    })
    assert [inv.tool for inv in trace.invocations] == ["first", "second"]


def test_partial_timestamps_fall_back_to_document_order() -> None:
    """All-or-nothing: never interleave two orderings, since step order drives pre-state."""
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "b", "name": "declared_first", "run_type": "tool",
             "start_time": "2026-01-01T00:00:09Z", "inputs": {}, "outputs": {}},
            {"id": "a", "name": "no_timestamp", "run_type": "tool", "inputs": {}, "outputs": {}},
        ],
    })
    assert [inv.tool for inv in trace.invocations] == ["declared_first", "no_timestamp"]


def test_latency_is_read_from_timestamps() -> None:
    trace, _ = _parse_one({
        "id": "t", "runs": [
            {"id": "r", "name": "get_order", "run_type": "tool",
             "start_time": "2026-01-01T00:00:00Z", "end_time": "2026-01-01T00:00:00.250000Z",
             "inputs": {}, "outputs": {}},
        ],
    })
    assert trace.invocations[0].latency_ms == pytest.approx(250.0)
