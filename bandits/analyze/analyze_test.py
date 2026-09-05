"""End-to-end checks over both fixture domains.

Two domains on purpose: a commerce trace and a coding trace, so the extraction
layer cannot quietly grow assumptions that only hold for one of them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bandits.analyze import analyze_corpus, extract_task, load_analysis, save_analysis
from bandits.analyze.models import EvidenceKind, Visibility
from bandits.analyze.outcomes import extract_outcome_evidence
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, Trace, TraceCorpus

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@pytest.fixture
def commerce() -> TraceCorpus:
    return load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp")


@pytest.fixture
def coding() -> TraceCorpus:
    return load_corpus(FIXTURES / "traces.coding.otlp.jsonl", "otlp")


def _claims(evidence) -> set[str]:
    return {e.claim for e in evidence}


def test_task_prompt_holds_only_the_instruction(commerce: TraceCorpus) -> None:
    analysis = analyze_corpus(commerce)
    by_id = analysis.evidence_by_id()

    for task in analysis.tasks:
        prompt = [by_id[eid] for eid in task.prompt_evidence_ids]
        assert prompt, "every fixture trace declares an instruction"
        assert all(e.visibility is Visibility.AT_START for e in prompt)
        assert _claims(prompt) == {"instruction"}


def test_tools_called_is_during_not_at_start(coding: TraceCorpus) -> None:
    """The tools an episode happened to call are not the toolset it was offered."""
    analysis = analyze_corpus(coding)
    tools_called = [e for e in analysis.evidence if e.claim == "tools_called"]

    assert tools_called
    assert all(e.visibility is Visibility.DURING for e in tools_called)
    for task in analysis.tasks:
        assert not set(task.prompt_evidence_ids) & {e.evidence_id for e in tools_called}


def test_exit_codes_are_read_as_outcome_evidence(coding: TraceCorpus) -> None:
    trace = next(t for t in coding.traces if t.trace_id == "code-1")

    evidence = extract_outcome_evidence(trace)
    exit_codes = [e for e in evidence if e.claim == "command_exit_code"]

    assert {e.value["value"] for e in exit_codes} == {0, 1}
    assert all(e.strength == "strong" for e in exit_codes)
    # The final run is terminal evidence; the earlier failing run is not.
    terminal = [e for e in exit_codes if e.visibility is Visibility.TERMINAL]
    assert [e.span_id for e in terminal] == ["span-4"]


def test_absent_tool_result_is_recorded_not_treated_as_success(coding: TraceCorpus) -> None:
    """code-2's model claims the work is done while its only tool returned nothing."""
    trace = next(t for t in coding.traces if t.trace_id == "code-2")

    evidence = extract_outcome_evidence(trace)

    assert "missing_tool_result" in _claims(evidence)
    assert "final_state_field" not in _claims(evidence)


def test_error_span_is_evidence_but_not_a_verdict(commerce: TraceCorpus) -> None:
    trace = next(t for t in commerce.traces if t.trace_id == "trace-2")

    evidence = extract_outcome_evidence(trace)
    errors = [e for e in evidence if e.claim == "span_error"]

    assert [e.span_id for e in errors] == ["span-4"]
    assert all(e.provenance == "observed" for e in evidence)


def test_missing_instruction_becomes_a_limitation() -> None:
    trace = Trace(trace_id="bare", source="otlp", source_digest="a" * 64, task=None, spans=())

    task, evidence = extract_task(trace)

    assert task.instruction is None
    assert task.prompt_evidence_ids == ()
    assert any("no instruction" in limitation for limitation in task.limitations)
    assert any("no spans" in limitation for limitation in task.limitations)
    assert "instruction" not in _claims(evidence)


def test_unnormalizable_source_records_are_surfaced(commerce: TraceCorpus) -> None:
    """The commerce fixture contains one unparseable line; the analysis must say so."""
    analysis = analyze_corpus(commerce)

    assert any("could not be normalized" in limitation for limitation in analysis.limitations)


def test_analysis_round_trips_through_the_derived_store(tmp_path, coding: TraceCorpus) -> None:
    store = DerivedStore(tmp_path / ".bandits")
    analysis = analyze_corpus(coding)

    envelope = save_analysis(analysis, store)

    assert envelope.kind == "analysis"
    assert envelope.parent_artifact_id == analysis.corpus_id
    assert envelope.summary["tasks"] == len(analysis.tasks)
    assert load_analysis(envelope.artifact_id, store) == analysis


def test_analyzing_the_same_corpus_twice_is_stable(tmp_path, coding: TraceCorpus) -> None:
    store = DerivedStore(tmp_path / ".bandits")

    first = save_analysis(analyze_corpus(coding), store)
    second = save_analysis(analyze_corpus(coding), store)

    assert first.artifact_id == second.artifact_id


def test_tool_result_before_final_model_is_terminal_state_evidence() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    spans = (
        Span(
            span_id="tool",
            kind=SpanKind.TOOL,
            name="refund_order",
            started_at=start,
            ended_at=start,
            output={"status": "refunded"},
        ),
        Span(
            span_id="reply",
            kind=SpanKind.MODEL,
            name="assistant",
            started_at=start + timedelta(seconds=1),
            ended_at=start + timedelta(seconds=1),
            output="Refund completed",
        ),
    )
    trace = Trace(
        trace_id="refund", source="otlp", source_digest="a" * 64, task="Refund it", spans=spans
    )

    task, _ = extract_task(trace)
    evidence = extract_outcome_evidence(trace)
    final_state = next(e for e in evidence if e.claim == "final_state_field")

    assert task.terminal_span_ids == ("tool", "reply")
    assert final_state.span_id == "tool"
    assert final_state.visibility is Visibility.TERMINAL
    assert final_state.value["field"] == "refund_order.status"


def test_a_closing_claim_is_recorded_as_self_report_and_ranked_last() -> None:
    """The chat fixture ends with the agent asserting it finished."""
    corpus = load_corpus(FIXTURES / "traces.chat.jsonl", "chat-json")
    analysis = analyze_corpus(corpus)

    final_output = next(e for e in analysis.evidence if e.claim == "final_output")
    others = [e for e in analysis.evidence if e.claim != "final_output"]

    assert final_output.kind is EvidenceKind.AGENT_SELF_REPORT
    assert final_output.visibility is Visibility.TERMINAL
    assert all(e.trust_rank > final_output.trust_rank for e in others)


def test_recorded_score_outranks_the_agents_own_claim() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trace = Trace(
        trace_id="scored",
        source="otlp",
        source_digest="a" * 64,
        task="Do the thing",
        spans=(
            Span(
                span_id="reply",
                kind=SpanKind.MODEL,
                name="assistant",
                started_at=start,
                ended_at=start,
                output="All done!",
                attributes={"score": 0.0},
            ),
        ),
    )

    evidence = {e.claim: e for e in extract_outcome_evidence(trace)}

    assert evidence["recorded_score"].kind is EvidenceKind.TRUSTED_EVALUATOR
    assert evidence["recorded_score"].trust_rank > evidence["final_output"].trust_rank


def test_initial_and_terminal_state_are_both_extracted() -> None:
    """The before/after pair is what makes a non-trivial verifier possible."""
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    trace = next(t for t in corpus.traces if t.trace_id == "refund-1")

    evidence = extract_outcome_evidence(trace)
    initial = {
        e.value["key"]: e.value["value"] for e in evidence if e.claim == "initial_state_field"
    }
    final = {e.value["key"]: e.value["value"] for e in evidence if e.claim == "final_state_field"}

    assert initial["charged_amount"] == 48.0
    assert final["refunded_amount"] == 48.0
    assert final["status"] == "refunded"


def test_every_scalar_of_the_terminal_tool_is_recorded_not_just_status() -> None:
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    trace = next(t for t in corpus.traces if t.trace_id == "refund-1")

    keys = {
        e.value["key"] for e in extract_outcome_evidence(trace) if e.claim == "final_state_field"
    }

    assert keys == {"status", "refunded_amount"}


def test_a_single_tool_episode_has_no_initial_state() -> None:
    """Its one result is the terminal state; calling it 'initial' too is vacuous."""
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    trace = next(t for t in corpus.traces if t.trace_id == "cancel-1")

    evidence = extract_outcome_evidence(trace)

    assert not [e for e in evidence if e.claim == "initial_state_field"]
    assert [e for e in evidence if e.claim == "final_state_field"]


def test_span_order_breaks_ties_on_source_order_not_span_id() -> None:
    """addr-1 stamps every span with one timestamp; 's10' must not sort before 's2'."""
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    trace = next(t for t in corpus.traces if t.trace_id == "addr-1")

    assert trace.spans[-1].name == "override_address_policy"
