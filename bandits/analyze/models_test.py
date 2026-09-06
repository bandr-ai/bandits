from __future__ import annotations

import pytest

from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    LeakageError,
    Visibility,
    build_task_candidate,
    evidence_id,
)


def _evidence(visibility: Visibility, claim: str = "instruction") -> Evidence:
    return Evidence(
        evidence_id=evidence_id(trace_id="trace-1", claim=claim),
        claim=claim,
        value="Refund order 7741",
        visibility=visibility,
        provenance="observed",
        strength="strong",
        trace_id="trace-1",
    )


def test_prompt_built_from_at_start_evidence_is_accepted() -> None:
    task = build_task_candidate(
        task_id="task-trace-1",
        trace_id="trace-1",
        instruction="Refund order 7741",
        prompt_evidence=(_evidence(Visibility.AT_START),),
        trajectory_span_ids=("span-1",),
        terminal_span_ids=("span-2",),
        outcome_evidence=(),
    )
    assert task.prompt_evidence_ids[0].startswith("ev-trace-1-trace-instruction-")


@pytest.mark.parametrize(
    "visibility", [Visibility.DURING, Visibility.TERMINAL, Visibility.POST_HOC]
)
def test_prompt_built_from_later_evidence_raises(visibility: Visibility) -> None:
    with pytest.raises(LeakageError, match="not knowable at t=0"):
        build_task_candidate(
            task_id="task-trace-1",
            trace_id="trace-1",
            instruction="Refund order 7741",
            prompt_evidence=(_evidence(visibility, claim="final_state_field"),),
            trajectory_span_ids=(),
            terminal_span_ids=(),
            outcome_evidence=(),
        )


def test_evidence_ids_are_stable_for_the_same_fact() -> None:
    once = evidence_id(trace_id="trace-1", claim="command_exit_code", span_id="span-2")

    assert once.startswith("ev-trace-1-span-2-command-exit-code-")
    assert once == evidence_id(trace_id="trace-1", claim="command_exit_code", span_id="span-2")


def test_two_details_differing_only_in_punctuation_are_two_ids() -> None:
    """The readable half flattens them; the fact behind them is not the same."""
    literal = evidence_id(trace_id="t", claim="final_state_field", span_id="s", detail="tool.a\\.b")
    nested = evidence_id(trace_id="t", claim="final_state_field", span_id="s", detail="tool.a.b")

    assert literal != nested


def test_strength_ranks_order_conflicting_evidence() -> None:
    strong = _evidence(Visibility.TERMINAL).model_copy(update={"strength": "strong"})
    weak = _evidence(Visibility.TERMINAL).model_copy(update={"strength": "weak"})
    assert strong.strength_rank > weak.strength_rank


def test_evidence_kind_dominates_strength_when_resolving_conflicts() -> None:
    live = _evidence(Visibility.POST_HOC).model_copy(
        update={"kind": EvidenceKind.LIVE_QUERY, "strength": "weak"}
    )
    self_report = _evidence(Visibility.TERMINAL).model_copy(
        update={"kind": EvidenceKind.AGENT_SELF_REPORT, "strength": "strong"}
    )
    assert live.trust_rank > self_report.trust_rank


def test_analysis_rejects_dangling_evidence_reference() -> None:
    task = build_task_candidate(
        task_id="task-trace-1",
        trace_id="trace-1",
        instruction="Refund it",
        prompt_evidence=(_evidence(Visibility.AT_START),),
        trajectory_span_ids=(),
        terminal_span_ids=(),
        outcome_evidence=(),
    )
    with pytest.raises(ValueError, match="references missing evidence"):
        CorpusAnalysis(corpus_id="corpus-1", source="test", tasks=(task,), evidence=())


def test_agent_self_report_is_the_floor() -> None:
    """Nothing may outrank-by-default what the agent says about its own run."""
    self_report = _evidence(Visibility.TERMINAL).model_copy(
        update={"kind": EvidenceKind.AGENT_SELF_REPORT, "strength": "strong"}
    )
    others = [
        _evidence(Visibility.TERMINAL).model_copy(update={"kind": kind, "strength": "weak"})
        for kind in EvidenceKind
        if kind is not EvidenceKind.AGENT_SELF_REPORT
    ]

    assert all(other.trust_rank > self_report.trust_rank for other in others)
