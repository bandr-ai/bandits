from __future__ import annotations

import json

import pytest

from bandits.store import DerivedStore
from bandits_jev.cost import (
    compute_verifier_cost_id,
    load_verifier_cost,
    save_verifier_cost,
    verifier_cost_from_ledger,
)
from bandits_jev.dataset import (
    DecisionDataset,
    DecisionDatasetCounts,
    DecisionExample,
    DecisionJudgeInfo,
    DecisionLineage,
    DecisionTarget,
)

_JUDGE = "fireworks/judge"


def _dataset(turns: list[tuple[str, int]]) -> DecisionDataset:
    judge = DecisionJudgeInfo(
        model=_JUDGE, prompt_digest="p", temperature=0.0, votes_requested=2, votes_valid=2, settings_digest="s"
    )
    examples = tuple(
        DecisionExample(
            decision_id=f"d-{trace}-{turn}",
            family_id=trace,
            group_id=trace,
            state="s",
            question="q",
            primitive="choice",
            options={"success": "s", "unclear": "u", "failure": "f"},
            target=DecisionTarget(kind="soft", probabilities={"success": 1.0, "unclear": 0.0, "failure": 0.0}),
            label_source="judge_votes",
            split="test",
            lineage=DecisionLineage(source_kind="t", record_id=f"{trace}:{turn}", trace_id=trace, turn_index=turn),
            judge=judge,
        )
        for trace, turn in turns
    )
    return DecisionDataset(
        producer="t",
        source_artifact_ids=(),
        examples=examples,
        counts=DecisionDatasetCounts(examples=len(examples), train=0, dev=0, test=len(examples), quarantined=0),
    )


def _call(trace, turn, *, prompt=1000, completion=500, seconds=2.0, status="success", model=_JUDGE):
    return json.dumps(
        {
            "stage": "judge_turn",
            "event_type": "model_call",
            "trace_id": trace,
            "turn_index": turn,
            "model": model,
            "status": status,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion} if status == "success" else None,
            "duration_seconds": seconds,
        }
    )


def _retry(trace, turn):
    return json.dumps({"stage": "judge_turn", "event_type": "retry", "trace_id": trace, "turn_index": turn})


def _price(lines, dataset, **kwargs):
    return verifier_cost_from_ledger(
        lines, dataset, "ds-1", ledger="ledger.jsonl",
        input_usd_per_mtok=kwargs.get("inp", 0.2), output_usd_per_mtok=kwargs.get("out", 0.8),
    )


def test_every_vote_for_a_step_is_summed_onto_its_decision() -> None:
    dataset = _dataset([("t1", 0), ("t1", 1)])
    lines = [_call("t1", 0), _call("t1", 0, seconds=3.0), _call("t1", 1, prompt=2000, completion=0)]
    cost = _price(lines, dataset)

    first = cost.per_decision["d-t1-0"]
    assert (first.calls, first.prompt_tokens, first.completion_tokens) == (2, 2000, 1000)
    assert first.seconds == pytest.approx(5.0)
    assert first.usd == pytest.approx((2000 * 0.2 + 1000 * 0.8) / 1e6)
    assert cost.per_decision["d-t1-1"].usd == pytest.approx(2000 * 0.2 / 1e6)


def test_uncovered_decisions_are_counted_not_priced_at_zero() -> None:
    dataset = _dataset([("t1", 0), ("t2", 0)])
    cost = _price([_call("t1", 0), _call("other", 9)], dataset)

    assert set(cost.per_decision) == {"d-t1-0"}
    assert cost.uncovered_decisions == 1


def test_failed_calls_and_retries_are_recorded() -> None:
    dataset = _dataset([("t1", 0)])
    cost = _price([_retry("t1", 0), _call("t1", 0, status="error"), _call("t1", 0)], dataset)

    decision = cost.per_decision["d-t1-0"]
    assert decision.calls == 2 and decision.failed_calls == 1
    assert cost.retries == 1


def test_a_ledger_from_a_different_judge_model_is_refused() -> None:
    with pytest.raises(ValueError, match="labeled by"):
        _price([_call("t1", 0, model="someone/else")], _dataset([("t1", 0)]))


def test_negative_prices_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _price([_call("t1", 0)], _dataset([("t1", 0)]), inp=-1)


def test_verifier_cost_round_trips(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    cost = _price([_call("t1", 0)], _dataset([("t1", 0)]))
    envelope = save_verifier_cost(cost, store)

    assert envelope.artifact_id == compute_verifier_cost_id(cost)
    assert load_verifier_cost(envelope.artifact_id, store) == cost


def test_a_ledger_holding_two_runs_over_the_same_steps_is_flagged() -> None:
    """votes_requested is 2 in _dataset: four successful calls for one step
    means a second judge run's calls were summed in."""
    dataset = _dataset([("t1", 0), ("t1", 1)])
    lines = [_call("t1", 0)] * 4 + [_call("t1", 1)] * 2
    cost = _price(lines, dataset)

    assert cost.over_counted_decisions == ("d-t1-0",)
    assert cost.per_decision["d-t1-0"].calls == 4  # recorded as found, flagged, not silently trimmed


def test_a_step_judged_by_two_runs_in_one_dataset_cannot_be_priced() -> None:
    """Ledger rows carry no judge-run id, so two decisions for the same
    trace and turn would silently share (or lose) each other's calls."""
    dataset = _dataset([("t1", 0)])
    second = dataset.examples[0].model_copy(update={"decision_id": "d-t1-0-other-judge-run"})
    doubled = dataset.model_copy(
        update={
            "examples": dataset.examples + (second,),
            "counts": dataset.counts.model_copy(update={"examples": 2, "test": 2}),
        }
    )

    with pytest.raises(ValueError, match="more than one decision"):
        _price([_call("t1", 0)], doubled)
