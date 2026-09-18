"""Tests for task-level capability statistics and report integrity."""

import pytest

from bandits.diagnose.models import (
    ActionCall,
    ComponentResult,
    ResultStatus,
    RolloutResult,
    RolloutStep,
    TerminationReason,
)
from bandits.diagnose.report import build_capability_report, partition_rollouts


def _rollout(scenario: str, seed: int, passed: bool, *, candidate: str = "candidate") -> RolloutResult:
    status = ResultStatus.PASS if passed else ResultStatus.FAIL
    return RolloutResult(
        rollout_id=f"{scenario}-{seed}",
        scenario_id=scenario,
        binding_id="binding",
        candidate_id=candidate,
        seed=seed,
        terminated_by=TerminationReason.CANDIDATE_COMPLETED,
        operational_result=ComponentResult(status=status),
        process_result=ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        communication_result=ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        overall=status,
        tool_awm_version="awm-1",
        user_policy_version="user-1",
        retrieval_index_version="index-1",
        scenario_set_version="scenarios-1",
    )


def test_pass_at_k_is_macro_averaged_across_scenarios() -> None:
    # Easy gets nine attempts and hard gets one. Pooling reports 90%; treating
    # each task equally reports (100% + 0%) / 2 = 50%.
    rollouts = [_rollout("easy", seed, True) for seed in range(9)]
    rollouts.append(_rollout("hard", 0, False))
    report = build_capability_report(rollouts, candidate_id="candidate", ks=(1,))
    assert report.pass_at[1] == 0.5


def test_report_rejects_mixed_environment_versions() -> None:
    rows = [_rollout("a", 0, True), _rollout("b", 0, True)]
    rows[1] = rows[1].replace(tool_awm_version="awm-2")
    with pytest.raises(ValueError, match="tool_awm_version"):
        build_capability_report(rows, candidate_id="candidate")


def test_report_rejects_a_different_candidate() -> None:
    with pytest.raises(ValueError, match="candidate"):
        build_capability_report(
            [_rollout("a", 0, True, candidate="someone-else")],
            candidate_id="candidate",
        )



def _terminated(reason: TerminationReason, *, overall=ResultStatus.UNKNOWN, steps=()) -> RolloutResult:
    return RolloutResult(
        rollout_id=f"r-{reason.value}",
        scenario_id="s",
        binding_id="binding",
        candidate_id="candidate",
        seed=0,
        terminated_by=reason,
        steps=tuple(steps),
        operational_result=ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        process_result=ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        communication_result=ComponentResult(status=ResultStatus.NOT_APPLICABLE),
        overall=overall,
        tool_awm_version="awm-1",
        user_policy_version="user-1",
        retrieval_index_version="index-1",
        scenario_set_version="scenarios-1",
    )


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.AWM_ABSTAINED,
        TerminationReason.UNSUPPORTED_ACTION,
        TerminationReason.INVALID_TRANSITION,
    ],
)
def test_an_unscorable_rollout_is_partitioned_without_mutating_a_frozen_report(reason) -> None:
    """AbstentionReport is frozen, so counting used to raise mid-aggregation.

    A rollout set that happened to be entirely scorable never reached the
    counters, which is why the whole abstention table -- the thing campaigns
    report beside every capability number -- could crash on its first
    abstention while the tests looked green.
    """
    scorable, report = partition_rollouts([_terminated(reason)])

    assert scorable == []
    assert report.total == 1
    assert report.by_reason[reason.value] == 1


def test_a_verifier_unknown_rollout_is_counted_not_scored() -> None:
    scorable, report = partition_rollouts(
        [_terminated(TerminationReason.CANDIDATE_COMPLETED, overall=ResultStatus.UNKNOWN)]
    )

    assert scorable == []
    assert report.verifier_unknown == 1


def test_abstained_steps_are_counted_per_tool() -> None:
    rollout = _terminated(
        TerminationReason.AWM_ABSTAINED,
        steps=(
            RolloutStep(index=0, calls=(ActionCall(tool="cancel_reservation"),), abstained=True),
            RolloutStep(index=1, calls=(ActionCall(tool="get_user_details"),), abstained=False),
        ),
    )

    _, report = partition_rollouts([rollout])

    assert report.abstained == 1
    # Only the step that actually abstained contributes its tool.
    assert report.by_tool == {"cancel_reservation": 1}


def test_a_mixed_batch_reports_every_partition_together() -> None:
    """The counters must survive each other: an earlier abstention used to
    abort the run before a later scorable rollout was ever reached."""
    rollouts = [
        _terminated(TerminationReason.AWM_ABSTAINED),
        _terminated(TerminationReason.INVALID_TRANSITION),
        _rollout("easy", 0, True),
    ]

    scorable, report = partition_rollouts(rollouts)

    assert [r.rollout_id for r in scorable] == ["easy-0"]
    assert (report.abstained, report.invalid, report.total) == (1, 1, 3)
