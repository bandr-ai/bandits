"""Tests for task-level capability statistics and report integrity."""

import pytest

from bandits.diagnose.models import (
    ComponentResult,
    ResultStatus,
    RolloutResult,
    TerminationReason,
)
from bandits.diagnose.report import build_capability_report


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

