"""Static probes evaluate authentic prefixes without invoking an AWM."""

from bandits.diagnose.models import (
    CommunicationRequirement,
    Partition,
    ResultStatus,
    Scenario,
    ScenarioKind,
    SealedSuccessContract,
    SuccessShape,
)
from bandits.diagnose.rollout import CandidateAction
from bandits.diagnose.static import run_static_probes


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="scenario",
        kind=ScenarioKind.TASK_START,
        task="Help the user",
        success_contract=SealedSuccessContract(
            contract_id="contract",
            source_task_id="task",
            shape=SuccessShape.INFORMATIONAL,
            communication=(
                CommunicationRequirement(
                    requirement_id="say-answer",
                    assertion="answer the user",
                    kind="communicate_info",
                    expected_values=("answer",),
                ),
            ),
        ),
        source_trace_id="trace",
        source_task_id="task",
        family_id="family",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("trace",),
    )


def test_static_probe_uses_only_the_candidate_view_and_no_world() -> None:
    seen = {}

    def candidate(**kwargs):
        seen.update(kwargs)
        return CandidateAction(content="answer", done=True)

    report, outcomes = run_static_probes(
        (_scenario(),),
        candidate,
        candidate_id="candidate",
        judge=lambda _scenario, action, _expected: (
            ResultStatus.PASS if action.content == "answer" else ResultStatus.FAIL
        ),
    )
    assert report.next_action_capability == 1.0
    assert outcomes[0].status is ResultStatus.PASS
    assert "state" not in seen
    assert "scenario" not in seen
