"""Static next-action probes over authentic trace prefixes; no AWM involved."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bandits.diagnose.candidates import parse_action
from bandits.diagnose.models import GroundingTransition, ResultStatus, Scenario
from bandits.diagnose.report import StaticReport
from bandits.diagnose.rollout import Candidate, CandidateAction, reset
from bandits.traces import Contract


class StaticProbeOutcome(Contract):
    scenario_id: str
    expected_transition_id: str | None = None
    action: CandidateAction
    status: ResultStatus
    detail: str = ""


StaticJudge = Callable[[Scenario, CandidateAction, GroundingTransition | None], ResultStatus]


def run_static_probes(
    scenarios: Sequence[Scenario],
    candidate: Candidate,
    *,
    candidate_id: str,
    expected_by_scenario: dict[str, GroundingTransition] | None = None,
    judge: StaticJudge,
    scenario_set_version: str = "",
) -> tuple[StaticReport, tuple[StaticProbeOutcome, ...]]:
    """Evaluate one candidate decision at real task/prefix checkpoints.

    The caller supplies the judge because equality with the recorded action is
    not capability: a different action may also be correct. The expected real
    transition is evidence available to that judge, never to the candidate.
    """
    expected_by_scenario = expected_by_scenario or {}
    outcomes: list[StaticProbeOutcome] = []
    by_shape: dict[str, dict[str, float]] = {}

    for scenario in scenarios:
        view, _state, history = reset(scenario)
        raw = candidate(view=view, history=history)
        action = raw if isinstance(raw, CandidateAction) else parse_action(raw)
        expected = expected_by_scenario.get(scenario.scenario_id)
        status = judge(scenario, action, expected)
        if status not in (ResultStatus.PASS, ResultStatus.FAIL, ResultStatus.UNKNOWN):
            raise ValueError("a static judge must return pass, fail, or unknown")
        outcomes.append(
            StaticProbeOutcome(
                scenario_id=scenario.scenario_id,
                expected_transition_id=expected.transition_id if expected else None,
                action=action,
                status=status,
            )
        )
        row = by_shape.setdefault(
            scenario.success_contract.shape.value,
            {"considered": 0.0, "acceptable": 0.0, "unacceptable": 0.0, "unjudged": 0.0},
        )
        row["considered"] += 1
        key = {
            ResultStatus.PASS: "acceptable",
            ResultStatus.FAIL: "unacceptable",
            ResultStatus.UNKNOWN: "unjudged",
        }[status]
        row[key] += 1

    return (
        StaticReport(
            candidate_id=candidate_id,
            scenario_set_version=scenario_set_version,
            considered=len(outcomes),
            acceptable=sum(row.status is ResultStatus.PASS for row in outcomes),
            unacceptable=sum(row.status is ResultStatus.FAIL for row in outcomes),
            unjudged=sum(row.status is ResultStatus.UNKNOWN for row in outcomes),
            by_shape=by_shape,
        ),
        tuple(outcomes),
    )

