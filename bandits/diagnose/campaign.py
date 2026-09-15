"""One reproducible diagnose campaign, from fidelity gate to candidate reports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from bandits.diagnose.fidelity import FidelityReport, gate
from bandits.diagnose.models import (
    GroundingTransition,
    RolloutResult,
    Scenario,
    SupportPolicy,
    ToolEffectCatalog,
)
from bandits.diagnose.report import CapabilityReport, Comparison, build_capability_report
from bandits.diagnose.rollout import (
    Budget,
    Candidate,
    run_rollout,
)
from bandits.diagnose.world import ToolWorldPredictor, UserPolicyPredictor
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract


class CampaignBlocked(RuntimeError):
    """The requested capability claim failed a validity precondition."""


CandidateFactory = Callable[[], Candidate]


class CampaignSpec(Contract):
    campaign_id: str
    scenario_set_version: str
    retrieval_index_version: str
    tool_awm_version: str
    user_policy_version: str
    fidelity_report_id: str
    support_policy: SupportPolicy
    attempts_per_scenario: int = 1
    reference_id: str = ""
    control_id: str = ""


class CampaignResult(Contract):
    campaign_id: str
    fidelity_report_id: str
    support_policy_id: str
    reports: tuple[CapabilityReport, ...]
    rollouts: tuple[RolloutResult, ...] = ()
    comparison_sane: bool | None = None


def run_campaign(
    spec: CampaignSpec,
    *,
    scenarios: Sequence[Scenario],
    index: Sequence[GroundingTransition],
    candidates: Mapping[str, CandidateFactory],
    binding_ids: Mapping[str, str],
    tool_world: ToolWorldPredictor,
    user_policy: UserPolicyPredictor,
    fidelity: FidelityReport,
    fidelity_thresholds: dict[str, float],
    catalog: ToolEffectCatalog | None = None,
    judge=None,
    budget: Budget = Budget(),
) -> CampaignResult:
    """Run every candidate on the same pinned environment and scenarios.

    Factories—not candidate instances—are accepted so stateful adapters and
    scripted controls start fresh for every pass@k attempt.
    """
    passed, failures = gate(fidelity, thresholds=fidelity_thresholds)
    if not passed:
        raise CampaignBlocked("AWM fidelity gate failed: " + "; ".join(failures))
    if spec.attempts_per_scenario < 1:
        raise ValueError("attempts_per_scenario must be positive")
    missing = [scenario.scenario_id for scenario in scenarios if scenario.scenario_id not in binding_ids]
    if missing:
        raise CampaignBlocked(f"{len(missing)} scenarios have no verifier binding")

    versions = {
        "tool_awm": spec.tool_awm_version,
        "user_policy": spec.user_policy_version,
        "retrieval_index": spec.retrieval_index_version,
        "scenario_set": spec.scenario_set_version,
    }
    calibrated_budget = budget.replace(min_support=spec.support_policy.tool_minimum)
    reports: list[CapabilityReport] = []
    all_rollouts: list[RolloutResult] = []
    for candidate_id, factory in candidates.items():
        rollouts = []
        for scenario in scenarios:
            for seed in range(spec.attempts_per_scenario):
                rollouts.append(
                    run_rollout(
                        scenario,
                        factory(),
                        index=index,
                        tool_world=tool_world,
                        user_policy=user_policy,
                        binding_id=binding_ids[scenario.scenario_id],
                        candidate_id=candidate_id,
                        seed=seed,
                        budget=calibrated_budget,
                        catalog=catalog,
                        judge=judge,
                        versions=versions,
                    )
                )
        reports.append(build_capability_report(rollouts, candidate_id=candidate_id))
        all_rollouts.extend(rollouts)

    comparison = Comparison(
        reports=tuple(reports),
        reference_id=spec.reference_id,
        control_id=spec.control_id,
    )
    sane = comparison.sane if spec.reference_id and spec.control_id else None
    if sane is False:
        raise CampaignBlocked("reference candidate did not beat the negative control")
    if not comparison.environments_match:
        raise CampaignBlocked("candidate reports were produced by different environments")
    return CampaignResult(
        campaign_id=spec.campaign_id,
        fidelity_report_id=spec.fidelity_report_id,
        support_policy_id=spec.support_policy.policy_id,
        reports=tuple(reports),
        rollouts=tuple(all_rollouts),
        comparison_sane=sane,
    )


def campaign_artifact_id(result: CampaignResult) -> str:
    digest = hashlib.sha256(result.model_dump_json().encode()).hexdigest()[:16]
    return f"diagnosis-{digest}"


def save_campaign(
    result: CampaignResult, *, parent_artifact_id: str, store: DerivedStore
) -> DerivedEnvelope:
    """Persist the manifest, reports, and every attempted rollout together."""
    return store.write(
        campaign_artifact_id(result),
        kind="diagnosis-campaign",
        parent_artifact_id=parent_artifact_id,
        payload=result.model_dump_json().encode(),
        summary={
            "candidates": len(result.reports),
            "rollouts": len(result.rollouts),
            "scorable": sum(report.scorable for report in result.reports),
        },
    )


def load_campaign(artifact_id: str, store: DerivedStore) -> CampaignResult:
    return CampaignResult.model_validate_json(store.read_payload(artifact_id))
