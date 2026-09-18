"""One reproducible diagnose campaign, from fidelity gate to candidate reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from bandits.diagnose.fidelity import FidelityReport, gate
from bandits.diagnose.models import (
    GroundingTransition,
    ResultStatus,
    RolloutResult,
    Scenario,
    SupportPolicy,
    ToolEffectCatalog,
    VerifierBinding,
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
from bandits.verify.propose import FamilyVerifier, RejectedCheck, compile_check


class CampaignBlocked(RuntimeError):
    """The requested capability claim failed a validity precondition."""


CandidateFactory = Callable[[int], Candidate]
PUBLISHABLE_FIDELITY_METRICS = frozenset(
    {
        "status_accuracy",
        "field_accuracy",
        "supported_coverage",
        "wrong_abstention_rate",
        "validation_rejection_rate",
    }
)


class CampaignSpec(Contract):
    campaign_id: str
    scenario_set_version: str
    retrieval_index_version: str
    tool_awm_version: str
    user_policy_version: str
    fidelity_report_id: str
    support_policy: SupportPolicy
    attempts_per_scenario: int = 1
    mode: Literal["development", "publishable"] = "publishable"
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
    bindings: Mapping[str, VerifierBinding],
    family_verifiers: Mapping[str, FamilyVerifier] | None = None,
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
    family_verifiers = family_verifiers or {}
    passed, failures = gate(fidelity, thresholds=fidelity_thresholds)
    if not fidelity_thresholds:
        raise CampaignBlocked("fidelity policy names no required metrics")
    missing_metrics = PUBLISHABLE_FIDELITY_METRICS - fidelity_thresholds.keys()
    if spec.mode == "publishable" and missing_metrics:
        raise CampaignBlocked(
            "publishable fidelity policy omits required metrics: "
            + ", ".join(sorted(missing_metrics))
        )
    if fidelity.split != "held_out":
        raise CampaignBlocked("fidelity report is not held-out")
    version_mismatches = [
        name
        for name, measured, expected in (
            ("tool AWM", fidelity.awm_version, spec.tool_awm_version),
            ("user policy", fidelity.user_policy_version, spec.user_policy_version),
            ("retrieval index", fidelity.retrieval_index_version, spec.retrieval_index_version),
        )
        if measured != expected
    ]
    if version_mismatches:
        raise CampaignBlocked(
            "fidelity report version mismatch: " + ", ".join(version_mismatches)
        )
    if spec.support_policy.calibration_report_id != spec.fidelity_report_id:
        raise CampaignBlocked("support policy was calibrated against another fidelity report")
    if not passed:
        raise CampaignBlocked("AWM fidelity gate failed: " + "; ".join(failures))
    if spec.attempts_per_scenario < 1:
        raise ValueError("attempts_per_scenario must be positive")
    missing = [scenario.scenario_id for scenario in scenarios if scenario.scenario_id not in bindings]
    if missing:
        raise CampaignBlocked(f"{len(missing)} scenarios have no verifier binding")
    verifier_by_scenario: dict[str, FamilyVerifier | None] = {}
    for scenario in scenarios:
        binding = bindings[scenario.scenario_id]
        if (
            binding.scenario_id != scenario.scenario_id
            or binding.contract_id != scenario.success_contract.contract_id
            or binding.shape is not scenario.success_contract.shape
            or set(binding.required_reward_basis) != set(scenario.success_contract.reward_basis)
        ):
            raise CampaignBlocked(f"verifier binding does not join scenario {scenario.scenario_id}")
        verifier = (
            family_verifiers.get(binding.reviewed_verifier_id)
            if binding.reviewed_verifier_id
            else None
        )
        if verifier is not None and verifier.family_id != scenario.family_id:
            raise CampaignBlocked(f"verifier family does not join scenario {scenario.scenario_id}")
        if spec.mode == "publishable" and (verifier is None or not verifier.accepted()):
            raise CampaignBlocked(f"scenario {scenario.scenario_id} has no accepted family checks")
        verifier_by_scenario[scenario.scenario_id] = verifier
    if spec.mode == "publishable" and (not spec.reference_id or not spec.control_id):
        raise CampaignBlocked("publishable campaigns require reference and control candidates")

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
                        factory(seed),
                        index=index,
                        tool_world=tool_world,
                        user_policy=user_policy,
                        binding_id=bindings[scenario.scenario_id].binding_id,
                        candidate_id=candidate_id,
                        seed=seed,
                        budget=calibrated_budget,
                        catalog=catalog,
                        judge=judge,
                        versions=versions,
                    )
                )
                rollouts[-1] = _apply_family_guards(
                    rollouts[-1], verifier_by_scenario[scenario.scenario_id]
                )
        reports.append(build_capability_report(rollouts, candidate_id=candidate_id))
        all_rollouts.extend(rollouts)

    comparison = Comparison(
        reports=tuple(reports),
        reference_id=spec.reference_id,
        control_id=spec.control_id,
    )
    sane = comparison.sane if spec.reference_id and spec.control_id else None
    if spec.mode == "publishable" and sane is not True:
        raise CampaignBlocked("reference candidate did not beat the negative control")
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


def _apply_family_guards(
    rollout: RolloutResult, verifier: FamilyVerifier | None
) -> RolloutResult:
    """Run reviewed reaction checks without reusing historical judge verdicts."""
    if verifier is None:
        return rollout
    rows = []
    for step in rollout.steps:
        if step.responder == "none":
            continue
        rows.append(
            {
                "trace_id": rollout.rollout_id,
                "index": step.index,
                "task": None,
                "action": json.dumps(
                    {
                        "content": step.action_content,
                        "calls": [call.model_dump(mode="json") for call in step.calls],
                    },
                    default=str,
                ),
                "next_state": json.dumps(step.observation, default=str),
                "reactions": [
                    {
                        "kind": "tool" if step.responder == "tool_world" else "user",
                        "name": step.responder,
                        "text": json.dumps(step.observation, default=str),
                        "error": bool(step.validator_rejections),
                    }
                ],
                "observed": True,
                "errored": bool(step.validator_rejections),
            }
        )
    unresolved = not rows and bool(verifier.accepted())
    flagged = False
    for check in verifier.accepted():
        try:
            fn = compile_check(check.code)
        except RejectedCheck:
            unresolved = True
            continue
        for row in rows:
            try:
                result = fn(dict(row))
            except Exception:  # noqa: BLE001 - an unresolved guard fails closed
                result = None
            if result is True:
                flagged = True
            elif result is not False:
                unresolved = True
    if flagged:
        return rollout.replace(overall=ResultStatus.FAIL)
    if unresolved:
        return rollout.replace(overall=ResultStatus.UNKNOWN)
    return rollout


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
