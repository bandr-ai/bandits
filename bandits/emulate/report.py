"""Aggregate rollouts into numbers, keeping every separation that makes them true.

The arithmetic here is easy and the denominator is not. A rollout that abstained
tells you nothing about the candidate, so it leaves the denominator entirely —
counting it as a failure charges the candidate for the simulator's ignorance,
and counting it as a pass rewards walking off the supported manifold. A
verifier-unknown leaves for the same reason: nothing was established.

That makes the excluded counts part of the result rather than a footnote. With
zero error evidence for every tool in the target family, a campaign can abstain
on most interesting actions and report a fine pass@k over the handful of
rollouts that stayed on the happy path. The abstention table sits beside every
figure so that reading is available to anyone who looks.

Reported separately, never averaged and never merged into one artifact:

    static next-action capability   from complete-rollout capability
    task-start                      from prefix-started
    verified failure                from verifier unknown
    pass@k                          from pass^k
    per-family macro                from traffic-weighted
    capability                      from AWM fidelity (D3)
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import Field

from bandits.emulate.models import (
    ResultStatus,
    RolloutResult,
    TerminationReason,
)
from bandits.traces import Contract


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Probability that at least one of k draws from n attempts succeeds.

    The unbiased estimator, computed as the complement of drawing k failures.
    None when k exceeds what was actually run: reporting pass@10 from four
    attempts would be an extrapolation dressed as a measurement.
    """
    if n <= 0 or k <= 0 or k > n or c < 0:
        return None
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def reliability(n: int, c: int, k: int) -> float | None:
    """pass^k: the chance that *all* k draws succeed.

    The number that separates a candidate which can do the task from one that
    does it dependably. High pass@k with low pass^k is the route table's
    "capability exists but is unreliable" row, and it is invisible unless both
    are reported.
    """
    if n <= 0 or k <= 0 or k > n or c < 0:
        return None
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """A confidence interval that stays inside [0, 1] at small n.

    The normal approximation gives negative lower bounds on the sample sizes
    this lab actually runs — a family with eight held-out traces is normal here —
    and an interval that reports impossible values invites ignoring intervals.
    """
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class AbstentionReport(Contract):
    """Where the environment declined, and why.

    Published beside every capability figure. An abstention rate is not a
    caveat on the number; it says which slice of behaviour the number covers.
    """

    total: int = 0
    abstained: int = 0
    invalid: int = 0
    verifier_unknown: int = 0
    by_tool: dict[str, int] = Field(default_factory=dict)
    by_reason: dict[str, int] = Field(default_factory=dict)

    @property
    def rate(self) -> float | None:
        if not self.total:
            return None
        return (self.abstained + self.invalid) / self.total

    @property
    def scorable(self) -> int:
        """Rollouts that could carry a verdict at all."""
        return self.total - self.abstained - self.invalid - self.verifier_unknown


class CapabilityReport(Contract):
    """One candidate on one scenario set, under one environment version.

    Carries no fidelity figure. A capability claim that quoted its simulator's
    accuracy in the same artifact would invite reading one as evidence for the
    other, and the route table's "strong only on learned-AWM-heavy rollouts →
    block" row depends on keeping them apart.
    """

    schema_version: int = 1
    candidate_id: str
    scenario_set_version: str = ""
    tool_awm_version: str = ""
    user_policy_version: str = ""
    retrieval_index_version: str = ""
    verifier_id: str = ""

    attempts: int = 0
    scenarios: int = 0
    scorable_scenarios: int = 0
    scorable: int = 0
    verified_passes: int = 0
    verified_failures: int = 0
    abstention: AbstentionReport = AbstentionReport()

    pass_at: dict[int, float] = Field(default_factory=dict)
    pass_power: dict[int, float] = Field(default_factory=dict)
    pass_at_scenarios: dict[int, int] = Field(default_factory=dict)
    interval: tuple[float, float] | None = None

    simulation_conditioned: bool = True
    """False only where every verdict rested entirely on recorded state."""

    by_kind: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_shape: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_family: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def headline(self) -> float | None:
        """pass@1 on scorable rollouts, or None when nothing was scorable."""
        return self.pass_at.get(1)


class StaticReport(Contract):
    """Next-action capability. Deliberately not pass@k.

    A static probe reads the real next observation from the trace and asks
    whether the candidate's action was acceptable. It needs no AWM and is
    cheap, which makes it the right first screen — and it measures one decision
    rather than a completed task, which is why the field is named this way and
    why this contract has no ``pass_at`` field for a caller to reach for.
    """

    schema_version: int = 1
    candidate_id: str
    scenario_set_version: str = ""
    considered: int = 0
    acceptable: int = 0
    unacceptable: int = 0
    unjudged: int = 0
    """Cases no rule and no judge could decide. Never counted as either."""

    by_shape: dict[str, dict[str, float]] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def next_action_capability(self) -> float | None:
        judged = self.acceptable + self.unacceptable
        return self.acceptable / judged if judged else None


def partition_rollouts(
    rollouts: Sequence[RolloutResult],
) -> tuple[list[RolloutResult], AbstentionReport]:
    """Split rollouts into those that can carry a verdict and those that cannot.

    The single most consequential function here. Everything downstream counts
    only what this returns first, and everything it sets aside is reported by
    the second return value rather than disappearing.
    """
    scorable: list[RolloutResult] = []
    # Accumulated locally and assembled once at the end: AbstentionReport is a
    # frozen Contract, so `report.abstained += 1` raises. Only the dict fields
    # tolerated in-place mutation, which is why a rollout set that happened to
    # be entirely scorable ran clean and the crash stayed hidden -- the first
    # abstention, invalid transition or unknown verdict took the whole
    # aggregation down, and campaigns report abstention rates by design.
    abstained = 0
    invalid = 0
    verifier_unknown = 0
    by_reason: dict[str, int] = {}
    by_tool: dict[str, int] = {}

    for rollout in rollouts:
        reason = rollout.terminated_by
        by_reason[reason.value] = by_reason.get(reason.value, 0) + 1

        if reason in (TerminationReason.AWM_ABSTAINED, TerminationReason.UNSUPPORTED_ACTION):
            abstained += 1
            for step in rollout.steps:
                if step.abstained:
                    for call in step.calls:
                        by_tool[call.tool] = by_tool.get(call.tool, 0) + 1
            continue
        if reason is TerminationReason.INVALID_TRANSITION:
            invalid += 1
            continue
        if rollout.overall is ResultStatus.UNKNOWN:
            verifier_unknown += 1
            continue
        scorable.append(rollout)

    return scorable, AbstentionReport(
        total=len(rollouts),
        abstained=abstained,
        invalid=invalid,
        verifier_unknown=verifier_unknown,
        by_reason=by_reason,
        by_tool=by_tool,
    )


def _rates(rollouts: Sequence[RolloutResult]) -> dict[str, float]:
    passes = sum(1 for r in rollouts if r.overall is ResultStatus.PASS)
    return {
        "attempts": float(len(rollouts)),
        "passes": float(passes),
        "rate": passes / len(rollouts) if rollouts else 0.0,
    }


def _group(rollouts: Sequence[RolloutResult], key) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RolloutResult]] = {}
    for rollout in rollouts:
        grouped.setdefault(str(key(rollout)), []).append(rollout)
    return {name: _rates(rows) for name, rows in sorted(grouped.items())}


def _bootstrap_mean_interval(
    values: Sequence[float], *, draws: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Deterministic percentile bootstrap over independent scenarios."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws)
    )
    return (means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))])


def _assert_consistent(rollouts: Sequence[RolloutResult], candidate_id: str) -> None:
    for rollout in rollouts:
        if rollout.candidate_id != candidate_id:
            raise ValueError(
                f"candidate {rollout.candidate_id!r} does not match report candidate "
                f"{candidate_id!r}"
            )
    for field in (
        "tool_awm_version",
        "user_policy_version",
        "retrieval_index_version",
        "scenario_set_version",
    ):
        values = {getattr(rollout, field) for rollout in rollouts}
        if len(values) > 1:
            raise ValueError(f"mixed {field} values cannot be aggregated: {sorted(values)!r}")


def build_capability_report(
    rollouts: Sequence[RolloutResult],
    *,
    candidate_id: str,
    ks: Sequence[int] = (1, 2, 5),
    kind_of=None,
    shape_of=None,
    family_of=None,
    verifier_id: str = "",
) -> CapabilityReport:
    """Aggregate one candidate's attempts, excluding what cannot be scored.

    ``ks`` beyond the number of attempts are silently absent rather than
    extrapolated: a missing pass@5 is honest, and a computed one from three
    attempts is not.
    """
    _assert_consistent(rollouts, candidate_id)
    scorable, abstention = partition_rollouts(rollouts)
    passes = sum(1 for r in scorable if r.overall is ResultStatus.PASS)
    failures = sum(1 for r in scorable if r.overall is ResultStatus.FAIL)
    n = len(scorable)

    by_scenario: dict[str, list[RolloutResult]] = {}
    for rollout in scorable:
        by_scenario.setdefault(rollout.scenario_id, []).append(rollout)

    pass_at: dict[int, float] = {}
    pass_power: dict[int, float] = {}
    pass_at_scenarios: dict[int, int] = {}
    for k in ks:
        at_values: list[float] = []
        power_values: list[float] = []
        for rows in by_scenario.values():
            scenario_passes = sum(r.overall is ResultStatus.PASS for r in rows)
            value = pass_at_k(len(rows), scenario_passes, k)
            power = reliability(len(rows), scenario_passes, k)
            if value is not None:
                at_values.append(value)
            if power is not None:
                power_values.append(power)
        if at_values:
            pass_at[k] = sum(at_values) / len(at_values)
            pass_at_scenarios[k] = len(at_values)
        if power_values:
            pass_power[k] = sum(power_values) / len(power_values)

    first = scorable[0] if scorable else (rollouts[0] if rollouts else None)
    return CapabilityReport(
        candidate_id=candidate_id,
        scenario_set_version=first.scenario_set_version if first else "",
        tool_awm_version=first.tool_awm_version if first else "",
        user_policy_version=first.user_policy_version if first else "",
        retrieval_index_version=first.retrieval_index_version if first else "",
        verifier_id=verifier_id,
        attempts=len(rollouts),
        scenarios=len({r.scenario_id for r in rollouts}),
        scorable_scenarios=len(by_scenario),
        scorable=n,
        verified_passes=passes,
        verified_failures=failures,
        abstention=abstention,
        pass_at=pass_at,
        pass_power=pass_power,
        pass_at_scenarios=pass_at_scenarios,
        interval=_bootstrap_mean_interval(
            [
                sum(r.overall is ResultStatus.PASS for r in rows) / len(rows)
                for rows in by_scenario.values()
            ]
        ),
        simulation_conditioned=any(r.simulation_conditioned for r in rollouts),
        by_kind=_group(scorable, kind_of) if kind_of else {},
        by_shape=_group(scorable, shape_of) if shape_of else {},
        by_family=_group(scorable, family_of) if family_of else {},
    )


class Comparison(Contract):
    """Several candidates on the same scenario set and environment version.

    The controls are the measurement, not decoration. If a deliberately broken
    candidate is not ranked below a competent one, the lab has not shown that
    it ranks anything, and no result it produces should be used to choose a
    model.
    """

    reports: tuple[CapabilityReport, ...] = ()
    target_id: str = ""
    reference_id: str = ""
    control_id: str = ""

    @property
    def ordered(self) -> tuple[tuple[str, float | None], ...]:
        rows = [(report.candidate_id, report.headline) for report in self.reports]
        return tuple(sorted(rows, key=lambda row: (row[1] is None, -(row[1] or 0.0))))

    @property
    def sane(self) -> bool | None:
        """Whether the reference outscored the control.

        None when either was not run, because an untested sanity check is not a
        passed one.
        """
        scores = {report.candidate_id: report.headline for report in self.reports}
        reference, control = scores.get(self.reference_id), scores.get(self.control_id)
        if reference is None or control is None:
            return None
        return reference > control

    @property
    def environments_match(self) -> bool:
        """Whether every report measured the same environment.

        Comparing candidates across AWM versions compares environments, and the
        difference would be attributed to the models.
        """
        pinned = {
            (
                report.tool_awm_version,
                report.user_policy_version,
                report.retrieval_index_version,
                report.scenario_set_version,
            )
            for report in self.reports
        }
        return len(pinned) <= 1


def route_recommendation(
    report: CapabilityReport, static: StaticReport | None = None
) -> tuple[str, tuple[str, ...]]:
    """The plan's route table, as evidence rather than a score.

    Returns a route and the reasons behind it. Every branch names what it saw,
    because a recommendation without its evidence is a number with a label, and
    no promotion in this system rests on one of those.
    """
    reasons: list[str] = []
    static_capability = static.next_action_capability if static else None
    headline = report.headline

    if report.abstention.rate is not None and report.abstention.rate > 0.5:
        reasons.append(
            f"the environment declined {report.abstention.rate:.0%} of attempts; "
            "the surviving rollouts are not a representative sample of behaviour"
        )
        return "block_low_support", tuple(reasons)

    if not report.scorable:
        return "block_unscorable", ("nothing was scorable",)

    if static_capability is not None and static_capability < 0.1:
        reasons.append(f"next-action capability {static_capability:.0%} is near zero")
        return "reject", tuple(reasons)

    if static_capability is not None and static_capability > 0.5 and (headline or 0) < 0.1:
        reasons.append(
            f"static capability {static_capability:.0%} does not survive interaction "
            f"({headline:.0%} complete-rollout)"
            if headline is not None
            else "static capability does not survive interaction"
        )
        return "sft_first", tuple(reasons)

    best = max(report.pass_at.values(), default=0.0)
    one = report.pass_at.get(1, 0.0)
    if best > 0.5 and one < 0.3:
        reasons.append(f"pass@k reaches {best:.0%} while pass@1 is {one:.0%}")
        return "opd_rl", tuple(reasons)

    if report.simulation_conditioned:
        reasons.append("every verdict is simulation-conditioned")
    reasons.append(f"pass@1 {one:.0%} on {report.scorable} scorable rollouts")
    return "advance_to_stronger_validation", tuple(reasons)
