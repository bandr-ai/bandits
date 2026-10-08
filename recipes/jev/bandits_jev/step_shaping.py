"""Turn observed browser steps and judge scores into one shaped rollout reward.

The rollout glue records an event for every browser action: the last policy
token that produced it, the action, and the real tool reply. After the rollout,
frozen-judge scores arrive for those events. This module decides which events
may earn reward (observed, first occurrence, one per turn, not truncated),
applies the lambda and caps through ``reward_vector``, and reports what it
rejected. Stock GRPO sums token rewards, so the total is what trains; the
token vector is kept for audit and for a later per-step advantage.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from bandits_jev.step_reward import reward_vector


@dataclass(frozen=True)
class StepEvent:
    position: int
    """Index of the last policy token of the action, before the reply was appended."""
    tool: str
    action: str
    observation: str | None
    """The real tool reply. None or blank means the result was never observed."""
    judge_score: float | None
    """P(positive) - P(negative) from the frozen judge, or None if unscored."""


@dataclass
class ShapedRollout:
    rewards: list[float]
    total: float
    step_sum_uncapped: float
    step_sum_capped: float
    rejected: Counter = field(default_factory=Counter)


def action_key(tool: str, action: str) -> tuple[str, str]:
    """Two actions are the same when tool and whitespace- and case-normalised text match."""
    return tool, " ".join(action.split()).lower()


def _action_key(event: StepEvent) -> tuple[str, str]:
    return action_key(event.tool, event.action)


def shape_rollout(
    response_mask: Sequence[int],
    events: Sequence[StepEvent],
    outcome: float,
    *,
    step_weight: float,
    step_cap: float = 1.0,
) -> ShapedRollout:
    """Return per-token rewards and the trajectory total.

    Rejected events earn nothing and are counted by reason. A repeated action
    can still be penalised but never rewarded. A mask-0 position is a harness
    bug and raises instead of being dropped.
    """
    policy_positions = [i for i, bit in enumerate(response_mask) if bit]
    last_policy = policy_positions[-1] if policy_positions else -1
    rejected: Counter = Counter()
    accepted: list[tuple[int, float]] = []
    seen_actions: set[tuple[str, str]] = set()
    seen_positions: set[int] = set()
    uncapped = 0.0

    for event in sorted(events, key=lambda e: e.position):
        if event.position > last_policy or event.position >= len(response_mask):
            rejected["truncated"] += 1
            continue
        if event.position < 0 or not response_mask[event.position]:
            raise ValueError(f"step event at {event.position} is not a policy token")
        if event.position in seen_positions:
            rejected["extra_call_in_turn"] += 1
            continue
        seen_positions.add(event.position)
        key = _action_key(event)
        repeated = key in seen_actions
        seen_actions.add(key)
        if not event.observation or not event.observation.strip():
            rejected["no_provenance"] += 1
            continue
        score = event.judge_score
        if score is None or not math.isfinite(score):
            rejected["unscored"] += 1
            continue
        score = max(-1.0, min(1.0, score))
        if repeated:
            rejected["repeat"] += 1
            score = min(score, 0.0)
        accepted.append((event.position, score))
        uncapped += step_weight * score

    rewards = reward_vector(
        response_mask, accepted, outcome, step_weight=step_weight, step_cap=step_cap
    )
    total = float(sum(rewards))
    return ShapedRollout(
        rewards=rewards,
        total=total,
        step_sum_uncapped=uncapped,
        step_sum_capped=total - outcome,
        rejected=rejected,
    )
