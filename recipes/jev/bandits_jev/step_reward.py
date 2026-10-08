"""Map observed browser rewards onto the policy tokens that caused them.

The rollout integration must provide token positions before tool responses are
appended. This pure function keeps the reward accounting separately testable.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def reward_vector(
    response_mask: Sequence[int],
    events: Iterable[tuple[int, float]],
    terminal_score: float,
    *,
    step_weight: float,
    step_cap: float = 1.0,
) -> list[float]:
    """Return token rewards; mask 1 marks policy tokens and 0 tool tokens.

    An event is ``(last assistant token of observed action, judge score)``.
    The final answer's outcome score lands on the last assistant token.
    Positive and negative step totals are bounded separately so extra calls
    cannot overwhelm the outcome and a late penalty is not erased by a cap.
    """
    if not response_mask or any(bit not in (0, 1) for bit in response_mask):
        raise ValueError("response_mask must contain policy/tool bits")
    if not math.isfinite(terminal_score) or not 0 <= step_weight <= 1 or step_cap <= 0:
        raise ValueError("invalid reward scale")
    policy_positions = [i for i, bit in enumerate(response_mask) if bit]
    if not policy_positions:
        raise ValueError("rollout has no policy token")
    terminal_position = policy_positions[-1]
    rewards = [0.0] * len(response_mask)
    positive_total = negative_total = 0.0
    used = set()
    for position, score in events:
        if not isinstance(position, int) or position in used or position > terminal_position:
            raise ValueError("step event must identify a distinct policy token")
        if position < 0 or not response_mask[position] or not math.isfinite(score) or abs(score) > 1:
            raise ValueError("invalid step event position or score")
        used.add(position)
        scaled = step_weight * score
        if scaled >= 0:
            awarded = min(scaled, max(0.0, step_cap - positive_total))
            positive_total += awarded
        else:
            awarded = max(scaled, min(0.0, -step_cap - negative_total))
            negative_total += awarded
        rewards[position] = awarded
    rewards[terminal_position] += terminal_score
    return rewards
