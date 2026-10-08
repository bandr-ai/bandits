import pytest

from bandits_jev.step_reward import reward_vector


def test_step_reward_lands_on_action_tokens_and_terminal_outcome() -> None:
    mask = [1, 1, 0, 0, 1, 0, 1, 1]
    rewards = reward_vector(mask, [(1, 0.8), (4, -0.5)], 1.0, step_weight=0.3)
    assert rewards == [0.0, 0.24, 0.0, 0.0, -0.15, 0.0, 0.0, 1.0]


def test_step_reward_caps_excess_calls_without_erasing_penalties() -> None:
    mask = [1] * 6
    rewards = reward_vector(mask, [(0, 1), (1, 1), (2, 1), (3, -1), (4, -1)], 0, step_weight=0.5)
    assert rewards == [0.5, 0.5, 0.0, -0.5, -0.5, 0.0]


@pytest.mark.parametrize("events", [[(1, 1.0)], [(0, 1.0), (0, 0.0)], [(0, 1.1)]])
def test_invalid_step_reward_events_fail_closed(events: list[tuple[int, float]]) -> None:
    with pytest.raises(ValueError):
        reward_vector([1, 0, 1], events, 1.0, step_weight=0.1)


def test_truncated_after_tool_response_can_receive_step_reward() -> None:
    assert reward_vector([1, 0, 0], [(0, 1.0)], 0.0, step_weight=0.3) == [0.3, 0.0, 0.0]
