from __future__ import annotations

from bandits_jev.jev_remote import FULL_STATE_CHARS, full_outcome_state


def events(n, size):
    return [{"tool": "browser.search", "action": f'{{"query": "q{i}"}}', "observation": f"r{i} " + "x" * size} for i in range(n)]


def test_full_state_keeps_every_step_and_the_final_answer():
    state = full_outcome_state("Q?", events(5, 100), "Answer: Paris")
    assert state.startswith("Research question: Q?") and state.endswith("Final answer: Answer: Paris")
    assert all(f"Step {i + 1}:" in state and f"r{i} " in state for i in range(5))


def test_long_trajectories_are_cut_evenly_to_fit():
    state = full_outcome_state("Q?", events(12, 50_000), "Answer: x")
    assert len(state) <= FULL_STATE_CHARS + 2_000
    assert all(f"r{i} " in state for i in range(12))  # no step dropped


def test_no_steps_and_no_answer():
    assert full_outcome_state("Q?", [], None) == "Research question: Q?\nFinal answer: (no answer)"
