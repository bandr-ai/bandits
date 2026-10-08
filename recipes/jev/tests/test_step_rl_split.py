from __future__ import annotations

from recipes.jev.scripts.step_rl_split import HELD_OUT_EVAL, split


def test_split_is_disjoint_complete_and_reproducible():
    ids = [str(i) for i in range(769, 1599)]
    result = split(ids)
    assert not set(result["train_pool"]) & set(result["held_out"])
    assert set(result["train_pool"]) | set(result["held_out"]) == set(ids)
    assert len(result["held_out"]) == 415 and len(result["held_out_eval"]) == HELD_OUT_EVAL
    assert split(list(reversed(ids))) == result
    assert result["held_out_eval"] == result["held_out"][:HELD_OUT_EVAL]
