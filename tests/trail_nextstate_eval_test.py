from __future__ import annotations

from scripts.trail_nextstate_eval import prf


def test_disjoint_non_empty_sets_score_f1_zero_not_undefined() -> None:
    """precision and recall are real 0.0 values here, not undefined -- only
    an empty denominator set (nothing predicted, or nothing true) means the
    metric itself is undefined and should read None."""
    result = prf(predicted={"a"}, truth={"b"}, universe={"a", "b"})

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_an_empty_predicted_set_leaves_precision_and_f1_undefined() -> None:
    result = prf(predicted=set(), truth={"a"}, universe={"a", "b"})

    assert result["precision"] is None
    assert result["recall"] == 0.0
    assert result["f1"] is None


def test_a_perfect_match_still_scores_f1_one() -> None:
    result = prf(predicted={"a"}, truth={"a"}, universe={"a", "b"})

    assert result["f1"] == 1.0
