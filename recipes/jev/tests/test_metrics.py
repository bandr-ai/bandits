from __future__ import annotations

import math
import random

import pytest

from bandits_jev.metrics import (
    NLL_PROBABILITY_FLOOR,
    argmax_option,
    ece,
    gold_option,
    grouped_bootstrap_interval,
    is_correct,
    macro_f1,
    row_brier,
    row_nll,
    total_variation,
)


def _plain_bootstrap(values, *, draws=2000, seed=0):
    """The per-row bootstrap bandits.emulate.report used before it was
    generalized, kept here verbatim as the reference."""
    if not values:
        return None
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return (means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))])


def test_ungrouped_bootstrap_matches_the_plain_per_row_bootstrap_exactly() -> None:
    values = [0.1, 0.9, 0.4, 0.4, 0.7, 0.0, 1.0, 0.3]
    assert grouped_bootstrap_interval(values) == _plain_bootstrap(values)
    assert grouped_bootstrap_interval(values, [str(i) for i in range(len(values))]) == _plain_bootstrap(values)


def test_grouped_bootstrap_widens_when_rows_move_together() -> None:
    """Ten groups of ten identical rows carry ten rows' worth of evidence,
    not a hundred; resampling rows would overstate the precision."""
    rng = random.Random(1)
    group_values = [rng.random() for _ in range(10)]
    values = [v for v in group_values for _ in range(10)]
    groups = [str(g) for g in range(10) for _ in range(10)]

    by_row = grouped_bootstrap_interval(values)
    by_group = grouped_bootstrap_interval(values, groups)

    assert (by_group[1] - by_group[0]) > 2 * (by_row[1] - by_row[0])


def test_bootstrap_edge_cases() -> None:
    assert grouped_bootstrap_interval([]) is None
    assert grouped_bootstrap_interval([0.5, 0.7], ["g", "g"]) == (0.6, 0.6)
    with pytest.raises(ValueError):
        grouped_bootstrap_interval([1.0, 2.0], ["only-one"])


def test_row_metrics_for_a_hard_target() -> None:
    probs = {"a": 0.7, "b": 0.2, "c": 0.1}
    target = {"a": 0.0, "b": 1.0, "c": 0.0}

    assert row_nll(probs, target) == pytest.approx(-math.log(0.2))
    assert row_brier(probs, target) == pytest.approx(0.49 + 0.64 + 0.01)
    assert argmax_option(probs) == "a"


def test_nll_floors_a_zero_probability_on_gold() -> None:
    assert row_nll({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == pytest.approx(
        -math.log(NLL_PROBABILITY_FLOOR)
    )


def test_argmax_ties_break_by_option_order_like_the_scorer() -> None:
    assert argmax_option({"b": 0.5, "a": 0.5}) == "b"


def test_a_tied_target_is_neither_right_nor_wrong_whatever_the_option_order() -> None:
    """One tie rule everywhere: a tied row is left out of accuracy (None) and
    counted, never credited or charged by dict order."""
    target = {"failure": 0.5, "success": 0.5, "unclear": 0.0}

    for prediction in (
        {"success": 0.8, "failure": 0.2, "unclear": 0.0},
        {"failure": 0.8, "success": 0.2, "unclear": 0.0},
        {"unclear": 0.8, "success": 0.2, "failure": 0.0},
    ):
        assert is_correct(prediction, target) is None
    assert is_correct({"failure": 0.9, "success": 0.1, "unclear": 0.0}, {"failure": 1.0, "success": 0.0, "unclear": 0.0})


def test_macro_f1_is_unweighted_over_gold_and_predicted_labels() -> None:
    # a: tp=1 fp=1 fn=0 -> 2/3 ; b: tp=0 fp=0 fn=1 -> 0 ; c: tp=1 -> 1
    pairs = [("a", "a"), ("a", "b"), ("c", "c")]
    assert macro_f1(pairs) == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert macro_f1([]) is None


def test_top_label_ece_of_a_perfectly_calibrated_bin_is_zero() -> None:
    pairs = [(0.75, True)] * 3 + [(0.75, False)]
    assert ece(pairs) == pytest.approx(0.0)


def test_total_variation() -> None:
    assert total_variation({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == 1.0
    assert total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0


def test_gold_option_is_none_on_a_tie_never_the_first_option() -> None:
    assert gold_option({"a": 0.2, "b": 0.8}) == "b"
    assert gold_option({"a": 0.5, "b": 0.5}) is None
    assert gold_option({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) is None
