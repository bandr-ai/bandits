import pytest

from bandits.rank import stats


def test_pass_at_k_is_the_unbiased_estimator_and_refuses_extrapolation():
    assert stats.pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert stats.pass_at_k(4, 1, 2) == pytest.approx(0.5)
    assert stats.pass_at_k(4, 1, 4) == 1.0
    assert stats.pass_at_k(8, 0, 4) == 0.0
    assert stats.pass_at_k(4, 1, 5) is None


def test_summary_headroom_learnable_and_prediction():
    s = stats.summarize("m", [(0, 4), (4, 4), (1, 4), (2, 4)])
    assert s.pass_at_1 == pytest.approx(0.4375)
    assert s.pass_at_k == pytest.approx(0.75) and s.k == 4
    assert s.learnable == 0.5
    assert s.predicted == pytest.approx(s.pass_at_1 + 0.5 * s.headroom)
    assert s.interval[0] <= s.predicted <= s.interval[1]


def test_summary_rejects_impossible_counts():
    with pytest.raises(ValueError):
        stats.summarize("m", [(5, 4)])


def test_order_separates_clear_gaps_and_flags_ties():
    strong = stats.summarize("a", [(4, 4)] * 40)
    twin = stats.summarize("c", [(4, 4)] * 40)
    weak = stats.summarize("b", [(0, 4)] * 40)
    out = stats.order([weak, strong, twin])
    assert out[2][0].model == "b" and out[2][1] == []
    assert out[0][1]


def test_rank_agreement():
    assert stats.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1
    assert stats.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1
    assert stats.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert stats.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert stats.spearman_rho([1, 1, 1], [1, 2, 3]) is None
    assert stats.spearman_rho([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(0.9487, abs=1e-3)
