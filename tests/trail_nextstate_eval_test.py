from __future__ import annotations

from bandits.verify.nextstate import TurnVerdict
from scripts.trail_nextstate_eval import brier, calibration_pairs, ece, prf, reliability_bins


def _verdict(trace_id: str, index: int, judge_votes: dict[str, float] | None) -> TurnVerdict:
    score = None
    if judge_votes is not None:
        score = int(max(judge_votes, key=lambda k: judge_votes[k]))
    return TurnVerdict(
        trace_id=trace_id,
        index=index,
        action_span_id=f"{trace_id}-{index}",
        observed=True,
        score=score,
        judge_votes=judge_votes,
        votes=(score,) if judge_votes else (),
    )


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


def test_brier_is_none_on_empty_and_zero_on_perfect_predictions() -> None:
    assert brier([]) is None
    assert brier([(1.0, True), (0.0, False)]) == 0.0
    assert brier([(1.0, False)]) == 1.0
    assert brier([(0.5, True), (0.5, False)]) == 0.25


def test_reliability_bins_report_every_bin_including_empty_ones() -> None:
    pairs = [(0.05, False), (0.95, True), (0.95, True), (0.95, False)]
    bins = reliability_bins(pairs, n_bins=10)
    assert len(bins) == 10
    low, high = bins[0], bins[9]
    assert low["range"] == [0.0, 0.1] and low["n"] == 1 and low["observed_rate"] == 0.0
    assert high["range"] == [0.9, 1.0] and high["n"] == 3
    assert high["observed_rate"] == 2 / 3
    assert abs(high["mean_predicted"] - 0.95) < 1e-9
    empty = bins[5]
    assert empty["n"] == 0
    assert empty["mean_predicted"] is None and empty["observed_rate"] is None


def test_ece_is_none_on_empty_and_zero_when_perfectly_calibrated() -> None:
    assert ece([]) is None
    # every prediction in its bin matches the bin's observed rate exactly
    assert ece([(1.0, True), (1.0, True), (0.0, False)]) == 0.0
    # a confident-wrong prediction pulls ece up from zero
    assert ece([(0.9, False)]) == 0.9


def test_calibration_pairs_includes_zero_negative_examples() -> None:
    """A verdict whose votes were all "0" or "1" has judge_votes["-1"] == 0.0
    -- a real observed frequency, dense since the nextstate.py fix -- and
    must contribute (0.0, False) to the calibration set, not be dropped.
    Dropping it would restrict prevalence/Brier/ECE to the subset that
    happened to see a negative vote, biasing every number in the report."""
    verdicts = {
        ("a", 0): _verdict("a", 0, {"-1": 0.0, "0": 0.0, "1": 1.0}),  # no error, no -1 votes
        ("a", 1): _verdict("a", 1, {"-1": 0.0, "0": 1.0, "1": 0.0}),  # no error, no -1 votes
        ("a", 2): _verdict("a", 2, {"-1": 1.0, "0": 0.0, "1": 0.0}),  # the one true error
    }
    observed = set(verdicts)
    truth_any = {("a", 2)}

    pairs = calibration_pairs(verdicts, observed, truth_any)

    assert len(pairs) == 3
    assert (0.0, False) in pairs
    assert pairs.count((0.0, False)) == 2
    assert (1.0, True) in pairs
    prevalence = sum(1.0 for _, y in pairs if y) / len(pairs)
    assert prevalence == 1 / 3
    assert brier(pairs) == 0.0  # every prediction matches its outcome exactly


def test_calibration_pairs_excludes_unobserved_and_unjudged_turns() -> None:
    verdicts = {
        ("a", 0): _verdict("a", 0, {"-1": 0.0, "0": 0.0, "1": 1.0}),
        ("a", 1): _verdict("a", 1, None),  # failed judgment: no distribution to report
    }
    observed = {("a", 0)}  # turn 1 is unobserved

    pairs = calibration_pairs(verdicts, observed, set())

    assert len(pairs) == 1
    assert pairs == [(0.0, False)]
