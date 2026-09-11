"""Per-prediction attribution of a shared DSPy call history."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bandits.analyze.rlm_history import scoped_to_history, summarize_history


def test_tokens_are_summed_only_over_the_calls_that_reported_them():
    """A missing usage block contributes nothing, never a zero.

    Summing a zero in for an unreported call produces a total that looks
    authoritative and understates the bill.
    """
    calls, tokens = summarize_history(
        [
            {"usage": {"prompt_tokens": 10, "total_tokens": 12}},
            {"usage": None},
            {},
            {"usage": {"prompt_tokens": 5, "total_tokens": 6}},
        ]
    )

    assert calls == 4, "every entry is a physical call, reported usage or not"
    assert tokens == {"prompt_tokens": 15, "total_tokens": 18}


def test_no_reported_usage_stays_empty_rather_than_becoming_zero():
    calls, tokens = summarize_history([{}, {}])

    assert calls == 2
    assert tokens == {}, "empty reads as unknown; a zero would read as free"


class _SharedHistoryModel:
    """Stands in for a ``dspy.LM``: one history list, appended to forever."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def call(self, count: int, tokens: int) -> None:
        for _ in range(count):
            self.history.append({"usage": {"total_tokens": tokens}})


def test_one_family_is_never_charged_for_the_calls_of_another():
    """`lm.history` is shared for the life of the process, not per prediction.

    Reading it whole would attribute every earlier family's calls to the family
    running now, which is the failure that makes per-family cost meaningless.
    """
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(int(members), tokens=10)
        return SimpleNamespace()

    wrapped = scoped_to_history(predict, language_model)

    wrapped(members="3", question="q")
    assert wrapped.spend() == (3, {"total_tokens": 30})

    wrapped(members="2", question="q")
    assert wrapped.spend() == (2, {"total_tokens": 20}), (
        "the second family spent two calls, not the five in the shared history"
    )
    assert len(language_model.history) == 5, "the underlying history still accumulates"


def test_a_failed_prediction_keeps_the_calls_it_completed():
    """The subcalls made before the failure are what explain it."""
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(4, tokens=25)
        raise RuntimeError("the sandbox died")

    wrapped = scoped_to_history(predict, language_model)

    with pytest.raises(RuntimeError, match="the sandbox died"):
        wrapped(members="x", question="q")

    assert wrapped.spend() == (4, {"total_tokens": 100})


def test_the_recorded_slice_is_copied_rather_than_referenced():
    """A reference into a growing list would describe later calls too."""
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(1, tokens=5)
        return SimpleNamespace()

    wrapped = scoped_to_history(predict, language_model)
    wrapped(members="x", question="q")

    language_model.call(9, tokens=5)

    assert wrapped.spend() == (1, {"total_tokens": 5})
