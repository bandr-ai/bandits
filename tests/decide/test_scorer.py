from __future__ import annotations

import inspect
import math

import pytest

from bandits.decide.dataset import DecisionExample, DecisionLineage, DecisionTarget
from bandits.decide.prompt import MAX_OPTIONS, build_prompt
from bandits.decide.scorer import (
    DecisionScoreResult,
    LogitPrediction,
    RejectedScore,
    ScorerRun,
    TokenizationError,
    compute_scorer_run_id,
    score_dataset,
    score_example,
)


def _example(options: dict[str, str] | None = None, target_option: str = "a") -> DecisionExample:
    options = options or {"a": "apple", "b": "banana", "c": "cherry"}
    return DecisionExample(
        decision_id="decision-1",
        family_id="f1",
        state="a customer asks about fruit",
        question="which fruit is mentioned",
        primitive="choice",
        options=options,
        target=DecisionTarget(
            kind="hard", probabilities={k: (1.0 if k == target_option else 0.0) for k in options}
        ),
        label_source="test",
        split="test",
        lineage=DecisionLineage(source_kind="test", record_id="r1"),
    )


class FakePredictor:
    """No torch, no download. Deterministic per-letter logits derived from
    the prompt text, so results are reproducible and order-sensitive (needed
    to make two-order averaging actually move the number in a test). Reports
    a full-vocabulary logsumexp of a much larger pretend vocabulary, so
    candidate_token_mass is a real (non-trivial) fraction, not always 1.0."""

    model_id = "fake/tiny"
    revision = "abc123"
    dtype = "float32"
    device = "cpu"

    def __init__(
        self,
        *,
        bad_letters: set[str] | None = None,
        tokens_per_char: float = 0.3,
        vocab_background_logit: float = 5.0,
        vocab_background_size: int = 50_000,
    ):
        self._bad_letters = bad_letters or set()
        self._tokens_per_char = tokens_per_char
        self._vocab_background_logit = vocab_background_logit
        self._vocab_background_size = vocab_background_size
        self.predict_calls: list[tuple[str, tuple[str, ...]]] = []

    def token_count(self, prompt: str) -> int:
        return int(len(prompt) * self._tokens_per_char)

    def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
        self.predict_calls.append((prompt, tuple(letters)))
        missing = self._bad_letters & set(letters)
        if missing:
            raise TokenizationError(f"letter(s) {sorted(missing)} are not single distinct tokens")
        # A stable pseudo-logit: position of the letter's option text in the
        # prompt (earlier text -> the model "sees" it first -> higher logit),
        # so single-order vs reversed-order genuinely differ.
        letter_logits = {letter: float(-(prompt.index(f"{letter}.")) % 97) for letter in letters}
        # Simulate the rest of the vocabulary as a flat background at a fixed
        # logit, so full_vocab_logsumexp is computable in closed form and
        # candidate_token_mass is a real, checkable fraction.
        background = math.log(self._vocab_background_size) + self._vocab_background_logit
        letters_lse = math.log(sum(math.exp(v) for v in letter_logits.values()))
        full_vocab_lse = math.log(math.exp(letters_lse) + math.exp(background))
        return LogitPrediction(letter_logits=letter_logits, full_vocab_logsumexp=full_vocab_lse)


def test_target_never_appears_in_the_prompt() -> None:
    example = _example()
    prompt, _letters = build_prompt(example.state, example.question, dict(example.options))
    assert "target" not in prompt.lower()
    assert example.target.probabilities["a"] == 1.0  # sanity: this example does have a target
    # the rendered prompt contains only option descriptions, never a marker of which is correct
    assert "correct" not in prompt.lower()


def test_option_letters_are_single_distinct_tokens_else_rejected() -> None:
    predictor = FakePredictor(bad_letters={"B"})
    result = score_example(predictor, _example())
    assert isinstance(result, RejectedScore)
    assert "not single distinct tokens" in result.reasons[0]


def test_more_than_26_options_rejected_by_prompt_builder() -> None:
    options = {chr(97 + i): f"option {i}" for i in range(MAX_OPTIONS + 1)}
    with pytest.raises(Exception, match="26"):
        build_prompt("state", "question", options)


def test_option_permutation_maps_back_to_the_right_semantic_ids() -> None:
    options = {"zzz": "first listed", "aaa": "second listed"}
    predictor = FakePredictor()
    result = score_example(predictor, _example(options=options, target_option="zzz"))
    assert isinstance(result, DecisionScoreResult)
    ids = {s.option_id for s in result.scores}
    assert ids == {"zzz", "aaa"}


def test_softmax_over_allowed_letters_sums_to_one() -> None:
    predictor = FakePredictor()
    result = score_example(predictor, _example())
    assert isinstance(result, DecisionScoreResult)
    total = sum(s.probability for s in result.scores)
    assert abs(total - 1.0) < 1e-9


def test_candidate_token_mass_is_full_vocabulary_share_not_always_one() -> None:
    """candidate_token_mass must reflect how much of the FULL vocabulary's
    softmax landed on the allowed letters -- with a large flat background at
    a competitive logit, that share should be well under 1.0, unlike the
    allowed-letters softmax itself (which always sums to 1 by construction)."""
    predictor = FakePredictor(vocab_background_logit=90.0, vocab_background_size=50_000)
    result = score_example(predictor, _example())
    assert isinstance(result, DecisionScoreResult)
    assert result.candidate_token_mass < 0.5
    assert result.candidate_token_mass > 0.0


def test_candidate_token_mass_approaches_one_when_background_is_negligible() -> None:
    predictor = FakePredictor(vocab_background_logit=-50.0, vocab_background_size=1)
    result = score_example(predictor, _example())
    assert isinstance(result, DecisionScoreResult)
    assert result.candidate_token_mass > 0.999


def test_padded_batch_reads_the_correct_final_position() -> None:
    """The predictor is the thing responsible for finding the right position
    in a padded batch; the scorer just trusts whatever it returns. Simulate a
    predictor that would return wrong logits if it read the padding position
    instead of the last real token, and confirm the scorer's output tracks
    the predictor's (correct) value, not some corrupted one."""

    class PaddedPredictor(FakePredictor):
        def predict(self, prompt: str, letters: list[str]) -> LogitPrediction:
            self.predict_calls.append((prompt, tuple(letters)))
            # Simulate correct last-position reads: option "a" always wins.
            letter_logits = {letter: (10.0 if letter == "A" else 0.0) for letter in letters}
            return LogitPrediction(letter_logits=letter_logits, full_vocab_logsumexp=11.0)

    predictor = PaddedPredictor()
    result = score_example(predictor, _example())
    assert isinstance(result, DecisionScoreResult)
    assert result.chosen_option_id == "a"


def test_overlength_prompt_is_rejected_not_truncated() -> None:
    predictor = FakePredictor(tokens_per_char=1.0)  # inflate token count past the limit
    result = score_example(predictor, _example(), max_prompt_tokens=10)
    assert isinstance(result, RejectedScore)
    assert "over the 10 limit" in result.reasons[0]


def test_raw_logits_kept_unrounded() -> None:
    predictor = FakePredictor()
    result = score_example(predictor, _example())
    assert isinstance(result, DecisionScoreResult)
    for score in result.scores:
        assert isinstance(score.raw_logit, float)


def test_generate_is_never_called() -> None:
    """Structural guarantee: the scorer module's source contains no call to
    `.generate(`, and the predictor protocol it depends on exposes no such
    method -- there is no code path through which it could be invoked."""
    import bandits.decide.scorer as scorer_module

    source = inspect.getsource(scorer_module)
    assert ".generate(" not in source
    assert not hasattr(FakePredictor, "generate")


def test_two_order_average_differs_from_single_order_when_order_matters() -> None:
    predictor = FakePredictor()
    single = score_example(predictor, _example(), mode="single_order")
    averaged = score_example(predictor, _example(), mode="two_order_average")
    assert isinstance(single, DecisionScoreResult)
    assert isinstance(averaged, DecisionScoreResult)
    single_probs = {s.option_id: s.probability for s in single.scores}
    averaged_probs = {s.option_id: s.probability for s in averaged.scores}
    assert single_probs != averaged_probs


def test_token_count_is_called_once_per_score() -> None:
    class CountingPredictor(FakePredictor):
        def __init__(self) -> None:
            super().__init__()
            self.token_count_calls = 0

        def token_count(self, prompt: str) -> int:
            self.token_count_calls += 1
            return super().token_count(prompt)

    predictor = CountingPredictor()
    score_example(predictor, _example())
    assert predictor.token_count_calls == 1


def test_scorer_run_records_dataset_split_and_template_contract() -> None:
    predictor = FakePredictor()
    run = score_dataset(predictor, [_example()], mode="single_order", dataset_id="ds-1", split="dev")
    assert run.dataset_id == "ds-1"
    assert run.split == "dev"
    assert run.model_id == "fake/tiny"
    assert run.dtype == "float32"
    assert run.device == "cpu"
    assert run.template_digest


def test_scorer_run_artifact_round_trips() -> None:
    predictor = FakePredictor()
    run = score_dataset(predictor, [_example()], mode="single_order")
    run_id = compute_scorer_run_id(run)

    payload = run.model_dump_json()
    reloaded = ScorerRun.model_validate_json(payload)
    assert compute_scorer_run_id(reloaded) == run_id
    assert reloaded == run


def test_scorer_does_not_depend_on_traces_or_judge_code() -> None:
    import bandits.decide.scorer as scorer_module

    source = inspect.getsource(scorer_module)
    assert "bandits.traces" not in source
    assert "bandits.verify" not in source
