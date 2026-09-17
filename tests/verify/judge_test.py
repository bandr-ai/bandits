"""Rubric judge tests. No network: the completion callable is injected."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path

import pytest
from pydantic import ValidationError

from bandits.analyze.models import EvidenceKind, Visibility
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.verify.execute import execute_verifier
from bandits.verify.judge import (
    JudgeRun,
    PromptDigestMismatch,
    Rubric,
    judge_trace,
    judge_traces,
    load_judge_run,
    render_transcript,
    save_judge_run,
)
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierMode,
    VerifierSpec,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _replies(*values: str):
    """A stub completion returning the given replies in order, then repeating."""
    stream = cycle(values)
    return lambda model, prompt, temperature: next(stream)


def _rubric(**overrides) -> Rubric:
    return Rubric(
        **{
            "rubric_id": "resolved-v1",
            "family_id": "family-one",
            "criterion": "the agent resolved the customer's issue",
            **overrides,
        }
    )


def _trace(trace_id: str):
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    return next(t for t in corpus.traces if t.trace_id == trace_id)


def test_a_failed_tool_call_is_visible_to_the_judge() -> None:
    """Its absent result renders as '-> null', which otherwise reads as success."""
    transcript = render_transcript(_trace("refund-4"))

    assert "[TOOL REPORTED AN ERROR]" in transcript


def test_the_transcript_carries_only_what_the_trace_recorded() -> None:
    """A judge shown an outcome label would be measuring nothing."""
    transcript = render_transcript(_trace("refund-1"))

    assert "Refund order 7001" in transcript
    assert "lookup_order" in transcript
    for leak in ("verdict", "human_label", "success", "failure"):
        assert leak not in transcript.lower()


def test_an_episode_with_no_tool_call_says_so() -> None:
    assert "(none - no tool was called)" in render_transcript(_trace("vague-1"))


def test_unanimous_samples_are_not_contested() -> None:
    verdict = judge_trace(_trace("vague-1"), _rubric(), complete=_replies("1"))

    assert verdict.samples == (1, 1, 1, 1, 1)
    assert verdict.score == 0.0
    assert verdict.agreement == 1.0
    assert not verdict.contested


def test_split_samples_are_contested_and_downgraded() -> None:
    rubric = _rubric()
    verdict = judge_trace(_trace("addr-1"), rubric, complete=_replies("5", "1", "5", "1", "1"))

    assert verdict.contested
    assert verdict.agreement == pytest.approx(0.6)
    assert verdict.as_evidence(rubric).strength == "weak"


def test_a_judgement_is_model_evidence_and_never_strong() -> None:
    rubric = _rubric()
    evidence = judge_trace(_trace("vague-1"), rubric, complete=_replies("1")).as_evidence(rubric)

    assert evidence.kind is EvidenceKind.MODEL_JUDGMENT
    assert evidence.provenance == "model"
    assert evidence.visibility is Visibility.POST_HOC
    assert evidence.strength != "strong"


def test_replies_with_no_usable_digit_score_unknown() -> None:
    verdict = judge_trace(_trace("vague-1"), _rubric(), complete=_replies("I cannot say"))

    assert verdict.samples == ()
    assert verdict.score is None


def test_a_digit_is_read_from_the_reply_not_the_reasoning() -> None:
    """This model interleaves reasoning, so the first digit it utters may not be its answer."""
    verdict = judge_trace(_trace("vague-1"), _rubric(samples=1), complete=_replies("4"))

    assert verdict.samples == (4,)


def test_changing_the_criterion_changes_the_prompt_digest() -> None:
    assert _rubric().prompt_digest != _rubric(criterion="something else").prompt_digest


def test_a_verdict_from_another_prompt_is_refused() -> None:
    """A score is only interpretable alongside the wording that produced it."""
    rubric = _rubric()
    verdict = judge_trace(_trace("vague-1"), rubric, complete=_replies("1"))

    with pytest.raises(PromptDigestMismatch):
        verdict.as_evidence(_rubric(criterion="a different question entirely"))


def test_a_run_cannot_mix_verdicts_from_different_prompts() -> None:
    rubric = _rubric()
    verdict = judge_trace(_trace("vague-1"), rubric, complete=_replies("1"))

    with pytest.raises(ValidationError):
        JudgeRun(
            rubric=_rubric(criterion="a different question entirely"),
            task_set_id="ts-1",
            verdicts=(verdict,),
        )


def test_sampling_repeatedly_at_temperature_zero_is_refused() -> None:
    """It would repeat one answer and report unanimous agreement it had not earned."""
    with pytest.raises(ValidationError, match="false agreement"):
        _rubric(samples=5, temperature=0.0)


def _rubric_spec(threshold: float) -> VerifierSpec:
    return VerifierSpec(
        verifier_id="verifier-rubric",
        family_id="family-one",
        task_set_id="ts-1",
        mode=VerifierMode.REPLAY,
        inputs=(),
        checks=(
            CheckSpec(
                check_id="check-rubric",
                claim="rubric_score:resolved-v1",
                operator=CheckOperator.RUBRIC_AT_LEAST,
                expected=threshold,
                supporting_evidence_ids=(),
                description="d",
                evidence_kind=EvidenceKind.MODEL_JUDGMENT,
            ),
        ),
        unknown_when=(),
        blind_spots=(),
        gaming_hypotheses=(),
    )


def test_a_confident_judgement_is_thresholded() -> None:
    rubric = _rubric()
    high = judge_trace(_trace("refund-1"), rubric, complete=_replies("5")).as_evidence(rubric)
    low = judge_trace(_trace("vague-1"), rubric, complete=_replies("1")).as_evidence(rubric)

    assert execute_verifier(_rubric_spec(0.5), (high,)).score == 1.0
    assert execute_verifier(_rubric_spec(0.5), (low,)).score == 0.0


def test_a_contested_judgement_is_unknown_not_thresholded() -> None:
    """refund-4 is a failed refund the judge split on before it could see the error."""
    rubric = _rubric()
    evidence = judge_trace(
        _trace("refund-4"), rubric, complete=_replies("2", "5", "2", "5", "5")
    ).as_evidence(rubric)

    result = execute_verifier(_rubric_spec(0.5), (evidence,))

    assert result.score is None, "a mean of 0.7 would have passed a failed refund"
    assert "disagreed with itself" in result.subscores[0].details["reason"]


def test_a_judge_run_round_trips_and_lists_what_needs_a_human(tmp_path) -> None:
    store = DerivedStore(tmp_path / ".bandits")
    corpus = load_corpus(FIXTURES / "traces.support.otlp.jsonl", "otlp")
    traces = [t for t in corpus.traces if t.trace_id in ("vague-1", "vague-2")]
    run = judge_traces(traces, _rubric(), "ts-1", complete=_replies("1", "1", "1", "1", "1", "5"))

    envelope = save_judge_run(run, store)

    assert envelope.kind == "judge_run"
    assert load_judge_run(envelope.artifact_id, store) == run
    assert run.contested_trace_ids() == ("vague-2",)
