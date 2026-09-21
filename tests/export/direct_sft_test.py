from __future__ import annotations

import json
from datetime import UTC, datetime

from bandits.export.direct_sft import SFTBucket, build_direct_sft, write_direct_sft
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus
from bandits.verify.judge import JudgeError

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _trace(trace_id: str = "run-1", *, error: bool = False) -> Trace:
    return Trace(
        trace_id=trace_id,
        source="test",
        source_digest="digest",
        task="Fix the broken test",
        spans=(
            Span(
                span_id="answer",
                kind=SpanKind.MODEL,
                name="claude-test",
                started_at=NOW,
                ended_at=NOW,
                status=SpanStatus.ERROR if error else SpanStatus.OK,
                output="Implemented the fix and the tests pass.",
            ),
        ),
    )


def _reply(recommendation: str = "accept", success: int = 5, quality: int = 5) -> str:
    return json.dumps(
        {
            "outcome": "success" if recommendation == "accept" else "uncertain",
            "task_clarity": 5,
            "demonstrated_success": success,
            "trajectory_quality": quality,
            "self_contained": 4,
            "recommendation": recommendation,
            "rationale": "The trace records a clear task and a completed response.",
            "concerns": [],
        }
    )


def test_direct_sft_accepts_only_when_model_and_structure_agree(tmp_path) -> None:
    corpus = TraceCorpus(source="test", traces=(_trace(),))
    bundle = build_direct_sft(
        corpus,
        "corpus-test",
        samples=3,
        complete=lambda model, prompt, temperature: _reply(),
    )

    assert bundle.candidates[0].bucket is SFTBucket.ACCEPTED
    assert len(bundle.candidates[0].model_review.samples) == 3
    paths = write_direct_sft(bundle, tmp_path / "dataset")
    assert '"messages"' in paths["accepted"].read_text()
    assert paths["review"].read_text() == ""
    assert paths["rejected"].read_text() == ""


def test_direct_sft_routes_uncertain_model_decision_to_review() -> None:
    corpus = TraceCorpus(source="test", traces=(_trace(),))
    bundle = build_direct_sft(
        corpus,
        "corpus-test",
        complete=lambda model, prompt, temperature: _reply("review", success=3, quality=3),
    )

    assert bundle.candidates[0].bucket is SFTBucket.REVIEW


def test_structural_failure_rejects_even_when_model_says_accept() -> None:
    corpus = TraceCorpus(source="test", traces=(_trace(error=True),))
    bundle = build_direct_sft(
        corpus,
        "corpus-test",
        complete=lambda model, prompt, temperature: _reply(),
    )

    candidate = bundle.candidates[0]
    assert candidate.bucket is SFTBucket.REJECTED
    assert "episode contains an error or recovery path" in candidate.structural_reasons


def test_direct_sft_refuses_an_unknown_selected_trace() -> None:
    corpus = TraceCorpus(source="test", traces=(_trace(),))
    try:
        build_direct_sft(corpus, "corpus-test", trace_ids=("missing",))
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown trace should have been refused")


def test_direct_sft_tolerates_one_failed_model_sample() -> None:
    replies = iter((JudgeError("timeout"), _reply(), _reply()))

    def complete(model: str, prompt: str, temperature: float) -> str:
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result

    bundle = build_direct_sft(
        TraceCorpus(source="test", traces=(_trace(),)),
        "corpus-test",
        samples=3,
        complete=complete,
    )

    review = bundle.candidates[0].model_review
    assert review.attempted_samples == 3
    assert review.failed_samples == 1
    assert len(review.samples) == 2


def test_direct_sft_reports_backend_failure_when_every_sample_fails() -> None:
    def fail(model: str, prompt: str, temperature: float) -> str:
        raise JudgeError("HTTP 403")

    try:
        build_direct_sft(
            TraceCorpus(source="test", traces=(_trace(),)),
            "corpus-test",
            samples=2,
            complete=fail,
        )
    except JudgeError as exc:
        assert "all 2 model review call(s) failed" in str(exc)
        assert "HTTP 403" in str(exc)
    else:
        raise AssertionError("an unavailable model backend should fail the build")
