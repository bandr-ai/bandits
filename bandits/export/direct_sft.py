"""Direct, model-reviewed SFT construction from a normalized corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from bandits.export.sft import _quality_reasons, build_transcript, generating_policy
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract, Trace, TraceCorpus
from bandits.verify.judge import DEFAULT_MODEL, JudgeError, fireworks_completion, render_transcript


class SFTBucket(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


class ModelSFTReview(Contract):
    outcome: Literal["success", "uncertain", "failure"]
    task_clarity: int = Field(ge=1, le=5)
    demonstrated_success: int = Field(ge=1, le=5)
    trajectory_quality: int = Field(ge=1, le=5)
    self_contained: int = Field(ge=1, le=5)
    recommendation: Literal["accept", "review", "reject"]
    rationale: str
    concerns: tuple[str, ...] = ()


class AggregatedSFTReview(Contract):
    model: str
    prompt_digest: str
    samples: tuple[ModelSFTReview, ...]
    attempted_samples: int
    failed_samples: int
    agreement: float
    mean_task_clarity: float
    mean_demonstrated_success: float
    mean_trajectory_quality: float
    mean_self_contained: float
    recommendation: Literal["accept", "review", "reject"]

    @model_validator(mode="after")
    def has_samples(self) -> AggregatedSFTReview:
        if not self.samples:
            raise ValueError("an aggregated SFT review requires at least one parsed sample")
        return self


class DirectSFTCandidate(Contract):
    trace_id: str
    corpus_id: str
    bucket: SFTBucket
    messages: tuple[dict, ...]
    generating_policy: dict
    warnings: tuple[str, ...] = ()
    structural_reasons: tuple[str, ...] = ()
    model_review: AggregatedSFTReview

    def jsonl_row(self) -> dict:
        return self.model_dump(mode="json")


class DirectSFTBundle(Contract):
    schema_version: int = 1
    corpus_id: str
    review_model: str
    requested_trace_ids: tuple[str, ...]
    candidates: tuple[DirectSFTCandidate, ...]

    @property
    def accepted(self) -> tuple[DirectSFTCandidate, ...]:
        return tuple(item for item in self.candidates if item.bucket is SFTBucket.ACCEPTED)

    @property
    def review(self) -> tuple[DirectSFTCandidate, ...]:
        return tuple(item for item in self.candidates if item.bucket is SFTBucket.REVIEW)

    @property
    def rejected(self) -> tuple[DirectSFTCandidate, ...]:
        return tuple(item for item in self.candidates if item.bucket is SFTBucket.REJECTED)


Completion = Callable[[str, str, float], str]

_PROMPT_VERSION = 1
_PROMPT = """You select high-quality demonstrations for supervised fine-tuning.

Review the agent trace below. Judge the whole trajectory, not merely whether the
final answer sounds confident. A successful but wasteful, confused, or overly
context-dependent run may still be unsuitable to imitate.

{transcript}

Return ONLY one JSON object with exactly these fields:
{{
  "outcome": "success" | "uncertain" | "failure",
  "task_clarity": 1-5,
  "demonstrated_success": 1-5,
  "trajectory_quality": 1-5,
  "self_contained": 1-5,
  "recommendation": "accept" | "review" | "reject",
  "rationale": "brief explanation grounded in the trace",
  "concerns": ["specific concern"]
}}

Recommend accept only when the trace provides a clear task, persuasive evidence
of success, and behavior worth imitating. Use review when the evidence is
ambiguous. Never infer missing tool results or unseen repository state.
"""


def _prompt(trace: Trace) -> str:
    return _PROMPT.format(transcript=render_transcript(trace))


def _prompt_digest(model: str) -> str:
    data = json.dumps({"model": model, "prompt": _PROMPT, "version": _PROMPT_VERSION})
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _parse_review(reply: str) -> ModelSFTReview | None:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return ModelSFTReview.model_validate(json.loads(reply[start : end + 1]))
    except (json.JSONDecodeError, ValueError):
        return None


def review_trace(
    trace: Trace,
    *,
    model: str = DEFAULT_MODEL,
    samples: int = 3,
    complete: Completion | None = None,
) -> AggregatedSFTReview:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    completion = complete or fireworks_completion
    parsed_items: list[ModelSFTReview] = []
    failures = 0
    backend_errors: list[str] = []
    for _ in range(samples):
        try:
            item = _parse_review(completion(model, _prompt(trace), 0.7))
        except JudgeError as exc:
            failures += 1
            backend_errors.append(str(exc))
            continue
        if item is None:
            failures += 1
        else:
            parsed_items.append(item)
    parsed = tuple(parsed_items)
    if not parsed:
        if backend_errors and len(backend_errors) == samples:
            raise JudgeError(
                f"all {samples} model review call(s) failed for trace {trace.trace_id}: "
                f"{backend_errors[-1]}"
            )
        raise ValueError(f"model returned no valid SFT review for trace {trace.trace_id}")

    counts = Counter(item.recommendation for item in parsed)
    recommendation, modal_count = counts.most_common(1)[0]

    def mean(field: str) -> float:
        return sum(getattr(item, field) for item in parsed) / len(parsed)

    return AggregatedSFTReview(
        model=model,
        prompt_digest=_prompt_digest(model),
        samples=parsed,
        attempted_samples=samples,
        failed_samples=failures,
        agreement=modal_count / len(parsed),
        mean_task_clarity=mean("task_clarity"),
        mean_demonstrated_success=mean("demonstrated_success"),
        mean_trajectory_quality=mean("trajectory_quality"),
        mean_self_contained=mean("self_contained"),
        recommendation=recommendation,
    )


def _bucket(review: AggregatedSFTReview, structural_reasons: Sequence[str]) -> SFTBucket:
    if structural_reasons or review.recommendation == "reject":
        return SFTBucket.REJECTED
    if (
        review.recommendation == "accept"
        and len(review.samples) / review.attempted_samples >= 2 / 3
        and review.agreement >= 2 / 3
        and review.mean_demonstrated_success >= 4
        and review.mean_trajectory_quality >= 4
        and review.mean_task_clarity >= 4
    ):
        return SFTBucket.ACCEPTED
    return SFTBucket.REVIEW


def build_direct_sft(
    corpus: TraceCorpus,
    corpus_id: str,
    *,
    trace_ids: Sequence[str] = (),
    model: str = DEFAULT_MODEL,
    samples: int = 3,
    complete: Completion | None = None,
) -> DirectSFTBundle:
    by_id = {trace.trace_id: trace for trace in corpus.traces}
    requested = tuple(dict.fromkeys(trace_ids)) or tuple(by_id)
    missing = [trace_id for trace_id in requested if trace_id not in by_id]
    if missing:
        raise ValueError(f"corpus contains no trace(s): {', '.join(missing)}")

    candidates: list[DirectSFTCandidate] = []
    for trace_id in requested:
        trace = by_id[trace_id]
        messages, defects, warnings = build_transcript(trace)
        structural = list(defects)
        if not trace.task:
            structural.append("trace has no user task")
        structural.extend(_quality_reasons(trace, messages, max_steps=max(1, len(trace.spans))))
        review = review_trace(trace, model=model, samples=samples, complete=complete)
        candidates.append(
            DirectSFTCandidate(
                trace_id=trace_id,
                corpus_id=corpus_id,
                bucket=_bucket(review, structural),
                messages=tuple(message.as_chat_message() for message in messages),
                generating_policy=generating_policy(trace),
                warnings=warnings,
                structural_reasons=tuple(dict.fromkeys(structural)),
                model_review=review,
            )
        )
    return DirectSFTBundle(
        corpus_id=corpus_id,
        review_model=model,
        requested_trace_ids=requested,
        candidates=tuple(candidates),
    )


def compute_direct_sft_id(bundle: DirectSFTBundle) -> str:
    digest = hashlib.sha256(bundle.model_dump_json().encode()).hexdigest()[:16]
    return f"direct-sft-{digest}"


def save_direct_sft(bundle: DirectSFTBundle, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_direct_sft_id(bundle),
        kind="direct_sft",
        parent_artifact_id=bundle.corpus_id,
        payload=bundle.model_dump_json().encode(),
        summary={
            "accepted": len(bundle.accepted),
            "review": len(bundle.review),
            "rejected": len(bundle.rejected),
        },
    )


def _write_jsonl(path: Path, rows: Sequence[DirectSFTCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row.jsonl_row(), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_direct_sft(bundle: DirectSFTBundle, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "accepted": output / "sft.jsonl",
        "review": output / "review.jsonl",
        "rejected": output / "rejected.jsonl",
        "report": output / "selection-report.json",
    }
    _write_jsonl(paths["accepted"], bundle.accepted)
    _write_jsonl(paths["review"], bundle.review)
    _write_jsonl(paths["rejected"], bundle.rejected)
    report = {
        "corpus_id": bundle.corpus_id,
        "review_model": bundle.review_model,
        "requested": len(bundle.requested_trace_ids),
        "accepted": len(bundle.accepted),
        "review": len(bundle.review),
        "rejected": len(bundle.rejected),
    }
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return paths
