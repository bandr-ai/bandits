"""Score a trajectory with a model, for tasks no deterministic check can reach.

Some families record no structured state at all — the agent talks, and nothing
in the trace says whether the talking worked. A rubric judge is the only thing
that can score those, and it enters the system as ordinary evidence with
``MODEL_JUDGMENT`` trust: above the agent's own claim, below anything a human or
an external system asserted, and unpromotable until calibrated against labels.

Uncertainty comes from sampling, not from token probabilities. Measured against
deepseek-v4-flash, a greedy decode reports the runner-up score around eighteen
logprobs down even on a genuinely contestable run — the model reasons itself
into confidence it has not earned. Sampling the same rubric K times separates
the two cases honestly: a clear run returns the same score every time, while a
contestable one splits, and that split is the signal worth spending a label on.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import Field, model_validator

from bandits.analyze.models import Evidence, EvidenceKind, Visibility
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract, SpanKind, SpanStatus, Trace

DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
DEFAULT_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7
"""Greedy decoding hides disagreement; this is what makes the spread meaningful."""

PROMPT_VERSION = 1
SCALE_LOW, SCALE_HIGH = 1, 5

_PROMPT_TEMPLATE = """You grade whether an AI agent completed its task.

{transcript}
Criterion: {criterion}

Reply with ONLY a single digit {low}-{high}. \
{low} = clearly did not meet the criterion, {high} = clearly did."""

_MAX_OUTPUT_CHARS = 400


class JudgeError(RuntimeError):
    """The judge could not be reached or returned nothing usable."""


class PromptDigestMismatch(ValueError):
    """A verdict was produced under a different prompt than the rubric now defines.

    Refused rather than reconciled: a score is only interpretable alongside the
    exact wording that produced it, and silently reusing it under new wording
    would make calibration a measurement of two different things at once.
    """


class Rubric(Contract):
    rubric_id: str
    family_id: str
    criterion: str
    model: str = DEFAULT_MODEL
    prompt_version: int = PROMPT_VERSION
    samples: int = Field(default=DEFAULT_SAMPLES, ge=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0)

    @model_validator(mode="after")
    def validate_rubric(self) -> Rubric:
        if not self.criterion.strip():
            raise ValueError("a rubric needs a criterion to grade against")
        if self.samples > 1 and self.temperature == 0.0:
            raise ValueError(
                "sampling more than once at temperature 0 repeats one answer and "
                "reports false agreement"
            )
        return self

    @property
    def prompt_digest(self) -> str:
        """Pins wording, model and scale. Recorded on every verdict it produces."""
        payload = json.dumps(
            {
                "template": _PROMPT_TEMPLATE,
                "criterion": self.criterion,
                "model": self.model,
                "version": self.prompt_version,
                "scale": [SCALE_LOW, SCALE_HIGH],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def render(self, transcript: str) -> str:
        return _PROMPT_TEMPLATE.format(
            transcript=transcript,
            criterion=self.criterion,
            low=SCALE_LOW,
            high=SCALE_HIGH,
        )


class JudgeVerdict(Contract):
    trace_id: str
    rubric_id: str
    samples: tuple[int, ...]
    score: float | None
    """Mean of the samples, normalized to 0-1. None when nothing parsed."""

    agreement: float
    """Fraction of samples landing on the most common score."""

    prompt_digest: str
    model: str

    @property
    def contested(self) -> bool:
        """Samples disagreed, so this run is worth a human label more than a score."""
        return self.agreement < 1.0

    def as_evidence(self, rubric: Rubric) -> Evidence:
        if self.prompt_digest != rubric.prompt_digest:
            raise PromptDigestMismatch(
                f"verdict for {self.trace_id} was produced under prompt "
                f"{self.prompt_digest}, but the rubric now digests to {rubric.prompt_digest}"
            )
        return Evidence(
            evidence_id=f"ev-{self.trace_id}-rubric-{self.rubric_id}",
            claim=f"rubric_score:{self.rubric_id}",
            value={
                "score": self.score,
                "samples": list(self.samples),
                "agreement": self.agreement,
                "prompt_digest": self.prompt_digest,
                "model": self.model,
            },
            visibility=Visibility.POST_HOC,
            provenance="model",
            # Never strong. An uncalibrated judge has not earned it, and a
            # contested one has actively argued against itself.
            strength="moderate" if not self.contested else "weak",
            kind=EvidenceKind.MODEL_JUDGMENT,
            trace_id=self.trace_id,
        )


class JudgeRun(Contract):
    schema_version: int = 1
    rubric: Rubric
    verdicts: tuple[JudgeVerdict, ...]
    task_set_id: str

    @model_validator(mode="after")
    def validate_digests(self) -> JudgeRun:
        stale = [v.trace_id for v in self.verdicts if v.prompt_digest != self.rubric.prompt_digest]
        if stale:
            raise PromptDigestMismatch(
                f"verdicts produced under a different prompt: {sorted(stale)}"
            )
        return self

    def contested_trace_ids(self) -> tuple[str, ...]:
        return tuple(sorted(v.trace_id for v in self.verdicts if v.contested))


def render_transcript(trace: Trace) -> str:
    """Flatten one episode into the text a judge grades.

    Only what the trace recorded. No label, score, or verifier result is
    rendered: a judge shown the answer is measuring nothing.
    """
    lines = [f"Task: {trace.task or '(no instruction recorded)'}"]
    actions = [
        f"  - {span.name}({json.dumps(span.arguments, default=str)[:200]})"
        f" -> {json.dumps(span.output, default=str)[:200]}"
        # A failed call whose result was simply absent renders as "-> null",
        # which reads as an empty success. The status is the whole difference.
        f"{' [TOOL REPORTED AN ERROR]' if span.status is SpanStatus.ERROR else ''}"
        for span in trace.spans
        if span.kind is SpanKind.TOOL
    ]
    lines.append("Agent actions:")
    lines.extend(actions or ["  (none - no tool was called)"])

    final = next(
        (
            span.output
            for span in reversed(trace.spans)
            if span.kind is SpanKind.MODEL and span.output
        ),
        None,
    )
    lines.append(
        f'Final message: "{str(final)[:_MAX_OUTPUT_CHARS]}"' if final else "Final message: (none)"
    )
    return "\n".join(lines)


Completion = Callable[[str, str, float], str]
"""(model, prompt, temperature) -> the model's reply text."""


def resolve_api_key() -> str:
    """The Fireworks key, read per call so it is never held in an artifact.

    Shared with the family audit, which reaches the same backend through a
    different client: two lookups would drift and one of them would start
    reporting a missing key that is plainly there.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        # Keep local dogfooding one command wide without executing arbitrary
        # shell from .env. Only the one key this backend owns is read.
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "FIREWORKS_API_KEY":
                    api_key = value.strip().strip("'\"")
                    break
    if not api_key:
        raise JudgeError("FIREWORKS_API_KEY is not set and was not found in .env")
    return api_key


def fireworks_completion(model: str, prompt: str, temperature: float) -> str:
    """Call Fireworks with one rendered rubric prompt."""
    api_key = resolve_api_key()

    request = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "temperature": temperature,
                # Reasoning models may spend a substantial part of this budget
                # before emitting their short visible answer. Seven hundred
                # truncated real structured SFT reviews halfway through JSON.
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JudgeError(f"judge request failed: {exc}") from exc
    return payload["choices"][0]["message"]["content"]


def _parse_score(reply: str) -> int | None:
    """Read the graded digit from the reply text.

    Deliberately reads the reply rather than the token stream: this model
    interleaves its reasoning into the tokens, so a digit found there may be one
    it mentioned while thinking rather than the one it settled on.
    """
    for token in reply.strip().split():
        cleaned = token.strip(".,:;()[]'\"")
        if cleaned.isdigit() and SCALE_LOW <= int(cleaned) <= SCALE_HIGH:
            return int(cleaned)
    return None


def judge_trace(
    trace: Trace, rubric: Rubric, *, complete: Completion = fireworks_completion
) -> JudgeVerdict:
    """Sample the rubric K times and report the spread as well as the score."""
    prompt = rubric.render(render_transcript(trace))
    scores = [
        parsed
        for parsed in (
            _parse_score(complete(rubric.model, prompt, rubric.temperature))
            for _ in range(rubric.samples)
        )
        if parsed is not None
    ]

    if not scores:
        return JudgeVerdict(
            trace_id=trace.trace_id,
            rubric_id=rubric.rubric_id,
            samples=(),
            score=None,
            agreement=0.0,
            prompt_digest=rubric.prompt_digest,
            model=rubric.model,
        )

    modal = Counter(scores).most_common(1)[0][1]
    mean = sum(scores) / len(scores)
    return JudgeVerdict(
        trace_id=trace.trace_id,
        rubric_id=rubric.rubric_id,
        samples=tuple(scores),
        score=(mean - SCALE_LOW) / (SCALE_HIGH - SCALE_LOW),
        agreement=modal / len(scores),
        prompt_digest=rubric.prompt_digest,
        model=rubric.model,
    )


def judge_traces(
    traces: Sequence[Trace],
    rubric: Rubric,
    task_set_id: str,
    *,
    complete: Completion = fireworks_completion,
) -> JudgeRun:
    return JudgeRun(
        rubric=rubric,
        task_set_id=task_set_id,
        verdicts=tuple(judge_trace(trace, rubric, complete=complete) for trace in traces),
    )


def compute_judge_run_id(run: JudgeRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode()).hexdigest()
    return f"judge-run-{digest[:16]}"


def save_judge_run(run: JudgeRun, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_judge_run_id(run),
        kind="judge_run",
        parent_artifact_id=run.task_set_id,
        payload=run.model_dump_json().encode(),
        summary={
            "verdicts": len(run.verdicts),
            "contested": len(run.contested_trace_ids()),
        },
    )


def load_judge_run(run_id: str, store: DerivedStore) -> JudgeRun:
    return JudgeRun.model_validate_json(store.read_payload(run_id))
