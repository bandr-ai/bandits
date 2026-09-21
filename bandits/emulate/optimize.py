"""Improve the environment prompt with GEPA. Not weights, not facts.

Retrieval supplies enterprise facts at inference time; this optimizes the stable
instructions that tell the model how to use them and obey the transition
contract. The two solve different halves and neither substitutes for the other.

    fit-optimize sample ──► candidate prompt ──► replay ──► fidelity + critique
             ▲                                                    │
             └──────────── GEPA proposes a revision ◄─────────────┘
                                      │
                    select on a DISJOINT FIT subset
                                      │
                    paired recheck against base; base wins ties
                                      │
                    sealed split opened ONCE, at the end

Three rules carry the design and each closes a specific way the number lies.

**Selection never touches held-out.** Held-out measures the verifier and seals
the fidelity claim. Selecting on it would spend the only clean measurement
available and report the result as if it were still clean, so both selection
samples come from fit.

**A winner that does not survive a paired recheck is discarded.** GEPA optimizes
against a judge and can overfit it; running longer is not guaranteed to keep
improving anything. Ties go to base, because an unproven change is a change.

**Stop on the selection score, never on the fit score.** A prompt that keeps
improving on the sample it was optimized against is the expected behaviour of
overfitting, not evidence of progress.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from typing import Any

from bandits.emulate.fidelity import FidelityReport
from bandits.emulate.models import GroundingTransition
from bandits.traces import Contract


class OptimizationError(RuntimeError):
    """The optimizer could not be built or was asked for something unsound."""


class LeakageError(ValueError):
    """A sample would have been drawn from a partition it must not touch."""


class PromptVersion(Contract):
    """One environment prompt, and what selected it.

    Content-addressed so a prompt cannot be confused with a revision of itself,
    and parented so the chain from base to winner is readable without re-running
    anything.
    """

    prompt_id: str = ""
    text: str
    parent_id: str | None = None
    round_number: int = 0
    critique: str = ""
    """The judge's structured criticism that motivated this revision."""

    select_score: float | None = None
    """Fidelity on the disjoint selection sample. The number that decides."""

    fit_score: float | None = None
    """Fidelity on the sample it was optimized against. Recorded, never decisive."""

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def identified(self) -> PromptVersion:
        return self.replace(prompt_id=f"prompt-{self.digest}")


class OptimizationSample(Contract):
    """The three disjoint sets an optimization run is allowed to see.

    ``sealed`` is named here only so the run can assert it never reads it.
    """

    optimize: tuple[str, ...] = ()
    select: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    """Held-out and sealed transition ids, asserted against at every step."""


class OptimizationRound(Contract):
    """One proposal and what happened to it."""

    round_number: int
    proposed: PromptVersion
    accepted: bool
    reason: str = ""
    incumbent_id: str = ""
    cost: float | None = None


class OptimizationRun(Contract):
    """The complete record of one optimization, including what it rejected.

    A run that reported only its winner would hide how many revisions it took
    and how close the rejected ones were — which is most of what says whether
    the winner is a real improvement or the best of a noisy sample.
    """

    schema_version: int = 1
    base: PromptVersion
    winner: PromptVersion
    rounds: tuple[OptimizationRound, ...] = ()
    sample: OptimizationSample = OptimizationSample()
    seed: int = 0
    budget: int = 0
    stop_reason: str = ""
    model: str = ""
    total_cost: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def improved(self) -> bool:
        """Whether the winner is anything other than the base prompt."""
        return self.winner.prompt_id != self.base.prompt_id

    @property
    def accepted_rounds(self) -> int:
        return sum(1 for row in self.rounds if row.accepted)


def split_optimization_sample(
    transitions: Sequence[GroundingTransition],
    *,
    fit_trace_ids: Sequence[str],
    held_out_trace_ids: Sequence[str] = (),
    sealed_trace_ids: Sequence[str] = (),
    select_fraction: float = 0.3,
    seed: int = 0,
) -> OptimizationSample:
    """Split fit-side transitions into optimize and select, by lineage.

    Whole lineages move, for the same reason the corpus split does: two runs of
    one request carry the same answer, and a prompt selected on a lineage it was
    optimized against is being rechecked on its own training example.
    """
    fit = set(fit_trace_ids)
    forbidden_traces = set(held_out_trace_ids) | set(sealed_trace_ids)

    eligible = [t for t in transitions if t.trace_id in fit and t.trace_id not in forbidden_traces]
    lineages: dict[str, list[str]] = {}
    for transition in eligible:
        lineages.setdefault(transition.lineage_id or transition.trace_id, []).append(
            transition.transition_id
        )

    ordered = sorted(lineages)
    random.Random(seed).shuffle(ordered)
    cut = max(1, round(len(ordered) * select_fraction)) if ordered else 0

    select_ids = [tid for name in ordered[:cut] for tid in lineages[name]]
    optimize_ids = [tid for name in ordered[cut:] for tid in lineages[name]]

    if optimize_ids and select_ids and set(optimize_ids) & set(select_ids):
        raise LeakageError("optimize and select samples overlap")

    return OptimizationSample(
        optimize=tuple(optimize_ids),
        select=tuple(select_ids),
        forbidden=tuple(t.transition_id for t in transitions if t.trace_id in forbidden_traces),
    )


def assert_clean(sample: OptimizationSample) -> None:
    """Refuse a sample that reaches into a partition it must not read."""
    forbidden = set(sample.forbidden)
    touched = (set(sample.optimize) | set(sample.select)) & forbidden
    if touched:
        raise LeakageError(
            f"the optimization sample reaches {len(touched)} held-out or sealed transitions"
        )


EvaluateFn = Callable[[PromptVersion, Sequence[str]], FidelityReport]
"""Replay a prompt over named transitions and report its fidelity."""

ProposeFn = Callable[[PromptVersion, str], PromptVersion | None]
"""Given the incumbent and a critique, propose a revision."""


def objective(report: FidelityReport) -> float:
    """One number for GEPA to climb, built from the structured halves.

    Weighted toward status correctness because a simulator that mistakes an
    error for a success changes what the rollout *is*, while a wrong field
    changes only its detail. Abstention is not penalised here: declining where
    nothing is supported is the behaviour being asked for, and scoring it as
    failure would teach the optimizer to invent.
    """
    parts: list[tuple[float, float]] = []
    if report.status_accuracy is not None:
        parts.append((report.status_accuracy, 2.0))
    if report.field_accuracy is not None:
        parts.append((report.field_accuracy, 1.0))
    if report.delta_accuracy is not None:
        parts.append((report.delta_accuracy, 1.0))
    if not parts:
        return 0.0
    violations = sum(len(row.invariant_violations) for row in report.transitions)
    penalty = min(0.5, 0.05 * violations)
    total = sum(value * weight for value, weight in parts) / sum(w for _, w in parts)
    # Conditional accuracy may not rise by declining hard, supported examples.
    # Correct abstentions are deliberately outside this coverage denominator.
    coverage = report.supported_coverage
    wrong_abstention = report.wrong_abstention_rate
    if coverage is not None:
        total *= coverage
    if wrong_abstention is not None:
        penalty += 0.5 * wrong_abstention
    return max(0.0, total - min(1.0, penalty))


def evaluate_prompt(
    prompt: PromptVersion, sample: Sequence[str], evaluate: EvaluateFn
) -> tuple[float, FidelityReport]:
    report = evaluate(prompt, tuple(sample))
    return objective(report), report


def paired_recheck(
    challenger: PromptVersion,
    incumbent: PromptVersion,
    sample: Sequence[str],
    evaluate: EvaluateFn,
    *,
    margin: float = 0.0,
) -> tuple[bool, str]:
    """Re-measure both on the same transitions before swapping.

    Paired because the sample is small enough that two prompts scored on
    different draws differ by sampling noise alone. Ties go to the incumbent: an
    unproven change is still a change, and every accepted revision is one more
    thing between the measurement and the prompt someone reviewed.
    """
    challenger_score, _ = evaluate_prompt(challenger, sample, evaluate)
    incumbent_score, _ = evaluate_prompt(incumbent, sample, evaluate)
    if challenger_score > incumbent_score + margin:
        return True, f"{challenger_score:.4f} beats {incumbent_score:.4f} on a paired recheck"
    return False, (
        f"{challenger_score:.4f} does not beat {incumbent_score:.4f}; the incumbent is kept"
    )


def optimize_prompt(
    base: PromptVersion,
    *,
    sample: OptimizationSample,
    evaluate: EvaluateFn,
    propose: ProposeFn,
    budget: int = 8,
    patience: int = 3,
    seed: int = 0,
    model: str = "",
    margin: float = 0.0,
) -> OptimizationRun:
    """Propose, measure, and keep only what survives a paired recheck.

    Stops on budget, on convergence, or on ``patience`` consecutive rejections.
    The last is the signal that matters: GEPA proposing revisions that no longer
    survive rechecking is what the end of useful optimization looks like, and
    continuing past it buys overfitting at full price.
    """
    assert_clean(sample)
    if not sample.select:
        raise OptimizationError(
            "selection needs a disjoint sample; optimizing and selecting on one set "
            "measures how well a prompt fits the examples it was written from"
        )

    incumbent = base.identified()
    incumbent_score, _ = evaluate_prompt(incumbent, sample.select, evaluate)
    fit_score, base_report = evaluate_prompt(incumbent, sample.optimize, evaluate)
    incumbent = incumbent.replace(select_score=incumbent_score, fit_score=fit_score)

    rounds: list[OptimizationRound] = []
    rejected_in_a_row = 0
    stop_reason = "budget exhausted"

    for round_number in range(1, budget + 1):
        critique = _critique(base_report)
        proposed = propose(incumbent, critique)
        if proposed is None:
            stop_reason = "the optimizer proposed nothing further"
            break

        candidate = proposed.replace(
            parent_id=incumbent.prompt_id, round_number=round_number, critique=critique
        ).identified()

        if candidate.digest == incumbent.digest:
            rounds.append(
                OptimizationRound(
                    round_number=round_number,
                    proposed=candidate,
                    accepted=False,
                    reason="the proposal was identical to the incumbent",
                    incumbent_id=incumbent.prompt_id,
                )
            )
            stop_reason = "converged: the optimizer stopped changing the prompt"
            break

        # GEPA/reflection may inspect this report. It therefore comes only from
        # the optimize split; selection feedback is never returned to the proposer.
        candidate_fit_score, candidate_fit_report = evaluate_prompt(
            candidate, sample.optimize, evaluate
        )
        candidate = candidate.replace(fit_score=candidate_fit_score)

        accepted, reason = paired_recheck(
            candidate, incumbent, sample.select, evaluate, margin=margin
        )
        candidate_score, _ = evaluate_prompt(candidate, sample.select, evaluate)
        candidate = candidate.replace(select_score=candidate_score)

        rounds.append(
            OptimizationRound(
                round_number=round_number,
                proposed=candidate,
                accepted=accepted,
                reason=reason,
                incumbent_id=incumbent.prompt_id,
            )
        )

        if accepted:
            incumbent = candidate
            base_report = candidate_fit_report
            rejected_in_a_row = 0
        else:
            rejected_in_a_row += 1
            if rejected_in_a_row >= patience:
                stop_reason = f"{patience} consecutive proposals failed their recheck"
                break

    return OptimizationRun(
        base=base.identified(),
        winner=incumbent,
        rounds=tuple(rounds),
        sample=sample,
        seed=seed,
        budget=budget,
        stop_reason=stop_reason,
        model=model,
    )


def _critique(report: FidelityReport) -> str:
    """What went wrong, in terms a prompt revision could act on.

    Built from the structured comparison rather than from a model's impression
    of it: the fields that were missed and the statuses that were inverted are
    exactly what the instruction has to address, and a prose summary would lose
    which they were.
    """
    if not report.transitions:
        return "no transitions were evaluated"

    lines: list[str] = []
    if report.status_accuracy is not None and report.status_accuracy < 1.0:
        inverted = [
            row
            for row in report.transitions
            if row.status_correct is False and row.error_recorded is not None
        ]
        missed_errors = sum(1 for row in inverted if row.error_recorded)
        invented_errors = sum(1 for row in inverted if not row.error_recorded)
        if missed_errors:
            lines.append(
                f"{missed_errors} transitions returned success where the real tool errored"
            )
        if invented_errors:
            lines.append(f"{invented_errors} transitions returned an error the real tool did not")

    missing: dict[str, int] = {}
    wrong: dict[str, int] = {}
    for row in report.transitions:
        for field in row.fields:
            if not field.present:
                missing[field.path] = missing.get(field.path, 0) + 1
            elif not field.correct:
                wrong[field.path] = wrong.get(field.path, 0) + 1
    for label, table in (("omitted", missing), ("got wrong", wrong)):
        worst = sorted(table.items(), key=lambda row: -row[1])[:5]
        if worst:
            named = ", ".join(f"{path} ({count})" for path, count in worst)
            lines.append(f"most often {label}: {named}")

    violations = [v for row in report.transitions for v in row.invariant_violations]
    if violations:
        lines.append(f"{len(violations)} invariant violations, e.g. {violations[0]}")

    if report.premature_disclosure_rate:
        lines.append(
            f"the user policy disclosed unprompted in {report.premature_disclosure_rate:.0%} of turns"
        )

    return "\n".join(lines) or "no structured errors were found on this sample"


def compile_with_gepa(
    student: Any,
    *,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    metric: Any,
    reflection_lm: Any,
    auto: str = "medium",
    seed: int = 0,
    max_metric_calls: int | None = None,
    gepa_factory: Any | None = None,
) -> Any:
    """Optimize a DSPy program through DSPy's public GEPA implementation.

    `gepa_factory` is an injection seam for tests. Production deliberately
    resolves it to ``dspy.GEPA``; the separately installed ``gepa`` package is
    DSPy's implementation detail, not an API this project calls.
    """
    if not trainset:
        raise OptimizationError("GEPA needs a non-empty training set")
    if not valset:
        raise OptimizationError("GEPA needs a disjoint non-empty validation set")
    if set(map(id, trainset)) & set(map(id, valset)):
        raise LeakageError("GEPA trainset and valset contain the same example objects")

    if gepa_factory is None:
        try:
            import dspy
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise OptimizationError(
                "prompt optimization needs the 'emulate' extra: uv sync --extra emulate"
            ) from exc
        gepa_factory = dspy.GEPA

    options: dict[str, Any] = {
        "metric": metric,
        "reflection_lm": reflection_lm,
        "auto": auto,
        "seed": seed,
    }
    if max_metric_calls is not None:
        # DSPy requires choosing either an auto budget or an explicit budget.
        options["auto"] = None
        options["max_metric_calls"] = max_metric_calls
    optimizer = gepa_factory(**options)
    return optimizer.compile(student, trainset=list(trainset), valset=list(valset))


def build_reflective_proposer(
    *, model: str, api_key: str | None = None, max_tokens: int = 4000
) -> ProposeFn:
    """A simple one-shot reflective baseline behind the propose interface.

    This is intentionally not named GEPA. It lacks GEPA's population and
    frontier search and exists only as a cheap baseline for the real
    ``compile_with_gepa`` path above.
    """
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise OptimizationError(
            "prompt optimization needs the 'emulate' extra: uv sync --extra emulate"
        ) from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=api_key or resolve_api_key(),
        temperature=1.0,
        # Proposals are the one place variation is wanted: an optimizer that
        # returns the same revision every round has stopped searching.
        max_tokens=max_tokens,
    )

    class _Revise(dspy.Signature):
        """Revise an environment prompt so it makes fewer of the listed errors.

        Change the instructions only. Do not add facts about the enterprise —
        those come from retrieved real transitions at inference time, and an
        instruction asserting one would be fabricating evidence. Keep the
        output contract and the abstention rule intact."""

        current: str = dspy.InputField(desc="the prompt being revised")
        critique: str = dspy.InputField(desc="what it got wrong, measured")
        revised: str = dspy.OutputField(desc="the improved prompt, in full")

    def propose(incumbent: PromptVersion, critique: str) -> PromptVersion | None:
        with dspy.context(lm=language_model):
            reply = dspy.Predict(_Revise)(current=incumbent.text, critique=critique)
        text = getattr(reply, "revised", "") or ""
        if not text.strip():
            return None
        return PromptVersion(text=text.strip())

    return propose
