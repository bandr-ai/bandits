"""Tests that prompt optimization cannot leak or win by declining work."""

from bandits.diagnose.fidelity import (
    FidelityReport,
    FieldComparison,
    TransitionFidelity,
)
from bandits.diagnose.optimize import (
    OptimizationSample,
    PromptVersion,
    compile_with_gepa,
    objective,
    optimize_prompt,
)


def _report(*, path: str, correct: bool = False) -> FidelityReport:
    return FidelityReport(
        transitions=(
            TransitionFidelity(
                transition_id=path,
                trace_id=path,
                fields=(
                    FieldComparison(
                        path=path,
                        predicted="right" if correct else "wrong",
                        recorded="right",
                    ),
                ),
                status_correct=True,
            ),
        )
    )


def test_next_critique_never_reads_selection_feedback() -> None:
    critiques: list[str] = []

    def evaluate(prompt: PromptVersion, sample: tuple[str, ...]) -> FidelityReport:
        is_select = sample == ("select",)
        # Every proposed prompt beats the incumbent on selection. The marker lets
        # us observe which split supplied the following round's critique.
        return _report(
            path="selection-secret" if is_select else "training-error",
            correct=is_select and bool(prompt.round_number),
        )

    def propose(_incumbent: PromptVersion, critique: str) -> PromptVersion:
        critiques.append(critique)
        return PromptVersion(text=f"revision-{len(critiques)}")

    optimize_prompt(
        PromptVersion(text="base"),
        sample=OptimizationSample(optimize=("train",), select=("select",)),
        evaluate=evaluate,
        propose=propose,
        budget=2,
    )

    assert len(critiques) == 2
    assert "training-error" in critiques[1]
    assert "selection-secret" not in critiques[1]


def test_wrong_abstention_reduces_the_optimization_objective() -> None:
    accurate = _report(path="x", correct=True)
    gaming = FidelityReport(
        transitions=(
            TransitionFidelity(
                transition_id="x", trace_id="x", abstained=True, abstain_correct=False
            ),
        )
    )
    assert objective(gaming) < objective(accurate)


def test_gepa_compiles_through_the_dspy_optimizer_contract() -> None:
    seen = {}
    compiled = object()

    class FakeGEPA:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def compile(self, student, *, trainset, valset):
            seen["compile"] = (student, trainset, valset)
            return compiled

    student = object()
    metric = object()
    reflection_lm = object()
    result = compile_with_gepa(
        student,
        trainset=("train",),
        valset=("select",),
        metric=metric,
        reflection_lm=reflection_lm,
        auto="light",
        seed=7,
        gepa_factory=FakeGEPA,
    )

    assert result is compiled
    assert seen["init"]["metric"] is metric
    assert seen["init"]["reflection_lm"] is reflection_lm
    assert seen["init"]["auto"] == "light"
    assert seen["init"]["seed"] == 7
    assert seen["compile"] == (student, ["train"], ["select"])
