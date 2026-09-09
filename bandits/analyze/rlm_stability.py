"""Compare independent mining runs, and report how much of the taxonomy is real.

One run produces a taxonomy that looks plausible whether or not it means
anything, because a language model asked to name groups will always name groups.
The only evidence that a family is a property of the corpus rather than of one
sampling path is that independent runs, seeded differently, keep putting the same
traces together.

Everything here is measured on trace co-assignment, never on generated names. Two
runs will call the same family different things, and comparing names would report
disagreement that is purely lexical while missing two identically-named families
that hold different traces. A pair of traces placed together is a claim both runs
can actually agree or disagree about.

Deliberately not a verdict. This computes the numbers the go/no-go decision is
made from — stability, recurring disagreements, persistently unplaced traces —
and stops there. A threshold baked in here would decide the experiment's outcome
in the code that measures it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from pydantic import Field, model_validator

from bandits.analyze.rlm_models import AssignmentRun, AssignmentStatus, FrozenTaxonomy
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract


class PairDisagreement(Contract):
    """Two traces that some runs grouped and others separated.

    The unit of a split/merge disagreement. A pair every run agrees on tells you
    nothing about which run to trust; these are where the taxonomies genuinely
    differ, and they are what a reviewer should read first.
    """

    trace_ids: tuple[str, str]
    together: int = Field(ge=0)
    apart: int = Field(ge=0)

    @model_validator(mode="after")
    def pair_is_ordered_and_contested(self) -> PairDisagreement:
        left, right = self.trace_ids
        if left >= right:
            raise ValueError("a disagreement records its trace pair in sorted order")
        if self.together == 0 or self.apart == 0:
            raise ValueError("a pair every run agreed on is not a disagreement; it is agreement")
        return self

    @property
    def contested(self) -> float:
        """How close to evenly split the runs were. 1.0 is maximal disagreement."""
        total = self.together + self.apart
        return 2 * min(self.together, self.apart) / total


class UnplacedTrace(Contract):
    """A trace that stayed ambiguous or uncovered across runs.

    Kept as its own finding. A trace no independent run could place is either a
    genuine gap in the taxonomy or a request that is not one task, and either way
    it must not disappear into a coverage percentage.
    """

    trace_id: str
    ambiguous: int = Field(ge=0)
    uncovered: int = Field(ge=0)
    runs: int = Field(ge=1)

    @property
    def always_unplaced(self) -> bool:
        return self.ambiguous + self.uncovered == self.runs


class StabilityReport(Contract):
    """What repeated independent runs agreed and disagreed about.

    Its own artifact, parented to the analysis rather than to any one run: it is
    a statement about the corpus and the method, and parenting it to a single
    run would imply that run was the reference the others were judged against.
    """

    schema_version: int = 1
    analysis_id: str
    assignment_run_ids: tuple[str, ...]
    runs: int = Field(ge=2)
    stable_assignment_fraction: float = Field(ge=0.0, le=1.0)
    """Fraction of traces every run placed, and placed with the same partners.

    Defined on partners rather than on contract ids because ids are not
    comparable across runs. A trace is stable when the set of traces sharing its
    family is identical in every run that placed it.
    """

    pairwise_agreement: float = Field(ge=0.0, le=1.0)
    """Mean Jaccard over run pairs: shared co-assignments / all co-assignments.

    Jaccard rather than a raw match rate because most trace pairs are in
    different families in every run, and a measure counting those agreements
    would report near-perfect stability for any two taxonomies at all.
    """

    disagreements: tuple[PairDisagreement, ...] = ()
    unplaced: tuple[UnplacedTrace, ...] = ()
    recurring_contracts: tuple[tuple[str, int], ...] = ()
    """Contract fingerprints and how many runs proposed them, most-recurring first.

    A semantic claim several independent runs arrive at separately is the
    strongest evidence this path produces: it is a family the corpus contains
    rather than one a sampling path invented.
    """

    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def runs_match_the_ids(self) -> StabilityReport:
        if len(self.assignment_run_ids) != self.runs:
            raise ValueError("stability report counts a different number of runs than it names")
        if len(set(self.assignment_run_ids)) != len(self.assignment_run_ids):
            raise ValueError(
                "stability report names the same run twice; comparing a run with itself "
                "reports agreement that was never tested"
            )
        return self


def _partners(run: AssignmentRun) -> dict[str, frozenset[str]]:
    """Each assigned trace mapped to the other traces sharing its family."""
    partners: dict[str, frozenset[str]] = {}
    for traces in run.members().values():
        members = frozenset(traces)
        for trace_id in traces:
            partners[trace_id] = members - {trace_id}
    return partners


def compare_runs(
    runs: Sequence[AssignmentRun],
    run_ids: Sequence[str],
    *,
    analysis_id: str,
    taxonomies: Sequence[FrozenTaxonomy] = (),
    max_disagreements: int = 50,
) -> StabilityReport:
    """Measure agreement across independent runs over the same corpus.

    Runs must not share a taxonomy: two assignments made against identical
    contracts measure the classifier's repeatability, which is a real question
    but not this one. The plan asks whether *discovery* is stable, so a shared
    taxonomy is rejected rather than quietly averaged in.
    """
    if len(runs) < 2:
        raise ValueError("stability needs at least two runs to compare")
    if len(runs) != len(run_ids):
        raise ValueError("every run must be named by exactly one id")

    taxonomy_ids = [run.taxonomy_id for run in runs]
    if len(set(taxonomy_ids)) != len(taxonomy_ids):
        raise ValueError(
            "two runs share a taxonomy; that measures assignment repeatability, not "
            "whether independent discovery finds the same families"
        )
    if len({run.analysis_id for run in runs}) != 1:
        raise ValueError("runs over different analyses are not comparable")

    limitations: list[str] = []
    views = {run.view for run in runs}
    if len(views) > 1:
        limitations.append(
            "these runs used different trace views, so disagreement between them "
            "confounds the arm with the method"
        )

    partner_maps = [_partners(run) for run in runs]
    placed_everywhere = set.intersection(*(set(p) for p in partner_maps)) if partner_maps else set()
    stable = sum(
        1 for trace_id in placed_everywhere if len({p[trace_id] for p in partner_maps}) == 1
    )

    # Denominator is every trace any run classified, not just the ones all runs
    # placed: a trace one run left uncovered is an instability, and dropping it
    # would let a run that placed almost nothing score as perfectly stable.
    classified = {
        a.trace_id
        for run in runs
        for a in run.assignments
        if a.status is not AssignmentStatus.UNREADABLE
    }
    stable_fraction = stable / len(classified) if classified else 0.0

    pair_sets = [run.co_assignment_pairs() for run in runs]
    agreements = []
    for left, right in combinations(pair_sets, 2):
        union = left | right
        agreements.append(len(left & right) / len(union) if union else 1.0)
    pairwise = sum(agreements) / len(agreements) if agreements else 1.0

    together = Counter(pair for pairs in pair_sets for pair in pairs)
    disagreements = []
    for pair, count in together.items():
        left, right = pair
        # Only runs that placed *both* traces get a vote. A run that left one
        # uncovered did not separate them; it declined to say, and counting that
        # as "apart" would manufacture disagreement out of missing coverage.
        voting = sum(1 for p in partner_maps if left in p and right in p)
        apart = voting - count
        if count and apart > 0:
            disagreements.append(PairDisagreement(trace_ids=pair, together=count, apart=apart))
    disagreements.sort(key=lambda d: (-d.contested, d.trace_ids))

    unplaced: list[UnplacedTrace] = []
    for trace_id in sorted(classified):
        ambiguous = sum(
            1
            for run in runs
            for a in run.assignments
            if a.trace_id == trace_id and a.status is AssignmentStatus.AMBIGUOUS
        )
        uncovered = sum(
            1
            for run in runs
            for a in run.assignments
            if a.trace_id == trace_id and a.status is AssignmentStatus.UNCOVERED
        )
        if ambiguous or uncovered:
            unplaced.append(
                UnplacedTrace(
                    trace_id=trace_id,
                    ambiguous=ambiguous,
                    uncovered=uncovered,
                    runs=len(runs),
                )
            )

    recurring: Counter[str] = Counter()
    for taxonomy in taxonomies:
        recurring.update(taxonomy.fingerprints())
    recurring_contracts = tuple(
        (fingerprint, count)
        for fingerprint, count in sorted(recurring.items(), key=lambda kv: (-kv[1], kv[0]))
        if count > 1
    )

    if taxonomies and len(taxonomies) != len(runs):
        limitations.append(
            "contract recurrence was computed over fewer taxonomies than there are runs"
        )
    if not taxonomies:
        limitations.append(
            "no taxonomies were supplied, so recurring semantic contracts were not measured"
        )
    if len(disagreements) > max_disagreements:
        limitations.append(
            f"{len(disagreements)} contested pairs were found and the {max_disagreements} "
            "most contested are reported"
        )

    return StabilityReport(
        analysis_id=analysis_id,
        assignment_run_ids=tuple(run_ids),
        runs=len(runs),
        stable_assignment_fraction=stable_fraction,
        pairwise_agreement=pairwise,
        disagreements=tuple(disagreements[:max_disagreements]),
        unplaced=tuple(unplaced),
        recurring_contracts=recurring_contracts,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def compute_stability_report_id(report: StabilityReport) -> str:
    digest = hashlib.sha256(report.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-stability-{digest[:16]}"


def save_stability_report(report: StabilityReport, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_stability_report_id(report),
        kind="rlm_stability_report",
        parent_artifact_id=report.analysis_id,
        payload=report.model_dump_json().encode("utf-8"),
        summary={
            "runs": report.runs,
            "disagreements": len(report.disagreements),
            "unplaced": len(report.unplaced),
            "recurring_contracts": len(report.recurring_contracts),
        },
    )


def load_stability_report(report_id: str, store: DerivedStore) -> StabilityReport:
    return StabilityReport.model_validate_json(store.read_payload(report_id))
