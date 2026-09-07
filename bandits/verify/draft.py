"""Propose small deterministic replay verifiers from observed terminal evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    EvidenceKind,
    TaskFamily,
    TaskSet,
    Visibility,
    kind_rank,
)
from bandits.labels import LabelSet, Verdict
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.verify.execute import execute_verifier
from bandits.verify.models import (
    CandidateStats,
    CheckOperator,
    CheckSpec,
    Result,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _spec_id(family_id: str, claim: str, expected: Any) -> str:
    payload = f"{family_id}\0{claim}\0{_stable_value(expected)}"
    return f"verifier-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _composite_id(family_id: str, checks: tuple[CheckSpec, ...]) -> str:
    """Identity of a conjunction is the set of checks in it, in a fixed order."""
    parts = sorted(f"{check.claim}\0{_stable_value(check.expected)}" for check in checks)
    payload = "\0".join([family_id, *parts])
    return f"verifier-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _eligible(family: TaskFamily, analysis: CorpusAnalysis) -> list[Evidence]:
    """Everything a verifier may read about the family's fit traces.

    Anything except ``at_start``: a verifier runs after the episode, so the
    initial state a tool observed and a score recorded afterwards are both fair
    game. Restricting this to terminal evidence made before/after invariants
    inexpressible and hid recorded evaluator scores entirely. The at_start
    restriction belongs to the prompt, and only to the prompt.
    """
    trace_ids = set(family.fit_trace_ids or family.trace_ids)
    return [
        item
        for item in analysis.evidence
        if item.trace_id in trace_ids and item.visibility is not Visibility.AT_START
    ]


def _proposal(
    *,
    family: TaskFamily,
    task_set_id: str,
    claim: str,
    operator: CheckOperator,
    expected: Any,
    evidence: list[Evidence],
    description: str,
    blind_spots: tuple[str, ...],
    gaming: tuple[str, ...],
    evidence_kind: EvidenceKind,
) -> VerifierSpec:
    verifier_id = _spec_id(family.family_id, claim, expected)
    return VerifierSpec(
        verifier_id=verifier_id,
        family_id=family.family_id,
        task_set_id=task_set_id,
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=(f"terminal_evidence:{claim}",),
        checks=(
            CheckSpec(
                check_id=f"check-{verifier_id.removeprefix('verifier-')}",
                claim=claim,
                operator=operator,
                expected=expected,
                supporting_evidence_ids=tuple(sorted(e.evidence_id for e in evidence)),
                description=description,
                evidence_kind=evidence_kind,
            ),
        ),
        unknown_when=(f"terminal evidence for {claim!r} is absent",),
        blind_spots=blind_spots,
        gaming_hypotheses=gaming,
    )


_MAX_VALUES_PER_KEY = 2
"""Competing hypotheses for one field are useful; a long tail of them is noise."""

_MAX_COMPOSITE_SOURCES = 4
"""How many standalone candidates conjunctions may be built from.

Bounded because the pairs grow quadratically and every one of them costs a pass
over the family. The four best-ranked standalone checks are where a useful
conjunction comes from; a pair drawn from further down is two weak checks."""

_UNCALIBRATED_OFF_PARTITION = (
    "every adjudicated label names a trace outside the side these checks were drafted "
    "from, so nothing below was measured against an outcome; the candidates are "
    "frequency-based hypotheses"
)

_UNCALIBRATED = (
    "no adjudicated label was available for this family; every candidate below is a "
    "frequency-based hypothesis, and its support says how often a value was recorded, "
    "not that the runs recording it succeeded"
)


def _accepts(result: Result | None) -> bool | None:
    """Whether a verifier said this trace passed, or could not say at all.

    None is not a failure. A check that cannot see what it needs has established
    nothing, and counting its silence as a rejection would make an absent field
    look like evidence against the run.
    """
    if result is None or result.score is None:
        return None
    return result.score >= 1.0


def _measure(
    spec: VerifierSpec,
    evidence_by_trace: dict[str, tuple[Evidence, ...]],
    verdicts: dict[str, Verdict],
    *,
    derivation: str,
    rationale: str = "",
) -> tuple[CandidateStats, dict[str, bool | None]]:
    """Run a candidate over the family it was drawn from and count what happened.

    Measured by executing the spec rather than by re-reading the evidence it was
    built from: the numbers a reader sees have to come from the same code that
    will score this verifier later, or they describe a check nobody will run.
    """
    outcomes: dict[str, bool | None] = {}
    for trace_id, evidence in evidence_by_trace.items():
        outcomes[trace_id] = _accepts(execute_verifier(spec, evidence))

    scorable = sum(1 for verdict in outcomes.values() if verdict is not None)
    labeled = {
        trace_id: verdict for trace_id, verdict in verdicts.items() if trace_id in evidence_by_trace
    }
    successes = [t for t, verdict in labeled.items() if verdict is Verdict.SUCCESS]
    failures = [t for t, verdict in labeled.items() if verdict is Verdict.FAILURE]
    stats = CandidateStats(
        verifier_id=spec.verifier_id,
        derivation=derivation,  # type: ignore[arg-type]
        considered=len(evidence_by_trace),
        scorable=scorable,
        unknown=len(evidence_by_trace) - scorable,
        accepted=sum(1 for verdict in outcomes.values() if verdict is True),
        rejected=sum(1 for verdict in outcomes.values() if verdict is False),
        labeled_successes=len(successes),
        labeled_failures=len(failures),
        success_support=sum(1 for t in successes if outcomes.get(t) is True),
        false_negatives=sum(1 for t in successes if outcomes.get(t) is False),
        failure_rejection=sum(1 for t in failures if outcomes.get(t) is False),
        false_positives=sum(1 for t in failures if outcomes.get(t) is True),
        unknown_on_labeled=sum(1 for t in labeled if outcomes.get(t) is None),
        rationale=rationale,
    )
    return stats, outcomes


def _rank(spec: VerifierSpec, stats: CandidateStats) -> tuple:
    """Authority first, then how well the candidate separates outcomes, then reach.

    Evidence authority leads because a check reading a stronger class of evidence
    is making a stronger claim, and no amount of agreement with labels promotes a
    check that reads only what the agent said about itself. Discrimination is
    zero for every candidate when nothing is labeled, so an uncalibrated draft
    keeps the authority order it already had.
    """
    return (
        -kind_rank(spec.weakest_evidence_kind),
        -round(stats.discrimination, 6),
        -round(stats.coverage, 6),
        spec.verifier_id,
    )


def _identifier_keys(grouped: dict[tuple[str, str], list[Evidence]]) -> list[str]:
    """Keys whose value is unique to every trace that recorded it."""
    traces_per_key: dict[str, set[str]] = {}
    values_per_key: dict[str, set[str]] = {}
    for (key, value), support in grouped.items():
        traces_per_key.setdefault(key, set()).update(item.trace_id for item in support)
        values_per_key.setdefault(key, set()).add(value)
    return [
        key
        for key, traces in traces_per_key.items()
        # Two traces cannot distinguish an identifier from a real two-way split.
        if len(traces) > 2 and len(values_per_key[key]) == len(traces)
    ]


def _top_values_per_key(
    grouped: dict[tuple[str, str], list[Evidence]],
    successes: frozenset[str] = frozenset(),
) -> dict[tuple[str, str], list[Evidence]]:
    """The few values per field worth proposing a check for.

    Frequency alone answers the wrong question. Where a corpus records seventy
    failures and thirty successes, the most frequent status is the failing one,
    and a field with several values could see the successful one fall off the
    list entirely — the draft would then never contain the check that works.
    So when outcomes are known, each key also keeps whatever its labeled
    successes recorded, whether or not the corpus repeated it most.
    """
    kept: dict[tuple[str, str], list[Evidence]] = {}
    by_key: dict[str, list[tuple[str, list[Evidence]]]] = {}
    for (key, value), support in grouped.items():
        by_key.setdefault(key, []).append((value, support))
    for key, entries in by_key.items():
        frequent = sorted(entries, key=lambda pair: (-len(pair[1]), pair[0]))
        chosen = list(frequent[:_MAX_VALUES_PER_KEY])
        if successes:
            distinguishing = sorted(
                (
                    entry
                    for entry in entries
                    if any(item.trace_id in successes for item in entry[1])
                ),
                key=lambda pair: (
                    -sum(1 for item in pair[1] if item.trace_id in successes),
                    -len(pair[1]),
                    pair[0],
                ),
            )
            for entry in distinguishing[:_MAX_VALUES_PER_KEY]:
                if entry not in chosen:
                    chosen.append(entry)
        for value, support in chosen:
            kept[(key, value)] = support
    return kept


def _compose(
    family: TaskFamily,
    task_set_id: str,
    left: VerifierSpec,
    right: VerifierSpec,
    left_outcomes: dict[str, bool | None],
    right_outcomes: dict[str, bool | None],
    verdicts: dict[str, Verdict],
) -> tuple[VerifierSpec, str] | None:
    """Propose ``left AND right`` when the labels show the pair is needed.

    Co-occurrence is not the test. Two checks that always appear together say
    only that the corpus recorded both, and conjoining on that would narrow a
    verifier for no reason anyone could defend later. The test is incremental:
    the conjunction must reject a labeled failure that ``left`` alone accepts,
    and must not lose a labeled success ``left`` alone keeps.
    """
    successes = [t for t, verdict in verdicts.items() if verdict is Verdict.SUCCESS]
    failures = [t for t, verdict in verdicts.items() if verdict is Verdict.FAILURE]
    if not successes or not failures:
        return None
    if {check.claim for check in left.checks} & {check.claim for check in right.checks}:
        # Two equality checks on one field cannot both hold. The pair would score
        # zero on every trace forever, and read as a check that rejects failures
        # perfectly.
        return None

    leaked = [t for t in failures if left_outcomes.get(t) is True]
    rescued = [t for t in leaked if right_outcomes.get(t) is False]
    if not rescued:
        return None
    kept = [t for t in successes if left_outcomes.get(t) is True]
    if not kept:
        # The check being narrowed accepts no labeled success, so it is not a
        # success check and narrowing it further cannot make it one.
        return None
    if any(right_outcomes.get(t) is not True for t in kept):
        # The second check rejects, or cannot score, a run the labels call a
        # success. Conjoining would trade a false positive for a false negative
        # or an abstention, which is not an improvement anyone asked for.
        return None

    checks = left.checks + right.checks
    verifier_id = _composite_id(family.family_id, checks)
    rationale = (
        f"{right.checks[0].claim!r} rejects {len(rescued)} labeled failure(s) that "
        f"{left.checks[0].claim!r} accepts, and rejects none of the "
        f"{len(kept)} labeled success(es) it keeps"
    )
    return (
        VerifierSpec(
            verifier_id=verifier_id,
            family_id=family.family_id,
            task_set_id=task_set_id,
            mode=VerifierMode.REPLAY,
            status=VerifierStatus.EXECUTABLE,
            inputs=tuple(dict.fromkeys(left.inputs + right.inputs)),
            checks=checks,
            unknown_when=tuple(dict.fromkeys(left.unknown_when + right.unknown_when)),
            blind_spots=tuple(dict.fromkeys(left.blind_spots + right.blind_spots)),
            gaming_hypotheses=tuple(
                dict.fromkeys(left.gaming_hypotheses + right.gaming_hypotheses)
            ),
        ),
        rationale,
    )


def _field_name(item: Evidence) -> str:
    """A state field's name, qualified by the tool that reported it."""
    return str(item.value.get("field") or item.value.get("key"))


def _invariants(
    family: TaskFamily,
    task_set_id: str,
    evidence: list[Evidence],
    successes: frozenset[str] = frozenset(),
) -> list[VerifierSpec]:
    """Propose ``final.X == initial.Y`` wherever it held on every fit trace.

    A field equalling a fixed value says a step happened. A field equalling the
    initial state it derives from says the step was *correct*, and it keeps
    saying so when the value differs run to run — which a fixed value cannot do,
    because it was only ever the one value this corpus happened to record.

    What counts as a counterexample depends on whether outcomes are known. With
    no labels, any trace where the relation fails retires it: there is nothing to
    say whether that trace was a run the relation should have held for. With
    labels, only a *labeled success* can retire it, because a relation that
    fails on a failed run is the relation doing its job — retiring it there
    discards precisely the check that separates the two, which is how the
    strongest candidate in a skewed family disappears before it is ever ranked.
    """
    # A trace whose result was read only in part cannot retire an invariant: the
    # field that would have disproved it may be one of the ones not read. Left
    # in, it looks like supporting evidence for a rule nothing tested.
    partial = {item.trace_id for item in evidence if item.claim == "truncated_outcome_fields"}

    by_trace: dict[str, dict[tuple[str, str], Any]] = {}
    for item in evidence:
        if item.trace_id in partial:
            continue
        if item.claim in ("final_state_field", "initial_state_field"):
            side = "final" if item.claim == "final_state_field" else "initial"
            by_trace.setdefault(item.trace_id, {})[(side, _field_name(item))] = item

    candidates: dict[tuple[str, str], list[Evidence]] = {}
    disproved: set[tuple[str, str]] = set()
    for trace_id, fields in by_trace.items():
        counts_against = not successes or trace_id in successes
        finals = {key: item for (side, key), item in fields.items() if side == "final"}
        initials = {key: item for (side, key), item in fields.items() if side == "initial"}
        for final_key, final_item in finals.items():
            for initial_key, initial_item in initials.items():
                pair = (final_key, initial_key)
                if final_item.value.get("value") == initial_item.value.get("value"):
                    candidates.setdefault(pair, []).extend((final_item, initial_item))
                elif counts_against:
                    # One counterexample retires the invariant. A rule that held
                    # by coincidence is worse than no rule.
                    disproved.add(pair)

    return [
        _proposal(
            family=family,
            task_set_id=task_set_id,
            claim=f"invariant:{final_key}=={initial_key}",
            operator=CheckOperator.STATE_INVARIANT,
            expected=None,
            evidence=support,
            description=(f"Require terminal {final_key!r} to equal the initial {initial_key!r}."),
            blind_spots=(
                "The two fields agreeing does not prove the change reached the "
                "authoritative system.",
            ),
            gaming=(f"Copy {initial_key!r} into {final_key!r} without performing the action.",),
            evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        )
        for (final_key, initial_key), support in sorted(
            candidates.items(), key=lambda pair: (-len(pair[1]), pair[0])
        )
        if (final_key, initial_key) not in disproved
    ]


def draft_verifiers(
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    family_id: str,
    *,
    limit: int = 3,
    labels: LabelSet | None = None,
) -> VerifierDraft:
    """Draft up to ``limit`` checks, without claiming they establish success.

    Every candidate is measured over the family's fit traces and the results are
    carried on the draft, so a reader sees how far a proposal reaches and what it
    separates rather than only that it was proposed.

    ``labels`` is what makes the ranking mean anything. Without adjudicated
    outcomes there is nothing to contrast against, so proposals stay ordered by
    evidence authority and are marked as the frequency-based hypotheses they are:
    a value can be common because failures dominate the corpus. With labels, a
    candidate that accepts failures is ranked below one that separates them, and
    a pair of checks that is jointly required can be proposed as a conjunction.
    """
    family = task_set.family_by_id().get(family_id)
    if family is None:
        raise ValueError(f"unknown family id: {family_id!r}")
    if task_set.corpus_id != analysis.corpus_id:
        raise ValueError("task set and analysis refer to different corpora")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    if labels is not None:
        if labels.task_set_id != task_set_id:
            # Family ids are derived from the descriptor, so the same id recurs
            # across task sets whose traces, splits and evidence all differ. A
            # label set from another one would rank these candidates against
            # verdicts about other runs, and _measure would then quietly drop
            # every trace it could not find — leaving a stale label set looking
            # uncalibrated rather than wrong.
            raise ValueError(
                f"labels were adjudicated against task set {labels.task_set_id!r}, "
                f"not {task_set_id!r}"
            )
        if labels.family_id != family_id:
            raise ValueError("labels belong to another family")
        stray = sorted(set(labels.verdicts()) - set(family.trace_ids))
        if stray:
            raise ValueError(f"labels name trace(s) outside family {family_id}: {', '.join(stray)}")

    evidence = _eligible(family, analysis)
    verdicts = labels.adjudicated() if labels is not None else {}
    successes = frozenset(t for t, verdict in verdicts.items() if verdict is Verdict.SUCCESS)
    proposals: list[VerifierSpec] = []

    exits = [item for item in evidence if item.claim == "command_exit_code"]
    if exits:
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                expected=0,
                evidence=exits,
                description="Require the recorded terminal command to exit successfully.",
                blind_spots=(
                    "A zero exit code does not prove the command tested the intended behavior.",
                ),
                gaming=("Run a harmless command that exits zero instead of the required check.",),
                evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
            )
        )

    states = [item for item in evidence if item.claim == "final_state_field"]
    grouped: dict[tuple[str, str], list[Evidence]] = {}
    for item in states:
        key = _field_name(item)
        value = item.value.get("value")
        grouped.setdefault((key, _stable_value(value)), []).append(item)
    for key in _identifier_keys(grouped):
        # A field with a different value on every run is an identifier or a
        # measurement, not a status. Equality against one observed value would
        # draft a verifier per trace, each of which passes only its own.
        for pair in [p for p in grouped if p[0] == key]:
            del grouped[pair]
    grouped = _top_values_per_key(grouped, successes)
    for (key, _), support in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        expected = support[0].value.get("value")
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim=f"final_state_field:{key}",
                operator=CheckOperator.EQUALS,
                expected=expected,
                evidence=support,
                description=f"Require terminal field {key!r} to equal the observed value {expected!r}.",
                blind_spots=("The field may be necessary but not sufficient for task success.",),
                gaming=(f"Set {key!r} without completing the other intended state changes.",),
                evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
            )
        )

    proposals.extend(_invariants(family, task_set_id, evidence, successes))

    errors = [item for item in evidence if item.claim == "span_error"]
    if errors:
        # Only drafted when the exporter demonstrably records errors somewhere in
        # this family; otherwise a silent absence would read as a clean run.
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim="no_span_error",
                operator=CheckOperator.NO_SPAN_ERROR,
                expected=None,
                evidence=errors,
                description="Require the episode to complete without a reported tool error.",
                blind_spots=("An episode can fail the task without any tool reporting an error.",),
                gaming=("Swallow the error and report success in the final message.",),
                evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
            )
        )

    scores = [item for item in evidence if item.claim == "recorded_score"]
    score_groups: dict[tuple[str, str], list[Evidence]] = {}
    for item in scores:
        key = str(item.value.get("key"))
        score_groups.setdefault((key, _stable_value(item.value.get("value"))), []).append(item)
    for (key, _), support in sorted(
        score_groups.items(), key=lambda pair: (-len(pair[1]), pair[0])
    ):
        expected = support[0].value.get("value")
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim=f"recorded_score:{key}",
                operator=CheckOperator.EQUALS,
                expected=expected,
                evidence=support,
                description=f"Require the recorded {key!r} to equal {expected!r}.",
                blind_spots=(
                    "What produced this score is not recorded, so its trustworthiness "
                    "is unverified until someone names the evaluator.",
                ),
                gaming=(
                    "Influence the recorded score without changing the outcome it stands for.",
                ),
                # A score is a trusted evaluator's only where the source named
                # the evaluator. Extraction settles that and records the kind on
                # the evidence; reading it back here keeps one answer in one
                # place instead of two that can drift apart.
                evidence_kind=min(
                    (item.kind for item in support),
                    key=kind_rank,
                    default=EvidenceKind.OBSERVED_TRACE,
                ),
            )
        )

    exact_output_task = any(
        marker in family.descriptor
        for marker in ("return exactly", "respond exactly", "output exactly", "print exactly")
    )
    outputs = (
        [item for item in evidence if item.claim == "final_output"] if exact_output_task else []
    )
    output_groups: dict[str, list[Evidence]] = {}
    for item in outputs:
        output_groups.setdefault(_stable_value(item.value.get("output")), []).append(item)
    for _, support in sorted(output_groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        expected = support[0].value.get("output")
        proposals.append(
            _proposal(
                family=family,
                task_set_id=task_set_id,
                claim="final_output",
                operator=CheckOperator.EXACT_OUTPUT,
                expected=expected,
                evidence=support,
                description="Require the terminal output to exactly match the recorded value.",
                blind_spots=(
                    "Exact text can reject semantically correct answers and is based on agent output, not external state.",
                ),
                gaming=("Emit the expected text without completing the underlying task.",),
                evidence_kind=EvidenceKind.AGENT_SELF_REPORT,
            )
        )

    # Measured before ranking, because how a candidate behaved is what ranks it.
    fit_ids = tuple(family.fit_trace_ids or family.trace_ids)
    evidence_by_trace: dict[str, tuple[Evidence, ...]] = {
        trace_id: tuple(item for item in analysis.evidence if item.trace_id == trace_id)
        for trace_id in fit_ids
    }
    # Only a label on a trace being measured can calibrate anything. A label set
    # covering the held-out side alone is not stale — it names traces of this
    # family — but drafting never reads those traces, so calling the result
    # contrast-derived would mark every candidate as measured against outcomes
    # while each one reports itself uncalibrated.
    applicable = {
        trace_id: verdict for trace_id, verdict in verdicts.items() if trace_id in evidence_by_trace
    }
    derivation = "contrast" if applicable else "frequency"
    measured = [
        (spec, *_measure(spec, evidence_by_trace, applicable, derivation=derivation))
        for spec in proposals
    ]
    measured.sort(key=lambda item: _rank(item[0], item[1]))

    # Conjunctions are drawn from the best standalone candidates and kept
    # alongside them, never instead of them: validation has to be able to
    # compare a composite against the checks it was built from.
    composites: list[tuple[VerifierSpec, CandidateStats, dict[str, bool | None]]] = []
    composite_sources: dict[str, set[str]] = {}
    """Which standalone candidates each conjunction was built from."""
    if applicable:
        sources = measured[:_MAX_COMPOSITE_SOURCES]
        seen_ids = {spec.verifier_id for spec, _, _ in measured}
        for index, (left, _, left_outcomes) in enumerate(sources):
            for right, _, right_outcomes in sources[index + 1 :]:
                # Both directions, because the conjunction is symmetric and the
                # justification is not: one check covering the other's gap is
                # what makes the pair defensible, and only one of the two
                # orderings may be the one that shows it.
                composed = _compose(
                    family, task_set_id, left, right, left_outcomes, right_outcomes, applicable
                ) or _compose(
                    family, task_set_id, right, left, right_outcomes, left_outcomes, applicable
                )
                if composed is None:
                    continue
                spec, rationale = composed
                if spec.verifier_id in seen_ids:
                    continue
                seen_ids.add(spec.verifier_id)
                composite_sources[spec.verifier_id] = {left.verifier_id, right.verifier_id}
                composites.append(
                    (
                        spec,
                        *_measure(
                            spec,
                            evidence_by_trace,
                            applicable,
                            derivation="contrast",
                            rationale=rationale,
                        ),
                    )
                )

    # ``limit`` bounds the independent checks, which is what it says and what a
    # caller is choosing between. A composite is not one of those: it is derived
    # from checks in this draft, and returning it without them would leave
    # validation unable to do the one comparison it exists for — whether the
    # conjunction earns its narrowness against the checks it was built from.
    kept = measured[:limit]
    retained = {spec.verifier_id for spec, _, _ in kept}
    admitted = [
        item for item in composites if composite_sources.get(item[0].verifier_id, set()) <= retained
    ][:limit]

    ranked = sorted([*kept, *admitted], key=lambda item: _rank(item[0], item[1]))
    proposals = [spec for spec, _, _ in ranked]
    candidates = [stats for _, stats, _ in ranked]

    unresolved: list[str] = []
    if proposals and not applicable:
        unresolved.append(_UNCALIBRATED_OFF_PARTITION if verdicts else _UNCALIBRATED)
    if not proposals:
        if any(item.claim == "unstructured_final_result" for item in evidence):
            unresolved.append(
                "terminal results were recorded but hold no comparable field; deciding what "
                "they mean needs a judge or a domain extractor, not another replay check"
            )
        else:
            unresolved.append(
                "no deterministic terminal exit code or structured final-state field was recorded"
            )
    elif len(proposals) == 1:
        unresolved.append("only one independent deterministic verifier pattern was supported")
    if proposals and all(spec.rests_only_on_self_report for spec in proposals[:limit]):
        unresolved.append(
            "every drafted check reads only the agent's own claim; none can be promoted"
        )

    return VerifierDraft(
        task_set_id=task_set_id,
        analysis_id=task_set.analysis_id,
        family_id=family_id,
        verifiers=tuple(proposals),
        candidates=tuple(candidates),
        unresolved=tuple(unresolved),
    )


def compute_verifier_draft_id(draft: VerifierDraft) -> str:
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    return f"verifier-draft-{digest[:16]}"


def save_verifier_draft(draft: VerifierDraft, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_verifier_draft_id(draft),
        kind="verifier_draft",
        parent_artifact_id=draft.task_set_id,
        payload=draft.model_dump_json().encode(),
        summary={"verifiers": len(draft.verifiers), "unresolved": len(draft.unresolved)},
    )


def load_verifier_draft(draft_id: str, store: DerivedStore) -> VerifierDraft:
    return VerifierDraft.model_validate_json(store.read_payload(draft_id))
