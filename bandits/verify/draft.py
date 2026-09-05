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
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
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
) -> dict[tuple[str, str], list[Evidence]]:
    kept: dict[tuple[str, str], list[Evidence]] = {}
    by_key: dict[str, list[tuple[str, list[Evidence]]]] = {}
    for (key, value), support in grouped.items():
        by_key.setdefault(key, []).append((value, support))
    for key, entries in by_key.items():
        ranked = sorted(entries, key=lambda pair: (-len(pair[1]), pair[0]))
        for value, support in ranked[:_MAX_VALUES_PER_KEY]:
            kept[(key, value)] = support
    return kept


def _field_name(item: Evidence) -> str:
    """A state field's name, qualified by the tool that reported it."""
    return str(item.value.get("field") or item.value.get("key"))


def _invariants(
    family: TaskFamily, task_set_id: str, evidence: list[Evidence]
) -> list[VerifierSpec]:
    """Propose ``final.X == initial.Y`` wherever it held on every fit trace.

    A field equalling a fixed value says a step happened. A field equalling the
    initial state it derives from says the step was *correct*, and it keeps
    saying so when the value differs run to run — which a fixed value cannot do,
    because it was only ever the one value this corpus happened to record.
    """
    by_trace: dict[str, dict[tuple[str, str], Any]] = {}
    for item in evidence:
        if item.claim in ("final_state_field", "initial_state_field"):
            side = "final" if item.claim == "final_state_field" else "initial"
            by_trace.setdefault(item.trace_id, {})[(side, _field_name(item))] = item

    candidates: dict[tuple[str, str], list[Evidence]] = {}
    disproved: set[tuple[str, str]] = set()
    for fields in by_trace.values():
        finals = {key: item for (side, key), item in fields.items() if side == "final"}
        initials = {key: item for (side, key), item in fields.items() if side == "initial"}
        for final_key, final_item in finals.items():
            for initial_key, initial_item in initials.items():
                pair = (final_key, initial_key)
                if final_item.value.get("value") == initial_item.value.get("value"):
                    candidates.setdefault(pair, []).extend((final_item, initial_item))
                else:
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
) -> VerifierDraft:
    """Draft up to ``limit`` independent checks, without claiming they establish success."""
    family = task_set.family_by_id().get(family_id)
    if family is None:
        raise ValueError(f"unknown family id: {family_id!r}")
    if task_set.corpus_id != analysis.corpus_id:
        raise ValueError("task set and analysis refer to different corpora")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    evidence = _eligible(family, analysis)
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
    grouped = _top_values_per_key(grouped)
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

    proposals.extend(_invariants(family, task_set_id, evidence))

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
                evidence_kind=EvidenceKind.TRUSTED_EVALUATOR,
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

    proposals.sort(key=lambda spec: -kind_rank(spec.weakest_evidence_kind))

    unresolved: list[str] = []
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
        verifiers=tuple(proposals[:limit]),
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
