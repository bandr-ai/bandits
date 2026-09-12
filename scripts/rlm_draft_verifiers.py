#!/usr/bin/env python3
"""Have an RLM author executable family verifiers, then validate them canonically.

The RLM sees one mined family's FIT traces and FIT labels and writes checks in
the host's own operator vocabulary. The host grounds every check against the
fit evidence — a field the corpus never recorded or a value it never observed is
rejected, not executed. Fit-only validation drives a bounded repair loop; the
draft is then frozen and scored once on held-out beside the rule-drafted
baseline. Held-out traces and labels never reach the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bandits.analyze.analysis import CorpusAnalysis, load_analysis
from bandits.analyze.models import Evidence, EvidenceKind, TaskFamily, Visibility
from bandits.analyze.tasksets import load_task_set
from bandits.labels import LabelSet, Verdict, load_label_set
from bandits.store import DerivedStore
from bandits.verify.draft import draft_verifiers, save_verifier_draft
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.validate import Validation, save_validation, validate_draft

AUTHORABLE = {
    "equals": CheckOperator.EQUALS,
    "state_invariant": CheckOperator.STATE_INVARIANT,
    "no_span_error": CheckOperator.NO_SPAN_ERROR,
}


class ProposedCheck(BaseModel):
    claim: str
    operator: str
    expected: Any = None
    description: str = ""


class ProposedVerifier(BaseModel):
    name: str
    checks: list[ProposedCheck]
    rationale: str = ""
    unknown_when: list[str] = []
    blind_spots: list[str] = []
    gaming_hypotheses: list[str] = []


_INSTRUCTION = textwrap.dedent(
    """
    You are writing executable replay verifiers for ONE already-mined task family.
    Family mining is finished; never regroup traces. Use the REPL to study the FIT
    examples: the request, the initial state the tools observed, the terminal state
    the mutating tools returned, and the sealed outcome label.

    A verifier is a conjunction of checks that all must pass. Only these operators
    exist, and the host executes nothing else:
      - equals: claim "final_state_field:<field>" or "initial_state_field:<field>",
        expected = a value; passes when the recorded field equals expected.
      - state_invariant: claim "invariant:<final_field>==<initial_field>"; passes
        when the terminal field equals the initial one on the same trace.
      - no_span_error: claim "no_span_error"; passes when no tool reported an error.
    <field> must be a tool-qualified name from field_catalog exactly as written, and
    an expected value must be one the catalog observed. A trace missing any field a
    check reads scores unknown, not failed, so prefer fields most traces record.

    Goal: reject labelled failures while keeping labelled successes, on as many fit
    traces as possible. Say when the verifier cannot decide (unknown_when), what it
    cannot see (blind_spots) and how an agent could satisfy it without doing the task
    (gaming_hypotheses). Return at most three materially different verifiers.
    Held-out examples exist but are hidden; do not guess at them.
    """
).strip()


def build_predictor(*, model: str, max_iterations: int, max_llm_calls: int, max_tokens: int):
    import dspy

    from bandits.verify.judge import resolve_api_key

    lm = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
    )

    class Compose(dspy.Signature):
        family: str = dspy.InputField()
        field_catalog: str = dspy.InputField()
        fit_examples: str = dspy.InputField()
        correction: str = dspy.InputField(
            desc="empty initially; otherwise host rejections and fit disagreements to repair"
        )
        verifiers: list[ProposedVerifier] = dspy.OutputField()

    Compose.__doc__ = _INSTRUCTION
    rlm = dspy.RLM(Compose, max_iters=max_iterations, max_llm_calls=max_llm_calls, sub_lm=lm)

    def predict(**kwargs):
        with dspy.context(lm=lm):
            return rlm(**kwargs)

    return predict


# --- what the model may see ------------------------------------------------


def _field_name(item: Evidence) -> str:
    return str(item.value.get("field") or item.value.get("key"))


def fit_evidence(family: TaskFamily, analysis: CorpusAnalysis) -> dict[str, list[Evidence]]:
    """Fit-trace evidence a replay verifier may read: everything but the prompt."""
    fit = set(family.fit_trace_ids)
    out: dict[str, list[Evidence]] = {trace_id: [] for trace_id in family.fit_trace_ids}
    for item in analysis.evidence:
        if item.trace_id in fit and item.visibility is not Visibility.AT_START:
            out[item.trace_id].append(item)
    return out


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def field_catalog(evidence: dict[str, list[Evidence]], verdicts: dict[str, Verdict]) -> dict:
    """Which fields the fit traces recorded, how often, and with what values.

    Values are split by label so the model can see contrast; the catalog is also
    the set of claims and expected values the host will accept.
    """
    catalog: dict[str, dict] = {}
    for trace_id, items in evidence.items():
        label = verdicts.get(trace_id)
        for item in items:
            if item.claim not in ("final_state_field", "initial_state_field"):
                continue
            key = f"{item.claim}:{_field_name(item)}"
            entry = catalog.setdefault(key, {"traces": set(), "values": {}})
            entry["traces"].add(trace_id)
            bucket = entry["values"].setdefault(_stable(item.value.get("value")), Counter())
            bucket[label.value if label else "unlabeled"] += 1
    rendered = {}
    for key, entry in catalog.items():
        rendered[key] = {
            "traces_recording": len(entry["traces"]),
            "values": {value: dict(counts) for value, counts in entry["values"].items()},
        }
    rendered["_span_count_recorded"] = sum(
        1 for items in evidence.values() if any(e.claim == "episode_span_count" for e in items)
    )
    return rendered


def fit_examples(
    family: TaskFamily,
    analysis: CorpusAnalysis,
    evidence: dict[str, list[Evidence]],
    verdicts: dict[str, Verdict],
) -> list[dict]:
    tasks = {task.trace_id: task for task in analysis.tasks}
    rows = []
    for trace_id in family.fit_trace_ids:
        verdict = verdicts.get(trace_id)
        if verdict is None:
            continue
        items = evidence[trace_id]
        initial = {
            _field_name(e): e.value.get("value") for e in items if e.claim == "initial_state_field"
        }
        final = {
            _field_name(e): e.value.get("value") for e in items if e.claim == "final_state_field"
        }
        tools = next((e.value for e in items if e.claim == "tools_called"), None)
        output = next((e.value.get("output") for e in items if e.claim == "final_output"), None)
        rows.append(
            {
                "trace_id": trace_id,
                "request": tasks[trace_id].instruction if trace_id in tasks else None,
                "tools_called": tools,
                "initial_state": initial,
                "final_state": final,
                "final_output": (output or "")[:400],
                "errors": sum(1 for e in items if e.claim == "span_error"),
                "label": verdict.value,
            }
        )
    return rows


# --- host grounding ---------------------------------------------------------


def ground_check(proposed: ProposedCheck, evidence: dict[str, list[Evidence]]) -> CheckSpec | str:
    """Turn a proposal into a CheckSpec, or say exactly why it cannot run.

    Every claim must name a field the fit traces recorded, and every expected
    value must be one they observed. The model never gets to assert a fact the
    corpus does not hold.
    """
    operator = AUTHORABLE.get(proposed.operator)
    if operator is None:
        return f"operator {proposed.operator!r} is not executable; use {sorted(AUTHORABLE)}"
    flat = [item for items in evidence.values() for item in items]

    def support(claim: str, field: str) -> list[Evidence]:
        return [e for e in flat if e.claim == claim and _field_name(e) == field]

    if operator is CheckOperator.NO_SPAN_ERROR:
        anchors = [e for e in flat if e.claim == "episode_span_count"]
        if not anchors:
            return "no_span_error needs episode_span_count, which no fit trace recorded"
        return CheckSpec(
            check_id="check-rlm-no-span-error",
            claim="no_span_error",
            operator=operator,
            expected=None,
            supporting_evidence_ids=tuple(sorted(e.evidence_id for e in anchors)),
            description=proposed.description or "Require no tool to report an error.",
            evidence_kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        )

    if operator is CheckOperator.EQUALS:
        prefix, _, field = proposed.claim.partition(":")
        if prefix not in ("final_state_field", "initial_state_field") or not field:
            return f"equals claim {proposed.claim!r} must be final_state_field:<field> or initial_state_field:<field>"
        found = support(prefix, field)
        if not found:
            return f"no fit trace recorded {prefix}:{field!r}"
        observed = {_stable(e.value.get("value")) for e in found}
        if _stable(proposed.expected) not in observed:
            return (
                f"expected {proposed.expected!r} was never observed for {proposed.claim!r}; "
                f"observed values: {sorted(observed)[:8]}"
            )
        evidence_kind = EvidenceKind.STRUCTURED_EXTERNAL_RESULT
    elif operator is CheckOperator.STATE_INVARIANT:
        prefix, _, pair = proposed.claim.partition(":")
        final_field, sep, initial_field = pair.partition("==")
        if prefix != "invariant" or not sep or not final_field or not initial_field:
            return f"state_invariant claim {proposed.claim!r} must be invariant:<final_field>==<initial_field>"
        finals = support("final_state_field", final_field)
        initials = support("initial_state_field", initial_field)
        if not finals:
            return f"no fit trace recorded final_state_field:{final_field!r}"
        if not initials:
            return f"no fit trace recorded initial_state_field:{initial_field!r}"
        found = finals + initials
        proposed = proposed.model_copy(update={"expected": None})
        evidence_kind = EvidenceKind.STRUCTURED_EXTERNAL_RESULT
    else:  # pragma: no cover - AUTHORABLE is closed
        return f"unsupported operator {operator.value!r}"

    digest = hashlib.sha256(f"{proposed.claim}\0{_stable(proposed.expected)}".encode()).hexdigest()[
        :12
    ]
    return CheckSpec(
        check_id=f"check-rlm-{digest}",
        claim=proposed.claim,
        operator=operator,
        expected=proposed.expected,
        supporting_evidence_ids=tuple(sorted({e.evidence_id for e in found})),
        description=proposed.description or f"Require {proposed.claim} via {proposed.operator}.",
        evidence_kind=evidence_kind,
    )


def build_spec(
    proposal: ProposedVerifier,
    evidence: dict[str, list[Evidence]],
    family_id: str,
    task_set_id: str,
) -> tuple[VerifierSpec | None, list[str]]:
    checks: list[CheckSpec] = []
    rejections: list[str] = []
    seen: set[str] = set()
    for proposed in proposal.checks:
        grounded = ground_check(proposed, evidence)
        if isinstance(grounded, str):
            rejections.append(f"{proposal.name}: {grounded}")
        elif grounded.check_id not in seen:
            seen.add(grounded.check_id)
            checks.append(grounded)
    if not checks:
        rejections.append(f"{proposal.name}: no executable check survived")
        return None, rejections
    digest = hashlib.sha256(
        (family_id + "\0" + "\0".join(sorted(c.check_id for c in checks))).encode()
    ).hexdigest()[:16]
    spec = VerifierSpec(
        verifier_id=f"verifier-rlm-{digest}",
        family_id=family_id,
        task_set_id=task_set_id,
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=tuple(sorted({f"terminal_evidence:{c.claim}" for c in checks})),
        checks=tuple(checks),
        unknown_when=tuple(proposal.unknown_when) or ("any field a check reads is absent",),
        blind_spots=tuple(proposal.blind_spots) or ("Blind spots were not stated by the author.",),
        gaming_hypotheses=tuple(proposal.gaming_hypotheses)
        or ("Gaming hypotheses were not stated by the author.",),
        provenance="model",
    )
    return spec, rejections


# --- selection on fit only --------------------------------------------------


def fit_summary(validation: Validation) -> dict[str, dict]:
    out = {}
    for item in validation.agreements:
        if item.split != "fit":
            continue
        out[item.verifier_id] = {
            "labeled": item.labeled,
            "scored": item.scored,
            "coverage": item.coverage,
            "agreement": item.agreement,
            "false_positives": item.false_positives,
            "false_negatives": item.false_negatives,
            "failure_catch_rate": item.failure_catch_rate,
            "counterexamples": [c.model_dump(mode="json") for c in item.counterexamples],
        }
    return out


def rank_key(summary: dict) -> tuple:
    """Fewest failures admitted, then most failures caught, then most traces scored."""
    return (
        summary["false_positives"] if summary["false_positives"] is not None else 10**6,
        -(summary["failure_catch_rate"] or 0.0),
        -(summary["coverage"] or 0.0),
        -(summary["agreement"] or 0.0),
    )


def correction_text(
    rejections: list[str], summaries: dict[str, dict], specs: dict[str, VerifierSpec]
) -> str:
    lines = []
    if rejections:
        lines.append("HOST REJECTED THESE CHECKS (fix or drop them):")
        lines.extend(f"  - {item}" for item in rejections)
    for verifier_id, summary in summaries.items():
        spec = specs[verifier_id]
        lines.append(
            f"FIT RESULT for {verifier_id} ({[c.claim for c in spec.checks]}): "
            f"scored {summary['scored']}/{summary['labeled']}, "
            f"false_positives={summary['false_positives']}, "
            f"false_negatives={summary['false_negatives']}, "
            f"failure_catch_rate={summary['failure_catch_rate']}"
        )
        for item in summary["counterexamples"]:
            lines.append(
                f"    {item['kind']}: trace {item['trace_id']} scored {item['verifier_score']} "
                f"but is labelled {item['human_verdict']}"
            )
    lines.append(
        "Unscored traces are ones where a check's field was absent. A false positive "
        "admits a failed run into training data; reduce those first, then raise coverage."
    )
    return "\n".join(lines)


def _parse(reply: Any) -> list[ProposedVerifier]:
    out = []
    for item in getattr(reply, "verifiers", []) or []:
        try:
            out.append(
                item
                if isinstance(item, ProposedVerifier)
                else ProposedVerifier.model_validate(item)
            )
        except Exception as exc:  # noqa: BLE001 - a malformed proposal is data, not a crash
            out.append(ProposedVerifier(name="malformed", checks=[], rationale=str(exc)[:200]))
    return out


def author_family(
    *,
    predict,
    task_set,
    task_set_id: str,
    analysis: CorpusAnalysis,
    family: TaskFamily,
    labels: LabelSet,
    label_id: str,
    store: DerivedStore,
    rounds: int,
    log,
) -> dict:
    verdicts = labels.adjudicated()
    evidence = fit_evidence(family, analysis)
    catalog = field_catalog(evidence, verdicts)
    examples = fit_examples(family, analysis, evidence, verdicts)
    family_text = json.dumps(
        {
            "family_id": family.family_id,
            "descriptor": family.descriptor,
            "fit_traces": len(family.fit_trace_ids),
        }
    )
    correction = ""
    history = []
    best: tuple[tuple, VerifierSpec, dict] | None = None

    for round_index in range(rounds):
        reply = predict(
            family=family_text,
            field_catalog=json.dumps(catalog, sort_keys=True)[:120_000],
            fit_examples=json.dumps(examples, default=str)[:200_000],
            correction=correction,
        )
        proposals = _parse(reply)
        specs: dict[str, VerifierSpec] = {}
        rejections: list[str] = []
        for proposal in proposals:
            spec, rejected = build_spec(proposal, evidence, family.family_id, task_set_id)
            rejections.extend(rejected)
            if spec is not None:
                specs.setdefault(spec.verifier_id, spec)
        entry = {
            "round": round_index,
            "proposals": [p.model_dump(mode="json") for p in proposals],
            "rejections": rejections,
            "fit": {},
        }
        if specs:
            draft = VerifierDraft(
                task_set_id=task_set_id,
                analysis_id=task_set.analysis_id,
                family_id=family.family_id,
                verifiers=tuple(specs.values()),
            )
            envelope = save_verifier_draft(draft, store)
            # Fit only: the model repairs against these, so held-out stays sealed.
            validation = validate_draft(
                draft,
                envelope.artifact_id,
                task_set,
                analysis,
                labels,
                label_id,
                include_held_out=False,
            )
            summaries = fit_summary(validation)
            entry["draft_id"] = envelope.artifact_id
            entry["fit"] = summaries
            for verifier_id, summary in summaries.items():
                key = rank_key(summary)
                if summary["scored"] and (best is None or key < best[0]):
                    best = (key, specs[verifier_id], summary)
            correction = correction_text(rejections, summaries, specs)
        else:
            correction = correction_text(rejections, {}, {})
        history.append(entry)
        log(f"  round {round_index}: {len(specs)} executable, {len(rejections)} rejected")
        if (
            best is not None
            and best[2]["false_positives"] == 0
            and (best[2]["coverage"] or 0) >= 0.9
        ):
            break

    result: dict = {
        "family_id": family.family_id,
        "descriptor": family.descriptor,
        "rounds": history,
    }
    if best is None:
        result["status"] = "no_executable_verifier"
        return result

    # Freeze, then open held-out exactly once.
    frozen = VerifierDraft(
        task_set_id=task_set_id,
        analysis_id=task_set.analysis_id,
        family_id=family.family_id,
        verifiers=(best[1],),
    )
    envelope = save_verifier_draft(frozen, store)
    validation = validate_draft(frozen, envelope.artifact_id, task_set, analysis, labels, label_id)
    saved = save_validation(validation, store)

    baseline = draft_verifiers(
        task_set, task_set_id, analysis, family.family_id, limit=3, labels=labels
    )
    baseline_envelope = save_verifier_draft(baseline, store)
    baseline_validation = validate_draft(
        baseline, baseline_envelope.artifact_id, task_set, analysis, labels, label_id
    )
    baseline_saved = save_validation(baseline_validation, store)

    result.update(
        status="validated",
        selected=best[1].model_dump(mode="json"),
        draft_id=envelope.artifact_id,
        validation_id=saved.artifact_id,
        agreements=[a.model_dump(mode="json") for a in validation.agreements],
        gameability=[g.model_dump(mode="json") for g in validation.gameability_assessments],
        baseline={
            "draft_id": baseline_envelope.artifact_id,
            "validation_id": baseline_saved.artifact_id,
            "verifiers": [[c.claim for c in s.checks] for s in baseline.verifiers],
            "agreements": [a.model_dump(mode="json") for a in baseline_validation.agreements],
        },
    )
    return result


def eligible(family: TaskFamily, verdicts: dict[str, Verdict]) -> bool:
    fit = {verdicts[t] for t in family.fit_trace_ids if t in verdicts}
    held = {verdicts[t] for t in family.held_out_trace_ids if t in verdicts}
    return len(fit) == 2 and len(held) == 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_set_id")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--label-sets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-families", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--model", default="accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    )
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=24_000)
    args = parser.parse_args()

    store = DerivedStore(args.project / ".bandits")
    task_set = load_task_set(args.task_set_id, store)
    analysis = load_analysis(task_set.analysis_id, store)
    label_ids = json.loads(args.label_sets.read_text())
    predict = build_predictor(
        model=args.model,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_tokens=args.max_tokens,
    )
    report = {"task_set_id": args.task_set_id, "model": args.model, "families": [], "skipped": []}
    for family in sorted(task_set.families, key=lambda item: -len(item.trace_ids)):
        label_id = label_ids.get(family.family_id)
        if not label_id:
            report["skipped"].append({"family_id": family.family_id, "reason": "no label set"})
            continue
        labels = load_label_set(label_id, store)
        if not eligible(family, labels.adjudicated()):
            report["skipped"].append(
                {"family_id": family.family_id, "reason": "needs both verdicts on fit and held-out"}
            )
            continue
        print(
            f"family {family.family_id}: {len(family.fit_trace_ids)} fit / {len(family.held_out_trace_ids)} held-out"
        )
        report["families"].append(
            author_family(
                predict=predict,
                task_set=task_set,
                task_set_id=args.task_set_id,
                analysis=analysis,
                family=family,
                labels=labels,
                label_id=label_id,
                store=store,
                rounds=args.rounds,
                log=print,
            )
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        if len(report["families"]) >= args.max_families:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
