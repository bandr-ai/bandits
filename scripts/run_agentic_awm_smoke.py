#!/usr/bin/env python3
"""Standalone smoke runner for the agentic AWM. Real model, explicit transitions.

Does not integrate with rollouts/campaigns (Phase 10 in the plan, later, only
after this passes). One question only: can an AWM that actively calls
read-only grounding tools (inspect_tool_contract, read_world_state,
search_transitions, inspect_ledger) produce a valid, well-grounded transition
proposal for a handful of hand-picked, meaningfully different cases --

    A. known-state mutation (cancel_reservation on an already-read reservation)
    B. unsupported lookup (get_user_details for an entity genuinely absent
       from state and from retrieval)
    C. target-injected control (get_user_details for an entity whose held-out
       answer is copied directly into state) -- a plumbing/control check, NOT
       an AWM fidelity claim: it tests whether the AWM faithfully reports
       facts it was directly handed, not whether it can ground a genuine
       prediction from evidence it had to gather itself.

-- without gaining any authority validate_transition doesn't already check.
Every internal tool call is captured structurally (AWMToolCall), not parsed
from DSPy's trajectory text, and the final proposal goes through the exact
same validate_transition() gate every other predictor's output does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_awm_fidelity import excluded_trace_ids_for, load_family, render_history

from bandits.diagnose.agentic import (
    AGENTIC_TOOL_WORLD_INSTRUCTION,
    AWMExecutionTrace,
    AWMRuntimeContext,
    AWMToolCall,
    build_agentic_tool_world_predictor,
    step_agentic_tool_world,
)
from bandits.diagnose.fidelity import (
    _correlate_observations,
    _looks_like_error,
    _recorded_observation,
    compare_observation,
)
from bandits.diagnose.models import (
    GroundingTransition,
    ScenarioState,
    StateField,
    WorldOrigin,
)
from bandits.diagnose.retrieve import build_index
from bandits.diagnose.world import (
    ProposedTransition,
    ValidationOutcome,
    WorldModelError,
    prompt_digest,
    validate_transition,
)

DEFAULT_TAU_ROOT = Path("work/tau/run/proj/.bandits")
DEFAULT_CORPUS_ID = "corpus-ee3b33086ef177d7"
DEFAULT_TASK_SET_ID = "taskset-37654fb6a6bd578f"
DEFAULT_FAMILY_ID = "family-451ae91f975c"
DEFAULT_MARKERS = ("###TRANSFER###",)


def outcome_taxonomy(
    transition: GroundingTransition,
    proposal: ProposedTransition,
    validation: ValidationOutcome | None,
    trace: AWMExecutionTrace,
) -> dict[str, Any]:
    """The full outcome classification for one transition -- not field
    accuracy alone. Called unconditionally (not only when accepted): the
    previous version returned None for output_invalid/abstained/rejected
    rows, so a reader of the report saw field accuracy for the one case that
    happened to be scorable and nothing structured for the other three,
    which is exactly what the review flagged as an incomplete taxonomy.

    Deliberately not score_transition_fidelity() for the accepted case:
    that function's own step_tool_world call gates on
    `support_level(examples)`, and every smoke call here passes no fixed
    `examples` at all -- retrieval happens inside the AWM's own
    search_transitions tool, not as an argument that function receives.
    Calling it with `examples=()` doesn't skip that gate, it *hits* it:
    support_level(()) is "none", which short-circuits step_tool_world into a
    synthetic abstention before it ever looks at the real proposal.

    Field comparison (when reachable) uses the validator's own
    committed_observations, never the raw proposal directly -- the same D76
    principle fidelity.py's real scorer follows.
    """
    grounding_summary = {
        "grounding_call_count": len(trace.grounding_calls),
        "grounding_tools_used": sorted({c.tool for c in trace.grounding_calls}),
        "entities_grounded": sorted(f"{k}:{i}" for k, i in trace.entities_grounded()),
        "exhausted_budget": trace.exhausted_budget,
        "budget_rejection_attempted": trace.budget_rejection_attempted,
        "rejected_unsupported_claim_paths": list(trace.rejected_unsupported_claim_paths),
    }

    if proposal.output_invalid:
        # Persist the unprojected prediction and per-stage LM records here
        # specifically: this is the branch where there is no parsed proposal
        # to inspect, so without them a failure leaves nothing but an error
        # string, and "the model looped" cannot be told from "the response
        # was truncated just short of valid."
        return {
            "outcome": "output_invalid",
            "output_invalid_errors": list(proposal.output_invalid_errors),
            "raw_prediction": trace.raw_prediction,
            "lm_history": [dict(entry) for entry in trace.lm_history],
            **grounding_summary,
        }

    if proposal.abstain:
        # Two distinct reasons an agentic proposal can abstain, and they must
        # not be scored by the same rule:
        #   - the AWM itself declined (epistemic: "I don't have this entity")
        #     -- correct iff the candidate's own target entities were
        #     genuinely never grounded, mirroring fidelity.py's I39-fixed
        #     rule;
        #   - step_agentic_tool_world's own claim-level attribution rejected
        #     an insufficiently-grounded mutation/event claim (agentic.py's
        #     assess_proposal_claims) -- this is always a correct outcome by
        #     construction: it only fires when a specific fabricated claim
        #     was caught, which is exactly the behavior being verified.
        #     Scoring it under the entity-level rule above would understate
        #     it, since the entity itself may have been perfectly grounded
        #     (Experiment A's cancellation shape) while one specific
        #     additional field was invented.
        claim_level_rejection = bool(trace.rejected_unsupported_claim_paths)
        if claim_level_rejection:
            abstain_correct = True
        else:
            target_entities = set()
            for call in transition.action_calls:
                for arg_name, kind in (("user_id", "user"), ("reservation_id", "reservation")):
                    value = call.arguments.get(arg_name)
                    if isinstance(value, str) and value:
                        target_entities.add((kind, value))
            grounded = trace.entities_grounded()
            abstain_correct = bool(target_entities) and not (target_entities & grounded)
        return {
            "outcome": "abstained",
            "abstain_reason": proposal.abstain_reason,
            "abstain_correct": abstain_correct,
            "claim_level_rejection": claim_level_rejection,
            **grounding_summary,
        }

    if validation is None or not validation.accepted:
        return {
            "outcome": "validator_rejected",
            "validator_rejections": list(validation.rejections) if validation else [],
            **grounding_summary,
        }

    recorded = _recorded_observation(transition)
    if recorded is None:
        return {"outcome": "accepted", "scorable": False, "reason": "no recorded observation", **grounding_summary}

    pairs, unmatched = _correlate_observations(transition, validation.committed_observations)
    if unmatched or not pairs:
        return {
            "outcome": "accepted",
            "scorable": False,
            "unmatched_call_observations": list(unmatched),
            **grounding_summary,
        }

    fields = []
    for predicted_payload, recorded_obs in pairs.values():
        fields.extend(compare_observation(predicted_payload, recorded_obs.content))
    correct = sum(1 for f in fields if f.correct)
    status_correct = all(
        _looks_like_error(predicted_payload) == bool(recorded_obs.error)
        for predicted_payload, recorded_obs in pairs.values()
    )
    return {
        "outcome": "accepted",
        "scorable": True,
        "status_correct": status_correct,
        "field_count": len(fields),
        "fields_correct": correct,
        "field_accuracy": (correct / len(fields)) if fields else None,
        "incorrect_or_missing": [
            {"path": f.path, "predicted": f.predicted, "recorded": f.recorded}
            for f in fields
            if not f.correct
        ],
        **grounding_summary,
    }


def build_context(
    transition: GroundingTransition,
    *,
    all_transitions: tuple[GroundingTransition, ...],
    fit_index: tuple[GroundingTransition, ...],
    tool_schemas: tuple[dict[str, Any], ...],
    extra_state_fields: tuple[StateField, ...] = (),
) -> AWMRuntimeContext:
    """One transition, wrapped in exactly what the agentic AWM is allowed to see.

    extra_state_fields is Experiment C's hook: a manually constructed fact
    packet, injected the same way any other RECORDED state would arrive --
    the AWM has no separate "fact packet" tool, it just sees more state.
    """
    history_text = render_history(transition)
    excluded = excluded_trace_ids_for(transition, all_transitions)
    state = transition.state_before
    if extra_state_fields:
        state = ScenarioState(fields=state.fields + extra_state_fields)
    return AWMRuntimeContext(
        candidate_calls=transition.action_calls,
        action_content=transition.action_content,
        current_state=state,
        history_text=history_text,
        history_before=transition.history_before,
        offered_tool_schemas=tool_schemas,
        fit_index=fit_index,
        family_id=transition.family_id,
        task_context=transition.task_context,
        excluded_trace_ids=excluded,
    )


def run_one(
    transition: GroundingTransition,
    *,
    all_transitions: tuple[GroundingTransition, ...],
    fit_index: tuple[GroundingTransition, ...],
    tool_schemas: tuple[dict[str, Any], ...],
    predict,
    label: str,
    extra_state_fields: tuple[StateField, ...] = (),
) -> dict[str, Any]:
    context = build_context(
        transition,
        all_transitions=all_transitions,
        fit_index=fit_index,
        tool_schemas=tool_schemas,
        extra_state_fields=extra_state_fields,
    )
    try:
        proposal, trace = step_agentic_tool_world(predict, context=context)
    except WorldModelError as exc:
        return {"label": label, "transition_id": transition.transition_id, "provider_error": str(exc)}

    validation = None
    if not proposal.abstain and not proposal.output_invalid:
        validation = validate_transition(
            proposal,
            calls=transition.action_calls,
            state=context.current_state,
            step_index=transition.turn_index,
            allowed_evidence_ids=trace.all_evidence_ids,
            tool_schemas=tool_schemas,
        )

    # Applies to all four possible outcomes, not only "accepted": the
    # previous version returned None for output_invalid/abstained/rejected
    # rows entirely, invisible in the report except via the raw proposal
    # dict's own flags.
    taxonomy = outcome_taxonomy(transition, proposal, validation, trace)

    return {
        "label": label,
        "transition_id": transition.transition_id,
        "tool": transition.action_tool,
        "grounding_calls": [json.loads(c.model_dump_json()) for c in trace.grounding_calls],
        "exhausted_budget": trace.exhausted_budget,
        "budget_rejection_attempted": trace.budget_rejection_attempted,
        "all_evidence_ids": list(trace.all_evidence_ids),
        "all_exact_entity_evidence_ids": list(trace.all_exact_entity_evidence_ids),
        "all_state_paths_read": list(trace.all_state_paths_read),
        "proposal": json.loads(proposal.model_dump_json()),
        "validator_result": json.loads(validation.model_dump_json()) if validation else None,
        "expected_observation": next(
            (o.content for o in transition.observations if o.role == "tool"), None
        ),
        "outcome": taxonomy,
    }


def _global_lm_history(limit: int = 12) -> list[dict[str, Any]]:
    """Per-stage LM records from dspy's global history, for the crash path.

    A failure inside predict() leaves no AWMExecutionTrace, so the records
    cannot come from the trace the way they do on the output_invalid path.
    Best-effort: diagnostics must never mask the original exception.
    """
    try:
        from dspy.clients.base_lm import GLOBAL_HISTORY

        from bandits.diagnose.agentic import _summarize_lm_entry

        return [_summarize_lm_entry(entry, "unknown") for entry in list(GLOBAL_HISTORY)[-limit:]]
    except Exception:  # noqa: BLE001 -- never mask the real failure
        return []


def run(
    *,
    tau_root: Path,
    corpus_id: str,
    task_set_id: str,
    family_id: str,
    markers: tuple[str, ...],
    model: str,
    known_state_transition_id: str,
    unsupported_lookup_transition_id: str,
    fact_grounded_transition_id: str,
    output_dir: Path,
    max_tokens: int = 6000,
    select_max_tokens: int = 800,
    extract_max_tokens: int = 3000,
    cases: tuple[str, ...] = (),
    resume: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    family, transitions, tool_schemas_by_trace = load_family(
        tau_root, corpus_id=corpus_id, task_set_id=task_set_id, family_id=family_id, markers=markers
    )
    fit_index = build_index(transitions, fit_trace_ids=family["fit_trace_ids"])
    by_id = {t.transition_id: t for t in transitions}

    manifest = {
        "model": model,
        "prompt_digest": prompt_digest(AGENTIC_TOOL_WORLD_INSTRUCTION, model),
        "corpus_id": corpus_id,
        "task_set_id": task_set_id,
        "family_id": family_id,
        "markers": list(markers),
        "transitions": {
            "known_state": known_state_transition_id,
            "unsupported_lookup": unsupported_lookup_transition_id,
            "fact_grounded": fact_grounded_transition_id,
        },
    }
    (output_dir / "smoke.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # Live progress. Each case is a ReAct loop making up to max_iters model
    # calls and several grounding calls, so a run that printed only on
    # completion sat silent for minutes with no way to tell a slow call from
    # a hung one, or to see which tool the AWM actually reached for.
    def _report_grounding_call(call: AWMToolCall) -> None:
        detail = ""
        if call.entity is not None:
            detail = f" {call.entity[0]}:{call.entity[1]}"
        elif call.arguments:
            detail = " " + ", ".join(f"{k}={v!r}" for k, v in list(call.arguments.items())[:2])
        summary = f" -> {call.result_summary}" if call.result_summary else ""
        print(f"    [call {call.index}] {call.tool}{detail}{summary}", flush=True)

    predict = build_agentic_tool_world_predictor(
        model=model,
        max_tokens=max_tokens,
        select_max_tokens=select_max_tokens,
        extract_max_tokens=extract_max_tokens,
        on_grounding_call=_report_grounding_call,
    )

    def _control_extra_fields(control: GroundingTransition) -> tuple[StateField, ...]:
        recorded = next((o for o in control.observations if o.role == "tool"), None)
        target_call = control.action_calls[0]
        entity_id = target_call.arguments.get("user_id") or target_call.arguments.get("reservation_id")
        if not (recorded and entity_id):
            return ()
        from bandits.diagnose.grounding import _flatten_paths

        prefix = f"{target_call.tool}.{entity_id}."
        # revealed_by_span_id must cite the span that actually produced this
        # value -- the recorded OBSERVATION's own span, not the ACTION's (the
        # call that asked for it). Citing the action span attributes the data
        # to the wrong evidence source.
        return tuple(
            StateField(
                path=f"{prefix}{path}",
                value=value,
                origin=WorldOrigin.RECORDED,
                revealed_by_span_id=recorded.span_id or control.action_span_id,
            )
            for path, value in _flatten_paths(recorded.content).items()
        )

    # Experiment C is a target-injected control, not AWM fidelity: the
    # held-out answer itself is copied straight into state, so it tests
    # plumbing -- can the AWM report facts it was directly handed without
    # inventing or dropping any -- not whether it can ground a genuine
    # prediction from evidence it had to gather itself.
    planned = (
        ("A_known_state_mutation", by_id[known_state_transition_id], ()),
        ("B_unsupported_lookup", by_id[unsupported_lookup_transition_id], ()),
        (
            "C_target_injected_control",
            by_id[fact_grounded_transition_id],
            _control_extra_fields(by_id[fact_grounded_transition_id]),
        ),
    )

    if cases:
        wanted = {c.upper() for c in cases}
        planned = tuple(row for row in planned if row[0][0].upper() in wanted)
        if not planned:
            raise SystemExit(f"no cases matched {sorted(wanted)} (expected some of A, B, C)")

    checkpoint_path = output_dir / "smoke.raw.jsonl"
    # Resume rather than truncate: a case costs a real (paid) call, so a
    # rerun aimed at one case must not discard the others' completed results.
    # Reruns of a case already present replace that case's row, keeping the
    # newest result per label.
    previous: dict[str, dict[str, Any]] = {}
    if resume and checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("label"):
                previous[row["label"]] = row
    replaced = {label for label, _t, _e in planned}
    results: list[dict[str, Any]] = [
        row for label, row in previous.items() if label not in replaced
    ]
    if results:
        print(f"[resume] keeping {len(results)} prior result(s): "
              f"{', '.join(sorted(r['label'] for r in results))}", flush=True)

    # Checkpoint each real (paid) call as it completes, not after all three:
    # an exception on C must not lose A and B's already-paid-for results.
    with checkpoint_path.open("w") as checkpoint_file:
        for kept in results:
            checkpoint_file.write(json.dumps(kept, default=str) + "\n")
        checkpoint_file.flush()
        for label, transition, extra_fields in planned:
            print(f"[start] {label} {transition.transition_id}", flush=True)
            try:
                result = run_one(
                    transition,
                    all_transitions=transitions,
                    fit_index=fit_index,
                    tool_schemas=tool_schemas_by_trace.get(transition.trace_id, ()),
                    predict=predict,
                    label=label,
                    extra_state_fields=extra_fields,
                )
            except Exception as exc:  # noqa: BLE001 -- must not lose prior checkpoints
                # A crash inside predict() (an adapter parse failure, a
                # protocol failure) raises before any trace exists, so the
                # per-stage records have to come from dspy's own global
                # history here. This is the branch A hit: without it the only
                # artifact was an error string, and the LM records are what
                # separate a repetition loop from a near-miss truncation.
                result = {
                    "label": label,
                    "transition_id": transition.transition_id,
                    "unexpected_error": f"{type(exc).__name__}: {exc}",
                    "lm_history": _global_lm_history(),
                }
                results.append(result)
                checkpoint_file.write(json.dumps(result, default=str) + "\n")
                checkpoint_file.flush()
                print(f"[unexpected-error] {label} {transition.transition_id}: {exc}", flush=True)
                continue

            results.append(result)
            checkpoint_file.write(json.dumps(result, default=str) + "\n")
            checkpoint_file.flush()
            print(f"[done] {label} {transition.transition_id}", flush=True)

    report = {
        "model": model,
        "results": [
            {
                "label": r["label"],
                "transition_id": r["transition_id"],
                # The full taxonomy (I39-fixed abstain_correct, status_correct,
                # field_accuracy, or the reason for a structural failure) --
                # not field accuracy alone, and never None for the three
                # outcomes that used to be silently unscored.
                "outcome": r.get("outcome"),
                "entities_grounded": (r.get("outcome") or {}).get("entities_grounded"),
                "evidence_ids_cited": r.get("proposal", {}).get("evidence_ids"),
                "evidence_ids_actually_searched": r.get("all_evidence_ids"),
                "exact_entity_evidence_ids": r.get("all_exact_entity_evidence_ids"),
                "state_paths_read": r.get("all_state_paths_read"),
                "budget_rejection_attempted": r.get("budget_rejection_attempted"),
            }
            for r in results
            if "provider_error" not in r and "unexpected_error" not in r
        ],
        "provider_errors": [r for r in results if "provider_error" in r],
        "unexpected_errors": [r for r in results if "unexpected_error" in r],
    }
    (output_dir / "smoke.report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {output_dir}/smoke.manifest.json, smoke.raw.jsonl, smoke.report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, default=DEFAULT_TAU_ROOT)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--task-set-id", default=DEFAULT_TASK_SET_ID)
    parser.add_argument("--family-id", default=DEFAULT_FAMILY_ID)
    parser.add_argument("--marker", action="append", dest="markers", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--known-state-transition-id", required=True)
    parser.add_argument("--unsupported-lookup-transition-id", required=True)
    parser.add_argument("--fact-grounded-transition-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/awm-agentic"))
    # Raise when responses truncate: a truncated response parses into an
    # output_invalid row, which is a parser failure recorded as if it were
    # an epistemic result.
    parser.add_argument("--max-tokens", type=int, default=6000)
    # Budgeted per ReAct stage. Tool selection emits three short fields;
    # a large ceiling there lets a repetition loop run longer rather than
    # producing a valid selection.
    parser.add_argument("--select-max-tokens", type=int, default=800)
    parser.add_argument("--extract-max-tokens", type=int, default=3000)
    parser.add_argument(
        "--case", action="append", dest="cases", default=None,
        help="run only these cases (A, B, C); repeatable. Others are kept from the checkpoint.",
    )
    parser.add_argument("--no-resume", action="store_true", help="discard prior checkpoint rows")
    args = parser.parse_args()

    run(
        tau_root=args.tau_root,
        corpus_id=args.corpus_id,
        task_set_id=args.task_set_id,
        family_id=args.family_id,
        markers=tuple(args.markers) if args.markers else DEFAULT_MARKERS,
        model=args.model,
        known_state_transition_id=args.known_state_transition_id,
        unsupported_lookup_transition_id=args.unsupported_lookup_transition_id,
        fact_grounded_transition_id=args.fact_grounded_transition_id,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        select_max_tokens=args.select_max_tokens,
        extract_max_tokens=args.extract_max_tokens,
        cases=tuple(args.cases) if args.cases else (),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
