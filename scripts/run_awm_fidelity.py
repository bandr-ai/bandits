#!/usr/bin/env python3
"""Run the real tool-world AWM against held-out tau2 transitions and score fidelity.

The engine already exists (compile, retrieve, world, fidelity, store). This
script is only the wiring: load a real tau2 family, build the fit-only
retrieval index, predict each held-out tool transition with a real
Fireworks-backed AWM, score it against what was actually recorded, checkpoint
every raw result as it lands, and print the aggregate report.

No GEPA, no candidate rollouts, no user-policy scoring here -- see
docs/awm-plan.md for why those stay separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bandits.diagnose.compile import extract_transitions
from bandits.diagnose.fidelity import (
    TransitionFidelity,
    _recorded_observation,
    build_report,
    score_transition_fidelity,
)
from bandits.diagnose.models import GroundingTransition, Partition, SuccessShape
from bandits.diagnose.retrieve import RetrievalQuery, build_index, coverage_by_tool, retrieve
from bandits.diagnose.world import (
    TOOL_WORLD_INSTRUCTION,
    ToolWorldPredictor,
    WorldModelError,
    build_tool_world_predictor,
    prompt_digest,
    render_action,
    render_evidence,
    render_state,
    step_tool_world,
    validate_transition,
)
from bandits.store import ArtifactStore

DEFAULT_TAU_ROOT = Path("work/tau/run/proj/.bandits")
DEFAULT_CORPUS_ID = "corpus-ee3b33086ef177d7"
DEFAULT_TASK_SET_ID = "taskset-37654fb6a6bd578f"
DEFAULT_FAMILY_ID = "family-451ae91f975c"
DEFAULT_MARKERS = ("###TRANSFER###",)
VALIDATOR_SCHEMA_VERSION = 1
"""Bumped whenever validate_transition's rejection semantics change, so a
resumed run notices a validator upgrade the same way it notices a model swap."""


def load_family(
    tau_root: Path, *, corpus_id: str, task_set_id: str, family_id: str, markers: tuple[str, ...]
) -> tuple[dict[str, Any], tuple[GroundingTransition, ...], dict[str, tuple[dict[str, Any], ...]]]:
    """Real tau2 traces for one mined family, compiled to transitions.

    Mirrors bandits/diagnose/tau_smoke_test.py's loading pattern exactly, since
    that is the only place this real artifact has been read correctly before.
    Also keeps trace_id -> per-trace tool schemas, since GroundingTransition
    itself carries none.
    """
    corpus = ArtifactStore(tau_root).read(corpus_id)
    by_id = {trace.trace_id: trace for trace in corpus.traces}
    payload = json.loads((tau_root / "derived" / task_set_id / "payload.json").read_bytes())
    family = next(f for f in payload["families"] if f["family_id"] == family_id)
    traces = [by_id[tid] for tid in family["trace_ids"] if tid in by_id]

    rows: list[GroundingTransition] = []
    tool_schemas_by_trace: dict[str, tuple[dict[str, Any], ...]] = {}
    for trace in traces:
        rows.extend(extract_transitions(trace, family_id=family_id, markers=markers))
        tool_schemas_by_trace[trace.trace_id] = tuple(
            schema.simulation_projection() for schema in (trace.tools_available or ())
        )
    return family, tuple(rows), tool_schemas_by_trace


def assert_split_not_corrupt(
    family: dict[str, Any], transitions: tuple[GroundingTransition, ...]
) -> None:
    """Preflight: no lineage may straddle the fit/held-out split.

    excluded_trace_ids_for() excludes same-lineage fit traces from retrieval
    per held-out transition, which quietly repairs a corrupt split rather than
    reporting it. If a lineage genuinely straddles the split, the split itself
    is broken and the evaluation population silently shrinks; that must stop
    the run, not be filtered away.
    """
    fit_ids = set(family["fit_trace_ids"])
    held_out_ids = set(family["held_out_trace_ids"])
    fit_lineages = {
        t.lineage_id for t in transitions if t.trace_id in fit_ids and t.lineage_id
    }
    held_out_lineages = {
        t.lineage_id for t in transitions if t.trace_id in held_out_ids and t.lineage_id
    }
    overlap = fit_lineages & held_out_lineages
    if overlap:
        raise SystemExit(
            f"fit/held-out split is corrupt: {len(overlap)} lineage id(s) appear in both "
            f"partitions: {sorted(overlap)[:10]}. Refusing to run -- fix the split upstream."
        )


def dataset_report(family: dict[str, Any], transitions: tuple[GroundingTransition, ...]) -> dict[str, Any]:
    """Experiment 0: what can actually be measured, before any model call."""
    fit_ids = set(family["fit_trace_ids"])
    held_out_ids = set(family["held_out_trace_ids"])
    fit = [t for t in transitions if t.trace_id in fit_ids]
    held_out = [t for t in transitions if t.trace_id in held_out_ids]
    tool_world = [t for t in transitions if t.reaction_role in ("tool", "mixed")]
    user_world = [t for t in transitions if t.reaction_role in ("user", "mixed")]
    batched = [t for t in transitions if len(t.action_calls) > 1]
    single = [t for t in transitions if len(t.action_calls) == 1]
    with_delta = [t for t in transitions if t.inferred_state_delta]
    errored = [t for t in transitions if any(obs.error for obs in t.observations)]

    fit_index = build_index(transitions, fit_trace_ids=family["fit_trace_ids"])
    by_tool = coverage_by_tool(fit_index)

    return {
        "fit_transitions": len(fit),
        "held_out_transitions": len(held_out),
        "tool_world_transitions": len(tool_world),
        "user_policy_transitions": len(user_world),
        "single_call_transitions": len(single),
        "batched_call_transitions": len(batched),
        "transitions_with_state_deltas": len(with_delta),
        "transitions_with_errors": len(errored),
        "transitions_per_tool": {tool: row["total"] for tool, row in by_tool.items()},
        "error_transitions_per_tool": {tool: row["errors"] for tool, row in by_tool.items()},
        "reviewed_effect_catalog": "none -- no ToolEffectCatalog exists for this family; "
        "mutation-path and read/write review is NOT enforced by this run",
    }


def render_history(transition: GroundingTransition, *, limit: int = 4000) -> str:
    """Same two-step render bandits/diagnose/rollout.py uses for scenario.prefix."""
    rows = [
        {"role": step.role, "content": step.content, "tool": step.tool_name}
        for step in transition.history_before
    ]
    text = "\n".join(f"{row.get('role')}: {row.get('content')}" for row in rows)
    return text[-limit:]


def excluded_trace_ids_for(
    transition: GroundingTransition, all_transitions: tuple[GroundingTransition, ...]
) -> tuple[str, ...]:
    """A transition's own trace, plus every trace sharing its lineage_id."""
    excluded = {transition.trace_id}
    if transition.lineage_id:
        excluded |= {t.trace_id for t in all_transitions if t.lineage_id == transition.lineage_id}
    return tuple(sorted(excluded))


_UNUSED_SHAPE_FOR_TOOL_WORLD = SuccessShape.INFORMATIONAL
"""RetrievalQuery.shape is required, but retrieve() only reads it for
response_role="user_policy" (retrieve.py's _shape_compatible check is gated
on `wants_user`). extract_transitions() never populates
GroundingTransition.success_shape -- that field is only set later, on a
Scenario, from a per-task SealedSuccessContract this script has no access to.
For tool_world queries the value is inert, so a placeholder is honest here;
it must never be relied on if a user_policy query is added later."""


def build_query(
    transition: GroundingTransition, *, history_text: str, excluded_trace_ids: tuple[str, ...]
) -> RetrievalQuery:
    """A RetrievalQuery straight from a GroundingTransition (no Scenario here)."""
    return RetrievalQuery(
        family_id=transition.family_id,
        shape=transition.success_shape or _UNUSED_SHAPE_FOR_TOOL_WORLD,
        response_role="tool_world",
        task_context=transition.task_context,
        history_text=history_text,
        state_keys=tuple(field.path for field in transition.state_before.fields),
        tools=tuple(call.tool for call in transition.action_calls),
        arguments_text=" ".join(str(call.arguments) for call in transition.action_calls),
        excluded_trace_ids=excluded_trace_ids,
        partition=Partition.FIT,
    )


def assert_no_leakage(
    transition: GroundingTransition, examples: tuple[Any, ...], excluded_trace_ids: tuple[str, ...]
) -> None:
    """Post-retrieval check: no returned example may come from an excluded trace.

    Belt-and-suspenders over retrieve()'s own exclusion (retrieve.py excludes
    at query time already) -- this is the assertion the experiment promised,
    catching a future regression in retrieve() itself rather than trusting it.
    """
    excluded = set(excluded_trace_ids)
    leaked = [
        example.transition.transition_id
        for example in examples
        if example.transition.trace_id in excluded
    ]
    if leaked:
        raise AssertionError(
            f"retrieval leaked excluded-trace evidence into {transition.transition_id}: {leaked}"
        )


def select_held_out_tool_transitions(
    transitions: tuple[GroundingTransition, ...], family: dict[str, Any], *, limit: int
) -> tuple[GroundingTransition, ...]:
    """A deterministic sample stratified across tool, batch shape, and delta presence.

    Round-robins across (tool, is_batched, has_delta) buckets in a stable order
    (sorted by bucket key, then by transition_id within a bucket) so the sample
    isn't just "whatever appears first in trace order."
    """
    held_out_ids = set(family["held_out_trace_ids"])
    candidates = [
        t
        for t in transitions
        if t.trace_id in held_out_ids and t.reaction_role in ("tool", "mixed")
    ]

    buckets: dict[tuple[str, bool, bool], list[GroundingTransition]] = {}
    for transition in candidates:
        key = (
            transition.action_tool or "batch",
            len(transition.action_calls) > 1,
            bool(transition.inferred_state_delta),
        )
        buckets.setdefault(key, []).append(transition)
    for rows in buckets.values():
        rows.sort(key=lambda t: t.transition_id)

    ordered_keys = sorted(buckets)
    selected: list[GroundingTransition] = []
    while len(selected) < limit and any(buckets[key] for key in ordered_keys):
        for key in ordered_keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
            if len(selected) >= limit:
                break
    return tuple(selected)


class _RecordingPredictor:
    """Wraps a ToolWorldPredictor to capture its last raw call/response.

    score_transition_fidelity() owns the only call to step_tool_world() and
    validate_transition() and offers no hook to intercept their outputs. This
    wrapper sits at the one seam that exists -- the predictor itself -- so the
    raw model input/output can be captured for manual inspection without a
    second model call or reimplementing scoring logic.
    """

    def __init__(self, inner: ToolWorldPredictor) -> None:
        self._inner = inner
        self.last_input: dict[str, Any] | None = None
        self.last_raw_response: Any = None

    def __call__(self, *, instruction: str, state: str, history: str, action: str, evidence: str) -> Any:
        self.last_input = {
            "instruction": instruction,
            "state": state,
            "history": history,
            "action": action,
            "evidence": evidence,
        }
        response = self._inner(
            instruction=instruction, state=state, history=history, action=action, evidence=evidence
        )
        self.last_raw_response = response
        return response


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "toDict"):
        return value.toDict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def run_manifest(
    *,
    model: str,
    corpus_id: str,
    task_set_id: str,
    family_id: str,
    markers: tuple[str, ...],
    retrieval_limit: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_digest": prompt_digest(TOOL_WORLD_INSTRUCTION, model),
        "corpus_id": corpus_id,
        "task_set_id": task_set_id,
        "family_id": family_id,
        "markers": list(markers),
        "retrieval_limit": retrieval_limit,
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
    }


def run(
    *,
    tau_root: Path,
    corpus_id: str,
    task_set_id: str,
    family_id: str,
    markers: tuple[str, ...],
    model: str,
    limit: int,
    retrieval_limit: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    family, transitions, tool_schemas_by_trace = load_family(
        tau_root, corpus_id=corpus_id, task_set_id=task_set_id, family_id=family_id, markers=markers
    )
    assert_split_not_corrupt(family, transitions)
    report = dataset_report(family, transitions)
    print("=== Experiment 0: dataset report ===")
    print(json.dumps(report, indent=2, sort_keys=True))

    fit_index = build_index(transitions, fit_trace_ids=family["fit_trace_ids"])
    selected = select_held_out_tool_transitions(transitions, family, limit=limit)
    print(f"\n=== Experiment 1: {len(selected)} held-out tool transitions (stratified) ===")

    manifest = run_manifest(
        model=model,
        corpus_id=corpus_id,
        task_set_id=task_set_id,
        family_id=family_id,
        markers=markers,
        retrieval_limit=retrieval_limit,
    )
    manifest_path = output.with_suffix(".manifest.json")
    checkpoint_path = output.with_suffix(".raw.jsonl")

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise SystemExit(
                f"refusing to resume: {manifest_path} was written for a different run "
                f"configuration.\nexisting: {json.dumps(existing, sort_keys=True)}\n"
                f"requested: {json.dumps(manifest, sort_keys=True)}\n"
                f"use a different --output, or delete {manifest_path} and {checkpoint_path} "
                "to start fresh."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    already_done: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Provider-error rows carry no "fidelity" -- they are not completed
            # and must be retried on resume, not treated as cached results.
            if "fidelity" in row:
                already_done[row["transition_id"]] = row

    base_predict: ToolWorldPredictor = build_tool_world_predictor(model=model)

    scored: list[TransitionFidelity] = []
    tool_by_transition: dict[str, str] = {}

    with checkpoint_path.open("a") as checkpoint_file:
        for transition in selected:
            tool_by_transition[transition.transition_id] = transition.action_tool or "batch"

            if transition.transition_id in already_done:
                cached = already_done[transition.transition_id]
                scored.append(TransitionFidelity.model_validate(cached["fidelity"]))
                print(f"[cached] {transition.transition_id}")
                continue

            history_text = render_history(transition)
            excluded = excluded_trace_ids_for(transition, transitions)
            query = build_query(transition, history_text=history_text, excluded_trace_ids=excluded)
            examples = retrieve(query, fit_index, limit=retrieval_limit)
            assert_no_leakage(transition, examples, excluded)

            tool_schemas = tool_schemas_by_trace.get(transition.trace_id, ())
            recorder = _RecordingPredictor(base_predict)

            try:
                fidelity = score_transition_fidelity(
                    transition,
                    recorder,
                    examples=examples,
                    history_text=history_text,
                    tool_schemas=tool_schemas,
                )
            except WorldModelError as exc:
                # Known provider/packaging failure only. Anything else is a
                # bug in this script or the engine and must fail fast, not be
                # checkpointed as a provider error and hidden inside a run
                # that then looks like it finished cleanly.
                print(f"[provider-error] {transition.transition_id}: {exc}")
                checkpoint_file.write(
                    json.dumps(
                        {
                            "transition_id": transition.transition_id,
                            "trace_id": transition.trace_id,
                            "tool": transition.action_tool,
                            "provider_error": str(exc),
                        }
                    )
                    + "\n"
                )
                checkpoint_file.flush()
                continue

            # Reconstruct exactly what score_transition_fidelity saw internally,
            # by replaying the one real predict() call the recorder captured
            # back through step_tool_world() -- this reproduces its
            # retrieval-support capping (step_tool_world caps the model's
            # self-reported support at what retrieval actually found), so the
            # saved proposal/validation match what fidelity actually scored.
            # No second model call: the replay predictor returns the captured
            # response instead of calling out again.
            proposal = None
            validation = None
            if recorder.last_raw_response is not None:
                proposal = step_tool_world(
                    lambda response=recorder.last_raw_response, **_: response,
                    calls=transition.action_calls,
                    content=transition.action_content,
                    state=transition.state_before,
                    history=history_text,
                    examples=examples,
                )
                if not proposal.abstain:
                    validation = validate_transition(
                        proposal,
                        calls=transition.action_calls,
                        state=transition.state_before,
                        step_index=transition.turn_index,
                        catalog=None,
                        allowed_evidence_ids=tuple(e.transition.transition_id for e in examples),
                        tool_schemas=tool_schemas,
                    )

            rendered_input = recorder.last_input or {
                "instruction": TOOL_WORLD_INSTRUCTION,
                "state": render_state(transition.state_before),
                "history": history_text,
                "action": render_action(transition.action_calls, transition.action_content),
                "evidence": render_evidence(examples),
            }

            recorded = _recorded_observation(transition)
            checkpoint_file.write(
                json.dumps(
                    {
                        "transition_id": transition.transition_id,
                        "trace_id": transition.trace_id,
                        "tool": transition.action_tool,
                        "retrieved_transition_ids": [e.transition.transition_id for e in examples],
                        "retrieved_scores": [e.score for e in examples],
                        "rendered_input": rendered_input,
                        "raw_model_response": _jsonable(recorder.last_raw_response),
                        "raw_proposal": (
                            json.loads(proposal.model_dump_json()) if proposal is not None else None
                        ),
                        "validator_result": (
                            json.loads(validation.model_dump_json()) if validation is not None else None
                        ),
                        "tool_schemas_available": bool(tool_schemas),
                        "reviewed_effect_catalog_used": False,
                        "expected_observation": recorded.content if recorded else None,
                        "expected_state_delta": dict(transition.inferred_state_delta),
                        "expected_state_delta_status": transition.delta_ground_truth_status.value,
                        "expected_state_delta_unmatched_paths": list(transition.unmatched_post_paths),
                        "fidelity": json.loads(fidelity.model_dump_json()),
                    },
                    default=str,
                )
                + "\n"
            )
            checkpoint_file.flush()

            scored.append(fidelity)

            status = (
                "output_invalid"
                if fidelity.output_invalid
                else "abstained"
                if fidelity.abstained
                else "rejected"
                if fidelity.validator_rejected
                else f"field_accuracy={fidelity.field_accuracy}"
            )
            print(f"[{status}] {transition.transition_id} tool={transition.action_tool}")

    fidelity_report = build_report(
        scored,
        awm_version=model,
        split="held_out",
        tool_of=lambda tid: tool_by_transition.get(tid, "unknown"),
    )

    print("\n=== Aggregate fidelity ===")
    print(
        json.dumps(
            {
                "considered": fidelity_report.considered,
                "attempted": fidelity_report.attempted,
                "model_output_invalid_rate": fidelity_report.model_output_invalid_rate,
                "abstention_rate": fidelity_report.abstention_rate,
                "wrong_abstention_rate": fidelity_report.wrong_abstention_rate,
                "correct_abstention_rate": fidelity_report.correct_abstention_rate,
                "supported_coverage": fidelity_report.supported_coverage,
                "status_accuracy": fidelity_report.status_accuracy,
                "field_accuracy": fidelity_report.field_accuracy,
                "delta_accuracy": fidelity_report.delta_accuracy,
                "delta_ground_truth_coverage": fidelity_report.delta_ground_truth_coverage,
                "validation_rejection_rate": fidelity_report.validation_rejection_rate,
                "by_tool": fidelity_report.by_tool,
                "reviewed_effect_catalog": "none",
            },
            indent=2,
            sort_keys=True,
        )
    )

    output.write_text(fidelity_report.model_dump_json(indent=2))
    print(f"\nWrote report to {output}")
    print(f"Manifest at {manifest_path}")
    print(f"Raw per-transition checkpoints at {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, default=DEFAULT_TAU_ROOT)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--task-set-id", default=DEFAULT_TASK_SET_ID)
    parser.add_argument("--family-id", default=DEFAULT_FAMILY_ID)
    parser.add_argument("--marker", action="append", dest="markers", default=None)
    parser.add_argument("--model", required=True, help="Fireworks model slug, e.g. accounts/fireworks/models/...")
    parser.add_argument("--limit", type=int, default=10, help="held-out transitions to score")
    parser.add_argument("--retrieval-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("work/awm-fidelity/baseline.json"))
    args = parser.parse_args()

    if not (args.tau_root / "artifacts" / args.corpus_id).exists():
        raise SystemExit(f"no tau2 artifact at {args.tau_root}/artifacts/{args.corpus_id}")

    run(
        tau_root=args.tau_root,
        corpus_id=args.corpus_id,
        task_set_id=args.task_set_id,
        family_id=args.family_id,
        markers=tuple(args.markers) if args.markers else DEFAULT_MARKERS,
        model=args.model,
        limit=args.limit,
        retrieval_limit=args.retrieval_limit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
