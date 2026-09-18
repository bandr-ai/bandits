#!/usr/bin/env python3
"""Rescore a saved run's raw responses under the current fidelity scorer.

Zero model calls. Reloads the real transitions and rebuilds the same
retrieval that produced ``--input``'s checkpoints, then replays each
``raw_model_response`` back through ``score_transition_fidelity`` with a
predictor that returns the saved response instead of calling a model. This
is how a scorer fix (I39's grounding assessment) gets checked against a real
run without spending on it twice.

Writes a new report; never mutates ``--input``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_awm_fidelity import build_query, excluded_trace_ids_for, load_family, render_history

from bandits.diagnose.fidelity import TransitionFidelity, build_report, score_transition_fidelity
from bandits.diagnose.retrieve import build_index, retrieve
from bandits.diagnose.world import (
    TOOL_WORLD_INSTRUCTION,
    ToolWorldPredictor,
    render_action,
    render_evidence,
    render_state,
)


def _replay_predictor(response: Any) -> ToolWorldPredictor:
    def predict(**_: Any) -> Any:
        return response

    return predict


def rescore(*, tau_root: Path, input_path: Path, output: Path) -> None:
    manifest = json.loads(input_path.with_suffix(".manifest.json").read_text())
    checkpoint_path = input_path.with_suffix(".raw.jsonl")

    family, transitions, tool_schemas_by_trace = load_family(
        tau_root,
        corpus_id=manifest["corpus_id"],
        task_set_id=manifest["task_set_id"],
        family_id=manifest["family_id"],
        markers=tuple(manifest["markers"]),
    )
    by_id = {t.transition_id: t for t in transitions}
    fit_index = build_index(transitions, fit_trace_ids=family["fit_trace_ids"])

    scored: list[TransitionFidelity] = []
    tool_by_transition: dict[str, str] = {}
    rows = [
        json.loads(line)
        for line in checkpoint_path.read_text().splitlines()
        if line.strip() and "fidelity" in json.loads(line)
    ]

    for row in rows:
        transition = by_id.get(row["transition_id"])
        if transition is None:
            print(f"[skip] {row['transition_id']}: not found in reloaded family")
            continue
        tool_by_transition[transition.transition_id] = transition.action_tool or "batch"

        history_text = render_history(transition)
        excluded = excluded_trace_ids_for(transition, transitions)
        query = build_query(transition, history_text=history_text, excluded_trace_ids=excluded)
        examples = retrieve(query, fit_index, limit=len(row["retrieved_transition_ids"]) or 8)

        reconstructed_ids = [e.transition.transition_id for e in examples]
        if reconstructed_ids != row["retrieved_transition_ids"]:
            raise SystemExit(
                f"{transition.transition_id}: reconstructed retrieval does not match the "
                f"saved run -- rescoring would silently change the inputs, not just the "
                f"scorer.\nsaved:          {row['retrieved_transition_ids']}\n"
                f"reconstructed:  {reconstructed_ids}\n"
                "Refusing to rescore. If retrieve()/build_index() changed since this run, "
                "this artifact needs a fresh --model run, not a rescore."
            )

        # IDs matching is not enough -- rendering itself could have changed
        # since the saved run (a prompt/format edit), which would mean the
        # replayed raw_model_response is being scored against inputs the
        # model never actually saw. Reconstruct the exact four rendered
        # strings and require them to match the saved rendered_input
        # byte-for-byte before trusting the replay.
        saved_rendered = row.get("rendered_input") or {}
        reconstructed_rendered = {
            "instruction": TOOL_WORLD_INSTRUCTION,
            "state": render_state(transition.state_before),
            "history": history_text,
            "action": render_action(transition.action_calls, transition.action_content),
            "evidence": render_evidence(examples),
        }
        mismatched = {
            key
            for key in reconstructed_rendered
            if key not in saved_rendered or saved_rendered[key] != reconstructed_rendered[key]
        }
        if mismatched:
            raise SystemExit(
                f"{transition.transition_id}: reconstructed rendering differs from the saved "
                f"run in {sorted(mismatched)} -- the raw_model_response would be replayed "
                "against inputs the model never saw.\n"
                "Refusing to rescore. If world.py's rendering changed since this run, this "
                "artifact needs a fresh --model run, not a rescore."
            )

        fidelity = score_transition_fidelity(
            transition,
            _replay_predictor(row["raw_model_response"]),
            examples=examples,
            history_text=history_text,
            tool_schemas=tool_schemas_by_trace.get(transition.trace_id, ()),
        )
        scored.append(fidelity)

        old = row["fidelity"]
        changed = (
            old.get("abstained") != fidelity.abstained
            or old.get("abstain_correct") != fidelity.abstain_correct
            or old.get("output_invalid") != fidelity.output_invalid
        )
        marker = "CHANGED" if changed else "same"
        single_call = fidelity.grounding.single_call if fidelity.grounding else None
        grounding_kind = single_call.kind.value if single_call else None
        print(
            f"[{marker}] {transition.transition_id} tool={transition.action_tool} "
            f"abstained={fidelity.abstained} abstain_correct={fidelity.abstain_correct} "
            f"grounding={grounding_kind} "
            f"(was abstain_correct={old.get('abstain_correct')})"
        )

    report = build_report(
        scored,
        awm_version=manifest["model"],
        split="held_out",
        tool_of=lambda tid: tool_by_transition.get(tid, "unknown"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2))
    print(f"\nWrote rescored report to {output}")
    print(
        json.dumps(
            {
                "considered": report.considered,
                "attempted": report.attempted,
                "abstention_rate": report.abstention_rate,
                "wrong_abstention_rate": report.wrong_abstention_rate,
                "correct_abstention_rate": report.correct_abstention_rate,
                "model_output_invalid_rate": report.model_output_invalid_rate,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-root", type=Path, default=Path("work/tau/run/proj/.bandits"))
    parser.add_argument("--input", type=Path, required=True, help="e.g. work/awm-fidelity/baseline-fixed2.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rescore(tau_root=args.tau_root, input_path=args.input, output=args.output)


if __name__ == "__main__":
    main()
