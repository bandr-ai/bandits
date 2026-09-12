#!/usr/bin/env python3
"""Discover per-family verifier signals and score them on sealed TRAIL traces.

This starts after the existing RLM miner has produced a TaskSet. Family
membership and splits are inputs, never recomputed here. The discovery model
sees fit traces and fit labels only; held-out labels are opened only after a
candidate has survived the fit threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from signal_experiment import score_signal  # noqa: E402
from signal_synth import _compile_signal, _run_over_corpus  # noqa: E402

from bandits.analyze.models import TaskFamily
from bandits.analyze.tasksets import load_task_set
from bandits.store import ArtifactStore, DerivedStore


class ProposedSignal(BaseModel):
    name: str
    hypothesis: str
    code: str
    blind_spots: list[str] = []
    gaming_hypotheses: list[str] = []


def _render(trace: dict[str, Any], limit: int = 6_000) -> str:
    lines = [f"TASK: {trace.get('task') or '(missing)'}"]
    for span in trace.get("spans", []):
        role = "ASSISTANT" if span.get("kind") == "model" else "TOOL"
        payload = json.dumps(
            {"arguments": span.get("arguments"), "output": span.get("output")},
            sort_keys=True,
            default=str,
        )
        lines.append(f"{role} {span.get('name')} status={span.get('status')}: {payload[:1200]}")
    return "\n".join(lines)[:limit]


def _examples(
    traces: dict[str, dict[str, Any]], labels: dict[str, bool], trace_ids: tuple[str, ...]
) -> str:
    blocks = []
    for trace_id in trace_ids:
        if trace_id not in labels:
            continue
        blocks.append(
            f"--- TRACE {trace_id} LABEL={'RELIABLE' if labels[trace_id] else 'UNRELIABLE'} ---\n"
            f"{_render(traces[trace_id])}"
        )
    return "\n\n".join(blocks)


def _balanced_example_ids(labels: dict[str, bool], maximum: int) -> tuple[str, ...]:
    """Choose a deterministic, balanced discovery sample from fit only."""
    positives = sorted(trace_id for trace_id, value in labels.items() if value)
    negatives = sorted(trace_id for trace_id, value in labels.items() if not value)
    each = max(1, maximum // 2)
    chosen = positives[:each] + negatives[:each]
    if len(chosen) < maximum:
        remainder = [trace_id for trace_id in sorted(labels) if trace_id not in chosen]
        chosen.extend(remainder[: maximum - len(chosen)])
    return tuple(chosen)


_INSTRUCTION = textwrap.dedent(
    """
    You are discovering executable evidence for a replay verifier after task-family mining.
    The family and its split are already fixed. Do not regroup tasks and do not memorize ids.

    Use the Python REPL to inspect the labelled FIT examples. Propose small signals that detect
    whether the trajectory is reliable. A signal must be a pure function:

        def signal(trace):
            # trace is a dict containing task, user_turns, and spans
            # return 0..1 (higher means reliable), or None when not applicable

    Executable schema (this is the contract the returned code receives):
    - trace['task']: str | None
    - trace['user_turns']: user messages only; NEVER contains tool calls or tool results
    - trace['spans']: list[dict]
    - each span has kind ('model' or 'tool'), name, status ('ok' or 'error'), arguments, output
    - helpers already in scope: spans(trace), tool_spans(trace), model_texts(trace), json_dumps(x)
    Read tool activity from tool_spans(trace), never from user_turns.

    No imports, I/O, network, eval, exec, open, dunder access, or trace-id tests. Prefer checks
    that connect the requested outcome to recorded tool evidence over surface proxies. State
    blind spots and how an agent could game each signal. Return distinct candidates.
    """
).strip()


def build_predictor(
    *, model: str, max_iterations: int, max_llm_calls: int, max_tokens: int
) -> Callable[..., Any]:
    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError("install the audit extra to run verifier RLM discovery") from exc

    from bandits.verify.judge import resolve_api_key

    lm = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=resolve_api_key(),
        temperature=0.0,
        max_tokens=max_tokens,
    )

    class Discover(dspy.Signature):
        family_contract: str = dspy.InputField()
        fit_examples: str = dspy.InputField()
        correction: str = dspy.InputField(
            desc="empty initially; otherwise exact host-side rejection errors to repair"
        )
        signals: list[ProposedSignal] = dspy.OutputField()

    Discover.__doc__ = _INSTRUCTION
    rlm = dspy.RLM(
        Discover,
        max_iters=max_iterations,
        max_llm_calls=max_llm_calls,
        sub_lm=lm,
    )

    def predict(*, family_contract: str, fit_examples: str, correction: str = "") -> Any:
        with dspy.context(lm=lm):
            return rlm(
                family_contract=family_contract,
                fit_examples=fit_examples,
                correction=correction,
            )

    return predict


def _binary_labels(scores: dict[str, float]) -> tuple[dict[str, bool], dict[str, float]]:
    values = sorted(scores.values())
    low = values[len(values) // 3]
    high = values[-(len(values) // 3) - 1]
    labels = {
        trace_id: score >= high
        for trace_id, score in scores.items()
        if score <= low or score >= high
    }
    return labels, {"low": low, "high": high}


def _load_labels(path: Path, label_format: str) -> tuple[dict[str, bool], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if label_format == "tau2":
        labels = {
            trace_id: bool(row["success"])
            for trace_id, row in payload.items()
            if isinstance(row, dict) and isinstance(row.get("success"), bool)
        }
        return labels, {"format": "tau2 sealed success"}
    scores = {key: float(value) for key, value in payload.items()}
    labels, cuts = _binary_labels(scores)
    return labels, {"format": "TRAIL human overall reliability tertiles", **cuts}


def _metric(name: str, values: dict[str, float | None], labels: dict[str, bool]) -> dict[str, Any]:
    result = score_signal(name, values, labels)
    return result.__dict__


def _baseline_values(traces: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Cheap verifier evidence the RLM-assisted arm must improve upon."""
    clean: dict[str, float | None] = {}
    used_tool: dict[str, float | None] = {}
    concise: dict[str, float | None] = {}
    for trace in traces:
        trace_id = trace["trace_id"]
        spans = trace.get("spans", [])
        clean[trace_id] = 1.0 if all(span.get("status") != "error" for span in spans) else 0.0
        used_tool[trace_id] = 1.0 if any(span.get("kind") == "tool" for span in spans) else 0.0
        concise[trace_id] = 1.0 / (1.0 + len(spans) / 15.0)
    return {"no_span_error": clean, "used_tool": used_tool, "concise": concise}


def evaluate_family(
    family: TaskFamily,
    traces: dict[str, dict[str, Any]],
    labels: dict[str, bool],
    predict: Callable[..., Any],
    *,
    keep_auc: float,
    max_fit_examples: int = 12,
    repair_attempts: int = 1,
) -> dict[str, Any]:
    fit_labels = {
        trace_id: labels[trace_id] for trace_id in family.fit_trace_ids if trace_id in labels
    }
    held_labels = {
        trace_id: labels[trace_id] for trace_id in family.held_out_trace_ids if trace_id in labels
    }
    classes = set(fit_labels.values())
    held_classes = set(held_labels.values())
    if len(fit_labels) < 4 or len(classes) < 2 or len(held_classes) < 2:
        return {
            "family_id": family.family_id,
            "status": "skipped",
            "reason": (
                "fit needs at least four labelled traces and both classes; held-out needs "
                "both classes"
            ),
            "fit_labels": len(fit_labels),
            "held_out_labels": len(held_labels),
        }

    contract = json.dumps(
        {
            "family_id": family.family_id,
            "descriptor": family.descriptor,
            "limitations": family.limitations,
        },
        indent=2,
    )
    rendered_examples = _examples(
        traces,
        fit_labels,
        _balanced_example_ids(fit_labels, max_fit_examples),
    )
    proposed: list[ProposedSignal] = []
    correction = ""
    repairs_used = 0
    for attempt in range(repair_attempts + 1):
        reply = predict(
            family_contract=contract,
            fit_examples=rendered_examples,
            correction=correction,
        )
        raw = getattr(reply, "signals", [])
        proposed = [
            item if isinstance(item, ProposedSignal) else ProposedSignal.model_validate(item)
            for item in raw
        ]
        failures = []
        for proposal in proposed:
            try:
                _compile_signal(proposal.code)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{proposal.name}: {exc}")
        if proposed and len(failures) < len(proposed):
            break
        if attempt < repair_attempts:
            repairs_used += 1
            correction = (
                "Every proposal was rejected by the host. Return complete, self-contained "
                "`def signal(trace)` functions with no imports. Exact errors:\n- "
                + "\n- ".join(failures or ["no proposals returned"])
            )

    all_family = [traces[trace_id] for trace_id in family.trace_ids if trace_id in traces]
    baselines = {
        name: {
            "fit": _metric(name, values, fit_labels),
            "held_out": _metric(name, values, held_labels),
        }
        for name, values in _baseline_values(all_family).items()
    }
    rows = []
    for proposal in proposed:
        if "trace_id" in proposal.code:
            rows.append(
                {
                    **proposal.model_dump(),
                    "status": "rejected",
                    "reason": "trace-id access",
                    "fit": None,
                    "held_out": None,
                }
            )
            continue
        try:
            signal = _compile_signal(proposal.code)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    **proposal.model_dump(),
                    "status": "rejected",
                    "reason": str(exc),
                    "fit": None,
                    "held_out": None,
                }
            )
            continue
        values = _run_over_corpus(signal, all_family)
        fit = _metric(proposal.name, values, fit_labels)
        kept = fit.get("auc") is not None and fit["auc"] >= keep_auc
        row: dict[str, Any] = {
            **proposal.model_dump(),
            "status": "kept" if kept else "dropped",
            "fit": fit,
        }
        # This is the only point held-out labels are consulted.
        row["held_out"] = _metric(proposal.name, values, held_labels) if kept else None
        rows.append(row)
    return {
        "family_id": family.family_id,
        "status": "evaluated",
        "fit_labels": len(fit_labels),
        "held_out_labels": len(held_labels),
        "repair_attempts": repairs_used,
        "baselines": baselines,
        "signals": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_set_id")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--label-format", choices=("trail", "tau2"), default="trail")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="accounts/fireworks/models/deepseek-v4-flash-0731")
    parser.add_argument("--keep-auc", type=float, default=0.62)
    parser.add_argument("--max-families", type=int, default=5)
    parser.add_argument("--max-fit-examples", type=int, default=12)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-llm-calls", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=16_000)
    args = parser.parse_args()

    store = DerivedStore(args.project / ".bandits")
    task_set = load_task_set(args.task_set_id, store)
    corpus = ArtifactStore(args.project / ".bandits").read(task_set.corpus_id)
    traces = {trace.trace_id: trace.model_dump(mode="json") for trace in corpus.traces}
    labels, label_provenance = _load_labels(args.scores, args.label_format)
    predict = build_predictor(
        model=args.model,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_tokens=args.max_tokens,
    )

    eligible = sorted(task_set.families, key=lambda family: -len(family.trace_ids))
    results = []
    for family in eligible:
        result = evaluate_family(
            family,
            traces,
            labels,
            predict,
            keep_auc=args.keep_auc,
            max_fit_examples=args.max_fit_examples,
            repair_attempts=args.repair_attempts,
        )
        results.append(result)
        print(
            f"{family.family_id}: {result['status']} "
            f"fit={result.get('fit_labels', 0)} held={result.get('held_out_labels', 0)}"
        )
        if sum(row["status"] == "evaluated" for row in results) >= args.max_families:
            break

    report = {
        "task_set_id": args.task_set_id,
        "label_source": label_provenance,
        "held_out_was_hidden_during_discovery": True,
        "families": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
