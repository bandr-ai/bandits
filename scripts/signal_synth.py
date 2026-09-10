#!/usr/bin/env python3
"""Let a model write the success signals, then keep only the ones that separate.

The hand-written signals in ``signal_experiment.py`` are a guess at what a
successful trajectory looks like. This script removes the guess: it shows a model
a stratified handful of labelled trajectories, asks it for Python predicate
functions that it thinks tell the two classes apart, then executes every proposal
over the whole corpus and scores it against the labels. Only proposals that beat
a discrimination floor are written out.

The label the proposals are scored against is, in order of preference:
  1. sealed benchmark truth, when ``--truth`` is given;
  2. otherwise the sibling-consensus proxy -- a run is a positive when >=75% of
     its task siblings made the same mutations, a negative when <=25% did.

So on a corpus with no truth at all, this still runs: it discovers signals that
agree with the sibling jury, which is itself the strongest label-free signal
found so far.

Model code is run under an AST allowlist (no imports, no dunder access, no
open/eval/exec/getattr), restricted builtins, and a per-trace wall-clock guard.
It is still model-authored code executing locally; run it on a corpus you trust
and read ``discovered_signals.py`` before reusing it.

Usage:
    uv run python scripts/signal_synth.py \
        --corpus work/tau2-hitl-eval/.bandits/artifacts/corpus-85e4bd83c00ff7df \
        --truth  work/tau2-run/tau2.labels.json \
        --out    work/signal-synth \
        --model  deepseek-v4-flash-0731 --rounds 2 --proposals 6
"""

from __future__ import annotations

import argparse
import ast
import json
import signal as signal_mod
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from signal_experiment import (  # noqa: E402
    _fireworks,
    _load_corpus,
    _load_truth,
    _model_texts,
    _spans,
    _tool_spans,
    score_signal,
    sibling_consensus,
)

# ------------------------------------------------------------------- labelling


def weak_labels(traces: list[dict[str, Any]], truth: dict[str, bool] | None) -> dict[str, bool]:
    if truth:
        return dict(truth)
    share, _ = sibling_consensus(traces)
    out: dict[str, bool] = {}
    for trace_id, value in share.items():
        if value is None:
            continue
        if value >= 0.75:
            out[trace_id] = True
        elif value <= 0.25:
            out[trace_id] = False
    if not out:
        raise SystemExit(
            "no --truth and sibling consensus produced no confident labels; "
            "this corpus has nothing to synthesise against"
        )
    return out


# ------------------------------------------------------------- prompt / digest


def digest(trace: dict[str, Any], *, calls: int = 40) -> str:
    tools = _tool_spans(trace)
    texts = _model_texts(trace)
    lines = [
        f"task: {(trace.get('task') or '')[:400]}",
        f"spans: {len(_spans(trace))}  tool_calls: {len(tools)}  assistant_turns: {len(texts)}",
    ]
    seq = []
    for span in tools[:calls]:
        args = json.dumps(span.get("arguments", {}), default=str)
        seq.append(f"{span['name']}({args[:120]})={'ERR' if span.get('status') == 'error' else 'ok'}")
    lines.append("calls: " + " | ".join(seq) if seq else "calls: (none)")
    if texts:
        lines.append(f"final_assistant: {texts[-1][:400]}")
    return "\n".join(lines)


_SYSTEM = textwrap.dedent(
    """\
    You are finding features that predict whether a tool-using agent SUCCEEDED at its task.

    You will see labelled example trajectories. Propose Python functions, each of the form:

        def signal(trace):
            \"\"\"<one-line hypothesis about why this separates success from failure>\"\"\"
            ...
            return <float in 0..1, higher = more likely SUCCESS>  # or None if not applicable

    Rules for the code:
    - Pure function of `trace` only. No imports, no I/O, no `open`, no `eval`, no `__` attributes.
    - Available helpers (already in scope): spans(trace), tool_spans(trace), model_texts(trace).
      `trace` is a dict with keys: task (str|None), lineage_id (str|None), spans (list).
      Each span dict has: kind ('model'|'tool'), name, status ('ok'|'error'), arguments (dict), output.
    - Available builtins: len, any, all, sum, min, max, sorted, set, list, dict, str, int, float,
      bool, range, enumerate, zip, abs, round, isinstance, json_dumps (use instead of json.dumps).
    - Return None when the feature does not apply to a trace; do not guess.
    - Each function must be self-contained and under 25 lines.

    Return ONLY a JSON array of objects: [{"name": "...", "code": "def signal(trace): ..."}].
    Give each function a distinct, descriptive name. Aim for features that are cheap and general,
    not ones that memorise these specific tasks.
    """
)


def _examples_block(traces_by_id, labels, ids) -> str:
    out = []
    for trace_id in ids:
        out.append(f"--- {trace_id}  LABEL={'SUCCESS' if labels[trace_id] else 'FAILURE'} ---")
        out.append(digest(traces_by_id[trace_id]))
    return "\n".join(out)


# ----------------------------------------------------------------- safe exec

_ALLOWED_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len", "any", "all", "sum", "min", "max", "sorted", "set", "list", "dict",
        "str", "int", "float", "bool", "range", "enumerate", "zip", "abs", "round",
        "isinstance", "tuple", "map", "filter", "reversed",
    )
}


class _Rejected(Exception):
    pass


def _check_ast(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise _Rejected(f"syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise _Rejected("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise _Rejected(f"dunder attribute {node.attr!r}")
        if isinstance(node, ast.Name) and node.id in {
            "eval", "exec", "compile", "open", "globals", "locals", "vars",
            "getattr", "setattr", "delattr", "__import__", "input", "breakpoint",
        }:
            raise _Rejected(f"name {node.id!r} is not allowed")


class _Timeout(Exception):
    pass


def _compile_signal(code: str) -> Callable[[dict[str, Any]], float | None]:
    _check_ast(code)
    namespace: dict[str, Any] = {
        "__builtins__": _ALLOWED_BUILTINS,
        "spans": _spans,
        "tool_spans": _tool_spans,
        "model_texts": _model_texts,
        "json_dumps": lambda obj: json.dumps(obj, default=str, sort_keys=True),
    }
    exec(compile(code, "<signal>", "exec"), namespace)  # noqa: S102 - sandboxed above
    fn = namespace.get("signal")
    if not callable(fn):
        raise _Rejected("no callable named 'signal' defined")
    return fn


def _run_over_corpus(
    fn: Callable[[dict[str, Any]], float | None], traces: list[dict[str, Any]]
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}

    def _alarm(signum, frame):  # noqa: ANN001
        raise _Timeout()

    old = signal_mod.signal(signal_mod.SIGALRM, _alarm)
    try:
        for trace in traces:
            signal_mod.setitimer(signal_mod.ITIMER_REAL, 2.0)
            try:
                value = fn(trace)
                if value is None:
                    scores[trace["trace_id"]] = None
                else:
                    value = float(value)
                    scores[trace["trace_id"]] = max(0.0, min(1.0, value))
            except (_Timeout, Exception):  # noqa: BLE001 - a bad proposal must not kill the run
                scores[trace["trace_id"]] = None
            finally:
                signal_mod.setitimer(signal_mod.ITIMER_REAL, 0)
    finally:
        signal_mod.signal(signal_mod.SIGALRM, old)
    return scores


# --------------------------------------------------------------------- driver


def _parse_proposals(reply: str) -> list[dict[str, str]]:
    """Pull {name, code} objects out of the reply, tolerating a truncated tail.

    The model returns a JSON array; a long round hits the token cap mid-array. So
    try the whole array first, then fall back to decoding objects one at a time
    with a raw decoder and skipping whatever trailing fragment did not close.
    """
    start = reply.find("[")
    if start < 0:
        return []
    body = reply[start:]
    try:
        items = json.loads(body[: body.rfind("]") + 1])
        return _collect(items)
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    out: list[dict[str, str]] = []
    i = body.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(body[i:])
        except ValueError:
            break
        if isinstance(obj, dict):
            out.extend(_collect([obj]))
        nxt = body.find("{", i + end)
        i = nxt
    return out


def _collect(items: list[Any]) -> list[dict[str, str]]:
    out = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("code"), str) and "def signal" in item["code"]:
            out.append({"name": str(item.get("name") or f"signal_{len(out)}"), "code": item["code"]})
    return out


def _stratified(labels: dict[str, bool], n: int) -> list[str]:
    pos = [t for t, y in labels.items() if y]
    neg = [t for t, y in labels.items() if not y]
    half = n // 2
    return sorted(pos[:half]) + sorted(neg[: n - half])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--proposals", type=int, default=6)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--keep-auc", type=float, default=0.62)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = _load_corpus(args.corpus)
    truth = _load_truth(args.truth) if args.truth else None
    if truth:
        present = {t["trace_id"] for t in traces}
        truth = {k: v for k, v in truth.items() if k in present}
    labels = weak_labels(traces, truth)
    traces = [t for t in traces if t["trace_id"] in labels]
    by_id = {t["trace_id"]: t for t in traces}
    pos = sum(1 for v in labels.values() if v)
    print(
        f"{len(labels)} labelled ({pos} pos / {len(labels) - pos} neg) "
        f"from {'sealed truth' if truth else 'sibling consensus'}\n"
    )

    model_path = (
        args.model if args.model.startswith("accounts/") else f"accounts/fireworks/models/{args.model}"
    )
    example_ids = _stratified(labels, args.examples)
    kept: dict[str, dict[str, Any]] = {}
    transcript_note = ""

    for rnd in range(1, args.rounds + 1):
        print(f"=== round {rnd} ===")
        user = (
            f"Labelled examples:\n{_examples_block(by_id, labels, example_ids)}\n\n"
            f"Propose {args.proposals} signal functions."
        )
        if transcript_note:
            user += (
                f"\n\nSignals kept so far and their AUC:\n{transcript_note}\n"
                "Propose DIFFERENT features that would catch the cases those miss."
            )
        reply = _fireworks(
            model_path, f"{_SYSTEM}\n\n{user}", 0.4 if rnd > 1 else 0.2, max_tokens=6000
        )
        (args.out / f"round-{rnd}-reply.txt").write_text(reply)
        proposals = _parse_proposals(reply)
        print(f"  {len(proposals)} proposals parsed")

        for prop in proposals:
            name = prop["name"]
            try:
                fn = _compile_signal(prop["code"])
            except _Rejected as exc:
                print(f"  reject {name}: {exc}")
                continue
            scores = _run_over_corpus(fn, traces)
            applied = sum(1 for v in scores.values() if v is not None)
            if applied < 0.2 * len(traces):
                print(f"  drop   {name}: applied to only {applied}/{len(traces)}")
                continue
            m = score_signal(name, scores, labels)
            tag = "KEEP" if (m.auc or 0) >= args.keep_auc and name not in kept else "  ok"
            auc_s = "n/a  " if m.auc is None else f"{m.auc:.3f}"
            print(
                f"  {tag} {name:32} AUC={auc_s}  prec={(m.precision or 0):.2f}"
                f"  cover={m.coverage:.2f}  applied={applied}"
            )
            if tag == "KEEP":
                kept[name] = {"code": prop["code"], "metrics": m.__dict__}

        transcript_note = "\n".join(
            f"- {n}: AUC {d['metrics']['auc']:.3f}" for n, d in kept.items()
        ) or "(none yet)"

    # write the survivors
    out_py = args.out / "discovered_signals.py"
    header = (
        '"""Model-discovered success signals. Auto-generated by scripts/signal_synth.py.\n\n'
        f"corpus: {args.corpus}\n"
        f"label source: {'sealed truth' if truth else 'sibling consensus'}\n"
        f"kept AUC >= {args.keep_auc}\n"
        '"""\n\n'
        "from signal_experiment import model_texts, spans, tool_spans  # noqa: F401\n\n"
        "def json_dumps(obj):\n"
        "    import json\n"
        "    return json.dumps(obj, default=str, sort_keys=True)\n\n\n"
    )
    body = "\n\n".join(
        f"# AUC {d['metrics']['auc']:.3f}  precision {d['metrics']['precision']}\n"
        f"{d['code'].strip()}\n\n{name} = signal  # noqa: F821\ndel signal"
        for name, d in kept.items()
    )
    out_py.write_text(header + body + "\n")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "label_source": "sealed truth" if truth else "sibling consensus",
                "kept": {n: d["metrics"] for n, d in kept.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nkept {len(kept)} signal(s) -> {out_py}")


if __name__ == "__main__":
    main()
