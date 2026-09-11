#!/usr/bin/env python3
"""Score success signals against TRAIL's human reliability annotations.

TRAIL (Patronus AI, arXiv:2505.08638) is a public, ungated benchmark: 148 real
agent traces (GAIA + SWE-Bench) with human annotations naming every error, its
category, its location, and an overall reliability score from 1-5. The raw data
and annotations are checked into github.com/patronus-ai/trail-benchmark directly
-- no Hugging Face gate needed if you clone that repo.

This is the best ground truth available so far: a continuous, human-authored
quality score per trace, from a domain (open-web tool use, GAIA) completely
different from tau2 (retail dialogue) and AppWorld (multi-app task completion).
If trace-only signals also fail here, the finding is corroborated a third time
on a third domain by someone else's annotations, not ours.

Usage:
    git clone --depth 1 https://github.com/patronus-ai/trail-benchmark /tmp/trail-benchmark
    uv run python scripts/trail_signal.py \
        --trail-dir /tmp/trail-benchmark/benchmarking --split gaia \
        --out work/trail-signal --model deepseek-v4-flash-0731
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from signal_experiment import _fireworks  # noqa: E402


@dataclass
class TrailTrace:
    trace_id: str
    task: str = ""
    final_answer: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)  # {kind, name, text, is_error}
    overall: float | None = None
    n_errors: int = 0
    max_impact: str = ""


def _flatten(span: dict[str, Any], out: list[dict[str, Any]]) -> None:
    out.append(span)
    for child in span.get("child_spans") or []:
        _flatten(child, out)


def _text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return value[:limit]


def load_trail(trail_dir: Path, split: str) -> list[TrailTrace]:
    folder = {"gaia": "GAIA", "swe_bench": "SWE Bench"}[split]
    ann_folder = {"gaia": "processed_annotations_gaia", "swe_bench": "processed_annotations_swe_bench"}[
        split
    ]
    traces: list[TrailTrace] = []
    for path in sorted(glob.glob(str(trail_dir / "data" / folder / "*.json"))):
        trace_id = Path(path).stem
        ann_path = trail_dir / ann_folder / f"{trace_id}.json"
        try:
            raw = json.loads(Path(path).read_text())
            ann = json.loads(ann_path.read_text()) if ann_path.is_file() else {}
        except (json.JSONDecodeError, OSError):
            continue  # a handful of files ship malformed; skip rather than fake a label

        spans: list[dict[str, Any]] = []
        for root in raw.get("spans", []):
            _flatten(root, spans)

        traj = TrailTrace(trace_id=trace_id)
        for span in spans:
            attrs = span.get("span_attributes") or {}
            kind = attrs.get("openinference.span.kind")
            is_error = span.get("status_code") == "Error"
            if not traj.task and "task" in _text(attrs.get("input.value"), 2000):
                try:
                    parsed = json.loads(attrs.get("input.value") or "{}")
                    if isinstance(parsed, dict) and parsed.get("task"):
                        traj.task = str(parsed["task"])[:2500]
                except (json.JSONDecodeError, TypeError):
                    pass
            if kind == "LLM":
                text = _text(attrs.get("llm.output_messages.0.message.content"), 300)
                if text:
                    traj.steps.append({"kind": "llm", "name": "model", "text": text, "is_error": is_error})
            elif kind == "TOOL":
                name = attrs.get("tool.name") or span.get("span_name") or "?"
                args = _text(attrs.get("input.value"), 100)
                out = _text(attrs.get("output.value"), 150)
                traj.steps.append(
                    {"kind": "tool", "name": name, "text": f"{name}({args}) -> {out}", "is_error": is_error}
                )
                if str(name).lower() in ("final_answer", "finalanswertool", "submit"):
                    traj.final_answer = out

        scores = ann.get("scores") or [{}]
        traj.overall = scores[0].get("overall")
        errors = ann.get("errors") or []
        traj.n_errors = len(errors)
        impacts = [e.get("impact", "") for e in errors]
        traj.max_impact = "HIGH" if "HIGH" in impacts else ("MEDIUM" if "MEDIUM" in impacts else "LOW" if impacts else "")
        traces.append(traj)
    return traces


def render(traj: TrailTrace, limit_steps: int = 20) -> str:
    lines = [f"TASK: {traj.task or '(not recovered)'}"]
    for step in traj.steps[:limit_steps]:
        prefix = "ASSISTANT" if step["kind"] == "llm" else "TOOL"
        flag = " [ERROR]" if step["is_error"] else ""
        lines.append(f"{prefix}: {step['text']}{flag}")
    if traj.final_answer:
        lines.append(f"FINAL ANSWER: {traj.final_answer}")
    return "\n".join(lines)


# --------------------------------------------------------------- signals


def sig_step_count(traj: TrailTrace) -> float:
    import math

    return math.exp(-len(traj.steps) / 15.0)


def sig_no_span_errors(traj: TrailTrace) -> float:
    errs = sum(1 for s in traj.steps if s["is_error"])
    return 1.0 / (1.0 + errs)


def sig_has_final_answer(traj: TrailTrace) -> float:
    return 1.0 if traj.final_answer else 0.0


DETERMINISTIC = {
    "step_count": sig_step_count,
    "no_span_errors": sig_no_span_errors,
    "has_final_answer": sig_has_final_answer,
}


_JUDGE_PROMPT = """You audit an AI agent's execution trace for reliability, exactly as a human \
QA reviewer would. Judge whether the agent followed a sound process and reached a well-supported \
answer -- not just whether it produced *an* answer.

Score 1 (very unreliable: hallucinated steps, ignored the plan, no evidence for claims) to \
5 (fully reliable: every claim traces to a real tool result, no skipped steps).

TRACE:
{trace}

Reply with ONLY one JSON object: {{"score": 1-5, "reason": "one sentence"}}"""


def judge_one(traj: TrailTrace, model: str) -> float | None:
    reply = _fireworks(model, _JUDGE_PROMPT.format(trace=render(traj)), 0.0, max_tokens=500)
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0:
        return None
    try:
        obj = json.loads(reply[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    score = obj.get("score")
    return float(score) if isinstance(score, (int, float)) else None


# --------------------------------------------------------------- scoring


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(pairs)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trail-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["gaia", "swe_bench"], default="gaia")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = [t for t in load_trail(args.trail_dir, args.split) if t.overall is not None]
    print(f"{len(traces)} traces with a human overall score, split={args.split}")
    values = sorted(t.overall for t in traces)
    lo, hi = values[len(values) // 3], values[-(len(values) // 3) - 1]
    print(f"overall score range {values[0]}-{values[-1]}, tertile cuts <= {lo} / >= {hi}")

    truth_binary = {t.trace_id: t.overall for t in traces if t.overall <= lo or t.overall >= hi}
    truth_binary = {k: (v >= hi) for k, v in truth_binary.items()}

    scores: dict[str, dict[str, float | None]] = {}
    for name, fn in DETERMINISTIC.items():
        scores[name] = {t.trace_id: fn(t) for t in traces}

    if not args.skip_model:
        model = args.model if args.model.startswith("accounts/") else f"accounts/fireworks/models/{args.model}"
        cache_path = args.out / f"judge-{args.split}.jsonl"
        cache = {}
        if cache_path.is_file():
            cache = {
                json.loads(line)["trace_id"]: json.loads(line)
                for line in cache_path.read_text().splitlines()
                if line.strip()
            }
        todo = [t for t in traces if t.trace_id not in cache]
        print(f"judge: {len(cache)} cached, {len(todo)} to run", file=sys.stderr)

        def one(t: TrailTrace) -> dict[str, Any]:
            try:
                return {"trace_id": t.trace_id, "score": judge_one(t, model)}
            except Exception as exc:  # noqa: BLE001
                return {"trace_id": t.trace_id, "score": None, "error": str(exc)}

        if todo:
            with ThreadPoolExecutor(max_workers=args.workers) as pool, cache_path.open("a") as sink:
                for row in pool.map(one, todo):
                    cache[row["trace_id"]] = row
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
        scores["llm_judge"] = {tid: r.get("score") for tid, r in cache.items()}

    print(f"\n{'signal':<16} {'spearman':>9} {'AUC(tertiles)':>14} {'n_scored':>9}")
    print("-" * 52)
    for name, sc in scores.items():
        cont_pairs = [(sc[t.trace_id], t.overall) for t in traces if sc.get(t.trace_id) is not None]
        bin_pairs = [(sc[tid], y) for tid, y in truth_binary.items() if sc.get(tid) is not None]
        rho = spearman(cont_pairs)
        a = auc(bin_pairs)
        print(
            f"{name:<16} {('n/a' if rho is None else f'{rho:.3f}'):>9} "
            f"{('n/a' if a is None else f'{a:.3f}'):>14} {len(cont_pairs):>9}"
        )

    (args.out / f"report-{args.split}.json").write_text(
        json.dumps(
            {
                "n": len(traces),
                "score_range": [values[0], values[-1]],
                "signals": {
                    name: {
                        "spearman": spearman(
                            [(sc[t.trace_id], t.overall) for t in traces if sc.get(t.trace_id) is not None]
                        ),
                        "auc_tertiles": auc(
                            [(sc[tid], y) for tid, y in truth_binary.items() if sc.get(tid) is not None]
                        ),
                    }
                    for name, sc in scores.items()
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
