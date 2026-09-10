#!/usr/bin/env python3
"""Score per-trajectory success signals against sealed benchmark truth.

The verifier pipeline induces a per-family deterministic reward and needs labels
to rank its candidates. SFT selection does not: it only has to sort trajectories
that already happened, once, and can spend an expensive read per trajectory that
is never queried again.

This script measures whether cheap per-trajectory signals can stand in for that
reward. Each signal maps one trace to a score in [0, 1] (higher meaning more
likely a success) or to ``None`` when it cannot say. Every signal is then scored
against the sealed benchmark labels on three things that matter for demonstration
selection:

* positive-set precision at a high-precision operating point -- the rate at which
  an admitted trajectory really was a success, which is the number a bad value
  poisons;
* coverage -- the share of the corpus that operating point admits, since a
  perfect filter that keeps three rows is not useful;
* ROC AUC -- threshold-free separation, so a signal that ranks well but needs a
  dataset-specific cut is not dismissed for the cut.

The bar is the majority-class baseline: predicting "success" for everything.

Usage:
    uv run python scripts/signal_experiment.py \
        --corpus work/tau2-hitl-eval/.bandits/artifacts/corpus-85e4bd83c00ff7df \
        --truth work/tau2-run/tau2.labels.json \
        --out work/signal-experiment

    # add the model judge (cached under --out/judge-cache):
    uv run python scripts/signal_experiment.py ... --judge deepseek-v4-pro --judge-samples 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- io


def _load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    payload = corpus_dir / "corpus.json"
    if not payload.is_file():
        raise SystemExit(f"no corpus.json under {corpus_dir}")
    return json.loads(payload.read_text())["traces"]


def _load_truth(path: Path) -> dict[str, bool]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit("truth file must be a JSON object keyed by trace id")
    out: dict[str, bool] = {}
    for trace_id, value in raw.items():
        if isinstance(value, dict) and isinstance(value.get("success"), bool):
            out[trace_id] = value["success"]
        elif isinstance(value, bool):
            out[trace_id] = value
    if not out:
        raise SystemExit("truth file carried no boolean success labels")
    return out


# ------------------------------------------------------------------ transcript

_WRITE_TOOLS = {
    # tau2 retail mutations. Read-only tools are everything else.
    "modify_pending_order_items",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_user_address",
    "modify_pending_order_payment",
    "place_order",
}

_GIVE_UP = (
    "transfer_to_human_agents",
    "transfer_to_human",
    "escalate",
)

_DONE_PHRASES = (
    "has been submitted",
    "have submitted",
    "has been processed",
    "have processed",
    "has been cancelled",
    "have cancelled",
    "has been completed",
    "has been placed",
    "have updated",
    "has been updated",
    "successfully",
    "is now scheduled",
    "all set",
)

_REFUSAL_PHRASES = (
    "cannot",
    "can't",
    "unable to",
    "not able to",
    "i'm sorry",
    "i am sorry",
    "apologi",
    "against our policy",
    "not possible",
)


def _spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return trace.get("spans", [])


def _model_texts(trace: dict[str, Any]) -> list[str]:
    out = []
    for span in _spans(trace):
        if span["kind"] != "model":
            continue
        value = span.get("output")
        if isinstance(value, str) and value.strip():
            out.append(value)
    return out


def _tool_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [span for span in _spans(trace) if span["kind"] == "tool"]


def render_transcript(trace: dict[str, Any], *, limit: int = 9000) -> str:
    lines = [f"TASK: {trace.get('task') or '(none recorded)'}"]
    for span in _spans(trace):
        if span["kind"] == "model":
            value = span.get("output")
            if isinstance(value, str) and value.strip():
                lines.append(f"ASSISTANT: {value.strip()}")
        else:
            args = json.dumps(span.get("arguments", {}), default=str)[:400]
            out = json.dumps(span.get("output"), default=str)[:600]
            err = " [ERROR]" if span.get("status") == "error" else ""
            lines.append(f"TOOL {span['name']}({args}) -> {out}{err}")
    text = "\n".join(lines)
    if len(text) > limit:
        # keep the head and the tail: the opening ask and the closing state are
        # where a judgement is actually made.
        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        text = f"{head}\n...[{len(text) - limit} chars elided]...\n{tail}"
    return text


# --------------------------------------------------------------- deterministic

Signal = Callable[[dict[str, Any]], float | None]


def sig_no_giveup(trace: dict[str, Any]) -> float | None:
    names = {span["name"] for span in _tool_spans(trace)}
    return 0.0 if names & set(_GIVE_UP) else 1.0


def sig_made_a_write(trace: dict[str, Any]) -> float | None:
    names = [span["name"] for span in _tool_spans(trace)]
    return 1.0 if any(name in _WRITE_TOOLS for name in names) else 0.0


def sig_no_repeat_calls(trace: dict[str, Any]) -> float | None:
    calls = Counter(
        (span["name"], json.dumps(span.get("arguments", {}), sort_keys=True, default=str))
        for span in _tool_spans(trace)
    )
    repeated = sum(1 for count in calls.values() if count > 1)
    return 1.0 / (1.0 + repeated)


def sig_no_tool_error(trace: dict[str, Any]) -> float | None:
    errors = sum(1 for span in _spans(trace) if span.get("status") == "error")
    return 1.0 / (1.0 + errors)


def sig_final_claims_done(trace: dict[str, Any]) -> float | None:
    texts = _model_texts(trace)
    if not texts:
        return None
    final = texts[-1].lower()
    done = any(phrase in final for phrase in _DONE_PHRASES)
    refuse = any(phrase in final for phrase in _REFUSAL_PHRASES)
    if done and not refuse:
        return 1.0
    if refuse and not done:
        return 0.0
    return 0.5


def sig_short_trajectory(trace: dict[str, Any]) -> float | None:
    # Flailing runs are long. Score is 1 for the shortest, decaying with length.
    n = len(_spans(trace))
    return math.exp(-n / 40.0)


def sig_ends_on_assistant(trace: dict[str, Any]) -> float | None:
    spans = _spans(trace)
    if not spans:
        return None
    # A run that ends on a tool result never delivered a closing summary.
    return 1.0 if spans[-1]["kind"] == "model" else 0.5


def sig_read_before_write(trace: dict[str, Any]) -> float | None:
    """Did every mutation follow at least one read of the same order/user?

    A blind write with no preceding get_* is the shape of an agent acting on an
    assumption. Cheap proxy: index of first write must be greater than zero and
    at least two reads precede it.
    """
    tools = _tool_spans(trace)
    first_write = next((i for i, s in enumerate(tools) if s["name"] in _WRITE_TOOLS), None)
    if first_write is None:
        return None
    reads_before = sum(1 for s in tools[:first_write] if s["name"].startswith(("get_", "find_", "list_")))
    return min(1.0, reads_before / 2.0)


DETERMINISTIC: dict[str, Signal] = {
    "no_giveup": sig_no_giveup,
    "made_a_write": sig_made_a_write,
    "no_repeat_calls": sig_no_repeat_calls,
    "no_tool_error": sig_no_tool_error,
    "final_claims_done": sig_final_claims_done,
    "short_trajectory": sig_short_trajectory,
    "ends_on_assistant": sig_ends_on_assistant,
    "read_before_write": sig_read_before_write,
}


# --------------------------------------------------------- sibling consensus

_READ_VERBS = (
    "get", "list", "find", "search", "read", "fetch", "show", "view", "lookup",
    "describe", "query", "check", "count", "grep", "glob", "cat", "ls",
)
_WRITE_VERBS = (
    "modify", "update", "set", "create", "add", "remove", "delete", "cancel",
    "return", "exchange", "place", "submit", "send", "post", "put", "patch",
    "write", "edit", "apply", "transfer", "refund", "issue", "assign", "move",
    "rename", "insert", "drop", "exec", "run", "install", "deploy",
)


def _looks_mutating(tool_name: str) -> bool:
    """Classify a tool as state-changing from its name alone, no domain list.

    tau2's ``modify_pending_order_items`` and Claude Code's ``Edit`` both land on
    the write side; ``get_order_details`` and ``Read`` on the read side. A name
    that matches neither verb set is treated as mutating: for a fingerprint that
    is meant to catch a run doing something different, a false "it changed state"
    is safer than missing one.
    """
    head = re.split(r"[_\-\s]", tool_name.strip().lower(), maxsplit=1)[0]
    if head in _READ_VERBS or tool_name[:1].islower() and any(
        tool_name.lower().startswith(v) for v in _READ_VERBS
    ):
        return False
    if head in _WRITE_VERBS or any(tool_name.lower().startswith(v) for v in _WRITE_VERBS):
        return True
    # CamelCase single-word tools (Claude Code style): first token.
    camel = re.match(r"[A-Z][a-z]+", tool_name)
    if camel:
        word = camel.group(0).lower()
        if word in _READ_VERBS:
            return False
        if word in _WRITE_VERBS:
            return True
    return True


def _mutation_fingerprint(trace: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """The set of state-changing calls a run made, name plus verbatim arguments.

    Verbatim on purpose: two runs that both call ``exchange_delivered_order_items``
    with the same item ids and the same new-variant ids did the same thing, and
    any normalisation that merges "added the items in a different order" also
    merges "exchanged for a different variant".
    """
    calls = []
    for span in _tool_spans(trace):
        if _looks_mutating(span["name"]):
            args = json.dumps(span.get("arguments", {}), sort_keys=True, default=str)
            calls.append((span["name"], args))
    return tuple(sorted(calls))


def sibling_consensus(
    traces: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Score each run by how many of its task siblings made the identical mutations.

    Repeated independent rollouts of one task are a free jury. When all four
    converge on the same writes they are almost all correct; the run whose
    mutation set is unique among its siblings is the one that went somewhere
    else. Needs no labels and no model -- only that the corpus declares which
    runs share a task, which ``lineage_id`` does.

    Returns two signals: ``share`` (fraction of the task's runs, including this
    one, with this exact fingerprint) and ``is_modal`` (1.0 when this run's
    fingerprint is the task's most common).
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        by_task.setdefault(trace.get("lineage_id") or trace["trace_id"], []).append(trace)

    share: dict[str, float | None] = {}
    is_modal: dict[str, float | None] = {}
    for group in by_task.values():
        prints = {t["trace_id"]: _mutation_fingerprint(t) for t in group}
        counts = Counter(prints.values())
        modal_print, _ = counts.most_common(1)[0]
        for trace_id, fingerprint in prints.items():
            if len(group) < 2:
                share[trace_id] = None
                is_modal[trace_id] = None
                continue
            share[trace_id] = counts[fingerprint] / len(group)
            is_modal[trace_id] = 1.0 if fingerprint == modal_print else 0.0
    return share, is_modal


# ---------------------------------------------------------------- model judge

_JUDGE_PROMPT = """You check whether a retail customer-service agent got the customer the outcome they asked for.

Focus on the end state, not on style or on extra clarifying questions:
- Identify every concrete thing the customer wanted changed (which order, which items, which new variants, which address, cancel vs return vs exchange).
- Check the agent actually performed each one: the matching modify / exchange / return / cancel tool call was made and returned without an error.
- The items, variant options, quantities and destinations in that call must match what the customer asked for.
- An agent that made no state-changing call, made the wrong one, changed the wrong order, or only did part of the request did NOT succeed.
- An agent that correctly declined something genuinely disallowed (e.g. modifying an already-delivered order) and did everything else DID succeed.

Transcript:
{transcript}

Reply with ONLY one JSON object:
{{"verdict": "success" | "failure", "confidence": 0.0-1.0, "reason": "one sentence naming the deciding action or gap"}}"""


def _fireworks_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key and Path(".env").is_file():
        for line in Path(".env").read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "FIREWORKS_API_KEY":
                key = value.strip().strip("'\"")
                break
    if not key:
        raise SystemExit("FIREWORKS_API_KEY not set and not in .env")
    return key


def _fireworks(model: str, prompt: str, temperature: float, *, max_tokens: int = 2400) -> str:
    body = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            # Reasoning models spend most of this before the visible answer; a
            # tight budget truncates the JSON and reads back as a parse failure.
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    last: Exception | None = None
    for _ in range(5):
        request = urllib.request.Request(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {_fireworks_key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"].get("content") or ""
            if content.strip():
                return content
            last = RuntimeError("model returned empty content")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f"model {model} is not deployed (404)") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            last = exc
    raise RuntimeError(f"fireworks call failed after retries: {last}")


def _parse_verdict(reply: str) -> tuple[float | None, str]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        return None, reply[:200]
    try:
        parsed = json.loads(reply[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None, reply[:200]
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict not in ("success", "failure"):
        return None, json.dumps(parsed)[:200]
    score = 1.0 if verdict == "success" else 0.0
    return score, str(parsed.get("reason", ""))[:300]


@dataclass
class JudgeConfig:
    model_id: str
    samples: int
    temperature: float

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {"prompt": _JUDGE_PROMPT, "model": self.model_id, "temp": self.temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_judge(
    traces: list[dict[str, Any]],
    cfg: JudgeConfig,
    cache_dir: Path,
    *,
    workers: int = 8,
) -> dict[str, float | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.model_id.replace('/', '_')}-{cfg.digest}.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["trace_id"]] = row

    todo = [t for t in traces if t["trace_id"] not in cached]
    print(f"judge {cfg.model_id}: {len(cached)} cached, {len(todo)} to run", file=sys.stderr)

    model_path = (
        cfg.model_id
        if cfg.model_id.startswith("accounts/")
        else f"accounts/fireworks/models/{cfg.model_id}"
    )

    def judge_one(trace: dict[str, Any]) -> dict[str, Any]:
        prompt = _JUDGE_PROMPT.format(transcript=render_transcript(trace))
        scores: list[float] = []
        reasons: list[str] = []
        notes: list[str] = []
        for _ in range(cfg.samples):
            temp = cfg.temperature if cfg.samples > 1 else 0.0
            try:
                reply = _fireworks(model_path, prompt, temp)
            except RuntimeError as exc:
                notes.append(f"call failed: {exc}")
                continue
            score, reason = _parse_verdict(reply)
            if score is None:
                notes.append(f"unparsed: {reason}")
                continue
            scores.append(score)
            reasons.append(reason)
        mean = statistics.mean(scores) if scores else None
        return {
            "trace_id": trace["trace_id"],
            "score": mean,
            "samples": scores,
            "reasons": reasons,
            "notes": notes,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as sink:
            for i, row in enumerate(pool.map(judge_one, todo), 1):
                cached[row["trace_id"]] = row
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                if i % 25 == 0:
                    print(f"  judged {i}/{len(todo)}", file=sys.stderr)

    return {tid: row.get("score") for tid, row in cached.items()}


# --------------------------------------------------------------------- scoring


@dataclass
class Metrics:
    name: str
    n: int
    scored: int
    auc: float | None
    # high-precision operating point
    threshold: float
    admitted: int
    precision: float | None
    recall: float | None
    coverage: float
    # balanced view at 0.5
    accuracy_at_half: float | None
    balanced_at_half: float | None
    extra: dict[str, Any] = field(default_factory=dict)


def _auc(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def score_signal(
    name: str,
    scores: dict[str, float | None],
    truth: dict[str, bool],
    *,
    target_precision: float = 0.90,
) -> Metrics:
    pairs = [
        (scores[tid], truth[tid])
        for tid in truth
        if tid in scores and scores[tid] is not None
    ]
    n = len(truth)
    scored = len(pairs)
    auc = _auc(pairs)

    # Sweep thresholds; pick the lowest that still clears target precision, to
    # maximise coverage subject to the precision floor. Fall back to the
    # highest-precision point if the floor is never met.
    candidates = sorted({s for s, _ in pairs} | {0.0, 0.5, 1.0})
    best: tuple[float, float, float, float] | None = None  # thr, prec, rec, cov
    fallback: tuple[float, float, float, float] | None = None
    for thr in candidates:
        admitted = [(s, y) for s, y in pairs if s >= thr]
        if not admitted:
            continue
        tp = sum(1 for _, y in admitted if y)
        prec = tp / len(admitted)
        rec = tp / max(1, sum(1 for _, y in pairs if y))
        cov = len(admitted) / scored
        if fallback is None or prec > fallback[1] or (prec == fallback[1] and cov > fallback[3]):
            fallback = (thr, prec, rec, cov)
        if prec >= target_precision:
            if best is None or cov > best[3]:
                best = (thr, prec, rec, cov)
    chosen = best or fallback or (1.0, 0.0, 0.0, 0.0)

    half = [(s, y) for s, y in pairs]
    acc_half = bal_half = None
    if half:
        tp = sum(1 for s, y in half if s >= 0.5 and y)
        tn = sum(1 for s, y in half if s < 0.5 and not y)
        fp = sum(1 for s, y in half if s >= 0.5 and not y)
        fn = sum(1 for s, y in half if s < 0.5 and y)
        acc_half = (tp + tn) / len(half)
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        bal_half = (tpr + tnr) / 2

    return Metrics(
        name=name,
        n=n,
        scored=scored,
        auc=auc,
        threshold=chosen[0],
        admitted=int(round(chosen[3] * scored)),
        precision=chosen[1],
        recall=chosen[2],
        coverage=chosen[3],
        accuracy_at_half=acc_half,
        balanced_at_half=bal_half,
    )


def and_combo(
    names: list[str],
    all_scores: dict[str, dict[str, float | None]],
    truth: dict[str, bool],
    *,
    thresholds: dict[str, float],
) -> Metrics:
    combined: dict[str, float | None] = {}
    for tid in truth:
        vals = []
        ok = True
        for name in names:
            s = all_scores[name].get(tid)
            if s is None:
                ok = False
                break
            vals.append(1.0 if s >= thresholds.get(name, 0.5) else 0.0)
        combined[tid] = (min(vals) if ok and vals else None)
    return score_signal(" AND ".join(names), combined, truth)


# ------------------------------------------------------------------------ main


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "  n/a"
    return f"{value * 100:5.1f}" if pct else f"{value:5.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="dir holding corpus.json")
    parser.add_argument("--truth", type=Path, required=True, help="sealed labels JSON")
    parser.add_argument("--out", type=Path, required=True, help="output dir for report + cache")
    parser.add_argument("--judge", default=None, help="fireworks model id for the LLM judge")
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--judge-temperature", type=float, default=0.6)
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="cap traces (debug)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = _load_corpus(args.corpus)
    truth = _load_truth(args.truth)
    traces = [t for t in traces if t["trace_id"] in truth]
    if args.limit:
        traces = traces[: args.limit]
    present = {t["trace_id"] for t in traces}
    truth = {tid: v for tid, v in truth.items() if tid in present}

    pos = sum(1 for v in truth.values() if v)
    baseline = pos / len(truth)
    print(f"\n{len(truth)} labelled traces  |  {pos} success / {len(truth) - pos} failure")
    print(f"majority-class baseline accuracy = {baseline:.3f}\n")

    all_scores: dict[str, dict[str, float | None]] = {}
    for name, fn in DETERMINISTIC.items():
        all_scores[name] = {t["trace_id"]: fn(t) for t in traces}

    share, is_modal = sibling_consensus(traces)
    all_scores["consensus_share"] = share
    all_scores["consensus_is_modal"] = is_modal

    judge_name: str | None = None
    if args.judge:
        cfg = JudgeConfig(args.judge, args.judge_samples, args.judge_temperature)
        judge_name = f"judge:{args.judge}"
        all_scores[judge_name] = run_judge(
            traces, cfg, args.out / "judge-cache", workers=args.judge_workers
        )
        # The combination the experiment is really testing: trust a unanimous
        # sibling jury outright, and spend the judge only on the runs it split on.
        combined: dict[str, float | None] = {}
        for tid in truth:
            s = share.get(tid)
            if s is not None and s >= 1.0:
                combined[tid] = 1.0
            elif s is not None and s <= 0.25:
                combined[tid] = 0.0
            else:
                combined[tid] = all_scores[judge_name].get(tid)
        all_scores["consensus>=4 else judge"] = combined

    metrics = [score_signal(name, scores, truth) for name, scores in all_scores.items()]

    # A couple of AND combos of the signals that separated best.
    ranked = sorted(
        (m for m in metrics if m.auc is not None),
        key=lambda m: m.auc,
        reverse=True,
    )
    combos: list[Metrics] = []
    if len(ranked) >= 2:
        top = [m.name for m in ranked[:3]]
        thr = {m.name: m.threshold for m in metrics}
        combos.append(and_combo(top[:2], all_scores, truth, thresholds=thr))
        if len(top) >= 3:
            combos.append(and_combo(top, all_scores, truth, thresholds=thr))

    header = (
        f"{'signal':<26} {'AUC':>6} {'acc@.5':>7} {'bal@.5':>7} "
        f"{'thr':>5} {'prec':>6} {'cover':>6} {'admit':>6}"
    )
    print(header)
    print("-" * len(header))
    baseline_row = (
        f"{'[baseline: all success]':<26} {'  n/a':>6} {baseline * 100:6.1f} "
        f"{'  50.0':>7} {' n/a':>5} {baseline * 100:5.1f} {'100.0':>6} {len(truth):>6}"
    )
    print(baseline_row)
    for m in sorted(metrics + combos, key=lambda m: (m.auc or 0), reverse=True):
        print(
            f"{m.name:<26} {_fmt(m.auc):>6} {_fmt(m.accuracy_at_half, True):>7} "
            f"{_fmt(m.balanced_at_half, True):>7} {m.threshold:5.2f} "
            f"{_fmt(m.precision, True):>6} {_fmt(m.coverage, True):>6} {m.admitted:>6}"
        )

    report = {
        "corpus": str(args.corpus),
        "truth": str(args.truth),
        "n": len(truth),
        "success": pos,
        "failure": len(truth) - pos,
        "baseline_accuracy": baseline,
        "signals": [m.__dict__ for m in metrics + combos],
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
