#!/usr/bin/env python3
"""Test success signals on long AppWorld trajectories, sealed-truth scored.

tau2 was the wrong corpus: short retail dialogues, user turns and tool errors
stripped, success defined against a hidden database. AppWorld is the opposite --
20-70 step multi-app agent runs (pay a debt list over Venmo/Splitwise, update a
playlist from phone messages) where the tool responses in the trace actually
show what happened.

Signals tested, none of which need a per-task checker:

* structural: step count, terminal error status, "ran out of turns", whether the
  closing message claims completion or gives up;
* grounded decomposed judge: a model first lists the concrete sub-goals in the
  request, then for each one is asked whether the trace's tool calls and results
  show it was done -- score is the fraction of sub-goals with positive evidence,
  not a holistic verdict;
* reverse reconstruction: a model sees ONLY the state-changing calls the agent
  made and reconstructs what task they accomplish; a second model rates how well
  that reconstruction matches the real request. A run that did the wrong thing
  reconstructs to the wrong task.

Each is scored against the 30 sealed AppWorld outcomes (10 success / 20 failure,
so the majority-class baseline is 66.7%).

Usage:
    uv run python scripts/rederive_signal.py \
        --otlp  work/exgentic-dogfood/exgentic.blind.otlp.jsonl \
        --truth work/exgentic-dogfood/exgentic.sealed-outcomes.json \
        --out   work/rederive --model deepseek-v4-flash-0731
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from signal_experiment import _fireworks, score_signal  # noqa: E402

# --------------------------------------------------------------- otlp parsing


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
    return value if isinstance(value, list) else []


def _part_text(part: dict[str, Any]) -> str:
    # text parts carry "content"; tool_call_response parts carry "result".
    value = part.get("content")
    if value is None:
        value = part.get("result")
    if value is None:
        value = part.get("thinking")
    if isinstance(value, str):
        return value
    if value is not None:
        return json.dumps(value, default=str)
    return ""


class Trajectory:
    __slots__ = ("trace_id", "task", "calls", "final_text", "steps", "status")

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.task = ""
        self.calls: list[dict[str, Any]] = []  # {name, args, result, is_error}
        self.final_text = ""
        self.steps = 0
        self.status = "ok"


_TOKEN_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")


def _scrub(text: str) -> str:
    return _TOKEN_RE.sub("<token>", text)


def load_trajectories(otlp_path: Path) -> list[Trajectory]:
    spans_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in otlp_path.read_text().splitlines():
        if line.strip():
            span = json.loads(line)
            spans_by_trace[span["trace_id"]].append(span)

    out: list[Trajectory] = []
    for trace_id, spans in spans_by_trace.items():
        spans.sort(key=lambda s: s.get("start_time", ""))
        traj = Trajectory(trace_id)
        traj.steps = len(spans)

        # A chat span carries OTel status {"code": 2} and an error.type when that
        # LLM call itself failed. The run ending on one means the agent crashed
        # rather than finishing -- distinct from the benchmark judging the
        # finished result wrong.
        last = spans[-1]
        if (last.get("status") or {}).get("code") == 2 or "error.type" in last.get(
            "attributes", {}
        ):
            traj.status = "error"

        # Every span holds the conversation so far; the last one holds all of it.
        # Its output.messages is null on an error span, so fall back to the last
        # span that has one.
        richest = max(
            spans,
            key=lambda s: len(str(s.get("attributes", {}).get("gen_ai.input.messages", ""))),
        )
        attrs = richest.get("attributes", {})
        messages = _as_list(attrs.get("gen_ai.input.messages"))
        for span in reversed(spans):
            out_msgs = _as_list(span.get("attributes", {}).get("gen_ai.output.messages"))
            if out_msgs:
                messages = messages + out_msgs
                break

        by_id: dict[str, dict[str, Any]] = {}
        last_assistant_text = ""
        for index, message in enumerate(messages):
            for part in message.get("parts", []):
                kind = part.get("type")
                if kind == "text":
                    text = _part_text(part).strip()
                    if index == 0 and not traj.task:
                        traj.task = _real_task(_scrub(text))
                    elif message.get("role") == "assistant" and text:
                        last_assistant_text = text
                elif kind == "tool_call":
                    call = {
                        "id": part.get("id"),
                        "name": part.get("name", "?"),
                        "args": {
                            k: v
                            for k, v in (part.get("arguments") or {}).items()
                            if k != "access_token"
                        },
                        "result": None,
                        "is_error": False,
                    }
                    traj.calls.append(call)
                    if call["id"]:
                        by_id[call["id"]] = call
                elif kind == "tool_call_response":
                    result = _scrub(_part_text(part))[:800]
                    call = by_id.get(part.get("id")) or (traj.calls[-1] if traj.calls else None)
                    if call is not None:
                        call["result"] = result
                        low = result.lower()
                        # AppWorld tool errors come back as a result whose text
                        # starts "Error:"; also catch raised exceptions.
                        call["is_error"] = (
                            '": "error:' in low
                            or '": "error ' in low
                            or low.lstrip('[{" ').startswith("error")
                            or "traceback (most recent call last)" in low
                            or "exception:" in low
                        )
        traj.final_text = _scrub(last_assistant_text)[:1500]
        out.append(traj)
    return out


def _real_task(text: str) -> str:
    """Strip the AppWorld environment preamble down to the actual instruction."""
    for marker in ("Task from supervisor:", "Task:", "Your task is"):
        if marker in text:
            return text.split(marker, 1)[1].strip()[:2000]
    # Fall back: drop a leading ``Context: {...}`` JSON blob if present.
    if text.startswith("Context:"):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 :].strip()[:2000]
    return text[:2000]


def load_truth(path: Path) -> dict[str, bool]:
    raw = json.loads(path.read_text())
    out: dict[str, bool] = {}
    for outcome in raw["outcomes"]:
        for trace_id in outcome["trace_ids"]:
            out[trace_id] = bool(outcome["success"])
    return out


# ------------------------------------------------------------------- render


_MUT_VERBS = (
    "send", "create", "add", "update", "delete", "remove", "pay", "post", "set",
    "modify", "cancel", "make", "transfer", "write", "edit", "upload", "share",
    "book", "order", "schedule", "assign", "mark", "like", "follow", "comment",
)


def _is_mutation(name: str) -> bool:
    tail = name.split("__")[-1].lower()
    return any(tail.startswith(v) or f"_{v}" in tail for v in _MUT_VERBS)


def render_calls(traj: Trajectory, *, only_mutations: bool = False, limit: int = 60) -> str:
    lines = []
    for call in traj.calls:
        if only_mutations and not _is_mutation(call["name"]):
            continue
        args = json.dumps(call["args"], default=str)[:200]
        result = (call["result"] or "")[:180]
        flag = " ERROR" if call["is_error"] else ""
        lines.append(f"{call['name']}({args}) -> {result}{flag}")
    return "\n".join(lines[:limit]) or "(no calls)"


def render_full(traj: Trajectory) -> str:
    return (
        f"TASK:\n{traj.task}\n\n"
        f"AGENT TOOL CALLS ({len(traj.calls)} total, {traj.steps} steps, "
        f"status={traj.status}):\n{render_calls(traj)}\n\n"
        f"AGENT FINAL MESSAGE:\n{traj.final_text or '(none)'}"
    )


# ---------------------------------------------------------------- signals


def sig_not_errored(traj: Trajectory) -> float | None:
    return 0.0 if traj.status == "error" else 1.0


def sig_no_error_results(traj: Trajectory) -> float | None:
    if not traj.calls:
        return None
    errs = sum(1 for c in traj.calls if c["is_error"])
    return 1.0 / (1.0 + errs)


def sig_final_claims_complete(traj: Trajectory) -> float | None:
    text = traj.final_text.lower()
    if not text:
        return None
    done = any(
        p in text
        for p in (
            "completed", "successfully", "all done", "have processed", "have sent",
            "have created", "task is complete", "finished processing", "all set",
        )
    )
    gave_up = any(
        p in text
        for p in (
            "unable to", "could not", "couldn't", "cannot complete", "ran out",
            "i was not able", "failed to", "stuck", "need more", "unclear",
        )
    )
    if done and not gave_up:
        return 1.0
    if gave_up and not done:
        return 0.0
    return 0.5


def sig_made_mutations(traj: Trajectory) -> float | None:
    muts = sum(1 for c in traj.calls if _is_mutation(c["name"]) and not c["is_error"])
    return min(1.0, muts / 2.0) if traj.calls else None


DETERMINISTIC = {
    "not_errored": sig_not_errored,
    "no_error_results": sig_no_error_results,
    "final_claims_complete": sig_final_claims_complete,
    "made_mutations": sig_made_mutations,
}


_DECOMPOSE_CHECK = """You are checking whether an agent completed a multi-app task, sub-goal by sub-goal.

STEP 1. From the task, list the concrete separately-checkable sub-goals. Each is one atomic
outcome: a specific message sent, a specific record created or updated, a specific file written.
If the task says "for each X do Y", expand it into the individual Y's the evidence shows are
required (one per person, per song, per item).

STEP 2. For each sub-goal, look at the agent's tool calls and their results and decide whether
the evidence shows it was actually done -- the right state-changing call was made and its
result is not an error.

TASK:
{task}

AGENT TOOL CALLS (name(args) -> result):
{calls}

AGENT FINAL MESSAGE:
{final}

Reply with ONLY one JSON object:
{{"subgoals": [{{"goal": "...", "done": true|false, "evidence": "call id/result, or what is missing"}}],
  "fraction_done": 0.0-1.0}}"""

_RECONSTRUCT = """These are the state-changing actions an agent took, in order. \
Infer what task the user most likely asked for. Be specific about entities and amounts.

ACTIONS:
{muts}

Reply with ONLY a JSON object: {{"inferred_task": "..."}}"""

_MATCH = """Two descriptions of a task. Rate how well the second matches the first.

ACTUAL REQUEST:
{task}

RECONSTRUCTED FROM THE AGENT'S ACTIONS:
{inferred}

Reply ONLY one JSON object: {{"match": 0.0-1.0, "why": "one sentence"}}. \
1.0 = the agent's actions accomplish exactly the actual request; \
0.0 = the actions are for a different task or miss most of it."""


def _json_obj(reply: str) -> dict[str, Any] | None:
    a, b = reply.find("{"), reply.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        return json.loads(reply[a : b + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _json_arr(reply: str) -> list[Any]:
    a, b = reply.find("["), reply.rfind("]")
    if a < 0 or b < a:
        return []
    try:
        got = json.loads(reply[a : b + 1])
        return got if isinstance(got, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def decomposed_judge(traj: Trajectory, model: str) -> tuple[float | None, dict[str, Any]]:
    prompt = _DECOMPOSE_CHECK.format(
        task=traj.task,
        calls=render_calls(traj, limit=45),
        final=traj.final_text or "(none)",
    )
    obj = _json_obj(_fireworks(model, prompt, 0.0, max_tokens=2600))
    if not obj or not isinstance(obj.get("subgoals"), list) or not obj["subgoals"]:
        return None, {"error": "no sub-goals parsed", "raw": str(obj)[:300]}
    subgoals = obj["subgoals"]
    done = sum(1 for s in subgoals if isinstance(s, dict) and s.get("done") is True)
    score = done / len(subgoals)
    return score, {
        "n_subgoals": len(subgoals),
        "done": done,
        "model_fraction": obj.get("fraction_done"),
        "subgoals": subgoals[:20],
    }


def reverse_reconstruction(traj: Trajectory, model: str) -> tuple[float | None, dict[str, Any]]:
    muts = render_calls(traj, only_mutations=True, limit=40)
    if muts == "(no calls)":
        return 0.0, {"note": "no state-changing actions at all"}
    inferred = _json_obj(
        _fireworks(model, _RECONSTRUCT.format(muts=muts), 0.2, max_tokens=1600)
    )
    inferred_task = (inferred or {}).get("inferred_task", "")
    if not inferred_task:
        return None, {"error": "no reconstruction"}
    match = _json_obj(
        _fireworks(
            model, _MATCH.format(task=traj.task, inferred=inferred_task), 0.0, max_tokens=1200
        )
    )
    if not match or not isinstance(match.get("match"), (int, float)):
        return None, {"inferred": inferred_task, "error": "no match score"}
    return float(match["match"]), {"inferred": inferred_task, "why": match.get("why")}


# -------------------------------------------------------------------- main


def _cached(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {r["trace_id"]: r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}
    return {}


def _run_model_signal(name, fn, trajs, model, out_dir, workers):
    cache_path = out_dir / f"{name}.jsonl"
    cache = _cached(cache_path)
    todo = [t for t in trajs if t.trace_id not in cache]
    print(f"{name}: {len(cache)} cached, {len(todo)} to run", file=sys.stderr)

    def one(traj):
        try:
            score, detail = fn(traj, model)
        except Exception as exc:  # noqa: BLE001
            score, detail = None, {"exception": str(exc)}
        return {"trace_id": traj.trace_id, "score": score, "detail": detail}

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as sink:
            for row in pool.map(one, todo):
                cache[row["trace_id"]] = row
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    return {tid: r["score"] for tid, r in cache.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--otlp", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-model", action="store_true", help="deterministic signals only")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trajs = load_trajectories(args.otlp)
    truth = load_truth(args.truth)
    trajs = [t for t in trajs if t.trace_id in truth]
    pos = sum(1 for t in trajs if truth[t.trace_id])
    baseline = max(pos, len(trajs) - pos) / len(trajs)
    print(f"\n{len(trajs)} trajectories | {pos} success / {len(trajs) - pos} failure")
    print(f"majority-class baseline = {baseline:.3f}")
    print(f"steps: {sorted(t.steps for t in trajs)}\n")

    all_scores: dict[str, dict[str, float | None]] = {}
    for name, fn in DETERMINISTIC.items():
        all_scores[name] = {t.trace_id: fn(t) for t in trajs}

    if not args.skip_model:
        model = (
            args.model
            if args.model.startswith("accounts/")
            else f"accounts/fireworks/models/{args.model}"
        )
        all_scores["decomposed_judge"] = _run_model_signal(
            "decomposed_judge", decomposed_judge, trajs, model, args.out, args.workers
        )
        all_scores["reverse_reconstruction"] = _run_model_signal(
            "reverse_reconstruction", reverse_reconstruction, trajs, model, args.out, args.workers
        )

    truth_bt = {t.trace_id: truth[t.trace_id] for t in trajs}
    metrics = [score_signal(n, s, truth_bt) for n, s in all_scores.items()]

    header = f"{'signal':<26} {'AUC':>6} {'acc@.5':>7} {'bal@.5':>7} {'thr':>5} {'prec':>6} {'cover':>6}"
    print(header)
    print("-" * len(header))
    print(f"{'[baseline]':<26} {'  n/a':>6} {baseline * 100:6.1f}  {'  50.0':>6}   n/a   n/a    n/a")
    for m in sorted(metrics, key=lambda m: (m.auc or 0), reverse=True):
        auc = "  n/a" if m.auc is None else f"{m.auc:.3f}"
        acc = "  n/a" if m.accuracy_at_half is None else f"{m.accuracy_at_half * 100:5.1f}"
        bal = "  n/a" if m.balanced_at_half is None else f"{m.balanced_at_half * 100:5.1f}"
        prec = "  n/a" if m.precision is None else f"{m.precision * 100:5.1f}"
        print(
            f"{m.name:<26} {auc:>6} {acc:>7} {bal:>7} {m.threshold:5.2f} {prec:>6} "
            f"{m.coverage * 100:5.1f}"
        )

    (args.out / "report.json").write_text(
        json.dumps(
            {"n": len(trajs), "baseline": baseline, "signals": [m.__dict__ for m in metrics]},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
