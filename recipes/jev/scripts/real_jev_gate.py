"""Gate the hosted Jev as a verifier and step judge on held-out data, before RL.

Final answers: the base run's 480 held-out rollouts, judged from the full
trajectory, against ground-truth correctness. Steps: the 1,172 held-out step
states, against gold-evidence labels. Same AUC measures as the small judges
(outcome 0.92, step 0.786). Results are cached per item so a rerun resumes.

    JEV_API_KEY=... python recipes/jev/scripts/real_jev_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from analyze_rollouts import auc

from bandits_jev import outcome_judge, step_judge
from bandits_jev.jev_remote import RemoteJev, full_outcome_state, usage_tokens

WORK = Path(__file__).resolve().parents[3] / "work/step-rl"
PRICE_PER_MTOK = 0.042  # OpenRouter list price for jev-1.13 input tokens; output is free


def cached(path: Path, keys: list[str], items: list[tuple], jev: RemoteJev) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done[row["key"]] = row["p"]
    todo = [(k, it) for k, it in zip(keys, items, strict=True) if k not in done]
    for start in range(0, len(todo), 64):
        chunk = todo[start : start + 64]
        results = jev.choose_many([it for _, it in chunk])
        with path.open("a") as handle:
            for (key, _), probabilities in zip(chunk, results, strict=True):
                if probabilities is not None:
                    done[key] = probabilities
                    handle.write(json.dumps({"key": key, "p": probabilities}) + "\n")
        print(f"{path.name}: {len(done)}/{len(keys)}", flush=True)
    return done


def main() -> None:
    jev = RemoteJev(workers=32)
    tasks = {json.loads(line)["query_id"]: json.loads(line) for line in (WORK / "tasks.jsonl").read_text().splitlines()}

    base = json.loads((WORK / "rollouts/base.json").read_text())["results"]
    keys = [f"{r['query_id']}-{r['sample']}" for r in base]
    items = [
        (full_outcome_state(tasks[r["query_id"]]["query"], r["events"], r["final"]), outcome_judge.QUESTION, outcome_judge.OPTIONS)
        for r in base
    ]
    outcome = cached(WORK / "real-jev-gate-outcome.jsonl", keys, items, jev)
    truth = {k: r["correct"] for k, r in zip(keys, base, strict=True)}
    p_correct = [(outcome[k]["correct"], truth[k]) for k in keys if k in outcome]

    rows = [json.loads(line) for line in (WORK / "judge-v2-heldout.jsonl").read_text().splitlines()]
    step_items = [(r["state"], step_judge.QUESTION, step_judge.OPTIONS) for r in rows]
    steps = cached(WORK / "real-jev-gate-step.jsonl", [r["id"] for r in rows], step_items, jev)
    scored = [(steps[r["id"]]["positive"] - steps[r["id"]]["negative"], r["target"]) for r in rows if r["id"] in steps]

    tokens = usage_tokens(jev.usage)
    report = {
        "outcome": {
            "scored": len(p_correct), "of": len(keys),
            "auc_correct_vs_incorrect": auc([p for p, ok in p_correct if ok], [p for p, ok in p_correct if not ok]),
            "accuracy_at_0.5": sum((p > 0.5) == ok for p, ok in p_correct) / len(p_correct),
            "mean_p_correct_when_correct": sum(p for p, ok in p_correct if ok) / max(1, sum(ok for _, ok in p_correct)),
            "mean_p_correct_when_incorrect": sum(p for p, ok in p_correct if not ok) / max(1, sum(not ok for _, ok in p_correct)),
            "small_outcome_jev_auc": 0.921,
        },
        "step": {
            "scored": len(scored), "of": len(rows),
            "auc_evidence_vs_rest": auc([s for s, t in scored if t == "positive"], [s for s, t in scored if t != "positive"]),
            "small_step_jev_auc": 0.786,
        },
        "input_tokens_this_run": tokens,
        "estimated_cost_usd_at_openrouter_price": round(tokens / 1e6 * PRICE_PER_MTOK, 4),
    }
    (WORK / "real-jev-gate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
