"""Label finished rollouts correct or incorrect, for training the outcome Jev.

The label comes from the gold answer (string match); the judge's input is
`outcome_state`, which has no gold. Training rows come from training-pool
questions only; the eval file holds held-out questions for the gate.

    python recipes/jev/scripts/outcome_labels.py train OUT.jsonl --per-class 400 FILE...
    python recipes/jev/scripts/outcome_labels.py eval OUT.jsonl FILE...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bandits_jev.outcome_judge import OPTIONS, QUESTION, outcome_state


def load_rollouts(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else (data.get("eval") or data["results"])


def split_for(query_id: str) -> str:
    bucket = int(hashlib.sha256(f"outcome-judge:{query_id}".encode()).hexdigest()[:8], 16) % 20
    return "train" if bucket < 16 else "dev" if bucket < 18 else "calibration" if bucket < 19 else "test"


def build(files: list[Path], questions: dict[str, str], *, per_class: int | None, splits: bool) -> list[dict]:
    rows, unique = [], set()
    for path in files:
        for r in load_rollouts(path):
            state = outcome_state(questions[r["query_id"]], r["events"], r.get("final"))
            if state in unique:
                continue
            unique.add(state)
            row = {
                "id": f"{path.stem}-{r['query_id']}-{r.get('sample', 0)}",
                "group_id": f"bcp-{r['query_id']}",
                "state": state, "question": QUESTION, "options": OPTIONS,
                "target": "correct" if r["correct"] else "incorrect",
                "label_source": "gold-answer-string-match", "source": "browsecomp-plus-rollouts",
            }
            if splits:
                row["split"] = split_for(r["query_id"])
            rows.append(row)
    if per_class is None:
        return rows
    chosen = []
    for target in OPTIONS:
        chosen += sorted((r for r in rows if r["target"] == target), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:per_class]
    return sorted(chosen, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("output", type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--tasks", type=Path, default=Path("work/step-rl/tasks.jsonl"))
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines()]
    questions = {t["query_id"]: t["query"] for t in tasks}
    split_of = {t["query_id"]: t["split"] for t in tasks}
    allowed = {"train": {"train_pool"}, "eval": {"held_out_eval", "held_out"}}[args.mode]
    rows = [
        r for r in build(args.files, questions, per_class=args.per_class, splits=args.mode == "train")
        if split_of[r["group_id"].removeprefix("bcp-")] in allowed
    ]
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["target"]] = counts.get(r["target"], 0) + 1
    print(len(rows), "rows", counts)


if __name__ == "__main__":
    main()
