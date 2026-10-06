"""Label logged search steps from ground truth, for training a stricter Jev.

positive: the step surfaced a gold evidence document new to the rollout, or
          opened one.
negative: the step repeated an earlier action, or the tool rejected it
          ("Error: ..."): padding and misuse.
neutral:  everything else, including plausible searches that found nothing.

Gold evidence is used only to make the label. The judge's input is the same
state the current judge reads (question, two previous steps, action, real
reply), so a judge trained on these rows learns to recognise evidence-finding
steps from the trace alone and serves through the unchanged reward code.

    python recipes/jev/scripts/gold_step_labels.py train OUT.jsonl --per-class 400 FILE...
    python recipes/jev/scripts/gold_step_labels.py eval OUT.jsonl FILE...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bandits_jev.step_judge import OPTIONS, QUESTION, step_state
from bandits_jev.step_shaping import action_key

LABEL_SOURCE = "gold-evidence-v1"


def load_rollouts(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("eval") or data["results"]


def step_rows(rollout: dict, question: str, tag: str) -> list[dict]:
    rows, history, seen = [], [], set()
    for index, event in enumerate(rollout["events"]):
        key = action_key(event["tool"], event["action"])
        if key in seen or event["observation"].startswith("Error:"):
            target = "negative"
        elif event.get("new_evidence_seen") or event.get("evidence_opened"):
            target = "positive"
        else:
            target = "neutral"
        seen.add(key)
        rows.append(
            {
                "id": f"{tag}-{rollout['query_id']}-{rollout.get('sample', 0)}-{index}",
                "group_id": f"bcp-{rollout['query_id']}",
                "state": step_state(question, history, event["tool"], event["action"], event["observation"]),
                "question": QUESTION,
                "options": OPTIONS,
                "target": target,
                "label_source": LABEL_SOURCE,
                "source": "browsecomp-plus-rollouts",
            }
        )
        history.append((event["tool"], event["action"], event["observation"]))
    return rows


def split_for(query_id: str) -> str:
    bucket = int(hashlib.sha256(f"judge-v2:{query_id}".encode()).hexdigest()[:8], 16) % 20
    return "train" if bucket < 16 else "dev" if bucket < 18 else "calibration" if bucket < 19 else "test"


def build(files: list[Path], questions: dict[str, str], *, per_class: int | None, splits: bool) -> list[dict]:
    rows, unique = [], set()
    for path in files:
        for rollout in load_rollouts(path):
            for row in step_rows(rollout, questions[rollout["query_id"]], path.stem):
                if row["state"] in unique:
                    continue
                unique.add(row["state"])
                if splits:
                    row["split"] = split_for(rollout["query_id"])
                rows.append(row)
    if per_class is None:
        return rows

    def order(row: dict) -> str:
        return hashlib.sha256(row["id"].encode()).hexdigest()

    chosen = []
    for target in OPTIONS:
        chosen += sorted((r for r in rows if r["target"] == target), key=order)[:per_class]
    return sorted(chosen, key=order)


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
    allowed = {"train": {"train_pool"}, "eval": {"held_out_eval", "held_out"}}[args.mode]
    split_of = {t["query_id"]: t["split"] for t in tasks}
    rows = build(args.files, questions, per_class=args.per_class, splits=args.mode == "train")
    rows = [r for r in rows if split_of[r["group_id"].removeprefix("bcp-")] in allowed]
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["target"]] = counts.get(r["target"], 0) + 1
    print(len(rows), "rows", counts)


if __name__ == "__main__":
    main()
