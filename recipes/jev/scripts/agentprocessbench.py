#!/usr/bin/env python3
"""AgentProcessBench (human step labels) as a `jev import` JSONL.

AgentProcessBench (arXiv 2603.14465; MIT) labels every assistant step of
1,000 agent trajectories over HotpotQA, GAIA, BFCL and tau2-bench as +1, 0
or -1. Each labeled step becomes one decision row:

- state: the task, the agent's instructions (clipped), the last eight turns
  before the step, the step itself and what came back until the agent's next
  turn. The history is there because a label marks a step that builds on an
  earlier uncorrected error as -1, so the step alone is not enough.
- target: +1 -> success, 0 -> unclear, -1 -> failure.
- split: by task, 60/10/10/20 train/dev/calibration/test from a hash of the
  task, so it is fixed. A task is its text: in tau2-bench, twelve query
  indices (34, 39-49) are one task, and splitting by index put it in train,
  dev and test. `--group-by index` reproduces that split, which the
  2026-09-27 locked results used (they report the test set without it).

    uv run python scripts/agentprocessbench.py --out apb.jsonl
    JEV_IMPORT=apb.jsonl uvx modal run scripts/modal_run.py --phase 2 --tag apb ...

The four source files are downloaded from Hugging Face into --src unless
they are already there.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import urllib.request
from pathlib import Path

SOURCES = ("gaia_dev", "tau2", "bfcl", "hotpotqa")
URL = "https://huggingface.co/datasets/LulaCola/AgentProcessBench/resolve/main/{source}/test.jsonl"
LABEL = {1: "success", 0: "unclear", -1: "failure"}
QUESTION = "Judge this agent step, given the task and what happened so far."
OPTIONS = {
    "success": "The step is correct and clearly advances the task.",
    "unclear": "The step is reasonable but has little or no effect on progress.",
    "failure": "The step is wrong or counterproductive, or builds on an earlier error the agent has not corrected.",
}
HISTORY_TURNS = 8


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]}…[{len(text) - limit} chars omitted]…{text[-(limit - head):]}"


def render(message: dict, limit: int) -> str:
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        parts.append(f"→ {fn.get('name')}({fn.get('arguments')})")
    return clip("\n".join(p for p in parts if p), limit)


def split_of(group: str) -> str:
    x = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if x < 0.6 else "dev" if x < 0.7 else "calibration" if x < 0.8 else "test"


def task_group(trajectory: dict, source: str, group_by: str) -> str:
    if group_by == "index":
        return f"{source}:{trajectory['query_index']}"
    text = re.sub(r"\s+", " ", trajectory["question"]).strip().lower()
    return f"{source}:task-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def rows_for(trajectory: dict, source: str, group_by: str = "task") -> list[dict]:
    messages = trajectory["messages"]
    group = task_group(trajectory, source, group_by)
    system = "\n".join(m["content"] or "" for m in messages if m["role"] == "system")
    rows = []
    for index_text, label in trajectory["step_labels"].items():
        i = int(index_text)
        reaction = []
        for m in messages[i + 1 :]:
            if m["role"] == "assistant":
                break
            name = f":{m['name']}" if m.get("name") else ""
            reaction.append(f"[{m['role']}{name}] {render(m, 600)}")
        history = [f"[{m['role']}] {render(m, 300)}" for m in messages[:i] if m["role"] != "system"]
        state = "\n\n".join(
            [
                f"Task:\n{clip(trajectory['question'], 2000)}",
                f"Agent instructions (clipped):\n{clip(system, 1500)}",
                "Recent history:\n" + ("\n".join(history[-HISTORY_TURNS:]) if history else "(none)"),
                f"Current step:\n{render(messages[i], 1600)}",
                "What came back:\n" + (clip("\n".join(reaction), 1200) if reaction else "(nothing)"),
            ]
        )
        rows.append(
            {
                "id": f"apb:{source}:{trajectory['total_index']}:{i}",
                "group_id": group,
                "split": split_of(group),
                "state": state,
                "question": QUESTION,
                "options": OPTIONS,
                "target": LABEL[label],
                "label_source": "human",
                "source": f"AgentProcessBench/{source}",
                "license": "MIT",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("agentprocessbench"), help="Where the source files are or go.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--group-by",
        choices=("task", "index"),
        default="task",
        help="task: one group per distinct task text (default). index: per query index, as the 2026-09-27 run.",
    )
    args = parser.parse_args()

    args.src.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in SOURCES:
        path = args.src / f"{source}.jsonl"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(source=source), path)
        for line in path.read_text(encoding="utf-8").splitlines():
            rows += rows_for(json.loads(line), source, args.group_by)

    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = collections.Counter(r["split"] for r in rows)
    groups = collections.defaultdict(set)
    for r in rows:
        groups[r["split"]].add(r["group_id"])
    print(f"{len(rows)} rows -> {args.out}")
    for split in ("train", "dev", "calibration", "test"):
        print(f"  {split:12s} {counts[split]:5d} steps  {len(groups[split]):3d} tasks")


if __name__ == "__main__":
    main()
