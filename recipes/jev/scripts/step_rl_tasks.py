"""Compact task file for the step-RL experiment: question, gold answer, gold evidence ids.

The decrypted benchmark file is about 2 GB because it embeds full document
text, so this reads it one line at a time. The output holds answers and so
stays under the ignored work/ directory; it is evaluation data and the answers
must never reach a judge prompt or the policy.

    python recipes/jev/scripts/step_rl_tasks.py work/step-rl/browsecomp-plus-decrypted.jsonl work/step-rl/split.json work/step-rl/tasks.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(source: Path, split_path: Path, output: Path) -> None:
    split = json.loads(split_path.read_text())
    group = {}
    for name in ("train_pool", "held_out"):
        group.update({qid: name for qid in split[name]})
    eval_ids = set(split["held_out_eval"])
    count = 0
    with source.open() as handle, output.open("w") as out:
        for line in handle:
            row = json.loads(line)
            qid = row["query_id"]
            out.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": row["query"],
                        "answer": row["answer"],
                        "evidence_ids": sorted({d["docid"] for d in row["evidence_docs"]}),
                        "split": "held_out_eval" if qid in eval_ids else group[qid],
                    }
                )
                + "\n"
            )
            count += 1
    print(count, "tasks")


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:4]))
