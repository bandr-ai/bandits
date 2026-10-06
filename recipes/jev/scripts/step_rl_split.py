"""Split the 830 BrowseComp-Plus questions into a training pool and a held-out set.

Our retriever only covers the BrowseComp-Plus corpus, so policy training
questions must come from the same 830. The held-out ids are never used for
training, step-judge data, or tuning; only the training pool is. The split is a
hash of the query id, so it is reproducible and cannot be tuned by hand.

    python recipes/jev/scripts/step_rl_split.py work/step-rl/browsecomp-plus-decrypted.jsonl work/step-rl/split.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

QUERY_ID = re.compile(r'\{"query_id": "([^"]+)"')
HELD_OUT_SHARE = 0.5
HELD_OUT_EVAL = 60
"""The held-out evaluation set used in the scoped experiment: the first ids by hash."""


def bucket(query_id: str) -> str:
    return hashlib.sha256(f"step-rl-split:{query_id}".encode()).hexdigest()


def split(query_ids: list[str]) -> dict:
    ordered = sorted(query_ids, key=bucket)
    cut = round(len(ordered) * HELD_OUT_SHARE)
    held_out, train = ordered[:cut], ordered[cut:]
    return {
        "rule": f"sort by sha256('step-rl-split:'+id); first {HELD_OUT_SHARE:.0%} held out, rest training pool",
        "train_pool": train,
        "held_out": held_out,
        "held_out_eval": held_out[:HELD_OUT_EVAL],
    }


def main(source: Path, output: Path) -> None:
    # The file holds full document text (about 2 GB), so read one line at a time.
    with source.open() as handle:
        ids = [QUERY_ID.match(line).group(1) for line in handle]
    result = split(ids)
    assert not set(result["train_pool"]) & set(result["held_out"])
    output.write_text(json.dumps(result, indent=1))
    print({k: len(v) for k, v in result.items() if isinstance(v, list)})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
