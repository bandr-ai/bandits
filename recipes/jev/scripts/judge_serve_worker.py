"""Long-running judge scorer, run by modal_jev_judge.py inside the recipe venv.

Reads one JSON request per line on stdin and answers each with one line on
stdout starting with RESULT_PREFIX. The first answer is the provenance, sent
once the model is loaded. Anything else the libraries print is not a reply.
"""

from __future__ import annotations

import json
import sys

from bandits_jev.hf_predictor import HFPredictor
from bandits_jev.prompt import PROMPT_VERSION
from bandits_jev.scorer import adapter_digest
from bandits_jev.step_judge import judge_step

MODEL = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
ADAPTER = "/runs/search-judge-pilot/checkpoints/step-50"
RESULT_PREFIX = "@@RESULT@@ "


def reply(payload: dict) -> None:
    sys.stdout.write(RESULT_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def score(predictor: HFPredictor, steps: list[dict]) -> list[dict]:
    results = []
    for step in steps:
        previous = [tuple(p) for p in step["previous"]]
        judged = judge_step(
            predictor, step["question"], previous, step["tool"], step["action"], step["observation"]
        )
        results.append(
            {
                "probabilities": judged.probabilities,
                "score": judged.score,
                "reason": judged.reason,
                "prompt_tokens": judged.prompt_tokens,
            }
        )
    return results


def main() -> None:
    predictor = HFPredictor(MODEL, revision=REVISION, adapter_path=ADAPTER)
    provenance = {
        "model": MODEL,
        "revision": REVISION,
        "adapter": ADAPTER,
        "adapter_digest": adapter_digest(ADAPTER),
        "prompt_version": PROMPT_VERSION,
    }
    reply({"ready": provenance})
    for line in sys.stdin:
        reply({"results": score(predictor, json.loads(line)["steps"])})


if __name__ == "__main__":
    main()
