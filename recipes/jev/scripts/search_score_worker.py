"""Run inside the Jev virtual environment on a Modal GPU."""

from __future__ import annotations

import json
from pathlib import Path

from bandits_jev.hf_predictor import HFPredictor
from bandits_jev.importer import import_jsonl
from bandits_jev.scorer import score_dataset

ADAPTER = "/runs/phase2-seed1-apb-20260927T134501Z/checkpoints/Qwen__Qwen3.5-4B-Base/step-250"
MODEL = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def main() -> None:
    source = Path("/work/search-labeled.jsonl")
    dataset = import_jsonl(source.read_text(), source_file=source.name)
    predictor = HFPredictor(MODEL, revision=REVISION, adapter_path=ADAPTER)
    run = score_dataset(predictor, dataset.examples, max_prompt_tokens=8000)
    by_id = {row.decision_id: row for row in dataset.examples}
    result = []
    for row in run.results:
        example = by_id[row.decision_id]
        gold = max(example.target.probabilities, key=example.target.probabilities.get)
        result.append({
            "id": example.lineage.record_id,
            "split": example.split,
            "target": gold,
            "predicted": row.chosen_option_id,
            "probabilities": {score.option_id: score.probability for score in row.scores},
        })
    report = {
        "model": MODEL,
        "revision": REVISION,
        "adapter": ADAPTER,
        "rows": len(dataset.examples),
        "scored": len(result),
        "rejected": [r.model_dump() for r in run.rejections],
        "accuracy": sum(r["target"] == r["predicted"] for r in result) / len(result) if result else None,
        "predictions": result,
    }
    Path("/runs/search-reward-gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
