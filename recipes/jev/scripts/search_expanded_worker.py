"""Score the frozen search-judge pilot on a fresh rubric-v3 audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bandits_jev.hf_predictor import HFPredictor
from bandits_jev.importer import import_jsonl
from bandits_jev.scorer import score_dataset

MODEL = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
OLD_ADAPTER = "/runs/phase2-seed1-apb-20260927T134501Z/checkpoints/Qwen__Qwen3.5-4B-Base/step-250"


def score(adapter: str, dataset) -> dict:
    predictor = HFPredictor(MODEL, revision=REVISION, adapter_path=adapter)
    run = score_dataset(predictor, dataset.examples, max_prompt_tokens=8000)
    examples = {e.decision_id: e for e in dataset.examples}
    rows = []
    for result in run.results:
        example = examples[result.decision_id]
        target = max(example.target.probabilities, key=example.target.probabilities.get)
        rows.append({
            "id": example.lineage.record_id,
            "group_id": example.group_id,
            "target": target,
            "predicted": result.chosen_option_id,
            "probabilities": {s.option_id: s.probability for s in result.scores},
        })
    return {
        "adapter": adapter,
        "rows": len(dataset.examples), "scored": len(rows),
        "rejected": [r.model_dump() for r in run.rejections],
        "accuracy": sum(row["target"] == row["predicted"] for row in rows) / len(rows) if rows else None,
        "gold_counts": dict(Counter(row["target"] for row in rows)),
        "predicted_counts": dict(Counter(row["predicted"] for row in rows)),
        "predictions": rows,
    }


def main() -> None:
    pilot = json.loads(Path("/runs/search-judge-pilot/report.json").read_text())
    dataset = import_jsonl(Path("/work/search-audit-v3.jsonl").read_text(), source_file="search-audit-v3.jsonl")
    report = {
        "model": MODEL, "revision": REVISION,
        "majority_baseline": max(Counter(max(e.target.probabilities, key=e.target.probabilities.get) for e in dataset.examples).values()) / len(dataset.examples),
        "old_judge": score(OLD_ADAPTER, dataset),
        "new_judge": score(pilot["best_adapter"], dataset),
    }
    out = Path("/runs/search-judge-pilot/v3-audit.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "rows": len(dataset.examples), "majority_baseline": report["majority_baseline"],
        "old_accuracy": report["old_judge"]["accuracy"],
        "new_accuracy": report["new_judge"]["accuracy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
