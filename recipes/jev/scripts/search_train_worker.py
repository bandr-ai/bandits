"""Train and gate a search-specific Jev pilot inside the Modal GPU image."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from bandits_jev.hf_predictor import HFPredictor
from bandits_jev.importer import import_jsonl
from bandits_jev.scorer import score_dataset

MODEL = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
ROOT = Path("/runs/search-judge-pilot")
PYTHON_ENV = Path("/repo/recipes/jev/.venv/bin")


def invoke(*args: str) -> str:
    result = subprocess.run(
        [str(PYTHON_ENV / "jev"), *args],
        cwd="/repo/recipes/jev", text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"jev {' '.join(args[:2])} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def score_file(predictor: HFPredictor, path: Path) -> dict:
    dataset = import_jsonl(path.read_text(), source_file=path.name)
    run = score_dataset(predictor, dataset.examples, max_prompt_tokens=8000)
    by_id = {e.decision_id: e for e in dataset.examples}
    predictions = []
    for result in run.results:
        row = by_id[result.decision_id]
        target = max(row.target.probabilities, key=row.target.probabilities.get)
        predictions.append({
            "id": row.lineage.record_id,
            "target": target,
            "predicted": result.chosen_option_id,
            "probabilities": {s.option_id: s.probability for s in result.scores},
        })
    return {
        "rows": len(dataset.examples), "scored": len(predictions),
        "rejected": [r.model_dump() for r in run.rejections],
        "accuracy": sum(p["target"] == p["predicted"] for p in predictions) / len(predictions) if predictions else None,
        "gold_counts": dict(Counter(p["target"] for p in predictions)),
        "prediction_counts": dict(Counter(p["predicted"] for p in predictions)),
        "predictions": predictions,
    }


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    project = ROOT / "project"
    checkpoint_dir = ROOT / "checkpoints"
    imported = invoke(
        "import", "/work/search-train.jsonl", "--project", str(project),
        "--output", str(ROOT / "import-review"),
    )
    dataset_id = re.search(r"decision_dataset_id:\s+(\S+)", imported)
    if dataset_id is None:
        raise RuntimeError(f"could not find imported dataset id:\n{imported}")
    trained = invoke(
        "train", dataset_id.group(1), "--model", MODEL, "--revision", REVISION,
        "--seed", "1", "--epochs", "2", "--effective-batch", "4",
        "--eval-every-steps", "25", "--max-prompt-tokens", "4500",
        "--checkpoint-dir", str(checkpoint_dir), "--project", str(project),
    )
    best_adapter = re.search(r"best_adapter_path:\s+(\S+)", trained)
    if best_adapter is None:
        raise RuntimeError(f"could not find best adapter path:\n{trained}")
    predictor = HFPredictor(MODEL, revision=REVISION, adapter_path=best_adapter.group(1))
    report = {
        "model": MODEL, "revision": REVISION,
        "dataset_id": dataset_id.group(1), "best_adapter": best_adapter.group(1),
        "import_log": imported, "train_log": trained,
        "external_audit": score_file(predictor, Path("/work/search-audit.jsonl")),
        "attack_probes": score_file(predictor, Path("/work/search-probes.jsonl")),
    }
    (ROOT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({
        "dataset_id": report["dataset_id"], "best_adapter": report["best_adapter"],
        "external_audit_accuracy": report["external_audit"]["accuracy"],
        "attack_probe_accuracy": report["attack_probes"]["accuracy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
