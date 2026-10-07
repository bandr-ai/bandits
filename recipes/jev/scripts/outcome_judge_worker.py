"""Train the outcome Jev and gate it on held-out final answers (Modal GPU)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from analyze_rollouts import auc
from search_train_worker import MODEL, REVISION, invoke, score_file

from bandits_jev.hf_predictor import HFPredictor

ROOT = Path("/runs/outcome-judge-v1")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    project = ROOT / "project"
    imported = invoke("import", "/work/outcome-train.jsonl", "--project", str(project), "--output", str(ROOT / "import-review"))
    dataset_id = re.search(r"decision_dataset_id:\s+(\S+)", imported)
    if dataset_id is None:
        raise RuntimeError(f"could not find imported dataset id:\n{imported}")
    trained = invoke(
        "train", dataset_id.group(1), "--model", MODEL, "--revision", REVISION,
        "--seed", "1", "--epochs", "2", "--effective-batch", "4",
        "--eval-every-steps", "25", "--max-prompt-tokens", "6000",
        "--checkpoint-dir", str(ROOT / "checkpoints"), "--project", str(project),
    )
    best = re.search(r"best_adapter_path:\s+(\S+)", trained)
    if best is None:
        raise RuntimeError(f"could not find best adapter path:\n{trained}")
    scored = score_file(HFPredictor(MODEL, revision=REVISION, adapter_path=best.group(1)), Path("/work/outcome-heldout.jsonl"))
    p_correct = [(p["probabilities"]["correct"], p["target"] == "correct") for p in scored["predictions"]]
    gate = {
        "rows": scored["rows"], "scored": scored["scored"], "accuracy": scored["accuracy"],
        "auc_correct_vs_incorrect": auc([p for p, ok in p_correct if ok], [p for p, ok in p_correct if not ok]),
        "mean_p_correct_when_correct": sum(p for p, ok in p_correct if ok) / max(1, sum(ok for _, ok in p_correct)),
        "mean_p_correct_when_incorrect": sum(p for p, ok in p_correct if not ok) / max(1, sum(not ok for _, ok in p_correct)),
    }
    print(json.dumps({"gate": gate}), flush=True)
    report = {"dataset_id": dataset_id.group(1), "best_adapter": best.group(1), "import_log": imported,
              "train_log": trained, "gate": gate, "heldout": scored}
    (ROOT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
