"""Train Jev v2 on gold-evidence step labels and compare it with v1 on held-out steps (Modal GPU)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from analyze_rollouts import auc
from search_train_worker import MODEL, REVISION, invoke, score_file

from bandits_jev.hf_predictor import HFPredictor

ROOT = Path("/runs/search-judge-v2")
OLD_ADAPTER = "/runs/search-judge-pilot/checkpoints/step-50"


def heldout_report(predictor: HFPredictor, path: Path) -> dict:
    scored = score_file(predictor, path)
    scores = [p["probabilities"]["positive"] - p["probabilities"]["negative"] for p in scored["predictions"]]
    positive = [s for s, p in zip(scores, scored["predictions"], strict=True) if p["target"] == "positive"]
    other = [s for s, p in zip(scores, scored["predictions"], strict=True) if p["target"] != "positive"]
    scored["auc_positive_vs_rest"] = auc(positive, other)
    scored["share_scored_above_0.25_among_non_positive"] = sum(s > 0.25 for s in other) / len(other)
    return scored


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    project = ROOT / "project"
    imported = invoke("import", "/work/judge-v2-train.jsonl", "--project", str(project), "--output", str(ROOT / "import-review"))
    dataset_id = re.search(r"decision_dataset_id:\s+(\S+)", imported)
    if dataset_id is None:
        raise RuntimeError(f"could not find imported dataset id:\n{imported}")
    trained = invoke(
        "train", dataset_id.group(1), "--model", MODEL, "--revision", REVISION,
        "--seed", "1", "--epochs", "2", "--effective-batch", "4",
        "--eval-every-steps", "25", "--max-prompt-tokens", "4500",
        "--checkpoint-dir", str(ROOT / "checkpoints"), "--project", str(project),
    )
    best = re.search(r"best_adapter_path:\s+(\S+)", trained)
    if best is None:
        raise RuntimeError(f"could not find best adapter path:\n{trained}")
    heldout = Path("/work/judge-v2-heldout.jsonl")
    report = {"dataset_id": dataset_id.group(1), "best_adapter": best.group(1), "import_log": imported, "train_log": trained}
    for name, adapter in (("v1", OLD_ADAPTER), ("v2", best.group(1))):
        result = heldout_report(HFPredictor(MODEL, revision=REVISION, adapter_path=adapter), heldout)
        report[name] = result
        print(json.dumps({name: {k: result[k] for k in ("rows", "scored", "accuracy", "auc_positive_vs_rest",
                                                         "share_scored_above_0.25_among_non_positive")}}), flush=True)
    (ROOT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
