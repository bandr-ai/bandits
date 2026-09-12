#!/usr/bin/env python3
"""Import a family-local handoff bundle into a runnable bandits project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bandits.analyze.analysis import analyze_corpus, compute_analysis_id, save_analysis
from bandits.analyze.models import TaskSet
from bandits.analyze.tasksets import save_task_set
from bandits.ingest.chat_json import _convert_conversation
from bandits.store import ArtifactStore, DerivedStore, compute_artifact_id
from bandits.traces import TraceCorpus, TraceIssue


def import_bundle(bundle: Path, project: Path) -> tuple[str, Path]:
    source_task_set = TaskSet.model_validate_json((bundle / "taskset.json").read_bytes())
    traces = []
    issues: list[TraceIssue] = []
    labels: dict[str, dict[str, bool]] = {}

    for family_dir in sorted(path for path in bundle.iterdir() if path.is_dir()):
        trajectories = json.loads((family_dir / "trajectories.json").read_text())
        for item in trajectories:
            wrapper = item["trajectory"]
            raw = json.dumps(wrapper, sort_keys=True, separators=(",", ":")).encode()
            trace = _convert_conversation(
                wrapper["messages"],
                trace_id=item["trace_id"],
                source_digest=hashlib.sha256(raw).hexdigest(),
                lineage_id=wrapper.get("session_id"),
                wrapper=wrapper,
                location=str(family_dir / "trajectories.json"),
                issues=issues,
            )
            if trace is not None:
                traces.append(trace)

        label_set = json.loads((family_dir / "labels.json").read_text())
        for label in label_set["labels"]:
            if label["verdict"] in ("success", "failure"):
                labels[label["trace_id"]] = {"success": label["verdict"] == "success"}

    corpus = TraceCorpus(
        source="verifier-family-bundle", traces=tuple(traces), issues=tuple(issues)
    )
    corpus_id = compute_artifact_id(corpus)
    ArtifactStore(project / ".bandits").write(corpus, source_path=str(bundle.resolve()))
    analysis = analyze_corpus(corpus)
    analysis_id = compute_analysis_id(analysis)
    derived = DerivedStore(project / ".bandits")
    save_analysis(analysis, derived)
    task_set = source_task_set.replace(corpus_id=corpus_id, analysis_id=analysis_id)
    task_set_artifact = save_task_set(task_set, derived)

    labels_path = project / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n")
    return task_set_artifact.artifact_id, labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    task_set_id, labels = import_bundle(args.bundle, args.project)
    print(f"task_set_id: {task_set_id}")
    print(f"labels:      {labels}")


if __name__ == "__main__":
    main()
