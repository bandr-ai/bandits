#!/usr/bin/env python3
"""Prepare TRAIL as native bandits artifacts for the unchanged RLM family miner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trail_signal import load_trail  # noqa: E402

from bandits.analyze.analysis import analyze_corpus, save_analysis
from bandits.store import ArtifactStore, DerivedStore
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, UserTurn


def prepare(trail_dir: Path, split: str) -> tuple[TraceCorpus, dict[str, float]]:
    source = trail_dir.resolve()
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    traces: list[Trace] = []
    scores: dict[str, float] = {}
    for item in load_trail(source, split):
        if item.overall is None or not item.task:
            continue
        spans = tuple(
            Span(
                span_id=f"{item.trace_id}:span-{index}",
                kind=SpanKind.MODEL if step["kind"] == "llm" else SpanKind.TOOL,
                name=str(step["name"]),
                started_at=epoch + timedelta(seconds=index),
                ended_at=epoch + timedelta(seconds=index),
                status=SpanStatus.ERROR if step["is_error"] else SpanStatus.OK,
                output=step["text"],
                attributes={"synthetic_time": True, "source": "trail"},
            )
            for index, step in enumerate(item.steps)
        )
        if not spans:
            continue
        digest = hashlib.sha256(
            (
                source
                / "data"
                / ("GAIA" if split == "gaia" else "SWE Bench")
                / f"{item.trace_id}.json"
            ).read_bytes()
        ).hexdigest()
        traces.append(
            Trace(
                trace_id=item.trace_id,
                source=f"trail-{split}",
                source_digest=digest,
                task=item.task,
                user_turns=(UserTurn(text=item.task),),
                spans=spans,
            )
        )
        scores[item.trace_id] = float(item.overall)
    return TraceCorpus(source=f"trail-{split}", traces=tuple(traces)), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("gaia", "swe_bench"), default="gaia")
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--labels-out", type=Path, required=True)
    args = parser.parse_args()

    corpus, scores = prepare(args.trail_dir, args.split)
    artifact = ArtifactStore(args.project / ".bandits").write(
        corpus, source_path=str(args.trail_dir.resolve())
    )
    analysis = analyze_corpus(corpus)
    analysis_artifact = save_analysis(analysis, DerivedStore(args.project / ".bandits"))
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.labels_out.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    print(f"corpus_id:   {artifact.artifact_id}")
    print(f"analysis_id: {analysis_artifact.artifact_id}")
    print(f"traces:      {len(corpus.traces)}")
    print(f"labels:      {args.labels_out}")


if __name__ == "__main__":
    main()
