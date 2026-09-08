#!/usr/bin/env python3
"""Build a small, outcome-blind OTLP corpus from Exgentic traces v2.

The source dataset stores one session per Parquet row and keeps benchmark
outcomes beside the spans.  Bandits must not see those outcomes during the
dogfood run, so this script writes them to a separate sealed file and emits
only the source spans as OTLP JSONL.

Run with an ephemeral dependency instead of adding PyArrow to Bandits:

    uv run --with pyarrow scripts/prepare_exgentic_dogfood.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO = "Exgentic/agent-llm-traces-v2"
REVISION = "main"
SHARDS = ("0000", "0001")
PROFILES = {
    "fast": {"success": 2, "unsuccessful": 2, "unfinished": 2, "error": 2},
    "core": {"success": 10, "unsuccessful": 10, "unfinished": 5, "error": 5},
}
OUTCOME_FIELDS = {
    "score",
    "success",
    "status",
    "steps",
    "action_count",
    "agent_cost",
    "benchmark_cost",
    "execution_time",
}


def _download(shard: str, cache: Path) -> Path:
    destination = cache / f"{shard}.parquet"
    if destination.exists():
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/train/{shard}.parquet"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _stable_key(row: dict[str, Any]) -> str:
    identity = f"{row['run_id']}\0{row['session_id']}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _select(rows: list[dict[str, Any]], quotas: dict[str, int]) -> list[dict[str, Any]]:
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_status[str(row["status"])].append(row)

    selected: list[dict[str, Any]] = []
    for status, quota in quotas.items():
        candidates = sorted(by_status[status], key=_stable_key)
        if len(candidates) < quota:
            raise RuntimeError(f"wanted {quota} {status!r} rows, found {len(candidates)}")
        selected.extend(candidates[:quota])
    return sorted(selected, key=_stable_key)


def _clean(value: Any) -> Any:
    """Drop Parquet nulls recursively without changing populated source values."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _blind_span(span: dict[str, Any]) -> dict[str, Any]:
    # Only the nested source span is copied. In particular, none of the
    # session-level outcome columns from its containing Parquet row are merged
    # into attributes. Span ``status`` remains: it describes whether that LLM
    # request errored, not whether the benchmark task succeeded.
    return _clean(span)


def prepare(output: Path, cache: Path, profile: str) -> None:
    paths = [_download(shard, cache) for shard in SHARDS]
    selected = _select(_rows(paths), PROFILES[profile])
    output.mkdir(parents=True, exist_ok=True)

    blind_path = output / "exgentic.blind.otlp.jsonl"
    truth_path = output / "exgentic.sealed-outcomes.json"
    manifest_path = output / "exgentic.selection.json"

    span_count = 0
    with blind_path.open("w", encoding="utf-8") as blind:
        for row in selected:
            for span in row["spans"]:
                blind.write(json.dumps(_blind_span(span), sort_keys=True) + "\n")
                span_count += 1

    truth = {
        "warning": "SEALED: do not expose this file to Bandits during analysis or drafting.",
        "source": REPO,
        "outcomes": [
            {
                "run_id": row["run_id"],
                "session_id": row["session_id"],
                "trace_ids": sorted(
                    {
                        span["trace_id"]
                        for span in row["spans"]
                        if isinstance(span.get("trace_id"), str) and span["trace_id"]
                    }
                ),
                "benchmark": row["benchmark"],
                "benchmark_subset": row["benchmark_subset"],
                **{field: row[field] for field in sorted(OUTCOME_FIELDS)},
            }
            for row in selected
        ],
    }
    truth_path.write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")

    manifest = {
        "source": REPO,
        "revision": REVISION,
        "source_shards": list(SHARDS),
        "selection": "deterministic sha256(run_id + NUL + session_id) within status strata",
        "profile": profile,
        "runs": len(selected),
        "spans": span_count,
        "blind_otlp": blind_path.name,
        "sealed_outcomes": truth_path.name,
        "outcomes_present_in_blind_otlp": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(selected)} blind runs / {span_count} spans to {blind_path}")
    print(f"sealed outcomes in {truth_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("work/exgentic-dogfood"))
    parser.add_argument("--cache", type=Path, default=Path("work/exgentic-dogfood/source-cache"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    args = parser.parse_args()
    prepare(args.output, args.cache, args.profile)


if __name__ == "__main__":
    main()
