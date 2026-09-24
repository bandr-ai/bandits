#!/usr/bin/env python3
"""Convert the Decision Models demo-dataset candidates (issue #67) into
``jev import`` JSONL with fixed train/dev/calibration/test splits.

Each candidate keeps its publisher's own held-out split as the locked test
set whenever one exists. Where a split is missing it is carved from a stable
hash of the row's content (never its position), so rerunning the conversion
can never move a row between splits:

    case_hold   train -> train | calibration (10%)   validation -> dev    test -> test
    twitter     train -> train | calibration (10%)   validation -> dev | test (50/50)
    jailbreak   train -> train | calibration (10%)   test       -> dev | test (50/50)
    injections  train -> train | calibration (10%)   test       -> dev | test (50/50)
    vitaminc    test, real revisions only             -> dev | test (50/50 by case_id)

VitaminC is the external check set, never trained on. Its dev half is the
go/no-go guardrail (#68); its test half is only for the final report (#69),
so the set the go/no-go looked at is never the one the launch claim rests on.

Rows carry no explicit id, so the importer identifies them by content: an
exact duplicate that lands in two different splits is quarantined on both
sides, never kept on either, so it can never sit across the train/test line.

Usage (pyarrow is only needed here, not by Bandits):

    uv run --with pyarrow python scripts/decision_candidates.py case_hold --out work/candidates
    uv run jev import work/candidates/case_hold.jsonl --source hf:coastalcph/lex_glue
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

CONVERSION_VERSION = 1

_HELD_OUT_FIRST = ("test", "dev", "calibration", "train")

TWITTER_TOPICS = (
    "Analyst Update",
    "Fed | Central Banks",
    "Company | Product News",
    "Treasuries | Corporate Debt",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Financials",
    "Currencies",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Politics",
    "Personnel Change",
    "Stock Commentary",
    "Stock Movement",
)

VITAMINC_OPTIONS = {
    "SUPPORTS": "The evidence supports the claim.",
    "REFUTES": "The evidence refutes the claim.",
    "NOT ENOUGH INFO": "The evidence neither supports nor refutes the claim.",
}


def bucket(candidate: str, key: str) -> float:
    """A stable number in [0, 1) from the row's content, salted by candidate."""
    return hashlib.sha256(f"{candidate}:{key}".encode()).digest()[0] / 256.0


_HELD_OUT_SPLITS: dict[tuple[str, str], str] = {
    ("case_hold", "validation"): "dev",
    ("case_hold", "test"): "test",
    ("twitter", "validation"): "dev|test",
    ("jailbreak", "test"): "dev|test",
    ("injections", "test"): "dev|test",
    ("vitaminc", "test"): "dev|test",
}
"""Publisher held-out split -> ours. "dev|test" halves it by content hash
because the publisher has no second held-out split."""


def carve(candidate: str, source_split: str, key: str) -> str:
    """Map a publisher split to ours; see the module docstring's table."""
    b = bucket(candidate, key)
    if source_split == "train":
        return "calibration" if b < 0.10 else "train"
    try:
        rule = _HELD_OUT_SPLITS[(candidate, source_split)]
    except KeyError:
        raise ValueError(f"{candidate}: no split mapping for publisher split {source_split!r}") from None
    if rule == "dev|test":
        return "dev" if b < 0.5 else "test"
    return rule


def _row(state, question, options, target, split, *, source, license, label_source, group_id=None):
    row = {
        "state": state,
        "question": question,
        "options": options,
        "target": target,
        "split": split,
        "source": source,
        "license": license,
        "label_source": label_source,
    }
    if group_id is not None:
        row["group_id"] = group_id
    return row


def convert_case_hold(record: dict, split: str, index: int, source: str) -> dict | None:
    endings = record["endings"]
    options = {str(i): text for i, text in enumerate(endings)}
    return _row(
        record["context"],
        "Which holding belongs in the <HOLDING> placeholder of the cited case?",
        options,
        str(record["label"]),
        carve("case_hold", split, record["context"]),
        source=f"{source} {split}[{index}]",
        license="CC-BY-4.0",
        label_source="found:court_opinion_citation",
    )


def convert_twitter(record: dict, split: str, index: int, source: str) -> dict | None:
    options = {str(i): topic for i, topic in enumerate(TWITTER_TOPICS)}
    return _row(
        record["text"],
        "What is the topic of this finance tweet?",
        options,
        str(record["label"]),
        carve("twitter", split, record["text"]),
        source=f"{source} {split}[{index}]",
        license="MIT",
        label_source="dataset_annotation:undocumented",
    )


def convert_jailbreak(record: dict, split: str, index: int, source: str) -> dict | None:
    return _row(
        record["prompt"],
        "Is this prompt an attempt to jailbreak an LLM?",
        {"benign": "No, it is a benign request.", "jailbreak": "Yes, it is a jailbreak attempt."},
        record["type"],
        carve("jailbreak", split, record["prompt"]),
        source=f"{source} {split}[{index}]",
        license="Apache-2.0",
        label_source="source_of_origin",
    )


def convert_injections(record: dict, split: str, index: int, source: str) -> dict | None:
    return _row(
        record["text"],
        "Is this text a prompt injection?",
        {"0": "No, it is a legitimate input.", "1": "Yes, it is a prompt injection."},
        str(record["label"]),
        carve("injections", split, record["text"]),
        source=f"{source} {split}[{index}]",
        license="Apache-2.0",
        label_source="dataset_annotation:undocumented",
    )


def convert_vitaminc(record: dict, split: str, index: int, source: str) -> dict | None:
    """External set only: no train or calibration rows. Synthetic revisions
    are dropped -- only rows built from real Wikipedia edits are kept. Split
    by ``case_id`` so a contrastive pair never straddles dev and test."""
    if record.get("revision_type") != "real" or record.get("label") not in VITAMINC_OPTIONS:
        return None
    return _row(
        f"Claim: {record['claim']}\n\nEvidence: {record['evidence']}",
        "What does the evidence establish about the claim?",
        dict(VITAMINC_OPTIONS),
        record["label"],
        carve("vitaminc", split, record["case_id"]),
        source=f"{source} {split}[{index}]",
        license="CC-BY-SA-3.0",
        label_source="crowd_annotation:real_wikipedia_revision",
        group_id=f"vitaminc:{record['case_id']}",
    )


@dataclass(frozen=True)
class Candidate:
    repo: str
    config: str
    splits: tuple[str, ...]
    convert: Callable[[dict, str, int, str], dict | None]


CANDIDATES: dict[str, Candidate] = {
    "case_hold": Candidate("coastalcph/lex_glue", "case_hold", ("train", "validation", "test"), convert_case_hold),
    "twitter": Candidate(
        "zeroshot/twitter-financial-news-topic", "default", ("train", "validation"), convert_twitter
    ),
    "jailbreak": Candidate("jackhhao/jailbreak-classification", "default", ("train", "test"), convert_jailbreak),
    "injections": Candidate("deepset/prompt-injections", "default", ("train", "test"), convert_injections),
    "vitaminc": Candidate("tals/vitaminc", "default", ("test",), convert_vitaminc),
}


def order_held_out_first(rows: Iterable[dict]) -> list[dict]:
    """Stable sort: test, dev, calibration, train; publisher order within each."""
    rank = {split: i for i, split in enumerate(_HELD_OUT_FIRST)}
    return sorted(rows, key=lambda r: rank[r["split"]])


def convert(name: str, records_by_split: dict[str, list[dict]], source: str, limit: int | None = None) -> list[dict]:
    """Pure conversion, no I/O. ``limit`` keeps a deterministic sample of
    about ``limit`` rows: whole groups (or single ungrouped rows) taken in
    content-hash order until the cap, so a sample never splits a group --
    VitaminC's contrastive pairs stay together."""
    candidate = CANDIDATES[name]
    rows = []
    for split in candidate.splits:
        for index, record in enumerate(records_by_split.get(split, [])):
            row = candidate.convert(record, split, index, source)
            if row is not None:
                rows.append(row)
    if limit is not None and len(rows) > limit:
        units: dict[str, list[dict]] = {}
        for row in rows:
            units.setdefault(row.get("group_id") or row["state"], []).append(row)
        kept: set[int] = set()
        # Salted differently from ``bucket``: sampling by the same hash that
        # assigns splits would take only the low-hash (dev) side.
        for key in sorted(units, key=lambda k: hashlib.sha256(f"sample:{k}".encode()).hexdigest()):
            if len(kept) >= limit:
                break
            kept.update(id(r) for r in units[key])
        rows = [r for r in rows if id(r) in kept]
    return order_held_out_first(rows)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def fetch(name: str) -> tuple[dict[str, list[dict]], dict]:
    """Download the candidate's parquet shards from the Hugging Face datasets
    server. Returns the records by publisher split and a provenance record
    (dataset commit, every shard's URL and sha256) so a conversion can be
    matched to exactly what it read."""
    import pyarrow.parquet as pq

    candidate = CANDIDATES[name]
    info = _get_json(f"https://huggingface.co/api/datasets/{candidate.repo}")
    listing = _get_json(f"https://datasets-server.huggingface.co/parquet?dataset={candidate.repo}")
    shards = sorted(
        (f for f in listing["parquet_files"] if f["config"] == candidate.config and f["split"] in candidate.splits),
        key=lambda f: (f["split"], f["url"]),
    )
    records: dict[str, list[dict]] = {}
    provenance_shards = []
    for shard in shards:
        data = _download(shard["url"])
        provenance_shards.append(
            {"split": shard["split"], "url": shard["url"], "sha256": hashlib.sha256(data).hexdigest()}
        )
        records.setdefault(shard["split"], []).extend(pq.read_table(io.BytesIO(data)).to_pylist())
    missing = set(candidate.splits) - set(records)
    if missing:
        raise RuntimeError(f"{name}: datasets server has no parquet for split(s) {sorted(missing)}")
    provenance = {
        "candidate": name,
        "repo": candidate.repo,
        "config": candidate.config,
        "dataset_commit": info.get("sha"),
        "card_license": (info.get("cardData") or {}).get("license"),
        "conversion_version": CONVERSION_VERSION,
        "shards": provenance_shards,
        "publisher_rows": {split: len(rows) for split, rows in sorted(records.items())},
    }
    return records, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", nargs="+", choices=sorted(CANDIDATES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Deterministic cap on rows per candidate.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.candidates:
        records, provenance = fetch(name)
        source = f"hf:{provenance['repo']}/{provenance['config']}@{provenance['dataset_commit']}"
        rows = convert(name, records, source, args.limit)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        provenance["converted_rows"] = dict(sorted(counts.items()))
        provenance["limit"] = args.limit
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (args.out / f"{name}.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(rows)} rows {provenance['converted_rows']} -> {args.out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
