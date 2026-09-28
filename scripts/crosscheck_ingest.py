#!/usr/bin/env python3
"""Compare Bandits' OTLP message decoding with an independent mapper.

Each OTLP model span is decoded twice: by Bandits (``otlp-std``) and by
genai-interlingua (https://github.com/Grace/genai-interlingua, a Go CLI). For
every span both decoded, the roles, text, tool calls and tool results of the
input and output messages are compared. A disagreement names the span and the
field; it does not say which side is wrong.

Redaction is disabled on the Bandits side so content can be compared exactly;
nothing is written to a store.

Sessions are split by a fixed hash into ``dev`` and ``holdout``. Fix readers
against dev disagreements only, then report holdout.

    uv run --with pyarrow scripts/crosscheck_ingest.py otlp FILE_OR_DIR...
    uv run --with pyarrow scripts/crosscheck_ingest.py exgentic --shards 0000 --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bandits.ingest import load_corpus
from bandits.redact import RedactionRuleset
from bandits.traces import SpanKind

NO_REDACTION = RedactionRuleset("none-crosscheck", ())
HOLDOUT_PERCENT = 20
FIELDS = (
    "input.roles",
    "input.text",
    "input.tool_calls",
    "input.tool_results",
    "output.roles",
    "output.text",
    "output.tool_calls",
)
_LEGACY_FUNCTION_NAME = re.compile(
    r"^gen_ai\.completion\.\d+\.(?:function_call\.name|tool_calls\.\d+\.function\.name)$"
)


def split_of(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    return "holdout" if bucket < HOLDOUT_PERCENT else "dev"


# ---------- message features ----------


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _canon(value: Any) -> str:
    value = _json(value)
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _text(part: dict[str, Any]) -> str | None:
    for key in ("content", "text"):
        if isinstance(part.get(key), str):
            return part[key]
    return None


def features(messages: Any) -> dict[str, list] | None:
    messages = _json(messages)
    if not isinstance(messages, list):
        return None
    roles, texts, calls, results = [], [], [], []
    for message in messages:
        if not isinstance(message, dict):
            roles.append(f"<{type(message).__name__}>")
            continue
        roles.append(message.get("role"))
        text = []
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "tool_call":
                calls.append((part.get("name"), _canon(part.get("arguments"))))
            elif kind == "tool_call_response":
                results.append(_canon(part.get("response", part.get("result"))))
            elif kind in ("text", None) and _text(part) is not None:
                text.append(_text(part))
        texts.append(" ".join(" ".join(text).split()))
    return {"roles": roles, "text": texts, "tool_calls": calls, "tool_results": results}


def span_features(input_messages: Any, output_messages: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for direction, value in (("input", input_messages), ("output", output_messages)):
        found = features(value)
        for key in ("roles", "text", "tool_calls", "tool_results"):
            out[f"{direction}.{key}"] = None if found is None else found[key]
    return out


# ---------- the two decoders ----------


def bandits_spans(path: Path) -> dict[str, dict[str, Any]]:
    corpus = load_corpus(path, "otlp-std", NO_REDACTION)
    decoded = {}
    for trace in corpus.traces:
        for span in trace.spans:
            if span.kind != SpanKind.MODEL:
                continue
            if span.span_id in decoded:
                raise ValueError(f"model span ID repeats across traces: {span.span_id}")
            decoded[span.span_id] = span_features(
                span.attributes.get("gen_ai.input.messages"),
                span.attributes.get("gen_ai.output.messages"),
            )
    return decoded


def _any_value(value: dict[str, Any]) -> Any:
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    if "intValue" in value:
        return int(value["intValue"])
    if "arrayValue" in value:
        return [_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {kv["key"]: _any_value(kv["value"]) for kv in value["kvlistValue"]["values"]}
    return None


def interlingua_binary() -> str | None:
    return os.environ.get("INTERLINGUA") or shutil.which("interlingua")


def interlingua_spans(path: Path, binary: str) -> tuple[dict[str, dict[str, Any]], Counter]:
    decoded: dict[str, dict[str, Any]] = {}
    dialects: Counter = Counter()
    files = sorted(path.rglob("*.json*")) if path.is_dir() else [path]
    for file in files:
        raw = file.read_text()
        documents = (
            [raw] if file.suffix == ".json" else [ln for ln in raw.splitlines() if ln.strip()]
        )
        for document in documents:
            result = subprocess.run(
                [binary, "-target", "v1.41.0"],
                input=document,
                capture_output=True,
                text=True,
                check=True,
            )
            for resource in json.loads(result.stdout).get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        attributes = {
                            a["key"]: _any_value(a["value"]) for a in span.get("attributes", [])
                        }
                        dialects[attributes.get("interlingua.dialect")] += 1
                        if attributes.get("gen_ai.operation.name") not in (
                            "chat",
                            "text_completion",
                            "generate_content",
                        ):
                            continue
                        if _hex_id(span["spanId"]) in decoded:
                            raise ValueError(
                                f"model span ID repeats across traces: {span['spanId']}"
                            )
                        decoded[_hex_id(span["spanId"])] = span_features(
                            attributes.get("gen_ai.input.messages"),
                            attributes.get("gen_ai.output.messages"),
                        )
    return decoded, dialects


def _hex_id(value: str) -> str:
    return value.lower()


def source_facts(path: Path) -> dict[str, dict[str, Any]]:
    """Content visible in raw OTLP that both decoders might otherwise miss."""
    facts: dict[str, dict[str, Any]] = {}
    files = sorted(path.rglob("*.json*")) if path.is_dir() else [path]
    for file in files:
        raw = file.read_text()
        documents = (
            [raw] if file.suffix == ".json" else [ln for ln in raw.splitlines() if ln.strip()]
        )
        for document in documents:
            request = json.loads(document)
            for resource in request.get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        attributes = {
                            a["key"]: _any_value(a["value"]) for a in span.get("attributes", [])
                        }
                        if attributes.get("gen_ai.operation.name") not in (
                            "chat",
                            "text_completion",
                            "generate_content",
                        ):
                            continue
                        span_id = _hex_id(span["spanId"])
                        if span_id in facts:
                            raise ValueError(f"model span ID repeats across traces: {span_id}")
                        required: dict[str, Any] = {}
                        for direction in ("input", "output"):
                            parsed = features(attributes.get(f"gen_ai.{direction}.messages"))
                            if parsed is not None:
                                for field in ("roles", "text", "tool_calls", "tool_results"):
                                    required[f"{direction}.{field}"] = parsed[field]
                        names = {
                            value
                            for key, value in attributes.items()
                            if _LEGACY_FUNCTION_NAME.fullmatch(key) and isinstance(value, str)
                        }
                        if names:
                            required["output.tool_call_names"] = names
                        facts[span_id] = required
    return facts


def source_losses(
    decoded: dict[str, dict[str, Any]], facts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    losses = []
    for span_id, required in facts.items():
        actual = decoded.get(span_id)
        if actual is None:
            losses.append({"span_id": span_id, "fields": ["model_span"]})
            continue
        missing = []
        for field, expected in required.items():
            if field == "output.tool_call_names":
                seen = {call[0] for call in actual["output.tool_calls"] or []}
                if not expected <= seen:
                    missing.append(field)
            elif field.endswith((".tool_calls", ".tool_results")):
                seen = actual.get(field) or []
                if any(item not in seen for item in expected):
                    missing.append(field)
            elif expected != actual.get(field):
                missing.append(field)
        if missing:
            losses.append({"span_id": span_id, "fields": missing})
    return losses


# ---------- comparison ----------


def compare(
    ours: dict[str, dict[str, Any]],
    theirs: dict[str, dict[str, Any]],
    session_of: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "only_bandits": sorted(set(ours) - set(theirs)),
        "only_interlingua": sorted(set(theirs) - set(ours)),
        "splits": {},
        "disagreements": [],
    }
    per_split: dict[str, Counter] = defaultdict(Counter)
    for span_id in sorted(set(ours) & set(theirs)):
        split = split_of(session_of.get(span_id, span_id))
        per_split[split]["spans"] += 1
        diffs = [f for f in FIELDS if ours[span_id][f] != theirs[span_id][f]]
        for field in FIELDS:
            per_split[split][f"agree.{field}"] += field not in diffs
        if diffs:
            per_split[split]["spans_disagreeing"] += 1
            report["disagreements"].append(
                {
                    "span_id": span_id,
                    "split": split,
                    "fields": {
                        f: {"bandits": ours[span_id][f], "interlingua": theirs[span_id][f]}
                        for f in diffs
                    },
                }
            )
    report["splits"] = {k: dict(v) for k, v in per_split.items()}
    return report


def print_report(report: dict[str, Any], dialects: Counter) -> None:
    print(f"interlingua dialects: {dict(dialects)}")
    print(f"model spans only in bandits:     {len(report['only_bandits'])}")
    print(f"model spans only in interlingua: {len(report['only_interlingua'])}")
    for split, counts in sorted(report["splits"].items()):
        spans = counts["spans"]
        print(
            f"[{split}] spans compared: {spans}, disagreeing: {counts.get('spans_disagreeing', 0)}"
        )
        for field in FIELDS:
            agree = counts.get(f"agree.{field}", 0)
            print(f"    {field:<20} {agree}/{spans} agree")
    kinds = Counter(field for d in report["disagreements"] for field in d["fields"])
    print(f"disagreements by field: {dict(kinds)}")


# ---------- inputs ----------


def _nanos(value: str) -> str:
    return str(int(datetime.fromisoformat(value).timestamp() * 1_000_000) * 1000)


def _kv(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool):
            encoded = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        elif isinstance(value, float):
            encoded = {"doubleValue": value}
        elif isinstance(value, list):
            encoded = {"arrayValue": {"values": [{"stringValue": str(v)} for v in value]}}
        else:
            encoded = {"stringValue": str(value)}
        out.append({"key": key, "value": encoded})
    return out


def exgentic_to_otlp(row: dict[str, Any]) -> dict[str, Any]:
    """One Exgentic session (flat GenAI spans) as one OTLP/JSON request."""
    spans = []
    resource: dict[str, Any] = {}
    for span in row["spans"]:
        resource = span.get("resource_attributes") or resource
        entry = {
            "traceId": span["trace_id"],
            "spanId": span["span_id"],
            "name": span["name"],
            "kind": 3,
            "startTimeUnixNano": _nanos(span["start_time"]),
            "endTimeUnixNano": _nanos(span["end_time"]),
            "attributes": _kv(span["attributes"] or {}),
            "status": {"code": (span.get("status") or {}).get("code") or 0},
        }
        if span.get("parent_span_id"):
            entry["parentSpanId"] = span["parent_span_id"]
        spans.append(entry)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _kv(resource)},
                "scopeSpans": [{"scope": {"name": "exgentic"}, "spans": spans}],
            }
        ]
    }


def run(paths: list[Path], session_of: dict[str, str], out: Path | None) -> int:
    binary = interlingua_binary()
    if binary is None:
        print(
            "interlingua not found (set INTERLINGUA or add it to PATH); skipping", file=sys.stderr
        )
        return 2
    ours: dict[str, dict[str, Any]] = {}
    theirs: dict[str, dict[str, Any]] = {}
    dialects: Counter = Counter()
    facts: dict[str, dict[str, Any]] = {}
    for path in paths:
        ours.update(bandits_spans(path))
        decoded, found = interlingua_spans(path, binary)
        theirs.update(decoded)
        dialects.update(found)
        new_facts = source_facts(path)
        if set(facts) & set(new_facts):
            raise ValueError("model span IDs repeat across source files")
        facts.update(new_facts)
    report = compare(ours, theirs, session_of)
    report["source_spans_checked"] = len(facts)
    report["source_losses_bandits"] = source_losses(ours, facts)
    report["source_losses_interlingua"] = source_losses(theirs, facts)
    print_report(report, dialects)
    print(f"source spans checked: {len(facts)}")
    print(f"source losses in bandits: {len(report['source_losses_bandits'])}")
    print(f"source losses in interlingua: {len(report['source_losses_interlingua'])}")
    if out:
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        print(f"full report: {out}")
    return int(
        bool(
            report["disagreements"]
            or report["source_losses_bandits"]
            or report["source_losses_interlingua"]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    otlp = sub.add_parser("otlp")
    otlp.add_argument("paths", nargs="+", type=Path)
    otlp.add_argument("--out", type=Path)
    exg = sub.add_parser("exgentic")
    exg.add_argument("--data", type=Path, default=Path("datasets/exgentic-v2/data/train"))
    exg.add_argument("--shards", nargs="+", default=["0000"])
    exg.add_argument("--limit", type=int, default=200, help="sessions per shard")
    exg.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.mode == "otlp":
        return run(args.paths, {}, args.out)

    import pyarrow.parquet as pq

    session_of: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for shard in args.shards:
            path = Path(tmp) / f"{shard}.jsonl"
            count = 0
            with path.open("w") as stream:
                for batch in pq.ParquetFile(args.data / f"{shard}.parquet").iter_batches(
                    batch_size=4, columns=["session_id", "spans"]
                ):
                    for row in batch.to_pylist():
                        if count >= args.limit:
                            break
                        for span in row["spans"]:
                            if span["span_id"] in session_of:
                                raise ValueError(f"model span ID repeats: {span['span_id']}")
                            session_of[span["span_id"]] = row["session_id"]
                        stream.write(json.dumps(exgentic_to_otlp(row)) + "\n")
                        count += 1
                    if count >= args.limit:
                        break
            paths.append(path)
        return run(paths, session_of, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
