#!/usr/bin/env python3
"""Per-source content audit: present -> retained -> interpreted -> undetermined.

For every model call Bandits decoded, and for each direction (input, output):

- present:      the source record holds a non-empty value
- retained:     Bandits kept that raw value on the span
- interpreted:  Bandits derived structured messages (or a model answer) from it
- undetermined: present but not interpreted; reported by value *shape* only

Counts and shapes only; no trace content is printed.

    uv run scripts/audit_source.py langfuse datasets/public/langfuse-docs/lf_full.jsonl
    uv run scripts/audit_source.py langsmith datasets/public/langsmith-shared/x.json
    uv run scripts/audit_source.py otlp-std FILE --mode workflow
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bandits.ingest import load_corpus
from bandits.redact import RedactionRuleset
from bandits.traces import SpanKind, SpanStatus, WorkflowDeclaration

NO_REDACTION = RedactionRuleset("none-audit", ())
EMPTY = (None, "", [], {})

# Where each native source records a model call's raw input and output.
NATIVE_FIELDS = {
    "langfuse": ("input", "output"),
    "langsmith": ("inputs", "outputs"),
}
OTLP_INPUT_KEYS = ("gen_ai.input.messages", "input.value", "gen_ai.prompt", "ai.prompt")
OTLP_OUTPUT_KEYS = ("gen_ai.output.messages", "output.value", "gen_ai.completion", "ai.response")


def _parsed(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def shape(value: Any, depth: int = 0) -> str:
    """A content-free structural fingerprint, e.g. list[dict{key,value}]."""
    value = _parsed(value)
    if depth > 2:
        return "…"
    if isinstance(value, dict):
        return "dict{" + ",".join(sorted(value)[:6]) + ("…" if len(value) > 6 else "") + "}"
    if isinstance(value, list):
        return f"list[{shape(value[0], depth + 1)}]" if value else "list[]"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _present(source: str, attributes: dict[str, Any], direction: int) -> tuple[bool, Any]:
    if source in NATIVE_FIELDS:
        record = _parsed(attributes.get("bandits.native.record"))
        value = record.get(NATIVE_FIELDS[source][direction]) if isinstance(record, dict) else None
        return _parsed(value) not in EMPTY, value
    if source == "phoenix":
        record = _parsed(attributes.get("bandits.native.record")) or {}
        value = (record.get("attributes") or {}).get(("input.value", "output.value")[direction])
        return _parsed(value) not in EMPTY, value
    keys = (OTLP_INPUT_KEYS, OTLP_OUTPUT_KEYS)[direction]
    prefix = ("gen_ai.prompt.", "gen_ai.completion.")[direction]
    oi = ("llm.input_messages.", "llm.output_messages.")[direction]
    for key in keys:
        if _parsed(attributes.get(key)) not in EMPTY:
            return True, attributes[key]
    flat = [
        k
        for k in attributes
        if k.startswith((prefix, oi)) and k.endswith((".content", ".role", ".name", ".arguments"))
    ]
    return bool(flat), {k: None for k in flat}


def _has_parts(messages: Any) -> bool:
    messages = _parsed(messages)
    return isinstance(messages, list) and any(
        isinstance(m, dict) and m.get("parts") for m in messages
    )


def _empty_messages(messages: Any) -> list[str]:
    messages = _parsed(messages)
    if not isinstance(messages, list):
        return []
    return [m.get("role", "?") for m in messages if isinstance(m, dict) and not m.get("parts")]


def _only_json_text(value: Any) -> bool:
    value = _parsed(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        texts = [p.get("content") for p in value[0].get("parts") or [] if p.get("type") == "text"]
        if len(texts) != 1 or len(value[0]["parts"]) != 1:
            return False
        value = texts[0]
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped[:1] not in "[{":
        return False
    try:
        return isinstance(json.loads(stripped), (list, dict))
    except json.JSONDecodeError:
        return False


def audit(source: str, path: Path, workflow: WorkflowDeclaration | None) -> dict[str, Any]:
    corpus = load_corpus(path, source, NO_REDACTION, workflow=workflow)
    models = [s for t in corpus.traces for s in t.spans if s.kind == SpanKind.MODEL]
    counts: Counter[str] = Counter()
    undetermined: dict[str, Counter[str]] = {"input": Counter(), "output": Counter()}
    for span in models:
        a = span.attributes
        for direction, name in enumerate(("input", "output")):
            present, raw = _present(source, a, direction)
            if not present:
                if name == "output" and span.status == SpanStatus.ERROR:
                    counts["output.absent_failed_call"] += 1
                else:
                    counts[f"{name}.absent_in_source"] += 1
                continue
            counts[f"{name}.present"] += 1
            retained = (
                a.get(f"{name}.value") is not None
                or a.get(f"gen_ai.{name}.messages") is not None
                or "bandits.native.record" in a
                or any(k.startswith(("gen_ai.prompt.", "gen_ai.completion.", "llm.")) for k in a)
            )
            counts[f"{name}.retained"] += retained
            messages = a.get(f"gen_ai.{name}.messages")
            interpreted = _has_parts(messages) or (name == "output" and span.output not in EMPTY)
            derived = a.get(f"bandits.{name}_messages_from") is not None or not _has_parts(messages)
            if (
                interpreted
                and derived
                and _only_json_text(messages if _has_parts(messages) else span.output)
            ):
                # One text blob that is itself a JSON structure: a record was
                # copied through as "text" rather than read as messages.
                counts[f"{name}.raw_json_text"] += 1
                undetermined[name]["raw JSON as text: " + shape(raw)] += 1
            elif interpreted:
                counts[f"{name}.interpreted"] += 1
                empty = _empty_messages(messages)
                if empty:
                    # Parsed, but some turns came out with nothing in them.
                    counts[f"{name}.calls_with_empty_turns"] += 1
                    undetermined[name][f"empty turns: {','.join(sorted(set(empty)))}"] += 1
            else:
                counts[f"{name}.undetermined"] += 1
                undetermined[name][shape(raw)] += 1
    return {
        "source": source,
        "traces": len(corpus.traces),
        "model_calls": len(models),
        "issues": dict(Counter(i.kind for i in corpus.issues)),
        "counts": dict(counts),
        "undetermined_shapes": {k: v.most_common(10) for k, v in undetermined.items()},
    }


def print_audit(report: dict[str, Any]) -> None:
    c = report["counts"]
    print(
        f"source: {report['source']}  traces: {report['traces']}  model calls: {report['model_calls']}"
    )
    print(f"issues: {report['issues']}")
    for name in ("input", "output"):
        present = c.get(f"{name}.present", 0)
        print(
            f"{name:<7} present {present:>5} | retained {c.get(f'{name}.retained', 0):>5}"
            f" | interpreted {c.get(f'{name}.interpreted', 0):>5}"
            f" | raw JSON text {c.get(f'{name}.raw_json_text', 0):>4}"
            f" | undetermined {c.get(f'{name}.undetermined', 0):>5}"
            f" | absent in source {c.get(f'{name}.absent_in_source', 0):>5}"
            f" | calls with empty turns {c.get(f'{name}.calls_with_empty_turns', 0)}"
        )
        for fingerprint, n in report["undetermined_shapes"][name]:
            print(f"          undetermined {n:>4} x {fingerprint}")
    if c.get("output.absent_failed_call"):
        print(f"failed calls without output: {c['output.absent_failed_call']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("conversation", "workflow"), default="conversation")
    parser.add_argument("--json", type=Path, help="write the reports here")
    args = parser.parse_args()
    workflow = WorkflowDeclaration() if args.mode == "workflow" else None
    reports = []
    for path in args.paths:
        report = audit(args.source, path, workflow)
        report["path"] = str(path)
        print(f"== {path}")
        print_audit(report)
        reports.append(report)
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1), encoding="utf-8")
    bad = any(
        r["counts"].get(f"{d}.{kind}")
        for r in reports
        for d in ("input", "output")
        for kind in ("undetermined", "raw_json_text", "calls_with_empty_turns")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
