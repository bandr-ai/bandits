#!/usr/bin/env python3
"""Deterministic verifier shipped in tests/. Stdlib only. No LLM.

Always writes a reward (0 or 1) unless something is wrong with the verifier
itself (missing/invalid expected.json), in which case it crashes without
writing one so the failure is reported as an infrastructure error.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def normalize(text, opts):
    s = str(text)
    if "strip" in opts:
        s = s.strip()
    if "collapse_ws" in opts:
        s = re.sub(r"\s+", " ", s)
    if "lower" in opts:
        s = s.lower()
    if "thousands" in opts:
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    if "percent" in opts:
        s = s.replace("%", "").strip()
    if "sorted_list" in opts:
        items = [x.strip() for x in re.split(r"[,\n;]", s) if x.strip()]
        s = ",".join(sorted(items))
    return s


def as_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def matches(spec, got_raw):
    opts = spec.get("normalize", ["strip"])
    got = normalize(got_raw, opts)
    exp = spec["expected"]
    kind = spec["check"]
    if kind == "exact":
        return got == normalize(exp, opts)
    if kind == "allowed_set":
        return any(got == normalize(e, opts) for e in exp)
    if kind == "numeric":
        g, e = as_float(got), as_float(normalize(exp, opts))
        if g is None or e is None:
            return False
        tol = max(spec.get("tolerance", 0.0), abs(e) * spec.get("rel_tolerance", 0.0))
        return abs(g - e) <= tol
    raise ValueError(f"unknown check kind {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", required=True)
    ap.add_argument("--answer-dir", required=True)
    ap.add_argument("--reward", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.expected).read_text())  # crash here = verifier defect
    ans = Path(a.answer_dir) / spec.get("file", "answer.txt")
    ok = False
    if ans.is_file():
        try:
            ok = matches(spec, ans.read_text(encoding="utf-8", errors="replace"))
        except (ValueError, TypeError):
            ok = False
    Path(a.reward).parent.mkdir(parents=True, exist_ok=True)
    Path(a.reward).write_text("1\n" if ok else "0\n")
    print("reward", 1 if ok else 0)


if __name__ == "__main__":
    sys.exit(main())
