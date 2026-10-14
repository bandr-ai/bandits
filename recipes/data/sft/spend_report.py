"""Fireworks spend from every llm_calls.jsonl under the given roots (deduplicated across copies of the same run).

Two numbers per group: `logged` is the cost the pipeline wrote (config prices; deepseek's are stale and too high),
`list` reprices every call from its token usage at Fireworks list prices. Failed calls that returned usage (e.g. hit
the token limit) are billed and counted; calls that never got a response cost nothing.

    python spend_report.py <root> [<root> ...] [--json out.json]
"""
import argparse
import collections
import json
import os

# USD per 1M tokens, Fireworks list prices (checked 2026-10-13): input, cached input, output
LIST = {"deepseek-v4p1-flash": (0.22, 0.22, 0.66), "glm-5p3-flash": (0.15, 0.029, 0.50)}


def list_cost(model: str, usage: dict | None) -> float:
    if not usage:
        return 0.0
    p = LIST.get(model)
    if not p:
        return 0.0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    prompt = (usage.get("prompt_tokens") or 0) - cached
    return (prompt * p[0] + cached * p[1] + (usage.get("completion_tokens") or 0) * p[2]) / 1e6


def run_name(path: str) -> str:
    """A readable run label: the directory under work/, a scratchpad dir, or the box folder."""
    d = os.path.dirname(path)
    for marker in ("/work/", "/scratchpad/", "/sft-ae/", "/sft-analyzer-pilot/"):
        if marker in d:
            return marker.strip("/") + ":" + d.split(marker, 1)[1]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    seen, rows = set(), []
    for root in a.roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [x for x in dn if x != ".venv"]
            if "llm_calls.jsonl" not in fn:
                continue
            p = os.path.join(dp, "llm_calls.jsonl")
            with open(p, errors="ignore") as fh:
                lines = fh.readlines()
            for line in lines:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "fireworks" not in str(r.get("model")):
                    continue
                key = (r.get("ts"), r.get("role"), r.get("attempt"), json.dumps(r.get("usage"), sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                model = r["model"].split("/")[-1]
                rows.append({"day": (r.get("ts") or "?")[:10], "run": run_name(p), "role": r.get("role") or "?", "model": model,
                             "ok": bool(r.get("ok")), "logged": float(r.get("cost") or 0), "list": list_cost(model, r.get("usage")),
                             "in": (r.get("usage") or {}).get("prompt_tokens") or 0, "out": (r.get("usage") or {}).get("completion_tokens") or 0})
    def table(field):
        g = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
        for r in rows:
            x = g[r[field]]
            x[0] += 1; x[1] += r["logged"]; x[2] += r["list"]; x[3] += r["in"]; x[4] += r["out"]
        return dict(sorted(g.items(), key=lambda kv: -kv[1][2]))
    out = {"calls": len(rows), "failed_calls": sum(not r["ok"] for r in rows),
           "logged_total": round(sum(r["logged"] for r in rows), 2), "list_total": round(sum(r["list"] for r in rows), 2),
           "by_day": table("day"), "by_model": table("model"), "by_role": table("role"), "by_run": table("run")}
    print(f"calls {out['calls']} (failed {out['failed_calls']})  logged ${out['logged_total']}  list-price ${out['list_total']}")
    for k in ("by_day", "by_model", "by_role", "by_run"):
        print(f"\n{k}  (calls, logged $, list $, Mtok in, Mtok out)")
        for name, (n, lg, ls, i, o) in out[k].items():
            print(f"  {str(name)[:60]:60} {n:6d}  {lg:8.2f}  {ls:8.2f}  {i/1e6:7.2f}  {o/1e6:6.2f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
