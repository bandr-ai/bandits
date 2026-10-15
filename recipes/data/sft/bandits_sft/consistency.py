"""Consistency check (any domain): one call lists each observation's facts, then contradictions, results impossible for
their own call, and unsupported answer claims (CONSISTENCY_CHECK). Code labels the trace bad when any list is
non-empty.

    python -m bandits_sft.consistency --config <cfg> --cases <cases.jsonl> --out <dir> [--workers 6]

cases.jsonl rows: {"id", "trace", optional "seed" (the real trace it was varied from), optional "expected": "good"|"bad"}; with expected labels it reports how many bad
traces are caught and how many good ones are rejected.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

from .config import load_config
from .io import read_jsonl
from .llm import BudgetExhausted, chat_json, configure_call_log, current_item, get_llm
from .prompts import CONSISTENCY_CHECK
from .schema import render
from .trace_text import tools_text

FULL = 10**7
PROBLEMS = ("contradictions", "impossible_results", "unsupported_claims")


def check(cfg, trace: dict, seed: dict | None = None, notes: list[str] | None = None) -> dict:
    """{"label": "good"|"bad", "problems": {kind: [...]}, "facts": [...]}; label is decided here, not by the model.
    seed: the real trace this one was varied from, shown as the reference for the agent's conventions."""
    reference = (f'\n<reference_seed note="a real recorded conversation of this agent">\n{render(seed, max_obs_chars=FULL)}\n'
                 f'</reference_seed>\n') if seed else ""
    if notes:
        from .rubric_judge import notes_block
        reference += "\n" + notes_block(notes) + "\n"
    prompt = CONSISTENCY_CHECK.format(system=trace.get("system", ""), reference=reference,
                                      tools=tools_text(trace.get("tools") or []) or "(none)",
                                      conversation=render(trace, max_obs_chars=FULL))
    for attempt in (1, 2):  # an incomplete answer gets one retry, then fails closed (an error, never "good")
        out = chat_json(get_llm(cfg, "judge"), "You check agent conversations for internal consistency. Output JSON only.",
                        prompt)
        missing = [k for k in PROBLEMS if not isinstance(out, dict) or not isinstance(out.get(k), list)]
        if not missing:
            break
        if attempt == 2:
            raise ValueError(f"consistency check answer lacks lists: {missing}")
    problems = {k: [p for p in out[k] if p] for k in PROBLEMS}
    return {"label": "bad" if any(problems.values()) else "good", "problems": problems, "facts": out.get("facts") or []}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in ok if r.get("expected") == "bad"]
    good = [r for r in ok if r.get("expected") == "good"]
    return {"checked": len(ok), "errors": len(rows) - len(ok),
            "bad_caught": f"{sum(r['label'] == 'bad' for r in bad)}/{len(bad)}",
            "good_rejected": f"{sum(r['label'] == 'bad' for r in good)}/{len(good)}",
            "flagged_by_kind": {k: sum(bool(r["problems"][k]) for r in ok) for k in PROBLEMS}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bandits_sft.consistency")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    configure_call_log(out / "llm_calls.jsonl")
    cases = read_jsonl(a.cases)

    def one(c):
        current_item.set(f"consistency/{c['id']}")
        row = {"id": c["id"], "expected": c.get("expected")}
        try:
            row.update(check(cfg, c["trace"], c.get("seed")))
        except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
            raise
        except Exception as e:  # noqa: BLE001 - an unchecked case is reported as an error, never as good
            row["error"] = repr(e)[:300]
        return row
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, cases))
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    rep = summarize(rows)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
