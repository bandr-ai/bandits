"""Judge self-test and REVISE loop, from good seeds only (any domain).

build:  for each seed, cases with a known label: the seed unchanged and a harmless edit (expected good), and one planted
        generator mistake per defect type (expected bad). An LLM writes find/replace edits; code applies them.
score:  the judge (rubric_judge) on every case; bad caught and good wrongly rejected, per defect type.
loop:   score the rubric on dev cases, ask REVISE (rubrics/induce_rubric_prompt_v2.md) for edits from the misses, keep a
        revision only if the dev score improves; report the first and the final rubric on test cases, which never feed
        a revision. Dev cases come from the seeds the rubric was induced from; test cases from held-out seeds.

    python -m simia_plus.selftest build --config <cfg> --seeds <seeds.jsonl> --out <cases.jsonl> [--per-seed-limit N]
    python -m simia_plus.selftest loop  --config <cfg> --rubric-dir <dir with domain.md> --dev <cases> --test <cases>
                                        --out <dir> [--rounds 1]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .config import load_config
from .io import read_jsonl
from .llm import BudgetExhausted, chat_json, configure_call_log, get_llm
from .rubric_judge import describe_change, judge_rubric
from .rubric_loop import RUBRICS, SLOTS_V2, assemble_v2, fill, generic_v2, parse_slots, prompt_section
from .schema import render

DEFECTS = {
    "contradiction": "change what one observation says about a fact the answer depends on, and leave another observation "
                     "of the same thing unchanged, so the two now disagree; adjust the final answer to follow the changed one.",
    "wrong_answer": "keep every observation; change the final answer to a different answer that the agent's own rules do "
                    "not allow for this evidence (a wrong kind of answer, not just different wording).",
    "impossible_content": "make one observation contain something its source could not have provided in this trace: data "
                          "from a source shown as unavailable elsewhere, a name or value that appears nowhere else, or a "
                          "conclusion written inside raw source content.",
    "decisive_removed": "remove or empty the observation that decided the answer, keep the final answer unchanged.",
    "unsupported_claim": "keep every observation; change the final answer so it states something about the evidence that "
                         "the observations contradict or do not contain.",
}


def _prompt(name: str) -> str:
    text = (RUBRICS / "selftest_prompts.md").read_text()
    m = re.search(rf"^## {name}\n(.*?)(?=^---$|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip()


def apply_edits(seed: dict, edits: list[dict]) -> dict | None:
    """The seed with the edits applied, or None when any edit does not match exactly one place."""
    t = copy.deepcopy(seed)
    for e in edits:
        try:
            m = t["messages"][int(e["message"])]
        except (KeyError, IndexError, ValueError, TypeError):
            return None
        content = m.get("content")
        if not isinstance(content, str) or not e.get("find") or content.count(e["find"]) != 1:
            return None
        m["content"] = content.replace(e["find"], e.get("replace", ""), 1)
    return t if edits else None


def make_case(cfg, seed: dict, kind: str) -> dict:
    """kind: 'unchanged', 'harmless' or a DEFECTS key. Returns a case, or one with 'skip' set."""
    base = {"seed_id": seed["id"], "kind": kind, "expected": "good" if kind in ("unchanged", "harmless") else "bad"}
    if kind == "unchanged":
        return {**base, "id": f"{seed['id']}:unchanged", "trace": seed, "change": None, "defect": "none", "with_seed": False}
    vals = {"system": seed.get("system", ""), "trace": render(seed, max_obs_chars=10**7)}
    if kind == "harmless":
        user = fill(_prompt("HARMLESS"), vals)
    else:
        user = fill(_prompt("CORRUPT"), {**vals, "defect": kind, "definition": DEFECTS[kind]})
    try:
        out = chat_json(get_llm(cfg, "selftest"), "You edit traces for a judge test. Output JSON only.", user)
    except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
        raise
    except Exception as e:  # noqa: BLE001
        return {**base, "id": f"{seed['id']}:{kind}", "skip": f"call failed: {e!r}"[:200]}
    trace = apply_edits(seed, out.get("edits") or []) if isinstance(out, dict) else None
    if trace is None:
        return {**base, "id": f"{seed['id']}:{kind}", "skip": (out or {}).get("reason") or "edits did not apply"}
    trace["id"] = f"{seed['id']}:{kind}"
    return {**base, "id": trace["id"], "trace": trace, "change": out.get("declared_change"), "defect": out.get("defect"),
            "with_seed": True, "edits": out.get("edits")}


def build(cfg, seeds: list[dict], workers: int = 8) -> list[dict]:
    jobs = [(s, k) for s in seeds for k in ("unchanged", "harmless", *DEFECTS)]
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: make_case(cfg, *j), jobs))


def judge_cases(cfg, cases: list[dict], seeds: dict, rubric: str, workers: int = 8) -> list[dict]:
    def one(c):
        try:
            r = judge_rubric(cfg, c["trace"], seed=seeds[c["seed_id"]] if c["with_seed"] else None, rubric=rubric,
                             change=describe_change(c["change"]) if c["change"] else None)
        except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
            raise
        except Exception as e:  # noqa: BLE001 - an unjudged case is reported as an error, never as a verdict
            r = {"label": "error", "error": repr(e)[:200]}
        return {"id": c["id"], "kind": c["kind"], "expected": c["expected"], "label": r.get("label"),
                "critique": (r.get("critique") or r.get("reason") or "")[-700:], "integrity": r.get("integrity")}
    usable = [c for c in cases if not c.get("skip")]
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, usable))


def score(verdicts: list[dict]) -> dict:
    """Bad caught (label not good) and good wrongly rejected; errors are counted apart, never as verdicts."""
    judged = [v for v in verdicts if v["label"] not in ("error", None)]
    bad = [v for v in judged if v["expected"] == "bad"]
    good = [v for v in judged if v["expected"] == "good"]
    caught = sum(v["label"] != "good" for v in bad)
    rejected = sum(v["label"] != "good" for v in good)
    per = Counter(v["kind"] for v in bad if v["label"] != "good")
    tot = Counter(v["kind"] for v in bad)
    recall = caught / len(bad) if bad else 0.0
    specificity = 1 - rejected / len(good) if good else 0.0
    return {"bad_caught": f"{caught}/{len(bad)}", "good_rejected": f"{rejected}/{len(good)}",
            "errors": len(verdicts) - len(judged), "balanced": round((recall + specificity) / 2, 3),
            "per_defect": {k: f"{per[k]}/{tot[k]}" for k in tot}}


def failures_text(cases: dict, verdicts: list[dict]) -> str:
    rows = []
    for v in verdicts:
        if v["label"] in ("error", None) or (v["label"] == "good") == (v["expected"] == "good"):
            continue
        c = cases[v["id"]]
        rows.append(f"- case {v['id']} | made as: {c['kind']} | expected: {v['expected']} | judge said: {v['label']}\n"
                    f"  planted defect: {c.get('defect')}\n  declared change: {c.get('change')}\n"
                    f"  judge critique (end): {v['critique'][-400:]}")
    return "\n".join(rows) or "(no disagreements)"


def revise(cfg, slots: dict, failures: str) -> tuple[dict, str]:
    sections = "\n".join(f"<{k}>\n{slots[k]}\n</{k}>" for k in SLOTS_V2)
    user = fill(prompt_section("REVISE", "induce_rubric_prompt_v2.md"),
                {"generic_judge": generic_v2(), "sections": sections, "failures": failures})
    out = get_llm(cfg, "rubric_induce").chat([
        {"role": "system", "content": "You revise grading rubrics from measured mistakes. Cite sources. Output the blocks only."},
        {"role": "user", "content": user}])
    text = out.get("content") or ""
    m = re.search(r"<changelog>(.*?)</changelog>", text, flags=re.DOTALL)
    return parse_slots(text), (m.group(1).strip() if m else "")


def loop(cfg, slots: dict, dev: list[dict], test: list[dict], seeds: dict, out: Path, rounds: int) -> dict:
    devc, testc = {c["id"]: c for c in dev}, {c["id"]: c for c in test}
    history = []
    best = slots
    best_v = judge_cases(cfg, dev, seeds, assemble_v2(best))
    best_s = score(best_v)
    history.append({"round": 0, "dev": best_s, "accepted": True})
    (out / "rubric_r0.md").write_text(assemble_v2(best))
    for r in range(1, rounds + 1):
        try:
            cand, changelog = revise(cfg, best, failures_text(devc, best_v))
        except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
            raise
        except Exception as e:  # noqa: BLE001
            history.append({"round": r, "error": repr(e)[:200]})
            continue
        v = judge_cases(cfg, dev, seeds, assemble_v2(cand))
        s = score(v)
        ok = s["balanced"] > best_s["balanced"]
        history.append({"round": r, "dev": s, "accepted": ok, "changelog": changelog})
        (out / f"rubric_r{r}.md").write_text(assemble_v2(cand))
        if ok:
            best, best_v, best_s = cand, v, s
    test_first = judge_cases(cfg, test, seeds, (out / "rubric_r0.md").read_text())
    test_final = judge_cases(cfg, test, seeds, assemble_v2(best)) if best is not slots else test_first
    (out / "domain_final.md").write_text("\n".join(f"<{k}>\n{best[k]}\n</{k}>" for k in SLOTS_V2) + "\n")
    (out / "rubric_final.md").write_text(assemble_v2(best))
    for name, v in (("dev_final", best_v), ("test_first", test_first), ("test_final", test_final)):
        (out / f"verdicts_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in v))
    report = {"history": history, "test_first": score(test_first), "test_final": score(test_final)}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m simia_plus.selftest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", required=True)
    b.add_argument("--seeds", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, help="use at most this many seeds")
    lp = sub.add_parser("loop")
    lp.add_argument("--config", required=True)
    lp.add_argument("--rubric-dir", required=True)
    lp.add_argument("--dev", required=True)
    lp.add_argument("--test", required=True)
    lp.add_argument("--seeds", nargs="+", required=True, help="seed files the cases were built from")
    lp.add_argument("--out", required=True)
    lp.add_argument("--rounds", type=int, default=1)
    sp = sub.add_parser("score", help="judge cases with a fixed rubric (no revision): compare generic texts or rubrics")
    sp.add_argument("--config", required=True)
    sp.add_argument("--domain", required=True, help="file with the two domain slots (domain.md or domain_final.md)")
    sp.add_argument("--generic", help="generic judge text to fill instead of rubrics/generic_parts_v2.md")
    sp.add_argument("--cases", required=True)
    sp.add_argument("--seeds", nargs="+", required=True)
    sp.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out)
    log_dir = out if a.cmd in ("loop", "score") else out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_call_log(log_dir / "llm_calls.jsonl")
    if a.cmd == "build":
        seeds = read_jsonl(a.seeds)[: a.limit or None]
        cases = build(cfg, seeds)
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
        print(json.dumps({"cases": len(cases), "usable": sum(not c.get("skip") for c in cases),
                          "by_kind": Counter(c["kind"] for c in cases if not c.get("skip"))}, indent=1))
    elif a.cmd == "score":
        seeds = {s["id"]: s for f in a.seeds for s in read_jsonl(f)}
        slots = parse_slots(Path(a.domain).read_text())
        generic = re.sub(r"<!--.*?-->", "", Path(a.generic).read_text(), flags=re.DOTALL).strip() if a.generic else generic_v2()
        rubric = fill(generic, {k: slots[k] for k in SLOTS_V2})
        (out / "rubric.md").write_text(rubric)
        v = judge_cases(cfg, read_jsonl(a.cases), seeds, rubric)
        (out / "verdicts.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in v))
        report = score(v)
        (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(json.dumps(report, indent=1))
    else:
        seeds = {s["id"]: s for f in a.seeds for s in read_jsonl(f)}
        slots = parse_slots((Path(a.rubric_dir) / "domain.md").read_text())
        print(json.dumps(loop(cfg, slots, read_jsonl(a.dev), read_jsonl(a.test), seeds, out, a.rounds), indent=1))


if __name__ == "__main__":
    sys.exit(main())
