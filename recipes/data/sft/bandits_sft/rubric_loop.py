"""Seed-only rubric induction for the rubric judge.

The generic parts of the rubric (evidence integrity, minor issues, labels) are fixed text in rubrics/generic_parts.md.
The domain part (situations, per-agent expected answers, hard rules, evidence traps) is induced from reviewed seeds
by one model call (the INDUCE prompt in rubrics/induce_rubric_prompt.md) and inserted into the generic text.

v2 (--v2): rubrics/generic_parts_v2.md has two domain slots (consistency map, answer rules), filled by the INDUCE prompt
in rubrics/induce_rubric_prompt_v2.md, which also sees the generic text so it does not repeat it. lint() flags what the
prompt forbids (table rows copied from a system prompt, unverified rows, unknowns, length) without changing the text.

    python -m bandits_sft.rubric_loop induce --config <cfg.json> --out <dir> [--seeds <seeds.jsonl>] [--v2]
"""
import argparse
import json
import re
import sys
from pathlib import Path

from .config import Config, load_config
from .io import read_jsonl
from .llm import get_llm
from .schema import render
from .verify import system_key

RUBRICS = Path(__file__).resolve().parent / "rubrics"  # package data: ships in the wheel


SLOTS_V2 = ("consistency_map", "answer_rules")
MAX_WORDS_V2 = 1500


def prompt_section(name: str, file: str = "induce_rubric_prompt.md") -> str:
    """One prompt (INDUCE or REVISE) from a prompt file in rubrics/."""
    text = (RUBRICS / file).read_text()
    m = re.search(rf"^## {name}\n(.*?)(?=^---$|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"no '## {name}' section in {file}")
    return m.group(1).strip()


def generic_v2() -> str:
    """The fixed v2 judge text with comments stripped, as both the judge and the induction prompt see it."""
    return re.sub(r"<!--.*?-->", "", (RUBRICS / "generic_parts_v2.md").read_text(), flags=re.DOTALL).strip()


def parse_slots(text: str) -> dict[str, str]:
    """The two <slot>...</slot> blocks of a v2 induction answer; a missing block is an error."""
    out = {}
    for k in SLOTS_V2:
        m = re.search(rf"<{k}>(.*?)</{k}>", text, flags=re.DOTALL)
        if not m or not m.group(1).strip():
            raise ValueError(f"induction answer has no <{k}> block")
        out[k] = m.group(1).strip()
    return out


def assemble_v2(slots: dict[str, str]) -> str:
    return fill(generic_v2(), {k: slots[k] for k in SLOTS_V2})


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def lint(slots: dict[str, str], system_prompts: list[str], n: int = 8) -> dict:
    """Problems the INDUCE prompt forbids, found mechanically. Hard rules are quoted on purpose, so only table rows are
    checked for copied prompt text (n words in a row that also appear in a system prompt)."""
    grams = set()
    for p in system_prompts:
        w = _words(p)
        grams.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    copied = []
    for line in slots.get("answer_rules", "").splitlines():
        if not line.lstrip().startswith("|") or set(line.strip()) <= set("|-: "):
            continue
        w = _words(line)
        if any(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1)):
            copied.append(line.strip()[:160])
    text = "\n".join(slots.values())
    words = len(_words(text))
    return {"copied_table_rows": copied,
            "unverified_mentions": len(re.findall(r"unverified", text, flags=re.IGNORECASE)),
            "unknowns_sections": len(re.findall(r"^\W*unknowns?\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "words": words, "over_length": words > MAX_WORDS_V2}


def induce_inputs(seeds: list[dict], max_obs_chars: int = 1500) -> dict[str, str]:
    """System prompts once each (with ids), tool definitions once, seeds rendered against their prompt id."""
    prompts: dict[str, str] = {}
    tools: dict[str, dict] = {}
    rendered = []
    for s in seeds:
        pid = system_key(s.get("system", ""))[:8]
        prompts.setdefault(pid, s.get("system", ""))
        for t in s.get("tools") or []:
            tools.setdefault(t["name"], t)
        rendered.append(f'<seed id="{s["id"]}" prompt="{pid}">\n{render(s, max_obs_chars=max_obs_chars)}\n</seed>')
    return {"system_prompts": "\n\n".join(f'<prompt id="{k}">\n{v}\n</prompt>' for k, v in prompts.items()),
            "tool_definitions": json.dumps(list(tools.values()), ensure_ascii=False, indent=1),
            "seeds": "\n\n".join(rendered)}


def fill(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def assemble(domain: str) -> str:
    return (RUBRICS / "generic_parts.md").read_text().replace("{domain}", domain.strip())


def induce(cfg: Config, seeds: list[dict], v2: bool = False) -> str:
    """The domain part, as markdown, from one model call (v2: the two slot blocks)."""
    if v2:
        user = fill(prompt_section("INDUCE", "induce_rubric_prompt_v2.md"),
                    {"generic_judge": generic_v2(), **induce_inputs(seeds)})
    else:
        user = fill(prompt_section("INDUCE"), induce_inputs(seeds))
    system = ("You write grading rubrics from examples. Use only the supplied material, cite sources, never invent rules. "
              "Output markdown only.")
    out = get_llm(cfg, "rubric_induce").chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = (out.get("content") or "").strip()
    text = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", text)
    if not text:
        raise RuntimeError("induction returned no text")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bandits_sft.rubric_loop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("induce", help="write the domain part once from the seeds and assemble the rubric")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", help="seed file (default: the config's seeds_path)")
    p.add_argument("--v2", action="store_true", help="two-slot rubric (generic_parts_v2.md, induce_rubric_prompt_v2.md)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = read_jsonl(a.seeds or cfg.seeds_path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .llm import configure_call_log
    configure_call_log(out / "llm_calls.jsonl")
    domain = induce(cfg, seeds, v2=a.v2)
    (out / "domain.md").write_text(domain + "\n")
    report = None
    if a.v2:
        slots = parse_slots(domain)
        (out / "rubric.md").write_text(assemble_v2(slots))
        report = lint(slots, list({s.get("system", "") for s in seeds}))
        (out / "lint.json").write_text(json.dumps(report, indent=1))
    else:
        (out / "rubric.md").write_text(assemble(domain))
    calls = [json.loads(line) for line in (out / "llm_calls.jsonl").open()] if (out / "llm_calls.jsonl").exists() else []
    usage = [c.get("usage") or {} for c in calls if c.get("ok")]
    print(json.dumps({"seeds": len(seeds), "rubric": str(out / "rubric.md"), "calls": len(calls),
                      "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                      "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                      "cost_usd": sum(c.get("cost") or 0 for c in calls) or None, "lint": report}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
