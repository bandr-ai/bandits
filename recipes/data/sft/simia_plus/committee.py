"""APIGen-MT-style review: three independent reviewers, each checking one aspect, then a vote.

Each reviewer sees the same material (the agent's system prompt, the real seed, the generated conversation) but answers ONE
question and must quote what it relies on. verdict(): "pass" if at most one reviewer fails (majority), and "any_fail" is also
reported so the stricter rule can be compared. This module only judges; regeneration with the reasons is done by the caller.
Like APIGen it votes without executing anything (we have no tool replay), so it can only be as good as the reviewers' reading."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .llm import chat_json, get_llm
from .schema import render

ASPECTS = {
    "consistency": (
        "CONSISTENCY. Is the generated conversation still the SEED's situation, and do all its observations agree with each other? "
        "Fail if a changed observation contradicts an observation that was kept, if something is attached to a place, context or state that "
        "the kept observations say it does not belong to, if a call that established a fact the conversation still relies on was dropped, "
        "or if the seed's identifying facts were replaced. Quote BOTH sides of any contradiction."),
    "contract": (
        "CONTRACT. Does the final assistant message obey every explicit rule in the system prompt: the answer format and required or "
        "conditional fields, the allowed labels and causes, what evidence a conclusion or cause requires, and any tie, default or "
        "ordering rule? Fail only for a rule you can quote from the system prompt that the answer or the calls break; a rule you cannot "
        "find is not a violation. Quote the rule and the offending part."),
    "grounding": (
        "GROUNDING. Is every name, id, URL, number, node, tool-result shape and claim in the generated conversation supported by the seed, "
        "the tool definitions, or an earlier observation, and does each claim in the final answer rest on an observation that can "
        "establish it? Fail for invented specifics or for a claim its cited observation cannot establish. A name already present in the "
        "seed is not invented. Quote the unsupported item."),
}

PROMPT = """<system_prompt_of_the_agent>
{system}
</system_prompt_of_the_agent>

<seed note="the real, reviewed trajectory">
{seed}
</seed>

<generated note="the synthetic trajectory to review">
{generated}
</generated>

<request>
You are one of three independent reviewers. Review ONLY this aspect:
{aspect}

The change from the seed is intentional; do not fail a trace merely for differing from the seed. Return JSON only:
{{"verdict": "pass" | "fail", "problems": [{{"quote": "<exact words>", "why": "<one sentence>"}}]}}
Use "pass" with an empty list when you find nothing you can quote.
</request>"""


def _one(cfg: Config, name: str, seed: dict, trace: dict) -> dict:
    try:
        out = chat_json(get_llm(cfg, "judge"), "You are a strict, precise reviewer. Output JSON only.",
                        PROMPT.format(system=trace.get("system", ""), seed=render(seed, max_obs_chars=2500)[:14000],
                                      generated=render(trace, max_obs_chars=2500)[:14000], aspect=ASPECTS[name]))
    except Exception as e:  # truncated or failed call: recorded as "error", never a pass
        out = {"error": str(e)[:120]}
    ok = isinstance(out, dict) and out.get("verdict") in ("pass", "fail")
    return {"aspect": name, "verdict": out["verdict"] if ok else "error",
            "problems": (out.get("problems") if ok and isinstance(out.get("problems"), list) else [])[:4]}


def review(cfg: Config, seed: dict, trace: dict) -> dict:
    """Run the three reviewers in parallel; a reviewer that errors counts as a fail so a broken call never passes a trace."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        votes = list(ex.map(lambda n: _one(cfg, n, seed, trace), ASPECTS))
    fails = [v for v in votes if v["verdict"] != "pass"]
    return {"votes": votes, "n_fail": len(fails), "majority_pass": len(fails) <= 1, "any_fail": bool(fails),
            "reasons": [p for v in fails for p in v["problems"]][:6]}
