"""Checks on generated traces.

rule_check     Simia's post-processing as checks (fix_arguments / tool_correct / validate_conversation),
               generalised: parseable calls, known tools, required args and types, results follow calls,
               ends with an assistant reply. Always on: this is the baseline.
provenance     SAP (2609.06124): every ID-like argument value must appear earlier in what the agent saw.
judge          Proxy-State Eval (2602.16246) + Adaption checklists: LLM audit against the spec, with the
               reconstructed final state diffed in code against the expected one.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from .config import Config
from .llm import chat_json, get_llm
from .prompts import DEFAULT_CHECKLIST, JUDGE
from .schema import render
from .state import state_match

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_IDLIKE = re.compile(r"^(?=.*\d)[\w\-#.@:/]{3,}$|^[^@\s]+@[^@\s]+\.\w+$")


def rule_check(trace: dict, min_assistant_turns: int) -> list[str]:
    issues = []
    if trace.get("meta", {}).get("simia_deleted"):
        issues.append("leaked tool markup (Simia should_delete_conversation)")
    msgs = trace["messages"]
    tools = {t["name"]: t for t in trace.get("tools", [])}
    if not msgs or msgs[0]["role"] != "user":
        issues.append("does not start with a user message")
    if not msgs or msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls") or not msgs[-1]["content"].strip():
        issues.append("does not end with an assistant reply")
    answered = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
    call_ids = {c["id"] for m in msgs for c in m.get("tool_calls", [])}
    orphans = sum(1 for m in msgs if m["role"] == "tool" and m["tool_call_id"] not in call_ids)
    if orphans:
        issues.append(f"{orphans} tool result(s) without a matching call")
    if any(m["role"] == "assistant" and not m.get("tool_calls") and not (m.get("content") or "").strip() for m in msgs):
        issues.append("empty assistant turn (no text, no tool call)")
    seen = Counter()
    for m in msgs:
        if "<tool_" in (m.get("content") or "") or "\n[{" in (m.get("content") or ""):
            issues.append("leaked tool markup in message text")
        for c in m.get("tool_calls", []):
            seen[(c["name"], json.dumps(c["arguments"], sort_keys=True))] += 1
            if c["id"] not in answered:
                issues.append(f"call {c['name']} has no result")
            if "_raw" in c["arguments"] or "_value" in c["arguments"]:
                issues.append(f"call {c['name']} has unparseable arguments")
                continue
            spec = tools.get(c["name"])
            if spec is None:
                issues.append(f"unknown tool {c['name']}")
                continue
            params = spec.get("parameters", {})
            props = params.get("properties", {})
            for req in params.get("required", []):
                if req not in c["arguments"]:
                    issues.append(f"{c['name']} missing required argument {req}")
            for k, v in c["arguments"].items():
                if k not in props:
                    if params.get("additionalProperties") is False:
                        issues.append(f"{c['name']} unknown argument {k}")
                    continue
                t = props[k].get("type")
                py = _TYPES.get(t) if isinstance(t, str) else None
                if (py and not isinstance(v, py)) or (t in ("integer", "number") and isinstance(v, bool)):
                    issues.append(f"{c['name']}.{k} should be {t}")
    if sum(1 for m in msgs if m["role"] == "assistant") < min_assistant_turns:
        issues.append("too few assistant turns")
    if any(n >= 3 for n in seen.values()):
        issues.append("same call repeated 3+ times (loop)")
    return issues


def _values(x) -> list:
    if isinstance(x, dict):
        return [v for val in x.values() for v in _values(val)]
    if isinstance(x, list):
        return [v for val in x for v in _values(val)]
    return [x]


def provenance_check(trace: dict, spec: dict | None) -> list[str]:
    """ID-like argument values that never appeared in the system prompt, user turns, tool results or spec facts."""
    seen = (trace.get("system") or "") + "\n" + json.dumps((spec or {}).get("user_facts") or {}, ensure_ascii=False)
    ungrounded = []
    for m in trace["messages"]:
        if m["role"] in ("user", "tool"):
            seen += "\n" + m["content"]
        for c in m.get("tool_calls", []):
            for v in _values(c["arguments"]):
                s = str(v)
                if isinstance(v, (str, int)) and not isinstance(v, bool) and _IDLIKE.match(s) and s.lower() not in seen.lower():
                    ungrounded.append(f"{c['name']}: {s}")
    return ungrounded


def judge(cfg: Config, trace: dict, spec: dict | None) -> dict:
    checklist = cfg.checklist or DEFAULT_CHECKLIST
    out = chat_json(get_llm(cfg, "judge"), "You are a strict, precise auditor. Output JSON only.", JUDGE.format(
        system=trace.get("system", ""), spec_json=json.dumps(spec or {"note": "no scenario spec; judge on the conversation itself"},
                                                             ensure_ascii=False, indent=1),
        transcript=render(trace), checklist="\n".join(f"- {c}" for c in checklist)))
    if not isinstance(out, dict):
        raise TypeError("judge returned non-object JSON")
    if spec and spec.get("expected_final_state"):
        final = trace["meta"].get("final_state") or out.get("final_state") or {}
        out["state_match"] = state_match(spec["expected_final_state"], final)
    return out


def ngrams(text: str, n: int = 8) -> set[tuple]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def user_text(trace: dict) -> str:
    return " ".join(m["content"] for m in trace["messages"] if m["role"] == "user")


def verify(cfg: Config, trace: dict, eval_ngrams: set[tuple]) -> dict:
    f = cfg.features
    spec = trace["meta"].get("spec")
    v: dict = {"rule_issues": rule_check(trace, cfg.min_assistant_turns)}
    v["contaminated"] = bool(eval_ngrams and ngrams(user_text(trace)) & eval_ngrams)
    if f.provenance:
        v["ungrounded_args"] = provenance_check(trace, spec)
    keep = not v["rule_issues"] and not v["contaminated"] and not v.get("ungrounded_args")
    if f.judge and keep:
        j = judge(cfg, trace, spec)
        v["judge"] = j
        consistent = bool(j.get("obs_consistent", True)) and not j.get("checklist_failed") and not j.get("hallucinations")
        success = bool(j.get("task_success"))
        keep = consistent and (success or cfg.keep_failures)
        trace["meta"]["bad_steps"] = j.get("bad_steps", []) if not success else []
    v["kept"] = keep
    trace["meta"]["verify"] = v
    return trace
