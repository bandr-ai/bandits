"""Checks on generated traces.

rule_check     Simia's post-processing as checks (fix_arguments / tool_correct / validate_conversation),
               generalised: parseable calls, known tools, required args and types, results follow calls,
               ends with an assistant reply. Always on: this is the baseline.
provenance     SAP (2609.06124): every ID-like argument value must appear earlier in what the agent saw.
judge          Proxy-State Eval (2602.16246) + Adaption checklists: LLM audit against the spec, with the
               reconstructed final state diffed in code against the expected one.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .config import Config
from .llm import chat_json, get_llm
from .prompts import BLIND_ANSWER, DEFAULT_CHECKLIST, JUDGE, JUDGE_CHECKS, JUDGE_EXPANSION
from .schema import render
from .state import state_match

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_IDLIKE = re.compile(r"^(?=.*\d)[\w\-#.@:/]{3,}$|^[^@\s]+@[^@\s]+\.\w+$")


def rule_check(trace: dict, min_assistant_turns: int, arg_defaults: dict | None = None) -> list[str]:
    issues = []
    if trace.get("meta", {}).get("simia_deleted"):
        issues.append("leaked tool markup (Simia should_delete_conversation)")
    msgs = trace["messages"]
    tools = {t["name"]: t for t in trace.get("tools", [])}
    if not msgs or msgs[0]["role"] != "user":
        issues.append("does not start with a user message")
    if not msgs or msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls") or not msgs[-1]["content"].strip():
        issues.append("does not end with an assistant reply")
    call_ids = [c["id"] for m in msgs for c in m.get("tool_calls", [])]
    if len(call_ids) != len(set(call_ids)):
        issues.append("duplicate tool call id")
    # each result must answer a call of the latest assistant turn, before the next user/assistant turn
    answered, open_ids, orphans = set(), set(), 0
    for m in msgs:
        if m["role"] == "tool":
            if m["tool_call_id"] in open_ids:
                open_ids.discard(m["tool_call_id"])
                answered.add(m["tool_call_id"])
            else:
                orphans += 1
        else:
            open_ids = {c["id"] for c in m.get("tool_calls", [])} if m["role"] == "assistant" else set()
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
            args = {**(arg_defaults or {}).get(c["name"], {}), **c["arguments"]}  # the harness fills defaults first
            issues += [f"{c['name']}{e}" for e in schema_errors(args, spec.get("parameters", {}))]
    if sum(1 for m in msgs if m["role"] == "assistant") < min_assistant_turns:
        issues.append("too few assistant turns")
    if any(n >= 3 for n in seen.values()):
        issues.append("same call repeated 3+ times (loop)")
    return issues


def schema_errors(value, schema: dict, path: str = "") -> list[str]:
    """The JSON Schema subset tool and answer schemas use: type, enum, required, properties,
    additionalProperties: false, items, minimum, maximum."""
    errs = []
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    if types:
        ok = any(value is None if x == "null" else
                 isinstance(value, _TYPES[x]) and not (x in ("integer", "number") and isinstance(value, bool))
                 for x in types if x == "null" or x in _TYPES)
        if not ok:
            return [f"{path or ' value'} should be {'/'.join(types)}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path or ' value'}={value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errs += [f"{path} missing required argument {k}" if not path else f"{path} missing {k}"
                 for k in schema.get("required", []) if k not in value]
        for k, v in value.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path} unknown argument {k}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            errs += schema_errors(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path} below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path} above {schema['maximum']}")
    return errs


def system_key(system: str) -> str:
    return hashlib.sha256((system or "").encode()).hexdigest()


def final_answer(trace: dict):
    """The final assistant reply parsed as JSON, or None when it is not exactly one JSON value."""
    last = trace["messages"][-1] if trace["messages"] else {}
    if last.get("role") != "assistant":
        return None
    try:
        return json.loads((last.get("content") or "").strip())
    except json.JSONDecodeError:
        return None


def tool_result_check(trace: dict, shapes: dict, prefix: str | None) -> list[str]:
    """Tool results must look like the real tool's output: the harness prefix with a running index, then a JSON
    object whose top-level keys are one of that tool's real shapes (from the harness code)."""
    if not shapes and not prefix:
        return []
    issues, n = [], 0
    for m in trace["messages"]:
        if m["role"] != "tool":
            continue
        n += 1
        body = m["content"]
        if prefix:
            mt = re.match(prefix, body)
            if not mt:
                issues.append(f"tool result {n} ({m.get('name')}) lacks the harness prefix")
            elif mt.groups() and mt.group(1) != str(n):
                issues.append(f"tool result {n} ({m.get('name')}) has index {mt.group(1)}")
            body = body[mt.end():] if mt else body
        if m.get("name") not in shapes:  # unlisted tools are not checked; "*" adds shapes (e.g. errors) to listed ones
            continue
        allowed = [set(k) for k in shapes[m.get("name")] + shapes.get("*", [])]
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            issues.append(f"tool result {n} ({m.get('name')}) is not JSON")
            continue
        if not isinstance(obj, dict) or set(obj) not in allowed:
            issues.append(f"tool result {n} ({m.get('name')}) has a shape the real tool never returns: "
                          f"{sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    return issues


def answer_check(trace: dict, schemas: dict[str, dict]) -> list[str]:
    """The final reply must be exactly the JSON object the system prompt's output contract asks for."""
    schema = schemas.get(system_key(trace.get("system", "")))
    if schema is None:
        return []
    ans = final_answer(trace)
    if ans is None:
        return ["final answer is not a bare JSON value"]
    return [f"answer{e}" for e in schema_errors(ans, schema)]


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


EXPANSION_CHECKS = ("only_declared_change", "dependencies_updated", "tool_contracts_respected")
FULL = 10**7  # judge and blind check see every tool result in full: an audit must not miss truncated evidence


def _norm(x) -> str:
    """Lowercase, no backslashes (JSON inside tool results is quoted with escapes the judge drops), single spaces."""
    return " ".join(str(x).replace("\\", "").split()).lower()


def _trigrams(text: str) -> set[tuple]:
    words = re.findall(r"\w+", text)
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def _grounded(item: dict, corpus: str, corpus_grams: set[tuple]) -> tuple[bool, list[str]]:
    """A verdict counts as grounded when it cites at least one excerpt and every excerpt is found in the corpus:
    each piece between "..." elisions verbatim, or with at least 80% of its word trigrams present (tolerates wrapping
    quotes and a skipped key, not invented content). Pieces under 3 words only count when nothing longer is quoted."""
    ev = item.get("evidence") if isinstance(item, dict) else None
    excerpts = [e.get("excerpt", "") for e in ev or [] if isinstance(e, dict)]
    missing = []
    def piece_found(text: str) -> bool:
        grams = _trigrams(text)
        return text in corpus or bool(grams) and len(grams & corpus_grams) / len(grams) >= 0.8

    for x in excerpts:
        pieces = [p for p in (_norm(q) for q in str(x).replace("…", "...").split("...")) if re.search(r"\w", p)]
        # "..." marks an elision: each piece must be found; one-word scraps between elisions carry no evidence
        checked = [p for p in pieces if len(re.findall(r"\w+", p)) >= 3] or pieces
        if not checked or not all(piece_found(p) for p in checked):
            missing.append(x)
    return bool(excerpts) and not missing, missing


def judge_expansion(cfg: Config, trace: dict, expansion: dict) -> dict:
    """Judge a seed-expansion trace against its seed and assigned variation (the seed is a reference, not an answer key).

    Each check gets a verdict with quoted evidence. Code verifies the quotes: a "pass" whose excerpts are missing or do
    not occur in the generated trace or the seed does not count as a pass. The result also carries the flat fields
    verify() gates on (task_success, obs_consistent, policy_followed, the fidelity checks, checklist_failed)."""
    checklist = cfg.checklist or DEFAULT_CHECKLIST
    case = expansion.get("case")
    case_text = (f'<case note="the generator\'s description of the changed situation; the agent never sees it">\n'
                 f"{json.dumps(case, ensure_ascii=False, indent=1)}\n</case>\n" if case else "")
    out = chat_json(get_llm(cfg, "judge"), "You are a strict, precise auditor. Output JSON only.", JUDGE_EXPANSION.format(
        harness=expansion.get("harness") or "(none)", system=trace.get("system", ""),
        seed=render(expansion["seed"], max_obs_chars=FULL), variation=json.dumps(expansion["variation"], ensure_ascii=False, indent=1),
        case=case_text, transcript=render(trace, max_obs_chars=FULL), checklist="\n".join(f"- {c}" for c in checklist)))
    if not isinstance(out, dict):
        raise TypeError("judge returned non-object JSON")
    corpus = _norm(render(trace, max_obs_chars=FULL)) + "\n" + _norm(render(expansion["seed"], max_obs_chars=FULL))
    corpus_grams = _trigrams(corpus)
    checks = out.get("checks") if isinstance(out.get("checks"), dict) else {}
    unverified: dict[str, list[str]] = {}

    def passed(name: str, item) -> bool:
        if not isinstance(item, dict) or item.get("verdict") != "pass":
            return False
        ok, missing = _grounded(item, corpus, corpus_grams)
        if not ok:
            unverified[name] = missing or ["(no excerpt)"]
        return ok

    required = [n for n in JUDGE_CHECKS if case or n != "case_consistent"]
    verdicts = {name: passed(name, checks.get(name)) for name in required}
    out.update({k: verdicts[k] for k in EXPANSION_CHECKS}, task_success=verdicts["task_success"],
               obs_consistent=verdicts["observations_consistent"], policy_followed=verdicts["policy_followed"],
               case_consistent=verdicts.get("case_consistent", True))
    out["turn_issues"] = turn_issues(trace, out.get("turns"))
    out["bad_steps"] = sorted({int(i.split()[1]) for i in out["turn_issues"] if i.startswith("turn ") and i.split()[1].isdigit()})
    out["checklist_failed"] = [str(c.get("item")) for c in out.get("checklist") or [] if isinstance(c, dict)
                               and c.get("verdict") in ("fail", "cannot_determine")]
    if not isinstance(out.get("checklist"), list):
        out["checklist_failed"] = ["(checklist missing)"]
    out["unverified_excerpts"] = unverified
    return out


def turn_issues(trace: dict, turns) -> list[str]:
    """Every assistant message needs a "justified" entry citing at least one earlier message (0 <= n < its number) or
    "system" (the system policy alone), and nothing at or after itself (no hindsight)."""
    if not isinstance(turns, list):
        return ["turns missing"]
    by_msg: dict[int, dict] = {}
    for e in turns:
        try:
            by_msg[int(e.get("message"))] = e
        except (AttributeError, TypeError, ValueError):
            continue
    issues = []
    for i, m in enumerate(trace["messages"]):
        if m["role"] != "assistant":
            continue
        e = by_msg.get(i)
        if e is None:
            issues.append(f"turn {i} not judged")
            continue
        if e.get("verdict") != "justified":
            issues.append(f"turn {i} {e.get('verdict')}: {e.get('reason', '')}"[:300])
        refs = e.get("relies_on")
        if not isinstance(refs, list) or not refs:
            issues.append(f"turn {i} cites nothing (relies_on must list earlier messages or \"system\")")
            continue
        bad = [r for r in refs if r != "system" and not (str(r).isdigit() and int(r) < i)]
        if bad:
            issues.append(f"turn {i} cites messages not before it: {bad}")
    return issues


def _categorical_fields(schema: dict | None, fallback: list[str]) -> list[str]:
    props = (schema or {}).get("properties") or {}
    picked = [k for k, s in props.items() if isinstance(s, dict) and ("enum" in s or s.get("type") == "boolean")]
    return picked or list(fallback)


def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower() if a is not None and b is not None else a is None and b is None


def blind_check(cfg: Config, trace: dict) -> dict:
    """A separate call answers from the system prompt and the conversation without its final answer; it never sees the
    seed, variation, case or verdicts. Compares the contract's categorical fields (enums, booleans), not wording or
    confidence. Agreement corroborates; it does not prove the observations are valid."""
    final = final_answer(trace)
    msgs = trace["messages"][:-1]
    out = chat_json(get_llm(cfg, "blind"), "You are the agent described in the system prompt. Output JSON only.",
                    BLIND_ANSWER.format(system=trace.get("system", ""), transcript=render(msgs, max_obs_chars=FULL)))
    if not isinstance(out, dict):
        return {"agree": False, "insufficient": False, "answer": out, "diffs": ["non-object answer"]}
    if out.get("insufficient_evidence") is True:
        return {"agree": False, "insufficient": True, "answer": out, "diffs": []}
    schema = load_answer_schemas(cfg.answer_schemas_path).get(system_key(trace.get("system", "")))
    fields = _categorical_fields(schema, cfg.outcome_fields)
    final = final if isinstance(final, dict) else {}
    diffs = [f"{k}: generated {final.get(k)!r} vs blind {out.get(k)!r}" for k in fields if not _same(final.get(k), out.get(k))]
    return {"agree": not diffs, "insufficient": False, "fields": fields, "answer": out, "diffs": diffs}


def ngrams(text: str, n: int = 8) -> set[tuple]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def user_text(trace: dict) -> str:
    return " ".join(m["content"] for m in trace["messages"] if m["role"] == "user")


@lru_cache(maxsize=4)
def load_answer_schemas(path: str | None) -> dict[str, dict]:
    return json.loads(Path(path).read_text()) if path else {}


def verify(cfg: Config, trace: dict, eval_ngrams: set[tuple], expansion: dict | None = None) -> dict:
    """expansion: {"seed", "variation", "harness"} for seed-expansion traces; the judge then checks fidelity too."""
    f = cfg.features
    spec = trace["meta"].get("spec")
    v: dict = {"rule_issues": [*trace["meta"].get("format_issues", []),
                               *rule_check(trace, cfg.min_assistant_turns, cfg.tool_arg_defaults),
                               *tool_result_check(trace, cfg.tool_result_shapes, cfg.tool_result_prefix),
                               *answer_check(trace, load_answer_schemas(cfg.answer_schemas_path))]}
    v["contaminated"] = bool(eval_ngrams and ngrams(user_text(trace)) & eval_ngrams)
    if f.provenance:
        v["ungrounded_args"] = provenance_check(trace, spec)
    keep = not v["rule_issues"] and not v["contaminated"] and not v.get("ungrounded_args")
    if f.judge and keep:
        j = judge_expansion(cfg, trace, expansion) if expansion else judge(cfg, trace, spec)
        v["judge"] = j
        # fail closed: a missing field is a failed check, not a pass
        consistent = (j.get("obs_consistent") is True and j.get("policy_followed") is True
                      and j.get("checklist_failed") == [] and j.get("hallucinations") == []
                      and all(j.get(k) is True for k in (EXPANSION_CHECKS + ("case_consistent",) if expansion else ()))
                      and (not expansion or j.get("turn_issues") == []))
        success = bool(j.get("task_success"))
        keep = consistent and (success or cfg.keep_failures)
        trace["meta"]["bad_steps"] = j.get("bad_steps", []) if not success else []
        if keep and expansion and f.blind_check:
            v["blind"] = blind_check(cfg, trace)
            if not v["blind"]["agree"]:
                keep, v["unresolved"] = False, True
    v["kept"] = keep
    trace["meta"]["verify"] = v
    return trace
