"""Checks on generated traces.

rule_check     baseline's post-processing as checks (fix_arguments / tool_correct / validate_conversation),
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
from .llm import BudgetExhausted, chat_json, get_llm
from .prompts import DEFAULT_CHECKLIST, JUDGE, JUDGE_CHECKS, JUDGE_EXPANSION
from .schema import render
from .state import state_match

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_IDLIKE = re.compile(r"^(?=.*\d)[\w\-#.@:/]{3,}$|^[^@\s]+@[^@\s]+\.\w+$")


def rule_check(trace: dict, min_assistant_turns: int, arg_defaults: dict | None = None,
               optional_args: dict | None = None, loose_type_args: dict | None = None) -> list[str]:
    issues = []
    if trace.get("meta", {}).get("baseline_deleted"):
        issues.append("leaked tool markup (baseline should_delete_conversation)")
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
            skip = {f" missing required argument {a}" for a in (optional_args or {}).get(c["name"], ())}
            loose = tuple(f".{a} should be " for a in (loose_type_args or {}).get(c["name"], ()))
            issues += [f"{c['name']}{e}" for e in schema_errors(args, spec.get("parameters", {}))
                       if e not in skip and not e.startswith(loose)]
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
            if v is None and k not in schema.get("required", []):
                continue  # null for an optional argument means omitted; real seeds pass it this way
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


def observed_optional_args(seeds: list[dict]) -> dict[str, list[str]]:
    """Schema-required arguments that reviewed seeds themselves omit in real calls: optional in practice (a runtime
    default exists). Read from the traces only; a schema alone cannot say which required fields are really required."""
    out: dict[str, set] = {}
    for s in seeds:
        req = {t["name"]: (t.get("parameters") or {}).get("required") or [] for t in s.get("tools", [])}
        for m in s.get("messages", []):
            for c in m.get("tool_calls") or []:
                out.setdefault(c["name"], set()).update(a for a in req.get(c["name"], []) if a not in c["arguments"])
    return {k: sorted(v) for k, v in out.items() if v}


def observed_loose_type_args(seeds: list[dict]) -> dict[str, list[str]]:
    """Arguments whose type in reviewed seeds' real calls differs from the schema (an int or object where it says
    string): the harness accepts them, so generated calls passing the same are not schema errors. Read from the traces."""
    out: dict[str, set] = {}
    for s in seeds:
        props = {t["name"]: (t.get("parameters") or {}).get("properties") or {} for t in s.get("tools", [])}
        for m in s.get("messages", []):
            for c in m.get("tool_calls") or []:
                p = props.get(c["name"], {})
                bad = {k for k, v in c["arguments"].items() if v is not None and k in p and schema_errors(v, p[k])}
                if bad:
                    out.setdefault(c["name"], set()).update(bad)
    return {k: sorted(v) for k, v in out.items()}


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


_JSON_DEMAND = re.compile(r"\b(return|respond|reply|output|answer|emit|produce|write)\b[^.\n]{0,80}\bJSON\b|\bJSON\b[^.\n]{0,40}\b(only|object)\b", re.IGNORECASE)


def requires_json_answer(system: str) -> bool:
    """True when the supplied system prompt itself says the answer is JSON. Nothing is inferred from seeds here."""
    return bool(_JSON_DEMAND.search(system or ""))


def explicit_answer_check(trace: dict, seed: dict | None) -> list[str]:
    """Hard check only for what the supplied system prompt states outright: a JSON answer must parse as bare JSON.
    Seed-expansion only, and only when the reviewed seed's own final reply is bare JSON too, so a loosely worded
    prompt ("output the call as JSON") cannot reject answers its reviewed examples do not give as JSON."""
    if seed is None or final_answer(seed) is None or not requires_json_answer(trace.get("system", "")):
        return []
    last = trace["messages"][-1] if trace["messages"] else {}
    text = (last.get("content") or "").strip() if last.get("role") == "assistant" else ""
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return [f"final answer is not valid JSON, which the system prompt requires ({e.msg})"]
    return []


def extra_answer_keys(trace: dict, seed: dict | None) -> list[str]:
    """Hard reject: the final JSON answer carries a key the reviewed seed's own answer never has (a contract violation,
    not a stylistic difference). Missing keys are not rejected here."""
    a, s = final_answer(trace), (final_answer(seed) if seed else None)
    if isinstance(a, dict) and isinstance(s, dict) and set(a) - set(s):
        return [f"final answer has key(s) the seed's answer does not: {sorted(set(a) - set(s))}"]
    return []


def _json_block(text: str):
    """(ok, error) for the first JSON object/array in text, or None when text has none."""
    i = min((k for k in (text.find("{"), text.find("[")) if k >= 0), default=-1)
    if i < 0:
        return None
    try:
        json.JSONDecoder().raw_decode(text[i:])
        return True, None
    except json.JSONDecodeError as e:
        return False, f"{e.msg} at char {i + e.pos}"


def input_json_flags(trace: dict, seed: dict | None) -> list[str]:
    """Soft: a user turn whose JSON block parsed in the seed but no longer parses (or is gone) in the trace, i.e. likely
    damaged while copying. Never gating: a variation may change an input's format on purpose; the judge decides."""
    if not seed:
        return []
    users = lambda t: [(i, m.get("content")) for i, m in enumerate(t["messages"]) if m["role"] == "user" and isinstance(m.get("content"), str)]
    out = []
    for (_, s), (i, t) in zip(users(seed), users(trace)):
        if _json_block(s) != (True, None):
            continue  # only when the seed's own block is valid JSON
        got = _json_block(t)
        if got is None:
            out.append(f"message {i}: the seed's JSON block in this user turn is missing")
        elif not got[0]:
            out.append(f"message {i}: JSON block no longer parses ({got[1]}); the seed's does")
    return out


def answer_soft_flags(trace: dict, seed: dict | None) -> list[str]:
    """Differences from the seed's answer that are logged for diagnosis, never gating: seeds are examples, not a schema."""
    a, s = final_answer(trace), final_answer(seed) if seed else None
    if isinstance(a, dict) and isinstance(s, dict) and set(a) != set(s):
        return [f"answer keys differ from seed: +{sorted(set(a) - set(s))} -{sorted(set(s) - set(a))}"]
    return []


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
FULL = 10**7  # the judge sees every tool result in full: an audit must not miss truncated evidence


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


def assistant_delta(trace: dict, seed: dict) -> bool:
    """True when at least one assistant message (text or tool call) differs from the seed's assistant messages."""
    def sig(t):
        return [((m.get("content") or "").strip(), json.dumps([(c.get("name"), c.get("arguments")) for c in m.get("tool_calls") or []], sort_keys=True))
                for m in t["messages"] if m["role"] == "assistant"]
    return sig(trace) != sig(seed)


_COSMETIC = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                           "\u2011": "-", "\u00a0": " "})


def _norm_text(text: str) -> str:
    """Cosmetic normalization: typographic quotes and dashes to ASCII, single vs double quotes alike, whitespace collapsed."""
    return re.sub(r"\s+", " ", (text or "").translate(_COSMETIC).replace("'", '"')).strip()


def _inputs(t: dict) -> list:
    """Everything the agent was given or did to get evidence: user turns, tool calls and tool results, in order."""
    out = []
    for m in t["messages"]:
        if m["role"] in ("user", "tool"):
            out.append((m["role"], _norm_text(m.get("content") or "")))
        for c in m.get("tool_calls") or []:
            out.append(("call", c.get("name"), _norm_text(json.dumps(c.get("arguments"), sort_keys=True))))
    return out


def _assistant_text(t: dict) -> str:
    return " || ".join(_norm_text((m.get("content") or "") + json.dumps([(c.get("name"), c.get("arguments"))
                                                                         for c in m.get("tool_calls") or []], sort_keys=True))
                       for m in t["messages"] if m["role"] == "assistant")


def change_gate(trace: dict, seed: dict, min_assistant_change: float) -> list[str]:
    """Issues that make a variation worthless whatever its domain: nothing it was given changed (a no-op, or only
    quoting/whitespace changed), or its assistant side is near-identical to the seed's. Empty list = passes."""
    import difflib
    issues = []
    if _inputs(trace) == _inputs(seed):
        issues.append("no_input_change: inputs equal the seed's after cosmetic normalization (the change was not applied)")
    if min_assistant_change > 0:
        a, b = _assistant_text(trace), _assistant_text(seed)
        sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
        if 1 - sim < min_assistant_change:
            issues.append(f"assistant_unchanged: assistant side {sim:.3f} similar to the seed's")
    return issues


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
        observed_contracts=expansion.get("observed_contracts") or "(none)", system=trace.get("system", ""),
        seed=render(expansion["seed"], max_obs_chars=FULL), variation=json.dumps(expansion["variation"], ensure_ascii=False, indent=1),
        case=case_text, transcript=render(trace, max_obs_chars=FULL), checklist="\n".join(f"- {c}" for c in checklist)))
    if not isinstance(out, dict):
        raise TypeError("judge returned non-object JSON")
    corpus = _norm(render(trace, max_obs_chars=FULL)) + "\n" + _norm(render(expansion["seed"], max_obs_chars=FULL))
    corpus_grams = _trigrams(corpus)
    checks = out.get("checks") if isinstance(out.get("checks"), dict) else {}
    unverified: dict[str, list[str]] = {}

    # "not_applicable" is accepted only where code can see the check is structurally empty: tool-result checks when the
    # conversation has no tool results. Never for correctness or policy: those fail closed.
    has_tools = any(m.get("role") == "tool" for m in trace["messages"])
    structural_na = [] if has_tools else ["tool_contracts_respected", "observations_consistent"]

    def passed(name: str, item) -> bool:
        if isinstance(item, dict) and item.get("verdict") == "not_applicable" and name in structural_na:
            return True
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
    out["structural_na"] = structural_na
    return out


def judge_flags(j) -> list[str]:
    """What a judge result criticised, whether or not the judge gates `kept`. Feeds the analyzer's diagnosis and the
    choice of best version. `not_applicable` and `cannot_determine` verdicts are not criticism; a judge error gives none."""
    if not isinstance(j, dict) or j.get("error"):
        return []
    flags = [f"check:{n}" for n, c in (j.get("checks") or {}).items() if isinstance(c, dict) and c.get("verdict") == "fail"]
    ok_na = set(j.get("structural_na") or []) | {"case_consistent"}  # a judge cannot declare correctness or policy not applicable
    flags += [f"check:{n}:not_applicable" for n, c in (j.get("checks") or {}).items()
              if isinstance(c, dict) and c.get("verdict") == "not_applicable" and n not in ok_na]
    flags += [f"checklist:{c}"[:80] for c in j.get("checklist_failed") or [] if "cannot_determine" not in str(c)][:3]
    if j.get("hallucinations"):
        flags.append("hallucinations")
    if j.get("turn_issues"):
        flags.append("turn_issues")
    if j.get("task_success") is False and not any(f == "check:task_success" for f in flags):
        flags.append("task_success")
    return flags


def malformed(j, expansion: dict | None) -> list[str]:
    """Required parts missing from a judge response: an evaluation error (retried once), not a verdict."""
    if not isinstance(j, dict):
        return ["not an object"]
    if not expansion:
        return []  # the plain judge has no required structure beyond its flat fields, which fail closed
    checks = j.get("checks") if isinstance(j.get("checks"), dict) else {}
    required = [k for k in JUDGE_CHECKS if expansion.get("case") or k != "case_consistent"]
    out = [f"check {k}" for k in required if not isinstance(checks.get(k), dict)]
    if not isinstance(j.get("turns"), list):
        out.append("turns")
    if not isinstance(j.get("checklist"), list):
        out.append("checklist")
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
        bad = [r for r in refs if r != "system" and not (_msg_no(r) is not None and 0 <= _msg_no(r) < i)]
        if bad:
            issues.append(f"turn {i} cites messages not before it: {bad}")
    return issues


def _msg_no(r) -> int | None:
    """A cited message number: an int (or integral float) or a digit string; anything else is not a message."""
    if isinstance(r, bool):
        return None
    if isinstance(r, int) or (isinstance(r, float) and r.is_integer()):
        return int(r)
    return int(r) if isinstance(r, str) and r.strip().isdigit() else None


def ngrams(text: str, n: int = 8) -> set[tuple]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def user_text(trace: dict) -> str:
    return " ".join(m["content"] for m in trace["messages"] if m["role"] == "user")


def boilerplate(traces: list[dict]) -> frozenset:
    """8-grams shared by many traces' user text (input templates): ignored when testing for copies, else every
    templated input looks copied (one real set: 117 of 118 seeds over Jaccard 0.5)."""
    df = Counter(g for t in traces for g in ngrams(user_text(t)))
    cut = max(3, 0.05 * len(traces))
    return frozenset(g for g, c in df.items() if c >= cut)


MIN_SHORT_COPY = 4  # a shorter input (a word or three) is too generic to call a copy


def copy_profile(text: str, common: frozenset = frozenset(), n: int = 8) -> dict:
    """What copied() compares, built once per text: its words, its distinctive n-grams, and its words with template
    spans (words inside a common n-gram) blanked out."""
    toks = re.findall(r"\w+", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    tpl = [False] * len(toks)
    for i, g in enumerate(grams):
        if g in common:
            tpl[i:i + n] = [True] * n
    free = " ".join("|" if m else t for t, m in zip(toks, tpl))  # "|" never matches a word
    return {"words": " ".join(toks), "len": len(toks), "grams": set(grams) - common, "free": f" {free} ", "n": n}


def copied_profiles(a: dict, b: dict, threshold: float = 0.5) -> bool:
    if a["words"] == b["words"]:
        return True
    short, long_ = (a, b) if a["len"] <= b["len"] else (b, a)
    if short["len"] < short["n"]:  # no n-grams to compare: the whole short input inside the other's non-template text
        return short["len"] >= MIN_SHORT_COPY and f" {short['words']} " in long_["free"]
    x, y = a["grams"], b["grams"]
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= threshold


def copied(a: str, b: str, common: frozenset = frozenset(), threshold: float = 0.5) -> bool:
    """Same text, or most of the smaller input's distinctive 8-grams (template removed) found in the other; an input
    under 8 words is a copy when it appears whole outside the other's template text."""
    return copied_profiles(copy_profile(a, common), copy_profile(b, common), threshold)


@lru_cache(maxsize=4)
def load_answer_schemas(path: str | None) -> dict[str, dict]:
    return json.loads(Path(path).read_text()) if path else {}


def verify(cfg: Config, trace: dict, eval_ngrams: set[tuple], expansion: dict | None = None) -> dict:
    """expansion: {"seed", "variation", "harness"} for seed-expansion traces; the judge then checks fidelity too."""
    f = cfg.features
    spec = trace["meta"].get("spec")
    if trace["meta"].get("refusal"):  # the generator declined the pairing: its own outcome, never judged or kept
        trace["meta"]["verify"] = {"rule_issues": [], "refused": True, "kept": False}
        return trace
    v: dict = {"rule_issues": [*trace["meta"].get("format_issues", []),
                               *rule_check(trace, cfg.min_assistant_turns, cfg.tool_arg_defaults, cfg.tool_optional_args,
                                          cfg.tool_loose_type_args),
                               *tool_result_check(trace, cfg.tool_result_shapes, cfg.tool_result_prefix),
                               *answer_check(trace, load_answer_schemas(cfg.answer_schemas_path)),
                               *explicit_answer_check(trace, (expansion or {}).get("seed"))]}
    if expansion:
        broken = input_json_flags(trace, expansion.get("seed"))
        if f.input_json_gate:  # precise on real runs (4/4 copy breaks, 0 false positives) and the judges ignored it as a note
            v["rule_issues"] += [f"input JSON broken: {b}" for b in broken]
        v["soft_flags"] = [*answer_soft_flags(trace, expansion.get("seed")), *extra_answer_keys(trace, expansion.get("seed")),
                           *([] if f.input_json_gate else broken)]  # soft: a prompt may make a key conditional
        v["no_assistant_delta"] = (not assistant_delta(trace, expansion["seed"])) if expansion.get("seed") else None
        if f.change_gate and expansion.get("seed"):
            v["change_issues"] = change_gate(trace, expansion["seed"], cfg.min_assistant_change)
    v["contaminated"] = bool(eval_ngrams and ngrams(user_text(trace)) & eval_ngrams)
    if f.provenance:
        v["ungrounded_args"] = provenance_check(trace, spec)
    keep = not v["rule_issues"] and not v["contaminated"] and not v.get("ungrounded_args") and not v.get("change_issues")
    if f.rubric_judge and keep:
        import contextvars
        from concurrent.futures import ThreadPoolExecutor

        from .consistency import check as consistency_check
        from .rubric_judge import describe_change, judge_rubric, load_rubric
        seed = (expansion or {}).get("seed")

        def guarded(fn):  # each gate fails closed: an unchecked trace is not kept, and says why
            try:
                return fn()
            except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                raise
            except Exception as e:  # noqa: BLE001
                return {"label": "error", "error": f"{type(e).__name__}: {e}"[:300]}
        notes = v.get("soft_flags") or None  # code-found differences from the seed: the judges weigh them, code never gates on them
        run_judge = lambda: guarded(lambda: judge_rubric(cfg, trace, seed=seed, rubric=load_rubric(cfg.rubric_path),
                                                     change=describe_change((expansion or {}).get("variation")), notes=notes))
        if f.consistency_check:  # two independent calls at once: latency of one, either can reject
            with ThreadPoolExecutor(2) as ex:
                fj = ex.submit(contextvars.copy_context().run, run_judge)
                fc = ex.submit(contextvars.copy_context().run, lambda: guarded(lambda: consistency_check(cfg, trace, seed, notes)))
                r, c = fj.result(), fc.result()
            v["consistency"] = c
            keep = r.get("label") == "good" and c.get("label") == "good"
        else:
            r = run_judge()
            keep = r.get("label") == "good"
        v["rubric"] = r
    if f.judge and keep:
        j, v["judge_attempts"] = None, 0
        for _ in range(2):  # a malformed response gets one retry; a failed call (incl. token limit) gets none
            v["judge_attempts"] += 1
            try:
                j = judge_expansion(cfg, trace, expansion) if expansion else judge(cfg, trace, spec)
            except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                raise
            except Exception as e:  # noqa: BLE001 - an evaluator failure is recorded, never read as a verdict on the trace
                v["eval_error"] = f"{type(e).__name__}: {e}"[:300]
                break
            missing = malformed(j, expansion)
            v["eval_error"] = f"malformed judge output: {', '.join(missing)}" if missing else None
            if not missing:
                break
        if not f.judge_gate:  # critique only: record whatever the judge said, rule checks alone decide `kept`
            v["judge"] = j if not v.get("eval_error") else {"error": v["eval_error"]}
            v.pop("eval_error", None)
            v["kept"] = keep
            trace["meta"]["verify"] = v
            return trace
        if v.get("eval_error"):
            v["judge"], v["kept"] = j, False
            trace["meta"]["verify"] = v
            return trace
        v.pop("eval_error", None)
        v["judge"] = j
        # fail closed: a missing field is a failed check, not a pass
        consistent = (j.get("obs_consistent") is True and j.get("policy_followed") is True
                      and j.get("checklist_failed") == [] and j.get("hallucinations") == []
                      and all(j.get(k) is True for k in (EXPANSION_CHECKS + ("case_consistent",) if expansion else ()))
                      and (not expansion or j.get("turn_issues") == []))
        success = bool(j.get("task_success"))
        keep = consistent and (success or cfg.keep_failures)
        trace["meta"]["bad_steps"] = j.get("bad_steps", []) if not success else []
    v["kept"] = keep
    trace["meta"]["verify"] = v
    return trace
