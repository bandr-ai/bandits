"""Seed-only analyzer -> generator -> checks -> diagnosis -> revision.

Only supplied dev seeds, embedded system prompts and tool definitions enter analysis.
Signals and variations are metadata; observations are examples, not reconstructed runtime rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .rollout import agent_rollout
from .banks import ObsBank, ResultFormat, UserBank, format_input_examples, format_obs_examples
from .config import Config
from .io import parallel_map, read_jsonl, write_jsonl
from .llm import BudgetExhausted, chat_json, get_llm
from .prompts import (
    SEED_VARIATIONS,
    SEED_VARIATIONS_WIDE,
    ANALYZER,
    ANALYZER_REQUIRED_SLOTS,
    ANALYZER_REVISION,
    ANALYZER_SLOTS,
    CASE_KEYS,
    DIAGNOSE,
    TASK_NOTE,
    ENV_ONLY,
    OUTPUT_FORMAT,
    OUTPUT_FORMAT_CASE,
    VARIATION_KEYS,
    VARIATION_KINDS,
)
from .schema import render
from .select import _shingles, action_signature, body_text, dedup, diversity_report
from .simia_text import tools_text
from .verify import final_answer, judge_flags, observed_loose_type_args, observed_optional_args, system_key, verify

_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def seed_json(trace: dict) -> str:
    """A trace in the generator's output format, so the example and the required output look alike."""
    msgs = []
    for m in trace["messages"]:
        if m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in m["tool_calls"]]
            msgs.append(msg)
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    return json.dumps({"messages": msgs}, ensure_ascii=False, indent=1)


def lint(template: str, holdout_texts: list[str] = ()) -> list[str]:
    errs = [f"missing required slot {{{s}}}" for s in ANALYZER_REQUIRED_SLOTS if f"{{{s}}}" not in template]
    errs += [f"unknown slot {{{s}}}" for s in sorted(set(_SLOT.findall(template)) - set(ANALYZER_SLOTS))]
    errs += ["template contains held-out seed text" for t in holdout_texts if len(t) >= 40 and t in template][:1]
    return errs


def check_variations(variations, labels: dict, seed_ids: list[str] | None = None, contracts: str | None = None) -> list[str]:
    if not isinstance(variations, list) or len(variations) < 3:
        return ["variations must be a list of at least 3 objects"]
    errs, names = [], set()
    for i, v in enumerate(variations):
        if not isinstance(v, dict):
            errs.append(f"variation {i} is not an object")
            continue
        missing = [k for k in VARIATION_KEYS if k != "requires_tools" and not v.get(k)]
        if not isinstance(v.get("requires_tools"), list):
            missing.append("requires_tools (a list, may be empty)")
        if missing:
            errs.append(f"variation {i} ({v.get('name')}) lacks {missing}")
        if v.get("name") in names:
            errs.append(f"duplicate variation name {v.get('name')}")
        names.add(v.get("name"))
        bad = [a for a in v.get("applies_to") or [] if a != "all" and a not in labels]
        if bad:
            errs.append(f"variation {v.get('name')} applies_to unknown labels {bad}")
        if seed_ids is not None:  # a real analyzer run: every variation names the seeds it can be applied to
            el = v.get("eligible_seeds")
            if not isinstance(el, list) or not el or not all(isinstance(x, str) for x in el):
                errs.append(f"variation {v.get('name')} needs eligible_seeds: a non-empty list of shown seed ids")
            elif [x for x in el if x not in seed_ids]:
                errs.append(f"variation {v.get('name')} eligible_seeds has unknown ids {[x for x in el if x not in seed_ids]}")
        if contracts is not None:
            errs += basis_errors(v, contracts)
            for k in ("decision_type", "answer_transition"):
                if not isinstance(v.get(k), str) or not v[k].strip():
                    errs.append(f"variation {v.get('name')} needs {k}")
    return errs


MAX_PER_CLASS = 2


def trim_classes(variations: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep at most MAX_PER_CLASS variations per (decision_type, answer_transition); the rest add no breadth.
    Order is the analyzer's own, so its first choices survive."""
    seen: Counter = Counter()
    kept, dropped = [], []
    for v in variations:
        key = (str(v.get("decision_type", "")).strip().lower(), str(v.get("answer_transition", "")).strip().lower())
        if key != ("", "") and seen[key] >= MAX_PER_CLASS:
            dropped.append(v.get("name"))
            continue
        seen[key] += 1
        kept.append(v)
    return kept, dropped


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).strip().lower()


def basis_errors(v: dict, contracts: str) -> list[str]:
    """A variation needing a result shape the seeds do not already show must quote the Explicit/Observed line that
    establishes it. basis == "seed" means it only reuses shapes present in the seed. The quote must occur in the
    contracts text before its Unknown section, so a shape the analyzer itself marked unknown cannot be required."""
    b = v.get("basis")
    if not isinstance(b, str) or not b.strip():
        return [f"variation {v.get('name')} needs basis: a verbatim quote from observed_contracts (Explicit/Observed), or \"seed\""]
    if b.strip().lower() == "seed":
        return []
    known = re.split(r"(?im)^#+\s*unknown\b", contracts, maxsplit=1)[0]
    if _squash(b) in _squash(known):
        return []
    # a lightly paraphrased quote is fine; a sentence the contracts never said is not: most of its words must occur there
    words = [w for w in re.findall(r"[a-z0-9_]{3,}", _squash(b))]
    have = set(re.findall(r"[a-z0-9_]{3,}", _squash(known)))
    if words and sum(w in have for w in words) / len(words) >= 0.8:
        return []
    return [f"variation {v.get('name')} basis is not a quote from the Explicit/Observed part of observed_contracts"]


def check_signals(signals, variations, seed_ids: list[str]) -> list[str]:
    """Validate provenance and references, not a task-specific skill taxonomy.
    Signals are OPTIONAL: the current analyzer prompt does not ask for them, because the
    signals -> variations -> assignment layer added instructions without evidence it helped. When present they are
    still validated, so older analyzer outputs and tests keep working."""
    if not signals and not any(isinstance(v, dict) and v.get("signals") for v in variations or []):
        return []
    if not isinstance(signals, list) or not signals:
        return ["signals must be a non-empty list"]
    errs, names = [], set()
    for sig in signals:
        if not isinstance(sig, dict):
            errs.append("signal must be an object")
            continue
        sid = sig.get("id")
        if not isinstance(sid, str) or not sid or sid in names:
            errs.append("signals need unique string ids")
        if isinstance(sid, str):
            names.add(sid)
        for key in ("teaches", "decision", "driving_evidence"):
            if not sig.get(key):
                errs.append(f"signal {sid}: missing {key}")
        sources = sig.get("sources")
        if not isinstance(sources, list) or not sources or any(x not in seed_ids for x in sources):
            errs.append(f"signal {sid}: sources must reference supplied dev seeds")
    for v in variations if isinstance(variations, list) else []:
        if not isinstance(v, dict):
            continue
        refs = v.get("signals")
        if not isinstance(refs, list) or not refs or any(x not in names for x in refs):
            errs.append(f"variation {v.get('name')}: unknown or missing signals")
    return errs


def check_combinations(combinations, variations: list[dict]) -> list[str]:
    if combinations is None:
        return []
    if not isinstance(combinations, list):
        return ["combinations must be a list of [name, name] pairs"]
    names = {v.get("name") for v in variations if isinstance(v, dict)}
    errs = []
    for c in combinations:
        if not (isinstance(c, list) and len(c) == 2 and c[0] != c[1] and set(c) <= names):
            errs.append(f"combination {c!r} must be two different known variation names")
            continue
        by = {v["name"]: v for v in variations if isinstance(v, dict)}
        if _both_eligible(by[c[0]], by[c[1]]) == []:
            errs.append(f"combination {c!r} has no seed eligible for both variations")
    return errs


def _both_eligible(a: dict, b: dict):
    ea, eb = a.get("eligible_seeds"), b.get("eligible_seeds")
    return sorted(set(ea) & set(eb)) if ea and eb else (ea or eb or None)


def combine(a: dict, b: dict) -> dict:
    """One job spec applying two variations together; the judge checks both are applied and neither undoes the other."""
    def both(k: str) -> str:
        return f"(1) {a.get(k, '')} (2) {b.get(k, '')}"

    return {"name": f"{a['name']}+{b['name']}", "combines": [a["name"], b["name"]],
            "applies_to": a["applies_to"], "eligible_seeds": _both_eligible(a, b),
            "requires_tools": sorted(set(a.get("requires_tools") or []) | set(b.get("requires_tools") or [])),
            "signals": sorted(set(a.get("signals", [])) | set(b.get("signals", []))),
            "signal_details": a.get("signal_details", []) + b.get("signal_details", []),
            "teaches": both("teaches"), "change": both("change") + " Apply both changes together.",
            "preserve": both("preserve"), "dependencies": both("dependencies"),
            "correct_when": both("correct_when") + " Both hold at once."}


def applicable(variations: list[dict], seed: dict, labels: dict) -> list[dict]:
    """Variations for this seed's agent group whose required tools the seed actually calls and, when the variation lists
    eligible_seeds, that name this seed. Ids prove nothing about semantics; the generator may still refuse."""
    label = next((lab for lab, keys in labels.items() if system_key(seed.get("system", "")) in label_keys(keys)), None)
    called = {c["name"] for m in seed["messages"] for c in m.get("tool_calls", [])}
    return [v for v in variations if ("all" in v["applies_to"] or label in v["applies_to"])
            and set(v.get("requires_tools") or []) <= called
            and (v.get("eligible_seeds") is None or seed.get("id") in v["eligible_seeds"])]


def fill(template: str, slots: dict[str, str], case: bool = False) -> str:
    out = template
    for name in ANALYZER_SLOTS:
        out = out.replace(f"{{{name}}}", slots.get(name, ""))
    return out + (OUTPUT_FORMAT_CASE if case else OUTPUT_FORMAT)


def parse_output(text: str, case: bool = False) -> tuple[list[dict] | None, list[str]] | tuple[dict | None, list[dict] | None, list[str]]:
    """Strict check of the fixed output format. Returns (messages, issues), or (case, messages, issues) with case=True;
    messages is None on any issue."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as e:
        issues = [f"format: not a single JSON object ({e.msg})"]
        return (None, None, issues) if case else (None, issues)
    case_obj, case_issues = None, []
    if case:
        case_obj = obj.get("case") if isinstance(obj, dict) else None
        if not isinstance(case_obj, dict) or any(not isinstance(case_obj.get(k), str) or not case_obj[k].strip() for k in CASE_KEYS):
            case_issues = [f"format: \"case\" must be an object with non-empty string fields {list(CASE_KEYS)}"]
    messages, issues = _parse_messages(obj)
    issues = case_issues + issues
    if case:
        return (case_obj if not issues else None), (messages if not issues else None), issues
    return messages, issues


def parse_refusal(text: str) -> str | None:
    """The generator's reason when it declined the pairing (a normal outcome, not a format failure), else None."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(obj, dict) and obj.get("invalid_pairing") is True and "messages" not in obj:
        return str(obj.get("reason") or "(no reason given)")
    return None


def _parse_messages(obj) -> tuple[list[dict] | None, list[str]]:
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list) or not obj["messages"]:
        return None, ["format: not {\"messages\": [non-empty list]}"]
    issues = []
    for i, m in enumerate(obj["messages"]):
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant", "tool"):
            issues.append(f"format: message {i} has role {role!r}")
            continue
        content = m.get("content")
        if role == "tool":
            if not isinstance(m.get("tool_call_id"), str) or not isinstance(content, str):
                issues.append(f"format: tool message {i} needs string tool_call_id and string content")
        elif role == "user" and not isinstance(content, str):
            issues.append(f"format: user message {i} content is not a string")
        elif role == "assistant":
            if content is not None and not isinstance(content, str):
                issues.append(f"format: assistant message {i} content is not a string")
            for c in m.get("tool_calls") or []:
                if not (isinstance(c, dict) and isinstance(c.get("id"), str) and isinstance(c.get("name"), str)
                        and isinstance(c.get("arguments"), dict)):
                    issues.append(f"format: assistant message {i} has a malformed tool call")
    return (None, issues) if issues else (obj["messages"], [])


def to_trace(job: dict, seed: dict, messages: list[dict] | None) -> dict:
    calls: dict[str, str] = {}
    out = []
    for m in messages or []:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "name": c["name"], "arguments": c["arguments"]} for c in m["tool_calls"]]
                calls.update({c["id"]: c["name"] for c in m["tool_calls"]})
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": calls.get(m["tool_call_id"], ""),
                        "content": m["content"]})
        else:
            out.append({"role": "user", "content": m["content"]})
    return {"id": job["job_id"], "system": seed.get("system", ""), "tools": seed["tools"], "messages": out, "meta": {}}


def generate_analyzer(cfg: Config, job: dict, seed: dict, template: str, obs: ObsBank) -> dict:
    rng = random.Random(f"{cfg.random_seed}:{job['job_id']}")
    split = cfg.features.split_answer
    hidden = ("applies_to", "eligible_seeds", "decision_type", "answer_transition") + (("teaches", "correct_when") if split else ())
    variation = {k: v for k, v in (job.get("variation") or {}).items() if k not in hidden}  # split: the outcome stays unsaid
    slots = {"seed_trace": seed_json(seed), "variation": json.dumps(variation, ensure_ascii=False, indent=1),
             "tools": tools_text(seed["tools"]), "system_prompt": seed.get("system", ""),
             "obs_examples": format_obs_examples(obs.examples_for_tools(seed["tools"], cfg.retrieved_obs_per_tool, rng), 800)
             if seed["tools"] else format_input_examples(obs.input_examples(cfg.retrieved_obs_per_tool, rng, seed["id"]))}
    discarded = []
    attempts = max(1, cfg.generation_attempts)
    use_case = bool(job.get("case_block"))
    case = None
    for attempt in range(1, attempts + 1):
        prompt = fill(template, {**slots, "generation_id": f"{job['job_id']}/{attempt}"}, case=use_case) + (ENV_ONLY if split else "") \
            + (TASK_NOTE if variation.get("kind") == "task" else "")
        out = get_llm(cfg, "generator").chat([{"role": "user", "content": prompt}], json_mode=True)
        refusal = parse_refusal(out["content"])
        if refusal:
            messages, issues, case = None, [], None
            break
        if use_case:
            case, messages, issues = parse_output(out["content"], case=True)
        else:
            messages, issues = parse_output(out["content"])
        if not issues or attempt == attempts:
            break
        discarded.append({"attempt": attempt, "problem": issues, "raw_output": out["content"]})
    answer_check = None
    if split and messages and not issues:
        while messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            messages = messages[:-1]  # a final reply written anyway is dropped: the agent writes its own
        prefix = to_trace(job, seed, messages)
        situation = json.dumps({k: variation[k] for k in ("change", "preserve", "dependencies") if k in variation}, ensure_ascii=False)
        fmt = ResultFormat(cfg.tool_result_prefix, [seed])
        examples = lambda name: format_obs_examples([{**e, "result": fmt.strip(e["result"])} for e in
                                                     obs.retrieve(name, None, cfg.retrieved_obs_per_tool, rng)], 800)
        added, answer_check = agent_rollout(cfg, prefix, situation, examples, cfg.split_max_rounds, fmt.apply)
        if added is None:
            issues = [answer_check["issue"]]
        else:
            messages = prefix["messages"] + added
    trace = to_trace(job, seed, messages)
    trace["meta"] = {"raw_output": out["content"], "format_issues": issues, "refusal": refusal, "attempts": attempt,
                     "discarded_attempts": discarded, "variation": variation.get("name"), "variation_spec": variation,
                     "case_block": use_case, "case": case, "answer_check": answer_check,
                     **{k: job.get(k) for k in ("job_id", "seed_id", "strategy", "mode", "version", "arm")}}
    trace["meta"]["fidelity"] = None if refusal else fidelity(trace, seed, cfg.tool_result_prefix)
    return trace


def variation_for(job_id: str, seed: dict, variations: list[dict], labels: dict) -> dict | None:
    """Stable choice for the main pipeline (prompt_source=analyzer), where jobs are planned without a ledger."""
    pool = applicable(variations, seed, labels)
    return pool[int(hashlib.sha256(job_id.encode()).hexdigest(), 16) % len(pool)] if pool else None


# ---- diagnostics: fidelity to the seed and decision cells (never reject on their own) ----

def _context(trace: dict) -> dict | None:
    """The JSON object in the first user message (e.g. 'Failure context: {...}'), or None."""
    text = next((m["content"] for m in trace["messages"] if m["role"] == "user"), "")
    i = text.find("{")
    try:
        obj = json.JSONDecoder().raw_decode(text[i:])[0] if i >= 0 else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _strip(content: str, prefix: str | None) -> str:
    return re.sub(prefix, "", content, count=1) if prefix else content


def fidelity(trace: dict, seed: dict, prefix: str | None) -> dict:
    """What changed vs the seed: failure-context fields (one level into objects) and reuse of the seed's tool results."""
    a, b = _context(seed), _context(trace)
    changed = None
    if a is not None and b is not None:
        flat = lambda d: {f"{k}.{k2}" if isinstance(v, dict) else k: (v2 if isinstance(v, dict) else v)
                          for k, v in d.items() for k2, v2 in (v.items() if isinstance(v, dict) else [(None, v)])}
        fa, fb = flat(a), flat(b)
        changed = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    seed_obs = {_strip(m["content"], prefix) for m in seed["messages"] if m["role"] == "tool"}
    obs = [_strip(m["content"], prefix) for m in trace["messages"] if m["role"] == "tool"]
    reused = sum(o in seed_obs for o in obs)
    calls = lambda t: [(c["name"], json.dumps(c["arguments"], sort_keys=True))
                       for m in t["messages"] for c in m.get("tool_calls", [])]
    # Everything the agent was given (user turns and tool results, in order): a no-op leaves all of it unchanged,
    # whatever its format (a JSON context, documents in the message, tool results).
    inputs = lambda t: [(m["role"], _strip(m["content"], prefix) if m["role"] == "tool" else m["content"])
                        for m in t["messages"] if m["role"] in ("user", "tool")]
    return {"context_parsed": changed is not None, "context_fields_changed": changed,
            "platform_changed": (a or {}).get("platform") != (b or {}).get("platform") if changed is not None else None,
            "tool_results": len(obs), "tool_results_reused_from_seed": reused,
            "no_op": inputs(trace) == inputs(seed) and calls(trace) == calls(seed)}


def _label_value(v) -> bool:
    """A short categorical answer value (a label, a flag, null), not prose, a number or a structure."""
    return v is None or isinstance(v, bool) or isinstance(v, str) and len(v) <= 40 and not re.search(r"\s", v)


def answer_labels(ans, fields: list[str] | None = None) -> tuple:
    """The categorical part of a final answer: the given fields, or else every top-level key whose value is a short
    label. A non-JSON answer (free text) has none."""
    if not isinstance(ans, dict):
        return ()
    keys = fields or sorted(k for k, v in ans.items() if _label_value(v))
    return tuple((k, ans.get(k)) for k in keys)


def _result_labels(content: str) -> tuple:
    """Short categorical top-level values of a tool result that holds a JSON object (any leading marker skipped)."""
    i = (content or "").find("{")
    try:
        o = json.loads(content[i:]) if i >= 0 else None
    except json.JSONDecodeError:
        o = None
    return tuple(sorted((k, v) for k, v in o.items() if _label_value(v))) if isinstance(o, dict) else ()


def decision_cells(trace: dict) -> list[tuple]:
    """One cell per assistant turn: evidence seen so far (tool, its categorical result values) -> action taken."""
    seen, cells = [], []
    for m in trace["messages"]:
        if m["role"] == "tool":
            seen.append((m.get("name"), _result_labels(m["content"])))
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                action = tuple(sorted(c["name"] for c in m["tool_calls"]))
            else:
                ans = final_answer(trace) if m is trace["messages"][-1] else None
                action = ("answer", answer_labels(ans)) if isinstance(ans, dict) else ("text",)
            cells.append((system_key(trace.get("system", ""))[:8], tuple(seen), action))
    return cells


# ---- analyzer and diagnosis agents ----

# System prompts whose word 5-gram Jaccard similarity is at least this belong to one agent group: the same task
# prompt with different platform/context notes. Chosen on 12 dev seeds of one dataset, where same-task pairs scored >= 0.31 and
# different-task pairs <= 0.21 (a thin margin: one dataset). Identical prompts score 1.0; unrelated ones near 0.
GROUP_MIN_SIMILARITY = 0.25


def _grams(text: str, n: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def label_keys(value) -> list[str]:
    """Member system keys of a label (older system_labels.json files store one key as a string)."""
    return [value] if isinstance(value, str) else list(value)


def system_labels(seeds: list[dict]) -> dict[str, list[str]]:
    """S1..Sn for groups of similar system prompts of the dev seeds (stable across rounds): {label: [system keys]}.
    Single-link grouping on text similarity, so one agent's prompt variants (e.g. platform-specific notes) share a label and a
    variation can apply to all of them."""
    texts: dict[str, str] = {}
    for s in sorted(seeds, key=lambda s: s["id"]):
        texts.setdefault(system_key(s.get("system", "")), s.get("system", ""))
    grams = {k: _grams(t) for k, t in texts.items()}
    sim = lambda x, y: len(grams[x] & grams[y]) / len(grams[x] | grams[y])
    groups: list[list[str]] = []
    for k in texts:  # first-appearance order
        hit = [g for g in groups if any(sim(k, m) >= GROUP_MIN_SIMILARITY for m in g)]
        merged = [k] + [m for g in hit for m in g]
        groups = [g for g in groups if g not in hit] + [merged]
    order = {k: i for i, k in enumerate(texts)}
    groups = sorted((sorted(g, key=order.get) for g in groups), key=lambda g: order[g[0]])
    return {f"S{i + 1}": g for i, g in enumerate(groups)}


def _analyzer_inputs(cfg: Config, seeds: list[dict], obs: ObsBank, labels: dict) -> dict:
    rng = random.Random(cfg.random_seed)
    shown = seeds  # the analyzer sees the entire supplied dev seed set
    by_key = {key: label for label, keys in labels.items() for key in label_keys(keys)}
    texts = {system_key(s.get("system", "")): s.get("system", "") for s in seeds}
    sys_text = "\n".join(
        f'<system_prompt label="{label}"' + (f' variant="{i + 1}of{len(label_keys(keys))}"' if len(label_keys(keys)) > 1 else "")
        + f'>\n{texts[key]}\n</system_prompt>' for label, keys in labels.items() for i, key in enumerate(label_keys(keys)))
    parts = [f'<seed id="{s["id"]}" system="{by_key[system_key(s.get("system", ""))]}">\n{seed_json(s)}\n</seed>' for s in shown]
    tools = {t["name"]: t for s in shown for t in s["tools"]}
    obs_ex = obs.examples_for_tools(list(tools.values()), 2, rng)
    return {"seed_ids": [s["id"] for s in shown], "n_seeds": len(shown), "dev_seeds": shown,
            "system_prompts": sys_text, "seeds": "\n".join(parts), "tools": tools_text(list(tools.values())),
            "obs": format_obs_examples(obs_ex, 800)}


def variation_errors(v: dict, labels: dict, seed_ids: list[str], contracts: str) -> list[str]:
    """check_variations for ONE variation (it wants a list of at least 3, so pad with renamed copies and drop their noise)."""
    pad = [v, {**v, "name": f"{v.get('name')}__pad2"}, {**v, "name": f"{v.get('name')}__pad3"}]
    return [e for e in check_variations(pad, labels, seed_ids, contracts) if "__pad" not in e]


def propose_for_seed(cfg: Config, seed: dict, label: str, labels: dict, dev: list[dict], contracts: str, existing: list[dict],
                     covered: list[dict], n: int) -> list[dict]:
    """One short analyzer call for one seed: n new variations of that seed (siblings under the same label shown for contrast)."""
    keys = set(label_keys(labels.get(label, [])))
    sibs = [s for s in dev if s["id"] != seed["id"] and system_key(s.get("system", "")) in keys][:2]
    cov = [f'- {r["variation"]}: ' + ("kept" if r.get("kept") else "refused" if r.get("refused") else "rejected") for r in covered if r["seed_id"] == seed["id"]]
    prompt = SEED_VARIATIONS.format(
        observed_contracts=contracts, label=label, system=seed.get("system", ""), seed_id=seed["id"], seed=seed_json(seed),
        siblings="\n".join(f'<sibling id="{s["id"]}">\n{seed_json(s)[:6000]}\n</sibling>' for s in sibs) or "(none)",
        covered=("cells finished on this seed:\n" + "\n".join(cov) if cov else "(none yet)") + "\nnames already used: " + ", ".join(sorted({x["name"] for x in existing})[:80]),
        n=n)
    out = chat_json(get_llm(cfg, "analyzer"), "You are an expert in agent training data. Output JSON only.", prompt)
    vs = out.get("variations") if isinstance(out, dict) else None
    ids = {s["id"] for s in dev}
    good = []
    rejected = propose_for_seed.rejected = getattr(propose_for_seed, "rejected", [])
    for v in vs if isinstance(vs, list) else []:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        v["applies_to"] = [label]
        v["eligible_seeds"] = [seed["id"]] + [x for x in (v.get("eligible_seeds") or []) if isinstance(x, str) and x in ids and x != seed["id"]]
        errs = variation_errors(v, labels, sorted(ids), contracts)
        if not errs:
            good.append(v)
        else:
            rejected.append({"seed": seed["id"], "name": v.get("name"), "errors": errs[:3]})
    return good


def write_version(cfg: Config, vdir: Path, version: int, inputs: dict, prev: dict | None,
                  holdout_texts: list[str], labels: dict) -> dict:
    """Ask the analyzer for version <version>; saves observed_contracts.md, variations.json, prompt.txt, rationale.md,
    analyzer_raw.json. Returns {"template", "variations", "observed_contracts"}."""
    if (vdir / "prompt.txt").exists():
        if not (vdir / "signals.json").exists() or not (vdir / "observed_contracts.md").exists():
            raise RuntimeError("Legacy analyzer output: use a fresh output directory for seed-only synthesis")
        combos = vdir / "combinations.json"
        return {"template": (vdir / "prompt.txt").read_text(), "observed_contracts": (vdir / "observed_contracts.md").read_text(),
                "variations": json.loads((vdir / "variations.json").read_text()),
                "signals": json.loads((vdir / "signals.json").read_text()),
                "combinations": json.loads(combos.read_text()) if combos.exists() else []}
    revision = ANALYZER_REVISION.format(**prev) if prev else ""
    request = ANALYZER.format(**inputs, revision=revision,
                              rationale_extra="; and what you changed in response to the diagnosis" if prev else "")
    llm = get_llm(cfg, "analyzer")
    attempts = []
    for _ in range(3):
        msg = request + (f"\n\nYour previous answer failed these checks; fix them: {attempts[-1]['errors']}" if attempts else "")
        out = chat_json(llm, "You are an expert in agent training data and prompt design. Output JSON only.", msg)
        out = out if isinstance(out, dict) else {}
        template = out.get("template", "")
        errs = (lint(template, holdout_texts) if template else ["no template in output"]) + \
            check_variations(out.get("variations"), labels, inputs.get("seed_ids"), out.get("observed_contracts") or "") + check_signals(out.get("signals"), out.get("variations"), inputs.get("seed_ids", [])) + ([] if out.get("observed_contracts") else ["no observed contracts"])
        if not errs:
            errs += check_combinations(out.get("combinations"), out["variations"])
        attempts.append({"output": out, "errors": errs})
        if not errs:
            break
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "analyzer_raw.json").write_text(json.dumps({"request": request, "attempts": attempts}, ensure_ascii=False, indent=1))
    errs = attempts[-1]["errors"]
    if errs and all("basis" in e for e in errs):  # only unsupported result shapes remain: drop those variations, keep the rest
        bad = {e.split()[1] for e in errs}
        kept = [x for x in out.get("variations", []) if x.get("name") not in bad]
        if len(kept) >= 3:
            out["variations"], errs = kept, []
            attempts[-1]["dropped_for_basis"] = sorted(bad)
    if errs:
        raise RuntimeError(f"analyzer v{version} output failed checks: {errs}")
    trimmed, dropped = trim_classes(out["variations"])
    if dropped and len(trimmed) >= 3:
        out["variations"] = trimmed
        attempts[-1]["dropped_for_class_cap"] = dropped
    if cfg.per_seed_variations and inputs.get("dev_seeds"):
        from concurrent.futures import ThreadPoolExecutor
        existing = list(out["variations"])
        covered = read_jsonl(vdir.parent / "ledger.jsonl")
        by_key = {k: lab for lab, ks in labels.items() for k in label_keys(ks)}
        dev = inputs["dev_seeds"]

        def one(seed):
            try:
                return propose_for_seed(cfg, seed, by_key.get(system_key(seed.get("system", "")), "all"), labels, dev,
                                        out["observed_contracts"], existing, covered, cfg.variations_per_seed)
            except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                raise
            except Exception:  # noqa: BLE001 - a failed proposal call only means fewer variations for that seed
                return []
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            extra = list(ex.map(one, dev))
        names = {x["name"] for x in existing}
        for vs in extra:
            for x in vs:
                if x["name"] in names:
                    x["name"] = f'{x["name"]}__{x["eligible_seeds"][0][-6:]}'
                names.add(x["name"])
                out["variations"].append(x)
        attempts[-1]["per_seed_added"] = len(out["variations"]) - len(existing)
        attempts[-1]["per_seed_rejected"] = list(getattr(propose_for_seed, "rejected", []))
        (vdir / "analyzer_raw.json").write_text(json.dumps({"request": request, "attempts": attempts}, ensure_ascii=False, indent=1))
    out["signals"] = out.get("signals") or []  # optional, see check_signals
    by_signal = {s["id"]: s for s in out["signals"]}
    for variation in out["variations"]:
        if variation.get("signals"):
            variation["signal_details"] = [by_signal[sid] for sid in variation["signals"]]
    (vdir / "signals.json").write_text(json.dumps(out["signals"], ensure_ascii=False, indent=1))
    (vdir / "observed_contracts.md").write_text(out["observed_contracts"])
    (vdir / "variations.json").write_text(json.dumps(out["variations"], ensure_ascii=False, indent=1))
    (vdir / "prompt.txt").write_text(template)
    (vdir / "rationale.md").write_text(str(out.get("rationale", "")))
    combos = out.get("combinations") or []
    (vdir / "combinations.json").write_text(json.dumps(combos, ensure_ascii=False, indent=1))
    return {"template": template, "variations": out["variations"], "observed_contracts": out["observed_contracts"], "signals": out["signals"], "combinations": combos}


def diagnose(cfg: Config, vdir: Path, version: int, spec: dict, report: dict, verified: list[dict],
             seeds_by_id: dict[str, dict]) -> str:
    if (vdir / "diagnosis.md").exists():
        return (vdir / "diagnosis.md").read_text()
    rng = random.Random(f"{cfg.random_seed}:diag:{version}")
    def flagged(t: dict) -> bool:  # rejected by the checks, or criticised by the judge (gated or not)
        v = t["meta"]["verify"]
        return not v["kept"] or bool(judge_flags(v.get("judge")))

    rejected = [t for t in verified if flagged(t)]
    kept = [t for t in verified if not flagged(t)]
    rej_s, kept_s = rng.sample(rejected, min(8, len(rejected))), rng.sample(kept, min(6, len(kept)))

    def show(t: dict) -> str:
        v = t["meta"]["verify"]
        why = {"refused": t["meta"].get("refusal"), "rule_issues": v["rule_issues"], "soft_flags": v.get("soft_flags"), "ungrounded_args": v.get("ungrounded_args"),
               "judge": {k: (v.get("judge") or {}).get(k) for k in ("checks", "task_success", "obs_consistent", "policy_followed",
                                                                    "hallucinations", "checklist_failed", "unverified_excerpts",
                                                                    "turn_issues", "bad_steps", "notes")},
               "judge_flags": judge_flags(v.get("judge")),
               "rubric": {k: (v.get("rubric") or {}).get(k) for k in ("label", "situation", "expected_answer",
                                                                    "integrity_failed", "minor_issues", "reason", "error")}
                         if v.get("rubric") else None,
               "fidelity": t["meta"].get("fidelity")}
        seed = seeds_by_id.get(t["meta"].get("seed_id"))
        seed_text = render(seed, max_obs_chars=1500)[:7000] if seed else "(seed not found)"
        body = (render(t, max_obs_chars=1500) if t["messages"] else
                f"(generator declined this pairing: {t['meta']['refusal']})" if t["meta"].get("refusal") else
                f"(unparseable output)\n{t['meta'].get('raw_output', '')[:3000]}")
        return (f'<trace id="{t["id"]}" seed="{t["meta"].get("seed_id")}" variation="{t["meta"].get("variation")}">\n'
                f"<seed>\n{seed_text}\n</seed>\n<reasons>{json.dumps(why, ensure_ascii=False)}</reasons>\n"
                f"<generated>\n{body[:7000]}\n</generated>\n</trace>")

    out = chat_json(get_llm(cfg, "diagnose"), "You are a rigorous reviewer of agent training data. Output JSON only.",
                    DIAGNOSE.format(version=version, observed_contracts=spec["observed_contracts"],
                                    variations=json.dumps(spec["variations"], ensure_ascii=False, indent=1),
                                    signals=json.dumps(spec["signals"], ensure_ascii=False, indent=1),
                                    template=spec["template"], report=json.dumps(report, indent=1, default=str),
                                    n_rejected=len(rej_s), total_rejected=len(rejected), n_kept=len(kept_s),
                                    total_kept=len(kept), rejected="\n".join(map(show, rej_s)) or "(none)",
                                    kept="\n".join(map(show, kept_s)) or "(none)"))
    notes = str(out.get("notes", "")) if isinstance(out, dict) else str(out)
    (vdir / "diagnosis.md").write_text(notes)
    return notes


# ---- planning, ledger and reports ----

def plan_jobs(prefix: str, seeds: list[dict], n: int, variations: list[dict], labels: dict,
              ledger: list[dict], combinations: list | None = None, combo_share: float = 0.0,
              max_per_variation: int = 2, max_per_seed: int = 3, max_per_family: int | None = None, **extra) -> list[dict]:
    """Up to n jobs over UNTRIED (seed, variation) pairs only: a pair already in the ledger (any round, refused or not)
    or already planned this round is never scheduled again, because a second sample of the same pair is the same lesson.
    Fewer than n jobs are planned when the pool of untried pairs is smaller; the caller reports that. Seeds used least
    so far come first and each seed advances one variation per sweep, so jobs spread over the pool.
    With combo_share > 0, that share of jobs uses a combination of two variations both applicable to the seed.
    max_per_family caps jobs per variation family (the name before "__": per-seed copies of one proposed idea), so one
    cheap kind of change cannot fill the round."""
    done = {(r["seed_id"], r["variation"]) for r in ledger}
    used = Counter(r["seed_id"] for r in ledger)
    by_name = {v["name"]: v for v in variations}
    queues: dict[str, list[dict]] = {}
    combos_for: dict[str, list[dict]] = {}
    for s in sorted(seeds, key=lambda s: used[s["id"]]):
        app = applicable(variations, s, labels)
        names = {v["name"] for v in app}
        combos_for[s["id"]] = [c for c in (combine(by_name[x], by_name[y]) for x, y in combinations or [] if x in names and y in names)
                               if (s["id"], c["name"]) not in done]
        pool = [v for v in app if (s["id"], v["name"]) not in done]
        if pool:
            queues[s["id"]] = pool
    jobs, sweep, used_name, used_seed, used_family = [], 0, Counter(), Counter(), Counter()
    while len(jobs) < n and queues:
        for sid in list(queues):
            if len(jobs) >= n:
                break
            pool = queues[sid]
            if used_seed[sid] >= max_per_seed:  # spread the round over seeds instead of piling onto the ones with many variations
                del queues[sid]
                continue
            capped = [i for i, x in enumerate(pool) if used_name[x["name"]] < max_per_variation
                      and (max_per_family is None or used_family[_family(x["name"])] < max_per_family)]
            if not capped:  # everything left on this seed was already used enough this round: leave it for a later round
                del queues[sid]
                continue
            pool.insert(0, pool.pop(capped[0]))
            combo_slot = combo_share > 0 and int((len(jobs) + 1) * combo_share) > int(len(jobs) * combo_share)
            v = combos_for[sid].pop(0) if combo_slot and combos_for[sid] else pool.pop(0)
            used_name[v["name"]] += 1
            used_family[_family(v["name"])] += 1
            used_seed[sid] += 1
            jobs.append({"job_id": f"{prefix}/{sid}__{sweep}", "seed_id": sid, "strategy": "analyzer", "variation": v, **extra})
            if not pool:
                del queues[sid]
        sweep += 1
    return jobs


def _family(name: str) -> str:
    return name.split("__")[0]


def outcomes(traces: list[dict], fields: list[str]) -> dict:
    """Counts of each categorical answer value (fields, or else every short label in the answer)."""
    c: Counter = Counter()
    for t in traces:
        c.update(f"{k}={v}" for k, v in answer_labels(final_answer(t), fields))
    return dict(sorted(c.items()))


def _fidelity_summary(traces: list[dict]) -> dict:
    fs = [t["meta"].get("fidelity") or {} for t in traces]
    parsed = [f for f in fs if f.get("context_parsed")]
    fields = Counter(k for f in parsed for k in f["context_fields_changed"])
    n_obs = sum(f.get("tool_results", 0) for f in fs)
    r = lambda x: round(x, 3)
    return {"context_parsed": f"{len(parsed)}/{len(fs)}",
            "mean_context_fields_changed": r(sum(len(f["context_fields_changed"]) for f in parsed) / len(parsed)) if parsed else None,
            "platform_changed": sum(bool(f.get("platform_changed")) for f in parsed),
            "no_op": sum(bool(f.get("no_op")) for f in fs),
            "tool_results_reused_from_seed_share": r(sum(f.get("tool_results_reused_from_seed", 0) for f in fs) / n_obs) if n_obs else None,
            "most_changed_fields": dict(fields.most_common(8))}


def pilot_report(cfg: Config, verified: list[dict], seeds: list[dict], attempted: int | None = None) -> dict:
    kept = [t for t in verified if t["meta"]["verify"]["kept"]]
    unique, dup = dedup(kept, near=cfg.features.near_dedup)
    best: dict[tuple, dict] = {}  # one trace per (seed, variation): a second sample of a pair is the same lesson
    for t in unique:
        key = (t["meta"].get("seed_id"), t["meta"].get("variation"))
        if key not in best or (bool(judge_flags(t["meta"]["verify"].get("judge"))), len(json.dumps(t["messages"]))) < (
                bool(judge_flags(best[key]["meta"]["verify"].get("judge"))), len(json.dumps(best[key]["messages"]))):
            best[key] = t
    if len(best) < len(unique):
        dup = {**dup, "same_pair": len(unique) - len(best)}
        unique = [t for t in unique if best[(t["meta"].get("seed_id"), t["meta"].get("variation"))] is t]
    lesson_dups = [t for t in unique if t["meta"]["verify"].get("lesson_dup_of")]
    if lesson_dups:
        dup = {**dup, "lesson_dup": len(lesson_dups)}
        unique = [t for t in unique if not t["meta"]["verify"].get("lesson_dup_of")]
    reasons: Counter = Counter()
    items: Counter = Counter()
    by_var: dict[str, Counter] = {}
    for t in verified:
        v = t["meta"]["verify"]
        reasons.update(r.split(":")[0] if r.startswith("format") else r for r in v["rule_issues"])
        if v.get("ungrounded_args"):
            reasons["ungrounded_args"] += 1
        j = v.get("judge")
        if v.get("eval_error"):
            reasons["eval_error"] += 1
        elif j is not None and not v["kept"]:
            reasons["judge_rejected"] += 1
            if (j or {}).get("turn_issues"):
                reasons["turn_issues"] += 1
        items.update((j or {}).get("checklist_failed") or [])
        b = by_var.setdefault(t["meta"].get("variation") or "-", Counter())
        b["jobs"] += 1
        b["kept"] += v["kept"]
        b["no_op"] += bool((t["meta"].get("fidelity") or {}).get("no_op"))
        b["refused"] += bool(t["meta"].get("refusal"))
    cells = Counter(c for t in unique for c in decision_cells(t))
    seed_cells = {c for s in seeds for c in decision_cells(s)}
    n = attempted if attempted is not None else len(verified)  # jobs planned, including ones that raised before producing a trace
    judge_errors = [t for t in verified if isinstance((t["meta"]["verify"].get("judge")), dict) and t["meta"]["verify"]["judge"].get("error")]
    unevaluated = lambda t: bool(cfg.features.judge) and not isinstance(t["meta"]["verify"].get("judge"), dict) or \
        (isinstance(t["meta"]["verify"].get("judge"), dict) and bool(t["meta"]["verify"]["judge"].get("error")))
    refused = [t for t in verified if t["meta"].get("refusal")]
    convs = [t for t in verified if not t["meta"].get("refusal") and t["messages"]]
    exec_errors = n - len(refused) - len(convs)  # generation raised, or no parseable conversation
    by_var_out: dict[str, Counter] = {}
    for t in unique:
        ans = final_answer(t)
        key = " / ".join(f"{k}={v}" for k, v in answer_labels(ans, cfg.outcome_fields)) if isinstance(ans, dict) else "unparsed"
        by_var_out.setdefault(t["meta"].get("variation") or "-", Counter())[key] += 1
    return {"jobs": n,
            # attempted jobs = conversations + refusals + execution errors; quality below is also given among conversations only
            "outcome_split": {"conversations": len(convs), "refusals": len(refused), "execution_errors": exec_errors},
            "refusal_reasons": {t["id"]: t["meta"]["refusal"][:300] for t in refused},
            "usable_of_conversations": round(len(unique) / len(convs), 3) if convs else None,
            "format_ok": sum(1 for t in verified if not t["meta"].get("format_issues") and not t["meta"].get("refusal")),
            "kept": len(kept), "unique_kept": len(unique), "usable_rate": round(len(unique) / n, 3) if n else 0.0,
            # critique-aware: kept by the checks AND not criticised by the judge; this, not usable_rate, picks the version
            "judge_flagged_kept": sum(bool(judge_flags(t["meta"]["verify"].get("judge"))) for t in unique),
            # clean = kept by the checks, judged without error, and not criticised; an unevaluated trace is NOT clean
            "clean_rate": round(sum(not judge_flags(t["meta"]["verify"].get("judge")) and not unevaluated(t) for t in unique) / n, 3) if n else 0.0,
            "judge_unevaluated": len(judge_errors),
            "no_assistant_delta": sum(bool(t["meta"]["verify"].get("no_assistant_delta")) for t in unique),
            "judge_flag_counts": dict(Counter(f.split(":")[0] + (":" + f.split(":")[1] if f.startswith("check:") else "")
                                              for t in verified for f in judge_flags(t["meta"]["verify"].get("judge"))).most_common()),
            "eval_errors": sum(bool(t["meta"]["verify"].get("eval_error")) for t in verified),
            "judge_retried": sum(t["meta"]["verify"].get("judge_attempts", 1) > 1 for t in verified),
            "outcomes_by_variation": {k: dict(v) for k, v in sorted(by_var_out.items())},
            "nearest_kept": nearest(unique, seeds),
            "dedup": dup, "reject_reasons": dict(reasons.most_common()), "checklist_failed": dict(items.most_common()),
            "by_variation": {k: dict(v) for k, v in sorted(by_var.items())},
            "outcomes_kept": outcomes(unique, cfg.outcome_fields), "outcomes_seeds": outcomes(seeds, cfg.outcome_fields),
            "fidelity_all": _fidelity_summary(verified), "fidelity_kept": _fidelity_summary(unique),
            "decision_cells_kept": {"distinct": len(cells), "max_repeats": max(cells.values(), default=0),
                                    "not_in_seeds": sum(1 for c in cells if c not in seed_cells)},
            "diversity_kept": diversity_report(unique),
            "attempts": dict(Counter(t["meta"].get("attempts", 1) for t in verified))}


def nearest(kept: list[dict], seeds: list[dict], top: int = 8) -> dict:
    """For each kept trace, its most similar other kept trace or seed (word 3-gram Jaccard over non-user text).
    A diagnostic for meaningful repetition, never a filter: useful minimal pairs are similar by design."""
    pool = [(t["id"], _shingles(body_text(t))) for t in kept] + [(f"seed:{s['id']}", _shingles(body_text(s))) for s in seeds]
    rows = []
    for t in kept:
        a = _shingles(body_text(t))
        best = max(((oid, len(a & b) / len(a | b) if a | b else 0.0) for oid, b in pool if oid != t["id"]),
                   key=lambda x: x[1], default=(None, 0.0))
        rows.append({"id": t["id"], "variation": t["meta"].get("variation"), "nearest": best[0], "similarity": round(best[1], 3)})
    rows.sort(key=lambda r: -r["similarity"])
    sims = [r["similarity"] for r in rows]
    return {"most_similar": rows[:top], "at_least_0.8": sum(s >= 0.8 for s in sims), "at_least_0.6": sum(s >= 0.6 for s in sims)}


def _append_ledger(root: Path, verified: list[dict]) -> None:
    done = {r["job_id"] for r in read_jsonl(root / "ledger.jsonl")}
    with (root / "ledger.jsonl").open("a") as f:
        for t in verified:
            if t["id"] in done:
                continue
            f.write(json.dumps({"job_id": t["id"], "version": t["meta"].get("version"), "seed_id": t["meta"].get("seed_id"),
                                "variation": t["meta"].get("variation"), "kept": t["meta"]["verify"]["kept"],
                                "refused": bool(t["meta"].get("refusal")),
                                "reasons": t["meta"]["verify"]["rule_issues"][:5]}, ensure_ascii=False) + "\n")


def expansion_context(trace: dict, seeds_by_id: dict[str, dict], observed_contracts: str) -> dict | None:
    """What the judge needs to check a seed-expansion trace: its seed, its variation, the observed_contracts description."""
    spec = trace["meta"].get("variation_spec")
    seed = seeds_by_id.get(trace["meta"].get("seed_id"))
    return {"seed": seed, "variation": spec, "observed_contracts": observed_contracts, "case": trace["meta"].get("case")} if spec and seed else None


LESSON_PROMPT = """Below are generated training conversations, grouped by the real seed they were expanded from. For each
conversation: the variation name, the generator's own note of what it changed (assigned_change, dependent_observations)
and its final answer fields. Within ONE seed, find conversations that teach the SAME lesson: the same evidence change
handled the same way, even if the variation names or wording differ. Different changes, or the same change leading to a
different decision, are NOT duplicates. Never compare across seeds.

{items}

Return JSON only: {{"duplicate_groups": [["<id>", "<id>"], ...]}} listing only groups of two or more duplicates (ids as given);
{{"duplicate_groups": []}} when there are none."""


LESSON_CHUNK = 40  # conversations per lesson-dedup call: whole seeds only, so a big batch never becomes one huge prompt


def mark_lesson_duplicates(cfg: Config, verified: list[dict]) -> list[dict]:
    """Judge-model calls after generation: within each seed, group kept conversations that teach the same lesson (read
    from the generator's own case notes, so the shared seed text does not hide the difference). All but the best of a
    group get meta.verify.lesson_dup_of; `kept` is untouched. Seeds are sent in chunks of whole seeds (LESSON_CHUNK
    conversations per call); a failed call records lesson_dup_error on its chunk and changes nothing."""
    by_seed: dict[str, list[dict]] = {}
    for t in verified:
        if t["meta"]["verify"]["kept"] and t["messages"]:
            by_seed.setdefault(t["meta"].get("seed_id"), []).append(t)
    multi = {s: ts for s, ts in by_seed.items() if len(ts) > 1}
    chunks: list[dict[str, list[dict]]] = []
    for sid, ts in multi.items():
        if not chunks or sum(map(len, chunks[-1].values())) + len(ts) > LESSON_CHUNK:
            chunks.append({})
        chunks[-1][sid] = ts
    for chunk in chunks:
        _lesson_chunk(cfg, chunk)
    return verified


def _lesson_chunk(cfg: Config, multi: dict[str, list[dict]]) -> None:
    lines = []
    for sid, ts in multi.items():
        lines.append(f'<seed id="{sid}">')
        for t in ts:
            case = t["meta"].get("case") or {}
            ans = final_answer(t)
            lines.append(f'<conversation id="{t["id"]}" variation="{t["meta"].get("variation")}">\n'
                         f'assigned_change: {str(case.get("assigned_change", ""))[:500]}\n'
                         f'dependent_observations: {str(case.get("dependent_observations", ""))[:500]}\n'
                         f'final: {json.dumps(ans, ensure_ascii=False)[:400] if ans is not None else "(not JSON)"}\n</conversation>')
        lines.append("</seed>")
    try:
        out = chat_json(get_llm(cfg, "judge"), "You are a careful data curator. Output JSON only.",
                        LESSON_PROMPT.format(items="\n".join(lines)))
        groups = out.get("duplicate_groups") if isinstance(out, dict) else None
        if not isinstance(groups, list):
            raise ValueError("no duplicate_groups list")
    except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
        raise
    except Exception as e:  # noqa: BLE001 - advisory step: never fail the run, record why nothing was grouped
        for ts in multi.values():
            for t in ts:
                t["meta"]["verify"]["lesson_dup_error"] = f"{type(e).__name__}: {e}"[:200]
        return
    ids = {t["id"]: t for ts in multi.values() for t in ts}
    for g in groups:
        members = [ids[i] for i in g if isinstance(i, str) and i in ids] if isinstance(g, list) else []
        if len(members) < 2 or len({m["meta"].get("seed_id") for m in members}) > 1:
            continue  # only within one seed
        members.sort(key=lambda t: (bool(judge_flags(t["meta"]["verify"].get("judge"))), len(json.dumps(t["messages"]))))
        for dup in members[1:]:
            dup["meta"]["verify"]["lesson_dup_of"] = members[0]["id"]


def run_arm(cfg: Config, adir: Path, jobs: list[dict], gen_fn, seeds: list[dict], eval_ng: set, observed_contracts: str | None = None) -> dict:
    """observed_contracts=None judges without seed context (the Simia baseline arm, which has no seed variation)."""
    by_id = {s["id"]: s for s in seeds}
    cfg.tool_optional_args = observed_optional_args(seeds)
    cfg.tool_loose_type_args = observed_loose_type_args(seeds)
    gen = parallel_map(gen_fn, jobs, key=lambda j: j["job_id"], out_path=adir / "generated.jsonl", workers=cfg.workers,
                       done_key=lambda t: t["id"], desc=f"generate {adir.name}")
    ver = parallel_map(lambda t: verify(cfg, t, eval_ng, expansion_context(t, by_id, observed_contracts) if observed_contracts is not None else None),
                       gen, key=lambda t: t["id"], out_path=adir / "verified.jsonl", workers=cfg.workers,
                       desc=f"verify {adir.name}")
    if observed_contracts is not None and cfg.features.lesson_dedup:  # seed-expansion arms only
        ver = mark_lesson_duplicates(cfg, ver)
        write_jsonl(adir / "verified.jsonl", ver)
    report = pilot_report(cfg, ver, seeds, attempted=len(jobs))
    (adir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def coverage_text(cfg: Config, root: Path, spec: dict, dev: list[dict], labels: dict) -> str:
    """What earlier rounds already covered, for the next revision: finished (seed, variation) cells with their outcome,
    and how many untried cells the current variations still leave. Those cells are never rerun, so a revision that
    keeps the same variations on the same seeds produces nothing new."""
    by_id = {s["id"]: s for s in dev}
    by_name = {v["name"]: v for v in spec["variations"]}
    rows = []
    ledger = read_jsonl(root / "ledger.jsonl")
    for r in ledger:
        v = by_name.get(r["variation"]) or {}
        outcome = "refused" if r.get("refused") else ("kept" if r.get("kept") else "rejected")
        rows.append(f'- seed {str(r["seed_id"])[:16]} x {r["variation"]} [{v.get("decision_type", "?")}]: {outcome}; '
                    f'{str(v.get("change", ""))[:110]}')
    done = {(r["seed_id"], r["variation"]) for r in ledger}
    untried = sum(1 for s in dev for v in applicable(spec["variations"], s, labels) if (s["id"], v["name"]) not in done)
    seeds_open = sorted({s["id"][:16] for s in dev if any((s["id"], v["name"]) not in done for v in applicable(spec["variations"], s, labels))})
    return (f"{len(done)} cells finished (listed below). Untried cells left under the current variations: {untried}; "
            f"the next round plans up to {cfg.pilot_jobs} jobs.\nSeeds that still have an untried cell: {seeds_open}\n" + "\n".join(rows))


def analyze(cfg: Config, dev: list[dict], heldout: list[dict], eval_ng: set) -> dict:
    root = cfg.out / "analyzer"
    root.mkdir(parents=True, exist_ok=True)
    obs = ObsBank.from_traces(dev)  # held-out seeds never reach the analyzer or the banks
    dev_by_id = {s["id"]: s for s in dev}
    labels = system_labels(dev)
    (root / "system_labels.json").write_text(json.dumps(labels, indent=1))
    inputs = _analyzer_inputs(cfg, dev, obs, labels)
    holdout_texts = [m["content"] for s in heldout for m in s["messages"] if m["role"] in ("user", "tool")]
    versions, prev = [], None
    for v in range(1, cfg.analyzer_rounds + 1):
        vdir = root / f"v{v}"
        spec = write_version(cfg, vdir, v, inputs, prev, holdout_texts, labels)
        jobs = read_jsonl(vdir / "jobs.jsonl")
        if not jobs:  # planned once per version, so a resumed run keeps the same jobs
            jobs = plan_jobs(f"v{v}", dev, cfg.pilot_jobs, spec["variations"], labels, read_jsonl(root / "ledger.jsonl"),
                             spec.get("combinations"), cfg.combo_share, max_per_seed=cfg.max_jobs_per_seed, mode="analyzer", version=v, case_block=cfg.case_block)
            (vdir / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
        report = run_arm(cfg, vdir, jobs, lambda j, t=spec["template"]: generate_analyzer(cfg, j, dev_by_id[j["seed_id"]], t, obs),
                         dev, eval_ng, observed_contracts=spec["observed_contracts"])
        _append_ledger(root, read_jsonl(vdir / "verified.jsonl"))
        versions.append({"version": v, "usable_rate": report["usable_rate"], "clean_rate": report["clean_rate"]})
        print(f"analyzer v{v}: usable {report['unique_kept']}/{report['jobs']}, format_ok {report['format_ok']}")
        if v == cfg.analyzer_rounds or report["usable_rate"] >= cfg.analyzer_round2_below:  # good enough: no revision
            break
        notes = diagnose(cfg, vdir, v, spec, report, read_jsonl(vdir / "verified.jsonl"), dev_by_id)
        prev = {"version": v, "template": spec["template"], "observed_contracts": spec["observed_contracts"],
                "variations": (json.dumps(spec["variations"], ensure_ascii=False, indent=1) if len(spec["variations"]) <= 20 else
                               "\n".join(f'- {x["name"]} [{x.get("decision_type")}/{x.get("answer_transition")}] seeds={len(x.get("eligible_seeds") or [])}: {str(x.get("change"))[:120]}'
                                         for x in spec["variations"])),
                "signals": json.dumps(spec["signals"], ensure_ascii=False, indent=1), "notes": notes,
                "covered": coverage_text(cfg, root, spec, dev, labels)}
    _all_rounds(cfg, root, len(versions), dev)
    best = max(versions, key=lambda x: (x["clean_rate"], x["usable_rate"], x["version"]))  # critique counts, not just code checks
    chosen = {"version": best["version"], "by": "highest clean rate (kept by code checks and not criticised by the judge; ties: usable rate, then later version); provisional, review the reports",
              "dev_usable_rates": {f"v{x['version']}": x["usable_rate"] for x in versions},
              "dev_clean_rates": {f"v{x['version']}": x["clean_rate"] for x in versions}}
    bdir = root / f"v{best['version']}"
    (root / "chosen.txt").write_text((bdir / "prompt.txt").read_text())
    (root / "chosen_variations.json").write_text((bdir / "variations.json").read_text())
    (root / "chosen_observed_contracts.md").write_text((bdir / "observed_contracts.md").read_text())
    (root / "chosen.json").write_text(json.dumps(chosen, indent=1))
    return chosen


def run_batch(cfg: Config, dev: list[dict], eval_ng: set, version: int, n: int) -> dict:
    """Scale step: freeze analyzer version <version> (its template and observed contracts), open fresh cells with one short
    proposal call per seed (finished cells are shown so they are not repeated), plan n jobs over UNTRIED cells only, and
    generate them all with that single template. Output goes to analyzer/batch; the ledger is appended."""
    import math
    from concurrent.futures import ThreadPoolExecutor
    root = cfg.out / "analyzer"
    vdir = root / f"v{version}"
    template = (vdir / "prompt.txt").read_text()
    contracts = (vdir / "observed_contracts.md").read_text()
    variations = json.loads((vdir / "variations.json").read_text())
    labels = system_labels(dev)
    obs = ObsBank.from_traces(dev)
    dev_by_id = {s["id"]: s for s in dev}
    bdir = root / "batch"
    bdir.mkdir(parents=True, exist_ok=True)
    if (bdir / "variations.json").exists():
        variations = json.loads((bdir / "variations.json").read_text())
    else:
        covered = read_jsonl(root / "ledger.jsonl")
        by_key = {k: lab for lab, ks in labels.items() for k in label_keys(ks)}

        def one(seed):
            try:
                return propose_for_seed(cfg, seed, by_key.get(system_key(seed.get("system", "")), "all"), labels, dev,
                                        contracts, variations, covered, cfg.variations_per_seed)
            except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                raise
            except Exception:  # noqa: BLE001 - a failed proposal call only means fewer cells for that seed
                return []
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            extra = list(ex.map(one, dev))
        names = {x["name"] for x in variations}
        for vs in extra:
            for x in vs:
                if x["name"] in names:
                    x["name"] = f'{x["name"]}__{x["eligible_seeds"][0][-6:]}'
                names.add(x["name"])
                variations.append(x)
        (bdir / "variations.json").write_text(json.dumps(variations, ensure_ascii=False, indent=1))
        (bdir / "proposal_rejections.json").write_text(json.dumps(getattr(propose_for_seed, "rejected", []), ensure_ascii=False, indent=1))
    jobs = read_jsonl(bdir / "jobs.jsonl")
    if not jobs:
        jobs = plan_jobs("batch", dev, n, variations, labels, read_jsonl(root / "ledger.jsonl"), None, 0.0,
                         max_per_seed=math.ceil(n / max(1, len(dev))) + 1, max_per_variation=max(2, len(dev)),
                         max_per_family=max(1, math.ceil(n * cfg.max_variation_share)) if cfg.max_variation_share < 1 else None,
                         mode="analyzer", version="batch", case_block=cfg.case_block)
        (bdir / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    print(f"batch: {len(variations)} variations, {len(jobs)} jobs planned (asked for {n})")
    report = run_arm(cfg, bdir, jobs, lambda j: generate_analyzer(cfg, j, dev_by_id[j["seed_id"]], template, obs), dev, eval_ng,
                     observed_contracts=contracts)
    _append_ledger(root, read_jsonl(bdir / "verified.jsonl"))
    return report


def _all_rounds(cfg: Config, root: Path, n_versions: int, seeds: list[dict]) -> None:
    """Kept traces of every version pooled and deduplicated together (an earlier round's trace may still be good)."""
    pooled = [t for v in range(1, n_versions + 1) for t in read_jsonl(root / f"v{v}" / "verified.jsonl") if t["meta"]["verify"]["kept"]]
    unique, dup = dedup(pooled, near=True)
    cells = Counter(c for t in unique for c in decision_cells(t))
    pairs = Counter((t["meta"].get("seed_id"), t["meta"].get("variation")) for t in unique)
    (root / "all_rounds.json").write_text(json.dumps({
        "kept_all_rounds": len(pooled), "unique_after_cross_round_dedup": len(unique), "dedup": dup,
        "by_version": dict(Counter(t["meta"].get("version") for t in unique)),
        "seed_variation_pairs": len(pairs), "max_per_pair": max(pairs.values(), default=0),
        "decision_cells": {"distinct": len(cells), "max_repeats": max(cells.values(), default=0)},
        "outcomes": outcomes(unique, cfg.outcome_fields)}, indent=1, default=str))


def compare_heldout(cfg: Config, root: Path, template: str, variations: list[dict], heldout: list[dict], dev: list[dict],
                    eval_ng: set) -> dict:
    """Chosen analyzer prompt vs simia_prompt=fixed (no other features), same held-out seeds and job count.
    Run only on request (`simia-plus heldout`), never as part of `analyze`."""
    from .generate import generate_simia

    obs, users = ObsBank.from_traces(dev), UserBank.from_traces(dev)
    by_id = {s["id"]: s for s in heldout}
    labels = json.loads((root / "system_labels.json").read_text())
    n = len(heldout) * cfg.heldout_jobs_per_seed
    hdir = root / "heldout"
    jobs = plan_jobs("heldout-analyzer", heldout, n, variations, labels, [], mode="analyzer")
    rep = {"analyzer": run_arm(cfg, hdir / "analyzer", jobs,
                               lambda j: generate_analyzer(cfg, j, by_id[j["seed_id"]], template, obs), heldout, eval_ng,
                               observed_contracts=(root / "chosen_observed_contracts.md").read_text())}
    base = copy.deepcopy(cfg)
    base.simia_prompt, base.prompt_source = "fixed", "simia"
    base.features.strategies = base.features.retrieval = False

    def gen_simia(j: dict) -> dict:
        t = generate_simia(base, j, by_id[j["seed_id"]], by_id, obs, users)
        t["meta"].update({k: j.get(k) for k in ("job_id", "seed_id", "strategy", "mode")})
        t["meta"]["fidelity"] = fidelity(t, by_id[j["seed_id"]], cfg.tool_result_prefix)
        return t

    simia_jobs = [{"job_id": f"heldout-simia/{s['id']}__{k}", "seed_id": s["id"], "strategy": "new_scenario", "mode": "simia"}
                  for k in range(cfg.heldout_jobs_per_seed) for s in heldout]
    rep["simia_fixed"] = run_arm(cfg, hdir / "simia_fixed", simia_jobs, gen_simia, heldout, eval_ng)
    summary = {arm: {k: r[k] for k in ("jobs", "format_ok", "kept", "unique_kept", "usable_rate", "checklist_failed",
                                       "outcomes_kept", "fidelity_kept")}
               | {"distinct_action_signatures": r["diversity_kept"]["distinct_action_signatures"]} for arm, r in rep.items()}
    (hdir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def run_ab(cfg: Config, dev: list[dict], eval_ng: set, version: int, n_jobs: int) -> dict:
    """Generator A/B on identical jobs: one analyzer version's template with the plain output format ("plain") vs with
    the case block first ("case"). Both arms get the same checks; only the case arm
    has a case for the judge to check. Run only on request (`simia-plus ab`)."""
    root = cfg.out / "analyzer"
    vdir = root / f"v{version}"
    if not (vdir / "prompt.txt").exists():
        raise SystemExit(f"no analyzer version {version} under {root}")
    spec = write_version(cfg, vdir, version, {}, None, [], {})
    labels = json.loads((root / "system_labels.json").read_text())
    obs, by_id = ObsBank.from_traces(dev), {s["id"]: s for s in dev}
    adir = root / f"ab_v{version}"
    adir.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(adir / "jobs.jsonl")
    if not base:  # planned once, so both arms and any resumed run share the same seed/variation assignments
        base = plan_jobs("ab", dev, n_jobs, spec["variations"], labels, [], spec.get("combinations"), cfg.combo_share,
                         mode="analyzer", version=version)
        (adir / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in base))
    abcfg = copy.deepcopy(cfg)
    abcfg.features.judge = True
    summary = {}
    for arm, case in (("plain", False), ("case", True)):
        prefix = f"ab_v{version}/{arm}/"
        jobs = [{**j, "job_id": prefix + j["job_id"].split("/", 1)[1], "case_block": case, "arm": arm} for j in base]
        r = run_arm(abcfg, adir / arm, jobs, lambda j, t=spec["template"]: generate_analyzer(abcfg, j, by_id[j["seed_id"]], t, obs),
                    dev, eval_ng, observed_contracts=spec["observed_contracts"])
        calls = read_jsonl(cfg.out / "llm_calls.jsonl")
        cost = sum(c.get("cost") or 0 for c in calls if str(c.get("item") or "").startswith(prefix))
        summary[arm] = {k: r[k] for k in ("jobs", "format_ok", "kept", "unique_kept",
                                          "eval_errors", "judge_retried", "reject_reasons",
                                          "checklist_failed", "by_variation", "outcomes_kept", "outcomes_by_variation",
                                          "fidelity_kept")} | {
            "nearest_kept": {k: r["nearest_kept"][k] for k in ("at_least_0.8", "at_least_0.6")},
            "decision_cells_kept": r["decision_cells_kept"], "cost_usd": round(cost, 4),
            "cost_per_unique_kept": round(cost / r["unique_kept"], 4) if r["unique_kept"] else None}
    (adir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def run_ab_split(cfg: Config, dev: list[dict], eval_ng: set, n_jobs: int) -> dict:
    """Split-generation A/B on identical jobs: the chosen analyzer version's template, generated as now ("current": one
    call writes observations and answer) vs split ("split": the generator writes the world without the intended
    outcome, the agent answers blind). Same judge and checks for both. Jobs come from the
    batch plan when there is one, else from a fresh plan. Run only on request (`simia-plus ab-split`)."""
    root = cfg.out / "analyzer"
    version = json.loads((root / "chosen.json").read_text())["version"]
    vdir = root / f"v{version}"
    template, contracts = (vdir / "prompt.txt").read_text(), (vdir / "observed_contracts.md").read_text()
    obs, by_id = ObsBank.from_traces(dev), {s["id"]: s for s in dev}
    adir = root / "ab_split"
    adir.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(adir / "jobs.jsonl")
    if not base:  # planned once, so both arms and any resumed run share the same seed/variation assignments
        pool = [j for j in read_jsonl(root / "batch" / "jobs.jsonl") if j["seed_id"] in by_id]
        if not pool:
            pool = plan_jobs("ab", dev, n_jobs, json.loads((vdir / "variations.json").read_text()),
                             json.loads((root / "system_labels.json").read_text()), [], mode="analyzer", version=version)
        rng = random.Random(cfg.random_seed)
        base = rng.sample(pool, min(n_jobs, len(pool)))
        (adir / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in base))
    summary = {}
    for arm, split in (("current", False), ("split", True)):
        acfg = copy.deepcopy(cfg)
        acfg.features.split_answer = split
        jobs = [{**j, "job_id": f"ab_split/{arm}/{j['job_id'].split('/', 1)[-1]}", "case_block": cfg.case_block, "arm": arm}
                for j in base]
        r = run_arm(acfg, adir / arm, jobs, lambda j, c=acfg: generate_analyzer(c, j, by_id[j["seed_id"]], template, obs),
                    dev, eval_ng, observed_contracts=contracts)
        ver = read_jsonl(adir / arm / "verified.jsonl")
        kept = [t for t in ver if t["meta"]["verify"].get("kept")]
        integ = Counter(k for t in ver for k in ((t["meta"]["verify"].get("rubric") or {}).get("integrity_failed") or []))
        checks = [t["meta"].get("answer_check") or {} for t in ver]
        agree = {"rollout_issues": dict(Counter((c.get("issue") or "").split(":")[0] for c in checks if c.get("issue"))),
                 "simulated_calls": sum(c.get("simulated_calls", 0) for c in checks)} if split else {}
        summary[arm] = {"jobs": len(jobs), "kept": len(kept), "reject_reasons": r.get("reject_reasons"),
                        "integrity_failed": dict(integ), "rollout": agree,
                        "judge_labels": dict(Counter((t["meta"]["verify"].get("rubric") or {}).get("label") for t in ver))}
    (adir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ---- wider variation A/B (ab-vary) ----

def coverage_summary(seeds: list[dict], variations: list[dict], fields: list[str] | None) -> str:
    """What already exists, most common first: decision types and answer transitions of existing variations, the seeds'
    own answer labels, and the tools the seeds call. Shown to the proposal call so it goes where little exists."""
    dt = Counter(str(v.get("decision_type") or "?").strip().lower() for v in variations)
    tr = Counter(str(v.get("answer_transition") or "?").strip().lower() for v in variations)
    outs = Counter(f"{k}={v}" for s in seeds for k, v in answer_labels(final_answer(s), fields))
    calls = Counter(c["name"] for s in seeds for m in s["messages"] for c in m.get("tool_calls", []))
    fmt = lambda c: ", ".join(f"{k} ({n})" for k, n in c.most_common(25)) or "(none)"
    return (f"decision types of existing variations: {fmt(dt)}\nanswer transitions of existing variations: {fmt(tr)}\n"
            f"answer labels in the seeds: {fmt(outs)}\ntool calls in the seeds: {fmt(calls)}")


def propose_wide(cfg: Config, seed: dict, label: str, labels: dict, dev: list[dict], contracts: str, existing: list[dict],
                 coverage: str, n: int, n_siblings: int = 2) -> list[dict]:
    """One proposal call for one seed: n variations of kinds task / evidence (SEED_VARIATIONS_WIDE)."""
    keys = set(label_keys(labels.get(label, [])))
    have = {t["name"] for t in seed["tools"]}
    sibs = [s for s in dev if s["id"] != seed["id"] and system_key(s.get("system", "")) in keys]
    sibs = random.Random(f"{cfg.random_seed}:{seed['id']}").sample(sibs, min(n_siblings, len(sibs)))
    prompt = SEED_VARIATIONS_WIDE.format(
        observed_contracts=contracts, label=label, system=seed.get("system", ""), seed_id=seed["id"], seed=seed_json(seed),
        tools=tools_text(seed["tools"]) if seed["tools"] else "(none: this agent has no tools)",
        siblings="\n".join(f'<sibling id="{s["id"]}">\n{_head_tail(seed_json(s), 4000)}\n</sibling>' for s in sibs) or "(none)",
        coverage=coverage, covered=", ".join(sorted({x["name"] for x in existing})[:80]) or "(none)", n=n)
    out = chat_json(get_llm(cfg, "analyzer"), "You are an expert in agent training data. Output JSON only.", prompt)
    vs = out.get("variations") if isinstance(out, dict) else None
    ids = {s["id"] for s in dev}
    good = []
    rejected = propose_wide.rejected = getattr(propose_wide, "rejected", [])
    for v in vs if isinstance(vs, list) else []:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        v["applies_to"] = [label]
        v["eligible_seeds"] = [seed["id"]]  # wider changes are written for this seed only
        errs = variation_errors(v, labels, sorted(ids), contracts)
        if v.get("kind") not in VARIATION_KINDS:
            errs.append(f"kind must be one of {list(VARIATION_KINDS)}")
        if set(v.get("requires_tools") or []) - have:
            errs.append(f"requires_tools not in this seed's tools: {sorted(set(v.get('requires_tools') or []) - have)}")
        if errs:
            rejected.append({"seed": seed["id"], "name": v.get("name"), "kind": v.get("kind"), "errors": errs[:3]})
        else:
            good.append(v)
    return good


def _head_tail(text: str, limit: int) -> str:
    """text, or its start and end around a cut marker when longer than limit (so the shown part is never mid-cut silently)."""
    if len(text) <= limit:
        return text
    return text[: limit * 2 // 3] + f"\n... [{len(text) - limit} characters not shown] ...\n" + text[-limit // 3:]


def _containment(a: set, b: set) -> float:
    return len(a & b) / len(a) if a else 0.0


def variety(kept: list[dict], seeds: list[dict], fields: list[str] | None, size: int | None = None, rng_seed: int = 0) -> dict:
    """How far kept traces move from the seeds and from each other, and how spread their answers and action sequences are.
    Similarity is containment of a trace's word 3-grams (non-user text) in the closest seed: the share of the trace that
    already exists in some seed, so length alone does not make a trace look new. Spread measures are
    taken on an equal-size random subsample (size) so an arm with more kept traces does not look more varied."""
    sample = random.Random(rng_seed).sample(kept, size) if size is not None and len(kept) > size else list(kept)
    sh = [_shingles(body_text(s)) for s in seeds]
    rows = [_shingles(body_text(t)) for t in sample]
    near = [max((_containment(a, b) for b in sh), default=0.0) for a in rows]
    pair = [_containment(rows[i], rows[j]) for i in range(len(rows)) for j in range(len(rows)) if i != j]
    outs = Counter(" / ".join(f"{k}={v}" for k, v in answer_labels(final_answer(t), fields)) for t in sample)
    labelled = "" not in outs or len(outs) > 1
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    sigs = Counter(action_signature(t) for t in sample)
    seed_sigs = {action_signature(s) for s in seeds}
    return {"n": len(sample), "mean_share_found_in_closest_seed": mean(near), "at_least_0.6_found_in_a_seed": sum(x >= 0.6 for x in near),
            "mean_share_found_in_another_kept": mean(pair),
            "distinct_answers": len(outs) if labelled else "n/a (free-text answers)",
            "top_answer_share": round(outs.most_common(1)[0][1] / len(sample), 3) if sample and labelled else None,
            "distinct_action_signatures": len(sigs), "action_signatures_in_no_seed": sum(1 for x in sigs if x not in seed_sigs)}


def _stack_pairs(variations: list[dict], seed: dict, labels: dict) -> list[dict]:
    """Two applicable variations of different decision types on one seed, combined into one job (combine())."""
    app = applicable(variations, seed, labels)
    out, used = [], set()
    for i, a in enumerate(app):
        for b in app[i + 1:]:
            if a["name"] in used or b["name"] in used or str(a.get("decision_type")).lower() == str(b.get("decision_type")).lower():
                continue
            out.append(combine(a, b))
            used |= {a["name"], b["name"]}
            break
    return out


def _wide_variations(cfg: Config, adir: Path, fname: str, seed_ids: list[str], dev: list[dict], labels: dict, contracts: str,
                     batch_vars: list[dict], item_prefix: str, n: int) -> list[dict]:
    """One wide proposal call per seed (cached in adir/fname); failures and rejected proposals are written next to it.
    Proposal calls are logged under item_prefix/<seed id> so their cost is counted."""
    from concurrent.futures import ThreadPoolExecutor
    if (adir / fname).exists():
        return json.loads((adir / fname).read_text())
    by_id = {s["id"]: s for s in dev}
    by_key = {k: lab for lab, ks in labels.items() for k in label_keys(ks)}
    cov = {lab: coverage_summary([s for s in dev if by_key.get(system_key(s.get("system", ""))) == lab],
                                 [v for v in batch_vars if lab in (v.get("applies_to") or []) or "all" in (v.get("applies_to") or [])],
                                 cfg.outcome_fields) for lab in labels}
    failed = []

    def one(sid):
        from .llm import current_item
        current_item.set(f"{item_prefix}/{sid}")
        lab = by_key.get(system_key(by_id[sid].get("system", "")), "all")
        try:
            return propose_wide(cfg, by_id[sid], lab, labels, dev, contracts, batch_vars, cov.get(lab, ""), n)
        except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
            raise
        except Exception as e:  # noqa: BLE001 - a failed proposal only means fewer cells for that seed; logged
            failed.append({"seed": sid, "error": repr(e)[:300]})
            return []
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        wide = [v for vs in ex.map(one, seed_ids) for v in vs]
    names = Counter()
    for v in wide:
        names[v["name"]] += 1
        if names[v["name"]] > 1:
            v["name"] = f'{v["name"]}__{v["eligible_seeds"][0][-6:]}'
    (adir / fname).write_text(json.dumps(wide, ensure_ascii=False, indent=1))
    (adir / fname.replace(".json", "_rejections.json")).write_text(json.dumps(
        {"failed_calls": failed, "rejected": getattr(propose_wide, "rejected", [])}, ensure_ascii=False, indent=1))
    return wide


def run_ab_vary(cfg: Config, dev: list[dict], eval_ng: set, n_jobs: int) -> dict:
    """Variety A/B over the same seeds: "current" (sampled batch jobs, one evidence change), "vary" (task / evidence kinds
    proposed with coverage, kinds rotated per seed) and "stack" (two existing variations of different
    decision types applied together). Every arm gets the same template, checks and judge, plus variety measures. Run only on request (`simia-plus ab-vary`)."""
    root = cfg.out / "analyzer"
    version = json.loads((root / "chosen.json").read_text())["version"]
    vdir = root / f"v{version}"
    template, contracts = (vdir / "prompt.txt").read_text(), (vdir / "observed_contracts.md").read_text()
    labels = json.loads((root / "system_labels.json").read_text())
    batch_vars = json.loads(((root / "batch" / "variations.json") if (root / "batch" / "variations.json").exists()
                             else (vdir / "variations.json")).read_text())
    obs, by_id = ObsBank.from_traces(dev), {s["id"]: s for s in dev}
    adir = root / "ab_vary"
    adir.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(adir / "jobs_current.jsonl")
    if not base:
        pool = [j for j in read_jsonl(root / "batch" / "jobs.jsonl") if j["seed_id"] in by_id]
        if not pool:
            pool = plan_jobs("ab", dev, n_jobs, batch_vars, labels, [], mode="analyzer", version=version)
        base = random.Random(cfg.random_seed).sample(pool, min(n_jobs, len(pool)))
        write_jsonl(adir / "jobs_current.jsonl", base)
    seeds_needed = sorted({j["seed_id"] for j in base})
    wide = _wide_variations(cfg, adir, "variations_vary.json", seeds_needed, dev, labels, contracts, batch_vars,
                            "ab_vary/vary/propose", min(cfg.variations_per_seed, 3))  # 1-2 jobs per seed; more proposals add latency
    # same seed multiset in every arm: the k-th job on a seed takes that seed's k-th option (vary: kinds rotated)
    vary_q, stack_q = {}, {}
    for i, sid in enumerate(seeds_needed):
        mine = [v for v in wide if v["eligible_seeds"][0] == sid]
        kinds = VARIATION_KINDS[i % len(VARIATION_KINDS):] + VARIATION_KINDS[:i % len(VARIATION_KINDS)]  # each seed starts at a different kind
        by_kind = [[v for v in mine if v["kind"] == k] for k in kinds]
        vary_q[sid] = [v for grp in zip(*[g + [None] * (len(mine) - len(g)) for g in by_kind]) for v in grp if v]
        stack_q[sid] = _stack_pairs(batch_vars, by_id[sid], labels)
    # only jobs every arm can fill, so all arms run the same seed multiset
    arms = {"current": [], "vary": [], "stack": []}
    k = Counter()
    for j in base:
        sid = j["seed_id"]
        if k[sid] < len(vary_q[sid]) and k[sid] < len(stack_q[sid]):
            arms["current"].append(j)
            arms["vary"].append({**j, "variation": vary_q[sid][k[sid]]})
            arms["stack"].append({**j, "variation": stack_q[sid][k[sid]]})
        k[sid] += 1
    summary = {"jobs_dropped_no_option": len(base) - len(arms["current"])}
    kept_by_arm = {}
    for arm, jobs in arms.items():
        jobs = [{**j, "job_id": f"ab_vary/{arm}/{j['job_id'].split('/', 1)[-1]}", "case_block": cfg.case_block, "arm": arm} for j in jobs]
        r = run_arm(cfg, adir / arm, jobs, lambda j: generate_analyzer(cfg, j, by_id[j["seed_id"]], template, obs),
                    dev, eval_ng, observed_contracts=contracts)
        ver = read_jsonl(adir / arm / "verified.jsonl")
        kept = [t for t in ver if t["meta"]["verify"].get("kept")]
        kinds = Counter(str((t["meta"].get("variation_spec") or {}).get("kind") or arm) for t in ver)
        kinds_kept = Counter(str((t["meta"].get("variation_spec") or {}).get("kind") or arm) for t in kept)
        calls = read_jsonl(cfg.out / "llm_calls.jsonl")
        cost = sum(c.get("cost") or 0 for c in calls if str(c.get("item") or "").startswith(f"ab_vary/{arm}/"))
        kept_by_arm[arm] = kept
        summary[arm] = {"jobs": len(jobs), "kept": len(kept), "refused": r["outcome_split"]["refusals"],
                        "reject_reasons": r.get("reject_reasons"), "kept_by_kind": {k: f"{kinds_kept[k]}/{kinds[k]}" for k in kinds},
                        "judge_labels": dict(Counter((t["meta"]["verify"].get("rubric") or {}).get("label") for t in ver)),
                        "logged_cost_usd": round(cost, 4)}
    size = min(len(v) for v in kept_by_arm.values())  # equal-size subsamples for the spread measures
    for arm, kept in kept_by_arm.items():
        summary[arm]["variety"] = variety(kept, dev, cfg.outcome_fields, size, cfg.random_seed)
    (adir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


MIXED_TYPES = ("old", "task", "evidence", "stack")


def run_mixed(cfg: Config, dev: list[dict], eval_ng: set, n_seeds: int) -> dict:
    """Pilot of generating every variation type and letting the checks and judge keep what is good. For up to n_seeds
    seeds, one job per available type: "old" (an existing batch variation, one evidence change), "task" and "evidence"
    (wide proposals, SEED_VARIATIONS_WIDE), "stack" (two existing variations of different decision types). Same template,
    checks and judge for all; per type: yield, variety. Run only on request (`simia-plus mixed`)."""
    root = cfg.out / "analyzer"
    version = json.loads((root / "chosen.json").read_text())["version"]
    vdir = root / f"v{version}"
    template, contracts = (vdir / "prompt.txt").read_text(), (vdir / "observed_contracts.md").read_text()
    labels = json.loads((root / "system_labels.json").read_text())
    batch_vars = json.loads(((root / "batch" / "variations.json") if (root / "batch" / "variations.json").exists()
                             else (vdir / "variations.json")).read_text())
    obs, by_id = ObsBank.from_traces(dev), {s["id"]: s for s in dev}
    adir = root / "mixed"
    adir.mkdir(parents=True, exist_ok=True)
    jobs = read_jsonl(adir / "jobs.jsonl")
    if not jobs:
        seed_ids = sorted(random.Random(cfg.random_seed).sample(sorted(by_id), min(n_seeds, len(by_id))))
        wide = _wide_variations(cfg, adir, "variations_wide.json", seed_ids, dev, labels, contracts, batch_vars,
                                "mixed/propose", min(cfg.variations_per_seed, 3))
        for sid in seed_ids:
            mine = [v for v in wide if v["eligible_seeds"][0] == sid]
            options = {"old": applicable(batch_vars, by_id[sid], labels)[:1],
                       "task": [v for v in mine if v["kind"] == "task"][:1],
                       "evidence": [v for v in mine if v["kind"] == "evidence"][:1],
                       "stack": _stack_pairs(batch_vars, by_id[sid], labels)[:1]}
            jobs += [{"job_id": f"mixed/{typ}/{sid}", "seed_id": sid, "strategy": "analyzer", "mode": "analyzer",
                      "version": version, "case_block": cfg.case_block, "arm": typ, "variation": vs[0]}
                     for typ, vs in options.items() if vs]
        write_jsonl(adir / "jobs.jsonl", jobs)
    run_arm(cfg, adir, jobs, lambda j: generate_analyzer(cfg, j, by_id[j["seed_id"]], template, obs), dev, eval_ng,
            observed_contracts=contracts)
    ver = read_jsonl(adir / "verified.jsonl")
    kept = [t for t in ver if t["meta"]["verify"].get("kept")]
    calls = read_jsonl(cfg.out / "llm_calls.jsonl")
    by_type = {typ: [t for t in ver if t["meta"].get("arm") == typ] for typ in MIXED_TYPES}
    kept_n = [sum(bool(t["meta"]["verify"].get("kept")) for t in ts) for ts in by_type.values()]
    size = min((x for x in kept_n if x), default=0)  # equal-size subsamples over the types that kept anything
    summary = {"seeds": len({j["seed_id"] for j in jobs}), "jobs": len(jobs), "kept": len(kept),
               "proposal_cost_usd": round(sum(c.get("cost") or 0 for c in calls if str(c.get("item") or "").startswith("mixed/propose/")), 4)}
    for typ, ts in by_type.items():
        k = [t for t in ts if t["meta"]["verify"].get("kept")]
        summary[typ] = {"jobs": len(ts), "kept": len(k), "refused": sum(bool(t["meta"].get("refusal")) for t in ts),
                        "rejected_by_judge": sum((t["meta"]["verify"].get("rubric") or {}).get("label") == "bad" for t in ts),
                        "rule_issues": dict(Counter(i for t in ts for i in t["meta"]["verify"].get("rule_issues") or [])),
                        "variety": variety(k, dev, cfg.outcome_fields, size, cfg.random_seed),
                        "logged_cost_usd": round(sum(c.get("cost") or 0 for c in calls if str(c.get("item") or "").startswith(f"mixed/{typ}/")), 4)}
    (adir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def run_ab_models(cfg: Config, dev: list[dict], eval_ng: set, n_jobs: int, arms: dict[str, dict]) -> dict:
    """Model-settings A/B on identical jobs (e.g. reasoning effort). arms = {arm: {role: {setting: value}}}; each arm
    overrides those roles' model settings (a role without its own model config starts from "default"; "extra" is merged
    into the API call). Jobs: a sample of the batch plan (one existing variation each), same template, checks and
    judge settings per arm as given. Per arm: yield, judge labels, cost, tokens and latency per role, from the call log.
    Run only on request (`simia-plus ab-models --arms <json>`)."""
    import dataclasses
    root = cfg.out / "analyzer"
    version = json.loads((root / "chosen.json").read_text())["version"]
    vdir = root / f"v{version}"
    template, contracts = (vdir / "prompt.txt").read_text(), (vdir / "observed_contracts.md").read_text()
    obs, by_id = ObsBank.from_traces(dev), {s["id"]: s for s in dev}
    adir = root / "ab_models"
    adir.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(adir / "jobs.jsonl")
    if not base:
        pool = [j for j in read_jsonl(root / "batch" / "jobs.jsonl") if j["seed_id"] in by_id]
        if not pool:
            pool = plan_jobs("ab", dev, n_jobs, json.loads((vdir / "variations.json").read_text()),
                             json.loads((root / "system_labels.json").read_text()), [], mode="analyzer", version=version)
        base = random.Random(cfg.random_seed).sample(pool, min(n_jobs, len(pool)))
        write_jsonl(adir / "jobs.jsonl", base)
    (adir / "arms.json").write_text(json.dumps(arms, indent=1))
    summary = {}
    for arm, roles in arms.items():
        acfg = copy.deepcopy(cfg)
        for role, over in roles.items():
            m = copy.deepcopy(acfg.models.get(role) or acfg.models["default"])
            over = dict(over)
            extra = {**m.extra, **over.pop("extra", {})}
            acfg.models[role] = dataclasses.replace(m, extra=extra, **over)
        prefix = f"ab_models/{arm}/"
        jobs = [{**j, "job_id": prefix + j["job_id"].split("/", 1)[-1], "case_block": cfg.case_block, "arm": arm} for j in base]
        r = run_arm(acfg, adir / arm, jobs, lambda j, c=acfg: generate_analyzer(c, j, by_id[j["seed_id"]], template, obs),
                    dev, eval_ng, observed_contracts=contracts)
        ver = read_jsonl(adir / arm / "verified.jsonl")
        calls = [c for c in read_jsonl(cfg.out / "llm_calls.jsonl") if str(c.get("item") or "").startswith(prefix)]
        per_role = {}
        for role in sorted({c.get("role") for c in calls}):
            rc = [c for c in calls if c.get("role") == role]
            ok = [c for c in rc if c.get("ok")]
            u = [c.get("usage") or {} for c in ok]
            mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
            per_role[role] = {"calls": len(rc), "failed": len(rc) - len(ok), "cost_usd": round(sum(c.get("cost") or 0 for c in rc), 4),
                              "mean_out_tokens": mean([x.get("completion_tokens") or 0 for x in u]),
                              "mean_reasoning_tokens": mean([(x.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for x in u]),
                              "mean_latency_s": mean([c.get("latency_s") or 0 for c in ok]),
                              "p90_latency_s": sorted(c.get("latency_s") or 0 for c in ok)[int(0.9 * (len(ok) - 1))] if ok else None}
        summary[arm] = {"settings": roles, "jobs": len(jobs), "kept": r["kept"], "refused": r["outcome_split"]["refusals"],
                        "exec_errors": r["outcome_split"]["execution_errors"], "reject_reasons": r.get("reject_reasons"),
                        "judge_labels": dict(Counter((t["meta"]["verify"].get("rubric") or {}).get("label") for t in ver)),
                        "cost_usd": round(sum(c.get("cost") or 0 for c in calls), 4),
                        "cost_per_kept": round(sum(c.get("cost") or 0 for c in calls) / r["kept"], 4) if r["kept"] else None,
                        "by_role": per_role}
    (adir / "comparison.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
