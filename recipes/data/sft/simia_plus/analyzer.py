"""Analyzer-written generation prompt for seed expansion (ANALYZER_SPEC.md).

An analyzer agent studies the harness source and the dev seeds and writes three things: an inspectable harness
description, a list of grounded variations, and the generation prompt (a template). Each job pairs ONE seed with ONE
applicable variation, chosen by code from a ledger so seed/variation pairs rotate instead of repeating. Code fills the
per-job slots and appends the one fixed rule: output a single {"messages": [...]} JSON object, validated strictly
(a malformed output is rejected and logged, never repaired). A pilot runs each version; a separate diagnosis agent
reads the results and the analyzer revises, for at most `analyzer_rounds` versions.

Fidelity to the seed (which failure-context fields changed, which tool results were reused) and decision-cell
coverage are measured in code and reported as diagnostics; they never reject a trace on their own.

Everything lands under <out_dir>/analyzer/: per version harness.md, variations.json, prompt.txt, rationale.md,
analyzer_raw.json, generated/verified traces, report.json, diagnosis.md; plus ledger.jsonl, system_labels.json,
all_rounds.json, chosen.json and heldout/.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .banks import ObsBank, UserBank, format_obs_examples
from .config import Config
from .io import parallel_map, read_jsonl
from .llm import chat_json, get_llm
from .prompts import (
    ANALYZER,
    ANALYZER_REQUIRED_SLOTS,
    ANALYZER_REVISION,
    ANALYZER_SLOTS,
    DIAGNOSE,
    OUTPUT_FORMAT,
    VARIATION_KEYS,
)
from .schema import render
from .select import dedup, diversity_report
from .simia_text import tools_text
from .verify import final_answer, system_key, verify

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


def check_variations(variations, labels: dict[str, str]) -> list[str]:
    if not isinstance(variations, list) or len(variations) < 3:
        return ["variations must be a list of at least 3 objects"]
    errs, names = [], set()
    for i, v in enumerate(variations):
        if not isinstance(v, dict):
            errs.append(f"variation {i} is not an object")
            continue
        missing = [k for k in VARIATION_KEYS if not v.get(k)]
        if missing:
            errs.append(f"variation {i} ({v.get('name')}) lacks {missing}")
        if v.get("name") in names:
            errs.append(f"duplicate variation name {v.get('name')}")
        names.add(v.get("name"))
        bad = [a for a in v.get("applies_to") or [] if a != "all" and a not in labels]
        if bad:
            errs.append(f"variation {v.get('name')} applies_to unknown labels {bad}")
    return errs


def applicable(variations: list[dict], seed: dict, labels: dict[str, str]) -> list[dict]:
    label = next((lab for lab, key in labels.items() if key == system_key(seed.get("system", ""))), None)
    return [v for v in variations if "all" in v["applies_to"] or label in v["applies_to"]]


def fill(template: str, slots: dict[str, str]) -> str:
    out = template
    for name in ANALYZER_SLOTS:
        out = out.replace(f"{{{name}}}", slots.get(name, ""))
    return out + OUTPUT_FORMAT


def parse_output(text: str) -> tuple[list[dict] | None, list[str]]:
    """Strict check of the fixed output format. Returns (messages, issues); messages is None on any issue."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError as e:
        return None, [f"format: not a single JSON object ({e.msg})"]
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
    variation = {k: v for k, v in (job.get("variation") or {}).items() if k != "applies_to"}
    slots = {"seed_trace": seed_json(seed), "variation": json.dumps(variation, ensure_ascii=False, indent=1),
             "tools": tools_text(seed["tools"]), "system_prompt": seed.get("system", ""),
             "obs_examples": format_obs_examples(obs.examples_for_tools(seed["tools"], cfg.retrieved_obs_per_tool, rng), 800)}
    discarded = []
    attempts = max(1, cfg.generation_attempts)
    for attempt in range(1, attempts + 1):
        prompt = fill(template, {**slots, "generation_id": f"{job['job_id']}/{attempt}"})
        out = get_llm(cfg, "generator").chat([{"role": "user", "content": prompt}], json_mode=True)
        messages, issues = parse_output(out["content"])
        if not issues or attempt == attempts:
            break
        discarded.append({"attempt": attempt, "problem": issues, "raw_output": out["content"]})
    trace = to_trace(job, seed, messages)
    trace["meta"] = {"raw_output": out["content"], "format_issues": issues, "attempts": attempt,
                     "discarded_attempts": discarded, "variation": variation.get("name"),
                     **{k: job.get(k) for k in ("job_id", "seed_id", "strategy", "mode", "version")}}
    trace["meta"]["fidelity"] = fidelity(trace, seed, cfg.tool_result_prefix)
    return trace


def variation_for(job_id: str, seed: dict, variations: list[dict], labels: dict[str, str]) -> dict | None:
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
    return {"context_parsed": changed is not None, "context_fields_changed": changed,
            "platform_changed": (a or {}).get("platform") != (b or {}).get("platform") if changed is not None else None,
            "tool_results": len(obs), "tool_results_reused_from_seed": reused,
            "no_op": changed == [] and reused == len(obs) and calls(trace) == calls(seed)}


def decision_cells(trace: dict) -> list[tuple]:
    """One cell per assistant turn: evidence seen so far (tool, available, status) -> action taken."""
    seen, cells = [], []
    for m in trace["messages"]:
        if m["role"] == "tool":
            body = re.sub(r"^\[obs:\d+\]\s*", "", m["content"])
            try:
                o = json.loads(body)
            except json.JSONDecodeError:
                o = {}
            seen.append((m.get("name"), o.get("available") if isinstance(o, dict) else None,
                         o.get("status") if isinstance(o, dict) else None))
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                action = tuple(sorted(c["name"] for c in m["tool_calls"]))
            else:
                ans = final_answer(trace) if m is trace["messages"][-1] else None
                action = ("answer", (ans or {}).get("conclusion"), "cause" if (ans or {}).get("root_cause") else "null") \
                    if isinstance(ans, dict) else ("text",)
            cells.append((system_key(trace.get("system", ""))[:8], tuple(seen), action))
    return cells


# ---- analyzer and diagnosis agents ----

def system_labels(seeds: list[dict]) -> dict[str, str]:
    """S1..Sn for the distinct system prompts of the dev seeds (stable across rounds)."""
    keys: dict[str, str] = {}
    for s in sorted(seeds, key=lambda s: s["id"]):
        keys.setdefault(system_key(s.get("system", "")), f"S{len(keys) + 1}")
    return {label: key for key, label in keys.items()}


def _harness_text(paths: list[str]) -> str:
    parts = []
    for p in paths:
        path = Path(p)
        files = sorted(f for f in path.rglob("*") if f.is_file()) if path.is_dir() else [path]
        parts += [f"### FILE {f}\n{f.read_text(errors='replace')}" for f in files if f.suffix in (".py", ".json", ".md", ".yml", ".yaml")]
    return "\n\n".join(parts) or "(none given)"


def _analyzer_inputs(cfg: Config, seeds: list[dict], obs: ObsBank, labels: dict[str, str]) -> dict:
    rng = random.Random(cfg.random_seed)
    shown = seeds if len(seeds) <= cfg.analyzer_seeds else rng.sample(seeds, cfg.analyzer_seeds)
    by_key = {key: label for label, key in labels.items()}
    texts = {system_key(s.get("system", "")): s.get("system", "") for s in seeds}
    sys_text = "\n\n".join(f"### System prompt {label}\n{texts[key]}" for label, key in labels.items())
    parts = [f"### Seed {s['id']} (system prompt {by_key[system_key(s.get('system', ''))]})\n{seed_json(s)}" for s in shown]
    tools = {t["name"]: t for s in shown for t in s["tools"]}
    obs_ex = obs.examples_for_tools(list(tools.values()), 2, rng)
    return {"harness": _harness_text(cfg.harness_paths), "n_seeds": len(shown),
            "seeds": sys_text + "\n\n" + "\n\n".join(parts), "tools": tools_text(list(tools.values())),
            "obs": format_obs_examples(obs_ex, 800)}


def write_version(cfg: Config, vdir: Path, version: int, inputs: dict, prev: dict | None,
                  holdout_texts: list[str], labels: dict[str, str]) -> dict:
    """Ask the analyzer for version <version>; saves harness.md, variations.json, prompt.txt, rationale.md,
    analyzer_raw.json. Returns {"template", "variations", "harness"}."""
    if (vdir / "prompt.txt").exists():
        return {"template": (vdir / "prompt.txt").read_text(), "harness": (vdir / "harness.md").read_text(),
                "variations": json.loads((vdir / "variations.json").read_text())}
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
            check_variations(out.get("variations"), labels) + ([] if out.get("harness") else ["no harness description"])
        attempts.append({"output": out, "errors": errs})
        if not errs:
            break
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "analyzer_raw.json").write_text(json.dumps({"request": request, "attempts": attempts}, ensure_ascii=False, indent=1))
    if attempts[-1]["errors"]:
        raise RuntimeError(f"analyzer v{version} output failed checks: {attempts[-1]['errors']}")
    (vdir / "harness.md").write_text(out["harness"])
    (vdir / "variations.json").write_text(json.dumps(out["variations"], ensure_ascii=False, indent=1))
    (vdir / "prompt.txt").write_text(template)
    (vdir / "rationale.md").write_text(str(out.get("rationale", "")))
    return {"template": template, "variations": out["variations"], "harness": out["harness"]}


def diagnose(cfg: Config, vdir: Path, version: int, spec: dict, report: dict, verified: list[dict],
             seeds_by_id: dict[str, dict]) -> str:
    if (vdir / "diagnosis.md").exists():
        return (vdir / "diagnosis.md").read_text()
    rng = random.Random(f"{cfg.random_seed}:diag:{version}")
    rejected = [t for t in verified if not t["meta"]["verify"]["kept"]]
    kept = [t for t in verified if t["meta"]["verify"]["kept"]]
    rej_s, kept_s = rng.sample(rejected, min(8, len(rejected))), rng.sample(kept, min(6, len(kept)))

    def show(t: dict) -> str:
        v = t["meta"]["verify"]
        why = {"rule_issues": v["rule_issues"], "ungrounded_args": v.get("ungrounded_args"),
               "judge": {k: (v.get("judge") or {}).get(k) for k in ("task_success", "obs_consistent", "policy_followed",
                                                                    "hallucinations", "checklist_failed", "notes")},
               "fidelity": t["meta"].get("fidelity")}
        seed = seeds_by_id.get(t["meta"].get("seed_id"))
        seed_user = next((m["content"] for m in seed["messages"] if m["role"] == "user"), "") if seed else ""
        body = render(t, max_obs_chars=1500) if t["messages"] else f"(unparseable output)\n{t['meta'].get('raw_output', '')[:3000]}"
        return (f"### {t['id']} (seed {t['meta'].get('seed_id')}, variation {t['meta'].get('variation')})\n"
                f"seed first user message: {seed_user[:1500]}\nreasons: {json.dumps(why, ensure_ascii=False)}\n{body[:7000]}")

    out = chat_json(get_llm(cfg, "diagnose"), "You are a rigorous reviewer of agent training data. Output JSON only.",
                    DIAGNOSE.format(version=version, harness=spec["harness"],
                                    variations=json.dumps(spec["variations"], ensure_ascii=False, indent=1),
                                    template=spec["template"], report=json.dumps(report, indent=1, default=str),
                                    n_rejected=len(rej_s), total_rejected=len(rejected), n_kept=len(kept_s),
                                    total_kept=len(kept), rejected="\n\n".join(map(show, rej_s)) or "(none)",
                                    kept="\n\n".join(map(show, kept_s)) or "(none)"))
    notes = str(out.get("notes", "")) if isinstance(out, dict) else str(out)
    (vdir / "diagnosis.md").write_text(notes)
    return notes


# ---- planning, ledger and reports ----

def plan_jobs(prefix: str, seeds: list[dict], n: int, variations: list[dict], labels: dict[str, str],
              ledger: list[dict], **extra) -> list[dict]:
    """Round-robin over seeds; for each, the applicable variation tried least often with that seed so far
    (ledger across all rounds), ties broken by rotation. Seeds with no applicable variation are skipped."""
    tried = Counter((r["seed_id"], r["variation"]) for r in ledger)
    eligible = [s for s in seeds if applicable(variations, s, labels)]
    jobs = []
    for j in range(n if eligible else 0):
        seed = eligible[j % len(eligible)]
        pool = applicable(variations, seed, labels)
        rot = j // len(eligible)
        v = min(enumerate(pool), key=lambda iv: (tried[(seed["id"], iv[1]["name"])], (iv[0] - rot) % len(pool)))[1]
        tried[(seed["id"], v["name"])] += 1
        jobs.append({"job_id": f"{prefix}/{seed['id']}__{rot}", "seed_id": seed["id"], "strategy": "analyzer",
                     "variation": v, **extra})
    return jobs


def outcomes(traces: list[dict], fields: list[str]) -> dict:
    c: Counter = Counter()
    for t in traces:
        ans = final_answer(t)
        for f in fields:
            if isinstance(ans, dict) and f in ans:
                c[f"{f}={ans[f]}"] += 1
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


def pilot_report(cfg: Config, verified: list[dict], seeds: list[dict]) -> dict:
    kept = [t for t in verified if t["meta"]["verify"]["kept"]]
    unique, dup = dedup(kept, near=cfg.features.near_dedup)
    reasons: Counter = Counter()
    items: Counter = Counter()
    by_var: dict[str, Counter] = {}
    for t in verified:
        v = t["meta"]["verify"]
        reasons.update(r.split(":")[0] if r.startswith("format") else r for r in v["rule_issues"])
        if v.get("ungrounded_args"):
            reasons["ungrounded_args"] += 1
        j = v.get("judge")
        if j is not None and not v["kept"]:
            reasons["judge_rejected"] += 1
        items.update((j or {}).get("checklist_failed") or [])
        b = by_var.setdefault(t["meta"].get("variation") or "-", Counter())
        b["jobs"] += 1
        b["kept"] += v["kept"]
        b["no_op"] += bool((t["meta"].get("fidelity") or {}).get("no_op"))
    cells = Counter(c for t in unique for c in decision_cells(t))
    seed_cells = {c for s in seeds for c in decision_cells(s)}
    n = len(verified)
    return {"jobs": n, "format_ok": sum(1 for t in verified if not t["meta"].get("format_issues")),
            "kept": len(kept), "unique_kept": len(unique), "usable_rate": round(len(unique) / n, 3) if n else 0.0,
            "dedup": dup, "reject_reasons": dict(reasons.most_common()), "checklist_failed": dict(items.most_common()),
            "by_variation": {k: dict(v) for k, v in sorted(by_var.items())},
            "outcomes_kept": outcomes(unique, cfg.outcome_fields), "outcomes_seeds": outcomes(seeds, cfg.outcome_fields),
            "fidelity_all": _fidelity_summary(verified), "fidelity_kept": _fidelity_summary(unique),
            "decision_cells_kept": {"distinct": len(cells), "max_repeats": max(cells.values(), default=0),
                                    "not_in_seeds": sum(1 for c in cells if c not in seed_cells)},
            "diversity_kept": diversity_report(unique),
            "attempts": dict(Counter(t["meta"].get("attempts", 1) for t in verified))}


def _append_ledger(root: Path, verified: list[dict]) -> None:
    done = {r["job_id"] for r in read_jsonl(root / "ledger.jsonl")}
    with (root / "ledger.jsonl").open("a") as f:
        for t in verified:
            if t["id"] in done:
                continue
            f.write(json.dumps({"job_id": t["id"], "version": t["meta"].get("version"), "seed_id": t["meta"].get("seed_id"),
                                "variation": t["meta"].get("variation"), "kept": t["meta"]["verify"]["kept"],
                                "reasons": t["meta"]["verify"]["rule_issues"][:5]}, ensure_ascii=False) + "\n")


def run_arm(cfg: Config, adir: Path, jobs: list[dict], gen_fn, seeds: list[dict], eval_ng: set) -> dict:
    gen = parallel_map(gen_fn, jobs, key=lambda j: j["job_id"], out_path=adir / "generated.jsonl", workers=cfg.workers,
                       done_key=lambda t: t["id"], desc=f"generate {adir.name}")
    ver = parallel_map(lambda t: verify(cfg, t, eval_ng), gen, key=lambda t: t["id"], out_path=adir / "verified.jsonl",
                       workers=cfg.workers, desc=f"verify {adir.name}")
    report = pilot_report(cfg, ver, seeds)
    (adir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


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
                             mode="analyzer", version=v)
            (vdir / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
        report = run_arm(cfg, vdir, jobs, lambda j, t=spec["template"]: generate_analyzer(cfg, j, dev_by_id[j["seed_id"]], t, obs),
                         dev, eval_ng)
        _append_ledger(root, read_jsonl(vdir / "verified.jsonl"))
        versions.append({"version": v, "usable_rate": report["usable_rate"]})
        print(f"analyzer v{v}: usable {report['unique_kept']}/{report['jobs']}, format_ok {report['format_ok']}")
        if v == cfg.analyzer_rounds:
            break
        notes = diagnose(cfg, vdir, v, spec, report, read_jsonl(vdir / "verified.jsonl"), dev_by_id)
        prev = {"version": v, "template": spec["template"], "harness": spec["harness"],
                "variations": json.dumps(spec["variations"], ensure_ascii=False, indent=1), "notes": notes}
    _all_rounds(cfg, root, len(versions), dev)
    best = max(versions, key=lambda x: (x["usable_rate"], x["version"]))
    chosen = {"version": best["version"], "by": "highest dev usable rate (ties: later version); provisional, review the reports",
              "dev_usable_rates": {f"v{x['version']}": x["usable_rate"] for x in versions}}
    bdir = root / f"v{best['version']}"
    (root / "chosen.txt").write_text((bdir / "prompt.txt").read_text())
    (root / "chosen_variations.json").write_text((bdir / "variations.json").read_text())
    (root / "chosen.json").write_text(json.dumps(chosen, indent=1))
    return chosen


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
                               lambda j: generate_analyzer(cfg, j, by_id[j["seed_id"]], template, obs), heldout, eval_ng)}
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
