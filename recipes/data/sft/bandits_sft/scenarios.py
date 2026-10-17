"""Scenario synthesis jobs: plan card -> generate -> seed rules -> judges -> dimension audit, resumable per job.

Used by the scenario pipeline (scenario_pipeline.py). Real seeds remain the only generation references.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import scenario_prompts as prompts
from .analyzer import basis_errors, parse_output, to_trace
from .io import write_jsonl
from .llm import (
    EmptyResponse,
    Truncated,
    chat_json,
    get_llm,
)
from .seed_rules import SeedRules
from .verify import (
    boilerplate,
    copied,
    explicit_answer_check,
    final_answer,
    user_text,
    verify,
)

DEFAULTS = {"novelty": "scenario", "targeting": "failure", "evidence": "complete",
            "temporal": "single_state", "composition": "single"}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def leakage(trace: dict, protected: list[dict], threshold: float = 0.5, common: frozenset = frozenset()) -> bool:
    """Copy of a protected input, judged on distinctive text only (common = template 8-grams from the seeds)."""
    a = user_text(trace)
    return any(copied(a, user_text(t), common, threshold) for t in protected)


class EvaluatorError(RuntimeError):
    """A judge or consistency call failed: the job is not a reject, it is retried on resume."""


def label_values(ans: dict, fields) -> dict:
    return {k: ans[k] for k in fields if k in ans}


def _label_fields(answers: list[dict], min_share: float, max_values: int, max_len: int) -> set[str]:
    out = set()
    for key in {k for a in answers for k in a}:
        values = [a[key] for a in answers if key in a]
        if len(values) < min_share * len(answers):
            continue
        if not all(v is None or isinstance(v, bool) or isinstance(v, str) and len(v) <= max_len for v in values):
            continue
        distinct = {json.dumps(v) for v in values}
        if len(distinct) <= max(max_values, 0.3 * len(values)) and len(distinct) < len(values):
            out.add(key)
    return out


def label_fields(seeds: list[dict], min_share: float = 0.5, max_values: int = 12, max_len: int = 60,
                 min_agent: int = 3) -> list[str]:
    """Final-answer keys that act as labels, learned from the seeds: present in at least min_share of the seeds whose
    final answer is a JSON object, with few short string/bool values that repeat (few: at most max_values, or 30% of
    the answers that have the field when that is more). Learned over all seeds and within each agent (system prompt)
    with at least min_agent JSON answers, so agents with different answer formats do not dilute each other.
    [] for free-text finals."""
    return sorted(set().union(*label_fields_by_agent(seeds, min_share, max_values, max_len, min_agent).values()))


def label_fields_by_agent(seeds: list[dict], min_share: float = 0.5, max_values: int = 12, max_len: int = 60,
                          min_agent: int = 3) -> dict[str, list[str]]:
    """Per agent (system prompt): the label fields learned from that agent's own answers when it has at least
    min_agent JSON answers; otherwise (too few to learn from) those learned over all seeds. A field one agent uses
    never binds another agent that has enough answers of its own."""
    by_agent: dict[str, list[dict]] = {}
    for s in seeds:
        a = final_answer(s)
        by_agent.setdefault(s.get("system") or "", [])
        if isinstance(a, dict):
            by_agent[s.get("system") or ""].append(a)
    pooled = _label_fields([a for g in by_agent.values() for a in g], min_share, max_values, max_len)
    return {agent: sorted(_label_fields(g, min_share, max_values, max_len) if len(g) >= min_agent else pooled)
            for agent, g in by_agent.items()}


def canonical_answer(trace: dict, canon: dict, fields=()) -> dict:
    """The label part of the final answer (fields = learned label fields), through an optional exact map."""
    ans = final_answer(trace)
    if not isinstance(ans, dict):
        return {}
    label = label_values(ans, fields)
    return canon.get(json.dumps(label, sort_keys=True), label)


def target_matches(trace: dict, target: dict, canon: dict, fields=()) -> bool:
    ans = canonical_answer(trace, canon, fields)
    return all(k in ans and ans[k] == v for k, v in target.items())


def fit(trace: dict, max_chars: int) -> dict:
    """The trace with long message contents cut to head + tail until it serializes under max_chars (a copy)."""
    size = len(json.dumps(trace, ensure_ascii=False))
    if size <= max_chars:
        return trace
    cap = max(200, max_chars // max(1, len(trace["messages"])))
    while True:
        messages = []
        for m in trace["messages"]:
            c = m.get("content")
            if isinstance(c, str) and len(c) > cap:
                c = c[:cap // 2] + f" ...[{len(c) - cap} chars cut]... " + c[-(cap // 2):]
            messages.append({**m, "content": c} if "content" in m else m)
        out = {**trace, "messages": messages}
        if len(json.dumps(out, ensure_ascii=False)) <= max_chars or cap <= 200:
            return out
        cap //= 2


def references(seed: dict, seeds: list[dict], max_chars: int | None = None) -> list[dict]:
    siblings = [s for s in seeds if s["id"] != seed["id"] and s["system"] == seed["system"] and s["tools"] == seed["tools"]]
    refs = [seed, *siblings[:2]]
    return [fit(r, max_chars) for r in refs] if max_chars else refs


# One cache entry, holding the seed list itself: a later run's list can reuse a freed list's id(), so the entry is
# reused only for the very same list object (a long-lived worker runs many jobs in one process).
_RULES: list = [None, None]


def seed_rules(seeds: list[dict]) -> SeedRules:
    if _RULES[0] is not seeds:
        _RULES[:] = [seeds, SeedRules(seeds)]
    return _RULES[1]


_COMMON: list = [None, None]


def seed_template(seeds: list[dict]) -> frozenset:
    if _COMMON[0] is not seeds:
        _COMMON[:] = [seeds, boilerplate(seeds)]
    return _COMMON[1]


def append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def basis_text(contracts: str, seed: dict) -> str:
    """What a basis may quote: the Explicit/Observed contracts (before Unknown) and the seed's own system prompt."""
    known = re.split(r"(?im)^#+\s*unknown\b", contracts, maxsplit=1)[0]
    return known + "\n" + (seed.get("system") or "")


def _flatten(d: dict, keys: tuple) -> None:
    """A list of strings where a string was asked is the same content: join it instead of rejecting the card."""
    for k in keys:
        if isinstance(d.get(k), list) and d[k] and all(isinstance(x, str) for x in d[k]):
            d[k] = "; ".join(d[k])


def card_errors(card, job: dict, basis: str) -> str:
    if not isinstance(card, dict):
        return "planner returned a non-object"
    _flatten(card, ("situation", "workflow", "decisive_evidence", "dependencies", "novelty_reason"))
    if isinstance(card.get("variation"), dict):
        _flatten(card["variation"], ("name", "change", "preserve", "dependencies", "correct_when", "teaches", "basis"))
    if card.get("unsupported") is True:
        return ""
    required = ("situation", "workflow", "decisive_evidence", "dependencies", "novelty_reason")
    if any(not isinstance(card.get(k), str) or not card[k].strip() for k in required) or not isinstance(card.get("variation"), dict):
        return "card lacks required evidence/dependency fields"
    fields = ("name", "change", "preserve", "dependencies", "correct_when", "teaches", "basis")
    if any(not isinstance(card["variation"].get(k), str) or not card["variation"][k].strip() for k in fields):
        return "variation lacks required contract/change fields"
    state = card.get("episode_state")
    if not isinstance(state, dict) or not isinstance(state.get("facts"), list) or not state["facts"]:
        return "card lacks episode_state with facts"
    errors = basis_errors(card["variation"], basis)
    if errors:  # result shapes are enforced on the generated trace by seed rules; an unquoted basis is only noted
        card["basis_warning"] = errors[0]
    if job["pair"] and not isinstance(card.get("counterfactual_flip"), str):
        return "paired scenario lacks counterfactual_flip"
    return ""


def planned(out: Path, system: str, job_id: str, limit: int = 40) -> list[str]:
    """Situations already planned for this system prompt in this run: the planner must not repeat them."""
    seen = []
    for card in sorted((out / "jobs").rglob("card.json")) if (out / "jobs").exists() else []:
        if card.parent.name == job_id.rsplit("/", 1)[-1]:
            continue
        try:
            c = json.loads(card.read_text())
            meta = json.loads((card.parent / "system.txt").read_text()) if (card.parent / "system.txt").exists() else None
        except ValueError:
            continue
        if meta == system and isinstance(c.get("situation"), str):
            seen.append(c["situation"][:200])
    return seen[-limit:]


def reject(job: dict, category: str, reason: str, traces: list | None = None) -> dict:
    return {"id": job["job_id"], "job": job, "traces": traces or [], "kept": False, "category": category, "reason": reason}


def run_job(cfg, job: dict, seeds: list[dict], contracts: str, out: Path, protected: list[dict], spec: dict, canon: dict) -> dict:
    """One scenario job. Every paid answer is saved before it is checked, so a retry or resume never pays twice.
    Rejections are rows with a category; an unavailable judge raises EvaluatorError (never a false reject)."""
    seed = next(s for s in seeds if s["id"] == job["seed_id"])
    fields = tuple(spec.get("label_fields") or ())
    path = out / "jobs" / job["job_id"]
    path.mkdir(parents=True, exist_ok=True)
    context = {"job": job, "system": seed["system"], "tools": seed["tools"], "contracts": contracts,
               "tool_result_rules": seed_rules(seeds).describe(),
               "references": references(seed, seeds, getattr(cfg, "reference_max_chars", None)),
               "already_planned": planned(out, seed["system"], job["job_id"])}
    card_path, raw_card = path / "card.json", path / "card_raw.json"
    if not card_path.exists():
        if raw_card.exists():  # paid answer saved before checking: a resume never pays twice
            card = json.loads(raw_card.read_text())
        else:
            try:
                card = chat_json(get_llm(cfg, "analyzer"), prompts.PLAN, json.dumps(context, ensure_ascii=False))
            except (Truncated, EmptyResponse) as e:  # an over-long answer rejects this job only; saved so resume skips it
                card = {"unsupported": True, "category": "bad_generation", "reason": f"planner: {type(e).__name__}"}
            raw_card.write_text(json.dumps(card, ensure_ascii=False, indent=2))
        problem = card_errors(card, job, basis_text(contracts, seed))
        if problem:  # one bad card rejects this job only, with its reason; it never stops the run
            return reject(job, "bad_card", f"bad card: {problem}")
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2))
        (path / "system.txt").write_text(json.dumps(seed["system"]))
    card = json.loads(card_path.read_text())
    if card.get("unsupported") is True:
        return reject(job, card.get("category", "unsupported"), card.get("reason", "unsupported"))
    generated_path = path / "generated.json"
    if not generated_path.exists():
        raw_gen = path / "generated_raw.json"
        if raw_gen.exists():
            raw = json.loads(raw_gen.read_text())
        else:
            try:
                raw = chat_json(get_llm(cfg, "generator"), prompts.GENERATE,
                                json.dumps({**context, "card": card}, ensure_ascii=False))
            except (Truncated, EmptyResponse) as e:  # rejects this job only, never the run
                raw = {"unsupported": True, "category": "bad_generation", "reason": f"generator: {type(e).__name__}"}
            raw_gen.write_text(json.dumps(raw, ensure_ascii=False))
        if not isinstance(raw, dict):
            return reject(job, "bad_generation", "bad generation: non-object")
        if raw.get("unsupported") is True:
            return reject(job, raw.get("category", "unsupported"), raw.get("reason", "unsupported"))
        items = raw.get("traces")
        if not isinstance(items, list) or len(items) != (2 if job["pair"] else 1):
            return reject(job, "bad_generation", "bad generation: wrong number of conversations")
        traces = []
        for i, item in enumerate(items):
            messages, issues = parse_output(json.dumps(item))
            if issues:
                return reject(job, "bad_generation", f"bad generation: {str(issues)[:300]}")
            traces.append(to_trace({"job_id": f"{job['job_id']}/{i}"}, seed, messages))
        generated_path.write_text(json.dumps(traces, ensure_ascii=False))
    traces = json.loads(generated_path.read_text())
    nudge = spec.get("stop_text") or ""
    for t in traces:
        if job.get("stop_nudge") and nudge and not any(m["role"] == "user" for m in t["messages"][1:]) \
                and len(t["messages"]) >= 3 and t["messages"][-1]["role"] == "assistant":
            # the seeds' own wrap-up turn, placed before the final answer as in the real traces
            t["messages"].insert(len(t["messages"]) - 1, {"role": "user", "content": nudge})
        t["meta"].update(seed_id=seed["id"], lane="scenario", dimensions=job["dimensions"],
                         cell=job["cell"], failure_id=job["failure_id"], pair_id=job["job_id"] if job["pair"] else None,
                         allocation=job.get("allocation"), capability=job.get("capability"))
        if leakage(t, protected, common=seed_template(seeds)):
            return reject(job, "leakage", "protected input overlap", traces)
        want = job.get("outcome") or {}
        got = final_answer(t)
        if want and (not isinstance(got, dict) or any(got.get(k) != v for k, v in want.items())):
            # control keeps a real seed's outcome so its label mix matches the seeds; checked before paying judges
            return reject(job, "control_mismatch", "control outcome mismatch", traces)
        harness = seed_rules(seeds).errors(t)  # novel tool results must be ones the recorded harness could return
        if harness:
            return reject(job, "seed_rules", f"seed rules: {harness[0]}", traces)
        # Novel scenarios use seed-independent integrity checks; I3/I4 are n/a, not weakened globally.
        verify(cfg, t, set(), None)
        issues = explicit_answer_check(t, seed)
        if isinstance(final_answer(seed), dict) and not isinstance(final_answer(t), dict):
            issues.append("final answer must be a JSON object under this reference contract")
        if issues:
            t["meta"]["verify"]["rule_issues"] += issues
            t["meta"]["verify"]["kept"] = False
        v = t["meta"]["verify"]
        if any(v.get(k, {}).get("label") == "error" for k in ("rubric", "consistency")):
            raise EvaluatorError("evaluator error; generation saved for retry")
    if not all(t["meta"]["verify"]["kept"] for t in traces):
        return reject(job, "quality_gate", "quality gate", traces)
    if job["target"] and not any(target_matches(t, job["target"], canon, fields) for t in traces):
        return reject(job, "audit", "target not reached", traces)
    if job["pair"] and canonical_answer(traces[0], canon, fields) == canonical_answer(traces[1], canon, fields):
        return reject(job, "audit", "pair did not flip label", traces)
    audit = chat_json(get_llm(cfg, "judge"), prompts.AUDIT,
                      json.dumps({"job": job, "card": card, "references": context["references"], "traces": traces}, ensure_ascii=False))
    keys = ("supported", "novelty_ok", "dimensions_ok", "pair_ok")
    if not isinstance(audit, dict) or any(not isinstance(audit.get(k), bool) for k in keys):
        raise ValueError("dimension audit returned malformed verdicts")
    kept = all(audit[k] is True for k in keys)
    return {"id": job["job_id"], "job": job, "traces": traces, "audit": audit, "kept": kept,
            "category": "kept" if kept else "audit", "reason": audit.get("reason", "dimension audit")}


def export(out: Path, rows: list[dict], canon: dict, fields=()) -> dict:
    """Atomic units: contrastive pairs survive or drop together, including dedup."""
    kept, signatures, inputs, dup, dup_jobs = [], {}, {}, 0, 0
    by_cell = Counter()
    for r in rows:
        if not r["kept"]:
            continue
        pending = []
        cell_signatures = signatures.setdefault(r["job"]["cell"], [])
        for t in r["traces"]:
            # Dedup the actual evidence/answer body; shared context alone does not collapse a pair.
            body = tokens(json.dumps([m for m in t["messages"] if m["role"] != "user"], sort_keys=True))
            inp = json.dumps([t["system"], t["tools"], t["messages"][:-1]], sort_keys=True)
            label = canonical_answer(t, canon, fields)
            if any(jaccard(body, s) >= 0.7 for s in cell_signatures) or (inp in inputs and inputs[inp] != label):
                break
            pending.append((t, body, inp, label))
        if len(pending) != len(r["traces"]):
            dup += len(r["traces"])
            dup_jobs += 1
            continue
        for t, body, inp, label in pending:
            kept.append(t)
            cell_signatures.append(body)
            inputs[inp] = label
            by_cell[t["meta"]["cell"]] += 1
    write_jsonl(out / "synthetic.jsonl", kept)
    report = {"jobs_finished": len(rows), "traces_kept": len(kept), "kept_by_cell": dict(by_cell),
              "duplicate_or_conflicting_traces": dup,
              "rejected_by_category": {**Counter(r.get("category", "other") for r in rows if not r["kept"]),
                                       **({"duplicate": dup_jobs} if dup_jobs else {})}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
