"""`mint judge-calibrate`: how often does the judge agree with a known verdict?

Judge tasks are model opinions, so before trusting one we measure the judge on answers whose
true result is known: the deterministic verifier's verdict on a type's tasks (their reference,
wrong and alternative answers), or externally graded answers from a jsonl file. The same judge
verifier that ships in judge tasks does the grading, so config and failure modes are identical.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from . import harbor
from .gates import LocalRunner
from .output import MINTError, UsageError, progress
from .results import is_judged
from .validate import _meta, seeded_sample
from .workspace import Workspace, read_json, sha

GENERIC_RUBRIC = [
    "The answer directly and fully responds to what the task asks.",
    "Every claim in the answer is supported by the task evidence.",
    "The answer does not contradict the task evidence.",
]


def judge_view(task_dir: Path, dest: Path, model: str) -> Path:
    """A scratch package that judges a deterministic task: generic rubric, no reference answer."""
    spec = read_json(task_dir / "tests" / "expected.json", {})
    t = dest / "tests"
    t.mkdir(parents=True)
    instr = task_dir / "instruction.md"
    rubric = {
        "version": harbor.rubric_version(GENERIC_RUBRIC),
        "model": model,
        "task": instr.read_text() if instr.is_file() else "",
        "criteria": GENERIC_RUBRIC,
    }
    (t / "rubric.json").write_text(json.dumps(rubric))
    (t / "expected.json").write_text(
        json.dumps({"expected": "", "check": "judge", "file": spec.get("file", "answer.txt")})
    )
    (t / "grade.py").write_text(harbor.judge_source())
    (t / "test.sh").write_text(harbor.TEST_SH)
    data = task_dir / "environment" / "data"
    for p in sorted(q for q in data.rglob("*") if q.is_file()) if data.is_dir() else []:
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue  # binary evidence is not shown to the judge
        out = t / "evidence" / p.relative_to(data)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return dest


def _truth(runner: LocalRunner, task_dir: Path, fname: str, answer: str) -> bool | None:
    out = runner.run_verifier(task_dir, {fname: answer})
    return None if out.state != "ok" else (out.reward or 0.0) >= 1.0


def generated_samples(ws: Workspace, ttype: str, n: int, runner: LocalRunner) -> list[dict]:
    """Reference + wrong + alternative answers of a type's deterministic tasks, labelled by the
    deterministic verifier itself."""
    rows = [r for r in ws.result_rows() if r["type"] == ttype]
    if not rows:
        raise MINTError(f"no tasks of type {ttype!r}", "list types with `mint status`")
    det = [r for r in rows if not is_judged(r) and r["status"] != "blocked"]
    if not det:
        raise MINTError(
            f"type {ttype!r} has no deterministic verifier to compare against",
            "calibrate on a deterministic type (--type <other>) or pass --answers",
        )
    out = []
    for tid in seeded_sample(
        [r["task_id"] for r in det], n, ws.config()["seed"], "calibrate:" + ttype
    ):
        d = ws.task_dir(tid)
        spec = read_json(d / "tests" / "expected.json", {})
        meta = _meta(ws, tid)
        exp = spec.get("expected")
        ref = [] if spec.get("check") == "forbidden" else [exp[0] if isinstance(exp, list) else exp]
        answers = list(
            dict.fromkeys(str(x) for x in ref + meta["alternatives"] + meta["wrong_answers"])
        )
        if spec.get("check") == "forbidden":
            answers += [str(x) for x in exp]
        for a in dict.fromkeys(answers):
            out.append(
                {
                    "task_id": tid,
                    "type": ttype,
                    "answer": a,
                    "truth": _truth(runner, d, spec.get("file", "answer.txt"), a),
                }
            )
    return out


def external_samples(ws: Workspace, path: Path, ttype: str | None, n: int) -> list[dict]:
    if not path.is_file():
        raise UsageError(f"--answers file {path} not found", "pass a jsonl file")
    rows = {r["task_id"]: r for r in ws.result_rows()}
    items = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            tid, ans, tr = str(rec["task_id"]), str(rec["answer"]), float(rec["true_reward"])
        except (ValueError, KeyError, TypeError):
            raise UsageError(
                f"{path.name} line {i} is not {{task_id, answer, true_reward}}", "fix the jsonl"
            ) from None
        if tid not in rows:
            raise MINTError(
                f"{path.name} line {i}: unknown task id {tid!r}", "ids are in out/results.jsonl"
            )
        if ttype and rows[tid]["type"] != ttype:
            continue
        items.append({"task_id": tid, "type": rows[tid]["type"], "answer": ans, "truth": tr >= 1.0})
    keep = seeded_sample(list(range(len(items))), n, ws.config()["seed"], "calibrate-answers")
    return [items[i] for i in keep]


def calibrate(ws: Workspace, ttype: str | None, n: int, answers: Path | None, dry: bool) -> dict:
    missing = [v for v in ("MINT_JUDGE_BASE_URL", "MINT_JUDGE_API_KEY") if not os.environ.get(v)]
    if missing:
        raise MINTError(
            f"judge-calibrate needs {' and '.join(missing)} in the environment",
            "export them (and optionally MINT_JUDGE_MODEL); the key is never stored",
        )
    cfg = ws.config()
    runner = LocalRunner(timeout=max(300.0, float(cfg["gate_timeout_sec"])))
    model = os.environ.get("MINT_JUDGE_MODEL") or cfg.get("judge_model") or ""
    if not model:
        raise MINTError(
            "judge-calibrate needs a judge model",
            "export MINT_JUDGE_MODEL or set judge_model in mint.json",
        )
    if answers:
        items = external_samples(ws, answers, ttype, n)
        source = f"external answers ({answers.name})"
    elif ttype:
        items = generated_samples(ws, ttype, n, runner)
        source = "reference/wrong/alternative answers"
    else:
        raise UsageError(
            "give --type and/or --answers", "e.g. mint judge-calibrate --type T --n 20"
        )
    skipped = sum(1 for x in items if x["truth"] is None)
    items = [x for x in items if x["truth"] is not None]
    if not items:
        raise MINTError("nothing to calibrate on", "check the type, the n, or the answers file")

    stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "agreed": 0, "fa": 0, "fr": 0, "err": 0, "tasks": set()}
    )
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mint-calib-") as tmp:
        for i, it in enumerate(items):
            s = stats[it["type"]]
            progress(f"calibrate {it['type']}/{it['task_id']} ({i + 1}/{len(items)})")
            d = ws.task_dir(it["task_id"])
            spec = read_json(d / "tests" / "expected.json", {})
            pkg = d
            if spec.get("check") != "judge":
                pkg = judge_view(d, Path(tmp) / f"v{i}", cfg.get("judge_model", ""))
            out = runner.run_verifier(pkg, {spec.get("file", "answer.txt"): it["answer"]})
            if pkg is not d:
                shutil.rmtree(pkg, ignore_errors=True)
            if out.state != "ok":
                s["err"] += 1
                errors.append(out.detail)
                continue
            said = (out.reward or 0.0) >= 1.0
            s["total"] += 1
            s["tasks"].add(it["task_id"])
            s["agreed"] += said == it["truth"]
            s["fa"] += said and not it["truth"]
            s["fr"] += (not said) and it["truth"]
    if not any(s["total"] for s in stats.values()):
        raise MINTError(
            f"every judge call failed: {errors[0] if errors else 'unknown'}",
            "fix the endpoint/config (infra errors are never counted as disagreement)",
        )
    result = {"model": model, "source": source, "skipped_unlabelled": skipped, "types": {}}
    for t, s in sorted(stats.items()):
        if not s["total"]:
            continue
        agreement = s["agreed"] / s["total"]
        why = (
            f"{s['total']} answers from {len(s['tasks'])} task(s), {source}; model {model or '(pinned)'}; "
            f"false accepts {s['fa']}, false rejects {s['fr']}, infra errors {s['err']}"
        )
        entry = {
            "kind": "judge_calibration",
            "what": f"judge agreement on {t}: {agreement:.0%} ({s['agreed']}/{s['total']})",
            "why": why,
            "type": t,
            "agreement": round(agreement, 4),
            "agreed": s["agreed"],
            "total": s["total"],
            "false_accept": s["fa"],
            "false_reject": s["fr"],
        }
        entry["id"] = sha(json.dumps(entry, sort_keys=True))[:10]
        if not dry and not any(r.get("id") == entry["id"] for r in ws.decision_rows()):
            with open(ws.decisions, "a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        result["types"][t] = entry
    return result
