#!/usr/bin/env python3
"""Prepare, score, and view a human-in-the-loop Bandits evaluation."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from bandits.analyze import analyze_corpus, load_analysis, load_task_set, save_analysis
from bandits.export import Partition, build_eval_export, save_export, write_jsonl
from bandits.ingest import load_corpus
from bandits.labels import LabelSet, Verdict, load_label_set, make_label, save_label_set
from bandits.redact import ruleset_by_name
from bandits.store import ArtifactStore, DerivedStore
from bandits.verify import (
    CheckReview,
    Interpretation,
    InterviewDecision,
    apply_decision,
    draft_verifiers,
    execute_verifier,
    load_interview,
    load_reviewed_verifier,
    load_verifier_draft,
    review_verifier,
    run_draft,
    save_interview,
    save_reviewed_verifier,
    save_validation,
    save_verifier_draft,
    start_review,
    validate_draft,
)
from bandits.verify.judge import fireworks_completion, render_transcript

MODEL_LABEL_MODEL = "accounts/fireworks/models/gpt-oss-120b"
MODEL_LABEL_VERSION = "auto-label-v3"


@dataclass
class Counts:
    labeled: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    unscored: int = 0

    @property
    def scored(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


def _rate(top: int, bottom: int) -> float | None:
    return top / bottom if bottom else None


def _truth(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("truth must be an object keyed by trace id")
    for trace_id, value in raw.items():
        if not isinstance(value, dict) or not isinstance(value.get("success"), bool):
            raise ValueError(f"truth for {trace_id!r} must contain boolean 'success'")
    return raw


def prepare(
    traces: Path, source: str, project: Path, redaction: str = "secrets-only-v1"
) -> dict[str, Any]:
    """Create the blind workspace. Deliberately accepts no truth path."""
    corpus = load_corpus(traces, source, ruleset_by_name(redaction))
    artifacts = ArtifactStore(project / ".bandits")
    derived = DerivedStore(project / ".bandits")
    corpus_env = artifacts.write(corpus, source_path=str(traces))
    analysis_env = save_analysis(analyze_corpus(corpus), derived)
    project.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "traces": str(traces),
        "source": source,
        "redaction": redaction,
        "corpus_id": corpus_env.artifact_id,
        "analysis_id": analysis_env.artifact_id,
        "trace_count": len(corpus.traces),
        "span_count": sum(len(t.spans) for t in corpus.traces),
        "issue_count": len(corpus.issues),
    }
    (project / "eval-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (project / "RUNBOOK.md").write_text(_runbook(project, analysis_env.artifact_id))
    return manifest


def _runbook(project: Path, analysis_id: str) -> str:
    return f"""# Bandits HITL evaluation

Do not open the benchmark truth until review and export are finished.

## Guided reviewer (recommended)

This shows progress, lets you choose the eval families, displays evidence, asks
for your labels and review decisions, and exports the accepted evals:

```bash
uv run python scripts/evaluate_bandits.py review --project {project} --labeler "<your-name>"
```

The commands below are the same flow written out for manual operation.

## Mine and inspect

```bash
uv run bandits mine {analysis_id} --project {project} --no-audit
uv run bandits families <task-set-id> --project {project}
```

## Complete the normal loop for chosen families

Choose families with both fit and held-out traces. Repeat this flow for each one:

```bash
uv run bandits draft-verifier <task-set-id> --family <family-id> --project {project}
uv run bandits label <verifier-draft-id> --labeler "<your-name>" --project {project}
uv run bandits draft-verifier <task-set-id> --family <family-id> --labels <label-set-id> --project {project}
uv run bandits validate-verifier <verifier-draft-id> --labels <label-set-id> --project {project}
uv run bandits interview-review <verifier-draft-id> --validation <validation-id> --round 1 --project {project}
uv run bandits review-verifier <verifier-draft-id> --validation <validation-id> --verifier <verifier-id> --interview <interview-id> --project {project}
uv run bandits export <task-set-id> --format eval --verifier <reviewed-verifier-id> --output {project}/exports/<family-id>.jsonl --project {project}
```

## Reveal truth, score, and view

```bash
uv run python scripts/evaluate_bandits.py score --project {project} --truth <sealed-labels.json> --output {project}/report.json
uv run python scripts/evaluate_bandits.py view --report {project}/report.json --output {project}/report.html
```
"""


def _reviewed(store: DerivedStore):
    chosen, unreadable = {}, []
    for env in store.list(kind="reviewed_verifier"):
        try:
            item = load_reviewed_verifier(env.artifact_id, store)
        except Exception as exc:
            unreadable.append({"artifact_id": env.artifact_id, "reason": str(exc)})
            continue
        chosen.setdefault((item.spec.task_set_id, item.spec.family_id), (env.artifact_id, item))
    return chosen, unreadable


def _run_bandits(project: Path, *arguments: str) -> None:
    executable = Path(sys.executable).with_name("bandits")
    command = [str(executable), *arguments, "--project", str(project)]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"Bandits stage failed ({completed.returncode}): {' '.join(command)}")


def _load_local_env(project: Path) -> tuple[Path, ...]:
    """Load simple KEY=VALUE entries without executing the env file as shell."""
    candidates = (Path.cwd() / ".env", project / ".env")
    loaded = []
    for path in candidates:
        if not path.is_file():
            continue
        loaded.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.removeprefix("export ").strip()
            if key.isidentifier():
                os.environ.setdefault(key, value.strip().strip("'\""))
    return tuple(loaded)


def _newest(store: DerivedStore, kind: str, *, family_id: str | None = None):
    for env in store.list(kind=kind):
        if family_id is None:
            return env
        try:
            payload = json.loads(store.read_payload(env.artifact_id))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("family_id") == family_id:
            return env
    return None


def _accepted_verifiers(interview) -> list[str]:
    superseded = {review.superseded_by for review in interview.reviews if review.superseded_by}
    accepted: dict[str, set[str]] = {}
    for review in interview.reviews:
        if review.review_id in superseded:
            continue
        checks = accepted.setdefault(review.verifier_id, set())
        if review.decision is InterviewDecision.ACCEPT:
            checks.add(review.check_id)
        else:
            checks.discard(review.check_id)
    return [
        spec.verifier_id
        for spec in interview.draft.verifiers
        if {check.check_id for check in spec.checks} <= accepted.get(spec.verifier_id, set())
    ]


def _label_cards(draft, taskset, analysis) -> list[dict[str, Any]]:
    family = taskset.family_by_id()[draft.family_id]
    run = run_draft(draft, analysis, taskset)
    disputed = [item.trace_id for item in run.disagreements]
    queue = disputed + [trace_id for trace_id in family.trace_ids if trace_id not in disputed]
    tasks = {task.trace_id: task for task in analysis.tasks}
    evidence: dict[str, list] = {}
    for fact in analysis.evidence:
        evidence.setdefault(fact.trace_id, []).append(fact)
    cards = []
    for trace_id in queue:
        facts = evidence.get(trace_id, [])
        finals = [f for f in facts if f.claim == "final_output"]
        final = (
            finals[-1].value.get("output")
            if finals and isinstance(finals[-1].value, dict)
            else None
        )
        states = []
        for fact in [f for f in facts if f.claim == "final_state_field"][-10:]:
            value = fact.value if isinstance(fact.value, dict) else {}
            states.append(
                {
                    "tool": value.get("tool", "result"),
                    "key": value.get("key", "value"),
                    "value": value.get("value"),
                }
            )
        errors = [
            {"kind": f.claim, "value": str(f.value)}
            for f in facts
            if f.claim in {"span_error", "missing_tool_result"}
        ]
        task = tasks.get(trace_id)
        cards.append(
            {
                "trace_id": trace_id,
                "instruction": task.instruction if task else None,
                "final": final,
                "states": states,
                "errors": errors,
                "disputed": trace_id in disputed,
            }
        )
    return cards


def _model_label_prompt(trace) -> str:
    return f"""You evaluate whether a tool-using agent completed a user's request.

Read the request first and identify every requested outcome and constraint. Then compare them
with the recorded tool actions, external results, and final message. External tool results are
stronger evidence than the agent's own claims. A successful tool call is not sufficient when it
performed the wrong action, completed only part of the request, violated a constraint, or made an
unrequested irreversible change. Use unclear when the trace lacks enough evidence to decide.

{render_transcript(trace)}

Return ONLY one JSON object:
{{"verdict":"success|failure|unclear","confidence":0.0,"requested_outcome":"...",
"observed_outcome":"...","supporting_evidence":["..."],"rationale":"..."}}"""


def _parse_model_label(reply: str) -> tuple[Verdict, str]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        return Verdict.UNCLEAR, f"{MODEL_LABEL_VERSION}: unparseable response: {reply[:500]}"
    try:
        parsed = json.loads(reply[start : end + 1])
        verdict = Verdict(str(parsed.get("verdict", "unclear")).lower())
    except (json.JSONDecodeError, ValueError, AttributeError):
        return Verdict.UNCLEAR, f"{MODEL_LABEL_VERSION}: unparseable response: {reply[:500]}"
    return verdict, f"{MODEL_LABEL_VERSION}: " + json.dumps(parsed, sort_keys=True)


def model_label_family(project: Path, taskset_id: str, family_id: str):
    """Return cached model labels, or judge one family and persist them."""
    store = DerivedStore(project / ".bandits")
    taskset = load_task_set(taskset_id, store)
    cached = _newest(store, "label_set", family_id=family_id)
    if cached is not None:
        label_set = load_label_set(cached.artifact_id, store)
        if label_set.labels and all(
            label.source == "model"
            and label.labeler == MODEL_LABEL_MODEL
            and label.rationale.startswith(f"{MODEL_LABEL_VERSION}:")
            for label in label_set.labels
        ):
            return label_set, cached.artifact_id, True

    family = taskset.family_by_id()[family_id]
    corpus = ArtifactStore(project / ".bandits").read(taskset.corpus_id)
    traces = {trace.trace_id: trace for trace in corpus.traces}

    def label_trace(trace_id: str) -> tuple[Verdict, str]:
        prompt = _model_label_prompt(traces[trace_id])
        reply = fireworks_completion(MODEL_LABEL_MODEL, prompt, 0.0)
        return _parse_model_label(reply)

    decisions = {trace_id: label_trace(trace_id) for trace_id in family.trace_ids}
    labels = tuple(
        make_label(
            trace_id=trace_id,
            family_id=family_id,
            verdict=decision[0],
            labeler=MODEL_LABEL_MODEL,
            source="model",
            rationale=decision[1],
            prompted_by=MODEL_LABEL_VERSION,
        )
        for trace_id, decision in decisions.items()
    )
    label_set = LabelSet(task_set_id=taskset_id, family_id=family_id, labels=labels)
    labels_env = save_label_set(label_set, store)
    return label_set, labels_env.artifact_id, False


def web_label(project: Path, draft_id: str, labeler: str) -> str:
    """Collect one family's labels in a local browser and save a LabelSet."""
    store = DerivedStore(project / ".bandits")
    draft = load_verifier_draft(draft_id, store)
    taskset = load_task_set(draft.task_set_id, store)
    analysis = load_analysis(draft.analysis_id, store)
    cards = _label_cards(draft, taskset, analysis)
    result: dict[str, str] = {}

    page = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width"><title>Bandits Review</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#eef2ff;font:17px system-ui}main{max-width:900px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;color:#aeb8d4}.bar{height:8px;background:#202b49;border-radius:8px;margin:12px 0 30px}.bar i{display:block;height:100%;background:#67e8b5;border-radius:8px}.card{background:#151d34;border:1px solid #2c3859;border-radius:18px;padding:28px}h2{color:#9fb2ff;font-size:14px;text-transform:uppercase;margin:24px 0 8px}.request{font-size:24px;line-height:1.35}.facts{display:grid;gap:7px}.fact{background:#0e1529;padding:10px 13px;border-radius:8px}.final{white-space:pre-wrap;line-height:1.5}.buttons{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:24px}button{border:0;border-radius:12px;padding:16px;font-weight:700;font-size:17px;cursor:pointer}.success{background:#55d69e}.failure{background:#ff7c83}.unclear{background:#bac4dd}textarea{width:100%;margin-top:15px;border-radius:10px;padding:12px;background:#0e1529;color:white;border:1px solid #354363}kbd{opacity:.65}.error{color:#ff9aa0}#done{text-align:center;padding:80px 0}</style></head><body><main><div id=app></div></main><script>
const cards=__CARDS__;let i=0;const decisions={};const app=document.querySelector('#app');
const esc=s=>String(s??'Not recorded').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){if(i>=cards.length){finish();return}const c=cards[i], facts=c.states.map(x=>`<div class=fact><b>${esc(x.tool)}</b> · ${esc(x.key)} = ${esc(JSON.stringify(x.value))}</div>`).join('')||'<div class=fact>No structured state recorded</div>';app.innerHTML=`<div class=top><span>${c.disputed?'⚡ Disputed case':'Review case'}</span><span>${i+1} / ${cards.length}</span></div><div class=bar><i style="width:${100*i/cards.length}%"></i></div><section class=card><h2>Request</h2><div class=request>${esc(c.instruction)}</div><h2>What changed</h2><div class=facts>${facts}${c.errors.map(e=>`<div class="fact error">${esc(e.kind)}: ${esc(e.value)}</div>`).join('')}</div><h2>Agent's final response</h2><div class=final>${esc(c.final)}</div><textarea id=why placeholder="Optional note: why?"></textarea><div class=buttons><button class=success onclick="pick('success')">Success <kbd>S</kbd></button><button class=failure onclick="pick('failure')">Failure <kbd>F</kbd></button><button class=unclear onclick="pick('unclear')">Unclear <kbd>U</kbd></button></div></section>`}
function pick(verdict){decisions[cards[i].trace_id]={verdict,rationale:document.querySelector('#why').value};i++;render()}
async function finish(){app.innerHTML='<div id=done><h1>Saving review…</h1></div>';const r=await fetch('/finish',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(decisions)});const d=await r.json();app.innerHTML=`<div id=done><h1>Review saved</h1><p>${esc(d.label_set_id)}</p><p>You can return to the terminal.</p></div>`}
addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='s')pick('success');if(e.key==='f')pick('failure');if(e.key==='u')pick('unclear')});render();
</script></body></html>""".replace("__CARDS__", json.dumps(cards).replace("</", "<\\/"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/finish":
                self.send_error(404)
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            decisions = json.loads(raw)
            labels = tuple(
                make_label(
                    trace_id=card["trace_id"],
                    family_id=draft.family_id,
                    verdict=Verdict(decisions[card["trace_id"]]["verdict"]),
                    labeler=labeler,
                    rationale=decisions[card["trace_id"]].get("rationale", ""),
                    prompted_by=draft_id if card["disputed"] else None,
                )
                for card in cards
            )
            label_set = LabelSet(
                task_set_id=draft.task_set_id, family_id=draft.family_id, labels=labels
            )
            env = save_label_set(label_set, store)
            result["id"] = env.artifact_id
            response = json.dumps({"label_set_id": env.artifact_id}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Browser reviewer: {url}")
    webbrowser.open(url)
    server.serve_forever()
    server.server_close()
    return result["id"]


def review_app(project: Path, labeler: str) -> None:
    """Run family selection, labeling, check review, and export in one browser app."""
    _load_local_env(project)
    store = DerivedStore(project / ".bandits")
    taskset_env = _newest(store, "taskset")
    if taskset_env is None:
        analysis_env = _newest(store, "analysis")
        if analysis_env is None:
            raise ValueError("run prepare first; this project contains no analysis")
        # Mining is the only long setup stage. It happens before interaction so
        # the browser never offers families that do not exist yet.
        print("Mining families once, then opening the reviewer…", flush=True)
        _run_bandits(project, "mine", analysis_env.artifact_id, "--no-audit")
        taskset_env = _newest(store, "taskset")
    if taskset_env is None:
        raise RuntimeError("mining completed without creating a task set")
    taskset = load_task_set(taskset_env.artifact_id, store)
    analysis = load_analysis(taskset.analysis_id, store)
    eligible = sorted(
        (f for f in taskset.families if f.fit_trace_ids and f.held_out_trace_ids),
        key=lambda f: (-f.workload_mass, f.family_id),
    )
    state: dict[str, Any] = {
        "phase": "select",
        "families": [
            {
                "id": f.family_id,
                "descriptor": f.descriptor,
                "fit": len(f.fit_trace_ids),
                "held": len(f.held_out_trace_ids),
                "traces": len(f.trace_ids),
            }
            for f in eligible
        ],
        "selected": [],
        "index": 0,
        "cards": [],
        "checks": [],
        "message": "Choose the proposed families to evaluate.",
        "exports": [],
    }
    runtime: dict[str, Any] = {}

    def open_family() -> None:
        family_id = state["selected"][state["index"]]
        family = taskset.family_by_id()[family_id]
        state.update(
            phase="working",
            current={"id": family_id, "descriptor": family.descriptor},
            message="The LLM is labeling this family and Bandits is testing its best verifier…",
        )
        label_set, labels_id, cached = model_label_family(
            project, taskset_env.artifact_id, family_id
        )
        labels = label_set.labels
        tasks = {task.trace_id: task for task in analysis.tasks}
        label_by_trace = {label.trace_id: label for label in labels}

        def label_reason(trace_id: str) -> str:
            label = label_by_trace.get(trace_id)
            if label is None:
                return "No model judgment recorded."
            raw = label.rationale.removeprefix(f"{MODEL_LABEL_VERSION}: ")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            return str(parsed.get("rationale") or parsed.get("observed_outcome") or raw)

        representative_ids = [family.medoid_trace_id]
        representative_ids.extend(
            trace_id for trace_id in family.trace_ids if trace_id != family.medoid_trace_id
        )
        representatives = [
            {
                "trace_id": trace_id,
                "request": tasks[trace_id].instruction if trace_id in tasks else None,
                "model_verdict": label_by_trace[trace_id].verdict.value
                if trace_id in label_by_trace
                else "unknown",
                "model_reason": label_reason(trace_id),
            }
            for trace_id in representative_ids[:3]
        ]
        # One proposal keeps human work proportional to families, not trajectories.
        draft = draft_verifiers(
            taskset,
            taskset_env.artifact_id,
            analysis,
            family_id,
            limit=1,
            labels=label_set,
        )
        draft_env = save_verifier_draft(draft, store)
        validation = validate_draft(
            draft,
            draft_env.artifact_id,
            taskset,
            analysis,
            label_set,
            labels_id,
        )
        validation_env = save_validation(validation, store)
        runtime.update(
            family=family,
            draft=draft,
            draft_id=draft_env.artifact_id,
            validation=validation,
            validation_id=validation_env.artifact_id,
        )
        checks = []
        for spec in draft.verifiers:
            held = validation.held_out(spec.verifier_id)
            assessment = next(
                (
                    a
                    for a in validation.gameability_assessments
                    if a.verifier_id == spec.verifier_id
                ),
                None,
            )
            attacks = [
                {
                    "hypothesis": attack.hypothesis,
                    "passed": attack.passed,
                    "forged_facts": attack.forged_facts,
                }
                for attack in validation.gameability
                if attack.verifier_id == spec.verifier_id
            ]
            counterexamples = []
            if held:
                for item in held.counterexamples[:3]:
                    task = tasks.get(item.trace_id)
                    counterexamples.append(
                        {
                            "trace_id": item.trace_id,
                            "kind": item.kind,
                            "request": task.instruction if task else None,
                            "model_verdict": item.human_verdict,
                            "model_reason": label_reason(item.trace_id),
                        }
                    )
            for check in spec.checks:
                checks.append(
                    {
                        "verifier_id": spec.verifier_id,
                        "check_id": check.check_id,
                        "description": check.description,
                        "expected": check.expected,
                        "evidence": check.evidence_kind.value,
                        "agreement": held.agreement if held else None,
                        "coverage": held.coverage if held else None,
                        "scored": held.scored if held else 0,
                        "labeled": held.labeled if held else 0,
                        "unscored": held.unscored if held else 0,
                        "false_positives": held.false_positives if held else None,
                        "false_negatives": held.false_negatives if held else None,
                        "gameable": assessment.attack_succeeded if assessment else None,
                        "attack_coverage": assessment.coverage if assessment else "none",
                        "attacks": attacks,
                        "counterexamples": counterexamples,
                        "blind_spots": list(spec.blind_spots),
                    }
                )
        state.update(
            phase="checks",
            cards=[],
            checks=checks,
            model_labels={
                "success": sum(label.verdict is Verdict.SUCCESS for label in labels),
                "failure": sum(label.verdict is Verdict.FAILURE for label in labels),
                "unclear": sum(label.verdict is Verdict.UNCLEAR for label in labels),
                "model": labels[0].labeler if labels else "unknown",
                "cached": cached,
            },
            representatives=representatives,
            message="The model labeled the traces. You only review the proposed verifier.",
        )

    def receive_reviews(decisions: dict[str, Any]) -> None:
        draft, validation = runtime["draft"], runtime["validation"]
        interview = start_review(draft, runtime["draft_id"], validation_id=runtime["validation_id"])
        for spec in draft.verifiers:
            for check in spec.checks:
                answer = decisions.get(
                    check.check_id, {"decision": "reject", "rationale": "No decision submitted."}
                )
                decision = InterviewDecision(answer["decision"])
                interpretation = Interpretation(
                    source="human", decision=decision, rationale=answer.get("rationale", "")
                )
                review = CheckReview(
                    review_id=f"web-{len(interview.reviews) + 1:03d}-{check.check_id}",
                    verifier_id=spec.verifier_id,
                    check_id=check.check_id,
                    reply=answer.get("rationale", ""),
                    decision=decision,
                    authoritative=True if decision is InterviewDecision.ACCEPT else None,
                    authoritative_why=answer.get("rationale", ""),
                    interpretation=interpretation,
                    model="human-web-review",
                )
                interview = apply_decision(interview, review)
        interview_env = save_interview(interview, store)
        accepted = _accepted_verifiers(interview)
        exported = None
        if accepted:
            reviewed = review_verifier(
                draft,
                runtime["draft_id"],
                validation,
                runtime["validation_id"],
                accepted[0],
                interview,
                interview_env.artifact_id,
            )
            reviewed_env = save_reviewed_verifier(reviewed, store)
            corpus = ArtifactStore(project / ".bandits").read(taskset.corpus_id)
            bundle = build_eval_export(
                corpus,
                taskset,
                taskset_env.artifact_id,
                analysis,
                reviewed,
                reviewed_env.artifact_id,
                partition=Partition.HELD_OUT,
            )
            save_export(bundle, store)
            output = project / "exports" / f"{runtime['family'].family_id}.jsonl"
            write_jsonl(bundle, output)
            exported = str(output)
            state["exports"].append(exported)
        state["index"] += 1
        if state["index"] < len(state["selected"]):
            open_family()
        else:
            state.update(
                phase="done",
                checks=[],
                message="HITL review complete. You can now reveal the sealed truth and score it.",
            )
        if exported:
            state["last_export"] = exported

    page = _review_app_html()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def reply(self, value: Any, status: int = 200):
            body = (value if isinstance(value, str) else json.dumps(value)).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8" if isinstance(value, str) else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(state if self.path == "/api/state" else page)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if self.path == "/api/select":
                    state["selected"] = body["families"]
                    if not state["selected"]:
                        raise ValueError("select at least one family")
                    open_family()
                elif self.path == "/api/reviews":
                    receive_reviews(body)
                elif self.path == "/api/score":
                    report = score(project, Path(body["truth"]))
                    report_path = project / "report.json"
                    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                    state.update(
                        phase="report",
                        report=report,
                        message="Evaluation scored against sealed truth.",
                    )
                else:
                    raise ValueError("unknown action")
                self.reply({"ok": True})
            except Exception as exc:
                self.reply({"ok": False, "error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Bandits HITL UI: {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def prelabel(project: Path) -> None:
    """Model-label every eligible family so browser review starts instantly."""
    _load_local_env(project)
    store = DerivedStore(project / ".bandits")
    taskset_env = _newest(store, "taskset")
    if taskset_env is None:
        raise ValueError("no task set found; run start once to mine families")
    taskset = load_task_set(taskset_env.artifact_id, store)
    families = sorted(
        (f for f in taskset.families if f.fit_trace_ids and f.held_out_trace_ids),
        key=lambda f: (-f.workload_mass, f.family_id),
    )
    for index, family in enumerate(families, 1):
        print(
            f"[{index}/{len(families)}] labeling {family.family_id} ({len(family.trace_ids)} traces)…",
            flush=True,
        )
        labels, _, cached = model_label_family(project, taskset_env.artifact_id, family.family_id)
        counts = {verdict: sum(x.verdict is verdict for x in labels.labels) for verdict in Verdict}
        suffix = " cached" if cached else ""
        print(
            f"  {counts[Verdict.SUCCESS]} success, {counts[Verdict.FAILURE]} failure, {counts[Verdict.UNCLEAR]} unclear{suffix}"
        )


def _review_app_html() -> str:
    return r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bandits HITL</title><style>
*{box-sizing:border-box}body{margin:0;background:#0a0f1f;color:#eef2ff;font:16px system-ui}main{max-width:980px;margin:auto;padding:30px}.muted{color:#9aa7c5}.panel{background:#151d34;border:1px solid #2c3859;border-radius:18px;padding:25px;margin:18px 0}.row{display:flex;gap:12px;align-items:flex-start;padding:13px;border-bottom:1px solid #293451}.row:last-child{border:0}input[type=checkbox]{width:20px;height:20px}.pill{background:#263252;padding:4px 9px;border-radius:20px;white-space:nowrap}button{border:0;border-radius:10px;padding:13px 18px;font-weight:700;cursor:pointer}.primary,.success{background:#58dda7}.failure{background:#ff858d}.unclear{background:#c3cbe0}.actions{display:flex;gap:10px;margin-top:20px}.request{font-size:23px;line-height:1.4}.fact{background:#0d1428;padding:9px 12px;border-radius:8px;margin:6px 0}.final{white-space:pre-wrap;line-height:1.5;max-height:260px;overflow:auto}textarea,input[type=text]{width:100%;background:#0d1428;color:white;border:1px solid #39496e;border-radius:9px;padding:11px;margin-top:10px}.bar{height:7px;background:#263252;border-radius:9px}.bar i{display:block;height:100%;background:#58dda7}.metric{display:inline-block;background:#0d1428;padding:7px 10px;border-radius:8px;margin:3px}.error{color:#ff9299}h3{color:#9fb2ff;text-transform:uppercase;font-size:13px;margin-top:23px}</style></head><body><main><h1>Bandits HITL Review</h1><p class=muted id=status>Loading…</p><div id=app></div></main><script>
const app=document.querySelector('#app'),statusEl=document.querySelector('#status');let state,cardIndex=0,labelDecisions={};const esc=s=>String(s??'Not recorded').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,data){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}),d=await r.json();if(!d.ok)throw Error(d.error);await load()}
async function load(){state=await(await fetch('/api/state')).json();statusEl.textContent=state.message;render()}
function render(){if(state.phase==='select')select();else if(state.phase==='working')app.innerHTML='<div class=panel><h2>LLM labeling in progress…</h2><p class=muted>This can take a moment. No human trajectory labeling is required.</p></div>';else if(state.phase==='checks')checks();else if(state.phase==='done')done();else if(state.phase==='report')report()}
function select(){app.innerHTML=`<div class=panel><h2>Select eval families</h2><p class=muted>Nothing is preselected. The LLM will label their trajectories; you will review one proposed verifier per family.</p>${state.families.map((f,i)=>`<label class=row><input type=checkbox value="${f.id}"><span style="flex:1"><b>${esc(f.descriptor)}</b></span><span class=pill>${f.fit} fit · ${f.held} held</span></label>`).join('')}<div class=actions><button id=all>Select all</button><button class=primary id=start>Generate verifiers</button></div></div>`;document.querySelector('#all').onclick=()=>document.querySelectorAll('input[type=checkbox]').forEach(x=>x.checked=true);document.querySelector('#start').onclick=()=>{const b=document.querySelector('#start');b.disabled=true;b.textContent='LLM labeling…';api('/api/select',{families:[...document.querySelectorAll('input:checked')].map(x=>x.value)}).catch(showError)}}
function label(){const c=state.cards[cardIndex],pct=100*cardIndex/state.cards.length;app.innerHTML=`<div class=bar><i style="width:${pct}%"></i></div><p class=muted>Family ${state.index+1}/${state.selected.length} · trajectory ${cardIndex+1}/${state.cards.length}</p><div class=panel><h3>Request</h3><div class=request>${esc(c.instruction)}</div><h3>What changed</h3>${c.states.map(x=>`<div class=fact><b>${esc(x.tool)}</b> · ${esc(x.key)} = ${esc(JSON.stringify(x.value))}</div>`).join('')||'<div class=fact>No structured result</div>'}<h3>Agent final response</h3><div class=final>${esc(c.final)}</div><textarea id=why placeholder="Optional rationale"></textarea><div class=actions><button class=success onclick="pick('success')">Success (S)</button><button class=failure onclick="pick('failure')">Failure (F)</button><button class=unclear onclick="pick('unclear')">Unclear (U)</button></div></div>`}
function pick(verdict){labelDecisions[state.cards[cardIndex].trace_id]={verdict,rationale:document.querySelector('#why').value};cardIndex++;if(cardIndex===state.cards.length){const d=labelDecisions;labelDecisions={};cardIndex=0;api('/api/labels',d).catch(showError)}else label()}
function checks(){const m=state.model_labels,reps=(state.representatives||[]).map(x=>`<div class=fact><b>${esc(x.model_verdict)}</b> · ${esc(x.request)}<div class=muted>${esc(x.model_reason)}</div></div>`).join('');app.innerHTML=`<div class=panel><h2>Review proposed verifier</h2><p class=muted>Your question is narrow: <b>would this recorded field necessarily prove that these requests succeeded?</b> The model labels are hypotheses, not ground truth.</p><h3>What this family asks for</h3>${reps}<p class=muted>${m.model} labeled all traces: ${m.success} success, ${m.failure} failure, ${m.unclear} unclear.</p>${state.checks.map(c=>`<div class=row style="display:block"><h2>${esc(c.description)}</h2><div><span class=metric>model agreement ${c.agreement==null?'n/a':Math.round(c.agreement*100)+'%'}</span><span class=metric>scorable ${c.scored}/${c.labeled}</span><span class=metric>FP ${c.false_positives??'n/a'}</span><span class=metric>FN ${c.false_negatives??'n/a'}</span></div><h3>Where verifier and model disagree</h3>${c.counterexamples.length?c.counterexamples.map(x=>`<div class=fact><b>${esc(x.kind)}</b> · model=${esc(x.model_verdict)}<br>${esc(x.request)}<div class=muted>${esc(x.model_reason)}</div></div>`).join(''):'<p class=muted>No disagreement among scorable held-out examples. This does not prove correctness.</p>'}<h3>Gaming test</h3>${c.attacks.length?c.attacks.map(a=>`<div class=fact>${a.passed?'⚠ Attack succeeded':'Attack resisted'}: ${esc(a.hypothesis)} (${a.forged_facts} forged fact${a.forged_facts===1?'':'s'})</div>`).join(''):'<p class=muted>No applicable attack template; safety was not established.</p>'}${c.blind_spots.length?`<h3>Known blind spots</h3><p>${c.blind_spots.map(esc).join('<br>')}</p>`:''}<h3>Your decision</h3><p>Does this check necessarily demonstrate successful completion—not merely that one tool emitted this value?</p><textarea data-note="${c.check_id}" placeholder="State why this field is or is not sufficient evidence."></textarea><div class=actions><label><input type=radio name="${c.check_id}" value=accept> Accept as sufficient</label><label><input type=radio name="${c.check_id}" value=reject checked> Reject as insufficient</label></div></div>`).join('')||'<p>No executable verifier could be proposed for this family.</p>'}<button class=primary id=submit>Save decision and continue</button></div>`;document.querySelector('#submit').onclick=()=>{const d={};state.checks.forEach(c=>d[c.check_id]={decision:document.querySelector(`input[name="${c.check_id}"]:checked`).value,rationale:document.querySelector(`[data-note="${c.check_id}"]`).value});api('/api/reviews',d).catch(showError)}}
function done(){app.innerHTML=`<div class=panel><h2>Review complete</h2><p>${state.exports.length} eval export(s) created.</p><p class=muted>Now—and only now—provide the sealed τ2 truth file.</p><input id=truth type=text value="work/tau2-run/tau2.labels.json"><div class=actions><button class=primary id=score>Reveal truth and score</button></div></div>`;document.querySelector('#score').onclick=()=>api('/api/score',{truth:document.querySelector('#truth').value}).catch(showError)}
function report(){const r=state.report,s=r.scores,w=r.workflow,h=r.held_out;app.innerHTML=`<div class=panel><h2>Final result</h2><div class=request>Trust score: ${s.trust_score==null?'n/a':Math.round(s.trust_score*100)+'%'}</div><p>Precision ${s.success_precision==null?'n/a':Math.round(s.success_precision*100)+'%'} · Coverage ${s.coverage==null?'n/a':Math.round(s.coverage*100)+'%'} · Failure recall ${s.failure_recall==null?'n/a':Math.round(s.failure_recall*100)+'%'}</p><p>${w.reviewed_families} reviewed families · ${w.exported_rows} exported rows · ${h.scored}/${h.labeled} held-out scored · ${s.leaked_task_groups} leaked task groups</p><p class=muted>Saved to the project report.json.</p></div>`}
function showError(e){statusEl.innerHTML=`<span class=error>${esc(e.message)}</span>`}addEventListener('keydown',e=>{if(state?.phase!=='label'||e.target.tagName==='TEXTAREA')return;if(e.key.toLowerCase()==='s')pick('success');if(e.key.toLowerCase()==='f')pick('failure');if(e.key.toLowerCase()==='u')pick('unclear')});load();
</script></body></html>"""


def guided_review(project: Path, labeler: str, limit: int) -> None:
    """Drive the normal HITL commands while preserving their native questions."""
    env_files = _load_local_env(project)
    if env_files:
        print("Loaded environment from " + ", ".join(str(path) for path in env_files))
    store = DerivedStore(project / ".bandits")
    taskset_env = _newest(store, "taskset")
    if taskset_env is None:
        analysis_env = _newest(store, "analysis")
        if analysis_env is None:
            raise ValueError("run prepare first; this project contains no analysis")
        print("\n[1/7] Mining task families…", flush=True)
        _run_bandits(project, "mine", analysis_env.artifact_id, "--no-audit")
        taskset_env = _newest(store, "taskset")
    if taskset_env is None:
        raise RuntimeError("mining completed without creating a task set")
    taskset = load_task_set(taskset_env.artifact_id, store)
    eligible = [f for f in taskset.families if f.fit_trace_ids and f.held_out_trace_ids]
    eligible.sort(key=lambda f: (-f.workload_mass, f.family_id))
    if not eligible:
        raise ValueError("no family has both fit and held-out traces")

    print("\nChoose the families that will become evals:")
    for index, family in enumerate(eligible[:20], 1):
        print(
            f"  {index:>2}. {family.descriptor[:74]}  (fit {len(family.fit_trace_ids)}, held {len(family.held_out_trace_ids)})"
        )
    default = ",".join(str(i) for i in range(1, min(limit, len(eligible)) + 1))
    answer = input(f"Family numbers [{default}]: ").strip() or default
    indexes = []
    for part in answer.split(","):
        index = int(part.strip())
        if index < 1 or index > min(20, len(eligible)):
            raise ValueError(f"family selection {index} is out of range")
        if index not in indexes:
            indexes.append(index)

    for position, index in enumerate(indexes, 1):
        family = eligible[index - 1]
        print(f"\n=== Eval {position}/{len(indexes)}: {family.descriptor} ===", flush=True)
        print("[2/7] Drafting blind verifier candidates…", flush=True)
        _run_bandits(
            project, "draft-verifier", taskset_env.artifact_id, "--family", family.family_id
        )
        blind = _newest(store, "verifier_draft", family_id=family.family_id)
        if blind is None:
            print("No verifier could be drafted; skipping this family.")
            continue
        print("[3/7] Opening the browser label reviewer…", flush=True)
        labels_id = web_label(project, blind.artifact_id, labeler)
        print("[4/7] Redrafting with your labels and validating held-out behavior…", flush=True)
        _run_bandits(
            project,
            "draft-verifier",
            taskset_env.artifact_id,
            "--family",
            family.family_id,
            "--labels",
            labels_id,
        )
        calibrated = _newest(store, "verifier_draft", family_id=family.family_id)
        if calibrated is None:
            continue
        _run_bandits(project, "validate-verifier", calibrated.artifact_id, "--labels", labels_id)
        validation = _newest(store, "validation", family_id=family.family_id)
        if validation is None:
            raise RuntimeError("validation completed without creating an artifact")
        print(
            "[5/7] Human review—the reviewer shows measurements and asks you about every check…",
            flush=True,
        )
        _run_bandits(
            project,
            "interview-review",
            calibrated.artifact_id,
            "--validation",
            validation.artifact_id,
            "--round",
            "1",
        )
        interview_env = _newest(store, "verifier_interview")
        if interview_env is None:
            raise RuntimeError("review completed without creating an interview")
        interview = load_interview(interview_env.artifact_id, store)
        accepted = _accepted_verifiers(interview)
        if not accepted:
            print("You accepted no complete verifier; this family is correctly left unresolved.")
            continue
        draft = load_verifier_draft(calibrated.artifact_id, store)
        print("[6/7] Select the accepted verifier to promote:")
        for number, verifier_id in enumerate(accepted, 1):
            spec = next(v for v in draft.verifiers if v.verifier_id == verifier_id)
            checks = " AND ".join(check.description for check in spec.checks)
            print(f"  {number}. {verifier_id}: {checks}")
        choice = input("Verifier number [1]: ").strip() or "1"
        verifier_id = accepted[int(choice) - 1]
        _run_bandits(
            project,
            "review-verifier",
            calibrated.artifact_id,
            "--validation",
            validation.artifact_id,
            "--verifier",
            verifier_id,
            "--interview",
            interview_env.artifact_id,
        )
        reviewed_env = store.list(kind="reviewed_verifier")[0]
        output = project / "exports" / f"{family.family_id}.jsonl"
        print("[7/7] Exporting held-out eval cases…", flush=True)
        _run_bandits(
            project,
            "export",
            taskset_env.artifact_id,
            "--format",
            "eval",
            "--verifier",
            reviewed_env.artifact_id,
            "--output",
            str(output),
        )
        print(f"Completed: {output}")

    print("\nReview finished. Keep truth sealed until you run the score command.")


def score(project: Path, truth_path: Path) -> dict[str, Any]:
    """Open sealed truth and score only completed, human-reviewed verifiers."""
    truth = _truth(truth_path)
    store = DerivedStore(project / ".bandits")
    reviewed, unreadable = _reviewed(store)
    totals, rows, leaked = Counts(), [], set()

    for (taskset_id, family_id), (reviewed_id, item) in reviewed.items():
        taskset = load_task_set(taskset_id, store)
        analysis = load_analysis(taskset.analysis_id, store)
        family = taskset.family_by_id()[family_id]
        evidence: dict[str, list] = {}
        for fact in analysis.evidence:
            evidence.setdefault(fact.trace_id, []).append(fact)
        fit_groups = {
            str(truth[t]["task_id"])
            for t in family.fit_trace_ids
            if t in truth and truth[t].get("task_id") is not None
        }
        held_groups = {
            str(truth[t]["task_id"])
            for t in family.held_out_trace_ids
            if t in truth and truth[t].get("task_id") is not None
        }
        leaked.update(fit_groups & held_groups)

        local = Counts()
        for trace_id in family.held_out_trace_ids:
            if trace_id not in truth:
                continue
            local.labeled += 1
            result = execute_verifier(item.spec, tuple(evidence.get(trace_id, ())))
            if result.score is None:
                local.unscored += 1
                continue
            predicted, actual = result.score >= item.success_threshold, truth[trace_id]["success"]
            if predicted and actual:
                local.tp += 1
            elif predicted:
                local.fp += 1
            elif actual:
                local.fn += 1
            else:
                local.tn += 1
        for field in ("labeled", "tp", "tn", "fp", "fn", "unscored"):
            setattr(totals, field, getattr(totals, field) + getattr(local, field))
        rows.append(
            {
                "family_id": family_id,
                "descriptor": family.descriptor,
                "reviewed_verifier_id": reviewed_id,
                "labeled": local.labeled,
                "scored": local.scored,
                "tp": local.tp,
                "tn": local.tn,
                "fp": local.fp,
                "fn": local.fn,
                "unscored": local.unscored,
            }
        )

    human_total = human_correct = model_total = model_correct = 0
    seen_labels: set[tuple[str, str]] = set()
    for env in store.list(kind="label_set"):
        for label in load_label_set(env.artifact_id, store).labels:
            identity = (label.source, label.trace_id)
            if identity in seen_labels:
                continue
            seen_labels.add(identity)
            if label.trace_id not in truth or label.verdict.value == "unclear":
                continue
            correct = (label.verdict.value == "success") == truth[label.trace_id]["success"]
            if label.source == "model":
                model_total += 1
                model_correct += correct
            else:
                human_total += 1
                human_correct += correct

    precision = _rate(totals.tp, totals.tp + totals.fp)
    coverage = _rate(totals.scored, totals.labeled)
    verifier_decisions = sum(
        len(load_interview(env.artifact_id, store).reviews)
        for env in store.list(kind="verifier_interview")
    )
    return {
        "schema_version": 1,
        "dataset": {"truth": str(truth_path), "labeled_trajectories": len(truth)},
        "workflow": {
            "reviewed_families": len(rows),
            "human_verifier_decisions": verifier_decisions,
            "human_labels": human_total,
            "human_label_accuracy": _rate(human_correct, human_total),
            "model_labels": model_total,
            "model_label_accuracy": _rate(model_correct, model_total),
            "exported_rows": sum(e.summary.get("rows", 0) for e in store.list(kind="eval_export")),
            "unreadable_legacy_reviews": unreadable,
        },
        "held_out": {
            "labeled": totals.labeled,
            "scored": totals.scored,
            "unscored": totals.unscored,
            "true_positive": totals.tp,
            "true_negative": totals.tn,
            "false_positive": totals.fp,
            "false_negative": totals.fn,
        },
        "scores": {
            "coverage": coverage,
            "success_precision": precision,
            "failure_recall": _rate(totals.tn, totals.tn + totals.fp),
            "accuracy": _rate(totals.tp + totals.tn, totals.scored),
            "trust_score": precision * coverage
            if precision is not None and coverage is not None
            else None,
            "leaked_task_groups": len(leaked),
        },
        "leaked_task_group_ids": sorted(leaked),
        "families": rows,
        "method": "Only human-reviewed verifiers are scored on held-out traces.",
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def print_report(report: dict[str, Any]) -> None:
    w, h, s = report["workflow"], report["held_out"], report["scores"]
    print("Bandits end-to-end HITL evaluation")
    print(f"reviewed families:  {w['reviewed_families']}")
    print(f"verifier decisions: {w.get('human_verifier_decisions', 0)}")
    print(f"model label accuracy: {_pct(w.get('model_label_accuracy'))}")
    print(f"exported rows:      {w['exported_rows']}")
    print(f"held-out scored:    {h['scored']} / {h['labeled']}")
    print(f"coverage:           {_pct(s['coverage'])}")
    print(f"success precision:  {_pct(s['success_precision'])}")
    print(f"failure recall:     {_pct(s['failure_recall'])}")
    print(f"trust score:        {_pct(s['trust_score'])}")
    print(f"leaked task groups: {s['leaked_task_groups']}")


def render_html(report: dict[str, Any]) -> str:
    w, h, s = report["workflow"], report["held_out"], report["scores"]
    cards = [
        ("Trust score", _pct(s["trust_score"])),
        ("Success precision", _pct(s["success_precision"])),
        ("Coverage", _pct(s["coverage"])),
        ("Failure recall", _pct(s["failure_recall"])),
        ("Reviewed families", str(w["reviewed_families"])),
        ("Verifier decisions", str(w.get("human_verifier_decisions", 0))),
        ("Model label accuracy", _pct(w.get("model_label_accuracy"))),
        ("Exported rows", str(w["exported_rows"])),
        ("Leakage", str(s["leaked_task_groups"])),
    ]
    card_html = "".join(
        f"<article><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></article>"
        for k, v in cards
    )
    keys = ("descriptor", "labeled", "scored", "tp", "tn", "fp", "fn", "unscored")
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys) + "</tr>"
        for row in report["families"]
    )
    rows = rows or '<tr><td colspan="8">No completed reviewed verifier yet.</td></tr>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Bandits HITL Eval</title><style>body{{font:16px system-ui;background:#0b1020;color:#eef2ff;max-width:1100px;margin:40px auto;padding:0 20px}}p,small{{color:#aeb8d4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:28px 0}}article{{background:#151d34;border:1px solid #293451;border-radius:12px;padding:18px}}small,strong{{display:block}}strong{{font-size:28px}}table{{width:100%;border-collapse:collapse;background:#151d34}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #293451}}th{{color:#aeb8d4}}</style></head><body><h1>Bandits HITL evaluation</h1><p>Sealed truth was opened only after human review and export.</p><section class="cards">{card_html}</section><h2>Held-out confusion matrix</h2><p>TP {h["true_positive"]} · TN {h["true_negative"]} · FP {h["false_positive"]} · FN {h["false_negative"]} · unscored {h["unscored"]}</p><h2>Reviewed families</h2><table><thead><tr><th>Family</th><th>Labeled</th><th>Scored</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Unknown</th></tr></thead><tbody>{rows}</tbody></table></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--traces", type=Path, required=True)
    prep.add_argument("--source", default="otlp")
    prep.add_argument("--redaction", default="secrets-only-v1")
    prep.add_argument("--project", type=Path, required=True)
    starter = sub.add_parser("start")
    starter.add_argument("--traces", type=Path, required=True)
    starter.add_argument("--source", default="otlp")
    starter.add_argument("--redaction", default="secrets-only-v1")
    starter.add_argument("--project", type=Path, required=True)
    starter.add_argument("--labeler", required=True)
    reviewer = sub.add_parser("review")
    reviewer.add_argument("--project", type=Path, required=True)
    reviewer.add_argument("--labeler", required=True)
    labeling = sub.add_parser("prelabel")
    labeling.add_argument("--project", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--project", type=Path, required=True)
    scoring.add_argument("--truth", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    viewer = sub.add_parser("view")
    viewer.add_argument("--report", type=Path, required=True)
    viewer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.traces, args.source, args.project, args.redaction)
        print(f"prepared {result['trace_count']} blind traces in {args.project}")
        print(f"runbook: {args.project / 'RUNBOOK.md'}")
    elif args.command == "start":
        prepare(args.traces, args.source, args.project, args.redaction)
        review_app(args.project, args.labeler)
    elif args.command == "review":
        review_app(args.project, args.labeler)
    elif args.command == "prelabel":
        prelabel(args.project)
    elif args.command == "score":
        result = score(args.project, args.truth)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print_report(result)
        print(f"report:             {args.output}")
    else:
        result = json.loads(args.report.read_text())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_html(result))
        print(f"viewer: {args.output}")


if __name__ == "__main__":
    main()
