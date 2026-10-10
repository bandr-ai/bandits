"""Side-by-side review of seed-expansion traces: each generated trace next to its seed, aligned message by message,
with word-level highlights of what changed, plus the assigned variation and the judge's verdicts.

Writes one self-contained HTML file (no network, nothing published): <out_dir>/analyzer/review.html.
"""
from __future__ import annotations

import difflib
import html
import json
import re
from pathlib import Path

from .io import read_jsonl


def _items(trace: dict) -> list[dict]:
    """Flatten a trace into display rows: user text, each tool call, each tool result, assistant text/answer."""
    rows = []
    for m in trace["messages"]:
        if m["role"] == "user":
            rows.append({"kind": "user", "key": "user", "text": m["content"]})
        elif m["role"] == "tool":
            rows.append({"kind": "result", "key": f"result:{m.get('name', '')}", "text": m["content"]})
        else:
            for c in m.get("tool_calls", []):
                rows.append({"kind": "call", "key": f"call:{c['name']}",
                             "text": f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})"})
            if (m.get("content") or "").strip():
                rows.append({"kind": "answer" if not m.get("tool_calls") else "assistant", "key": "assistant",
                             "text": m["content"]})
    return rows


def _words(text: str) -> list[str]:
    return re.findall(r"\s+|[\w.'-]+|[^\w\s]", text)


def _diff_html(a: str, b: str) -> tuple[str, str]:
    wa, wb = _words(a), _words(b)
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        sa, sb = html.escape("".join(wa[i1:i2])), html.escape("".join(wb[j1:j2]))
        if op == "equal":
            left.append(sa)
            right.append(sb)
        else:
            if sa:
                left.append(f"<del>{sa}</del>")
            if sb:
                right.append(f"<ins>{sb}</ins>")
    return "".join(left), "".join(right)


def _align(seed: list[dict], gen: list[dict]) -> list[tuple[dict | None, dict | None, str]]:
    """Pair rows by kind/tool so a changed result sits next to the result it replaced."""
    pairs = []
    sm = difflib.SequenceMatcher(None, [r["key"] for r in seed], [r["key"] for r in gen], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = seed[i1:i2], gen[j1:j2]
        for k in range(max(len(a), len(b))):
            x, y = (a[k] if k < len(a) else None), (b[k] if k < len(b) else None)
            status = ("same" if x["text"] == y["text"] else "changed") if x and y else ("removed" if x else "added")
            pairs.append((x, y, status))
    return pairs


def _cell(row: dict | None, body: str | None = None) -> str:
    if row is None:
        return '<div class="cell empty"></div>'
    return f'<div class="cell"><span class="tag {row["kind"]}">{row["kind"]}</span><pre>{body or html.escape(row["text"])}</pre></div>'


def _judge_html(v: dict) -> str:
    j = v.get("judge") or {}
    rows = []
    for name, c in (j.get("checks") or {}).items():
        if isinstance(c, dict):
            verdict = str(c.get("verdict"))
            rows.append(f'<tr><td>{html.escape(name)}</td><td class="v {html.escape(verdict)}">{html.escape(verdict)}</td>'
                        f'<td>{html.escape(str(c.get("reason", "")))}</td></tr>')
    issues = "".join(f"<li>{html.escape(r)}</li>" for r in [*v.get("rule_issues", []), *(j.get("turn_issues") or [])])
    unverified = j.get("unverified_excerpts") or {}
    extra = (f'<p class="warn">Unverified excerpts: {html.escape(", ".join(unverified))}</p>' if unverified else "")
    return ((f'<ul class="issues">{issues}</ul>' if issues else "")
            + (f'<table class="judge"><tr><th>check</th><th>verdict</th><th>reason</th></tr>{"".join(rows)}</table>' if rows else "")
            + extra)


CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ins:#dcfce7;--insfg:#14532d;--del:#fee2e2;--delfg:#7f1d1d;
--same:#f5f5f4;--changed:#fef9c3;--added:#dcfce7;--removed:#fee2e2;--pass:#15803d;--fail:#b91c1c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;
--line:#292524;--ins:#14532d;--insfg:#dcfce7;--del:#7f1d1d;--delfg:#fee2e2;--same:#1c1917;--changed:#422006;--added:#052e16;--removed:#450a0a;
--pass:#4ade80;--fail:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 12px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;border-bottom:1px solid var(--line)}
select,input{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
details.trace{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:10px 0}
summary{cursor:pointer;padding:10px 12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.badge{font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid var(--line)}.kept{color:var(--pass)}.rej{color:var(--fail)}
.meta{padding:0 12px 8px;color:var(--muted)}.meta b{color:var(--fg)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:3px 12px}.row.same{opacity:.65}
.row.changed .cell{background:var(--changed)}.row.added .cell:last-child{background:var(--added)}.row.removed .cell:first-child{background:var(--removed)}
.cell{border:1px solid var(--line);border-radius:6px;padding:6px;min-width:0;background:var(--same)}.cell.empty{border-style:dashed;background:transparent}
pre{margin:4px 0 0;white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,monospace}
ins{background:var(--ins);color:var(--insfg);text-decoration:none}del{background:var(--del);color:var(--delfg)}
.tag{font-size:11px;text-transform:uppercase;color:var(--muted)}.cols{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 12px;color:var(--muted);font-size:12px}
table.judge{border-collapse:collapse;margin:6px 12px 12px;font-size:12px;width:calc(100% - 24px)}
table.judge td,table.judge th{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.v.pass{color:var(--pass)}.v.fail,.v.cannot_determine{color:var(--fail)}.issues{margin:6px 12px;color:var(--fail)}.warn{margin:6px 12px;color:var(--fail)}
@media (max-width:700px){.row,.cols{grid-template-columns:1fr}}
"""

JS = """
const f={v:document.getElementById('fv'),x:document.getElementById('fx'),k:document.getElementById('fk'),s:document.getElementById('fs')};
function apply(){let n=0;document.querySelectorAll('details.trace').forEach(d=>{
 const ok=(!f.v.value||d.dataset.version===f.v.value)&&(!f.x.value||d.dataset.variation===f.x.value)
  &&(!f.k.value||d.dataset.kept===f.k.value);d.style.display=ok?'':'none';if(ok)n++;
 d.querySelectorAll('.row.same').forEach(r=>r.style.display=f.s.checked?'none':'');});
 document.getElementById('count').textContent=n+' shown';}
Object.values(f).forEach(e=>e.addEventListener('change',apply));apply();
"""


def build_review(root: Path, seeds: list[dict]) -> Path:
    by_id = {s["id"]: s for s in seeds}
    traces = []
    for vdir in sorted(p for p in root.glob("v*") if p.is_dir() and (p / "verified.jsonl").exists()):
        traces += read_jsonl(vdir / "verified.jsonl")
    blocks, versions, variations = [], set(), set()
    for t in traces:
        m, v = t["meta"], t["meta"].get("verify", {})
        seed = by_id.get(m.get("seed_id"))
        if seed is None:
            continue
        version, variation, kept = str(m.get("version")), str(m.get("variation")), "1" if v.get("kept") else "0"
        versions.add(version)
        variations.add(variation)
        rows = []
        for a, b, status in _align(_items(seed), _items(t)):
            if status == "changed":
                la, lb = _diff_html(a["text"], b["text"])
                rows.append(f'<div class="row changed">{_cell(a, la)}{_cell(b, lb)}</div>')
            else:
                rows.append(f'<div class="row {status}">{_cell(a)}{_cell(b)}</div>')
        spec = m.get("variation_spec") or {}
        meta = "".join(f"<div><b>{k}:</b> {html.escape(str(spec.get(k, '')))}</div>" for k in ("change", "dependencies", "correct_when"))
        if m.get("case"):
            meta += "<div><b>case:</b></div>" + "".join(f"<div>&nbsp;&nbsp;<b>{html.escape(k)}:</b> {html.escape(str(x))}</div>"
                                                     for k, x in m["case"].items())
        if v.get("blind"):
            b = v["blind"]
            meta += (f"<div><b>blind answer:</b> {'agrees' if b.get('agree') else 'insufficient evidence' if b.get('insufficient') else 'disagrees'}"
                     f" {html.escape('; '.join(b.get('diffs') or []))}</div>")
        fid = m.get("fidelity") or {}
        blocks.append(
            f'<details class="trace" data-version="{version}" data-variation="{html.escape(variation)}" data-kept="{kept}">'
            f'<summary><span class="badge">v{html.escape(version)}</span><b>{html.escape(variation)}</b>'
            f'<span class="badge {"kept" if kept == "1" else "rej"}">{"kept" if kept == "1" else "rejected"}</span>'
            f'<span class="badge">seed {html.escape(str(m.get("seed_id"))[:13])}</span>'
            f'<span class="badge">tool results reused {fid.get("tool_results_reused_from_seed", "?")}/{fid.get("tool_results", "?")}</span></summary>'
            f'<div class="meta">{meta}</div>{_judge_html(v)}<div class="cols"><div>SEED</div><div>GENERATED</div></div>'
            f'{"".join(rows)}</details>')
    opts = lambda vals, label: f'<option value="">{label}</option>' + "".join(
        f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in sorted(vals))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Seed expansion review</title><style>{CSS}</style></head><body>'
            f'<h1>Seed expansion review</h1><p class="sub">{len(blocks)} traces. Each generated trace next to its seed; '
            f'yellow rows changed (<del>removed</del> / <ins>added</ins> words), green rows added, red rows removed.</p>'
            f'<div class="bar"><select id="fv">{opts(versions, "all versions")}</select>'
            f'<select id="fx">{opts(variations, "all variations")}</select>'
            f'<select id="fk"><option value="">kept + rejected</option><option value="1">kept</option><option value="0">rejected</option></select>'
            f'<label><input type="checkbox" id="fs"> hide unchanged rows</label><span id="count" class="sub"></span></div>'
            f'{"".join(blocks)}<script>{JS}</script></body></html>')
    out = root / "review.html"
    out.write_text(page)
    return out
