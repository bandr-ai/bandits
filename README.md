<div align="center">

# Bandits

### Build trustworthy datasets from agent traces.

[![quality](https://github.com/vektori-ai/bandits/actions/workflows/quality.yml/badge.svg)](https://github.com/vektori-ai/bandits/actions/workflows/quality.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv)](https://docs.astral.sh/uv/)

Bandits turns real agent runs into labeled SFT data and reusable success checks, scored by what a trace's reactions actually show, without hiding missing evidence.

[Why Bandits?](#why-bandits) · [Quickstart](#quickstart) · [Workflow](#workflow) · [Trust model](#trust-is-a-data-model) · [CLI](#cli-reference)

</div>

---

Agent traces already contain the work: the request, the decisions, the tool calls, and the result. Bandits turns that history into a chain of evidence, from raw runs to reviewed eval and training data.

## Why Bandits?

- **Keep the truth.** Normalize OTLP, chat JSON, and Claude Code traces without inventing missing results or dropping malformed records.
- **Judge by reaction, not claim.** Score each turn by what happened right after it — the tool result, the execution log, the user's next message — never the agent's own summary of what it did. Full write-up and measured numbers: [`docs/next-state-verifier.md`](docs/next-state-verifier.md).
- **Turn a judge into a free check.** An RLM proposes deterministic predicates from the judge's verdicts, re-executes every one over the whole family, and keeps only what actually agrees with it — so scoring a new trace costs no model call.
- **Trace every row.** Follow an exported SFT row back through the checks that fired on it, the judge run, the verifier, and the original corpus, and know whether a human ever reviewed those checks or not.

## Quickstart

Bandits requires Python 3.11 or newer. The repository uses [uv](https://docs.astral.sh/uv/) for reproducible environments.

```bash
git clone https://github.com/vektori-ai/bandits.git
cd bandits
uv sync
```

Ingest the included OTLP fixture and inspect the resulting corpus:

```bash
uv run bandits ingest tests/fixtures/traces.otlp.jsonl --source otlp

# The fixture deterministically produces corpus-67e49fdc2268c1e5.
uv run bandits show corpus-67e49fdc2268c1e5
uv run bandits show corpus-67e49fdc2268c1e5 --issues
```

Bandits writes immutable artifacts to `.bandits/` in the selected project directory. Re-ingesting identical normalized content resolves to the same ID.

### Supported inputs

| Source | Flag | Expected shape |
| --- | --- | --- |
| OpenTelemetry | `--source otlp` | OTLP JSON or JSONL span exports |
| Chat transcripts | `--source chat-json` | One JSON conversation or an array of conversations |
| Claude Code | `--source claude-code` | One session JSONL file or a directory of sessions |

The input format is always explicit. Bandits does not guess and risk accepting a plausible-looking misparse.

## Workflow

Use this to build a verifier that reads what happened after an action, not what the agent claims about it.

```bash
# 1. Group traces into families first, if you want family-scoped checks
#    (skip this and the whole corpus is treated as one family, which is
#    what a benchmark split usually is).
uv run bandits analyze <corpus-id> --tasks
uv run bandits mine-rlm <analysis-id>
uv run bandits materialize-rlm-taskset <clustering-run-id>

# 2. Score every observed turn by its reaction -- the tool result, the
#    execution log, the user's next message -- never the agent's own claim.
#    The archetype (support, coding, computer-use, generic) changes only
#    what counts as a bad reaction in this kind of trace.
uv run bandits judge-turns <corpus-id> --archetype computer-use \
  --task-set <task-set-id> --family <family-id>

# 3. Have an RLM propose cheap check(turn) predicates from the judge's
#    verdicts, re-execute every one over the whole family in a sandbox, and
#    keep only what fires often enough and agrees with the judge.
uv run bandits propose-verifier <judge-run-id> --family <family-id>

# 4. Accept or reject each proposed check. One prompt each; resumable.
uv run bandits review-checks <family-verifier-id>

# 5. Apply the accepted checks, plus the judge, to every trace: pass/fail
#    and a score per trace.
uv run bandits score-traces <family-verifier-id>

# 6. Export scored traces as labeled positive/negative SFT rows.
uv run bandits export-nextstate <verifier-scores-id> --output sft.jsonl
```

Every export also writes a sibling `<name>.unresolved.jsonl`. A trace missing from the corpus, with no observed turns, with a transcript that cannot be rebuilt, or with a turn no check or the judge could actually confirm either way, is quarantined with a reason instead of vanishing or being labeled by a guess.

`score-traces --survivors` applies every check that cleared the automatic precision bar, including ones `review-checks` never saw — useful for a quick look, but the export carries `all_checks_reviewed: false` on every row it produces, so a dataset built this way can never be mistaken for one a human actually reviewed.

Full write-up and measured numbers (where this works, where it doesn't yet, and why): [`docs/next-state-verifier.md`](docs/next-state-verifier.md).

### A quicker path: direct LLM review

For a dataset straight from raw traces, with an LLM reviewing each candidate directly rather than the turn-by-turn pipeline above:

```bash
uv run bandits build-sft <corpus-id> --output work/direct-dataset
```

Writes `sft.jsonl`, `review.jsonl` (borderline candidates), `rejected.jsonl`, and a selection report. Independent of the workflow above — it does not use `judge-turns` or any verifier artifact.

### Duplicates and the held-out split

`materialize-rlm-taskset` moves whole groups when it splits a family, so a declared retry chain never straddles fit and held-out. Lineage ids are read from the source and never inferred, so a source that declares no lineage at all leaves every trace its own.

Two traces of the same normalized request are joined as well, and the two rules compose rather than one falling back to the other: two runs of one request from different sessions are held together despite carrying different lineage ids, and a group joined by lineage on one edge and by an identical request on another moves whole. Normalization changes case and separators only; it preserves identifiers and every other value, so `refund order 7741` and `refund order 8802` remain distinct.

Whole groups move, so the realized held-out share is whatever complete groups come nearest the requested fraction. `judge-turns`, `propose-verifier` and `score-traces` do not currently read this split — nothing in the active pipeline draws on it yet — but `families <task-set-id>` shows it, and it exists so a future calibration step (comparing checks against labels independent of what they were proposed from) has a leak-safe boundary to measure across instead of inventing one after the fact.

### What produced a grouping

A task set records the arm that produced it — the trace view the miner read — along with the model that proposed the families and the id of the clustering run it was materialized from. Families carry no coherence figure and no similarity threshold: nothing measured a distance, and a plausible number in those fields would be fabricated geometry in an artifact whose whole claim is that it used none. Each family's representative is its lexically first member, which is a real trace chosen by a rule that cannot be mistaken for a centrality claim.

A materialized task set also records what this path cannot claim: the miner named a family and placed its members in one context, so membership was never checked by an independent pass.

## Trust is a data model

Bandits keeps source evidence immutable and stores every interpretation beside it as a new derived artifact:

```text
.bandits/
├── artifacts/
│   └── corpus-…/
│       ├── corpus.json
│       └── envelope.json
└── derived/
    ├── analysis-…/
    ├── taskset-…/
    ├── turn_judge_run-…/
    ├── family_verifier-…/
    ├── verifier_scores-…/
    └── nextstate_sft_export-…/
```

That separation matters: revising a check or re-running the judge creates a new artifact; it never rewrites what the source trace recorded, and every downstream artifact names the exact id of what it was built from.

A check's `decision` carries a concrete meaning, set only by `review-checks`:

| Decision | What it means |
| --- | --- |
| `pending` | Proposed and automatically evaluated; no human has looked at it |
| `accepted` | A reviewer confirmed it |
| `rejected` | A reviewer refused it |
| `revised` | A reviewer sent it back with feedback; a new pending check was queued from it, linked by `parent_check_id` |

`score-traces` applies only `accepted` checks by default. `--survivors` widens that to every check that cleared the automatic precision bar regardless of decision — including ones still `pending`, but never ones a reviewer actively `rejected` or `revised` — and the resulting export is marked `all_checks_reviewed: false` so it can never be mistaken for a dataset a human actually reviewed. Review itself hasn't yet been run against the numbers in `docs/next-state-verifier.md`; `--survivors` stood in for it there, and that's stated, not hidden.

## Dataset contracts

Every SFT row, from either exporter, uses chat-completions-shaped `messages`, with assistant `tool_calls` correctly paired to `tool` results and never batched into one turn just because they share a parent span. Rebuilding that transcript is refused, not guessed at, when a trace's own record can't support it — a tool result with no announced call, a call with no recorded result, no recorded user instruction, or user turns that don't run forward through the trajectory all quarantine the row into `<name>.unresolved.jsonl` with the reason.

`export-nextstate` additionally carries: `label` (`positive`/`negative`, from whether the trace passed scoring), `flagged_turns` and `flagged_by` (which check id, or the judge, fired on each one — empty for a positive row), `checks_applied`, `all_checks_reviewed`, and the full `corpus_id` / `family_id` / `verifier_id` / `scores_id` / `judge_run_id` lineage. A trace with any turn no check or the judge could actually confirm is quarantined rather than labeled either way — see `unresolved` in the workflow section above.

`build-sft` additionally rejects a candidate for: recorded tool errors or recovery paths, repeated identical tool actions, an episode long relative to its own step count, or no recorded generating model — demonstration-quality gates, not claims that a successful outcome alone makes behavior worth imitating.

## CLI reference

| Command | Purpose |
| --- | --- |
| `ingest` | Normalize, redact, and store a trace export |
| `list` / `show` | Browse corpora, traces, spans, and ingest issues |
| `analyze` | Extract task candidates and outcome evidence |
| `mine-rlm` | Discover task families by reading raw user requests, with no embedding geometry |
| `audit-rlm` | Advisory: challenge each discovered family in a fresh adversarial context. Changes nothing |
| `materialize-rlm-taskset` / `families` | Turn a clustering run into a task set with lineage-safe fit/held-out splits, and read it back |
| `rlm-session` / `rlm-families` | Watch a running mining or audit session; read its families as reviewable cards |
| `judge-turns` | Score every observed turn by its reaction — the tool result, the execution log, the user's next message — never the agent's claim |
| `propose-verifier` | Have an RLM propose `check(turn)` predicates from the judge's verdicts; re-execute every one in a sandbox and keep survivors |
| `review-checks` | Accept, reject, or revise each proposed check, one prompt each, resumable |
| `score-traces` | Apply the accepted checks and the judge to every trace: flags, an unresolved list, and pass/score per trace |
| `export-nextstate` | Write scored traces as labeled positive/negative SFT rows, plus a quarantine file |
| `build-sft` | A quicker, independent path: LLM-reviewed SFT rows straight from raw traces, no verifier artifact |

Run `uv run bandits <command> --help` for every option.

## Redaction and local state

Ingest uses the `default-v1` redaction ruleset by default. Use `--redaction secrets-only-v1` when email addresses are task identifiers that must be retained:

```bash
uv run bandits ingest traces.jsonl \
  --source chat-json \
  --redaction secrets-only-v1 \
  --project ./my-experiment
```

The source digest and selected redaction ruleset are part of corpus identity, so changing redaction produces a different content-addressed artifact. Local `.bandits/` state and `.env` credentials are ignored by Git; choose or ignore export paths according to your own data-retention policy.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest --cov=bandits --cov-report=term-missing
```

The family coherence audit is an optional extra, since it pulls a REPL sandbox
and reaches a model:

```bash
uv sync --extra audit
```

The test suite injects a predictor instead of calling one, so neither the extra
nor a credential is needed to run it.

The test suite exercises ingestion fidelity, redaction, content-addressed storage, RLM task-family mining, the next-state judge and its RLM-proposed checks, and both export paths.

## Project map

```text
bandits/
├── ingest/      # OTLP, chat JSON, and Claude Code adapters
├── analyze/     # task extraction, evidence, and RLM family discovery
├── verify/      # turns, the next-state judge, RLM-proposed checks, and their review
├── export/      # next-state SFT export, and the direct LLM-reviewed exporter
├── traces.py    # immutable canonical trace contracts
├── store.py     # content-addressed corpus and derived-artifact storage
├── redact.py    # deterministic redaction policies
└── cli.py       # Typer command-line interface

tests/           # pytest suite, mirroring the package layout
├── fixtures/    # trace fixtures the suite reads
└── tasksets.py  # shared task-set helpers

scripts/         # manual and paid runs, not collected by pytest
```

Bandits is intentionally domain-agnostic: coding agents, support workflows, browser automation, research, API agents, and other tool-using systems all enter through the same evidence model. Domain-specific definitions of success belong in reviewable verifier checks, not hidden inside the trace format.
