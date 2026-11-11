# Cleanup plan: start from zero, keep only ingest

## The rule
We build bandits back up from scratch. **Only ingest stays**, plus the 3 files it can't run without. Everything else is deleted, not hidden or commented out. Git history is the archive. Anything needed later gets rebuilt fresh on top of ingest; nothing is restored by default.

## What we accept breaking
| Breaks | Accepted |
|---|---|
| **The `bandits` command (CLI)** | Yes. Deleted entirely. |
| **`recipes/`** (Jev, SFT) | Yes. The folder stays in the repo **untouched**, even though it imports code we delete. |
| **PyPI** | No release this week. `bandits-cli` 0.2.0 stays as the published version. |

---

## 1. What stays

```
bandits/
  __init__.py      # version
  traces.py        # the trace format ingest produces
  redact.py        # strips secrets (keys, tokens) while reading traces
  store.py         # saves an ingested corpus to disk under a content-hash id
  ingest/          # every trace reader (OTLP, chat JSON, JSONL, Claude Code, TRAIL, FailProofAI, …)
    genai.py       # MOVED from bandits/genai.py (OTel field names; only ingest uses them)
    jsonarray.py   # MOVED from bandits/jsonarray.py (streams big JSON arrays; only ingest uses it)

tests/
  ingest/          # minus the 2 script tests listed below
  fixtures/
  conftest.py, __init__.py
  traces_test.py, redact_test.py, store_test.py

scripts/
  smoke_wheel.sh   # rewritten: install the wheel, call ingest from Python on a fixture

docs/
  ingest-sources.md
  ingest-validation.md
  releasing.md

recipes/           # untouched, may be broken
README.md          # rewritten, short, ingest only
CHANGELOG.md       # new "0.3.0 (unreleased)" entry listing every removal
pyproject.toml     # trimmed (see §3)
.github/workflows/ # quality.yml runs ruff + pytest tests/ only
```

---

## 2. What gets DELETED (the complete list)

### 2.1 `bandits/`: about 23k lines

| Delete | What it was | Lines |
|---|---|---|
| `bandits/cli.py` | The `bandits` command and all 24 subcommands | 2,306 |
| `bandits/verify/` (`__init__`, `turns`, `nextstate`, `judge`, `propose`) | The old audit: turn judge, +1/0/−1 scoring, proposed checks | 2,180 |
| `bandits/analyze/` (all 17 files, RLM included) | Task candidates, task families, RLM mining and audit | 6,685 |
| `bandits/emulate/` (all 15 files) | AWM / TraceWorld simulation | 9,075 |
| `bandits/export/` (`__init__`, `sft`, `direct_sft`, `nextstate_sft`, `models`) | SFT exports and defect flags | 1,154 |
| `bandits/inspect.py` | The HTML trace viewer page | 819 |
| `bandits/fields.py` | The CLI's `fields` command | 345 |
| `bandits/ledger.py` | Records model calls | 247 |
| `bandits/providers.py` | LiteLLM model routing | 174 |
| `bandits/transport.py` | HTTP retries for model calls | 125 |

Also check `ingest/mapping.py` and `ingest/workflow.py`. They served the CLI's `mapping` commands and the old pipeline steps. Delete them if nothing left in `ingest/` imports them.

### 2.2 `tests/`: everything not about ingest
- `tests/analyze/` (all)
- `tests/emulate/` (all)
- `tests/export/` (all)
- `tests/verify/` (all)
- `tests/cli_test.py`, `tests/fields_test.py`, `tests/inspect_test.py`, `tests/ledger_test.py`, `tests/providers_test.py`, `tests/transport_test.py`
- `tests/tasksets.py`, `tests/tau2_to_otlp_test.py`, `tests/trail_nextstate_eval_test.py`
- `tests/ingest/crosscheck_ingest_test.py`, `tests/ingest/validate_openinference_export_test.py` (their scripts are deleted)

### 2.3 `scripts/`: everything except `smoke_wheel.sh`
- **AWM:** `audit_awm_coverage.py`, `rescore_awm_fidelity.py`, `run_agentic_awm_smoke.py`, `run_awm_fidelity.py`
- **RLM:** `rlm_smoke_test.py`
- **Judge evals:** `tau2_nextstate_eval.py`, `tau2_to_otlp.py`, `trail_nextstate_eval.py`
- **One-off ingest checks:** `audit_source.py`, `crosscheck_ingest.py`, `validate_openinference_export.py`
- **Other:** `prepare_exgentic_dogfood.py`, `replay/span_dump.py`

### 2.4 `docs/`: everything except the 3 ingest/release docs
- **AWM:** `awm-decisions-v3.md`, `awm-decisions-v4.md`, `awm-issues.md`, `awm-plan.md`, `reasoning-awm-decisions.md`, `reasoning-awm-decisions-v2.md`
- **RLM:** `rlm-task-family-mining-plan.md`
- **Old audit:** `next-state-verifier.md`
- **Paper figures:** `docs/assets/` (all 9 images) and `docs/figures/` (all 5 files)

### 2.5 `pyproject.toml`
- **dependencies:** remove `typer` and `rich`. Keep `pydantic`.
- **extras:** remove `llm`, `audit` and `emulate`. Keep `dev`.
- remove `[project.scripts]`, since there's no `bandits` command
- description: "Read agent traces from any format into one trace model."
- `pytest` collects `tests/` only, so the broken recipes aren't collected

### 2.6 PRs and branches
- **#148** (Langfuse ingest): **merge** if green, since it's ingest
- **close** **#150** (rank), **#140** (RLM), **#139** (model-run recording) and **#126** (Jev docs), each with a link to this plan
- **delete** the 42 remote branches already merged into `main`, and any branch older than 30 days with no open PR

---

## 3. Order

| # | Do | Done when |
|---|---|---|
| 1 | Branch `chore/ingest-only` off `main`. Merge #148 first. | |
| 2 | Delete everything in §2.1. Move `genai.py` and `jsonarray.py` into `ingest/` and fix their imports. | `python -c "import bandits.ingest, bandits.store, bandits.redact, bandits.traces"` works |
| 3 | Delete §2.2–2.4. Trim `pyproject.toml` (§2.5). Rewrite `smoke_wheel.sh`, the README and CI. | `pytest tests/` green, ruff clean |
| 4 | Open the PR, merge it, close the old PRs, delete the branches (§2.6) | `main` is ingest-only |

## 4. Checks before merging

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest tests/
uv build && bash scripts/smoke_wheel.sh dist/bandits_cli-*.whl

# nothing outside recipes/ points at deleted code
git grep -nE "bandits\.(cli|verify|analyze|emulate|export|inspect|fields|ledger|providers|transport|genai|jsonarray)\b" -- bandits tests scripts docs README.md pyproject.toml .github
# expect: no output (the moved genai/jsonarray are now bandits.ingest.genai / bandits.ingest.jsonarray)
```
