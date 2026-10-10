# Handoff: SFT data recipe (2026-10-10)

## State
- PR bandr-ai/bandits#143, branch `feat/recipe-sft-synth`.
- Analyzer built for seed expansion (`simia_plus/analyzer.py`, `simia-plus analyze`; see `ANALYZER_SPEC.md`). Not piloted yet: **no pilot runs without the user's go-ahead.**
- First ("invent") pilot recorded in `pilots/2026-10-10-fde-invent/` (prompts, rationales, diagnoses, reports; traces stay in fde-work).
- Student model for training: `Qwen/Qwen3.8-27B` (fde-ft, uncommitted on its `main`); its GPU settings are still sized for 4B.

## Gotchas
**LiteLLM gateway (`ai-gateway-staging.testsigma.com/v1`)**
- It caches identical requests. Keep `no_cache` on and the generation id in the prompt, or you get duplicate traces and useless retries.
- Key env var: `LITE_LLM_KEY` (in `fde-work/.env`).
- Model: `openrouter/deepseek/deepseek-v4.1-flash`. Pin Relace via `extra.extra_body.provider = {"only": ["relace"], "allow_fallbacks": false}`. Check `provider` in `llm_calls.jsonl`.

**DeepSeek v4.1 flash**
- It writes planning text, and sometimes fake role tags, into `content`.
- Untried fixes: `reasoning_effort` and BEGIN/END markers.
- It also puts reasoning before the call JSON without `<think>`; the parser now handles that.

**fde seeds**
- Stored in gateway-log format (`final_messages` + `turns`). Tools are names only, so set `tool_defs_path` to `fde-ft/tool_defs.json`.
- 4/17 seeds call `a11y_search` without `role`/`name`: valid, the harness fills them from `arg_defaults` (`tools/_base.py`). Set `tool_arg_defaults`.
- `query_har` / `walk_history` only ever return `available:false` in the seeds; generators invent their success shapes unless given `tool_result_shapes` and the harness source.

**Release 1 checks**
- Plain-Simia stats from before `c2ead2e` are invalid: that parser dropped most tool calls, and `rule_check` did not catch orphan tool results.
- Bundles for app.bandr.ai: check every step and history entry before upload; the first batch rendered empty.

## Analyzer pilot (durable: `fde-work/sft-analyzer-pilot/`)
- `pilot.json` config; `seeds_golden.jsonl` (the 17); `out/` all outputs (layout in ANALYZER_SPEC); `analyze.log`.
- `answer_schemas.json` from `derive_answer_schemas.py`: parses each seed system prompt's output contract (source of truth: agentic-test `failure_analyzer/prompts.py` @ 3fd5e445). 14 distinct prompts (sweep/escalation prompts embed run-specific cause rows). All 17 seeds pass.
- Split by seed id, `random_seed` 0: 12 dev / 5 held out (`out/split.json`).
- Rerun: `set -a; . fde-work/.env; set +a; simia-plus ingest|analyze --config fde-work/sft-analyzer-pilot/pilot.json` (resumes; delete `out/analyzer/vN` to redo a version).

## Data
- Seeds: the 30 traces in `fde-work/eval-labeling/traces`, with reviewer verdicts in `outcome-check/binary-audit.json` (`expected_binary`). Used: the 17 marked `success`, all `faa.reason`.
- Seed ID = `turns[0].cid` of a row in `filtered/02_capped_cap2/kept.jsonl`.

## Results (17 seeds, 51 jobs)
| Run | Usable / unique | Notes |
|---|---|---|
| Plain Simia | 18 / 6 | Cache on; old parser |
| Fixed prompt | 25 / 25 | Cache off; new parser |

Total spend ≈ $0.75. Run outputs are in the session scratchpad (not durable). Rerun via `configs/` and the gateway settings above.

**Open issues**
- Seed-expansion pilot not run yet (needs the user's go-ahead). Config ready: `fde-work/sft-analyzer-pilot/pilot_expand.json` (copy with scrubbed paths in `pilots/next-seed-expansion/`) (fresh `out_expand/`, same split copied in; harness = agentic-test `failure_analyzer/tools`, `cause_catalog.json`, `check_specs.py`, `shaping.py`, `fde-ft/tool_defs.json`, ~114k chars). With its arg defaults, result shapes and prefix, all 17 seeds pass every code check. Run: `ingest` then `analyze`.
- Judge was lenient (passed ~all) when it could not see the seed; now it gets seed + variation + harness. Smoke test (`fde-work/sft-analyzer-pilot/judge_smoke.py`, 5 hand-built cases from 2 dev seeds x 2 runs, $0.03): 10/10 as expected after fixing excerpt matching (judge quotes unescape JSON and use "..."; matching now tolerates that, invented quotes still fail). Unchanged seed and a legit evidence change pass; fake tool shape (code), unsupported cause and undeclared platform drift fail. Tiny sample, not a calibration.
- 17 reviewed seeds bound how many distinct situations exist; measure coverage (decision cells) before scaling.
- No held-out evaluation and no training test yet.

## app.bandr.ai
Workspace `eae26b28…` (the FAA expert workspace). Import goes through S3 `bandr-data-478499050241-aps1/expert-review/<key>/` plus SSM on `i-0128e67c5081e82c2`; see `fde-work/eval-labeling/expert-global-30-20261009/upload.py`.

| Batch | Contents |
|---|---|
| `522c8f53…` | 18 baseline traces, built from broken parsing; ignore |
| `c6d46241…` | 25 fixed-prompt traces, results visible |

Invite links expire 2026-10-17.
