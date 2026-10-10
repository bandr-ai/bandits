# Handoff: SFT data recipe (2026-10-10)

## State
- PR bandr-ai/bandits#143, branch `feat/recipe-sft-synth`.
- Next: build `ANALYZER_SPEC.md`.

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
- 20% of seeds call `a11y_search` without args that the schema marks required (a schema mismatch, not a recipe bug).

**Release 1 checks**
- Plain-Simia stats from before `c2ead2e` are invalid: that parser dropped most tool calls, and `rule_check` did not catch orphan tool results.
- Bundles for app.bandr.ai: check every step and history entry before upload; the first batch rendered empty.

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
- Tool results state the conclusion outright.
- Narrow diversity: the same consent-modal and promo stories keep recurring.
- No answer-schema check yet.
- No held-out evaluation yet.
- No training test yet.

## app.bandr.ai
Workspace `eae26b28…` (the FAA expert workspace). Import goes through S3 `bandr-data-478499050241-aps1/expert-review/<key>/` plus SSM on `i-0128e67c5081e82c2`; see `fde-work/eval-labeling/expert-global-30-20261009/upload.py`.

| Batch | Contents |
|---|---|
| `522c8f53…` | 18 baseline traces, built from broken parsing; ignore |
| `c6d46241…` | 25 fixed-prompt traces, results visible |

Invite links expire 2026-10-17.
