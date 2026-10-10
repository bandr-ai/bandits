# Pilot 2026-10-10: analyzer v1–v3 on FAA seeds ("invent" baseline)

The first analyzer pilot, built before seed expansion existed. The analyzer was asked for "a new, different trajectory" per seed, and it wrote prompts that invent new apps, sites and failures. That made it a useful baseline but not training data. Traces are not in the repo; they live in `fde-work/sft-analyzer-pilot/out/`.

## Setup
- **Seeds:** the 17 reviewed FAA inspector traces, split by seed id into 12 dev and 5 held out (`split.json`). The held-out comparison was stopped before anything was verified, so the 5 seeds are still unused.
- **Model:** `deepseek-v4.1-flash` pinned to Relace for every role. 20 jobs per round, 3 rounds. Cost: $0.40 over 127 calls.
- **Checks:** strict output format, call/result pairing, argument schemas, and an answer contract per system prompt. The contracts come from `derive_answer_schemas.py`, built on agentic-test `failure_analyzer/prompts.py`. On top of those: ID provenance, and a judge with the fixed 5-item checklist in `pilot.json`.

## Files
- `v{1,2,3}/prompt.txt`: the analyzer-written generator prompt.
- `rationale.md`: the analyzer's reasoning.
- `diagnosis.md`: the diagnosis agent's notes, which fed the next version.
- `report.json`: pilot metrics for that version.
- `comparison.json`: every set re-scored with the same code checks (`compare.py`): the real seeds, the old Simia runs, and v1–v3.

## Results (comparison.json; v3 partial, 18 of 20)
| | real seeds | Simia plain | Simia fixed | v1 | v2 | v3 |
|---|---|---|---|---|---|---|
| code checks pass | 13/17* | 3/51 | 25/51 | 20/20 | 20/20 | 18/18 |
| root_cause null | 94% | 18% | 22% | 30% | 40% | 100% |
| tool results naming a verdict | 0% | 8% | 4% | 0% | 0% | 0% |
| tool results unavailable | 17% | 1% | 4% | 4% | 4% | 9% |
| user context similarity to own seed | – | 0.14 | 0.21 | 0.10 | 0.11 | 0.09 |

\* The 4 failures were `a11y_search` calls without `role`/`name`. The harness fills both from `arg_defaults`, so the check was wrong; it is now fixed with `tool_arg_defaults`.

## Takeaways
- **Better than Simia on validity and tool honesty:** no leaked verdicts, no call loops.
- **Invented worlds.** Context similarity to the seed is lower than Simia's.
- **The outcome mix followed the prompt's wording, not the evidence.** It swung from 70% named causes in v1 to 0% in v3.
- **Invented tool output shapes.** 9 of 12 `query_har` results and 1 of 1 `walk_history` results don't match the real tool. The seeds never show these tools succeeding, so the generator guessed.
- **The judge passed nearly everything**, so usable rate didn't separate the versions.

These findings led to seed expansion: harness source as analyzer input, one assigned variation per job, the real tool-shape check, and fidelity diagnostics (`../../ANALYZER_SPEC.md`).
