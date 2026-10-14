# Seed-only synthesis

The analyzer reads all supplied dev seeds, their embedded system prompts and supplied tool definitions. Held-out seeds never enter analysis or observation banks. No harness source, external cause catalog, raw-corpus pool or world model is used. `harness_paths` is no longer a valid config field.

No agent receives an external harness or reconstructed implementation rules. `observed_contracts.md` is only a summary of the supplied traces, prompts and tool definitions, with unknowns left unknown. It is not a replacement harness.

## Generation

1. Extract signals: demonstrated decision, driving evidence, source seed ids, and a useful contrast or invariance where justified. No fixed skill taxonomy or signals-per-seed limit.
2. Write 3–5 simple variations, the generator prompt, and a short description of explicitly supplied contracts and observed formats. Observation is not proof of an implementation rule. Unknown tool branches remain unknown.
3. Assign one reviewed seed and one applicable variation per job using the existing ledger. Each variation includes its signal descriptions, what changes, what stays and dependencies.
4. Generate a complete conversation. With `case_block: true`, first describe changed/preserved facts, dependent observations and uncertainty, without a target verdict. The case is metadata and is removed from training messages. Update assistant calls and stopping as needed; do not freeze the seed's action sequence.
5. Existing structural checks and the seed-aware judge inspect the result. The judge uses only the supplied policy, demonstrated formats and visible evidence, not imagined implementation rules.
6. Diagnosis reviews failures and weak supervision; the analyzer revises signals, variations and prompt. The existing loop supports three rounds; the shipped defaults run a second round only when the first keeps under half of its jobs. The automatic prompt pick by usable rate remains provisional; it does not establish training usefulness.

The initial experiment uses single small changes, no combinations and no blind-answer call. Existing combination support remains available for later experiments; it is not part of this first run. Format, tool-use and answer checks remain useful infrastructure, not the focus of generation design.

## Artifacts

Each version stores `signals.json`, `observed_contracts.md`, `variations.json`, `combinations.json`, `prompt.txt`, `rationale.md`, `analyzer_raw.json`, jobs, candidates and reports. The generator receives the chosen variation's full signal descriptions as metadata. Legacy versions containing harness-derived prompts cannot resume through this flow; use a fresh output directory.

`observed_contracts.md` summarizes the supplied examples. It must distinguish explicit prompt/definition rules from examples and unknowns. It must not invent step limits, runtime hooks, defaults, successful tool formats or cause vocabularies.

## Run after reviewing config

Use `simia_plus/defaults/pipeline_fireworks.json` and set `seeds_path` to reviewed seed data supported by the existing ingest. If seed tool definitions are already embedded, no external registry is needed. Otherwise provide only the customer's actual tool definitions, never inferred implementation source. Export the API key the config names (`api_key_env`).

```bash
.venv/bin/python -m simia_plus ingest --config simia_plus/defaults/pipeline_fireworks.json
.venv/bin/python -m simia_plus analyze --config simia_plus/defaults/pipeline_fireworks.json
.venv/bin/python -m simia_plus review --config simia_plus/defaults/pipeline_fireworks.json
```

`ingest` makes no model calls with this config. `analyze` makes paid calls. No calls are needed to run `.venv/bin/python -m pytest -q`.

For a small first look, set `analyzer_rounds: 1` and `pilot_jobs: 3`; use a fresh output directory. Read the cases and generated conversations next to their seeds before increasing scale. The full three-round loop is already supported.
