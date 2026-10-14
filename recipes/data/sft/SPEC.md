# Spec: seed traces → more traces, no environment

## Goal
Take N good agent traces and return M new traces for SFT, e.g. 100 golden traces → 1,000 or 10,000. Assume no
environment: nothing can be executed, and only the traces themselves are available.

**Base:** Simia (Li et al., arXiv 2511.01824). Its code is copied into `simia_plus/simia.py` and its prompts into
`simia_plus/prompts.py` (MIT, `LICENSE-SIMIA`). With every feature off, a run is plain Simia: one LLM call per new
trace, using one seed as the example, followed by Simia's post-processing. Each improvement is a feature flag, so it
can be measured against that baseline.

**Not specific to one benchmark.** Simia's released generator only has prompts for τ²'s retail and airline domains.
Here those domain blocks are replaced by one generic block, and seeds can be ShareGPT (Simia),
OpenAI/LangChain-style messages, or LLM-gateway logs (`final_messages` + `turns`).

## Pipeline
| Stage | Output | What happens |
|---|---|---|
| `ingest` | `seeds.jsonl`, `obs_bank.jsonl`, `user_bank.jsonl` | Normalise the seeds. Optionally decontaminate them against eval traces (`decontam_paths`), and optionally run Simia's LLM pre-filter (`check_seeds`). Mine real tool results and real user turns |
| `plan` | `jobs.jsonl` | `ceil(target_count × overgen)` jobs, round-robin over **all** seeds. Each job gets a strategy, an optional persona, an optional failure, and a mode |
| `specs` | `specs.jsonl` | Only with `spec` on. One LLM call per seed writes that seed's specs as a batch |
| `generate` | `generated.jsonl` | One trace per job, in mode `simia`, `simia_env` or `split` |
| `verify` | `verified.jsonl` | Rules always run (Simia's post-processing as checks); provenance and the judge are optional |
| `select` | `final/` | Exact and optional near-duplicate removal, then balanced selection down to `target_count`. Writes `report.json` |

All stages resume: re-running skips work that is already done. Failed items go to `*.errors.jsonl` and are retried.

## Generation modes
- **`simia`** (default): Simia-SFT. One call writes user, assistant, calls and results in Simia's text format. The output is parsed with Simia's parser, then Simia's `fix_arguments` repair and `should_delete_conversation` filter run.
- **`simia_env`**: Simia-RL's environment. A teacher agent acts with native tool calling, and one simulator model plays both the user and the tools using Simia's prompt unchanged, with the seed as reference data. The conversation starts from the seed's first user message, as in Simia-RL.
- **`split`**: separate calls per step.
  - The user simulator gets a persona plus real user-turn style examples.
  - The agent is a teacher using native tool calling.
  - The tool simulator answers from an explicit JSON state and returns ops; code applies them (StateGen 2606.16307).
  - Splitting the roles stops the simulator from inventing a helpful user to rescue a failing agent.
  - Needs `spec`.

## Features (all off = Simia)
| Flag | Adds | Why (source) |
|---|---|---|
| `check_seeds` | LLM check on seeds: complete / logical / well-formatted | Simia §2.2 (described in the paper, not in its code) |
| `strategies` | A mix of six strategies instead of only "new scenario": rephrase, entity_swap, extend, compose, variant_outcome, new_scenario | BeyondWeb: "generate more like this" ≈ repetition, while several grounded transformation strategies keep improving |
| `spec` | Scenario spec per job: goal, user facts, initial state, expected final state, expected behavior. Written in batches per seed | Proxy-State Eval 2602.16246 (a careful spec gives near-zero simulator hallucination); batch diversity (Adaption) |
| `persona` | Realistic users. Applied automatically only to seeds with ≥2 user turns, because single-request agents have no conversation to simulate | Sim2Real user gap 2603.11245; PPol 2605.12894; MIMESIS 2610.09484 |
| `failure` | One injected tool failure (timeout, server_error, not_found, permission_denied, invalid_argument) at a sampled call; the agent must recover | When Simulation Lies 2605.11928; Kimi K2; HAT |
| `retrieval` | Real recorded tool results from the seeds, given as format reference | Simia-RL already replays reference results; MirrorAPI 2503.20527 |
| `loop` | A share (`loop_frac`) of jobs runs per step (`loop_style`: `simia_env` or `split`) | Simia-RL; StateGen |
| `provenance` | Drop traces whose ID-like tool arguments never appeared earlier | SAP 2609.06124 |
| `judge` | LLM audit against the spec and a checklist. The judge's reconstructed final state is diffed in code against the expected state. Failures can be kept with `bad_steps` for loss masking | Proxy-State Eval; Agent World Model 2602.10090; Adaption checklists; keep-failures (Terminal-World) |
| `near_dedup` | 3-gram Jaccard on the first user message, within the same tool-sequence bucket | Datology curation |

## Datology / Adaption principles and where they live
| Principle | Source | Implementation |
|---|---|---|
| Ground synthesis in real data; "more like this" ≈ repetition | BeyondWeb, Datology synthetic-data primer | Every job starts from a real seed. `strategies` adds grounded transformations. `retrieval` copies real result formats |
| Many strategies beat one | BeyondWeb | 6 strategies, weighted by `strategy_weights` |
| HQ seeds matter more than novel LQ ones | BeyondWeb | `check_seeds`; the rule check reports seed problems at ingest |
| Accumulate, never replace | Datology primer; Gerstgrasser et al. | `final/seeds_plus_synthetic_sharegpt.jsonl` always includes the seeds |
| Curate from a bigger pool | Datology (pool size) | `overgen` (default 1.5×), then filter and select |
| Keep coverage, not just the head | Datology 20/20 VLM ("rare-but-useful") | Selection round-robins over (seed, strategy) buckets, so no seed or strategy is crowded out |
| Decontaminate before synthesis | Datology Curation Studio | `decontam_paths` applies 8-gram overlap to the seeds at ingest and to outputs at verify |
| Brevity / cost-of-pass; length-controlled judges | Datology "Brevity" | The judge prompt says not to prefer longer answers; within a bucket, selection prefers successes, then shorter traces |
| One-sample-per-call collapses diversity; add constraints after | Adaption "Invent" | Specs are generated in batches per seed with "differ from each other". Persona and failure are attached in code after the strategy is chosen |
| Checklist QC; any applicable miss drops the trace | Adaption agentic checklists | `checklist` in config (default 5 items). `checklist_failed` non-empty means rejected |
| Measure diversity | Adaption "Invent" | `report.json`: distinct action signatures, signature entropy, first-message distinct 3-gram ratio, counts per strategy / persona / mode / seed, plus the seeds' own numbers for comparison |
| Match production conditions | Adaption | Seeds keep their own system prompt and tool schemas; generated traces reuse them |
| Synthetic SFT can be worse than the base model | Adaption "Invent" | Gate 1 below: always compare against the untuned base |

## Gates (outside this tool; matched count; evaluate on held-out real traces)
1. Untuned base vs seeds-only vs seeds + plain Simia.
2. Plain Simia vs Simia + one feature at a time. Keep only features that help.
3. `simia` vs `simia_env` vs `split` at equal count.

## Not built yet
- HSA harness variants (HAT 2608.15763): rename tools and skills, reorder prompt blocks, perturb limits.
- A fine-tuned tool simulator trained on `obs_bank.jsonl` (MirrorAPI).
- Embedding-based near-dedup and diversity (the current versions are lexical).
