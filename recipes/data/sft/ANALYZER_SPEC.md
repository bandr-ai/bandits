# Analyzer-written generation prompt: seed expansion (spec)

## Why
Simia's prompt is domain-tuned (hand-written retail/airline blocks) and failed on new domains. Instead, an analyzer agent writes the generation prompt from the seeds and the harness.

The first pilot (`pilots/2026-10-10-fde-invent/`) showed that asking for "new, different" trajectories makes the generator invent worlds. So generation is now **seed expansion**: keep one reviewed seed's real situation and apply one controlled variation. Only approved, reviewed seeds are used as starting points.

## Loop
analyzer → prompt + variations → generator (one seed + one variation per job) → checks + judge → diagnosis → same analyzer revises. At most `analyzer_rounds` versions.

## Analyzer (`simia-plus analyze`, after `ingest`)
**Input:**
- the harness source (`harness_paths`), treated as established facts;
- up to `analyzer_seeds` dev seeds, with system prompts labelled S1..Sn (`system_labels.json`);
- the tool schemas and real tool results;
- on later rounds: the previous version and its diagnosis.

**Output per version:**
- `harness.md`: task, contracts, tool contracts with exact result shapes, evidence→answer rules, unknowns. Every claim is tagged `[established: file]`, `[inferred]` or `[unknown]`.
- `variations.json`: 6–12 objects with `name, applies_to, requires_tools, teaches, change, preserve, dependencies, correct_when`. Each changes what the agent must recognize, handle or justify; the correct answer may stay the same. None are cosmetic. The analyzer proposes them, so no human labelling is needed.
- `prompt.txt`: the template.
- `rationale.md` and `analyzer_raw.json`.

**Checks on the analyzer's output** (up to 3 tries, with errors fed back):
- required slots `{seed_trace}`, `{variation}`, `{system_prompt}`, `{tools}`, `{generation_id}`; optional `{obs_examples}`; no unknown slots; no held-out seed text;
- at least 3 variations, all keys present, valid labels.

**Prompt layout** (analyzer, diagnosis, judge): material first, each document in its own tag (`<harness_file path>`, `<system_prompt label>`, `<seed id system>`, …), then the request and the JSON output contract last. Output stays JSON (DeepSeek JSON mode). On revision the analyzer must consolidate (replace, merge, resolve conflicts) rather than add rules, and the template's instruction part is capped at about 1,200 words.

## Jobs
- Round-robin over dev seeds. A variation is applicable when it matches the seed's inspector (`applies_to`) and the seed actually calls its `requires_tools`. Each job gets the applicable variation tried least often with that seed, counted over all rounds in `ledger.jsonl`.
- Jobs are planned once per version (`vN/jobs.jsonl`), so a resumed run keeps them.

## Output format (fixed)
- Code appends `OUTPUT_FORMAT`: one JSON object `{"messages": [...]}`.
- Parsed strictly. A failure is rejected and logged, never repaired.

## Acceptance (the same for every version)
- **Pairing:** calls and results paired in order, no duplicate ids.
- **Argument schemas,** after the harness's `tool_arg_defaults`.
- **Tool result shape:** each result must be one of that tool's real top-level shapes (`tool_result_shapes`, taken from the harness code) and carry the harness prefix with a running index (`tool_result_prefix`).
- **Final answer** matches the schema of its system prompt's contract (`answer_schemas_path`).
- **ID provenance.**
- **Judge:** sees the harness description, the seed (a reference, not an answer key), the variation and the generated trace, each in its own tagged block. For each check (`only_declared_change`, `dependencies_updated`, `tool_contracts_respected`, `observations_consistent`, `policy_followed`, `task_success` under `correct_when`) and each config checklist item, it returns a verdict (`pass` / `fail` / `cannot_determine` / `not_applicable`), the message numbers with exact excerpts, and a reason. Code checks every excerpt against the generated trace and the seed: a `pass` without an excerpt that really occurs there does not count. Missing fields and `cannot_determine` fail closed.
- **Dedup:** exact plus near, within a round and across all rounds (`all_rounds.json`).

## Diagnostics (reported, never reject on their own)
- **Diagnosis** reads each sampled trace next to its full seed.
- **Fidelity to the seed:** which failure-context fields changed, platform changes, the share of seed tool results reused, no-op outputs.
- **Per-variation results:** jobs, kept, no-op.
- **Decision cells:** evidence seen so far → action, per assistant turn. Reports the distinct count, max repeats, and how many cells are not in the seeds.
- Outcome counts vs the seeds.

## Held-out (`simia-plus heldout`, only on request)
- The split is by seed id at ingest (`holdout_frac`, `split.json`). Held-out seeds are kept out of `seeds.jsonl`, the banks and the analyzer.
- The chosen version and `simia_prompt=fixed` run on the same held-out seeds, with matched counts, under the same acceptance.
- With about 5 seeds this is a sanity check. Once results influence a revision, those seeds count as dev.

## Out of scope for now
Evaluator calibration sets; raw (unreviewed) traces as anchors; promoting synthetic traces to seeds; a reasoning-rewrite stage (training drops thinking, so there is no reasoning text to rewrite); the student-model training test, which is the real gate for scaling.

## Config
`prompt_source`, `analyzer_seeds`, `analyzer_rounds`, `pilot_jobs`, `heldout_jobs_per_seed`, `holdout_frac`, `harness_paths`, `tool_arg_defaults`, `tool_result_shapes`, `tool_result_prefix`, `answer_schemas_path`, `outcome_fields`.
