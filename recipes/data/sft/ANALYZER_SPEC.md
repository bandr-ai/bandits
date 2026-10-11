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
- `combinations.json`: pairs of variations that can be applied to one seed together (may be empty).
- `prompt.txt`: the template.
- `rationale.md` and `analyzer_raw.json`.

**Checks on the analyzer's output** (up to 3 tries, with errors fed back):
- required slots `{seed_trace}`, `{variation}`, `{system_prompt}`, `{tools}`, `{generation_id}`; optional `{obs_examples}`; no unknown slots; no held-out seed text;
- at least 3 variations, all keys present, valid labels.

**Prompt layout** (analyzer, diagnosis, judge): material first, each document in its own tag (`<harness_file path>`, `<system_prompt label>`, `<seed id system>`, …), then the request and the JSON output contract last. Output stays JSON (DeepSeek JSON mode). On revision the analyzer must consolidate (replace, merge, resolve conflicts) rather than add rules, and the template's instruction part is capped at about 1,200 words.

## Jobs
- Round-robin over dev seeds. A variation is applicable when it matches the seed's inspector (`applies_to`) and the seed actually calls its `requires_tools`. Each job gets the applicable variation tried least often with that seed, counted over all rounds in `ledger.jsonl`.
- With `combo_share` > 0, that share of jobs applies a combination (both variations applicable to the seed), as one merged spec.
- Jobs are planned once per version (`vN/jobs.jsonl`), so a resumed run keeps them.

## Output format (fixed)
- Code appends `OUTPUT_FORMAT`: one JSON object `{"messages": [...]}`.
- With `case_block` (or the `case` arm of `ab`): `OUTPUT_FORMAT_CASE`, `{"case": {assigned_change, kept, dependent_observations, unavailable_or_uncertain}, "messages": [...]}`. The case describes the changed situation, states no answer and cannot widen the variation. It is stored in `meta.case` for the judge and review, never in the training messages.
- Parsed strictly. A failure is rejected and logged, never repaired.

## Acceptance (the same for every version)
- **Pairing:** calls and results paired in order, no duplicate ids.
- **Argument schemas,** after the harness's `tool_arg_defaults`.
- **Tool result shape:** each result must be one of that tool's real top-level shapes (`tool_result_shapes`, taken from the harness code) and carry the harness prefix with a running index (`tool_result_prefix`).
- **Final answer** matches the schema of its system prompt's contract (`answer_schemas_path`).
- **ID provenance.**
- **Judge:** sees the harness description, the seed (a reference, not an answer key), the variation and the generated trace, each in its own tagged block. For each check (`only_declared_change`, `dependencies_updated`, `tool_contracts_respected`, `observations_consistent`, `policy_followed`, `task_success` under `correct_when`) and each config checklist item, it returns a verdict (`pass` / `fail` / `cannot_determine` / `not_applicable`), the message numbers with exact excerpts, and a reason. Code checks every excerpt against the generated trace and the seed: a `pass` without an excerpt that really occurs there does not count. Missing fields and `cannot_determine` fail closed.
- **Per-turn behaviour:** the judge returns one entry per assistant message (`verdict`, `relies_on`, reason). Code rejects a trace when any assistant message lacks a `justified` entry, cites nothing, or cites anything but earlier messages (0 ≤ n < its number) or `"system"` (hindsight). The judge and blind check see every tool result untruncated. The harness rules and system policy decide correctness; a variation's `correct_when` is a hypothesis the judge checks against them, and keeping the seed's answer is valid when the changed observations still support it. For behaviour, only earlier messages count as evidence, never the case or variation.
- **Case consistency** (case traces only): observations express the case and the case stays within the variation.
- **Blind answer** (`features.blind_check`): a separate call gets the system prompt and the conversation without its final answer, nothing else, and answers (or says `insufficient_evidence`). Compared on the contract's categorical fields (enums, booleans), not wording or confidence. Disagreement or insufficient evidence makes the trace **unresolved**: not exported, kept for diagnosis. Agreement corroborates; it does not validate the observations. With `features.blind_gate` false (the A/B default) the check is diagnostic only: disagreement is recorded and reported (`kept_unresolved`) but does not block the trace; quarantine those before any training export.
- **Evaluation errors:** a judge call that fails (including the token limit, never retried) or returns a malformed response (missing a required check, `turns` or `checklist`; retried once) is an `eval_error`: not kept, and reported separately, never counted as a rejection.
- **Quarantined seeds** (`quarantine_seeds`: id or prefix -> reason): kept on disk, never used as seeds or in observation banks.
- **Dedup:** exact plus near, within a round and across all rounds (`all_rounds.json`).

## Diagnostics (reported, never reject on their own)
- **Diagnosis** reads each sampled trace next to its full seed.
- **Fidelity to the seed:** which failure-context fields changed, platform changes, the share of seed tool results reused, no-op outputs.
- **Per-variation results:** jobs, kept, no-op.
- **Decision cells:** evidence seen so far → action, per assistant turn. Reports the distinct count, max repeats, and how many cells are not in the seeds.
- Outcome counts vs the seeds, and outcomes per variation (dataset-level shortcuts).
- Nearest similar kept trace or seed per kept trace (word 3-gram Jaccard): a repetition diagnostic, never a filter.

## Generator A/B (`simia-plus ab --version N --jobs 20`, only on request)
The same planned jobs run twice with version N's template: `plain` (OUTPUT_FORMAT) and `case` (OUTPUT_FORMAT_CASE). Both get every check including the blind answer; only `case` has a case to check. `analyzer/ab_vN/comparison.json`: kept, unresolved, reject reasons, per-variation results and outcomes, fidelity, nearest-similar counts, decision cells, cost per unique kept. A higher kept rate alone does not decide; read traces from both arms.

## Held-out (`simia-plus heldout`, only on request)
- The split is by seed id at ingest (`holdout_frac`, `split.json`). Held-out seeds are kept out of `seeds.jsonl`, the banks and the analyzer.
- The chosen version and `simia_prompt=fixed` run on the same held-out seeds, with matched counts, under the same acceptance.
- With about 5 seeds this is a sanity check. Once results influence a revision, those seeds count as dev.

## Out of scope for now
Evaluator calibration sets; raw (unreviewed) traces as anchors; promoting synthetic traces to seeds; a reasoning-rewrite stage (training drops thinking, so there is no reasoning text to rewrite); the student-model training test, which is the real gate for scaling.

## Config
`prompt_source`, `analyzer_seeds`, `analyzer_rounds`, `pilot_jobs`, `heldout_jobs_per_seed`, `holdout_frac`, `harness_paths`, `tool_arg_defaults`, `tool_result_shapes`, `tool_result_prefix`, `answer_schemas_path`, `outcome_fields`.
