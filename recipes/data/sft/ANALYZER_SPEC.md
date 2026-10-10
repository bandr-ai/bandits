# Analyzer-written generation prompt (spec)

## Why
Simia's prompt is domain-tuned (hand-written retail/airline blocks) and failed on new domains. Instead, an analyzer agent writes the entire generation prompt from the seeds.

## Contract
**Analyzer**
- Input: a sample of seeds (≤20, never the held-out ones), their tool schemas and system prompts, real tool-result examples, and, on later rounds, the diagnosis notes.
- Output: `prompt_vN.txt` (a template), `checklist_vN.json` (yes/no requirements, tagged by inspector/tool), `notes_vN.md` (rationale).

**Template**
- The analyzer writes all of the text.
- Slots that code fills per job: `{seed_trace}`, `{tools}`, `{system_prompt}`, `{obs_examples}`, `{generation_id}`. A template may use any subset, but it must include `{seed_trace}` and `{generation_id}`.

**Output format (fixed; the only rule the analyzer cannot change)**
- One JSON object: `{"messages": [...]}`.
- Roles: user / assistant / tool. Assistant turns may carry `tool_calls` `[{id, name, arguments}]`; tool turns carry `tool_call_id`.
- Validated against a JSON schema. A failure is rejected and logged, never silently fixed.

**Lint before use**
- Required slots present; no unknown slots.
- Output-format instruction present.
- No held-out seed text in the template.

## Loop (≤3 rounds)
1. The analyzer writes v1.
2. Pilot: about 20 jobs. Run schema validation, `rule_check`, the checklist judge (reasoning mode; any applicable miss means reject), and diversity.
3. A diagnosis agent reads the rejected and weak traces and writes notes. The analyzer then writes vN+1.
4. Every version, with its pilot report, is logged as a node in `runs.jsonl`.

## Independence
- The judge's checklist is frozen within a round, and the analyzer never scores its own prompt.
- Acceptance uses held-out seeds only.

## Acceptance (held-out seeds, matched count, cache off, same model)
**Compare against:**
- `simia_prompt=original`
- `simia_prompt=fixed`
- the previous analyzer version

**Metrics:**
- usable rate
- checklist pass rate
- distinct tool sequences and verdicts
- answer-leak rate: tool results that state the conclusion outright (judged)
- expert review on a sample

**Keep a version only if** it beats `fixed` on usable rate and leak rate without lower diversity.

## Config
- `prompt_source: simia | analyzer`
- `analyzer_seeds: 20`
- `analyzer_rounds: 3`
- `holdout_frac: 0.3`
- New stage: `simia-plus analyze`, run before `plan`.
