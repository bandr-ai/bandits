# Changelog

All notable changes to `bandits-cli` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
change the command line or the artifact format; such changes are called out.

## [Unreleased]

- Mint: package hash sorts outcome keys, so regenerating an unchanged task keeps
  its validation.
- Rename the logs-to-tasks tool from `l2h` to **Mint**: command/import `mint`,
  bundled skill `mint`, workspace `mint.json`, `MINT_*` environment variables,
  and Mint task/trajectory metadata. It remains part of `bandits-cli`.
  Existing experimental workspaces and parser scripts must be regenerated or
  migrated to these names; the old command/import are removed.
- `mint` verifier honesty: `BuildResult.verifier_timeout_sec` (own `[verifier]`
  budget), `oracle_source` and `Outcome.source` provenance shown in `mint report`,
  `answer` oracles are `ready` only with verified/computed trust, execution errors
  reported apart from grading failures; guidance on verifier-only dependencies,
  outcome-based graders and the `verified` label.
- `mint evidence`: starting-state facts keep their command, step, tool and
  `before_agent_setup`. System facts only come from version/OS query commands;
  absence is `not_on_path` / `module_missing`; `Unable to locate package` is
  `package_lookup_failed`. The "smallest matching base image" rule is gone. A job
  whose whole assignment is a decision is a normal answer-file job.
- `mint validate` runs gate runs in parallel (`-j/--jobs`, config `validate_jobs`, default 4)
  and saves each task's result as soon as it finishes; a rerun skips tasks whose package and
  runner evidence are current. Per-task progress on stderr and wall time in the summary.


### Added

- `mint`: a standalone logs-to-Harbor tool (library plus CLI, no dependency on
  `bandits`). It samples and queries agent logs, runs agent-written parse,
  check and task-type code over the whole corpus, writes Harbor task packages,
  scans them for answer leaks, and validates them with oracle, no-op and
  wrong-answer gates. Ships its own skill and guides (`mint docs`). Plan:
  `docs/plans/mint.md`.
  Review round: `validate` no longer reports `passed` or refunds the fix budget
  while rows stay blocked, resets structure/leak/secret evidence when a package
  changes, and records an exhausted type instead of aborting; attachment paths
  and `--grep`/`--field` errors are contained. New level-1 secret scan
  (`secret_allow`), label-vocabulary-aware leak scan, binary evidence, warnings
  (giveaway, missing alternatives, dangling `/app` path), `negative` gold kind
  (`forbidden` check), status/report show gold kind and flag unaudited
  `validated` tasks, and `mint sample --stratify <expr> --n N`.
  Follow-up: `Answer.choices` declares a label vocabulary (leak scan, giveaway and
  alternatives warnings use it; files that define the labels are not leaks), negative
  tasks need `pattern` or `choices`, and the secret scan ignores code expressions.
  ATIF: a parsed trajectory is now an ATIF (Agent Trajectory Interchange Format,
  ATIF-v1.8) document plus an `extra["mint"]` block (`id`, `source_refs`, `ended`, `meta`,
  `attachments`), built with `Trajectory.from_atif`. It is validated against Harbor's own ATIF
  model when `harbor` is importable (violations are parse errors grouped by cause); logs that are
  already ATIF need a few lines. New accessors `t.calls()`, `t.final_message`, `t.subagents`;
  `t["meta"]`, `r.meta` etc. keep working, but `events` is gone (use `steps`; breaking for
  existing `parse.py`). Task types must set `SPEC["task_scope"]` (`full_job` | `decision`), shown
  per type in `mint report`; the skill gains a rule to prefer rebuilding the agent's whole job and to
  label decision-from-evidence tasks as such.

  Job map: step 1 writes `jobs.json` (full jobs, sub-capabilities, repeats, outcomes); every task
  type names its job (`SPEC["job"]`), `task_scope` gains `subtask`, and the report lists jobs with
  nothing built for them. Pre-ATIF parsers get a migration error.
- `mint` full jobs first: a task type can grade the end state with its own
  `tests/test.sh` (no answer file needed; the verifier runs in the task
  environment), supply its own environment and oracle, and record provenance
  (`env_source`, `oracle_kind`, `grading_method`/`grading_reason`, `gaps`, in
  results and `task.toml`). Wrong answers and alternatives can be executable
  `Outcome` scripts. `oracle_kind="none"` packages run every other gate and stay
  `generated`. `Candidate(reason, gaps, next_action)` records a job that can't be
  packaged yet; a `full_job` may be `Blocked` only for a secret, a leak or no
  instruction. New `exec_failed` status (malformed or failed gates); `blocked` now
  means leak/secret. The report leads with full jobs and a computed `ready` count.
  New `mint evidence <id>` lists what a run read, wrote, ran and fetched. The skill
  no longer limits tasks to static evidence.

  Review round 2: the secret scan catches env-style names (`DB_PASSWORD=...`,
  `AWS_SECRET_ACCESS_KEY=...`) and "enter <value> in the password field" prose; the leak scan treats
  `hint`/`result`/`sum`/`total`/`it's` as answer labels, warns on a bare short hidden value in
  `instruction.md`, and the giveaway warning also checks `instruction.md`. Tool calls and results
  without ids pair by position, and `mint run parse` warns about calls with no matching result
  (parsers should synthesise `call_1`, ... when the logs lack ids). `"meta" in t` matches `t["meta"]`.
  "changed since last run" now counts content and generator-version changes, not only counts;
  `run parse` suggests the same next step as `mint status`. The task type scaffold leaves
  `task_scope` as `TODO` (choose `full_job` | `subtask` | `decision`); only the `jobs.json`
  scaffold's own placeholders are rejected; `solve.sh` comments name the gold kind.

## [0.3.0] - unreleased

`bandits` is now ingest only. Everything else was removed, to be rebuilt from
scratch on top of ingest.

### Kept

- `bandits.ingest` (every reader, detection, discovery, health, mappings),
  `bandits.traces`, `bandits.redact` and `bandits.store`.

### Removed (breaking)

- **The `bandits` command line**, with all of its commands. Use the Python API
  (`bandits.ingest.load_corpus`, `bandits.store.ArtifactStore`) instead.
- **`bandits.verify`**: the turn judge (`judge-turns`), check proposal
  (`propose-verifier`, `review-checks`) and scoring (`score-traces`).
- **`bandits.export`**: SFT exports (`build-sft`, `export-nextstate`).
- **`bandits.analyze`**: task candidates, task families and RLM mining
  (`analyze`, `families`, `mine-rlm`, `audit-rlm`, `materialize-rlm-taskset`,
  `rlm-session`, `rlm-families`).
- **`bandits.emulate`**: AWM / trace-grounded world simulation.
- `bandits.inspect` (the HTML trace page), `bandits.fields`, `bandits.ledger`,
  `bandits.providers` and `bandits.transport`.
- The `llm`, `audit` and `emulate` extras, and the `typer` and `rich`
  dependencies.
- The scripts and docs for the removed code (AWM, RLM, the tau2 and TRAIL judge
  evals, the paper figures).

### Changed

- `bandits.genai` and `bandits.jsonarray` moved to `bandits.ingest.genai` and
  `bandits.ingest.jsonarray`.
- `recipes/` is left as it was and is no longer tested in CI. It imports
  modules removed here.

## [0.2.0] - unreleased

The first release on PyPI.

### Install

- Published as `bandits-cli` (the PyPI name `bandits` belongs to another
  project). The command and the import are still `bandits`:
  `pipx install bandits-cli`.
- Extras: `llm` (model calls through LiteLLM), `audit` (RLM mining, check
  proposal and audit), `emulate` (trace-grounded world simulation).

### Added

- `bandits --version`.
- Ingest for OpenTelemetry (standard OTLP/JSON and flat spans), chat JSON and
  JSONL exports, Claude Code sessions, Langfuse, LangSmith, Phoenix, TRAIL and
  FailproofAI, with deterministic redaction and content-addressed storage.
- `list`, `show`, `inspect`, `fields` and `get` to browse what was ingested.
- `analyze`, `mine-rlm`, `audit-rlm` and `materialize-rlm-taskset` to find
  task families in real user requests.
- The next-state verifier: `judge-turns`, `propose-verifier`,
  `review-checks`, `score-traces` and `export-nextstate`.
- `build-sft` for LLM-reviewed SFT rows straight from traces.

### Changed

- `bandits list` prints full artifact ids, which other commands accept,
  instead of shortened ones they did not. The `derivation` column appears only
  when a corpus has one, so the table fits an 80-column terminal.

[Unreleased]: https://github.com/bandr-ai/bandits/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bandr-ai/bandits/releases/tag/v0.2.0
