# Logs → Harbor: final plan

Status: implemented in `mint/`; this is the design record. Updated 2026-11-11.

## Goal

A user points the tool at a folder of agent logs in any format. They get back Harbor tasks with an honest status on each one, plus a report on what was used, what was excluded, why, and what was blocked. Nothing in the tool assumes a particular source, domain, modality or training objective.

## Core idea

The agent reads samples and writes code, and that code processes every record.

- No step requires an LLM pass over every record. The agent investigates exceptions and ambiguous instances.
- The agent's effort grows with how varied the corpus is and how many exceptions it has, not with its size.
- We ship the workflow (a skill), the CLI and the gates. Code specific to a source is agent-written and lives in the workspace, never in our package.
- Runs are unattended by default. The user reviews the report at the end. `--review` pauses after steps 3 and 4.

## Agent autonomy

The skill gives the agent a good default path, not a fixed script. The agent works iteratively the way it would on any coding task: it tries something, looks at the result and revises.

- **Free:** the agent can change step order, loop back at any point (a bad generator can mean the parser is wrong), rewrite its own parse, check and generator code as many times as needed, write extra throwaway scripts, use its own tools (shell, grep, Python, image viewing) instead of `mint`, and choose how deep to investigate.
- **Fixed (the only hard rules):** raw logs stay read-only; every exclusion and every task traces back to source records; nothing is made up; a task is marked `validated` only if the gates actually ran and passed; every decision is logged with its evidence.
- `mint` exists to make the repeated, error-prone parts reliable, not to limit the agent.

## Workspace

```
workspace/
  source_notes.md      what a record/trajectory is, boundaries, tools, outputs, open questions
  parse.py             agent-written
  checks.py            agent-written
  tasktypes/<type>.py  agent-written: spec + generator
  decisions.jsonl      every decision plus its evidence (audit trail)
  out/
    trajectories.jsonl
    findings.jsonl     per record: check id, result, evidence
    excluded.jsonl     record id, the task-blocking reason
    tasks/<type>/<id>/
    results.jsonl      per task: sources, generator version, gates run, status
  report.md
```

Raw logs are never modified or copied into good/bad folders. Everything references the originals.

## Script contracts

`mint run` calls agent-written code through these fixed signatures.

```python
# parse.py
def parse(paths: list[Path]) -> Iterator[Trajectory | ParseError]
# Trajectory: an ATIF document (agent, steps[source/message/tool_calls/observation]) + extra.mint {id, source_refs:[{file, offset|key}], ended, attachments:[...], meta:{}}
# Unknown values stay None. Unfamiliar fields are kept under meta.

# checks.py
CHECKS = [Check(id, version, fn)]
def fn(t: Trajectory) -> Finding  # {result: pass|fail|unknown, evidence:[refs], note}

# tasktypes/<type>.py
SPEC = {unit, env, job, task_scope (full_job|subtask|decision), check, gold, hidden, applies_when}
def applies(t: Trajectory, findings) -> bool
def group_key(t: Trajectory) -> str           # attempts sharing a key become one task
def build(group: list[Trajectory], out: Path) -> BuildResult | Blocked
# BuildResult: {task_id, files, gold_type, assumptions:[...]}
# Blocked: {reason, missing:[...]}
```

Example `results.jsonl` row:
```json
{"task_id":"t_0193","type":"decide_x","sources":["f3.jsonl#41","f3.jsonl#57"],
 "generator":"decide_x@3","gold":"imitation","gates":{"structure":"pass","leak":"pass","exec":"not_run"},
 "status":"generated"}
```

## Flow

### 1. Understand the source (agent)
- `mint sample` samples the **native files and records** by file type, size, field shape and position. It assumes nothing about trajectories yet.
- The agent works out what the requested work is, kept separate from what the agent in the log actually did. It also works out how sessions, calls, tools, sub-agents and attachments relate, and what is recorded versus missing.
- It writes `parse.py` and `source_notes.md`, including which samples it read.
- Records the number of raw events that landed in no trajectory, plus parse errors, so a parser that silently drops data shows up in the report.

### 2. Findings (agent proposes checks, code runs them)
- The agent proposes checks from samples: missing responses, malformed output, loops, unfinished runs, corruption, duplicates, and so on.
- `mint run checks` writes findings for every record. **Findings are observations, not exclusions.**
- Three separate questions are asked of each record: recording quality, execution outcome, and usefulness for building a task. Failed and partial runs stay available.
- Each check is spot-checked on a fresh sample before it is trusted. Exclusion rates per check go in the report.

### 3. Task types (agent)
- The initial grouping is by prompt or job and output shape. The groups are then refined by **evidence access and completion requirements**. One episode can produce several subtasks, and many attempts can collapse into one task.
- The spec for each type covers:
  - **unit:** whole episode or a specific subtask
  - **env:** static evidence files in v1. Types needing a live system are marked blocked.
  - **check:** exact match, an allowed set of answers, or the grading ladder (exact → numeric within tolerance → list/percent normalising → symbolic), all deterministic with no LLM in the reward
  - **gold:** `verified` (an independent verdict, with evidence that it matches this task's exact scope), `solved` (opt-in: a strong model solved the task fresh and reproduced the logged answer), `imitation` (the logged answer), or none (blocked)
  - **hidden:** answer keys, the original final response, later observations, solution files
  - **applies_when:** which records and findings this type can use
- **Exclusion happens here, not in step 2:** a record is excluded only if one of its findings prevents building or evaluating the chosen task. Otherwise the record is kept along with its findings.
- The agent builds a few varied instances first and fixes environment and verifier problems before generating in bulk.

### 4. Generate (code, plus the agent on exceptions)
- `mint run generate` sends each group to every generator whose `applies` matches. Repeated attempts are merged through `group_key` so they don't become duplicate tasks.
- Each task package contains `instruction.md`, `environment/`, `tests/test.sh`, `solution/solve.sh` and `task.toml`. Provenance goes in metadata, never inside the agent-visible environment.
- `Blocked` and exception results are queued for the agent to investigate. The agent does not resolve them by inventing data.

### 5. Validate (three levels)
1. **Every task:** files and config are valid, inputs resolve, the instance is internally consistent, and the **leak scan** passes. Gold answers live only in `tests/`, which Harbor runs after the agent finishes and never shows to the solver. The leak scan checks that no answer key, original final response or solution file is in `instruction.md` or `environment/`. Legitimate evidence stays, even when it contains words that also appear in the answer.
2. **A sample per type**, plus any instance whose generator changed: fresh environment, the oracle passes, a no-op fails, a meaningful wrong answer fails, and an alternative valid answer passes where exact matching could be too strict. An oracle that just writes out the expected answer only proves the verifier is wired up correctly. It says nothing about whether the answer can be derived from the inputs, so the agent also reviews whether the sampled answers are derivable.
3. **Batch audit:** the agent reads sampled tasks for fidelity, evidence sufficiency and verifier behavior. A failed audit sends the affected batch back to step 3.

**Optional `solved` pass** (`mint validate --solve --model <m>`): a strong model attempts each `imitation` task fresh in its sandbox. If it reproduces the logged answer, the task becomes `solved`; if not, it stays `imitation` and is flagged. This is the only step that needs the dev's own model API key, and it is off by default. Agreement is evidence, not proof, so `solved` stays a separate label from `verified`.

Status per task: `validated` (exec gates ran and passed on that instance), `generated` (structure and leak checks passed, and the type passed its sampled validation), `blocked` (with a reason). Settings: `sample_size` and `max_fix_attempts` are separate. Infrastructure errors are recorded as errors, not as task failures.

### 6. Report (the user's single touchpoint)
- Counts per stage (including events that landed in no trajectory), plus findings and exclusion rates per check
- Tasks per type, split into `validated` / `generated` and `verified` / `imitation`
- Gate results, and blocked types and candidates with what is missing
- Every decision with its evidence, and about 5 sample tasks per type, so a spot-check takes minutes

## CLI

The agent's toolbox. Each command exists because the agent would otherwise write throwaway scripts or guess.

| Command | Does |
|---|---|
| `mint init <logs> <ws> [--assistant claude\|codex\|copy]` | Creates the workspace, inventories files, and checks that Docker and Harbor are available |
| `mint status` | Shows where the run is: what's done, what's next, any blockers. The first command to run after an interruption |
| `mint sample [--n --unseen --matching <check>]` | Picks varied records (native files before parse). `--unseen` skips records already shown; `--matching` tests a check on fresh matches and non-matches |
| `mint show <id> [--field --attachments]` | Shows a whole record. Large fields are cut short with a visible marker, and attachments are saved as files the agent can open |
| `mint query "<expr>" [--count --group-by --grep]` | Counts, groups and searches across the whole corpus, to turn a hunch into a measured number |
| `mint new parser\|check\|tasktype <name>` | Creates a file with the contract already filled in |
| `mint run parse\|checks\|generate [--limit --dry]` | Runs the agent-written code over the corpus. Errors are grouped by cause with example ids, and output shows what changed since the last run |
| `mint task show <id>` | Shows exactly what the solving agent will see, to catch answer leaks and missing evidence |
| `mint task try <id> --answer <x>` | Runs the verifier locally in about a second, without Docker, for quick iteration |
| `mint validate [--type --sample-size --max-fix-attempts]` | Runs the three validation levels. Failures are grouped by cause; infrastructure errors are reported separately |
| `mint decide "<what>" --why "<evidence>"` | Adds an entry to decisions.jsonl, so the reasoning survives a context reset |
| `mint report` | Writes report.md |
| `mint schema` / `mint docs [--topic]` | Prints the contracts and config schemas, and the bundled guides |

Output rules:
- Output is short by default; `--json` for exact data, `--full` to opt in to everything.
- Nothing is ever cut silently: a cut-short field shows a marker with the command to see all of it.
- Every command ends with a `next:` hint.
- Every error says how to fix it.
- Seeded sampling and cached parsing, so the same inputs give the same outputs.
- Raw logs are read-only and every command is safe to rerun.

## How users use it: CLI + skill

The dev doesn't type the CLI. Their coding agent (Claude Code, Codex) runs the loop.

```
dev:    "turn ./logs into harbor tasks"
agent:  loads SKILL.md → runs mint commands, writes parse/checks/generators, fixes failures
agent:  "Done. 412 tasks (380 generated, 32 validated), 2 types blocked. See report.md"
dev:    steers by chatting ("why exclude those?", "loosen that verifier")
```

- **The skill** is the playbook: the workflow order, the judgment rules, and when to stop.
- **The CLI** is the agent's toolbox: deterministic, versioned, tested. It keeps runs repeatable instead of improvised.
- **A Python library** sits under the CLI, so it's tested and reusable in other pipelines.
- **Unattended runs** use the same thing: `claude -p "turn ./logs into harbor tasks"`. No agent runtime of our own.
- **MCP is not in v1.** The workflow needs a shell anyway (to write files, run Python and Docker), so MCP adds a server and solves nothing extra.
- **Packaging:** `pip install bandits` installs the CLI. The skill installs into the agent's skills folder (or comes in a Claude Code plugin).

This matches how agent tooling commonly ships today: an agent-friendly CLI plus a skill or instructions file, rather than a custom agent.

### Conventions (borrowed from an existing agent-driven CLI + skill product; platform parts dropped)

- **Built for agents:** `--json` prints one result on stdout, with progress on stderr. Commands never prompt. Exit codes are `0` ok, `1` error, `2` bad args/config, `130` interrupted. `--dry-run` works on anything that writes.
- **Self-describing:** `mint schema` prints the script contracts and config schemas, and `mint docs [--topic]` prints the bundled guides. The agent reads the installed version and never guesses.
- **The skill ships inside the package:** `pip install bandits` includes `SKILL.md`, `guides/` and `examples/`, so the skill always matches the CLI version. `npx skills add` is an optional second install path.
- **One-command start:** `mint init <logs> --assistant claude|codex|copy` opens the dev's coding agent with the skill and task loaded. `copy` prints the prompt for any other agent.
- **`examples/`:** 1–2 public log sets converted end to end, as references the agent can adapt.
- **Report shape:** running / verified / blocked / unverified, plus what to inspect next, using file paths.
- **Honest endings:** the parser records how each logged run actually ended (done, cut off, error, unknown). A clean exit is never taken as success.
- **Short skill:** about 100 lines of principles. Details live in `guides/`, which the agent reads only when needed.

**No platform.** There are no accounts, orgs, API keys, links, cost confirmations or remote submissions. `init` only checks the local machine: the logs path exists and is readable, and Docker and Harbor are available. It stops with a fix message otherwise.

## Harbor output (matches Harbor conventions)

- **Generators follow Harbor's benchmark-converter pattern** (`adapters/` in the Harbor repo: a template folder plus a fill-in script). The skill points the agent at Harbor's own `create-task` / `create-adapter` skills and `harbor task init` instead of re-explaining Harbor.
- **Per task:** `task.toml`, `instruction.md`, `environment/`, `tests/test.sh` (writes 0–1 to `/logs/verifier/reward.txt`), `solution/solve.sh`.
- **The verifier runs separately** (`[verifier] environment_mode = "separate"`) where supported. The solver's answer file or directory must be declared as an artifact (e.g. `artifacts = [{ source = "/app/answer" }]` in `task.toml`) so the verifier receives it.
- **A canary GUID in every task**, so leaks into training corpora can be detected.
- **Suite files:** `registry.json` plus `manifest.parquet` (one row per task: type, gold, status, difficulty, sources), and lineage-safe splits so related attempts never cross train and test.
- **Precedent:** an existing open dataset (SmolDataEnvs) built 5.5K Harbor tasks from real agent notebooks, with deterministic grading ladders and gold answers checked by re-solving. That's the same shape as this plan.

**Not required for the first working conversion:** symbolic grading, suite publishing (`registry.json`/`manifest.parquet`), canaries and `--assistant` launching. Add each when needed. The first goal is real logs converted to tasks that run and pass the gates.

**Standalone:** this does not depend on any existing Bandits code. Bandits is being rebuilt, and where this lives will be decided then.

## Testing

The plan is proven by converting real logs and running the tasks, not by more planning.

**Unit tests (in repo, synthetic fixtures only):**
- Script contracts: parse, check and generator results round-trip, and errors are grouped correctly.
- Leak scan: catches an answer placed in `instruction.md` or `environment/`, and does not flag legitimate evidence.
- Gates on a tiny fixture task: the oracle passes, a no-op fails, a wrong answer fails, and an infrastructure error is recorded as an error.
- Status: a task is never marked `validated` without its execution gates having run.
- CLI: `--json` shape, exit codes, `--dry-run` writes nothing, and reruns are idempotent.

**End-to-end pilots (each run separately, nothing changed in the harness between sources):**
1. **Public source:** a public set of agent trajectories, run end to end. The result can be committed as `examples/`.
2. **Private source A:** a large, messy production log export (multi-agent sessions with attachments). Data and results stay outside this repo.
3. **Private source B:** a structurally different private source, run once with nothing tuned, to test that the method generalizes.

**What a pilot must show:**
- At least one task type end to end: real logs → tasks that run in Harbor and pass the oracle, no-op and wrong-answer gates.
- Every sampled record has a recorded outcome (used, excluded with a reason, or blocked).
- The report shows counts per stage, exclusion rates per check, gold and status splits, and blocked items with what is missing.
- A manual audit of about 20 generated tasks per type for fidelity, leaks and solvability, recording what the gates missed.
- An interrupted run resumes from `mint status`.
- Reported numbers: task yield, gate pass rate, agent time and tokens, and the number of human interventions.

## Rules

- Every exclusion and every task traces back to source records.
- No timestamps, labels, boundaries, pairings or success values are made up. Unknown is a valid value.
- Logged answers, logged success, confidence and agreement between models are not ground truth. The gold tag states which kind of answer a task has.
- Earlier dataset-specific filters (e.g. ones built for one training run) are not inherited. Every exclusion is reassessed for the task being built.
- The verifier checks the outcome, not a replay of the logged steps.
- Instructions describe the task and never include the recorded answer.

## v1 scope

- **In:** static evidence tasks, episode and subtask units, exact or allowed-set checks, any source format or modality the parser can read.
- **Out:** rebuilt live environments, rubric or LLM-judge verifiers, training-data generation.
