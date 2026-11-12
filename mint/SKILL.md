---
name: mint
description: Turn a folder of agent logs (any format) into Harbor tasks with honest validation status, using the `mint` CLI. Use when asked to convert agent logs/traces/trajectories into Harbor tasks or evals.
---

# Mint

You read samples, write code, and the code processes every record. No step needs an
LLM pass over every record. Your effort should track how varied the corpus is, not how big.
Nothing assumes a source, domain, modality or objective.

Before guessing a command or contract: `mint schema`, `mint docs --topic <name>`, `--help`.
Output is short by default; `--json` for exact data, `--full` for everything.
Every command ends with a `next:` hint. After any interruption, run `mint status`.

## Hard rules (the only fixed ones)

1. Raw logs are read-only. Never copy them into good/bad folders. Reference them.
2. Every exclusion and every task traces back to source records (`file#offset`).
3. Make nothing up: no timestamps, labels, boundaries, pairings or success values. Unknown is valid.
4. A task is `validated` only if the execution gates passed in a real Harbor run; host-local gate passes (the default fast runner) stay `generated`. Run `mint validate` with Harbor and Docker installed (`--runner harbor` to insist) to reach `validated`. `mint` enforces it; never edit results.jsonl by hand.
5. Log every decision with its evidence: `mint decide "<what>" --why "<evidence>"`.

Everything else is yours: reorder steps, loop back (a bad generator may mean a wrong parser),
rewrite your code as often as needed, use your own tools (shell, grep, Python) instead of `mint`.

## Default path

1. **Understand the source.** `mint init <logs> <workspace>` (add `--allow-missing-runtime` on a
   machine without Harbor/Docker: generation and host-local checks still work, nothing reaches
   `validated`), then `mint sample`, `mint sample --unseen`,
   `mint show <id>`, `mint query '<expr>' --count`. Work out the *requested work* separately from
   what the logged agent *did*; how sessions, calls, tools, sub-agents, attachments relate; what is
   recorded vs missing. Write `source_notes.md` (include which samples you read) and `parse.py`
   (`mint new parser`): each trajectory is an ATIF document (Harbor's trajectory format: steps with
   source/message/tool_calls/observation; logs that already are ATIF just get wrapped) plus mint's id,
   source_refs, ended and meta. Record how each run actually ended (done / cut_off / error / unknown). A clean
   exit is never success. `mint run parse`; investigate dropped records and parse-error groups.
   Guide: `mint docs --topic parsing`.
   Then **map the jobs** (`mint new jobs` → `jobs.json`): every job the logged agents were given,
   end to end (not their internal steps or sub-agents), how repeat attempts at the same instance
   show up, and how success or failure is known. Task types are built *from* this map (each names
   its job); jobs you don't build stay listed in the report.
2. **Findings.** `mint new check <name>`, `mint run checks`. Findings are observations, not exclusions.
   Ask three separate questions: recording quality, execution outcome, usefulness for a task.
   Failed and partial runs stay usable. Spot-check each check on fresh data:
   `mint sample --matching <check>`, then `mint decide "<what>" --kind spot_check --check <id> --verdict pass --why "..."`.
   Guide: `mint docs --topic checks`.
3. **Task types.** `mint new tasktype <name>`, one per job shape. Each task is one whole job: many
   attempts at the same instance collapse into one task via `group_key` (keep them all as evidence). Exclude a record only if a finding blocks
   building or evaluating *this* task. Build a few varied instances first and fix them before bulk.
   Guide: `mint docs --topic tasktypes`.
4. **Generate.** `mint run generate --limit 5`, then without the limit. Candidates, blocked and exception
   groups land in `out/queue.jsonl`: investigate them, never resolve them by inventing data.
   Look at what the solver gets: `mint task show <id>`. Try answers fast: `mint task try <id> --answer x`.
5. **Validate.** `mint validate` (level 1 on every task: structure + leak scan; level 2 on a seeded
   sample per type: oracle passes, no-op fails, wrong answers/outcomes fail, valid alternatives pass;
   without an oracle the rest still runs and the task stays `generated`).
   An `answer` oracle proves wiring only: *you* also check the answer is derivable from the inputs.
   Level 3 (`--level 3`) is you reading sampled tasks for fidelity, evidence sufficiency and verifier
   behavior; a failed audit goes back to step 3. Infra errors are not task failures. Stop after
   `max_fix_attempts` and report it.
   Guide: `mint docs --topic validation`.
6. **Report.** `mint report`, read `report.md`, then summarise for the user, per job: its task or
   candidate, environment source, grading method and trust, oracle, gaps, and whether it is ready;
   then exclusions and why, validated vs generated, what is missing.

## Judgment

- **Whole jobs only.** Each task asks a new agent to do what the logged agent was assigned, end to
  end: the original request, its starting state, its tools/action space, its output, graded on the
  real outcome. No subtasks or decision-from-evidence tasks. Logs only, or logs + an attached harness
  (use it, fill what it lacks).
- **Starting state = the world at the agent's first action in this task** (`mint evidence <id>`):
  - environment: files named in the request or read before anything in the run could have made them
    (content only when the read was complete), the base system shown by OS/version output, and setup
    done before the agent started (by the harness or image);
  - never environment: anything the agent installed, configured, downloaded, built or wrote. That is
    the job (it belongs in the solution), even when it makes the environment easier to rebuild;
  - absence is evidence: `command not found` / a missing module means the agent didn't have it, so
    don't pre-install it;
  - version output narrows the base image, rarely pins it: take the smallest base that matches;
  - anything still unknown is a gap (`affects: fidelity`), never a convenient guess.
- **Never block a job for being hard.** If a piece is missing, build it with the gap recorded
  (`env_source="partial"`, `gaps=[...]`, `oracle_kind="none"`), or return
  `Candidate(reason, gaps, next_action)` when no runnable package is possible yet. `Blocked` is only
  for an unremovable secret/leak or no recoverable instruction. Every job ends with at least a
  candidate; `mint report` flags any that don't.
- **Never fake.** Missing data stays a listed gap: no invented files, no solve.sh or verifier made up
  to satisfy packaging. Mark a read complete only when the log proves it.
- **Grading.** `grading_method` fits the job (own `tests/test.sh` on the end state, recomputation, or
  exact/allowed-set). `gold_type` = why trusted: `verified` needs an outside label for this job (your
  own check is `computed`), `independent`, `solved`, `imitation` (logged answer, last resort), `none`.
  Check the outcome the request asks for, never a printed marker; accept valid alternatives (layout,
  naming, approach). Verifier-only tools go in `tests/test.sh` after the solve, never `environment/`;
  `verifier_timeout_sec` is the verifier's own sufficient budget.
- **Oracle (`oracle_kind`).** Prefer `replay` of a logged attempt (kept only if it passes in the built
  environment), then `fresh` (ask first), then `reference`; `answer` proves wiring only; else `none`.
  Say where it came from in `oracle_source`. **Wrong outcomes are real mistakes:** `Outcome(script=...,
  source=...)` doing plausible bad work (a broken build, a stub printing the expected text); alternatives differ.
- Instructions never include the recorded answer; gold lives only in `tests/`. The verifier checks the outcome, not a replay of the logged steps. Deterministic only, no LLM.
- Reassess every exclusion for this task (no inherited filters); ask the user only for their decisions.

## Harbor

Use Harbor's `create-task` / `create-adapter` skills; tasks follow its conventions (`mint docs --topic harbor`).

## Command cheat-sheet

| Need | Command |
|---|---|
| Start / resume | `mint init <logs> <ws> [--allow-missing-runtime]` / `mint status` |
| What a run read/wrote/ran | `mint evidence <id>` (environment rebuild input) |
| Look at the corpus | `mint sample [--unseen] [--matching <check>] [--stratify <expr over r> --n N]`, `mint show <id> [--field p] [--attachments]` |
| Measure a hunch | `mint query '<expr over r>' [--count] [--group-by p] [--grep re]` |
| Scaffold code | `mint new parser`, `mint new jobs`, `mint new check <n>`, `mint new tasktype <n>` |
| Run it on everything | `mint run parse`, `mint run checks`, `mint run generate [--limit N] [--type T]` |
| See what the solver sees | `mint task show <id>`; fast verifier loop: `mint task try <id> --answer x` |
| Gates | `mint validate [--type T] [--task ID] [--level 1,2,3]` |
| Remember / report | `mint decide "<what>" --why "<evidence>"`, `mint report` |
| Contracts / guides | `mint schema`, `mint docs [--topic parsing\|checks\|tasktypes\|validation\|harbor]` |

Every command takes `--json` (result on stdout, progress on stderr), `--full`, `--dry-run`, `--workspace`.
Exit codes: 0 ok, 1 error or failed validation, 2 bad args/config, 130 interrupted.

## Ending a run

Tell the user, from `report.md` with file paths: each job and how far it got (task or candidate, ready or
not and why), tasks `validated` vs `generated` vs `exec_failed`, grading trust, exclusions and why,
what is missing, and which numbers you could not verify. Unattended runs stop at the report.
