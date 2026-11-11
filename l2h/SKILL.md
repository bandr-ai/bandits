---
name: logs-to-harbor
description: Turn a folder of agent logs (any format) into Harbor tasks with honest validation status, using the `l2h` CLI. Use when asked to convert agent logs/traces/trajectories into Harbor tasks or evals.
---

# Logs to Harbor

You read samples, write code, and the code processes every record. No step needs an
LLM pass over every record. Your effort should track how varied the corpus is, not how big.
Nothing assumes a source, domain, modality or objective.

Before guessing a command or contract: `l2h schema`, `l2h docs --topic <name>`, `--help`.
Output is short by default; `--json` for exact data, `--full` for everything.
Every command ends with a `next:` hint. After any interruption, run `l2h status`.

## Hard rules (the only fixed ones)

1. Raw logs are read-only. Never copy them into good/bad folders. Reference them.
2. Every exclusion and every task traces back to source records (`file#offset`).
3. Make nothing up: no timestamps, labels, boundaries, pairings or success values. Unknown is valid.
4. A task is `validated` only if the execution gates passed in a real Harbor run; host-local gate passes (the default fast runner) stay `generated`. `l2h` enforces it; never edit results.jsonl by hand.
5. Log every decision with its evidence: `l2h decide "<what>" --why "<evidence>"`.

Everything else is yours: reorder steps, loop back (a bad generator may mean a wrong parser),
rewrite your code as often as needed, use your own tools (shell, grep, Python) instead of `l2h`.

## Default path

1. **Understand the source.** `l2h init <logs> <workspace>`, then `l2h sample`, `l2h sample --unseen`,
   `l2h show <id>`, `l2h query '<expr>' --count`. Work out the *requested work* separately from
   what the logged agent *did*; how sessions, calls, tools, sub-agents, attachments relate; what is
   recorded vs missing. Write `source_notes.md` (include which samples you read) and `parse.py`
   (`l2h new parser`). Record how each run actually ended (done / cut_off / error / unknown). A clean
   exit is never success. `l2h run parse`; investigate dropped records and parse-error groups.
   Guide: `l2h docs --topic parsing`.
2. **Findings.** `l2h new check <name>`, `l2h run checks`. Findings are observations, not exclusions.
   Ask three separate questions: recording quality, execution outcome, usefulness for a task.
   Failed and partial runs stay usable. Spot-check each check on fresh data:
   `l2h sample --matching <check>`, then `l2h decide --kind spot_check --check <id> --verdict pass --why ...`.
   Guide: `l2h docs --topic checks`.
3. **Task types.** `l2h new tasktype <name>`. Group by prompt/job and output shape, then refine by
   evidence access and completion requirements (one episode can yield several subtasks; many
   attempts can collapse into one task via `group_key`). Exclude a record only if a finding blocks
   building or evaluating *this* task. Build a few varied instances first and fix them before bulk.
   Guide: `l2h docs --topic tasktypes`.
4. **Generate.** `l2h run generate --limit 5`, then without the limit. `Blocked` and exception groups
   land in `out/queue.jsonl`: investigate them, never resolve them by inventing data.
   Look at what the solver gets: `l2h task show <id>`. Try answers fast: `l2h task try <id> --answer x`.
5. **Validate.** `l2h validate` (level 1 on every task: structure + leak scan; level 2 on a seeded
   sample per type: oracle passes, no-op fails, wrong answer fails, valid alternatives pass).
   An oracle that writes the expected answer proves wiring only, so *you* also check the answer is
   derivable from the inputs. Level 3 (`--level 3`) is you reading sampled tasks for fidelity,
   evidence sufficiency and verifier behavior; a failed audit sends the batch back to step 3.
   Infrastructure errors are not task failures. Stop after `max_fix_attempts` and report it.
   Guide: `l2h docs --topic validation`.
6. **Report.** `l2h report`, read `report.md`, then summarise for the user: counts, what was
   excluded and why, validated vs generated, verified vs imitation, what is blocked and what is missing.

## Judgment

- Gold: `verified` only with an independent verdict matching this task's exact scope; `imitation` = the
  logged answer; logged success, confidence and model agreement are not ground truth. Label honestly.
- Instructions describe the task and never include the recorded answer. Gold lives only in `tests/`.
- The verifier checks the outcome, not a replay of the logged steps. Deterministic only, no LLM.
- v1 is static evidence. A type needing a live system is `Blocked` with what is missing.
- Earlier dataset-specific filters are not inherited; reassess every exclusion for the task at hand.
- Stop and ask the user only for decisions that are theirs (scope, what counts as success), not for routine fixes.

## Harbor

Don't re-explain Harbor: use its `create-task` / `create-adapter` skills and `harbor task init`.
Generated tasks follow its conventions (`l2h docs --topic harbor`).

## Command cheat-sheet

| Need | Command |
|---|---|
| Start / resume | `l2h init <logs> <ws>` / `l2h status` |
| Look at the corpus | `l2h sample [--unseen] [--matching <check>]`, `l2h show <id> [--field p] [--attachments]` |
| Measure a hunch | `l2h query '<expr over r>' [--count] [--group-by p] [--grep re]` |
| Scaffold code | `l2h new parser`, `l2h new check <n>`, `l2h new tasktype <n>` |
| Run it on everything | `l2h run parse`, `l2h run checks`, `l2h run generate [--limit N] [--type T]` |
| See what the solver sees | `l2h task show <id>`; fast verifier loop: `l2h task try <id> --answer x` |
| Gates | `l2h validate [--type T] [--task ID] [--level 1,2,3]` |
| Remember / report | `l2h decide "<what>" --why "<evidence>"`, `l2h report` |
| Contracts / guides | `l2h schema`, `l2h docs [--topic parsing\|checks\|tasktypes\|validation\|harbor]` |

Flags on every command: `--json` (one result on stdout, progress on stderr), `--full`, `--dry-run`,
`--workspace`. Exit codes: 0 ok, 1 error or failed validation, 2 bad args/config, 130 interrupted.

## Ending a run

Tell the user, from `report.md` and with file paths: how many tasks, how many `validated` vs
`generated`, `verified` vs `imitation`, what was excluded and why (rates per check), which types are
blocked and what is missing, and which numbers you could not verify. Unattended runs stop at the report.
