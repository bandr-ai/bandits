---
name: mint
description: Turn a folder of agent logs (any format) into Harbor tasks with honest validation status, using the `mint` CLI. Use when asked to convert agent logs/traces/trajectories into Harbor tasks or evals.
---

# Mint

Logs (plus any supplied harness, environment assets and data) in, whole-job Harbor tasks out.
Spend your effort building tasks. You read samples and write code; the code processes every record.
Nothing assumes a source, domain, modality or objective.

Contracts and help: `mint schema`, `mint docs --topic <name>`, `--help`. `--json` for exact data.
Every command ends with a `next:` hint. After any interruption, run `mint status`.

## Hard rules

1. Raw logs are read-only. Reference them (`file#offset`); every task and exclusion traces back to them.
2. Make nothing up: no invented files, labels, success values, timestamps or customer facts. Unknown
   is valid; missing pieces are listed, not faked.
3. `validated` means the execution gates passed in a real Harbor run (`mint validate --runner harbor`).
   Host-local passes stay `generated`. Never edit results by hand.

Everything else is yours: reorder, loop back, use your own tools instead of `mint`.

## Workflow

1. **Read and group.** `mint init <logs> <ws>` (`--allow-missing-runtime` without Docker), `mint sample`,
   `mint show <id>`, `mint query`. Write `parse.py` (`mint new parser`, guide: `parsing`), `mint run parse`.
   Map the jobs (`mint new jobs` → `jobs.json`): each job the logged agents were given, end to end, its
   repeat attempts, and how success is known.
2. **Build.** One task type per job shape (`mint new tasktype`, guide: `tasktypes`); each task is one
   whole job, repeat attempts collapse via `group_key`. Reuse supplied tools, data, environments and
   simulators. `mint run generate --limit 5`, fix, then all.
3. **Check and fix.** `mint validate` (guide: `validation`): structure, leak/secret scan, and in Harbor:
   oracle passes, no-op and wrong outcomes fail, valid alternatives pass. Fix demonstrated failures; stop
   after `max_fix_attempts`. Infra errors are not task failures.
4. **Report.** `mint report`, then tell the user per job: built or not, checks passed/failed/not run,
   what is missing, and how the task differs from the original job.

Checks (`mint new check`), decisions (`mint decide`) and notes are optional tools; use them when they
add information, not as paperwork.

## Judgment

- **Whole jobs.** A task asks a new agent to do what the logged agent was assigned: original request,
  starting state, tools, output, graded on the real outcome. No extracted sub-steps.
- **Starting state** = the world at the agent's first action (`mint evidence <id>`). What the agent
  installed, built or wrote is the job, not the environment. A supplied Dockerfile/harness wins.
- **Build every job you can.** Missing data, DB or runtime → build a scaffold with an explicit connection
  point (seed path, endpoint, env var) and list it in `gaps`; `env_source="partial"`. Only when not even
  a useful scaffold is possible, return `Candidate(reason, gaps, next_action)`. `Blocked` is only for an
  unremovable secret/leak or no recoverable instruction. A scaffold is not a validated task.
- **Interaction.** If the job needs further user replies: no simulator when none are needed; reuse a
  supplied simulator when compatible; else scripted replies for a bounded exchange, or a model-backed
  customer for open-ended conversation. The customer's goal, private facts, preferences and consent
  conditions live in its spec; the solver gets the opening request and learns the rest by asking. Never
  hand the solver future consent upfront unless an offline adaptation is explicitly wanted; then say so
  in the report. Missing simulator/credentials → connection point + gap, not silent offline conversion.
- **Grading.** Check the outcome the request asks for, under the policy that applies; accept valid
  alternatives. Enforce only requirements the original job supports (no added authentication,
  minimality, naming or exact-state rules without support). Check action history only where the
  requirement depends on it (e.g. consent before a mutation). Money: reject NaN/inf, compare exact
  units. Don't require extra report fields when the tool results already are the outcome. Prefer
  deterministic grading; if the original grades with a model, say so. Grader-only tools go in `tests/`.
  `gold_type` names what supports the expected outcome (outside label `verified`, recomputation
  `computed`, end-state check `independent`, fresh solve `solved`, logged answer `imitation`, `none`).
- **Oracle.** Prefer `replay` of a logged attempt that passes in the built environment, then
  `reference`; `answer` proves wiring only. Wrong outcomes are plausible real mistakes; no-op must fail.
  When the target is "no change" (a correct refusal), the verifier also checks the response or
  escalation, so doing nothing still fails.
- Instructions never contain the recorded answer; gold lives only in `tests/`.

## Harbor

Use Harbor's `create-task` / `create-adapter` skills; tasks follow its conventions (`mint docs --topic harbor`).

## Commands

| Need | Command |
|---|---|
| Start / resume | `mint init <logs> <ws>` / `mint status` |
| Look | `mint sample [--unseen] [--matching c]`, `mint show <id>`, `mint query '<expr>' [--count]`, `mint evidence <id>` |
| Scaffold | `mint new parser\|jobs\|check <n>\|tasktype <n>` |
| Run | `mint run parse\|checks\|generate [--limit N] [--type T]` |
| Task loop | `mint task show <id>`, `mint task try <id> --answer x` |
| Gates | `mint validate [--type T] [--task ID] [-j N]` |
| Report | `mint report` |

Exit codes: 0 ok, 1 error or failed validation, 2 bad args, 130 interrupted.
