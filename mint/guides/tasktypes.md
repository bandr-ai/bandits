# Task types

Every type rebuilds one job end to end: `SPEC["job"]` is its id from `jobs.json`. Mint builds whole
jobs only (no subtasks, no decision-from-evidence tasks).

A type file has `SPEC`, `VERSION`, `applies`, `group_key`, `build` (`mint new tasktype <name>`).

- `applies(t, findings)` decides if a record can feed this type. Return `(False, "reason")` to leave a
  recorded exclusion reason. Records no type applies to land in `out/excluded.jsonl`.
- `group_key(t)`: attempts sharing a key become one task. Merge repeated attempts here.
- A task is the original request, the starting state, the tools/action space and the output,
  graded on the real outcome. `mint report` has one row per job.
- `build(group, out)` returns one of:
  - `BuildResult`: a package. Either `answer=Answer(...)` (mint writes an answer-file verifier) or your
    own `files["tests/test.sh"]` that checks the end state in the task environment (no `answer`;
    must write `/logs/verifier/reward.txt`). Your own environment goes in
    `files["environment/Dockerfile"]` (+ context files) with `env_source` `harness`, `rebuilt` or
    `partial` (default `evidence` = mint's image plus `evidence` files). Your own oracle goes in
    `files["solution/solve.sh"]` with `oracle_kind` `replay` / `fresh` / `reference`; `oracle_kind="none"`
    when there is no credible one. Record `grading_method`, `grading_reason` and
    `gaps=[{"what", "affects": execution|fidelity|grading|none}]`.
  - `Candidate(reason, gaps=[...], next_action="...")`: the job is identified but no runnable package is
    possible yet. Kept in the queue and the report; never counted as a task.
  - `Blocked(reason, missing, kind=...)`: only `kind` `secret`, `leak` or `no_instruction`.
- Starting state = the world at the agent's first action in this task. `mint evidence <id>` lists what
  each run read (complete only when proven), wrote, ran and fetched, with step refs.
  - Environment: files named in the request or read before anything could have created them
    (`pre_existing`), OS/runtime shown by version output (smallest matching base), and setup done
    before the agent started (harness, image).
  - Not environment: whatever the agent installed, configured, downloaded, built or wrote. That is
    the job: it goes in the solution (`replay`), never the Dockerfile.
  - `command not found` / a missing module: the agent didn't have it; don't pre-install it.
  - Unknown → a gap (`affects: fidelity`), not a convenient guess.
- Wrong answers / alternatives: strings for answer-file tasks; `Outcome(script="...")` for any task
  (a bash script run in the environment instead of solve.sh). Wrong ones must be *meaningful* (a
  plausible mistake, not noise); give alternatives when the check might be too strict. `hidden` lists
  strings that must never reach the solver: the original final response, answer keys.
- Evidence goes in `evidence={"path": text_or_bytes}` and appears at `/app/path`. Use bytes for
  binary evidence such as images (`{"img/chart.png": png_bytes}`); the leak and secret scans skip
  binary files, so a secret or answer inside an image is not caught: review those by eye.
  The solver writes its answer to `/app/answer/<answer.file>`.
- Gold (`gold_type`) = why the grading is trusted: `verified` (independent label for this exact scope),
  `computed` (recomputed from the inputs), `independent` (end-state/behaviour check not tied to the
  logged answer), `solved` (a fresh solve passed), `imitation` (the logged answer), `none`. `ready` in the
  report needs validated, a non-partial environment, trust other than imitation/none, and no open gaps.
- Bump `VERSION` when the generator changes so changed instances are revalidated.
- Idempotence: unchanged packages keep their validation state across `mint run generate` reruns.
