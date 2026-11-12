# Task types

Every type names the job it rebuilds: `SPEC["job"]` is an id from `jobs.json`; `task_scope` is
`full_job`, `subtask` (one sub-capability) or `decision` (evidence given, decision asked).

A type file has `SPEC`, `VERSION`, `applies`, `group_key`, `build` (`l2h new tasktype <name>`).

- `applies(t, findings)` decides if a record can feed this type. Return `(False, "reason")` to leave a
  recorded exclusion reason. Records no type applies to land in `out/excluded.jsonl`.
- `group_key(t)`: attempts sharing a key become one task. Merge repeated attempts here.
- `SPEC["task_scope"]` (required; the scaffold leaves `TODO` so you choose):
  `full_job` | `subtask` | `decision`. Build each `full_job` first:
  the original request, starting state, tools/action space and output, graded on the real outcome.
  Subtasks and decisions are extras with their own honest scope; they never cover a full job.
  `l2h report` lists full jobs first.
- `build(group, out)` returns one of:
  - `BuildResult`: a package. Either `answer=Answer(...)` (l2h writes an answer-file verifier) or your
    own `files["tests/test.sh"]` that checks the end state in the task environment (no `answer`;
    must write `/logs/verifier/reward.txt`). Your own environment goes in
    `files["environment/Dockerfile"]` (+ context files) with `env_source` `harness`, `rebuilt` or
    `partial` (default `evidence` = l2h's image plus `evidence` files). Your own oracle goes in
    `files["solution/solve.sh"]` with `oracle_kind` `replay` / `fresh` / `reference`; `oracle_kind="none"`
    when there is no credible one. Record `grading_method`, `grading_reason` and
    `gaps=[{"what", "affects": execution|fidelity|grading|none}]`.
  - `Candidate(reason, gaps=[...], next_action="...")`: the job is identified but no runnable package is
    possible yet. Kept in the queue and the report; never counted as a task.
  - `Blocked(reason, missing, kind=...)`: for a `full_job`, only `kind` `secret`, `leak` or
    `no_instruction`. For optional children any reason is fine.
- Environment input: `l2h evidence <id>` lists what each run read (complete only when proven), wrote,
  ran and fetched, with step refs. Files read before being written are starting-state candidates;
  files the run wrote are outputs (hidden reference, not starting state). Logged setup commands are
  evidence for a Dockerfile, not commands to append blindly.
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
  logged answer), `solved` (a fresh solve passed), `imitation` (the logged answer), `negative` (see
  below), `none`. `ready` in the report needs validated, a non-partial environment, trust other than
  imitation/negative/none, and no open gaps.
- Negative tasks: the logs show a known-wrong answer and no known-right one. Use
  `gold_type="negative"` with `Answer(expected=[<known-wrong answers>], check="forbidden", pattern=<optional regex a well-formed answer must fully match>)`
  and at least one valid answer in `alternatives` (the oracle writes the first). The verifier rewards
  any non-empty, well-formed answer except the known-wrong ones. Level 2 checks: the oracle's
  different valid answer passes, each known-wrong answer fails, an empty answer fails. The instruction
  must say what a well-formed answer looks like; the known-wrong answers are not hidden from the scan.
- Bump `VERSION` when the generator changes so changed instances are revalidated.
- Idempotence: unchanged packages keep their validation state across `l2h run generate` reruns.
