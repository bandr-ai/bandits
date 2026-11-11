# Task types

Every type names the job it rebuilds: `SPEC["job"]` is an id from `jobs.json`; `task_scope` is
`full_job`, `subtask` (one sub-capability) or `decision` (evidence given, decision asked).

A type file has `SPEC`, `VERSION`, `applies`, `group_key`, `build` (`l2h new tasktype <name>`).

- `applies(t, findings)` decides if a record can feed this type. Return `(False, "reason")` to leave a
  recorded exclusion reason. Records no type applies to land in `out/excluded.jsonl`.
- `group_key(t)`: attempts sharing a key become one task. Merge repeated attempts here.
- `SPEC["task_scope"]` (required): `full_job` or `decision`. Prefer rebuilding the logged agent's whole
  job: starting inputs, tools/action space, output format, graded on its real output (`full_job`). When
  inputs or tools cannot be rebuilt from the logs (or a provided harness), a decision task is still
  valuable: give the evidence the agent gathered and ask for its decision (`decision`). Name and
  describe it as a decision-from-evidence task, not the full job (instruction, `description`). `l2h report`
  shows the scope per type.
- `build(group, out)` returns `BuildResult` or `Blocked(reason, missing)`. Needing a live system,
  missing evidence, or no independent gold means `Blocked`, not a made-up task.
- Always give `wrong_answers` that are *meaningful* (a plausible mistake, not noise) and `alternatives`
  when exact matching might be too strict. `hidden` lists strings that must never reach the solver:
  the original final response, answer keys.
- Evidence goes in `evidence={"path": text_or_bytes}` and appears at `/app/path`. Use bytes for
  binary evidence such as images (`{"img/chart.png": png_bytes}`); the leak and secret scans skip
  binary files, so a secret or answer inside an image is not caught: review those by eye.
  The solver writes its answer to `/app/answer/<answer.file>`.
- Gold: `verified` (independent verdict for this exact scope), `solved` (not in this version),
  `imitation` (logged answer), `negative` (see below), `none` (blocked).
- Negative tasks: the logs show a known-wrong answer and no known-right one. Use
  `gold_type="negative"` with `Answer(expected=[<known-wrong answers>], check="forbidden", pattern=<optional regex a well-formed answer must fully match>)`
  and at least one valid answer in `alternatives` (the oracle writes the first). The verifier rewards
  any non-empty, well-formed answer except the known-wrong ones. Level 2 checks: the oracle's
  different valid answer passes, each known-wrong answer fails, an empty answer fails. The instruction
  must say what a well-formed answer looks like; the known-wrong answers are not hidden from the scan.
- Bump `VERSION` when the generator changes so changed instances are revalidated.
- Idempotence: unchanged packages keep their validation state across `l2h run generate` reruns.
