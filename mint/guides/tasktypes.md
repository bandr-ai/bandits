# Task types

Every type rebuilds one job end to end: `SPEC["job"]` is its id from `jobs.json`. Mint builds whole
jobs only: no subtasks and no *extracted intermediate* decision tasks (a step's or sub-agent's
decision pulled out of a larger job). A job whose whole assignment is a classification/decision is a
normal job: an answer-file task.

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
    (`pre_existing`), OS/runtime shown by version/OS queries, and setup done before the agent
    started (harness, image). Base image: a supplied Dockerfile/harness wins; otherwise choose one
    consistent with the observed facts and record as gaps only the unknowns that matter for the job
    (not every package).
  - Not environment: whatever the agent installed, configured, downloaded, built or wrote. That is
    the job: it goes in the solution (`replay`), never the Dockerfile.
  - `not_on_path` / `module_missing` facts with `before_agent_setup`: the agent didn't have it at
    the start; don't pre-install it. Later ones only describe that moment. `Unable to locate package`
    (`package_lookup_failed`) is a failed lookup, not absence. System facts come only from
    version/OS query commands, never from the contents of files the agent read.
  - Unknown → a gap (`affects: fidelity`), not a convenient guess.
- Wrong answers / alternatives: strings for answer-file tasks; `Outcome(script="...")` for any task
  (a bash script run in the environment instead of solve.sh). Wrong ones must be *meaningful* (a
  plausible mistake, not noise); give alternatives when the check might be too strict. `hidden` lists
  strings that must never reach the solver: the original final response, answer keys.
- Grader quality (custom verifiers especially):
  - Check the outcome the request asks for (a computed result, real behaviour), never a printed
    marker or a file's mere presence: a fake binary that echoes the expected text must fail.
  - Accept valid alternatives (layout, naming, approach); do not hard-code one path or one tool.
  - Enforce only what the original job or its policy supports: no added steps (e.g. authentication),
    minimality or exact-state rules unless the original requires them; don't merge values the
    original treats as different.
  - Check action history only where a requirement depends on it (e.g. consent before a change).
  - Numbers/money: reject NaN/inf; compare exact units (integer cents or exact decimals).
  - Repeat attempts at the same instance: when the attempts known to succeed end in the same
    state, that state is a supported target; when they differ, list the gap.
  - Give wrong outcomes that try the obvious shortcut (a stub printing the expected text, hard-coding
    the expected value) and alternatives that do the job differently. Add `source="..."` to each
    `Outcome` (free text: "logged attempt X, outcome known from ...", "synthetic shortcut").
- Verifier dependencies: tools only the verifier needs (python3 for a grader, a test runner) never go
  in `environment/`: the solver would get them for free. Install them in `tests/test.sh` after the
  solve, or use a separate verifier image (the answer-file default already does).
- Verifier time: set `verifier_timeout_sec` (default 120) to cover the verifier's own setup and work, not
  a copy of the solver's budget; keep per-call timeouts inside tests generous so a correct, slower
  solution is not rejected.
- `oracle_source` (free text) says which logged attempt or solve `solution/solve.sh` came from and how
  its outcome is known. `mint report` shows it, and each wrong/alternative outcome's `source`.
- Evidence goes in `evidence={"path": text_or_bytes}` and appears at `/app/path`. Use bytes for
  binary evidence such as images (`{"img/chart.png": png_bytes}`); the leak and secret scans skip
  binary files, so a secret or answer inside an image is not caught: review those by eye.
  The solver writes its answer to `/app/answer/<answer.file>`.
- Verifier choice, in order: force a deterministic check first. (1) structured answer: `exact` or
  `allowed_set`; (2) tests that run the answer (override `tests/test.sh` via `BuildResult.files`, keep the
  `MINT_*` variables); (3) `numeric` with tolerance; (4) required elements checked by code (partial credit
  allowed); (5) an `alternatives` list for other valid answers. Use the `judge` check only when none of
  these can work, and state why in `SPEC["judge_reason"]`.
- Judge tasks (last resort): `Answer(expected=<reference answer>, check="judge")`, `rubric=[3-6 yes/no
  criteria]` written from the instruction and evidence, never from the logged answer, `wrong_answers`
  (required), optional `alternatives`, text-only evidence and `SPEC["judge_reason"]` (non-empty, or
  `mint validate` fails structure). The verifier grades the solver's answer against the evidence
  (copied to `tests/evidence/`) and the reward is the fraction of criteria judged supported. The
  reference is shown to the judge as "may be wrong", never as the key. The model is pinned in
  `mint.json` (`judge_model`) or `BuildResult.judge_model` and recorded in `task.toml` with the rubric
  version and `verifiable = false`. Judged tasks end `judged`, never `validated`.
- Gold (`gold_type`) = why the grading is trusted: `verified` (an outside label for this job; mint's
  own check is `computed`),
  `computed` (recomputed from the inputs), `independent` (end-state/behaviour check not tied to the
  logged answer), `solved` (a fresh solve passed), `imitation` (the logged answer), `none`. `ready` in the
  report needs validated, a non-partial environment, trust other than imitation/none, no open gaps,
  and, for an `answer` oracle, `verified`/`computed` trust. Execution errors are listed apart from
  grading failures.
- Bump `VERSION` when the generator changes so changed instances are revalidated.
- Idempotence: unchanged packages keep their validation state across `mint run generate` reruns.
