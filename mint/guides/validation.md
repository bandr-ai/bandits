# Validation

Level 1 (every task): structure (files, valid task.toml, inputs resolve, answer path stated in the
instruction, consistent expected answer) and the leak scan over `instruction.md` + `environment/`.
The scan flags: hidden strings, answer-like file names, copies of `tests/` or `solution/` files,
and short answers after labels like `answer:`. Evidence that merely shares words with the answer is fine.
For an `allowed_set` check, a label that appears only as one item of a list (comma/`or` list,
bullets, table rows) in the instruction or evidence is a label vocabulary, not a leak; "answer: X",
a lone mention, the original final response and copies of `tests/` files are still flagged.
Allow a legitimate flagged file with `BuildResult.leak_allow=["environment/data/x"]` and a `decide`.

Level 1 also records non-blocking **warnings** on the row (shown by `validate` and in the report):
`giveaway` (the exact expected answer appears verbatim in an evidence file), `alternatives`
(allowed-set check with no `alternatives` given), `missing_file` (instruction.md references an
`/app/...` path the package does not contain).

The secret scan (also level 1) covers `instruction.md`, `task.toml` and `environment/`: API-key and
token shapes, private keys, bearer tokens, `password`/`secret`/`token` followed by a literal value,
and URLs with embedded credentials. A hit blocks the task and names the file, line and a redacted
preview. Allow a legitimate hit with `BuildResult.secret_allow=["environment/data/x"]` and a
`decide`; allow-lists are recorded in `results.jsonl` (`allow`) and the report.

Level 2 (seeded sample per type + changed instances, `--sample-size N`): the real `tests/test.sh`
runs against: the oracle (solve.sh) must pass; doing nothing must fail; each wrong answer / wrong
`Outcome` script must fail; each alternative must pass. An `Outcome` runs in place of solve.sh, so it
can do real bad work (a broken build, a wrong service). With no oracle (`oracle_kind="none"`) the
other gates still run and `exec` is `no_oracle`: the task stays `generated`. Runner is `local` (no Docker): result rows record it.
Verifier crashes, missing reward files, timeouts and Harbor build/agent errors are **infra errors**,
not task failures: the task stays `generated` (never `exec_failed` or `blocked`) until a clean run.

Judge tasks (`check = "judge"`) are skipped at level 2 because every gate costs model calls. Opt in with
`mint validate --judge` (needs `MINT_JUDGE_BASE_URL` and `MINT_JUDGE_API_KEY` exported; `MINT_JUDGE_MODEL`
optionally overrides the pinned model). Gates: the reference (oracle) answer passes, an empty answer
fails (no call is made), each wrong answer fails, each alternative passes (pass = every criterion
judged supported), plus a stability gate: the reference is judged 3 times and any difference in
reward is a flip, which fails the gate. Missing config, endpoint errors and unparseable replies are
infrastructure errors (no reward is written). With Harbor the key reaches the verifier through
`[verifier.env]` (`${VAR}` taken from the host environment) and the verifier container needs outbound
network access (`network_mode = "public"`, set for judge tasks). Whatever the gates say, judge tasks
end `judged`, never `validated`: the judge is a model opinion, not ground truth.

`mint judge-calibrate --type T [--n 20]` measures the judge before you trust it: for a sample of
T's tasks that have a deterministic verifier it judges the reference, alternative and wrong answers
with a generic rubric (the reference is not shown), compares the judge's pass/fail (all criteria
supported) with the deterministic verdict, and records the agreement %, false accepts and false
rejects as a `judge_calibration` decision shown per type in the report. `--answers f.jsonl` takes
externally graded `{"task_id", "answer", "true_reward"}` lines instead (any task, judged ones use
their own rubric). It needs `MINT_JUDGE_BASE_URL`, `MINT_JUDGE_API_KEY` and a model
(`MINT_JUDGE_MODEL` or `judge_model` in `mint.json`); endpoint errors are never counted as disagreement.
The per-call limit is `MINT_JUDGE_TIMEOUT` (default 30 s); raise `gate_timeout_sec` in `mint.json` if
`validate --judge` times out.

A type whose sample fails marks its unvalidated tasks `exec_failed` and uses one fix attempt
(`--max-fix-attempts`, default 3). After that, stop and report. Fix, `mint run generate`, `mint validate`.

Level 3: `mint validate --level 3` lists a seeded sample per type. Read them with `mint task show`.
Record `mint decide "..." --kind audit --verdict pass|fail --type T --task ID --why ...`.
A fail marks the type's tasks `exec_failed` until regenerated. Also record `--kind derivability` after checking the
answer can really be derived from the visible inputs.

Status: `validated` = exec gates passed, oracle included, in a real Harbor run (host-local passes
and `no_oracle` stay `generated`); `generated` = structure+leak+secrets passed; `exec_failed` = the
package is malformed or its gates failed (fix and regenerate); `blocked` = leak or secret found.
Reasons are in results.jsonl. `ready` (report only) = validated, environment not partial, grading
trust not imitation/none, no open gaps.
`judged` = a judge task that passed structure + leak + secrets: model opinion, never `validated`.
`validated` means the gates passed, not that the answer is correct: `status` and `report` show the
gold kind next to it (`validated (imitation)`) and warn about validated tasks with no audit or
derivability decision.

Runners (`mint validate --runner auto|harbor|local`): `harbor` runs each gate with `harbor run`
(oracle agent, nop agent, and an oracle on a scratch copy whose solve.sh writes the wrong or
alternative answer), so images are built and the answer reaches the separate verifier. `auto`
uses it when `harbor` and `docker` are on PATH, otherwise the fast host-local runner, whose passes
leave tasks `generated`. Harbor build/timeout/agent errors are infra errors, not task failures.

Parallel and resumable: `mint validate -j N` (`--jobs`; config `validate_jobs` in mint.json, default 4;
`-j 1` = one run at a time) runs the independent gate runs (oracle, no-op, each wrong answer and
alternative) of all sampled tasks on one thread pool. Results and ordering are the same as `-j 1`.
Each task is saved (`out/results.jsonl` + `.mint/gates/<id>.json`) the moment its gates finish and
progress is printed per task on stderr, so a killed run keeps its finished tasks. Re-run `mint validate`
to continue: a task whose package, wrong answers and alternatives are unchanged and whose saved pass/fail
came from the same runner is not run again (`--task` still re-runs the ids you name). A saved failure
from an interrupted run is counted against the type's fix budget once, on the next run. The summary
shows wall seconds. Keep `-j` within what Docker and the machine can build at once.
