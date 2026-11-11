# Harbor output

Checked against harbor-framework/harbor (schema_version 1.4):

```
<task>/task.toml            [task] name = "mint/<type>-<id>", [metadata.mint] provenance, artifacts
      instruction.md        the prompt; says where to write the answer
      environment/Dockerfile, environment/data/...   copied to /app
      tests/test.sh         writes 0/1 to /logs/verifier/reward.txt
      tests/{Dockerfile,grade.py,expected.json}   answer-file tasks only
      solution/solve.sh     oracle (absent when oracle_kind = none)
```

- Two verifier shapes. Answer-file tasks (`answer=Answer(...)`) use a separate verifier (below).
  Custom-verifier tasks (no `answer`, your own `tests/test.sh`) run the verifier in the solver's
  environment and check its end state (files, a running service, a built binary); `task.toml` then
  has no `artifacts` and no separate verifier mode.
- Separate verifier: `[verifier] environment_mode = "separate"`; `tests/Dockerfile` bakes `/tests`.
  The solver's `/app/answer` is declared in top-level `artifacts` so the verifier receives it.
- Gold lives only in `tests/` and `solution/`; provenance only in `task.toml`.
- `test.sh` honours `MINT_TESTS_DIR`, `MINT_ANSWER_DIR`, `MINT_REWARD_DIR` so `mint task try` can run it
  locally. If you override `test.sh`, keep those variables.
- Your own environment: `files["environment/Dockerfile"]` plus its context files, with `env_source`.
  Harbor's `create-task` / `create-adapter` skills cover the package conventions.
- Verifier-only tools (python3 for a grader, test runners) never go in `environment/`; install them in
  `tests/test.sh` after the solve, or use a separate verifier image (answer-file tasks already do).
- `[verifier] timeout_sec` defaults to 120; set `BuildResult.verifier_timeout_sec` so the verifier has
  its own sufficient budget (setup included), not a copy of `[agent]`'s. `task.toml` metadata also
  records `oracle_source`.

## Dialogue tasks (user simulator)

For a job that is a conversation with a user (see the skill: conversations stay conversations):

- `environment/docker-compose.yaml`: the solver's `main` service plus a `user-sim` service (and, if
  the job has tools/state, a service that owns them). Reuse a supplied simulator/runtime when the
  harness has one.
- The user's spec (goal, private facts, preferences, what they agree to) lives only inside the
  `user-sim` service's files, never in `instruction.md` or any file the solver can read.
- `instruction.md`: the opening message and how to talk to the user (e.g. a `user say "<text>"`
  command or an HTTP endpoint), plus the tool interface.
- Model and key: `BuildResult(runtime_env={"USER_SIM_MODEL": "${USER_SIM_MODEL:-}",
  "USER_SIM_API_KEY": "${USER_SIM_API_KEY:-}"})`. Use `${VAR:-}` so the task still starts without
  them. The simulator must start without credentials and fail only when asked to reply, so the
  oracle / no-op / wrong-outcome gates run with no model.
- Log every turn (user and solver) and every tool call to `/logs/agent/` for the verifier.
- Verifier: deterministic; checks the end state (and required statements in the transcript). Never
  grade with a model.
- Oracle: `solution/solve.sh` replays a passing attempt's tool actions against the tool service (no
  user turns needed). Wrong outcomes: no-op, a plausible wrong change.
- `agent_timeout_sec`: conversations need more than the 900 s default.

## Judge tasks

Judge tasks (`check = "judge"`) add `tests/rubric.json`, `tests/evidence/`, a stdlib judge `tests/grade.py`,
  `verifiable = false` and the pinned `judge_model` / `judge_rubric_version` / `judge_reason` under
  `[metadata.mint]`. The key never enters the package: `[verifier.env]` forwards `MINT_JUDGE_BASE_URL`,
  `MINT_JUDGE_API_KEY` and `MINT_JUDGE_MODEL` (default: the pinned model) from the host environment with
  Harbor's `${VAR}` syntax, so export them before `harbor run` / `mint validate --judge`. The verifier
  needs outbound network access: judge tasks set `[verifier.environment] network_mode = "public"`
  (use `allowlist` + `allowed_hosts` if you must restrict it). Missing config or an endpoint error leaves
  no reward file, which Harbor and `mint` report as an infrastructure error, never as 0 or 1.
