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
