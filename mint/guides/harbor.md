# Harbor output

Checked against harbor-framework/harbor (schema_version 1.4):

```
<task>/task.toml            [task] name = "mint/<type>-<id>", [metadata.mint] provenance, artifacts
      instruction.md        the prompt; says where to write the answer
      environment/Dockerfile, environment/data/...   copied to /app
      tests/test.sh         writes 0/1 to /logs/verifier/reward.txt
      tests/{Dockerfile,grade.py,expected.json}
      solution/solve.sh     oracle
```

- Separate verifier: `[verifier] environment_mode = "separate"`; `tests/Dockerfile` bakes `/tests`.
  The solver's `/app/answer` is declared in top-level `artifacts` so the verifier receives it.
- Gold lives only in `tests/` and `solution/`; provenance only in `task.toml`.
- `test.sh` honours `MINT_TESTS_DIR`, `MINT_ANSWER_DIR`, `MINT_REWARD_DIR` so `mint task try` can run it
  locally. If you override `test.sh`, keep those variables.
- For anything beyond static evidence use Harbor's own `create-task` / `create-adapter` skills.
