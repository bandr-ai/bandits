# Harbor output

Checked against harbor-framework/harbor (schema_version 1.4):

```
<task>/task.toml            [task] name = "l2h/<type>-<id>", [metadata.l2h] provenance, artifacts
      instruction.md        the prompt; says where to write the answer
      environment/Dockerfile, environment/data/...   copied to /app
      tests/test.sh         writes 0/1 to /logs/verifier/reward.txt
      tests/{Dockerfile,grade.py,expected.json}
      solution/solve.sh     oracle
```

- Separate verifier: `[verifier] environment_mode = "separate"`; `tests/Dockerfile` bakes `/tests`.
  The solver's `/app/answer` is declared in top-level `artifacts` so the verifier receives it.
- Gold lives only in `tests/` and `solution/`; provenance only in `task.toml`.
- `test.sh` honours `L2H_TESTS_DIR`, `L2H_ANSWER_DIR`, `L2H_REWARD_DIR` so `l2h task try` can run it
  locally. If you override `test.sh`, keep those variables.
- For anything beyond static evidence use Harbor's own `create-task` / `create-adapter` skills.
