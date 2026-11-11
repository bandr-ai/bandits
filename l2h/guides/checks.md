# Checks

`CHECKS = [Check(id, version, fn)]`; `fn(t) -> Finding("pass"|"fail"|"unknown", evidence, note)`.

- A finding is an observation. Failing a check does not exclude a record; exclusion happens in a task type.
- Use `unknown` when you cannot tell. Never guess to get a clean `pass`.
- Bump `version` when logic changes; the run output shows what changed since last time.
- A crashing check is recorded as `unknown` and grouped by cause; fix the crash.
- Before trusting a check: `l2h sample --matching <id>` shows fresh fails and passes. Read them, then
  `l2h decide --kind spot_check --check <id> --verdict pass --why "<what you saw>"`. The report flags
  unchecked checks.
