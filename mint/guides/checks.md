# Checks

`CHECKS = [Check(id, version, fn)]`; `fn(t) -> Finding("pass"|"fail"|"unknown", evidence, note)`.

- A finding is an observation. Failing a check does not exclude a record; exclusion happens in a task type.
- Use `unknown` when you cannot tell. Never guess to get a clean `pass`.
- Bump `version` when logic changes; the run output shows what changed since last time.
- `mint run checks --limit N` previews the selected trajectories without saving findings or stage state.
  Run `mint run checks` without a limit to save the complete findings.
- A crashing check is recorded as `unknown` and grouped by cause; fix the crash.
- Before trusting a check: `mint sample --matching <id>` shows fresh fails and passes. Read them, then
  `mint decide "<what>" --kind spot_check --check <id> --verdict pass --why "<what you saw>"`. The report flags
  unchecked checks.
