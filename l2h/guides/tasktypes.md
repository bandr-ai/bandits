# Task types

A type file has `SPEC`, `VERSION`, `applies`, `group_key`, `build` (`l2h new tasktype <name>`).

- `applies(t, findings)` decides if a record can feed this type. Return `(False, "reason")` to leave a
  recorded exclusion reason. Records no type applies to land in `out/excluded.jsonl`.
- `group_key(t)`: attempts sharing a key become one task. Merge repeated attempts here.
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
  `imitation` (logged answer), `none` (blocked).
- Bump `VERSION` when the generator changes so changed instances are revalidated.
- Idempotence: unchanged packages keep their validation state across `l2h run generate` reruns.
