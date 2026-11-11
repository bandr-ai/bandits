# Validation

Level 1 (every task): structure (files, valid task.toml, inputs resolve, answer path stated in the
instruction, consistent expected answer) and the leak scan over `instruction.md` + `environment/`.
The scan flags: hidden strings, answer-like file names, copies of `tests/` or `solution/` files,
and short answers after labels like `answer:`. Evidence that merely shares words with the answer is fine.
Allow a legitimate flagged file with `BuildResult.leak_allow=["environment/data/x"]` and a `decide`.

Level 2 (seeded sample per type + changed instances, `--sample-size N`): in a scratch dir the real
`tests/test.sh` runs against: the oracle (solve.sh) must pass; an empty answer must fail; each wrong
answer must fail; each alternative must pass. Runner is `local` (no Docker): result rows record it.
Verifier crashes, missing reward files and timeouts are **errors**, not failures, and never block a task.

A type whose sample fails blocks its unvalidated tasks and uses one fix attempt
(`--max-fix-attempts`, default 3). After that, stop and report. Fix, `l2h run generate`, `l2h validate`.

Level 3: `l2h validate --level 3` lists a seeded sample per type. Read them with `l2h task show`.
Record `l2h decide "..." --kind audit --verdict pass|fail --type T --task ID --why ...`.
A fail blocks the type until regenerated. Also record `--kind derivability` after checking the
answer can really be derived from the visible inputs.

Status: `validated` = exec gates passed in a real Harbor run (host-local passes stay `generated`); `generated` = structure+leak passed;
`blocked` = reason in results.jsonl.
