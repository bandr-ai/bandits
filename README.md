# Bandr

**Turn agent traces into runnable Harbor tasks with verifiers.**

Mint, Bandr's command-line tool, helps a coding agent turn logs of past work into
new tasks: the original request, a starting environment, and a verifier that
checks the outcome. Use those tasks to evaluate agents or as environments for
training workflows.

Successful and failed attempts can both provide evidence. Repeated attempts at
the same job become one task. Missing evidence stays visible in the report.

## What you get

A [small synthetic example](examples/mint) turns three attempts into two tasks.
Two attempts have wrong answers; the example recomputes the expected results
from the starting files rather than trusting the logged answers.

```text
agent logs + starting-state evidence
                  ↓
          coding agent + mint
                  ↓
          Harbor task packages
          ├── instruction.md
          ├── task.toml
          ├── environment/
          ├── tests/test.sh
          └── solution/solve.sh   # when a credible oracle is available
                  +
          report.md              # provenance, gates, exclusions, gaps
```

[Harbor](https://github.com/harbor-framework/harbor) provides the task format and
container-based execution. Mint builds tasks and checks their verifiers; your
coding agent writes parsers and generators for the actual logs.

## Get started with your coding agent

Python **3.11+** is required. The distribution is `bandr`; the command is `mint`.

**Release preparation:** `bandr` is not published yet. To try this checkout now,
run `pip install .` from the repository root. After publication:

```bash
pip install bandr
mint --version
mint skill install
```

`mint skill install` installs the bundled skill into `~/.claude/skills/mint/` for
Claude Code. Start a new session, then ask:

```text
Use the Mint skill to turn the agent logs in ./logs into Harbor tasks in
./mint-workspace. Inspect the source first. Report what you can build,
what validation actually ran, and what evidence is missing.
```

For a project-scoped installation, use `mint skill install --project`, which
writes `.claude/skills/mint/SKILL.md` in the current directory. Re-run after an
upgrade; if the installed skill differs, review it and use `--force` to replace
that file. Other files are preserved. Other coding agents can read the
[Mint skill](mint/SKILL.md) directly; automatic installation currently targets
Claude Code.

Prefer an isolated tool install? Use `uv tool install bandr` or
`pipx install bandr`, then run the same `mint` commands.

## Try the example

The [runnable example](examples/mint/README.md) includes logs, a parser, a job
map, and a generator. No model calls or production data are needed. Its tested
local result is:

```text
3 trajectories → 2 task packages
oracle accepted; no-op and wrong answers rejected; valid alternatives accepted
status: generated (host-local checks)
```

To reach `validated`, install Harbor, start Docker, and execute the Harbor gates.
The example guide includes those commands. Its generated answer oracle proves
verifier wiring; it is not an independent agent solve.

## Work directly with the CLI

```bash
mint init ./logs ./mint-workspace
cd ./mint-workspace
mint sample
mint schema
mint docs
mint status
```

`init` starts a workspace; conversion requires source-specific code. Read samples,
write a parser and job map, build task types, then generate, validate, and report.
The skill guides the coding agent through this process. Findings checks are
optional. Add `--allow-missing-runtime` to `init` when Harbor/Docker are unavailable;
generation and host-local checks still work.

Commands provide next-step hints. Use `--json` for structured output and
`mint status` to resume a workspace.

## What the statuses mean

- **`generated`**: the task passed structure and content gates. Execution may be
  untested, local-only, incomplete, or missing an oracle.
- **`validated`**: the required execution gates passed in Harbor, including an
  accepted oracle and rejected no-op/wrong outcomes. This is evidence about the
  verifier, not proof that the task is faithful or the expected answer is correct.
- **`judged`**: a model grades the task. Model opinion is kept separate from
  deterministic validation; inspect calibration and execution evidence.
- **`blocked`**: a content scan found an answer leak or credential. Packages with
  detected credentials are removed from `out/tasks`; fix and regenerate.
- **`exec_failed`**: structure, execution gates, or a type audit failed. Inspect
  the recorded reason before using the task.

The report records source references, grading trust, oracle provenance, gaps,
and exclusion reasons. Some jobs remain candidates when the evidence cannot
support a runnable task. Binary files and explicitly allowed files are outside
the best-effort text secret scan; raw and parsed logs may remain sensitive.

Mint sends anonymous command usage (version, OS, duration, outcome and bucketed
counts), without log contents or paths. Use `mint telemetry off` or set
`MINT_TELEMETRY_DISABLED=1` to disable it.

## Guides and development

- [Parsing logs](mint/guides/parsing.md) · [Optional checks](mint/guides/checks.md)
- [Building whole-job task types](mint/guides/tasktypes.md)
- [Validation](mint/guides/validation.md) · [Harbor integration](mint/guides/harbor.md)
- [Release process](docs/releasing.md) · [Changelog](CHANGELOG.md)

For development, run `uv sync --extra dev`, then use `uv run mint ...`.
Run `uv run ruff check bandits mint tests`,
`uv run ruff format --check bandits mint tests`, and `uv run pytest tests`.
Existing Python imports remain `mint` and `bandits`; the latter provides the
trace-ingestion library. Older [SFT](recipes/data/sft) and [JEV](recipes/jev)
recipes are retained for reference and are not part of this quickstart.
