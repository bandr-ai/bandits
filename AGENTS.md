# Working on Bandr

The installable distribution is `bandr`; its command and conversion import are
`mint`. The `bandits` import remains the ingestion library.

Before using Mint from this checkout, run `uv sync --extra dev`, then
`uv run mint --version`. Use `uv run mint ...` while developing so the command
matches this source checkout. Do not install an older published version over it.

For trace-to-task conversion, read [the Mint skill](mint/SKILL.md) and follow the
installed command's `schema` and `docs`. Users can install that skill for Claude
Code with `mint skill install`, or `mint skill install --project`.

Run Ruff and the relevant pytest tests for changes. Packaging changes also need
`uv build` and `scripts/smoke_wheel.sh` against the built wheel. A local verifier
pass must never be described as Harbor validation.
