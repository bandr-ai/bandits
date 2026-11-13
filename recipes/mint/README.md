# Standalone Mint

Mint turns agent logs into Harbor tasks with a coding agent. This distribution contains only
`mint`, its CLI, workflow and content gates, so hosted integrations can use it alongside an older
Bandits ingestion library. It has no runtime dependencies.

The source is shared with the `bandits-cli` distribution. Install one Mint distribution per
Python environment; both provide the same `mint` import and command.

Run `mint --help`, `mint schema` and `mint docs` for the installed version's contracts and workflow.
Docker and Harbor are needed for execution validation. Local verifier checks remain unverified.
