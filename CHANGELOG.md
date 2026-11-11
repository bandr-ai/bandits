# Changelog

All notable changes to `bandits-cli` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
change the command line or the artifact format; such changes are called out.

## [Unreleased]

### Added

- `l2h`: a standalone logs-to-Harbor tool (library plus CLI, no dependency on
  `bandits`). It samples and queries agent logs, runs agent-written parse,
  check and task-type code over the whole corpus, writes Harbor task packages,
  scans them for answer leaks, and validates them with oracle, no-op and
  wrong-answer gates. Ships its own skill and guides (`l2h docs`). Plan:
  `docs/plans/logs-to-harbor.md`.

## [0.3.0] - unreleased

`bandits` is now ingest only. Everything else was removed, to be rebuilt from
scratch on top of ingest.

### Kept

- `bandits.ingest` (every reader, detection, discovery, health, mappings),
  `bandits.traces`, `bandits.redact` and `bandits.store`.

### Removed (breaking)

- **The `bandits` command line**, with all of its commands. Use the Python API
  (`bandits.ingest.load_corpus`, `bandits.store.ArtifactStore`) instead.
- **`bandits.verify`**: the turn judge (`judge-turns`), check proposal
  (`propose-verifier`, `review-checks`) and scoring (`score-traces`).
- **`bandits.export`**: SFT exports (`build-sft`, `export-nextstate`).
- **`bandits.analyze`**: task candidates, task families and RLM mining
  (`analyze`, `families`, `mine-rlm`, `audit-rlm`, `materialize-rlm-taskset`,
  `rlm-session`, `rlm-families`).
- **`bandits.emulate`**: AWM / trace-grounded world simulation.
- `bandits.inspect` (the HTML trace page), `bandits.fields`, `bandits.ledger`,
  `bandits.providers` and `bandits.transport`.
- The `llm`, `audit` and `emulate` extras, and the `typer` and `rich`
  dependencies.
- The scripts and docs for the removed code (AWM, RLM, the tau2 and TRAIL judge
  evals, the paper figures).

### Changed

- `bandits.genai` and `bandits.jsonarray` moved to `bandits.ingest.genai` and
  `bandits.ingest.jsonarray`.
- `recipes/` is left as it was and is no longer tested in CI. It imports
  modules removed here.

## [0.2.0] - unreleased

The first release on PyPI.

### Install

- Published as `bandits-cli` (the PyPI name `bandits` belongs to another
  project). The command and the import are still `bandits`:
  `pipx install bandits-cli`.
- Extras: `llm` (model calls through LiteLLM), `audit` (RLM mining, check
  proposal and audit), `emulate` (trace-grounded world simulation).

### Added

- `bandits --version`.
- Ingest for OpenTelemetry (standard OTLP/JSON and flat spans), chat JSON and
  JSONL exports, Claude Code sessions, Langfuse, LangSmith, Phoenix, TRAIL and
  FailproofAI, with deterministic redaction and content-addressed storage.
- `list`, `show`, `inspect`, `fields` and `get` to browse what was ingested.
- `analyze`, `mine-rlm`, `audit-rlm` and `materialize-rlm-taskset` to find
  task families in real user requests.
- The next-state verifier: `judge-turns`, `propose-verifier`,
  `review-checks`, `score-traces` and `export-nextstate`.
- `build-sft` for LLM-reviewed SFT rows straight from traces.

### Changed

- `bandits list` prints full artifact ids, which other commands accept,
  instead of shortened ones they did not. The `derivation` column appears only
  when a corpus has one, so the table fits an 80-column terminal.

[Unreleased]: https://github.com/bandr-ai/bandits/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bandr-ai/bandits/releases/tag/v0.2.0
