# Changelog

All notable changes to `bandits-cli` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
change the command line or the artifact format; such changes are called out.

## [Unreleased]

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
