# bandits

Read agent traces from any common format into one trace model.

`bandits` reads raw exports (OpenTelemetry, chat logs, Claude Code sessions and
the main tracing tools), strips secrets while it reads, and gives you one
`TraceCorpus`: traces made of spans, with every model call, tool call and
message in the same shape, whatever produced it.

> **0.3.0 (unreleased) is ingest only.** The command line, the turn judge,
> verifiers, exports, RLM mining and AWM emulation were removed. See
> [CHANGELOG.md](CHANGELOG.md). The last release with them is `bandits-cli`
> 0.2.0.

## Install

```bash
pip install "git+https://github.com/bandr-ai/bandits"
```

Python 3.11+. The only runtime dependency is `pydantic`.

## Use

```python
from pathlib import Path

from bandits.ingest import load_corpus
from bandits.store import ArtifactStore

corpus = load_corpus(Path("traces.jsonl"))   # source="auto" detects the format
for trace in corpus.traces:
    print(trace.trace_id, len(trace.spans))

# Optional: save it on disk under an id derived from its content.
envelope = ArtifactStore(Path(".bandits")).write(corpus, source_path="traces.jsonl")
print(envelope.artifact_id)
```

Pass `source=` when the format can't be detected. For large files,
`bandits.ingest.iter_corpus` streams traces instead of loading them all at once.

## Sources

| `source` | Reads |
| --- | --- |
| `otlp-std` | OTLP/JSON, including the GenAI, OpenInference, OpenLLMetry, Langfuse, Braintrust and Vercel AI SDK span conventions |
| `otlp` | Flat OTLP span exports |
| `chat-json` | Chat JSON and JSONL exports (OpenAI-style messages and the common chat export shapes) |
| `claude-code` | Claude Code session files |
| `langfuse` | Langfuse trace objects with `observations[]` |
| `langsmith` | LangSmith Run objects |
| `phoenix` | Phoenix Span JSON |
| `trail` | TRAIL benchmark traces |
| `failproofai` | FailProofAI sessions |

Details, including what each reader keeps and how formats are told apart:
[docs/ingest-sources.md](docs/ingest-sources.md). How each reader was checked:
[docs/ingest-validation.md](docs/ingest-validation.md).

## Redaction

Every reader redacts secrets (API keys, tokens, passwords, private keys) before
anything is stored. The default ruleset is `bandits.redact.DEFAULT_RULESET`.
Pass another `RedactionRuleset` as `ruleset=` to `load_corpus`.

## Layout

```
bandits/
  traces.py   # the trace model: TraceCorpus, Trace, Span
  redact.py   # secret redaction
  store.py    # content-addressed corpus storage
  ingest/     # one reader per source, plus format detection
recipes/      # older training recipes, not maintained against this version
```

## Development

```bash
uv sync --extra dev
uv run ruff check bandits tests
uv run pytest tests
```

Releasing: [docs/releasing.md](docs/releasing.md).
