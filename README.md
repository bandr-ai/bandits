# bandits

**Bring your agent traces. Get Harbor tasks back: for SFT data and as RL environments.**

```
your traces  →  bandits  →  Harbor tasks  →  SFT data
                                          →  RL environments
```

Most teams have thousands of production traces and nothing they can train on.
bandits turns them into tasks with tests.

## Status

Being rebuilt from the ground up. What works today is the first step:
**ingest**. It reads traces from any common format into one clean trace model.
Building Harbor tasks comes next.

## Install

```bash
pip install "git+https://github.com/bandr-ai/bandits"
```

Python 3.11+.

## Ingest your traces

```python
from bandits.ingest import load_corpus

corpus = load_corpus("traces.jsonl")   # format is detected automatically
for trace in corpus.traces:
    print(trace.trace_id, len(trace.spans))
```

Secrets (API keys, tokens, passwords) are stripped while reading.

**Reads:** OpenTelemetry (GenAI, OpenInference, OpenLLMetry), Langfuse,
LangSmith, Phoenix, Braintrust, Vercel AI SDK, chat JSON/JSONL, Claude Code
sessions, TRAIL and FailProofAI. Details: [docs/ingest-sources.md](docs/ingest-sources.md).

## Development

```bash
uv sync --extra dev
uv run pytest tests
```

---

## Recipes

Older recipes, kept for reference. They are not maintained against the current
version and may not run.

- [`recipes/jev`](recipes/jev): train a small decision model on your labeled traces
- [`recipes/data/sft`](recipes/data/sft): generate new SFT traces from a few good seed traces
