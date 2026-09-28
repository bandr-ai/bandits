# Trace inputs

All readers produce the same `TraceCorpus`. The source format is selected with
`bandits ingest PATH --source NAME`. A new workflow using a supported format
does not need a new reader.

| `--source` | Accepted recording |
| --- | --- |
| `otlp-std` | OTLP/JSON requests, including GenAI, OpenInference, OpenLLMetry and Langfuse span conventions |
| `langfuse` | Bundled Langfuse trace objects with `observations[]`, JSON or JSONL |
| `langsmith` | LangSmith Run objects or a `runs[]` wrapper, JSON or JSONL |
| `phoenix` | Phoenix Span JSON objects with `context.trace_id` and `context.span_id`, or a `spans[]` wrapper |
| `otlp`, `chat-json`, `claude-code`, `trail` | Existing source-specific readers |

OpenInference is an OTLP convention, not another native export format. Phoenix
OTLP exports use `otlp-std`; the `phoenix` source reads Phoenix Span JSON.
Other Phoenix exports, including dataframes with different schemas, are not
covered by this reader.

Native readers translate into OTLP **inside BANDITS** and use the same OTLP
decoder. The Langfuse JSONL reader processes complete trace records in batches
so the whole export need not fit in memory. LangSmith and Phoenix run/span
records that share a trace are grouped across the input file.

On CLI ingest, the artifact contains `corpus.json`, `source-manifest.json`, and
`source/000000.json` (more files when the input is a directory). The source
archive holds **redacted** source bytes; its manifest records original and
redacted SHA-256 digests. It retains source fields that have no normalized
meaning. `null`, empty values, and zero stay visible in that archive. Parsed
OTLP spans also keep separate resource, scope, span, event and link records in
`bandits.otlp.source_context`. A recorded link is preserved as a link; ingest
does not call it feedback or causation.

The convention index in `bandits.normalized_scalars` records the source key
used for model, provider and usage values. Its candidates were checked against
[MLflow's OTLP translators](https://github.com/mlflow/mlflow/tree/master/mlflow/tracing/otel/translation)
and [OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md).
Two OTLP fixtures from
[genai-interlingua](https://github.com/Grace/genai-interlingua/tree/ece3efa13745f8fe0a8b596f162118192d85d22a/testdata)
are included as regression inputs. No Collector or MLflow process is required.

The fixture tests check known mappings and archived bytes. They do not prove
compatibility with every version of every platform export; unsupported records
are reported as ingest issues. A source archive preserves what was recorded,
but it cannot restore content omitted upstream or removed by redaction.
