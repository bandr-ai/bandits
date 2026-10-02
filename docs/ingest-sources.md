# Trace inputs

All readers produce the same `TraceCorpus`. `bandits ingest PATH` recognizes
known source shapes automatically. `bandits ingest PATH --dry-run` checks recognition
and runs the reader without writing an artifact; it reports missing model input,
output, task, the recorded kind attribute used to recognize each model call,
and parsing issues without printing trace content. Use
`--source NAME` when the shape cannot be identified. For OTLP and native
exports, auto-ingest requires `--mode conversation` or `--mode workflow`:
their file format cannot establish who wrote a `user`-role model input. A new
workflow using a supported format does not need a new reader, but workflow
semantics still need `--mode workflow` and task/delivery field declarations.

| `--source` | Accepted recording |
| --- | --- |
| `otlp-std` | OTLP/JSON requests, including tested GenAI, OpenInference, OpenLLMetry, Langfuse, Braintrust, and Vercel AI SDK span conventions |
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
meaning. `null`, empty values, and zero stay visible in that archive.
The default `default-v2` redaction also inspects decoded JSON string values,
including JSON serialized inside an OTLP attribute; `default-v1` remains
available to reproduce old artifacts. It does not promise to find every kind
of sensitive data.

`envelope.json` records which code wrote the corpus (`bandits_version`, and
`git_commit`/`git_dirty` when bandits runs from its own checkout), the workflow
`derivation_version`, and `problem_count` (the warnings printed at ingest)
apart from `redaction_count`. `bandits list` shows problems, not raw issue
counts, and warns when workflow corpora in one project were derived by
different versions; envelopes written before these fields show `?`.

Parsed OTLP spans also keep separate resource, scope, span, event and link records in
`bandits.otlp.source_context`. A recorded link is preserved as a link; ingest
does not call it feedback or causation.

The convention index in `bandits.normalized_scalars` records the source key
used for model, provider and usage values. Its candidates were checked against
[MLflow's OTLP translators](https://github.com/mlflow/mlflow/tree/master/mlflow/tracing/otel/translation)
and [OpenInference](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md).
Seven independent OTLP captures from
[genai-interlingua](https://github.com/Grace/genai-interlingua/tree/ece3efa13745f8fe0a8b596f162118192d85d22a/testdata)
are included as regression inputs. They cover OpenInference, OpenLLMetry,
Braintrust, LiteLLM, Vercel AI SDK, LangChain, and an older OpenLLMetry shape.
The LangSmith reader has also been checked against one independently published
RunTree export. The Phoenix reader has a published API-contract check, but no
Phoenix-generated export check yet. See [the validation record](ingest-validation.md)
for the exact evidence and remaining limits. No Collector or MLflow process is
required.

The fixture tests check known mappings and archived bytes. They do not prove
compatibility with every version of every platform export; unsupported records
are reported as ingest issues. Auto-detection is a conservative structural
check, not proof that a mapping is semantically correct. It samples up to ten
records per file; the reader then checks the full export. If a sample record
exceeds 512 MiB, pass `--source NAME` explicitly. Unknown or mixed sources stop
with a reason; no LLM proposes a mapping yet. A source archive preserves what
was recorded, but it cannot restore content omitted upstream or removed by
redaction. A format match and a successful decode are the first two checks;
task origin, causal links, and stage quality require separate validation.
