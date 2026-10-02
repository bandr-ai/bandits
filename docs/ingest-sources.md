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

For OTLP and native sources, ingest prints where every record went:

```
records:  293 seen → 41 model · 73 pipeline_step · 179 node · 0 dropped (unreadable items: 0)
parents:  5 trace(s) have top-level steps whose parent was not exported (max 3 per trace)
```

"Seen" is every span the OTLP decoder read, or every observation a native
converter read (nested ones included). Each lands in exactly one bucket: kept
(`model`, `tool`, `invocation`, `pipeline_step`, `node`, `container`,
`step_with_calls`, `covered`) or dropped (`unconvertible`, `duplicate_native`,
`malformed_span`, `duplicate`, `cyclic`, `excluded`, `empty_trace`,
`root_step`, `unrepresented`), each dropped bucket with an issue. If the
buckets do not add up to what was seen, that is a Bandits bug: ingest stops
and saves nothing. Unreadable items (a line that is not JSON, a record with no
spans list, an unsupported native record) are counted separately, since the
spans inside them cannot be counted. Records are decoded records, so the
`tool` count can differ from the corpus when tool calls are also recovered
from model messages. The same report is saved as `report.json` beside the
corpus, outside its id.

It also groups traces by shape: each span as role, declared kind and name
over the set of its children, with every top-level step under one root, so
repeated steps and sibling order do not split a shape but different nesting
does. The five most common shapes are printed with their share, model calls,
task results and an example trace id; `report.json` keeps the top 50 and the
total count.

Issue kinds added with this report:

| kind | class | meaning |
| --- | --- | --- |
| `unconvertible_observation` | warning | native observations with no id, an unparseable or missing start/end time (a running observation), not an object, nested in one of those, or repeated within a record; examples name the reason |
| `trace_split_across_chunks` | warning | one trace id appears in native records read in different batches, so the corpus holds more than one trace with that id |
| `malformed_record` | warning | also raised now for a `resourceSpans`/`scopeSpans` entry that is not an object |
| `parent_not_exported` | notice | workflow traces whose top-level steps point to parents the export does not contain; one issue per ingest with up to three example traces |

Summary issues (`unparsed_value`, `unrepresented_span`, `source_container`,
`excluded_evaluator`, `task_unresolved`/`task_conflict`, and the ones above)
are issued once per ingest with the source path as location, never once per
native batch.

In workflow mode, when `--task-field` or `--delivered-field` is not given,
ingest first reads the export for them (no corpus is built in that pass). A
field qualifies by its last key (`query`, `question`, `input`, ... for the
request; `answer`, `output`, `response`, ... for the answer) at most two levels
below the run's `input`/`output`, or as the whole value when that is text. The
fields found on each kind of top-level run form one option; options with the
same fields are one option (one run recorded under two names). An option is
chosen only when it picks exactly one run in every trace and no other option
does; otherwise every option is printed with its coverage and the flag that
selects it:

```
task:     not chosen: input.payload.query → app(SPAN) 5/5 · input.question → graph(CHAIN) 5/5
```

The answer field is then read on the chosen runs only, and chosen only when
one path covers all of them. Declared flags are always used as given.

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
