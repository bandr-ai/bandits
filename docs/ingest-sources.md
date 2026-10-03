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
`source/000000.json` (more files when the input is a directory), plus
`inspect.html`: a self-contained page (no network requests) with the record
accounting, the trace shapes, each sampled trace as a tree of steps with their
inputs, outputs and fields, and the notes, redactions counted per kind. It holds
the corpus's redacted data, so it stays with the corpus; `bandits inspect ID`
rewrites it. The source
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

In workflow mode it also counts evidence links by kind, with the most any one
trace has and the time spent building them (`evidence:` line). This is a
measurement only: link rules are unchanged.

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
same fields are one option (one run recorded under two names). One option that
settles every trace is chosen; otherwise every option's fields are declared
together. Nothing needs a flag and nothing is guessed:

- **Agreement.** When several top-level runs hold the question and their texts
  are equal (surrounding whitespace aside), the question is that text. The
  invocation is the agreeing run whose output holds the answer field, else the
  first; `request.task_candidates` lists every field and run it was read from.
- **Conflict.** When their texts differ, no run is chosen: the trace is saved
  with `task_status: conflict` and every value in `request.task_candidates`
  (path, value, run), for whatever uses the corpus to choose from. Ingest
  reports it as a warning (`task_conflict`) and still saves.

The answer field is then read on the chosen runs (or any agreeing run), and
chosen when one path is present in every trace. Declared flags are always used
as given, and the options with their coverage stay visible in the output.

## Fields Bandits does not interpret

Each native converter names only the fields it reads. Every other field of the
record, including nulls, empty values, unknown keys inside an object it reads
only partly, and values of an unexpected shape, is kept on the step under its
original name in `bandits.unmapped`; the app's own `metadata` keys are kept as
`metadata.*`. Token counts and request parameters the source records under its
own names are also given their GenAI names (`gen_ai.usage.*`,
`gen_ai.request.*`). A field nobody has listed yet is therefore kept, not
dropped, and the inspect page shows these under **Misc**.

After a CLI ingest the sampled steps are compared with the raw records they
came from (in the redacted source archive): every field the converter does not
read must be in `bandits.unmapped` with the same name, type and value (exact),
and the share of raw values found anywhere in the parsed step is reported as an
informational signal (a value can match by coincidence). The run's own record
is kept as the trace's request and is not compared; OTLP sources have no raw
record pointer yet and are reported as not checked.

## Saved mappings

The choices an export needs can be saved once and reused:

```
bandits mapping propose FILE --source S --name NAME   # profile it, write an unconfirmed mapping
bandits mapping confirm NAME                          # confirm it as it stands
bandits ingest FILE --source S --mapping NAME         # apply exactly what was confirmed
bandits mapping show NAME
```

A mapping (`.bandits/mappings/NAME.json`) holds the source, `task_fields`,
`delivered_field` (may stay null), the identities (`KIND_LABEL|NAME`) allowed as
the invocation, optional `step_kinds` overrides (`tool`, `step`, `exclude`
per identity), and the trace shapes it was confirmed on. `propose` runs the
field discovery above; when that is ambiguous it lists the candidate runs, and
`propose --invocation "KIND_LABEL|NAME" --force` discovers within that run.
`confirm` refuses a mapping without task fields and records a digest of the
choices; a file edited afterwards is refused until confirmed again, as is one
made for another `--source`. Applying a mapping discovers nothing: an
invocation is chosen only when exactly one candidate has a listed identity,
`--task-field`/`--delivered-field` still override its fields, traces whose shape
it was not confirmed on are ingested and reported (`shape_not_in_mapping`),
spans it excludes stay in the source archive (`excluded_by_mapping`), and a
`tool` override on a span with calls beneath it is left unapplied
(`mapping_override_not_applicable`). The corpus records the mapping's name and
digest.

`envelope.json` records which code wrote the corpus (`bandits_version`, and
`git_commit`/`git_dirty` when bandits runs from its own checkout), the workflow
`derivation_version`, and `problem_count` (the warnings printed at ingest)
apart from `redaction_count`. `bandits list` shows problems, not raw issue
counts, and warns when workflow corpora in one project were derived by
different versions; envelopes written before these fields show `?`.

Parsed OTLP spans also keep separate resource, scope, span, event and link records in
`bandits.otlp.source_context`. Its `span_attributes` holds only the declarations
the span's attributes don't already carry as declared: a value that shadowed a
different resource value, or one the normalized view replaced (for example a
declared `gen_ai.input.messages`). Spans from native readers carry
`bandits.source.record`, a pointer (`line`, `index` or `document`, plus
`observation_id`) to their record in the redacted source archive;
`ArtifactStore.read_native_record` returns it. The native record is no longer
copied onto each span. A recorded link is preserved as a link; ingest
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


### Memory during ingest

The CLI writes OTLP-family corpora incrementally to a temporary directory beside
the project store. It publishes the directory only after health and record
accounting pass and the source archive is complete. Failed runs and dry runs
remove the staging directory. Written bytes and content IDs match serialization
of the same materialized corpus, including declared control markers.

Bundled Langfuse JSONL is decoded in chunks. Standard OTLP, LangSmith, and
Phoenix JSONL group interleaved spans in a temporary SQLite database and assemble
one complete trace at a time. This also applies to the discovery pre-pass;
standard OTLP directories group spans across files. Source order breaks timestamp
ties, the first duplicate span is retained, and the resulting traces keep their
existing order. The spool is removed on completion or failure.

Memory depends on the largest source record/trace and accumulated issues,
reports, and discovery summaries, rather than all recorded prompts and replies.
Grouping and canonical ordering use temporary disk space and add I/O. A native
export that is one top-level JSON array (pretty-printed, or a single line) is read
one element at a time, for ingest and for the source archive. A single JSON
document (one object, such as LangSmith's `{"runs": [...]}`) and the legacy flat
`otlp` conversation adapter still buffer input; convert such a file to JSONL
first (`jq -c '.runs[]' in.json > in.jsonl`) when it is large.
Library `load_corpus` and reading a saved corpus still materialize the corpus.
`iter_corpus` yields traces followed by a trace-free `TraceCorpus` footer; its
report is complete only when iteration finishes.
