# Ingestion validation record

This separates **format recognition**, **content preservation**, **task and
relationship interpretation**, and **step quality**. A pass at one level is
not evidence that the next level passed.

| Input | Evidence checked | Result | Limit |
| --- | --- | --- | --- |
| Exgentic GenAI OTLP | First 15 sessions from each of nine shards: 135 sessions, 4,425 model spans. Bandits vs. genai-interlingua, plus a direct source-content check. | Zero mapping disagreements and zero source-content losses; 1,482 spans are in the fixed hash holdout. | These spans already contain GenAI messages. Interlingua reports dialect `raw`, so this does not test translation of another convention. This sample has four harnesses and four benchmarks. The full nine shards have 10,056 sessions and 241,473 spans and were **not** run through both decoders. |
| Public OpenInference agent traces | [SearchAgentDemoTraces](https://huggingface.co/datasets/inference-net/SearchAgentDemoTraces), revision `6dd8e0422939749a5e839a6e1bda4291e4ca5e56`: 1,005 traces and 16,174 span JSONL records, wrapped into OTLP/JSON without changing attributes. `scripts/validate_openinference_export.py` compares decoded fields directly to the source. | All 5,783 source LLM spans are model actions; recorded raw input/output fields, input roles/text, 20,789 input tool calls, parent IDs and all 467 error statuses have zero mismatches in this check. All 16,174 recorded spans are represented as actions, workflow nodes or invocations. | **No model answer is available:** all 5,783 `output.value` fields describe an API response object and there are no `llm.output_messages`. Bandits now marks these outputs unusable. This validates input translation, not answer extraction, SFT or quality. The published file is not an OTLP envelope; the wrapper is part of this check. |
| Seven independent OTLP captures | One capture each from genai-interlingua's `testdata/`, covering OpenInference, OpenLLMetry, LiteLLM, Vercel, Braintrust and LangChain shapes. | The readers retain original attributes. A direct source-content check found a LiteLLM tool call under `gen_ai.completion.0.function_call.*` that both mappers initially missed. Bandits now recovers it; the two mappers disagree on that field, and interlingua still loses it. | One capture per dialect does not establish broad compatibility. |
| Private Langfuse JSON and converted OTLP | Same first 20 source records, 101 model observation IDs. Compare each model's `input.value`, `output.value`, normalized input and output messages, as well as declared request and delivery. | Zero ID or content mismatches; all 20 tasks and deliveries resolve. The separate strict workflow checker verifies recorded-span retention. | Both paths use Bandits' OTLP decoder; their agreement does not independently establish semantic correctness. The OTLP is converted from the same Langfuse records, not a separate dual-emission capture. |
| Langfuse upstream Agno trace | Langfuse's own six-observation trace fixture, outside the private export. | All six observation IDs retained. Two near-identical framework/provider model pairs are reported as duplicate instrumentation: two model actions remain and both enclosing records remain as workflow structure. Task stays unresolved without a declared field. | The same response text confirms one pair; the tool-call pair has different token counts, so its equivalence rests on direct nesting, matching model/status, near-identical timing and compatible reply shapes. Some raw input shapes do not normalize to structured messages. Langfuse's paired ChatML fixture also does not recover message roles and content for those inputs. |
| Public LangSmith RunTree export | [Public telemetry trace](https://gist.github.com/sidpan1/e9daf51795801269f7cae20792cc85de), SHA-256 `25811810a3f1c2884944abf44948b2d254c46321f6cfca84818dfb947ecd2f96`. | All 11 runs and their recorded parents retained; two model calls have normalized input/output messages; declared request and delivery resolve. | One run tree does not establish coverage of LangSmith variants. Parentage alone does not establish which output a later call consumed. |
| Phoenix native export | [Phoenix getSpans contract](https://github.com/Arize-ai/phoenix/blob/main/js/packages/phoenix-client/test/spans/getSpans.test.ts), plus a local Phoenix Python client 3.5.0 → server 13.9.0 → `get_spans` capture in `tests/fixtures/upstream/phoenix/sdk-server-getspans.json`. | The server-produced two-span export retains the request, answer, model input/output, span kind, IDs and parent link; the declared task and delivery resolve. A separate contract fixture covers top-level error status. | The two test spans were constructed and sent through Phoenix's SDK, not captured by automatic instrumentation. One run is insufficient for broad Phoenix compatibility. Parentage does not prove causal consumption. |
| Public Langfuse traces | 119 example traces linked from Langfuse's integration docs (40+ frameworks); list endpoint plus `observations.byId` for each of 773 observations, because the list endpoint returns input/output as null. `scripts/audit_source.py`. | 206 model calls. Input: 204 present, 195 interpreted. Output: 193 present, 193 interpreted. Found and fixed: null I/O became messages reading "null"; Gemini requests; role-less lists; OTLP key/value I/O; LangChain tool calls in `additional_kwargs` or as JSON strings; parts lists serialized into content; tool schemas listed as `role: tool` messages; embedding generations counted as model calls. | Nine inputs are application data (`args`, `state`, a tool list) or agno's duplicated outer record (Python repr strings), left uninterpreted. One `llm.chat` input records tool calls under `role: tool`. Traces were written as demos. |
| Public LangSmith shared runs | Four live public share links found in GitHub code, fetched with the public runs API: 828 runs, 89 LLM calls. | 89/89 inputs and outputs interpreted; decoded message count equals the source count for all 89; 680 tool calls and 680 tool results. | 43 of 47 links found were no longer shared. Four trees from four projects. |
| Instrumentation test replays | OpenLLMetry (OpenAI, Anthropic, LangChain) and OpenInference (OpenAI, LangChain) test suites run against their recorded responses; `scripts/replay/span_dump.py` keeps the spans each real instrumentor emitted. 1,421 upstream tests, 917 model calls. | Every recorded input and output interpreted, except: 64 OpenInference inputs from property tests with random role strings (correctly rejected); one content-filtered reply and two reasoning items with no text (empty in the source). Found and fixed: completion-request `prompt`/`prompts`. | Test inputs are short and synthetic. Many tests disable content capture, so those calls have nothing to interpret.  |

`scripts/crosscheck_ingest.py` compares message roles, text, tool calls, and
tool results with [genai-interlingua](https://github.com/Grace/genai-interlingua).
It now separately checks source-declared GenAI messages and flattened legacy
function-call names. This caught a loss shared by both decoders before Bandits
was fixed; it now exposes the disagreement and exits nonzero. Its
source-content checks do not yet cover every non-GenAI convention.

**Error spans:** on a real Exgentic session with two failed model calls,
`ingest --dry-run` now reports `failed calls with no output: 2` and exits 0. An
errored provider call without a reply is not a decoder loss. A successful call
with no usable recorded output still fails the check. A separate fixture
verifies that a transport response object's string representation is kept in
the raw attributes but never treated as a model completion.
The public OpenInference trace also exposed five failed invocation roots; the
workflow request now retains their recorded status rather than only their
input and output.

**Duplicate spans:** the reader does not merge spans just because they nest or
use the same model. A direct model child must nearly fill its parent in time,
match model and status, and have a compatible recorded reply. The outer record
is retained as structure and `ingest --dry-run` reports the pair. This rule found
the two Agno pairs and no pairs in the public OpenInference dataset; broader
false-positive measurement is still needed.

**Task and relationships:** the private-export and LangSmith task checks compare against
declared request fields. Stored parent IDs, tool-call IDs, text matches and
execution rounds have different meanings. The current checks preserve and
label these relationships; they do not prove causation for every edge. The
Langfuse nested-generation capture is a concrete unresolved interpretation
case.

**Step quality:** no result here says whether a model action was good. The
workflow judge still refuses these corpora pending stage-aware evaluation
(M2). The 16 model-generated reference labels from the private export are not human ground
truth. A quality claim needs reviewed labels and a held-out comparison of
stage-aware predictions to those labels; that work is not complete.
