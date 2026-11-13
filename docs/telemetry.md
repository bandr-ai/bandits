# Mint usage telemetry

Mint sends one `command_finished` event for these milestone commands: `init`,
`new`, `run`, `validate`, `judge-calibrate`, `decide`, `report`, and `task try`.
This is small product-usage instrumentation, not execution tracing or error logging.

Disable capture with `mint telemetry off`, `MINT_TELEMETRY_DISABLED=1`, or
`DO_NOT_TRACK=1`. It is also disabled in CI and pytest. `mint telemetry status`
shows the effective setting. A malformed configuration disables capture.

## Event contract (schema_version 2)

Every event includes Mint/Python versions, OS family, command, elapsed milliseconds,
CLI exit category, schema version, and `dry_run`. An error additionally includes a
closed `error_type` (`usage`, `contract`, `interrupted`, `operation`, `internal`)
and `failure_phase` (`workspace`, `execute`, `render`). These categories never
contain exception text. `operation` includes a command returning a nonzero result.

| Command | Additional properties |
| --- | --- |
| `init` | Docker/Harbor executables present, allow-missing-runtime flag |
| `new` | Built-in scaffold kind; never its custom name |
| `run` | Stage, limited scope, cache hit when known; output counts; parse/check errors; generation queued/blocked counts |
| `validate` | Requested runner, normalized requested levels, actual runner when available, judge flag; level 1 checked/failed; level 2 sampled/passed/failed/infra_error/resumed; any_failure/any_infra_error |
| `decide` | Built-in decision kind and optional pass/fail verdict |
| `task try` | Verifier pass/fail when a numeric reward is available |
| `judge-calibrate`, `report` | Common properties only |

All counts use buckets: `0`, `1`, `2-9`, `10-99`, `100-999`, `1000+`.
Missing properties mean unknown/not applicable, not zero. `levels` describes the
request, not evidence that execution completed. Actual `runner` is absent on early
validation errors.

`exit=ok` means the CLI completed successfully. It does not mean all records
parsed, all checks executed, tasks were generated, or an answer passed. Inspect
stage counts and `verifier_outcome` separately. For checks, findings include
pass/fail/unknown results; `errors` counts execution exceptions, not failed findings.
For generation, `outputs` counts constructed tasks, `queued` counts queue entries,
and `blocked` counts constructed tasks with blocked status. Queued work need not
represent a software error.

Validation counts come from the current invocation's summary, respecting task/type
selection and dry runs. They never count all saved workspace result rows. Level 2
counts reflect the validator's sample and may include reused evidence; `resumed`
counts that reuse. They are not an exact count of newly launched executions.
A host-local pass is never Harbor validation. Use the actual `runner` property.
Version 1's whole-workspace status counts have been removed; filter dashboards by
`schema_version=2` when using the new semantics.

## Privacy and delivery limits

Property names and string values are allow-listed. Capture excludes paths,
prompts, queries, answers, task IDs, custom check/type names, log contents,
exception messages, and stack traces. Person-profile processing and GeoIP
enrichment are disabled. The network service still receives the connection's IP;
disabling GeoIP enrichment does not hide it from the server.

A random UUID persists in the user's Mint config directory. It identifies that
configuration, not a verified person or an individual project. Different workspaces
share it; shared configs share it; deleting the config resets it. Concurrent first
invocations can race to initialize it. Funnels therefore describe approximate
installation activity, not a guaranteed project conversion funnel.

Delivery is one synchronous attempt with a 0.2-second socket timeout, no retry,
queue, or background process. The socket timeout is not a total wall-time deadline.
Offline events and failed requests are discarded, so counts underreport usage.
`telemetry status` reports permission to attempt capture, not delivery health.
Forced process termination cannot emit a completion event. Argument-parser errors,
help/version, read commands, and direct Python-library calls are not captured.
The notice appears once on a terminal, or on `report`; piped milestone commands
may capture before a notice has appeared.

## PostHog references

The implementation uses the documented [capture API](https://posthog.com/docs/api/capture)
and anonymous-event controls. Static names, useful outcome properties, and schema
versioning follow [product analytics best practices](https://posthog.com/docs/product-analytics/best-practices).
The [Python SDK guidance](https://posthog.com/docs/libraries/python) also explains
that short-lived processes need explicit delivery handling and that delivery is
not guaranteed. This CLI keeps its stdlib sender to avoid adding a dependency and
shutdown lifecycle for one event per invocation.
