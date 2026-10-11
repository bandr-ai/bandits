## What the seeds show

**The agent is a single-question inspector, not a general debugger.** Every seed is one failure context in, a few targeted tool calls, and one strict JSON answer out (S1 page-readiness, S2/S10 cause-sweep groups, S3/S6 element presence, S4/S9 page identity, S5 native overlay, S7 open escalation, S8 keyboard occlusion). The answer's keys and enum values differ per inspector, so the system prompt — not the seed — must govern the output.

**The trajectories are short and tool-poor.** Typically 1–3 tool calls; often the second call only corroborates the first. Voluminous exploration is not the style; a well-aimed call plus an honest answer is.

**Tools routinely come back empty or unavailable — and that is realistic, not failure.** `query_har("pending")` → `available:false`; `locator_probe` on runtime with no captured source → `result:"unavailable"` with a reason; `a11y_search` → `count:0, nodes:[]`. Several seeds end in `root_cause: null` or `page_ready` / `not_occluded` precisely because the evidence positively supports nothing. That honest-null ending is a first-class label, not a degenerate one.

**Availability is derived from the failure context.** Seeds where `runtime_a11y` or `runtime page_source` is absent from `evidence_available` get exactly the `unavailable` / `available:false` reads with the matching reason. This consistency is the thing a generator most easily breaks.

**Tool semantics are narrow and must be preserved.** `compare_screenshots` is a vision model: it reports only what is visible, never network or tree state, and its `read` is an escaped JSON *string*. `a11y_search` returns role/name nodes; a `side:"record"` search says nothing about the live page. `locator_probe` only counts. Seeds show the agent hitting unavailable probes and still concluding correctly from the screenshot alone (01a0de25-f8b3, 01a0de31).

**Platform vocabulary splits.** MOBILE seeds use `XCUIElementType*`, `android.widget.*`, permission alerts, keyboards, and have no URLs or `readyState`; WEB seeds carry `runtime_url`, `runtime_title`, `ready_state` and sometimes a HAR.

## Where a synthetic trajectory would go wrong

- **Answer-shaped tool results.** A `compare_screenshots` read saying "no pending requests" or naming a root cause, or an `a11y_search` returning a cause, would hand the agent a conclusion the tool cannot know.
- **Availability contradictions** — returning runtime source data when `evidence_available` has none, or `available:false` on a side the context says was captured.
- **Wrong-cause leakage.** Every seed's system prompt carries its own candidate catalog; a cause valid for S7 is invalid for S2's `wrong_search_context` group, and vice versa. The answer must come from the system prompt in the job, never from a neighbouring inspector.
- **Seed copying.** Same URL, same element name, same step ids, same observation sentence — the trajectory then teaches nothing about the *shape* of the work.
- **Unsupported conclusions**, especially the expensive ones: `absent` without a screenshot affirming the region is visible; `diverged` when only dynamic content differs; `keyboard_occluded` when the keyboard is merely present.
- **Perfection bias** — every job resolving to a confident cause. The seeds show the opposite, and the null/negative verdicts are the calibrated behaviour worth training.

## Why the template says what it says

- The seed, system prompt, schemas and observation examples are given as separate blocks with explicit roles so the generator cannot confuse *shape to copy* with *content to copy* and cannot let the seed's enums override the job's system prompt.
- The hard rules encode the contracts above: invent-everything (rule 1), enum/catalog discipline (rule 2), evidence-support and calibration (rule 3), per-tool honesty incl. the escaped `read` string and the record/runtime boundary (rule 4), availability derived from `evidence_available` (rule 5), platform vocabulary (rule 6).
- Rule 7 reproduces the mechanical texture the fine-tuning target needs: no system message, `content:""` on tool turns, `call-<uuid>-<n>` ids with parallel calls sharing a uuid, `[obs:N]` prefixed JSON tool contents, a bare final JSON answer, the occasional "almost out of steps" nudge, and a short total length.
- Rule 8 protects the distribution of outcomes: positive, honest-null-scenario, and divergence verdicts all appear across jobs so the model learns proportional confidence rather than always finding a cause.
- The template deliberately never describes the wrapper output format — code appends that requirement.