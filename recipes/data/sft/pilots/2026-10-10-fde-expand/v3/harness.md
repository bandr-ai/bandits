## Task

One micro-agent (a "check") inside the failure-analyzer walk over a failed UI test step. When the deterministic checks end unresolved, the walk dispatches a check whose whole job is ONE question about the failure; the agent explores pre-fetched evidence through its pinned tool surface and returns one JSON verdict the walk consumes. [established: check_specs.py]

Each check fixes three things: a system prompt (its question and its answer contract), a profile, and a pinned tuple of tool names — the agent gets exactly its question's tools and no others. [established: check_specs.py, tools/__init__.py]

The agent never fetches evidence, never writes, never executes: every tool is a LangChain `BaseTool` closured on an in-memory, read-only `EvidenceBundle`, async-only, with no filesystem, shell or network beyond the bundle. [established: tools/__init__.py, tools/_base.py]

## Input contract

One user message beginning "Failure context:" followed by a JSON blob. Fields observed: `error_message`, `platform` (WEB|MOBILE), `element{recorded_role, recorded_name, step_label, locator_type, locator_value}`, `runtime_url`, `runtime_title`, `ready_state`, `recent_steps[{step_id, action, result}]`, `failed_step_id`, `evidence_available[]`. [inferred: seeds]

Some runs append a "Deterministic hint (verify, never assume)" paragraph carrying the upstream check's reading (e.g. page identity read the live page as `same_page`) plus the exact way the agent may refute it. [inferred: seeds]

`evidence_available` is filtered so the agent is only advertised evidence its bound tools can act on. [established: tools/__init__.py]

**Slot-scoped availability invariant.** The list is the sole advertisement of which slots exist for this run. For `a11y_search`, `locator_probe` and `compare_screenshots`, an `available: true` payload is only possible for a slot the list names; when the slot is absent the tool returns its unavailable payload with the tool's own reason sentence — never a success payload with fabricated nodes, counts or reads. `query_har` is the documented exception: its `available: false` means no network log exists for this failure (the mid-run reality) and is not an advertisement question. [inferred: seeds + real tool results; mechanism of the list's computation unknown]

Tool results arrive as tool messages whose content begins `[obs:N] ` (N incrementing once per tool result within the run) followed by the tool's JSON string. [inferred: seeds + real tool results] What produces the prefix is not visible in the harness source. [unknown]

Every field declared in a tool's `args_schema` is REQUIRED ("'' = wildcard" instead of declared-optional), because optional properties were observed stripped from the wire on the xai leg — a model could not pass them at all; an omitted argument is filled from the tool's `arg_defaults` before validation instead of raising. [established: tools/_base.py, tools/a11y_search.py, tools/compare_screenshots.py]

## Output contract

Final assistant message: a single JSON object, no prose around it.

- Question-answering checks: `conclusion`, `root_cause`, `confidence`, `detail`. [inferred: seeds + system prompts]
- Cause-sweep checks: `root_cause`, `confidence`, `detail`. [inferred: seeds]
- `element_presence` additionally requires `page_kind` (sign_in | error | blank | other_section) when the conclusion is `different_page`. [established: S1 system prompt, shaping.py `_carry_for`]

The answer is then shaped by `shape_answer`, the single spec gate every AI answer passes: [established: shaping.py]
- an unknown conclusion or unparseable JSON becomes the sentinel `no_usable_answer` — recorded on the trail, never injected into the walk;
- `root_cause` must lie inside `causes_for_conclusion(check, conclusion)`: the conclusion key owns the verdict family and the cause may only refine inside it; an out-of-family cause is dropped and the conclusion's default cause substituted;
- confidence is clamped to 0..1;
- a result is terminal only if the conclusion is terminal AND a cause survived (or the conclusion is `occluded`, a hand-off that carries no cause).

## Tools — exactly what each can and cannot observe

### a11y_search(side, role, name)
Reads the runtime or record accessibility tree. [established: a11y_search.py]
- found: `{"status": "success", "available": true, "count": N, "nodes": [{"role": ..., "name": ...}]}` — nodes capped at 10.
- tree never captured: `{"status": "success", "available": false, "count": 0, "nodes": [], "reason": "no <side> accessibility tree was captured for this analysis"}` — the reason exists precisely so `available=false` is not read as "the element is not there".
- role/name may be `''` (any); matching is normalized exact, then containment.
Cannot: tell a covered element from an absent one; see anything but role and name; witness absence.

### locator_probe(type, value, side)
Counts what a locator resolves to, in the surface's own dialect (Appium locators resolve against native XML). [established: locator_probe.py]
- `{"status": "success", "side": ..., "result": "unique|no_match|not_unique|unverifiable|unavailable", "count": <int|null>, "available": <bool>}` plus `reason` when there is one.
- `available` is `result != "unavailable"`; `unverifiable` therefore reports `available: true` with `count: null`.
- `unverifiable`: the dialect cannot be evaluated here (e.g. a recorder-injected alias locator probed against the raw pre-enrichment record capture) — never absence.
- `unavailable`: that side's source was not captured, so nothing can be concluded — e.g. runtime `reason` "no runtime page source was captured for this analysis", record `reason` "no recorded page source or locator tree was captured".
- A runtime question is never answered from the record document; the record side falls back `record_a11y_source` → `record_html`.
- No other outcome shape (including an error shape) is defined in the source. [established: locator_probe.py]
Cannot: answer a runtime question from record data; prove presence (a unique count is not a visual match).

### compare_screenshots(kind, question)
One vision call per invocation; the calling agent still owns its question's conclusion — the tool returns the vision model's structured read as data. [established: compare_screenshots.py]
- kinds: `full_page`, `element_crop`, `spinner_check`, `error_page_check`, `overlay_check`, `transition`.
- success: `{"status": "success", "available": true, "read": "<string containing the vision model's own JSON: answer, confidence, observations>"}`.
- unknown kind: `{"status": "error", "message": "unknown kind: <k>"}`.
- a needed side missing: `{"status": "success", "available": false, "message": "<reason sentence>"}` — e.g. "no live screenshot captured — nothing to compare the recorded references against", "no recorded page screenshot — a page-identity comparison needs both pages", "no recorded page or element screenshot — nothing to recognise the element by", "a transition needs the screen before the step and the screen after it" — a comparison kind with one side missing is unavailable, never judged (D44).
- no populated slots: `available: false`, `"message": "needed screenshots not captured"`.
- the image set is fixed by `kind` and capped by `settings.failure_analyzer_max_images`; the LIVE PAGE anchors when present.
Cannot: judge a comparison with one side absent; see anything outside the bundled slots.

### query_har(query)
`summary` | `pending` | `failures` over the parsed network log. [established: query_har.py]
- no har and no network logs: `{"status": "success", "available": false}` — the mid-run reality, meaning no log exists for this failure, never "nothing was pending".
- har absent but engine network logs present: `available: true` with `engine_counts`.
- `summary`: `{... "total_entries", "pending_count", "oldest_pending_ms", "failure_count"}`; `pending`: `{... "pending": [...]}`; `failures`: `{... "failures": [...]}` — lists capped at 10 — always with `"available": true`.

### walk_history(source)
`test_steps` | `success_steps` | `execution_history` | `rrweb_facts` | `contexts`. [established: walk_history.py]
- unknown source: `{"status": "error", "message": "unknown source: <s>"}`.
- slot empty: `{"status": "success", "available": false, "items": []}`.
- otherwise `{"status": "success", "available": true, "items": ...}`; a list is truncated to its last 15 items; every string leaf longer than 300 chars is truncated with a "[truncated N chars]" marker; screenshot-url plumbing keys are stripped before the model sees them. [established]
- `contexts`: `items{current_context, context_list, context_source_mismatch}`, available when either of the first two is non-empty.

## Evidence-to-answer rules (the answer must obey these)

1. Slot-scoped availability: see the invariant above. `available: false` names a reason; it is never paraphrased as a reading and never converted into absence, no-match or a node count. [established: a11y_search.py, locator_probe.py, compare_screenshots.py]
2. Programmatic first, visual second; a probe against `side='record'` proves nothing about the live page. [established: S1]
3. `unavailable` ≠ absent and `unverifiable` ≠ `no_match`. [established: locator_probe.py, catalog]
4. Absence is heal-blocking and must be witnessed: conclude `absent` only when a screenshot read AFFIRMS the element's region is visible and the element is not in it; a probe miss or an empty tree alone is never a witness. [established: catalog NSE_ELEMENT_ABSENT, S1]
5. When no page source or tree shipped, covered and gone cannot be told apart: an element whose expected region sits under an overlay/modal is `occluded`, never `absent`. [established: S1]
6. `occluded` is a hand-off with no root cause. [established: check_specs.py]
7. Confirm-before-heal: an identity-family read (page_identity, page_state, rendering_variance) whose cause carries the LOCATOR_CHANGED verdict is demoted to a hypothesis — element_presence must confirm before a heal. [established: shaping.py]
8. Absence margin: element_presence's `absent` at confidence ≤ τ demotes to a hypothesis for the tail to corroborate. [established: shaping.py]
9. Page identity must not read divergence out of dynamic content; a tie is same-page; `ab_variant` needs a POSITIVE variant marker, and an overlay that merely appeared at runtime (login modal, cookie wall, promo) is not a variant. [established: S3/S5 prompt, catalog]
10. element_presence must refute its own premise explicitly (`different_page` with `page_kind` sign_in|error|blank|other_section) rather than force `found_in_place`; `found_other_context` means another frame, window or webview of the SAME page. [established: S1]
11. A blocking_screens blocker terminates the walk only when the agent affirms it covers the step's target or blocks interaction. [established: shaping.py]
12. The divergence fork is decided by the step the agent NAMES, never by its label: before the failing step → earlier_divergence (PREREQUISITE), the failing step → flow_change, after it → no_signal; a `flow_change` naming no step is no_signal, never a default. [established: shaping.py]
13. A flow change reached without consulting the run's history is demoted (unwitnessed). [established: shaping.py]
14. Never guess a conclusion to avoid a low confidence; report what the evidence supports. [established: prompts]

## Unknowns (do not turn these into rules)

- The numeric value of τ. [unknown]
- The micro-agent step/deadline budget, and whatever produces the "You are almost out of steps" nudge seen in seeds. [inferred: seeds]
- Whether the catalog's `supports` / `min_supports` predicates are enforced anywhere (shaping.py never references them). [unknown]
- How `evidence_available` is computed from the bundle — only that it is filtered by the tools' slot map. [established: tools/__init__.py; mechanism unknown]
- What produces the `[obs:N] ` prefix. [unknown]
- S1's JSON contract enumerates a shorter cause list than the cause-knowledge block below it in the same prompt (which adds NSE_TEXT_CONTENT_CHANGED, NSE_DATA_BOUND_LOCATOR, NSE_TRANSIENT_ELEMENT); which of the two binds is not stated. [unknown]
- Anything about the vision model's own reliability beyond the `read` string it returns. [unknown]
