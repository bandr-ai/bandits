# AWM / diagnose issue register

Status: post-verification. Most audit fixes are applied; statuses below are set by
executed checks, not by implementation (D74). New issues I31-I34 are recorded in
`awm-decisions-v4.md`.

## What diagnose must do

`diagnose` decides whether a candidate model already possesses the capabilities shown in
an enterprise's traces, and whether those capabilities survive interaction.

It has two evaluation paths:

1. **Static/off-policy probes:** give the candidate task-start, middle-prefix, and
   end-prefix scenarios from real traces and judge its next action without simulating the
   rest of the world.
2. **Interactive/on-policy rollouts:** let the candidate choose new actions; use a
   trace-grounded tool-world AWM and a separate user policy to respond; maintain state in
   an external ledger; score the completed rollout with an external verifier.

A usable run must report capability, pass@k/reliability, simulator fidelity, unsupported
coverage/abstention, failure modes, and a routing recommendation. Capability and simulator
fidelity remain separate artifacts.

## Why the AWM may abstain

An AWM is allowed to abstain because the source traces do not cover every action a weak
candidate can take. In the current tau2 target family, recorded tool behavior is almost
entirely happy-path behavior and every relevant tool has zero recorded error examples.

If a candidate calls a tool in an unsupported state, the AWM has three possible behaviors:

1. invent a success;
2. invent an error/refusal;
3. decline to claim what the tool would do.

The first two manufacture enterprise semantics and can change the candidate's verdict.
Therefore the environment fails closed with option 3. Abstention is valid only when an
external support policy, based on retrieved evidence, says the transition is unsupported;
it is not the AWM's subjective confidence.

Abstained rollouts do not establish candidate success or failure. They are excluded from
the capability denominator and reported by tool, action/effect class, scenario kind, and
reason. A high abstention rate blocks the capability claim because the surviving rollouts
are not representative.

## How simulator trust must be measured

Replay held-out recorded transitions: hide the real reaction, ask the AWM to predict it,
then compare prediction with the recorded reaction. Measure at least:

- success-versus-error accuracy;
- exact structured-field accuracy, including list contents;
- state-delta path **and value** accuracy;
- event and invariant violations;
- per-tool and per-effect accuracy and coverage;
- correct abstention, wrong abstention, and supported-transition coverage;
- multi-step state drift with predicted state fed into the next step;
- user-policy premature disclosure, invented information, correct withholding, goal
  consistency, and termination behavior;
- calibration by retrieval support level;
- performance on task-start, middle-prefix, and end-prefix slices.

These metrics must pass reviewed thresholds on a held-out set. The sealed set is opened
once after prompts, thresholds, validators, and reporting code are frozen. Even a high
fidelity score only validates behavior on measured support; unsupported coverage remains
visible and can still block use.

## P0 — correctness blockers

### I1. No end-to-end campaign runner — **FIXED; CLI DEFERRED**

The modules are not wired into one reproducible command that loads artifacts, builds the
fit index, runs fidelity gates, evaluates candidates and controls, builds reports, and
saves lineage/versioned outputs.

**Resolution:** `bandits/diagnose/campaign.py` provides the supported Python execution
surface. Every input version and partition is explicit, and every attempted rollout,
including invalid and abstained ones, is persisted. A CLI is convenience work and is
deliberately deferred (D65), not a validity blocker.

### I2. Static/off-policy diagnosis is not implemented — **FIXED**

`StaticReport` exists, but there is no evaluator for the start/middle/end next-action
probes. This omits the cheapest answer to “does the base capability exist at all?”

**Fix:** implement a static probe runner that presents `CandidateView` at each compiled cut
point, evaluates the candidate action against recorded next-action evidence and reviewed
checks, and reports unjudged cases separately.

### I3. pass@k and pass^k are pooled across scenarios — **FIXED**

`report.py` combines every rollout into one global `(n, c)`. Tasks with more attempts or
easier tasks can dominate the result.

**Fix:** group by `scenario_id`, compute pass@k and pass^k per scenario, then macro-average.
Report scenario coverage and use a scenario-level bootstrap interval. Reject inconsistent
attempt counts or define their weighting explicitly.

### I4. Optimization leaks selection examples into later prompt revisions — **FIXED**

After accepting a candidate prompt, `optimize.py` stores its selection-set report as the
next critique source. The next revision therefore trains on selection errors.

**Fix:** critiques always come from the optimize/train split. Use the selection split only
for choosing between candidates; never pass its report or feedback to the proposer. Keep a
final sealed fidelity split entirely outside optimization.

### I5. The current “GEPA” proposer is not `dspy.GEPA` — **FIXED**

It is a custom loop whose proposal step is one `dspy.Predict` prompt rewrite. It does not
use GEPA's population, reflective feedback, or frontier/search behavior.

**Fix:** represent the AWM/user policy as DSPy programs and invoke `dspy.GEPA(...).compile`
with a metric returning score plus textual feedback. Use DSPy's public interface; do not
call the underlying `gepa` package directly. Rename the current helper if retained as a
simple baseline.

### I6. Optimization can improve by wrongly abstaining — **FIXED**

Accuracy is computed only on attempted predictions and the objective does not constrain
supported coverage. A prompt can avoid hard supported cases and appear more accurate.

**Fix:** measure correct and wrong abstention separately. Require minimum supported
coverage and a maximum wrong-abstention rate before a prompt is eligible; optimize
conditional correctness only inside those constraints.

### I7. Batched tool execution is still semantically all-or-nothing — **FIXED**

`RolloutStep` has `committed_call_ids` and `executed_call_ids`, but the rollout marks all
calls executed and marks all calls committed whenever any state was committed. A batch
where call A succeeds and call B fails cannot be represented correctly.

**Fix:** add a per-call outcome contract keyed by `call_id`, containing execution status,
observation/error, deltas, and events. Validate and commit each call independently and
derive the aggregate step fields from those outcomes.

### I8. AWM evidence and mutation claims are insufficiently validated — **FIXED**

The implementation now checks claimed evidence IDs against the retrieved set, mutation
paths against reviewed tool-effect entries, candidate calls against offered input schemas,
and declared tool results with standards-compliant JSON Schema. Canonical events are
stamped by the validator with their originating call ID; spoofed IDs, unsupported effects,
invalid schemas, invalid outputs, and cross-call verifier matches are rejected. Missing
output schemas remain explicit as `unavailable`, while invalid results remain `invalid` on
the rejected persisted step.

**Verification run (v4/S7-S9).** The focused tests, the diagnose suite, and the full
repository suite were executed: 897 passed, 1 failed. Verified and holding: output-schema
validation including boolean schemas, `unavailable`/`invalid` statuses, validator-owned
`_call_id` provenance, spoofed-event rejection, and the cross-call verifier-matching block.

**Closed (v4/S11).** Per-call completeness was the last outstanding half: submitted call
IDs must now equal outcome call IDs, and no branch infers execution from the call list.
D67's aggregate single-call form is preserved -- the rule is about silence, not about the
encoding. Closed by executed checks; full suite 919 passed, 0 failed.

### I9. Fidelity, optimization, and reporting lack direct tests — **FIXED**

The repository has 858 passing tests, but there are no `fidelity_test.py`,
`optimize_test.py`, or `report_test.py` files. The most consequential numerical logic is
therefore not covered.

**Fix:** add adversarial unit tests before changing these modules, followed by a fake-model
end-to-end campaign test and a real-artifact tau2 smoke test.

## P1 — validity and measurement weaknesses

### I10. Structured fidelity ignores list contents — **FIXED**

The flattener compares a list only by length. Different flights, passengers, or transactions
with equal lengths count as identical.

**Fix:** recursively compare list elements using a declared ordered/unordered policy per
field and report missing, extra, and incorrect values.

### I11. Multi-step drift does not accumulate predicted state — **FIXED**

`multi_step_drift` resets the ledger from each recorded `state_before` instead of applying
the AWM's predicted delta to the next step.

**Fix:** teacher-force only the recorded actions; carry predicted state forward. Compare
the predicted and recorded ledgers at every horizon and report drift curves by tool/shape.

### I12. State-delta fidelity compares paths but not values — **FIXED**

Two predictions touching the same path count as delta-correct even if the new value is
wrong.

**Fix:** compare path, old value where known, new value, originating call ID, and event
semantics.

### I13. User-policy fidelity is mostly lexical — **PARTIAL**

Token overlap cannot reliably determine whether an agent requested a fact or whether the
simulated user revealed, withheld, negated, or invented it.

**Fix:** compile known/unknown information into normalized fact IDs and construct a
recorded disclosure timeline. Compare simulated disclosure events to that timeline; use a
rubric judge only for residual paraphrase mapping, not as the source of truth.

### I14. User-policy retrieval shape compatibility is ineffective — **FIXED**

The current compatibility predicate accepts essentially every bound success shape. User
turns from refusal, informational, and mutation tasks can therefore cross-ground one
another even though user behavior depends on the goal.

**Fix:** define and test an explicit compatibility matrix for the user policy. Keep tool
semantics shape-independent so the environment never refuses on the candidate's behalf.

### I15. Support threshold is unresolved but defaults to `LOW` — **FIXED FOR CAMPAIGNS**

Decision Q12 says the commit-versus-abstain threshold must be chosen from Gate 2 data, but
`Budget.min_support` and `validate_transition` currently default to `LOW`.

**Fix:** remove the production default. Require a versioned `SupportPolicy` selected from
held-out calibration, with thresholds per role/tool/effect where necessary.

### I16. “Abstention correct iff no examples” is too coarse — **FIXED**

The current fidelity label treats abstention as correct only when retrieval returned zero
examples. Irrelevant, weak, or wrong-role evidence can exist without supporting the
specific transition.

**Fix:** label support against transition compatibility and required semantics, not raw
example count. Evaluate calibration across `none/low/medium/high` support buckets.

### I17. Capability reports can silently mix versions or candidates — **FIXED**

`build_capability_report` copies environment versions from the first rollout and does not
reject mixed versions or a mismatched `candidate_id`.

**Fix:** validate candidate, scenario-set, AWM, user-policy, retrieval-index, verifier, and
support-policy versions across every rollout before aggregation.

### I18. Invented tool calls are missed when no tools are offered — **FIXED**

`unoffered_calls` returns no violations when the offered set is empty, although every tool
call is then unoffered.

**Fix:** distinguish “offered set not supplied” from “supplied and empty”; in the latter
case report every call.

### I19. Candidate identity and endpoint behavior are not fully reproducible — **FIXED**

The candidate digest omits endpoint and max-token settings, has no decoding seed, and the
adapter ignores `CandidateSpec.endpoint` in favor of a hard-coded provider prefix.

**Fix:** include provider/endpoint, model revision, instruction, temperature, seed,
max-tokens, tool schema version, and adapter version in candidate identity and honor them
when constructing the LM.

### I20. Scripted candidates are stateful across attempts — **FIXED FOR CAMPAIGNS**

A scripted candidate consumes its action list. Reusing it for pass@k produces different
behavior because earlier attempts exhausted it.

**Fix:** expose a candidate factory or resettable protocol and instantiate/reset once per
rollout.

### I21. User-policy support is not externally capped like tool-world support — **REVISED**

The tool-world proposal's support is capped by retrieval, while the user policy can claim
any support even when no user examples were retrieved.

**Resolution:** the hidden user profile is itself the primary grounding source, unlike tool
semantics. Empty behavioral retrieval does not imply no factual support. Profile coverage
and behavioral-example support still need separate calibration (D50).

### I22. Simulation-conditioned provenance needs verdict-level validation

The report should state whether the claims that actually determined the verdict relied on
simulated world state. Episode-level presence of simulated claims can overstate or
understate this.

**Fix:** preserve claim IDs consumed by each verifier result and compute conditioning from
only those claims, then aggregate per rollout and report stratum.

## P2 — integration, reporting, and documentation

### I23. No persisted campaign artifact or resumable execution — **PARTIAL**

There is no single artifact binding scenario set, partitions, prompts, candidate settings,
support policy, verifier, rollouts, fidelity gate, cost, and final report.

**Fix:** define a content-addressed `DiagnosisCampaign` manifest and append-only result
store; make retries idempotent and preserve failed/abstained attempts.

### I24. No explicit cost/rate/error-budget enforcement

The recovered tau2 budgets and provider costs are not wired into campaign execution.

**Fix:** add per-rollout and campaign limits for steps, repeated actions, model calls,
tokens, wall time, errors, and money; report exhaustion as candidate failure or environment
failure according to ownership.

### I25. Control-model acceptance gates are not enforced — **FIXED FOR CAMPAIGNS**

Controls exist, but nothing requires the competent reference to beat giving-up/looping
controls before publishing results.

**Fix:** make control separation a mandatory campaign gate and block reports when sanity
ordering fails.

### I26. No coverage requirement across scenario kinds and success shapes

A headline can be computed from whichever rollouts remained scorable, hiding missing
task-start, middle, end, refusal, informational, or mutation slices.

**Fix:** publish attempted/scorable/abstained counts per scenario kind, shape, family, tool,
and effect; require reviewed minimum coverage for each promised slice.

### I27. The v2 decision record contradicts itself — **FIXED**

- Q9 is resolved in D27 but still appears under open questions.
- the original D34 remains provisional although it is later reversed;
- D40's original shape rule contradicts the corrected tool-world behavior;
- Q11 is partly represented in code but remains unresolved in the record;
- Q12 says no threshold is guessed while code defaults to `LOW`;
- the session log stops at retrieval and omits later modules;
- the recovered user-policy prompt/config is paraphrased in code without complete
  versioned provenance.

**Fix:** edit superseded entries in place with explicit `SUPERSEDED` pointers, resolve or
retain Q11/Q12 accurately, correct D40 to separate tool-world and user-policy compatibility,
and extend the session/evidence log through the current implementation.

### I28. Sealed-partition wording is ambiguous — **FIXED**

The record says sealed is “excluded from retrieval.” The intended rule is that sealed
transitions can never be retrieval sources; a sealed scenario may still query the fit-only
index during its one final evaluation.

**Fix:** state this distinction explicitly in the decision record and assert both sides in
tests.

### I29. Re-ingestion/control-marker migration is not operationalized

D38 requires re-ingestion with `###TRANSFER###` declared and preservation of the legacy
artifact, but no campaign preflight enforces it.

**Fix:** add an ingest-contract preflight that records the ID delta, detects undeclared
known markers, and either requires the migrated corpus or records an explicit compatibility
warning.

### I30. Candidate-owned terminal failures could still pass — **FIXED**

Step/action-loop/token/time exhaustion was described as candidate failure but normal
verifier scoring could still pass a rollout that had performed an earlier required effect.

**Resolution:** these termination reasons now force an overall candidate failure (D60).

## Closed in v4/S11

I31 (outcome completeness), I32 (fidelity runs the runtime validator), I33 (explicit
serialization projections), I34 (real tau2 smoke test), D72 (standards-compliant input
validation), and I8. Full suite 919 passed, 0 failed; ruff clean.

## Closed in v4/S12

First real-model fidelity run (`scripts/run_awm_fidelity.py`, DeepSeek V4 Flash 0731,
against the pinned tau2 family) found and closed four new issues, all discovered because
they blocked interpreting the run's own result:

- **I35** (P0, FIXED): `_coerce` could not parse a real `dspy.Prediction` object at all
  (no `model_dump`, not a dict) — every real model answer silently read as abstention. See
  `awm-decisions-v4.md` D75.
- **I36** (P0, FIXED): malformed structured output collapsed into the same abstention path
  as a genuine decline, hiding a prompt/parser defect behind a metric that looked
  epistemic. D75.
- **I37** (P0, FIXED): fidelity compared the wrong observation representation for the
  (D67-sanctioned) batch `call_outcomes` form, and batch status accuracy aggregated "did
  any call error" across a whole batch instead of comparing per correlated call. D76, D78.
- **I38** (P1, FIXED): `inferred_state_delta` was every reported field, not a diff against
  `state_before`; fixing that exposed that cross-tool entity-path prefixes leave delta
  ground truth `unavailable`/`not_applicable` for 100% of the pinned family — an honest,
  reported limitation now (`delta_ground_truth_status`, `delta_ground_truth_coverage`), not
  a silently-wrong number. D77.

Full suite 932 passed, 0 failed (`uv run pytest -q`, no targeted subset). See
`awm-decisions-v4.md` S12/E33-E38/D75-D78 for the executed evidence and Q19/Q20 for what
remains open (a reviewed tau2 path canonicalizer; user-policy output-invalid taxonomy).

## I39 — CLOSED (narrowly) in v4/S13

**I39** (P0, FIXED — scorer-only, single-call record-materialization reads): `support_level()`
(`retrieve.py`) and the `abstain_correct` check it fed were entity-blind — a retrieved example
that called the *same tool* for a *different* entity counted as support, with no check that
the transition under test's specific entity (reservation/user id) was ever actually observed
anywhere in `state_before` or retrieval.

Found running the ten-transition stratified experiment (E39): six of ten held-out transitions
asked the AWM to reproduce a specific entity's private record it was never given (e.g.
`get_user_details("emma_kim_9957")`, expecting Emma Kim's exact name/address/email/DOB/payment
methods — none of which appear in `state_before` or in the retrieved examples, which concerned
other users entirely). The model correctly declined on all six; the scorer read two of those as
`wrong_abstention` and (via the separate, already-fixed D75/Q22 structural issue) four as
`output_invalid`, because retrieval supplied same-tool-different-entity evidence that
`support_level()` treated as sufficient grounding for an exact record it could not possibly
determine.

**What shipped.** `bandits/diagnose/grounding.py` (new) plus `CallGroundingAssessment`/
`GroundingAssessment`/`EntityRef`/`GroundingKind` in `models.py`. Per Q21's resolution (D79,
below), grounding is assessed at scoring time from the transition's actual `state_before` and
retrieved evidence, using evaluation demand read from the recorded result's field paths.
Deliberately narrow, per review of an over-general first draft:

- entity identity is tool-independent (`reservation:Q69X3R`, never
  `get_reservation_details.Q69X3R`) so the same entity resolves identically across tools;
- grounding is assessed **per call**, correlated by `tool_call_id` (falling back to
  positional pairing only for single-call transitions, where this corpus often carries no
  call id on either side) — never pooled across a batch;
- `GroundingAssessment.values_identifiable` is `None` for any multi-call batch, not just a
  mixed one: two calls that individually agree (both unidentifiable) must not let the
  aggregate collapse to a false "batch abstention correct," since proposal-wide abstain has
  no single correct label for a batch;
- behavioral evidence (`behavior_supported`) is filtered to examples whose retrieved
  transition actually called *this call's* tool, not the whole query's `exact_tool` reason;
- `state_before` wins outright on conflict with a retrieved value (it is this rollout's own
  history; a retrieved trace can be a stale snapshot of the same entity from a different
  episode); two retrieved examples about the same entity that disagree on a field have that
  field dropped rather than letting whichever example was iterated last silently win;
- restricted to exactly `get_user_details`/`get_reservation_details` single-call reads
  (`ENTITY_TOOLS` in `grounding.py`) — writes, batches, `book_reservation`, and non-entity
  tools (`transfer_to_human_agents`, search, calculate) report `GroundingKind.UNAVAILABLE`
  and fall back to the prior `SupportLevel`-based abstain-correctness rule, unchanged;
- `DERIVABLE_FROM_STATE` is reserved in the enum for a future mutation classifier (e.g.
  cancellation, whose post-state is a transformation of known pre-state) and is never emitted
  by the current read-only classifier — a record-materialization read reports a record, it
  does not derive one, so it is always `EXACT_ENTITY_FACT`;
- `TransitionFidelity.grounding` is attached on every scored row (abstained, output-invalid,
  validator-rejected, accepted), not only on abstentions — an auditor that only ran on
  declines could not answer whether an accepted success was state-grounded, or whether a
  non-abstaining answer was fabricated on an unidentifiable case;
- `output_invalid`'s short-circuit (D75) is untouched: a malformed response is never
  reclassified as a correct abstention regardless of what grounding finds.

**Verification executed this session (v4/S13).**

```text
uv run pytest bandits/diagnose/grounding_test.py bandits/diagnose/fidelity_test.py -q
  → 32 passed, 0 failed  (14 grounding, 18 fidelity)

uv run pytest -q   (full default testpaths: tests/ + bandits/)
  → 946 passed, 0 failed

ruff check bandits/diagnose/ scripts/rescore_awm_fidelity.py
  → all checks passed

git diff --check
  → clean
```

Real-artifact zero-model-call rescore, `scripts/rescore_awm_fidelity.py` (new) against the
real DeepSeek V4 Flash `baseline-fixed2` run — reloads the pinned tau2 family, rebuilds
retrieval, and replays each saved `raw_model_response` through the updated scorer. Refuses to
proceed (raises) if reconstructed retrieved-transition-ids or any of the four reconstructed
rendered strings (instruction/state/history/action/evidence) differ from what the original run
saved, so a rescore can never silently change the inputs alongside the scorer:

```text
uv run python scripts/rescore_awm_fidelity.py \
  --input work/awm-fidelity/baseline-fixed2.json \
  --output work/awm-fidelity/baseline-grounding-rescore.json

  → 2 of 2 clean lookup abstentions: wrong_abstention → correct_abstention
  → 4 of 4 malformed lookup responses: unchanged, still output_invalid
  → cancellation: unchanged, still accepted (36/36 fields)
  → 3 transfers: unchanged
  → wrong_abstention_rate: 1.0 → 0.0
  → correct_abstention_rate: 0.0 → 1.0
  → model_output_invalid_rate: 0.4 (unchanged)
  → attempted: 4/10 (unchanged)
```

Re-run a second time in this session: byte-identical output to the first rescore artifact.

**Scope of the close.** This closes I39 as: *exact-value abstention calibration for supported
single-call tau2 record-materialization reads (`get_user_details`, `get_reservation_details`)
is fixed and empirically verified against a real model run.* It does not extend to writes,
batches, or non-entity tools (still `SupportLevel`-based, unchanged), and it is not the
agentic AWM-centered runtime (`read_world_state`/`search_transitions`/`inspect_tool_contract`/
etc.) described earlier in this document — that remains separate, unbuilt future work and
should get its own issue when scoped.

## Current next steps

1. Inspect one persisted deterministic campaign, including invalid/abstained attempts and
   per-call output-validation/provenance fields.
2. Choose support thresholds from held-out fidelity results; do not guess them. This now
   includes a threshold for `validation_rejection_rate` (Q16).
3. Run small real AWM and candidate campaigns only after the fidelity gate passes.
4. Freeze prompts, thresholds, validators, and reporting before opening the sealed split.
5. Gate 0's re-ingest migration (D38/D69): declare `###TRANSFER###`, record the id delta,
   re-mine dependents.
