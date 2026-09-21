# AWM / traces2rlenv — decision record, v4

Continues `reasoning-awm-decisions.md` (D1–D26), `reasoning-awm-decisions-v2.md`
(D27–D42), and `awm-decisions-v3.md` (D43–D67). This record starts from the first
**executed** verification of the external-validation P0 work. v3's S6 shipped that
work without running it; everything below is measured, not asserted.

## Session log

| # | What | Outcome |
|---|---|---|
| S7 | Run the deferred verification from v3/S6 | 897 passed, 1 failed; 4 leak sites found behind the 1 failure |
| S8 | Independently re-derive a third-party defect report | 6/6 claims reproduced; 1 understated, 1 mis-prescribed |
| S9 | Record what the green suite did not cover | D68–D74; I8 held open, I31–I34 opened |
| S10 | User ratifies v4 and sets Q13–Q15 | Explicit projections; legacy ids pinned; input validation is a Gate 0 blocker |
| S11 | Close I31, I33, D72, I32, I34 in that order | 919 passed, 0 failed; each closed by an executed check |
| S12 | First real-model fidelity runs (`scripts/run_awm_fidelity.py`, DeepSeek V4 Flash) against the pinned tau2 family: one transition, then ten stratified; ten scorer defects found and fixed as they blocked interpretation | 932 passed, 0 failed; one cancellation scored 36/36 fields; the ten-transition run found the scorer's entity-blind support/abstention check (E39), documented as provisional, not yet fixed |
| S13 | Fix I39 narrowly: per-call, batch-safe, tool-independent-entity grounding for single-call `get_user_details`/`get_reservation_details` reads (D79 resolves Q21); zero-model-call rescore against the real S12 artifact | 946 passed, 0 failed; 32/32 focused; real rescore matches predicted outcome exactly and reproduces byte-identical on a second run; I39 closed narrowly, agentic AWM-centered runtime left as separate unscoped future work |

## Evidence

### E27. The suite is green except for one export test, and that test is the tip of four leak sites.

Executed: `uv run pytest -q` → **897 passed, 1 failed**. Focused external-validation
set (107 tests) passed; `bandits/emulate` (227 tests) passed; ruff clean.

The failure is `bandits/export/export_test.py::test_start_context_reaches_the_exported_row`.
Adding `ToolSchema.output_schema` (D63) made every exported tool carry
`"output_schema": null`.

The test is one *symptom*. The cause is that four call sites serialize `ToolSchema`
with a bare `model_dump()`, so **any** field added to the contract automatically
enters their payloads:

```text
bandits/export/sft.py:658      -> SFT training rows
bandits/export/eval.py:121     -> held-out eval cases
bandits/analyze/tasks.py:72    -> `available_tools` Evidence, hashed into the analysis id
bandits/emulate/compile.py:481 -> scenario offered_tools
```

Only the first is test-pinned. The other three changed shape silently.

### E28. `exclude_none` is the wrong fix, and the data says so.

Measured:

```text
bare dump    -> {'name':…, 'description': None, 'parameters':…, 'output_schema': None}
exclude_none -> {'name':…, 'parameters':…}
```

`exclude_none` also drops `description: None`, which the current export contract
*does* emit — so the obvious one-word fix silently changes a second field and
breaks the same test from the other direction. Confirmed separately that
`output_schema=False` survives `exclude_none` (it is not None), so D63's boolean
schemas are not the obstacle; `description` is.

### E29. An AWM can make a submitted call disappear from a batch.

Executed against `validate_transition`:

```text
calls    : a=cancel(AAA), b=cancel(BBB)
outcomes : only a
result   : accepted=True, rejections=(), executed=('a',), committed=()
```

The validator checks outcomes ⊆ calls (`unknown_ids`) but never calls ⊆ outcomes.

Two variants measured, one of which is worse than the batch case:

```text
forbidden call dropped from a batch -> accepted=True, executed=('a',)
batch with ZERO outcomes            -> accepted=True, executed=('a','b')   <-- fabricated
single call, zero outcomes          -> accepted=True, executed=('a',)      <-- fabricated
```

The zero-outcome path is the sharper defect: control falls through to the `elif calls:`
branch, which marks every submitted call executed *from the call list alone*, with no
AWM outcome behind it. The batch case loses a call; the zero-outcome case invents
execution the AWM never claimed.

### E30. The fidelity gate does not measure what the runtime enforces.

`validate_transition` appears exactly once outside tests, at `rollout.py:343`.
`fidelity.py` calls `step_tool_world` at lines 312 and 470 and scores the raw
proposal. So every rule added under D51/D63/D64/D67 — output schemas, mutation
paths, event provenance, per-call evidence — is invisible to the gate that decides
whether the simulator is trustworthy enough to run.

### E31. Input and output validation are not the same standard.

Outputs use real `jsonschema` (`world.py:219`). Inputs use a hand-rolled loop in
`rollout.py:127–156`: `required` presence plus a six-entry primitive `type` map.
Nested constraints, `enum`, `oneOf`, `additionalProperties`, formats, and array item
schemas are unenforced. A candidate emitting a well-typed but semantically invalid
argument is not a deterministic failure, contrary to D62's intent.

### E32. No test touches the real tau2 artifact.

The only occurrence of `corpus-ee3b33086ef177d7` under `bandits/emulate/` is a
docstring in `compile_test.py:40` recording D36's provenance. Every emulate test runs
on fixtures. D36 said a fixture is a claim about the data and an unverified claim is as
wrong as unverified code — that rule is currently satisfied by comment, not by a test.

### E33. `_coerce` never handled a real DSPy prediction; every real model answer read as abstention.

`build_tool_world_predictor` returns `dspy.Predict(...)`'s result, a `dspy.Prediction`
object. `_coerce` only handled `isinstance(payload, model)`, a JSON string, or
`hasattr(payload, "model_dump")`. `dspy.Prediction` has none of these — it exposes
`toDict()`, not `model_dump()` — so every real prediction fell straight through to
`return None`, which `step_tool_world` reports as abstention regardless of what the model
actually said. Measured directly: `hasattr(dspy.Prediction(...), "model_dump")` is `False`;
the first two live runs (Nemotron, DeepSeek) both logged `abstain=false` in the raw model
response and `abstained=true` in the scored `TransitionFidelity`, a contradiction that
traces to this one gap.

### E34. Malformed structured output and abstention shared one code path.

Once E33 was fixed, DeepSeek's response parsed far enough to reach `ProposedTransition`
validation and still failed it: `call_outcomes` items carried business fields
(`reservation_id`, `status`) directly rather than the `{call_id, executed, observation,
state_delta, ...}` shape, because the DSPy signature declared `call_outcomes: list[dict]`
with only a natural-language `desc=`, giving the model no actual nested schema.
`_coerce`'s `except (TypeError, ValueError): pass` swallowed the six resulting pydantic
errors and returned `None`, again read as abstention. A model that attempted a
well-reasoned answer and a model that declined outright were indistinguishable downstream.

### E35. `inferred_state_delta` was every reported field, not a diff.

`compile._delta()` updated a dict with every reported field's post-action value and never
consulted `state_before`. Measured against the real corpus: the target family's own
`cancel_reservation` transitions reported ~14 fields including `origin`, `cabin`, and
`passenger.length`, none of which the cancellation changed — only `status` and
`payment_history` did. `delta_correct` was therefore comparing a prediction against a
ground truth that was mostly unrelated context.

### E36. Confirmed mutations require entity-prefixed path alignment, which mostly does not exist in this family.

Fixing E35 to diff against `state_before` exposed a second problem: `state_before`'s paths
are entity-prefixed by `_entity_prefix` (`<tool>.<entity_id>.<field>`, from
`reconstruct_state`), but `_delta`'s reported paths were bare field names. Even after
prefixing reactions the same way, a mutation crossing tools — `get_reservation_details.X.status`
(read) vs `cancel_reservation.X.status` (write) — never aligns, because the two calls use
different tool prefixes for the same entity field. Measured: every one of the 371
transitions in the pinned family now reports `delta_ground_truth_status` of either
`not_applicable` (252) or `unavailable` (119) — zero `measured`. Delta ground truth is
currently unmeasurable for this entire family without a reviewed tau2 entity-path
canonicalizer, which does not exist (D26/D27's fail-closed rule: an unreviewed mapping is
not built here).

### E37. Fidelity scored the wrong observation representation for the batch form.

`score_transition_fidelity` diffed `proposal.observation` — the top-level aggregate field —
against the recorded result. A model using the (D67-sanctioned) batch `call_outcomes` form
correctly leaves that top-level field empty (`{}`) and puts its actual answer inside
`call_outcomes[i].observation`. Measured on the real DeepSeek run: the model's per-call
observation was a near-exact match to the recorded cancellation, but `field_accuracy` scored
`0.0` because the comparison was against the deliberately-empty top-level field. The correct
comparison target is `validate_transition`'s own canonical, call-correlated result — the same
one the runtime would commit — not a field the caller re-derives independently.

### E38. Batch status accuracy aggregated "did any call error" instead of comparing per call.

`status_correct` was `error_predicted == error_recorded`, each an `any()` over every call in
a batch. Constructed case: call A actually errors and call B does not; a prediction that
swaps which call errored (B errors, A succeeds) produces `error_predicted=True,
error_recorded=True` and scores correct, despite attributing the failure to the wrong call
entirely.

## Decisions

### D68. A contract field must not reach an artifact by default. **Decided**

The four leak sites share one defect: `model_dump()` is an *open* serializer, so the
blast radius of adding a field to `ToolSchema` is "everywhere it is dumped," discovered
only where a test happens to pin the shape.

Each site declares the projection it needs. Where an exported shape is a published
contract (SFT rows, eval cases, analysis evidence), the projection is explicit and its
own test pins it. `exclude_none` is rejected as the fix (E28): it is another implicit
rule, and it silently drops `description`.

This is the serialization analogue of D30. `CandidateView` is a declared type precisely
so a new `Scenario` field cannot leak by default; the same reasoning applies to every
boundary where a contract becomes an artifact.

### D69. Re-serializing an existing artifact kind is a migration, not a patch. **Decided**

`analyze/tasks.py:72` puts the tool dump in `available_tools` Evidence, and
`analysis.py:15` hashes `model_dump_json()` into the analysis id. Changing that
payload changes analysis identity for every re-analyzed corpus.

That is the same class of event as D38's control-marker re-ingest, and takes the same
handling: the new id is correct rather than a problem, the legacy artifact is preserved
as provenance, and the delta is recorded. It is not a thing to be discovered later from
a hash that moved.

### D70. Every submitted call requires an explicit outcome. **Decided**

From E29. A batch is accepted only when the set of outcome call IDs equals the set of
submitted call IDs. A call the AWM cannot answer is stated as `executed=False`, which
already validates today — silence is not an available encoding.

The zero-outcome fall-through is closed in the same change and is the more urgent half:
no branch may derive `executed_call_ids` from the submitted call list. Execution is a
claim the AWM makes, never one the validator makes on its behalf. This is D53 stated as
an enforced equality rather than an intent, and it is what D53 already meant.

### D71. The fidelity gate scores accepted transitions. **Decided**

From E30. A gate that scores proposals the runtime would reject measures a simulator
that will never run. `fidelity.py` routes proposals through `validate_transition` and
reports the validation-rejection rate as a first-class gate output beside accuracy and
coverage.

This follows D54's logic: accuracy on the subset the AWM chose to answer cannot
establish validity, and neither can accuracy on the subset the runtime would throw away.

### D72. Input validation is held to the output standard. **Decided**

From E31. Candidate arguments validate against the declared input schema with
`jsonschema`, as outputs already do under D63. Until then, the residual is stated in
the report rather than left implied by D62's wording: a call may satisfy validation and
still violate its declared contract.

### D73. The real-artifact smoke test is a gate deliverable, not a manual step. **Decided**

From E32. A test compiles `corpus-ee3b33086ef177d7`, builds the fit-only index, and
asserts the counts v2 measured — 371 transitions, all observed, 14 batched actions,
calls-per-action `{0:189, 1:168, 2:10, 4:4}` — skipping cleanly when `work/tau/` is
absent so CI needs no fixture.

Those numbers are already the evidence base for D39 and E26. Leaving them unpinned means
the next contract change moves them silently, which is exactly the failure E27 just
demonstrated on the export side.

### D74. Verification status is set by execution. **Decided**

v3 recorded S6 as "verification intentionally not run yet" and I8 as
`IMPLEMENTED; AWAITING VERIFICATION`. That was honest. The lesson from running it is
that the deferred run found four leak sites and two accept-path defects that the
implementation session believed were closed.

No issue moves to `FIXED` on implementation. It moves on a named executed check. I8
stays open until D70 and D71 land with tests.

### D75. A parseable-but-malformed answer is a distinct outcome from abstention. **Decided**

From E33/E34. `ProposedTransition` gains `output_invalid: bool` and
`output_invalid_errors: tuple[str, ...]`. `_coerce` returns three distinguishable things:
`None` only when nothing was returned at all (empty/absent payload); the parsed contract on
success; a new `_CoerceFailure` carrying the raw validation errors when a real, non-empty
payload existed but did not fit the contract (invalid JSON, wrong shape, or a schema
mismatch). `step_tool_world` and `validate_transition` both check `output_invalid` before
`abstain`. `TransitionFidelity` and `FidelityReport` carry the same split:
`model_output_invalid_rate` is now a first-class metric, disjoint from
`abstention_rate`/`wrong_abstention_rate` and from `validation_rejection_rate` (an
output-invalid row never reaches `validate_transition` and does not set
`validator_rejected` — that flag means the runtime specifically refused a parseable
proposal, which did not happen here).

This is the same shape as D27: an abstention is honest behavior the design asks for, and
counting a parser defect against it would teach the wrong lesson twice — once by hiding the
defect, once by penalizing legitimate declining as if it were the defect.

### D76. Fidelity compares the validator's canonical, call-correlated result — never a field the scorer re-derives. **Decided**

From E37/E38. `ValidationOutcome` gains `committed_observations: dict[str, Any]`, keyed by
call_id (or tool name only when the sole submitted call has no call_id). Both the batch
(`call_outcomes`) and aggregate (single-call top-level) forms normalize into this same
mapping inside `validate_transition`, so a caller never re-derives which model field answers
which call. `score_transition_fidelity` correlates each recorded `GroundingObservation` to
its `committed_observations` entry by `tool_call_id`/`call_id` (falling back to positional
pairing only in the unambiguous single-call, single-observation, single-committed-result
case — real tau2 data frequently omits `tool_call_id` even for unbatched calls). A batched
action reaching the aggregate top-level form with more than one call is rejected outright
(new validator check): attributing an unlabeled aggregate observation to `calls[0]` would be
a guess the design forbids elsewhere (D53/D70).

Status correctness (`status_correct`) is computed per correlated call and required to hold
for every pair, not aggregated across the batch — E38's swapped-error case is exactly what
an aggregate `any()` comparison cannot catch. The recorded side of a correlated pair keeps
the full `GroundingObservation`, not just its `.content`: a call flagged `error=True` whose
payload does not itself look like an error (no `"error"` key, no `"error"`-prefixed string)
must still score as an error, driven by the recorded provenance rather than payload shape
alone.

Incomplete correlation — any submitted call whose recorded observation could not be paired —
makes the *entire* transition's field/status accuracy unscorable (`fields=()`,
`status_correct=None`), not a partial score over whichever calls happened to match. A
two-call batch where one call correlates perfectly must not report `field_accuracy=1.0`
while the other call goes unscored and invisible; `unmatched_call_observations` names which
calls are missing so the gap is visible rather than silently shrinking what "1.0" claims to
cover.

### D77. `inferred_state_delta` carries an explicit ground-truth status, and reports only what before/after comparison actually established. **Decided**

From E35/E36. `compile._delta()` now diffs a reaction's reported fields against
`state_before`, entity-prefixed the same way `reconstruct_state` prefixes state (correlating
each reaction to the specific call it answers by `tool_call_id`, falling back to the sole
call of that tool only when unambiguous — never a bare positional guess). A field counts as
a confirmed mutation only when the pre-action value was known and differs from the
post-action value.

`GroundingTransition` gains `delta_ground_truth_status: DeltaGroundTruthStatus` (`measured`,
`unavailable`, `not_applicable`) and `unmatched_post_paths: tuple[str, ...]`. An empty
`inferred_state_delta` is ambiguous on its own — "verified no-op" and "could not compare
anything" are different facts — so the status makes the distinction explicit rather than
collapsing both into `{}`:

- `measured`: at least one field was compared and none were left unmatched. A `measured`
  delta with zero entries is a genuine verified no-op.
- `unavailable`: some reported field could not be aligned to any `state_before` path
  (typically a cross-tool prefix mismatch, E36) — conservatively applied even when other
  fields in the same reaction WERE confirmed changed, since a partial delta is not the same
  claim as a complete one, and the unmatched field might be exactly the one that changed.
  Also applied when a reaction reported real structured content but could not be correlated
  to any submitted call at all (an ambiguous batch, e.g. two calls to the same tool with no
  `tool_call_id` on either side) — mutation evidence existed, it simply could not be used,
  which is a different fact from no evidence existing.
- `not_applicable`: no reaction reported any comparable structured content at all.

`delta_correct` and `multi_step_drift`'s per-step `expected` state are both gated on
`delta_ground_truth_status is MEASURED`; an `unavailable`/`not_applicable` row contributes
`delta_correct=None` (excluded from `delta_accuracy`) rather than being scored against an
empty dict that looks identical to a real no-op. `multi_step_drift` additionally scores only
the paths `expected` has an opinion on for that step — `state.fields` is cumulative across
all prior steps, and scoring the union with `expected` penalized a predictor for any
self-consistent path `expected` was silent on, including that step's own unmeasurable delta.
`FidelityReport` gains `delta_ground_truth_coverage`, reported beside `delta_accuracy` always
— on the pinned family this measures `0.0`: delta fidelity is not yet a usable metric here,
which is the honest result E36 established, not a bug to paper over.

### D78. `predicted_delta` compares the validator's committed result, not the proposal's raw aggregate field. **Decided**

A narrower instance of D76's principle, called out separately because it was found and
fixed after D76 landed: `score_transition_fidelity` computed `predicted_delta` from
`proposal.state_delta`, the same top-level aggregate field E37 already identified as
deliberately empty under the batch form. `predicted_delta` now reads
`validation.committed` — the validator's own flattened, accepted result — so `delta_correct`
compares the same representation for both the batch and aggregate encodings.

### D79. Resolves Q21: identifiability is assessed at scoring time, from evaluation demand read out of the recorded result. **Decided**

Q21 asked whether a transition's "identifiability classification" (E39's provisional
`grounding` taxonomy) should be a compile-time property of the transition, like
`delta_ground_truth_status`, or a scoring-time property of what a specific query's retrieval
actually returned — since the same transition could be identifiable under one retrieval
configuration and not another.

**Decided: scoring-time, in two parts, computed by `bandits/emulate/grounding.py`'s
`assess_grounding`.**

- **Evaluation demand** comes from the recorded result: for held-out fidelity scoring, the
  transition's actual `GroundingObservation.content` is flattened into field paths
  (`required_fact_paths`). This is safe only because those values are read to determine
  *what the prediction would have needed to know*, never shown to the predictor being
  scored — the same non-leakage boundary `score_transition_fidelity` already holds for
  `recorded` generally. Runtime (no recorded answer to peek at, per the earlier
  AWM-centered-runtime discussion this issue is deliberately not building) needs a
  tool-contract-derived demand instead; out of scope here.
- **Identifiability** is assessed against the actual pre-action inputs a given call would
  see — `state_before` plus whatever was actually retrieved for that query — not stored on
  `GroundingTransition` itself. The same transition run through a different retrieval
  configuration, or scored earlier/later in a trace where more or less state had accumulated,
  can legitimately get a different verdict; hard-coding one verdict onto the transition would
  hide that dependency rather than measure it.

This does not extend to writes, batches, or tools outside `grounding.py`'s reviewed
`ENTITY_TOOLS` mapping (`get_user_details`, `get_reservation_details`) — those report
`GroundingKind.UNAVAILABLE` and are scored by the prior `SupportLevel`-based rule, unchanged.
Extending the taxonomy to mutations (where `DERIVABLE_FROM_STATE` is reserved but not yet
emitted) and to the agentic AWM-centered runtime's own grounding-tool calls
(`read_world_state`/`search_transitions`/etc.) is separate, unscoped future work — see I39's
closing note in `awm-issues.md`.

**Verification.** See I39's closing note in `awm-issues.md` for the executed test/rescore
commands and results (946 passed full suite; 32/32 focused; real-artifact rescore matches the
predicted outcome exactly; rescore is reproducibly byte-identical on a second run).

## Issues opened

- **I31** (P0): incomplete and zero-outcome batches accepted; execution fabricated from
  the call list. E29 → D70.
- **I32** (P1): fidelity scores unvalidated proposals. E30 → D71.
- **I33** (P1): `ToolSchema` fields leak into four artifact payloads via bare
  `model_dump()`. E27/E28 → D68, D69.
- **I34** (P2): no real-artifact tau2 smoke test. E32 → D73.
- **I35** (P0): `_coerce` cannot parse a real `dspy.Prediction`; every real model answer
  silently read as abstention. E33 → D75.
- **I36** (P0): malformed structured output collapses into abstention, hiding a
  prompt/parser defect behind a metric that looks epistemic. E34 → D75.
- **I37** (P0): fidelity's observation/status comparison used the wrong (uncorrelated,
  aggregate-only) representation, and batch status aggregated across calls instead of
  comparing per call. E37/E38 → D76, D78.
- **I38** (P1): `inferred_state_delta` was every reported field, not a diff, and even after
  diffing, cross-tool path prefixes leave the pinned family's delta ground truth entirely
  unmeasurable. E35/E36 → D77.

I8 remains **OPEN** (was `IMPLEMENTED; AWAITING VERIFICATION`). Its output-schema,
event-provenance, and cross-call-matching claims are verified and hold; its per-call
completeness claim does not (E29).

## Resolutions (S10, ratified)

**Q13 resolved: keep explicit nulls.** The published shape stays
`{name, description, parameters}`, nulls included. `ToolSchema.offered_projection()`
serves SFT, eval, and analysis evidence; `simulation_projection()` adds `output_schema`
for the simulator alone. `exclude_none` is rejected -- E28 measured that it also drops
`description`, so it would have broken the same test from the other side.

**Q14 resolved: pin the legacy ids.** The explicit legacy projection means the analysis
payload does not change, so no migration is triggered by a serialization fix. Introducing
output schemas into analysis evidence is a deliberate Gate 0 migration when it happens:
preserve the legacy artifact, re-ingest and re-analyze, record old id -> new id with the
contract-version delta, and re-mine dependent task sets. D69 stands; it is simply not
triggered yet.

**Q15 resolved: input validation is a Gate 0 blocker.** Candidate calls are deterministic
inputs. An invalid nested argument, enum, array, or `oneOf` branch reaching the AWM asks
the simulator to invent behavior for a call the tool contract rejects. The four outcomes
are distinct: input failure is `INVALID_CANDIDATE_ACTION`, output failure is
`INVALID_TRANSITION`, an invalid declared schema is a preflight/configuration failure, and
a missing schema stays `unavailable`. It does not block compilation or static probes; it
blocks simulator-backed campaigns.

**I32 raised to P0-for-campaigns.** Until fidelity exercises `validate_transition`,
passing the fidelity gate does not validate the simulator rollouts actually use.

## S11 — closures, each by an executed check (D74)

Full suite: **919 passed, 0 failed** (from 897/1). Ruff clean.

### I31 closed — outcome completeness, and a corrected boundary

`validate_transition` now requires submitted call IDs == outcome call IDs, and no branch
derives execution from the call list.

One correction found while fixing it, worth keeping: the first patch banned the aggregate
single-call form outright, which broke three existing tests. That form is sanctioned by
D67 ("aggregate single-call effects follow the same rule"), so the ban was wrong. The
actual defect is narrower than E29 suggested: a single call may still state its result as
the proposal's own observation/delta/events, but a proposal claiming **nothing at all**
cannot have its execution inferred. D70 is therefore about silence, not about the
aggregate encoding.

Both E29 repros now reject. Tests: `test_a_batch_missing_an_outcome_is_refused`,
`test_an_unanswerable_call_is_stated_not_omitted`,
`test_execution_is_never_inferred_from_the_submitted_calls`,
`test_a_single_call_claiming_nothing_at_all_is_refused`.

### I33 closed — the projections live on the contract

Both projections are methods on `ToolSchema` rather than repeated at each call site, so
the rule travels with the contract that has the fields. All four leak sites converted.
The originally-failing export test passes unchanged, which is the compatibility claim.

### D72 closed — one validator, both directions

`_validate_output` was never output-specific; it is now `validate_instance(..., label=)`
and `_argument_errors` calls it. The hand-rolled required/primitive-type loop is gone.
Enum, nested-object, and array-item violations now terminate as
`INVALID_CANDIDATE_ACTION`; a valid call under a rich schema still reaches the world.

### I32 closed — the gate scores accepted transitions

`score_transition_fidelity` routes every proposal through `validate_transition` and
returns a rejected row with **no accuracy fields populated**: a refused prediction has no
standing to be called correct, and filling them in would let it contribute to the accuracy
the gate reads. `validation_rejection_rate` is a reported and gateable output.

`multi_step_drift` had the same defect and worse consequences -- it commits each
prediction into the state the next step reasons from, so an invalid transition
contaminated every later horizon. It now refuses to carry an unvalidated prediction
forward.

### I34 closed — the real artifact is pinned

`bandits/emulate/tau_smoke_test.py` runs against the stored corpus and reproduces v2's
measurements exactly: **371 transitions, all observed, 14 batched, calls-per-action
`{0:189, 1:168, 2:10, 4:4}`**, 189 user-role reactions, and E26's zero-error finding
across the family. It also asserts the live facts behind D38 (the corpus still declares
`control_markers == ()`), D29 (markers stripped, not dropped), and D28 (held-out lineages
excluded by the index's own filter, not by the test). Skips cleanly without `work/tau/`;
verified by running it from a directory that has none.

## Status

I31, I32, I33, I34 closed. **I8 closed**: its per-call completeness claim, the last
outstanding half, now holds under D70.

D72 was the stated Gate 0 blocker and is closed. Per S10's condition -- no real
AWM/candidate experiments until I31, D72, and I32 are closed by executed checks -- that
condition is now met.

## S12 — first real-model fidelity run, ten defects found and fixed by running it

`scripts/run_awm_fidelity.py` is the thin orchestration layer D65 deferred: load the pinned
tau2 family, build the fit-only index, select a stratified held-out sample, call a real
Fireworks-backed AWM through `build_tool_world_predictor`, score with
`score_transition_fidelity`, checkpoint every raw model response/proposal/validator result,
and print the aggregate report. No GEPA, no candidate rollouts -- the first question was
narrower: does the real-model path even function, and is one real prediction trustworthy.

It was not, four times over, before it was. Two live single-transition runs (Nemotron
Lightning 3.5, then DeepSeek V4 Flash 0731) against the real `cancel_reservation`
transition `transition-06c1350d8743` both scored `abstained=true` despite the raw model
response showing `abstain=false` -- E33. Fixing that exposed E34 (malformed output still
read as abstention), which exposed a genuine model answer whose *content* was already
correct -- DeepSeek's observation matched the recorded cancellation on all 36 compared
fields, including the reversed gift-card refund -- but which the fidelity scorer still
reported as `field_accuracy=0.0` for an unrelated reason (E37: comparing the wrong
representation). Fixing E37 surfaced E38 (batch status aggregation) and prompted an audit
of `predicted_delta` that found D78's instance of the same class of bug. In parallel, D77's
work on `inferred_state_delta` (E35/E36) found that entity-path prefixes make delta ground
truth currently unmeasurable for the whole pinned family -- a real, reportable limit rather
than a code defect.

Every fix was verified against the real corpus or replayed against the real saved model
response (`work/awm-fidelity/baseline-fixed2.raw.jsonl`, gitignored scratch output, not
committed) -- zero additional model spend beyond the two live calls. Final replay after all
six fixes: `status_correct=True, field_accuracy=1.0, output_invalid=False,
validator_rejected=False, unmatched_call_observations=(), delta_ground_truth_status=
UNAVAILABLE, delta_correct=None`. That is the first trustworthy real-model fidelity
observation this project has produced.

D75-D78 land with the tests in `world_test.py`, `fidelity_test.py`, and `compile_test.py`
listed in their entries. Full suite: **932 passed, 0 failed** (`uv run pytest -q`, no
targeted subset). I35-I38 closed by the same executed checks (D74's rule).

Held open, deliberately not attempted this session: a reviewed tau2 entity-path
canonicalizer (D77 named the alternative and rejected inventing one unreviewed), and
user-policy output-invalid taxonomy (`ProposedUserTurn` still collapses a failed parse into
abstention -- noted inline in `step_user_policy`, out of scope for the tool-world work this
session did).

### E39. The ten-transition stratified run mostly tested record materialization, which no retrieved evidence in this family can support.

Ran (same DeepSeek model, same `.raw.jsonl`/manifest checkpointing, resumed from the cached
E37/E38 result): 10 held-out transitions, stratified across `cancel_reservation` (1, cached),
`get_reservation_details` (3), `get_user_details` (3), `transfer_to_human_agents` (3).

Raw counts: `field_accuracy=1.0` and `status_accuracy=1.0` in the aggregate report, but both
are computed over denominators far smaller than "10" suggests --

```text
fields compared at all         : 1 transition  (the cached cancellation, 36/36 correct)
status compared (non-abstain,
  non-output-invalid)          : 4 transitions (1 cancel + 3 transfers, all fieldless)
output_invalid                 : 4 transitions (all get_reservation_details/get_user_details)
abstained                      : 2 transitions (both get_reservation_details/get_user_details)
```

Inspecting the four `output_invalid` rows' raw model responses (not just the aggregate rate)
found they are not garbled output: each is `abstain=true` with a coherent, correct-sounding
reason ("No evidence for user emma_kim_9957; cannot ground the tool's output"), paired with a
`call_outcomes` entry stating `executed=false, observation=null` -- the model trying to be
maximally structured about *why* it declined. `ProposedTransition.abstention_proposes_nothing`
rejects any non-empty `call_outcomes` alongside `abstain=true`, with no exception for an
all-null/not-executed placeholder, so `_coerce` correctly (per D75) classifies these as
`output_invalid` rather than salvaging them -- but the *underlying model behavior* was a
reasonable abstention, misshaped by the contract, not noise.

Inspecting the two clean-abstention rows (`get_reservation_details` for `Q69X3R` and
`get_user_details` for `raj_sanchez_7340`) and the four output-invalid rows' `expected_observation`
against what was actually retrieved: every one of these six read transitions asked the AWM to
reproduce a specific entity's private record -- e.g. `get_user_details("emma_kim_9957")`
expected Emma Kim's exact name/address/email/DOB/payment methods, none of which appear
anywhere in `state_before` or in the retrieved examples (which were `get_user_details` calls
for *other* users). No amount of behavioral pattern-matching over other users' records can
recover Emma's specific values; declining was the only honest answer available. `support_level()`
(`retrieve.py`) awards `"high"`/`"medium"` support from `exact_tool` matches -- the retrieved
example called the *same tool* -- without checking whether the retrieved example concerns the
*same entity* the transition under test asks about. `abstain_correct`/`wrong_abstention_rate`
inherit that blindness: a correct "I don't have this specific record" reads as a wrong
abstention because generic same-tool evidence existed, even though no evidence about *this*
entity existed.

By contrast, the cached `cancel_reservation` transition is a different task shape entirely:
`state_before` already held the target reservation's full details from an earlier
`get_reservation_details` call *in the same trace*, so scoring it required transforming known
state under a recorded action, not materializing an unknown record. That is the transition
that scored 36/36 -- it is not representative of what the other nine transitions asked for,
and averaging it into one `field_accuracy=1.0` headline is misleading on its own.

The `transfer_to_human_agents` rows (3/3 "valid," `status_correct=true`) carry almost no
evidentiary weight: this tool's transitions have no observation fields to diff at all
(`fields=()` for all three) -- passing tells us only that the model predicted the right
coarse success/error shape for a call with no payload, not that any fact was reproduced
correctly.

One `max_tokens=4000` truncation warning fired during the run, log-order-attributable to the
5th call (`transition-233d53079e5b`'s `get_reservation_details` invocation) with high
confidence, but the persisted final response for that transition is short and well-formed --
not visibly truncated. Whether an earlier internal completion attempt was the one that hit
the limit, and whether DSPy retried before the value that was actually saved, cannot be
established from what the runner persists today (only the final `dspy.Prediction`, not
provider/DSPy attempt history). Recorded as unconfirmed rather than asserted either way.

### Provisional direction from E39 — not yet a ratified decision

Two structurally different things are currently conflated under one `support_level()`/
`abstain_correct` calculation, and likely need to stay separate rather than be merged into a
single "entity match" fix:

```text
grounding =
    exact_entity_fact        (the specific record/value is in state_before or a retrieved
                               example that names this exact entity)
  | derivable_from_state      (the expected result is a transformation of an already-known
                               entity's state under a recorded-shape action -- the
                               cancellation case)
  | behavioral_analogy        (same tool, different entity -- teaches output SHAPE/semantics,
                               never a specific field VALUE)
  | unsupported
```

Only the first two can license a fidelity claim about specific field values. `behavioral_analogy`
can support scoring transition *semantics* (does the tool's shape/error/effect pattern look
right) but must not be read as grounds to expect, or to score against, an exact reproduced
value -- which is what today's `exact_tool`-only support check effectively does.

This is deliberately left provisional. Before it becomes a D-series decision: an explicit
identifiability classification per transition (can the expected result actually be derived
from what the AWM was given, independent of what it produced) is needed first, and
`abstain_correct`/`wrong_abstention_rate` should be conditioned on that classification rather
than on retrieval support alone. Not implemented or re-run this session.

## Open questions

- **Q16.** `validation_rejection_rate` is gateable but has no threshold, like every other
  bar in `gate()`. Does it get one from Gate 2 calibration, or is any rejection
  disqualifying?
- **Q17.** Input validation now blocks campaigns. Static probes deliberately still accept
  contract-violating calls -- should a static report surface that count, since "the
  candidate emits invalid arguments" is itself a capability finding (the D9 route table's
  `tool-call SFT` row)?
- **Q18.** The smoke test pins counts from one family. Does Gate 0 extend it to the other
  15 mined families, or is one family's fidelity to the artifact sufficient?
- **Q19 (updated post-E39).** Delta ground truth was `unavailable`/`not_applicable` for
  100% of both the single-transition and ten-transition runs (D77/E36). A reviewed tau2
  entity-path canonicalizer remains unbuilt and unreviewed (D26/D27 fail-closed rule); the
  report carries `delta_ground_truth_coverage=0.0` as an honest, standing limitation rather
  than a guessed canonicalization. Still open: is this worth building at all before the
  identifiability work in E39's provisional direction, given delta fidelity and
  record-materialization identifiability may turn out to need overlapping machinery
  (both are, at root, "does the AWM's input actually determine this entity's specific
  field values")?
- **Q20.** `ProposedUserTurn`/`step_user_policy` still has no `output_invalid` distinction
  (D75 covers only the tool-world side). Does the user-policy half need the same taxonomy
  before Experiment 6 (user-policy fidelity) runs, given 189/371 of this family's
  transitions are user replies?
- **Q21 (from E39). RESOLVED by D79.** Identifiability is assessed at scoring time
  (`bandits/emulate/grounding.py::assess_grounding`), from `state_before` plus whatever was
  actually retrieved for that call's query; evaluation demand comes from the recorded
  result's field paths. See D79 for the full resolution and I39's closing note in
  `awm-issues.md` for the executed verification.
- **Q22 (from E39).** `abstention_proposes_nothing` (the `ProposedTransition` validator
  under D75) rejects an abstention that carries an all-empty/`executed=false` placeholder
  `call_outcomes` entry, even though it asserts no content. Four of ten real transitions hit
  this exact shape. Is the fix a narrower validator rule (only reject `call_outcomes` entries
  that assert actual content: non-null observation, non-empty deltas/events), a prompt
  change (tell the model `abstain=true` requires `call_outcomes=[]`), or both? Deliberately
  not decided or implemented this session -- see E39's note that fixing this reclassifies
  four `output_invalid` rows as clean abstentions but does not, by itself, make the AWM able
  to answer the underlying lookups.
- **Q23 (from E39).** The `max_tokens=4000` truncation warning during the ten-transition run
  is attributable to one call by log order but not confirmed against the actual persisted
  response, which shows no visible truncation. Is it worth persisting DSPy/provider attempt
  history (not just the final `dspy.Prediction`) in the runner's checkpoint so future
  truncation attribution is exact rather than inferred from log ordering?
