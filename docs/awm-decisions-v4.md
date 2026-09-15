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

## Evidence

### E27. The suite is green except for one export test, and that test is the tip of four leak sites.

Executed: `uv run pytest -q` → **897 passed, 1 failed**. Focused external-validation
set (107 tests) passed; `bandits/diagnose` (227 tests) passed; ruff clean.

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
bandits/diagnose/compile.py:481 -> scenario offered_tools
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

The only occurrence of `corpus-ee3b33086ef177d7` under `bandits/diagnose/` is a
docstring in `compile_test.py:40` recording D36's provenance. Every diagnose test runs
on fixtures. D36 said a fixture is a claim about the data and an unverified claim is as
wrong as unverified code — that rule is currently satisfied by comment, not by a test.

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

## Issues opened

- **I31** (P0): incomplete and zero-outcome batches accepted; execution fabricated from
  the call list. E29 → D70.
- **I32** (P1): fidelity scores unvalidated proposals. E30 → D71.
- **I33** (P1): `ToolSchema` fields leak into four artifact payloads via bare
  `model_dump()`. E27/E28 → D68, D69.
- **I34** (P2): no real-artifact tau2 smoke test. E32 → D73.

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

`bandits/diagnose/tau_smoke_test.py` runs against the stored corpus and reproduces v2's
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
