# AWM / traces2rlenv — decision record, v2

Continues [reasoning-awm-decisions.md](reasoning-awm-decisions.md) (D1–D26). That
record settled the architecture; this one runs from first code to a working slice and
records every decision, question and measurement taken along the way, in order.

Same rules as v1. Each entry is **Decided**, **Provisional** (acting on it; a named
measurement can overturn it), or **Open**. Nothing here is ratified by the user unless it
says so — the record exists so choices can be overturned cheaply, not so each one waits
for approval.

Numbering continues: decisions D27+, evidence E17+.

---

## Session log

| # | What | Outcome |
|---|---|---|
| S1 | Confirm grounding-loop description | Accurate; three corrections → D27, D28, D29 |
| S2 | Measure tau2 error-transition density | E17/E18 — the finding that reshapes Gate 2 |
| S3 | Write `bandits/diagnose/models.py` + tests | Shipped; 36 tests, suite green |
| S4 | Decisions taken while writing models | D30–D33 |
| S5 | Refusal-state seeding question | D34 |
| S6 | Check tau2 `initial_state` availability | E20 — 0/50; D34 revised to its fallback |
| S7 | Read tau2 `info.user_info` | E21 — the user simulator's own guidelines recovered |
| S8 | Write `compile.py` + tests | Shipped; 73 tests, suite green |
| S9 | Run `compile.py` on the real corpus | E22/E23 — two defects the fixtures hid; D36, D37 |
| S10 | Write `verify.py` + tests | Shipped; the rollout→verifier join, 103 tests |
| S11 | Review corrections on D27/D34/transitions | E25 — multi-call actions measured; D34 reversed, D39 |
| S12 | Write `retrieve.py` + tests | Shipped; E26 — every tool has zero error evidence; D40, D41 |
| S13 | Write `world.py`, `rollout.py`, and `candidates.py` | Shipped; grounded interaction loop and controls |
| S14 | Write `fidelity.py`, `optimize.py`, and `report.py` | Shipped; subsequently audited in `awm-issues.md` |
| S15 | Post-build correctness audit | Superseding repairs and new decisions continue in `awm-decisions-v3.md` |

---

## 1. Is the grounding-loop description accurate?

Yes. Retrieval supplies enterprise facts, the external ledger supplies cross-turn
consistency, GEPA supplies instructions, fidelity evaluation proves the grounding works,
and no weights are trained. The pipeline (existing artifacts → cut points + lossless
transitions → grounding index) is right, and `/diagnose` consuming rather than re-extracting
is right.

Three corrections follow, one of them serious.

### D27. Mine *policy constraints*, not tool refusals. Unsupported tool counterfactuals abstain. **Decided; earlier form corrected**

The description assumes retrieval can return "refused/error examples" and "contrasting
examples showing when the same action should behave differently." Measured against the
corpus, it mostly cannot.

**E17.** Across all 16 mined families, 1212 tool results contain **26** beginning with
`Error:` — 2.1%. Twenty of the 26 are `get_user_details` not-found.

**E18.** `cancel_reservation`, the target family's central write tool, was called 26 times
across the corpus and produced **zero** recorded errors. The target family's own 36
trajectories contain one error-shaped result in 204.

This is what the data *is*, not a gap in it: tau2's agents were competent, so the corpus
records the happy path. The decision-relevant transitions — ineligible cancellation, double
cancellation, a nonexistent reservation — are exactly what a weak candidate produces, and
retrieval has nothing to return for any of them.

**The correction to the first form of this decision.** It proposed mining refusals from the
policy text as evidence that *the tool refuses*. That conflates two different things:

```text
what the agent is permitted to do      (policy)
what the tool does when called anyway  (tool semantics)
```

tau2 contains tasks where the agent must refuse **even though the tool would execute the
cancellation** — the whole point of those tasks is that the guardrail is the agent's, not
the API's. Treating a written policy as evidence about tool behavior would build a
simulator that refuses on the tool's behalf, and a candidate that improperly calls
`cancel_reservation` would be *rescued by the environment* instead of failing the verifier.
That inverts what the diagnosis is measuring.

**Decision, in three parts:**

1. **Policy text yields constraints, not transitions.** It produces `ForbiddenEffect`s,
   communication requirements, and policy-violation checks on the *verifier* side. It may
   never ground a committed tool observation.
2. **Assistant-stated refusals are agent self-report.** A recorded assistant declining on
   policy grounds is evidence about what that agent believed, ranked accordingly. It cannot
   ground a tool observation either.
3. **A genuinely unsupported tool counterfactual abstains.** Nonexistent ids, double
   cancellation, ineligible mutation: absent a source of tool semantics, the environment
   declines rather than inventing either a success or a refusal. The abstention rate is
   reported per tool and per effect class before any fidelity number is trusted.

Where the improper call *is* supported, the simulator should execute it successfully and
let the verifier mark the candidate's behavior as the failure it was.

**Q9 is resolved strictly** by the above. The cost — rollouts abstaining on many
interesting actions — is real and is now a measurement rather than a guess: Gate 2 reports
it, and only a measured abstention rate justifies loosening.

### D28. The retrieval filter list needs one more entry: partition. **Decided**

The described filter excludes held-out traces, same-lineage examples, incompatible shapes,
incompatible tools/effects, and marker-contaminated content. It omits the sealed partition.

Transitions from `Partition.SEALED` (the 40 reserved tau tasks) must never be retrieval
sources and must be excluded from GEPA and prompt selection. A sealed scenario may query
the fit-only index during the one final audit; otherwise it could not run. Held-out and sealed
are different exclusions with different reasons, and naming only held-out is how the sealed
set quietly becomes a development set. Asserted at query time, not only at index build.

### D29. "Control-marker-contaminated content" is filtered *and* transformed, not only filtered. **Decided**

Filtering marker-carrying content out of retrieval would discard a large share of tau2's
user turns, since `###TRANSFER###` appears on most airline episodes rather than only
escalating ones. The correct handling is the one `GroundingTransition.stripped_markers`
already implements: strip the marker for the AWM-facing view, record that it was stripped,
keep the transition. Filtering is the fallback for content where the marker cannot be
cleanly separated.

---

## 2. Decisions taken while writing `models.py`

Shipped: [bandits/diagnose/models.py](../bandits/diagnose/models.py) and its tests. 36
tests, full suite green, ruff clean, no credential needed in CI.

### D30. `CandidateView` is a type, not a dict returned by a method. **Decided**

The candidate/environment boundary is the leak that inflates every score downstream, and a
`dict` return makes leaking the default failure mode: adding a private field to `Scenario`
would silently start including it. A declared type means anything the candidate sees had to
be given a home deliberately.

`test_candidate_view_carries_no_private_field` asserts the view's fields are a strict subset
of the scenario's, so the test fails if a future field appears on both.

### D31. `ToolEffectCatalog` has no naming fallback whatsoever. **Decided**

D26 said the effect class comes from reviewed metadata, "not a `get_*` naming heuristic."
Checking the data made that stronger than a preference:

**E19.** Every tool in the tau2 toolset declares `{"name", "parameters"}` and nothing else —
the conversion dropped all 13 `description` fields. There is no source metadata to classify
from at all.

So a `get_*` rule would not have been a heuristic *layered over* metadata; it would have
been the only input, while looking like metadata was consulted. An unreviewed tool returns
`ToolEffect.UNKNOWN` and counts toward neither operational success nor forbidden-effect
checks. Classification requires a named reviewer, enforced by validator.

### D32. `NOT_APPLICABLE` is distinct from `UNKNOWN`, and does not block a pass. **Decided**

An informational task owes no operational result. Reporting that as `UNKNOWN` would say a
measurement was attempted and failed, which would then fail the verdict closed — and every
read-only task in the family would report unknown forever.

`NOT_APPLICABLE` means nothing was owed. It does not block, which is what lets a read-only
task pass on communication alone. The degenerate case is closed separately:
`test_a_pass_needs_something_that_actually_passed` refuses an overall pass where every
component is not-applicable.

### D33. Budget and loop terminations are candidate failures, not abstentions. **Decided**

`TerminationReason.invalidates_rollout` is true only for `UNSUPPORTED_ACTION`,
`AWM_ABSTAINED`, and `INVALID_TRANSITION` — cases where *the environment declined to
answer*. `STEP_LIMIT`, `TOKEN_LIMIT`, `TIME_LIMIT` and `ACTION_LOOP` are real failures: the
environment answered every time and the candidate did not finish.

The distinction is load-bearing in both directions. Counting an abstention as failure
charges the candidate for the simulator's ignorance; counting a step-limit as an abstention
lets a candidate escape a bad score by looping until the budget dies.

---

## 3. Refusal-state seeding

### D34. Seed refusal-relevant state paths from the source contract. **SUPERSEDED by the reversal in section 6**

A refusal contract asserts that a reservation is *still* confirmed at termination. If the
candidate never looked the reservation up, the ledger has never heard of that path, and the
check returns unknown — so a candidate that correctly refuses by doing nothing at all scores
unknown rather than pass. That is the wrong answer for the shape that most needs to be
scorable.

**Decision.** At compile time, paths named by the contract's `ForbiddenEffect.state_path`
are seeded into `initial_state` from the source task's own initial state, with
`origin=RECORDED`. They are environment-private: seeded state is not in `CandidateView`, so
this does not tell the candidate anything.

**The leak this accepts, stated plainly.** The ledger then knows about entities the
authentic prefix never revealed. It is not a candidate-visible leak, but it does mean
`ScenarioState` no longer contains *only* what the prefix showed — and `Scenario`'s current
validator enforces exactly that, so this decision requires relaxing it to permit
contract-seeded recorded fields, distinguishable by a flag.

**Rejected: fail closed to UNKNOWN.** Honest, but makes the refusal shape nearly unscorable,
which defeats D25's reason for keeping refusals at all.

**Resolved against the primary mechanism.** Checked immediately:

**E20.** **0 of 50** tau2 tasks carry a non-null `initial_state`. `ticket` is null on all
50, and `env_assertions` is null on all 50. The only populated contract fields are
`actions`, `nl_assertions`, `communicate_info` (6/50) and `reward_basis`.

So there is no source initial state to seed from, and the decision reduces to its fallback:
a forbidden-effect path is seeded from the sealed contract's own `must_remain` value, which
the reviewer supplies when binding the refusal template. That is a *reviewed human
assertion*, not a recorded observation, so it cannot be marked `origin=RECORDED`.

**Revised decision.** `WorldOrigin` gains no new member — the axis is about which world
produced the fact, and a reviewer's assertion about the real world is `RECORDED` in origin
while being weaker in *authority*. The seeded field therefore carries
`origin=RECORDED` with `authority=HUMAN_LABEL` on the claim it produces, and a
`seeded_by_contract` flag on `StateField` so the scenario validator can permit it while
still refusing simulated fields in initial state. A refusal verdict resting on a seeded path
is reported as resting on a reviewed assertion, not an observation.

**Consequence for the target family.** Its two refusal tasks (0 and 28) can be bound, but
each needs a reviewer to state what must remain unchanged. That is real review work in
Gate 0, not a compile-time derivation, and it should be scoped before promising refusal
coverage.

---

## Open questions, v2

**Q9 resolved by D27.** Policy-derived evidence never supports a committed tool
transition. Unsupported tool counterfactuals abstain.

**Q11 resolved.** `compile.py` emits scenarios for unbindable contracts; `rollout.py`
returns an explicit `UNKNOWN` result without treating it as candidate failure. Campaign
coverage reports must retain these scenarios while excluding them from capability claims.

**Q12 resolved by D56 in v3.** Production campaigns require a versioned `SupportPolicy`
naming its reviewer and fidelity calibration report. The campaign supplies its measured
threshold; development-level defaults are not publishable policy.

---

## 4. A recovered input for the user policy

### D35. Adopt tau2's own user-simulation guidelines as the user policy's base prompt. **Decided**

**E21.** `airline.json`'s `info.user_info` carries the complete configuration of the
simulator that generated every user turn in this corpus: `implementation: user_simulator`,
`llm: gpt-5.2`, `reasoning_effort: low`, and a full `global_simulation_guidelines` text
stating the rules it followed — one message at a time, follow the scenario instructions
strictly, and *never* invent information the scenario did not provide.

This is directly load-bearing for D23 (the user-policy fidelity gate). The gate asks whether
our simulated user discloses facts the real persona withheld. The corpus now tells us the
exact rule the recorded user was following when it withheld them — "information not provided
in the scenario instructions should be considered unknown or unavailable" — which is the
disclosure discipline our user policy has to reproduce.

**Decision.** The user policy's base prompt starts from these recovered guidelines rather
than from a prompt written fresh. Two reasons: fidelity is measured against user turns
generated under exactly these rules, so starting elsewhere means optimizing toward a target
the reference behavior never had; and it makes the D23 gate's premature-disclosure check a
test of rule-following against a stated rule, not against our guess at one.

Recorded as provenance on the user-policy version, alongside the fact that the reference
user was `gpt-5.2` at low reasoning effort — a detail that matters when our own user policy
runs on a different model and the two are compared.

**Also recovered and worth pinning:** `max_steps: 200`, `max_errors: 10`, `num_trials: 4`,
`seed`, and the tau-bench `git_commit`. The step and error budgets are the ones the recorded
episodes actually ran under, so a rollout budget chosen differently is a deliberate
divergence rather than an arbitrary default.

---

## 5. Compiling, and what the real corpus corrected

Shipped: [bandits/diagnose/compile.py](../bandits/diagnose/compile.py) and its tests.
73 tests in the package, full suite green, no credential needed.

Measured on `corpus-ee3b33086ef177d7`, family `family-451ae91f975c` (36 traces):
**393 grounding transitions, 371 observed (94%)** — 182 with a tool reaction, 189 with a
user reaction — **204 tool actions across 10 tools**, and **155 scenarios**.

The user-reaction count is worth noting on its own: 189 of 371 observed transitions are
user replies, so **slightly over half of this family's transitions are the user policy's
responsibility, not the tool world's**. D23's user-policy fidelity gate is therefore not a
secondary concern; it governs the majority of transitions in the target family.

### D36. Fixtures must be verified against a real artifact before they are trusted. **Decided**

**E22.** The first `compile_test.py` fixture built a model span carrying a nested
`tool_calls` payload — the shape the OpenAI wire format uses. The chat-JSON adapter does not
produce that. It records a tool call as a MODEL span whose `name` *is* the tool and whose
`arguments` are the call's arguments flat, with `output: None`; speech gets
`name="assistant"` and the text as output.

Every one of the 20-odd compilation tests passed against that fixture while
`extract_transitions` produced **zero tool-typed transitions on the real corpus** — every
action read as plain speech. The suite was green and the module was inert.

**Decision.** `_render_action` handles both shapes, with `SPEAKER_SPAN_NAMES` separating a
speaker name from a tool name, and the fixture now documents that it was verified against
`corpus-ee3b33086ef177d7`. `test_both_recorded_call_shapes_are_read` pins both.

The general rule, which is the part worth keeping: **a fixture is a claim about the data,
and an unverified claim is exactly as wrong as unverified code — but harder to notice,
because it makes the tests agree with it.** Every new fixture in this plane gets a
real-artifact smoke run before its tests are believed.

### D37. State paths are namespaced by the entity the call named. **Decided**

**E23.** The first implementation looked up the pending call through an
order-independent dict, so it namespaced results by tool alone:
`get_reservation_details.status`. Two reservations both reporting `status` then collapse
onto one path and the second silently overwrites the first — with no error, and a verifier
checking "the reservation is cancelled" reading whichever happened to be observed last.

Corrected to track the immediately preceding call, giving
`get_reservation_details.3RK2T9.status`. Also fixed in the same pass: errored results
contribute no state, since an error reports what did *not* happen and reading state from it
invents the fact the failure denies.

### D38. tau2's control markers were never declared at ingest. **Decided, with a migration consequence**

**E24.** `corpus-ee3b33086ef177d7.control_markers` is `()`. The corpus was ingested before
that field existed, or without it being passed — yet 21 of the family's transitions carry
`###TRANSFER###` in user text, confirmed by compiling with the marker supplied explicitly.

So the marker is present in the data and undeclared in the artifact. `compile.py` takes
markers as a parameter rather than reading them off the corpus, which makes it correct
today, but it means the caller has to know something the artifact does not say.

**Decision.** Gate 0's re-ingestion declares `control_markers=("###TRANSFER###",)`. This
changes the corpus's content hash and therefore its id — which is the right outcome, not a
problem: the legacy corpus genuinely does not record a fact about itself that the new one
will, and that is precisely the "compare ingest-contract versions and migrate" case rather
than the "force reproduction of the old hash" case. The legacy artifact is preserved
unchanged as provenance.

Until re-ingestion, any diagnose command reading the legacy corpus must pass markers
explicitly, and should warn when a corpus declares none but its text contains a known one.

---

## 6. Three contract corrections before compiling further

### D34 reversed. A success contract does not establish initial state. **Decided**

The earlier form seeded a `ForbiddenEffect`'s `state_path` into `initial_state` as a
`RECORDED` field flagged `seeded_by_contract`. That was internally inconsistent with
`WorldOrigin.RECORDED`'s own definition — "read off a real span in a real historical
episode." A reviewer asserting `must_remain="confirmed"` is stating what *should* remain
true, which is not the same fact as what *was* true, and manufacturing a state field from
it puts an assertion in the ledger where an observation belongs.

**Corrected rule:**

```text
baseline observed in the authentic prefix  -> compare initial and final state
baseline not observed                      -> check the event ledger for forbidden writes
essential but unavailable                  -> that component is UNKNOWN
```

`must_remain` stays what it always was: an expected value the verifier compares against.
It never enters `ScenarioState`. `seeded_by_contract` is removed from `StateField`, and its
tests with it.

A refusal therefore passes on: no matching forbidden committed event, no forbidden tool
execution, and the required refusal communication — none of which needs a fabricated
baseline. `test_forbidden_write_that_never_happened_holds` is the ledger path;
`test_seeded_state_makes_a_refusal_checkable` now exercises the observed-baseline path only.

### D39. An action is a batch of calls, not one call. **Decided**

**E25.** Measured on `corpus-ee3b33086ef177d7`: runs of consecutive tool-call model spans
have length distribution `{1: 1058, 2: 29, 3: 12, 4: 11, 6: 1, 10: 1}` — **54 actions carry
several calls and one carries ten.** The chat-JSON adapter splits one assistant message
holding several calls into one span per call.

The first `GroundingTransition` had `action_tool: str | None` and a single `observation`.
Against that data it fragments a parallel batch into pseudo-sequential turns and pairs each
call with whichever result happened to follow — not necessarily its own. Both the ordering
and the pairing are lost, and neither is recoverable afterwards.

**Decision.** `ActionCall` and `GroundingObservation` as separate contracts;
`GroundingTransition` carries `action_calls` and `observations` tuples with
`tool_call_id` preserving the pairing; `RolloutStep` carries `calls` the same way.
`action_tool` survives as a *derived property* returning None for a batch, so a caller
reading it cannot silently score one call of several. `reaction_role` likewise derives, and
returns `mixed` where both a tool and a user reacted.

Verified after the change: the family now yields **371 transitions, all observed, 14
batched actions**, calls-per-action `{0: 189, 1: 168, 2: 10, 4: 4}`. The transition count
fell from 393 because fragmented batches merged back into single actions — which is the
correction, visible in the number.

`_committed_effects` now lists every call of a batched step, so an action carrying two
cancellations commits two effects rather than one.

---

## 7. The rollout-to-verifier join

Shipped: [bandits/diagnose/verify.py](../bandits/diagnose/verify.py) and its tests.
103 tests in the package, full suite green.

The load-bearing test is `test_reviewed_verifier_executes_against_rollout_claims`: a real
`VerifierSpec` runs through the real `execute_verifier` over evidence rendered from a
rollout. If claim names or value shapes ever drift from `analyze/outcomes.py`, it fails —
and the capability number stops being comparable to a historical one, which is the only
property that makes it mean anything.

Preserved across the join, each with a test: absence is unknown rather than failure;
every rendered row is `provenance="derived"` and never `observed`; `episode_span_count` is
always emitted so `NO_SPAN_ERROR` can tell a clean run from an unmeasured one; and
`is_simulation_conditioned` is computed **per claim**, so an end-prefix rollout resting
entirely on recorded state is not reported as conditioned.

`compose_overall` additionally refuses to drop a half the source declared: when
`reward_basis` names `DB` and the binding produced no operational claim, the verdict is
unknown rather than a communication-only pass.

---

## 8. Retrieval, and the coverage table that settles D27

Shipped: [bandits/diagnose/retrieve.py](../bandits/diagnose/retrieve.py) and its tests.
120 tests in the package, full suite green.

**E26.** Built the fit-side index for `family-451ae91f975c` — 371 transitions, 294 after
excluding the 8 held-out traces — and ran the per-tool coverage report:

| tool | transitions | with an error |
|---|---|---|
| `get_reservation_details` | 62 | **0** |
| `get_user_details` | 25 | **0** |
| `calculate` | 20 | **0** |
| `cancel_reservation` | 16 | **0** |
| `search_direct_flight` | 13 | **0** |
| `transfer_to_human_agents` | 9 | **0** |
| `get_flight_status` | 7 | **0** |
| `search_onestop_flight` | 5 | **0** |
| `update_reservation_flights` | 4 | **0** |
| `book_reservation` | 3 | **0** |

**Every tool in the target family has zero error evidence.** Not "sparse" — zero. D27's
strict resolution is therefore not a conservative preference; it is the only honest
reading of this index. Any error or refusal the AWM emits for this family would be
invented, and the abstention path is the entire off-happy-path behaviour available.

This also bounds what Gate 3 can measure on this family: a candidate that takes a
well-formed, supported action can be simulated, and a candidate that takes a malformed or
ineligible one hits abstention. The rollout report must carry the abstention rate per tool
beside every pass@k, or the number silently describes only the candidates that behaved.

### D40. Compatibility filtering runs before ranking, with separate relations by role. **Corrected by D48 in v3**

`GroundingTransition` carries `success_shape`. For user-policy retrieval, the source and
query shapes must match because the user's goal and disclosure behavior depend on the task.
For tool-world retrieval, shape does not gate evidence: what an API does is independent of
whether the agent was permitted to call it. In particular, a refusal scenario may retrieve
a successful cancellation transition so an improper call executes and the verifier catches
the policy violation instead of the environment rescuing the candidate.

Effect compatibility filters the same way, from the reviewed catalog. With no catalog the
filter does not fire at all: an unreviewed toolset cannot be filtered honestly, and
guessing from a tool's name is what D26 refuses.

### D41. Retrieval reserves slots for contrast rather than returning the top-k. **Decided**

Given E26, the top-k of this corpus is k near-identical successes of the same call. An AWM
shown only those learns that the action always succeeds, which is exactly the belief that
makes a simulator useless for diagnosing weak candidates.

`_diversify` keeps the highest-scoring results but reserves a slot for an error case and
one for a user reaction where either exists. On this family the error slot will usually go
unfilled, which the coverage table already says out loud — the mechanism is there for the
corpora that do have the evidence, and its emptiness here is itself reportable.

Support estimation is deliberately conservative for the same reason: two exact-tool matches
for `high`, one for `medium`, anything else `low` or `none`. A generous estimate is how an
invented refusal gets committed as though it were grounded.

### D42. Similarity is lexical, and says so. **Decided**

No embedding backend is configured anywhere in this pipeline. Ranking uses token overlap
plus explicit signals (exact tool, shared state paths, prior errors), and records which
signal fired in `RetrievedExample.reasons`.

This follows the precedent the RLM miner set when it refused to record a coherence figure
for a grouping that measured no distance: a plausible similarity number from a backend
that was never run is fabricated geometry. If retrieval quality turns out to bound
fidelity, adding an embedding backend is a measured decision with a baseline to beat, not a
default chosen before anything was measured.
