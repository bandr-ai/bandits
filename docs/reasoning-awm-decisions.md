# AWM / TraceWorld — decision record

Companion to [awm-plan.md](awm-plan.md). That document says *what* to build. This one
records *why each choice was made*, what was rejected, and what evidence in the current
codebase forced the choice. Every entry is written so a later reader can overturn it by
producing contrary evidence, not by preferring a different taste.

Scope of this record: the **emulate phase only** (plan §"Current implementation target").
Milestones 4, 7, 8 and the ADWM experiment are out of scope and are not decided here.

Status legend for each decision: **Decided** (act on it), **Provisional** (acting on it,
but a named measurement can overturn it), **Blocked** (waiting on an answer in §Open
questions).

---

## 0. Evidence gathered before deciding

Read in full before any decision below: `bandits/traces.py`, `bandits/store.py`,
`bandits/analyze/models.py`, `bandits/analyze/tasksets.py`, `bandits/verify/models.py`,
`bandits/verify/execute.py`, `bandits/verify/judge.py`, `bandits/transport.py`,
`bandits/analyze/rlm_mine.py` (predictor/budget/session idiom), `bandits/analyze/rlm_session.py`,
`bandits/export/eval.py`, `README.md`, `docs/rlm-task-family-mining-plan.md`, PR #44's
`bandits/verify/turns.py`, and the `handoffs/verifier-family-bundle-2026-09-12` bundle.

Facts established, which the decisions rest on:

| # | Fact | Where |
|---|---|---|
| E1 | `Span` has no state field. A tool result is `output: Any`; nothing records world state before/after. | `traces.py` |
| E2 | `execute_verifier` scores only `Evidence` read off **recorded spans** via `analyze/outcomes.py`. It cannot score a rollout that produced no spans. | `verify/execute.py` |
| E3 | `Evidence.provenance` is `observed \| derived \| model \| human`, and `EvidenceKind` bottoms out at `AGENT_SELF_REPORT`. There is no kind for "a simulator made this up". | `analyze/models.py` |
| E4 | Fit/held-out split lives on `TaskFamily.fit_trace_ids` / `held_out_trace_ids`, moved as whole lineage groups. It is the only leakage boundary that exists. | `analyze/models.py`, README |
| E5 | PR #44's `Turn` **clips** action to 1600 chars and each reaction to 1200 chars *at extraction*, and drops pre-first-model-call tool results. | PR #44 `turns.py` |
| E6 | PR #44 is open, authored by someone else, +3622 lines, unmerged. | `gh pr view 44` |
| E7 | The project-root store holds **one** corpus (`corpus-9487f569d11267de`, 40 SWE-bench traces, `tools_available: NO`), one analysis, one clustering run, audits — and **zero** task sets, **zero** verifier drafts, **zero** reviewed verifiers. The separate tau run artifacts have not yet been installed into this store. | `.bandits/` |
| E8 | The tau2 handoff bundle holds a frozen 16-family task set (`taskset-133062e911a95d3c`), 160 trajectories with a declared 13-tool schema, structured JSON tool results, real user turns, a system policy, and ground-truth pass/fail labels from tau-bench v1.0.1. Four families have a non-empty held-out side (sizes 36/28/16/12). | `handoffs/.../tau2-16-mined-families` |
| E9 | PR #44 measured tau2 reaction-only scoring at **chance** (AUC 0.51–0.54), because tau2 failures live in a hidden goal-state diff no reaction reveals. | PR #44 description, `docs/next-state-verifier.md` |
| E10 | `gepa==0.1.4` is already installed, but only as a transitive dep of `dspy[deno]` under the `audit` extra. It is not a declared dependency of this project. | `.venv`, `pyproject.toml` |
| E11 | Established model-call idiom: a `_Predictor` Protocol injected into pure functions; the real `dspy`/model import happens only inside `build_predictor`; tests inject a fake. Core install stays at three runtime deps. | `rlm_mine.py` |
| E12 | Established long-run idiom: a mutable `SessionState` scratchpad rewritten every chunk for resume/watch, explicitly **not evidence**, plus a content-addressed immutable run artifact written at the end. | `rlm_session.py` |
| E13 | `ClusteringProvenance` sets the precedent that every model-influenced step records resolved values (model, seed, prompt digest, budget, stop reason) structurally, never as prose. | `analyze/models.py` |
| E14 | tau2 corpora carry `control_markers` (e.g. `###TRANSFER###`) written into a user turn's own text. | `traces.py` |
| E15 | The previously missing tau source has been restored locally: `work/tau/data/airline.json` (15.7 MB), converted `work/tau/run/airline.json`, mappings, labels, and `run/test_tasks.json` all exist. | `work/tau/` checked 2026-09-14 |
| E16 | The original project store still exists at `work/tau/run/proj/.bandits/`. It contains the original corpus, analysis, frozen task set, labels, target-family verifier draft/run, and interviews, but no reviewed verifier artifact. | `work/tau/run/proj/.bandits/` checked 2026-09-14 |
| E17 | Corpus identity hashes the full serialized contract. Ingest changes to fields such as user turns or offered tools can legitimately change the corpus id even when the source export is byte-identical. | `store.py`, `ingest/chat_json.py`, git history |
| E18 | PR #44 produces a `FamilyVerifier` containing sandboxed `check(turn)` predicates, not a `VerifierSpec`. Its trace aggregation passes iff there is at least one observed turn and no turn is flagged; its fractional score is `1 - flagged / observed`. | PR #44 `propose.py`, `docs/next-state-verifier.md` |
| E19 | The legacy target-family draft contains instance constants (`passengers[].count == 2`, destination `PHX`, reservation id `59XX6W`) and explicitly says it was frequency-drafted without adjudicated labels. It is not a parameterized cancellation/refund success contract. | `verifier-draft-69661f703fad9c8d` |

---

## 1. Framing decisions

### D1. The AWM is a *derived artifact producer*, not a new evidence source. **Decided**

The entire repository rests on one invariant: source evidence is immutable, every
interpretation is a new derived artifact beside it, and trust is ranked (E3). A simulator
that emits observations is manufacturing things that *look exactly like* tool results but
were never observed.

**Decision.** Nothing the AWM produces may ever be written as a `Span`, ingested as a
`TraceCorpus`, or constructed as `Evidence` with `provenance="observed"`. Simulated
rollouts live in their own artifact kind under `DerivedStore`, parented to the scenario
they came from.

**Rejected: writing rollouts back as a corpus.** It would make them re-ingestable and
therefore minable, exportable as SFT, and drafted against — and one accidental `export
--format sft` would ship generated trajectories as demonstrations. The README's central
claim ("Keep the truth") would be false. The cost of the rejected option is invisible and
unrecoverable; the cost of this decision is one more artifact kind.

**Falsifiable by:** nothing. This is a hard boundary.

### D2. Keep simulation origin outside `EvidenceKind`'s authority ordering. **Decided; previous decision overturned**

E3 exposes two different questions that the earlier decision incorrectly collapsed:

1. **authority:** live query, external result, human label, model judgment, self-report;
2. **world/origin:** recorded production, executable environment, or learned simulation.

A simulated terminal-state field may be authoritative about what happened *inside that
simulator branch*, while remaining no evidence at all that the same event would happen in
production. Putting `SIMULATED_TRANSITION` below `AGENT_SELF_REPORT` in one linear enum loses
that distinction. It also creates a lifecycle bug: the current `rests_only_on_self_report`
guard checks equality with one enum member, so a simulated-only verifier would not
automatically inherit the promised promotion prohibition.

**Decision.** Do not add `SIMULATED_TRANSITION` to `EvidenceKind`. Introduce a separate
rollout-claim contract with `world_origin = recorded | executable | simulated` and retain
the authority of the underlying claim separately. Simulator output cannot validate,
calibrate, review, or promote a verifier. A verifier is calibrated and promoted only on
real held-out evidence; applying that frozen verifier to rollout claims produces a
`simulation_conditioned` result.

**Rejected: reusing either `MODEL_JUDGMENT` or `STRUCTURED_EXTERNAL_RESULT`.** The first
conflates judging a real run with generating one; the second falsely presents learned
state as external state.

### D3. "AWM fidelity" and "candidate capability" are two different artifacts and must never share a number. **Decided**

The plan's §"Non-negotiable reporting" already demands the split. Making it structural
rather than editorial: a fidelity report is parented to the AWM version; a capability
report is parented to the candidate + scenario set. Neither may contain the other's
headline figure.

**Why this is load-bearing.** A candidate scoring well on a low-fidelity simulator has
demonstrated nothing, and the plan's own route table (§Phase 9) has a row for exactly this
("Strong only on learned-AWM-heavy rollouts → Block"). If the two numbers live in one
report, that row cannot be evaluated.

### D4. Rename `fit transitions` → `grounding transitions` in code and UI. **Decided**

The plan asks for this explicitly. Adopted verbatim: `fit` in this repo already means a
partition side (E4), and "fitting" means training. Reusing it for retrieval evidence would
make `fit_transitions` read as "transitions we trained on", which is the one thing the
no-weight-training default does *not* do.

Naming settled now, before any code exists: `GroundingTransition` (the record),
`grounding_index` (the retrieval structure), `fit-side grounding evidence` (prose).

---

## 2. Corpus and scope decisions

### D5. Emulate targets the **tau2 airline** corpus, not the local SWE-bench one. **Decided**

E7 and E8 force this.

The local store's only corpus is 40 SWE-bench traces with `tools_available: NO` and a
single `run_in_shell` tool whose output is a terminal screen. The plan itself says
(§"Coding backend") that coding tasks need a real container and that "a textual trace
cannot reconstruct a missing repository or dependency state". Simulating `run_in_shell`
against an astropy checkout is precisely the case the plan excludes.

tau2 airline is the opposite: 13 declared tools with parameter schemas, structured JSON
results, a written system policy, real user turns, a frozen task set, and ground-truth
labels. It is the only asset in this repository on which AWM fidelity is measurable at all.

**Decision.** Emulate phase runs on tau2. SWE-bench stays as the negative control that
demonstrates the boundary, and is not simulated.

**Cost accepted:** tau2's user side is an LLM simulator, so "user reply" transitions are
simulations of a simulation. Recorded as a limitation on every tau2 fidelity figure, not
hidden.

### D6. Start with **one** family, not four. **Provisional**

`family-451ae91f975c` (cancellation and refund): 36 traces, 28 fit / 8 held-out, 22 pass /
14 fail. It is the largest, has the most held-out mass, and has both outcome classes.

**Why not all four.** Fidelity must be characterized before it is trusted, and four
families is four times the model spend to learn the same lesson. If the AWM cannot
faithfully simulate `cancel_reservation` — a tool with a small argument space and a
deterministic result — it will not do better on rebooking.

**Explicitly excluded:** `family-954be23fc311`, which the corpus flagged itself as
over-merged (widest pair 0.57 against a 0.40 link threshold, three unrelated operations).
Using a family the system already reported as incoherent would confound AWM error with
family error.

**Overturned by:** a fidelity result on 451ae91f975c good enough that the next question is
about generalization rather than about the AWM. Then add b3fbe10f04b6 and 224a9c361db3.

### D7. Recover the legacy tau artifact graph, then migrate deliberately. **Decided; revised**

E7: the task set referenced throughout the handoff (`taskset-133062e911a95d3c`) is a JSON
file in `handoffs/`, not an artifact in the store. Nothing in the codebase can load it —
`load_task_set` reads `DerivedStore`. E16 confirms that the original content-addressed graph
still exists under `work/tau/run/proj/.bandits/`, including the exact corpus, analysis, task
set, labels, and partial verifier work.

**Decision.** Preserve and audit the legacy graph first. Then run the current ingester against
the same converted source and compare contracts. Matching ids permit direct reuse. An id
mismatch is expected when the ingest contract changed (E17): record a field-level migration
delta, produce a new corpus/analysis/task set, and re-mine under current code. Never mutate
artifacts or chase the historical hash by suppressing newly represented fields. Emulate
must bind wholly to either the frozen legacy graph or the newly migrated graph, never mix
parent ids across them.

**Rejected: reading the handoff JSON directly or copying only the task-set JSON.** Either
would detach scenarios from the parent graph that gives their lineage meaning.

---

## 3. Transition representation decisions

### D8. Do **not** block on PR #44. Define `GroundingTransition` independently, adapting from `Turn` when it lands. **Decided**

The plan says "if it lands, emulate should reuse it". E6 says it has not landed, is
3622 lines, and is another person's work. E5 says its `Turn` is lossy in a way that matters.

**The specific problem.** `Turn` clips the action to 1600 chars and each reaction to 1200
chars *at extraction time*, and `Reaction.text` is a rendered string, not the structured
payload. For a next-state *judge* that is correct and cheap — a judge only needs enough to
say +1/0/−1. For an AWM it is fatal twice over:

1. A grounding example whose `get_reservation_details` result was truncated mid-JSON
   teaches the simulator to emit truncated JSON.
2. A *fidelity* comparison needs the real structured result to diff predicted fields
   against. `Turn` has already thrown the structure away.

**Decision.** `GroundingTransition` holds the **unclipped structured** action and next
observation, plus span ids back into the corpus. Clipping is a *rendering* concern applied
at prompt-build time against a context budget, never at extraction. Where PR #44's
`extract_turns` boundary logic is right — MODEL span starts an action, intervening
tool/user spans are its reaction, a trailing action with no reaction is *unobserved* and
never scored — that logic is reused, by importing it if PR #44 lands and by reimplementing
that same rule if it does not.

**Not a competing representation.** `Turn` is a judge's view; `GroundingTransition` is a
simulator's view. If #44 lands, `GroundingTransition` is built *from* the same span
boundary and carries `turn_index` so the two join.

**Falsifiable by:** #44 landing with the clipping moved to render time, in which case
`GroundingTransition` becomes a thin wrapper and this decision costs almost nothing either
way.

### D9. `Turn.observed == False` transitions are grounding evidence for *nothing*. **Decided**

PR #44's rule — a final action with no reaction is unobserved, never scored as if silence
were approval — applies with more force here. A transition with no recorded next
observation cannot teach a next-observation predictor and cannot be a fidelity test case.

**Decision.** They are excluded from the grounding index and from fidelity scoring, and
the count of excluded transitions is reported. They are *not* deleted from the transition
corpus, because "how much of this family's tail is unobserved" is itself a limitation the
rollout report needs.

### D10. Strip `control_markers` before a transition enters the grounding index. **Decided**

E14: tau2 appends `###TRANSFER###` to a user turn's own text on most airline episodes, not
only ones that escalate. `TraceCorpus.control_markers` exists precisely so a reader knows
this. A miner shown it reads it as evidence about the request; an **AWM** shown it will
learn to *emit* it, and a candidate will then learn to condition on a token the real
environment does not emit that way.

**Decision.** The grounding index honors `control_markers` at build time. The stripped
marker is recorded per transition rather than silently removed, so a fidelity comparison
is not penalized for a difference that was scaffolding.

---

## 4. State decisions

### D11. Scenario state is an **external ledger owned by TraceWorld**, never a field on `Span` or `Trace`. **Decided**

E1: the trace contracts have no state field, are frozen, and are content-addressed — adding
one changes every artifact id in the store. More importantly the plan's §"Complete
no-weight-training AWM stack" already names "externally maintained scenario state/event
ledger" as a required component.

**Decision.** A `ScenarioState` lives beside the rollout, is advanced by validated state
deltas the AWM proposes, and every field carries provenance (`observed` from the
authentic prefix, vs `simulated` after the cut point). `traces.py` is not touched.

**Consequence.** The initial state of a scenario is reconstructed from the authentic prefix
only. What the prefix never revealed is `unknown`, not `absent` — the same rule
`execute.py` already applies to checks (absence is never failure).

### D12. The AWM proposes a state delta; a **validator** commits it. **Decided**

The plan's safety section asks for this. The reason it is non-negotiable here: the external
verifier reads state, so a simulator that can write state unchecked can write itself a pass.
That is the plan's own "AWM can generate a state that a correct verifier then accepts".

**Decision.** Deltas are validated against the family's declared invariants before commit.
A rejected delta is an abstention, not a silent no-op, and is counted in the
unsupported-transition rate.

---

## 5. Verifier decisions

### D13. The existing verifier plane **cannot** score a rollout without an adapter. Build a provenance-safe adapter. **Decided; mechanism revised**

E2 is the sharpest constraint in this whole exercise and the plan does not address it.
`execute_verifier(spec, evidence)` takes `Evidence`, and every producer of `Evidence` in
this repo reads recorded spans. A rollout produces a `ScenarioState` and a simulated
transcript. Nothing joins them.

**Decision.** A `rollout_claims(rollout) -> tuple[VerifierInputClaim, ...]` adapter emits
the same claim names `analyze/outcomes.py` emits (`final_state_field`,
`initial_state_field`, `episode_span_count`, `span_error`, …), but with explicit
`world_origin` and field-level provenance. Factor the operator logic behind a small common
claim interface; preserve `execute_verifier(spec, evidence)` as the replay-compatible
wrapper. Do not manufacture `Evidence` objects that pretend simulated fields came from
recorded spans.

**Why an adapter and not an unrelated executor.** The verifier's operator semantics — unknown is
not failure, one unknown check makes the whole verifier unknown, ambiguous bare keys refuse
rather than guess — are the reviewed, tested, human-promoted part of this system. A second
executor would re-derive them and drift. The adapter means a reviewed verifier scores a
rollout *the same way* it scored history, which is the only basis on which the two numbers
are comparable at all.

**Risk accepted and named:** a verifier reviewed against real evidence was never reviewed
for its behavior on simulated claims. Every rollout verdict is therefore reported with
its transition-source composition, and a verdict resting on any simulated field is
`simulation_conditioned`. Such a verdict is usable for model comparison only after the AWM
passes its independent fidelity gate; it is never verifier-calibration evidence.

### D14. tau2 rollout verdicts come from the **reviewed verifier**, not from tau-bench reward, and not from a reaction judge. **Decided**

Three available scorers, and the choice matters:

- *tau-bench's own reward* grades a recorded episode against a hidden goal state. A
  candidate's simulated rollout has no tau-bench env to grade against. Unusable by
  construction.
- *PR #44's reaction judge* measured at chance on tau2 (E9), for a documented reason that
  applies identically here: the failure lives in state no reaction reveals. Using it would
  produce a capability number indistinguishable from noise.
- *A Bandits reviewed verifier* reads terminal state fields, which the `ScenarioState`
  ledger (D11) actually carries.

**Decision.** The reviewed verifier is the scorer. The tau-bench labels are used for their
correct purpose — calibrating and validating that verifier against history (they are
already imported as a `LabelSet`, E8) — never as the rollout grader.

Here “reviewed verifier” specifically means the existing `VerifierSpec` → validation →
`ReviewedVerifier` lifecycle, with task/scenario parameters materialized into state claims.
It does **not** mean PR #44's `review-checks` lifecycle. D13 is therefore aimed at the right
target: rollout state becomes verifier claims, not clipped `Turn`s.

The current `CheckSpec.expected` is fixed, so a heterogeneous family also needs a narrow
`VerifierBinding`: reviewed placeholders are resolved from the scenario's sealed success
contract into a concrete spec before execution. The candidate and AWM may never supply or
change those expected values. Historical validation binds the same template independently
for each labeled task; promotion applies to the template plus binding rules, not to one
reservation's constants.

**Blocking consequence.** E7 says no verifier has been promoted yet. Emulate cannot report
pass@k until one exists for 451ae91f975c. See Q2.

---

## 6. Retrieval and AWM decisions

### D15. Retrieval is lineage-safe by construction: the index is built from **fit-side transitions only**, per family. **Decided**

E4: the fit/held-out split is the only leakage boundary in the system and it moves whole
lineage groups for a reason the README states plainly — otherwise held-out agreement
reports memorisation as generalisation.

The same failure has a worse form here. If a held-out scenario's own trace is retrievable,
the AWM can copy the real next observation and fidelity scores near-perfect while having
learned nothing. That number would then authorize a rollout campaign.

**Decision.** The index is keyed by family and contains only `fit_trace_ids` transitions.
A held-out scenario's retrieval is filtered against its own lineage group *in addition*, so
a retry of the same request cannot leak either. The filter is asserted at query time, not
only at build time, because a shared index is exactly the thing that silently stops being
partitioned.

### D16. Sealed fidelity split is carved from held-out, and is **not** the same as the held-out used for prompt selection. **Decided**

The plan asks for fit / disjoint-selection / sealed-held-out. Bandits only offers two sides
(E4), and the tau2 SOURCE-NOTES already observed that Bandits' held-out becomes a
development set by the time anything is promoted — which is why the tau2 conversion
reserved 10 task ids (40 traces) *outside the corpus entirely*.

**Decision.** Three-way: GEPA optimizes on a fit subsample; prompt selection uses a disjoint
fit subsample (**not** held-out); held-out is the sealed fidelity split, opened once for
final reporting. Selecting on held-out would burn the only clean measurement available.

The reserved 40 traces in `work/tau/run/test_tasks.json` are present (E15), so they are the
final seal. They remain outside retrieval, optimization, prompt selection, and verifier
iteration until the one-time final audit.

### D17. Default AWM path is prompt + retrieval, no weight training. **Decided** (adopting the plan)

Adopted as written, with the reasoning recorded rather than restated: the escalation to
weight training is gated on a *measured* residual, and the plan is explicit that weight
training is an optimization, not a missing functional requirement. Nothing in the current
evidence justifies training a model before knowing whether a prompt can do it.

### D18. Declare `gepa` directly under a new `emulate` extra; do not rely on the transitive pull. **Decided**

E10: it is installed only because `dspy[deno]` happens to require `gepa[dspy]==0.1.4`.
Importing it directly while it is transitive means a dspy upgrade that drops or bumps it
breaks this code with no declared reason.

**Decision.** Add it explicitly under a new `emulate` optional extra. The core install's
three-runtime-dependency property (E11) is preserved, while neither audit users nor
emulate users acquire the other's machinery accidentally.

### D19. AWM calls follow the `_Predictor` Protocol + injection idiom. **Decided**

E11. Tests inject a fake predictor, the real model import happens only inside a
`build_*` function, and CI needs neither a credential nor a sandbox. Deviating would make
the AWM the first untestable-without-a-model component in the repo.

Likewise adopted from established practice: `prompt_digest` pinning wording + model +
version onto every artifact produced (as `Rubric.prompt_digest` and `rlm_mine.prompt_digest`
both do), `request_with_retry` for transport, and structural provenance in the
`ClusteringProvenance` style (E13) rather than prose.

### D20. Rollouts get a resumable `SessionState`-style scratchpad. **Decided**

E12. A rollout campaign is `n` samples × scenarios × steps of model calls — strictly longer
and more failure-prone than a mining run, which already needed this. The same split applies:
the session is a mutable scratchpad and **not evidence**; the immutable rollout artifact is
written at the end and is the only thing anything downstream may cite.

---

## 7. Measurement decisions

### D21. Fidelity is measured on structured fields and state effects, never text similarity. **Decided** (adopting the plan)

Recorded because tau2 makes it concrete and cheap: tool results are JSON, so a predicted
`cancel_reservation` result can be diffed field-by-field against the recorded one. Status
accuracy, error-vs-success accuracy, and per-field accuracy are all computable without a
judge. A fidelity judge is reserved for the natural-language reactions (the simulated user
turn), where no structured diff exists.

**Consequence:** the cheap deterministic half is built first. If the AWM cannot get
`status` right on a 13-tool API it will not get prose right, and that answer costs no model
spend to obtain.

### D22. Static `next_action_capability` is built first and is explicitly not pass@k. **Decided**

The plan permits static checkpoints as "a cheap preliminary screen" and requires the label.
Taken as sequencing: static probes need no AWM at all — they read the real next observation
from the trace — so they can run the day scenarios compile, and they answer whether a
candidate is worth spending simulator budget on.

**Hard rule, in the artifact and not only the prose:** the field is named
`next_action_capability`, a static report cannot contain a field called `pass_at_k`, and the
two are never averaged.

### D23. The user policy has an independent fidelity gate. **Decided**

In tau2 the user simulator is causally upstream of the candidate's later actions. A
too-helpful user can disclose an identifier, confirmation, preference, or constraint that
the source persona was instructed to withhold until asked. That can raise candidate pass@k
even when the tool-world simulator is perfect.

**Decision.** Full-rollout capability is blocked until both environment roles pass separate
fidelity gates. On held-out recorded transitions, the user-policy gate conditions on the
authentic prefix, authentic preceding agent action, and hidden tau persona/goal, then checks:

- prohibited or premature information disclosure;
- required response act and information revealed when correctly elicited;
- persona/goal and entity consistency;
- confirmation, refusal, escalation, and termination behavior;
- unsupported/off-policy response and abstention rate.

Natural-language wording is not expected to match. Deterministic disclosure/constraint
checks come first; an independent rubric judges semantic behavior where rules cannot. The
four tau trials per task provide multiple real responses and action variations, reducing
dependence on one transcript.

**Boundary.** Held-out next-user fidelity validates supported historical contexts, not an
arbitrary response to a novel candidate action. Off-support user interactions must abstain
or be reported separately. Final candidate rankings also receive a user-policy sensitivity
check; a ranking that changes materially across two qualified user-policy versions is not
stable enough for model selection.

### D24. PR #44 turn checks are diagnostics, never the tau2 success oracle. **Decided**

E18 confirms that #44 owns a separate object and lifecycle. Its “pass” means only that no
accepted predicate or next-state judge flagged a locally visible bad reaction. It does not
mean the task's hidden goal was achieved, and its fractional trace score is not task reward.

**Decision.** On simulated rollouts, #44 checks may be run over a deliberately rendered
turn projection to produce step-level warnings, recovery labels, and debugging slices. They
do not contribute positive success reward and are not ANDed/ORed into terminal pass@k. The
terminal verdict comes from the state/event ledger through the reviewed state-based
`VerifierSpec`. A reaction check reading AWM-generated `next_state` is explicitly
`simulation_conditioned` and cannot independently establish either success or failure in
the unavailable real world.

Tool/user fidelity gates reduce the risk of a simulator evading these predicates, but do
not cure circular grading: the AWM still generated the predicate input. That is why the
checks remain diagnostics. The state/event ledger is superior because it is structured,
effect-checked, and invariant-validated, but post-cut simulated fields were still proposed
by the AWM; its terminal verdict therefore remains simulation-conditioned. Only real or
executable state removes that circularity. For coding or computer-use corpora with real
execution reactions, #44 may be much more central; that does not transfer to tau2, where
its measured reaction-only signal was near chance.

**Legacy draft decision.** `verifier-draft-69661f703fad9c8d` is rejected as the Gate 0
fallback (E19). Keep it as provenance and a regression case for draft quality, but re-draft
a parameterized cancellation/refund verifier from imported labels, task requirements, and
state/event claims.

### D25. Bind reviewed success sub-shapes inside the family; do not force one outcome shape. **Decided**

The target family's nine task lineages are intent-similar but outcome-heterogeneous: pure
refusals, informational/read-only tasks, a booking mutation, and a compound
cancel-and-rebook workflow. A single fixed terminal shape cannot honestly verify all of
them. Dropping refusals would also remove the most safety-sensitive cases and bias pass@k
upward.

**Decision.** Keep the family for retrieval and workload reporting, but make
`SealedSuccessContract` a reviewed tagged union with at least:

- `refusal`: specified writes are forbidden/unchanged and communication is required;
- `informational`: no operational success is claimed; required communication carries the
  task while reads remain optional process evidence;
- `mutation`: one or more required state effects/events plus communication requirements;
- `compound`: several required effects/events with explicit multiplicity and optional
  partial-order constraints.

The scenario's sealed source contract selects the tag before rollout. Neither candidate nor
AWM behavior may select or change it. Promote the finite set of sub-shapes and their binding
rules together; report capability both for the parent family and separately per sub-shape.
If a task cannot be bound without an unreviewed interpretation, it is verifier-unknown, not
silently excluded or forced into the nearest tag.

The family remains the outer retrieval partition, but compatibility filtering happens
before similarity ranking: retrieval may cross sub-shapes only when tool/effect and workflow
constraints are compatible. A pure-refusal scenario must not receive a booking mutation as
an analogous success merely because both appeared in the same mined family.

### D26. Reads are process evidence; writes/effects carry operational success. **Decided**

Tool effect class comes from a reviewed, versioned `ToolEffectCatalog`, using source tool
metadata where available—not a `get_*` naming heuristic. Read calls can establish that the
agent gathered information, but never that it completed the task. A read-only informational
task therefore has `operational_result = not_applicable`, not pass. Unauthorized writes or
failed safety invariants can still make it fail.

Required mutations are matched against committed state effects/events, not merely attempted
calls. The default action/effect matching policy is an argument-aware multiset: multiplicity
matters, order does not. Thus two cancellations of different reservation ids require two
distinct matching committed events and cannot be satisfied by cancelling one id twice.
Order is enforced only when a reviewed sub-shape declares a sequence or partial-order
constraint. The resolved policy is stored on every `VerifierBinding` rather than inferred at
execution time.

Communication is structurally separate. Because `communicate_info` is sparse, most tau2
communication checks use `nl_assertions` through a rubric and retain
`MODEL_JUDGMENT` authority. Operational, process, communication, and overall results are
reported separately and never averaged.

---

## 8. What was considered and rejected outright

| Rejected | Why |
|---|---|
| Writing simulated rollouts back as a `TraceCorpus` | D1. Makes fabrication re-ingestable, minable, and SFT-exportable. |
| Reusing `MODEL_JUDGMENT` for simulated observations | D2. Conflates judging a real run with fabricating one; one is promotable, the other must never be. |
| Extending `Span` with state fields | D11. Frozen, content-addressed contracts; would change every artifact id in the store. |
| A second verifier executor for rollouts | D13. Would drift from the reviewed, tested operator semantics, and make history and rollout numbers incomparable. |
| Simulating `run_in_shell` over SWE-bench | D5. The plan's own coding-backend section excludes it; a trace cannot reconstruct a repository. |
| Using PR #44's reaction judge as the rollout grader on tau2 | D14. Measured at chance on this exact corpus, for a reason that applies unchanged. |
| Selecting the GEPA prompt on held-out | D16. Burns the only sealed fidelity measurement available. |
| Starting with all four gate-passing families | D6. Four times the spend to learn the same thing about fidelity. |
| Blocking all work on PR #44 merging | D8. Open, large, another author's; and its `Turn` is lossy for this purpose anyway. |

---

## Question resolutions

These answers are now part of the emulate implementation boundary.

**Q1. tau2 source data — resolved.** Both exports and the complete legacy project store are
present (E15/E16). Audit the frozen graph, compare current re-ingestion at field level, and
reuse or migrate according to D7. An id mismatch is a versioned contract change, not bad data.

**Q2. Promoted verifier — resolved as a required build gate.** The legacy store contains a
target-family draft, run, labels, and interviews, but no reviewed verifier artifact (E16).
Resume from those artifacts where compatible; produce, validate on real labels, review, and
promote the final verifier. Interactive pass@k waits on it; static diagnostics and AWM
fidelity engineering do not.

**Q3. The 40 reserved tau2 test traces — resolved.** `work/tau/run/test_tasks.json` exists,
and the original export still contains those simulations. Keep them entirely outside
retrieval, GEPA optimization, prompt selection, and verifier iteration; convert/open them
once for the final fidelity and capability audit.

**Q4. Which model for the AWM? — resolved architecturally.** The implementation is
provider-neutral through the existing injected `_Predictor` idiom and records model id,
endpoint, prompt digest, decoding settings, and seed. The first reproducible baseline uses
the current Fireworks/OpenAI-compatible transport; a frontier model is a configured second
arm, not a second architecture. A capability report must pin the selected AWM version.

**Q5. Which candidate(s) are being diagnosed? — resolved for implementation.** The
deliverable is a model-agnostic lab. Candidate identity is configuration, not a package
dependency. A valid first bakeoff requires at least the target OSS model, a stronger
incumbent/reference, and a deliberately weaker control; the exact target remains a run-time
input and does not block building the lab.

**Q6. Does the simulated user turn belong in scope? — resolved.** Yes, for full tau2
rollouts, but not as an undifferentiated next-observation model. Use two explicit,
versioned roles: a tool-world transition policy and a user policy. The user policy sees the
tau task's hidden persona/goal plus visible history; the candidate never sees that hidden
state. Authentic future user text is never replayed after candidate divergence. Also
publish a tool-only/stop-at-user metric as a diagnostic, never as full-rollout pass@k.
Every full pass@k is conditioned on both the tool-AWM version and user-policy version.

**Q7. Extra placement for `gepa`/`dspy` — resolved.** Add a new `emulate` optional extra.
Emulation should not imply installation of the RLM audit REPL/sandbox, and users of the
audit feature should not acquire the AWM stack accidentally. Declare every imported
dependency directly; do not rely on GEPA arriving transitively through DSPy.

**Q8. Does PR #44 land first? — resolved.** Do not block or coordinate this vertical slice
on it. Implement the lossless `GroundingTransition` against current `Span`s and keep the
segmentation rule in a narrow helper. If #44 lands with a lossless extraction hook, adapt
that helper; its clipped `Turn` remains a judge projection and is never AWM grounding data.
