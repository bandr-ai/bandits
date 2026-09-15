# TraceWorld: Enterprise Traces to a Post-Trained Agent

## Purpose and central design

This is the implementation plan for turning historical enterprise agent traces into:

1. an evidence-backed workload and verifier corpus;
2. a trace-grounded interactive environment, called **TraceWorld**;
3. base-model capability measurements over complete candidate rollouts;
4. verified SFT, OPD, and RL inputs;
5. a regression-gated, deployable post-trained agent.

The design is one rollout loop, not separate AWM, off-policy, and on-policy systems:

```text
Historical traces collected under behavior policy μ
                         |
                         v
              tasks + state + tool evidence
                         |
                         v
                  candidate policy π
                         |
                    chooses action
                         |
                         v
        TraceWorld transition runtime
          | retrieve relevant real evidence
          | AWM updates the simulated world
          | AWM returns the next observation
          | unsupported -> abstain
                         |
                 returns observation
                         |
                         +----> candidate continues
                         |
                         v
                external verifier
                         |
                         v
        pass@k, reliability, cost, regressions
```

The source evidence is **off-policy** because the real traces came from μ while the candidate is
π. A rollout becomes **on-policy with respect to TraceWorld** because every simulated state follows
the candidate's own actions. AWM is the learned part of the transition runtime inside that loop.

ADWM is optional research work. It is not on the critical path.

World-model weight training is also optional. The default AWM is an existing capable model wrapped
with trace retrieval, a GEPA-optimized environment prompt, strict transition contracts, external
state, support estimation, and abstention.

## Current implementation target: `bandits/diagnose`

The immediate project is diagnosis only. It does not build the complete destination architecture in
this document.

```text
tau2 source -> project-store corpus + task families + reviewed verifier
                         |
                         v
       compile task-start, middle-prefix, and end-prefix scenarios
                         |
                         v
       build a grounded AWM from existing trace evidence
       | existing capable model, no weight training required
       | retrieval over fit-side traces
       | GEPA-optimized environment prompt
       | strict response protocol
                         |
                         v
          run several candidate agents against the AWM
                         |
                         v
       rollout-claim adapter -> frozen reviewed verifier
       + simulation-conditioned pass@k/reliability report
```

For this phase, the grounded AWM **is the synthetic interactive environment**. There is no separate
CRM/ERP/Slack replica, executable connector runtime, universal enterprise state schema, or automatic
environment generator. Those are future options only if diagnosis demonstrates that prompt-and-RAG
simulation is insufficient or that subsequent OPD/RL needs stronger state guarantees.

The three scenario classes are:

1. **task start:** only the authentic task, system context, and offered tools are visible;
2. **middle prefix:** authentic history through one of many selected real observations is visible,
   then the candidate continues against the AWM;
3. **end prefix:** authentic history ends near the decisive/final action, then the candidate must
   finish against the AWM and verifier.

The candidate never receives future historical actions after the cut point.

One trace may produce many middle-prefix scenarios. Useful cut points include every observed tool
result, user reply, model error, retry, correction, branch decision, irreversible action, and recovery.
Deduplicate equivalent prefixes and cap per-trace/per-family sampling so long trajectories or common
families do not dominate the evaluation. Each cut point remains bound to the same trace lineage and
held-out partition as its source.

This is environment simulation. In the diagnose phase the simulator is generative and trace-grounded
rather than an executable clone: it repeatedly predicts the next observation after each candidate
action, appends that observation to the simulated session, and lets the candidate act again.

### What Bandits already supplies to diagnose

Bandits already stores the trace spans containing model actions, tool results, user turns, task
context, toolsets when declared, lineage, families, outcome evidence, and reviewed verifiers. Diagnose
must consume those artifacts rather than re-ingest or reinterpret the source independently.

The phrase `action/reaction extraction` means only a deterministic view over that existing corpus:

```text
one recorded model action
      + the tool results/user messages before the next model action
      = one observed AWM example
```

It is not a new trace-mining system. Open PR #44 implements a useful *judge projection* as `Turn`,
but clips actions/reactions and renders structured results to strings. Diagnose therefore reuses only
its span-boundary rule if a lossless hook lands. AWM grounding uses a separate lossless
`GroundingTransition` view over the original spans; clipping happens only while rendering a prompt.

The tau2 source is present under `work/tau/`, including the original 15.7 MB export, converted
160-trace corpus, mappings, labels, and the 40-trace reserved test partition. The original project
store also survives under `work/tau/run/proj/.bandits/`: it contains the frozen corpus, analysis,
task set, labels, target-family draft/run, and interviews, but no reviewed verifier. Diagnose first
audits that artifact graph and then either reuses it or performs an explicit migration.

The phrase `fit transitions` means the action/reaction examples belonging to the existing fit-side
lineages. It does not mean fitting a new model. Rename this artifact in code and UI to
`grounding transitions` or `fit-side grounding evidence` to avoid that ambiguity.

### Scaling GEPA and retrieval

Millions of trajectories do not imply putting every trajectory into every prompt or every GEPA
iteration:

- all eligible fit-side transitions may be indexed for retrieval;
- each AWM call receives only the top relevant examples within a fixed context budget;
- GEPA optimizes the shared environment prompt on a representative fit sample;
- a disjoint selection sample chooses/rechecks the prompt;
- sealed held-out lineages measure final fidelity;
- hard error clusters may be deliberately oversampled during optimization.

GEPA optimizes instructions, not weights. Running it longer is not guaranteed to keep improving the
AWM: it can overfit the optimization judge or selected examples. Stop on disjoint selection
performance, paired base-versus-candidate rechecks, budget, and convergence—not on fit score alone.

## Terminology

### TraceWorld

The complete interactive system: task/scenario compiler, state store, agent harness, connector
runtime, transition routing, learned AWM, provenance, rollout scheduler, and verifier execution.

### Executable environment

Code- and state-backed tool behavior:

\[
s_{t+1}=T_{code}(s_t,a_t), \qquad o_{t+1}=O_{code}(s_{t+1})
\]

This is the Snowflake AWM-style part of the design: generate or author scoped, executable,
database-backed environments rather than cloning production products.

### Learned AWM

A Qwen-AgentWorld/WebEvolver-style language world model:

\[
(\hat{o}_{t+1},\hat{\Delta s}_t,\hat d_{t+1})
\sim M_\phi(h_t,a_t,R_t)
\]

Here `R_t` is lineage-safe retrieved evidence from real transitions. The learned AWM handles
unavailable, expensive, stochastic, or natural-language behavior that executable connectors do not
cover confidently.

### Off-policy source evidence

The candidate did not generate the production traces used to reconstruct tasks and transitions:

\[
D\sim\mu, \qquad \pi_{candidate}\ne\mu
\]

This creates an identification and support problem whenever the candidate takes actions absent from
the trace corpus.

### Candidate-on-TraceWorld rollout

The candidate chooses every action after reset, and TraceWorld generates each subsequent
observation:

\[
a_t\sim\pi(\cdot\mid h_t), \qquad
o_{t+1}\sim\hat E(\cdot\mid h_t,a_t)
\]

This is on-policy relative to the reconstructed environment, not proof of performance in the
unavailable production environment.

For tau2, “environment” contains two separately versioned policies: a tool-world policy that returns
tool observations/state deltas, and a user policy that returns user messages. The user policy may see
the task's hidden persona and goal; the candidate may not. Once the candidate diverges, historical
future user turns are not replayable counterfactuals and must not be copied. Full-rollout pass@k is
conditioned on both policy versions. A stop-at-next-user score may be reported as a cheaper diagnostic,
but never as full-rollout pass@k.

## End-to-end workflow

### Phase 1: Ingest and preserve traces

Inputs may include OTLP, chat JSON, native agent sessions, tool calls/results, system prompts, tool
schemas, rewards, human corrections, external outcomes, final state, model metadata, and coding
workspace identities.

Normalize these into immutable trajectories while preserving causal order, paired tool observations,
user turns, lineage, offered tools, generating model metadata, evidence authority, missing fields,
source digests, and transformation provenance.

Never infer a missing tool result, silently drop an incomplete trace, or treat the agent's claim of
success as external outcome evidence.

**Artifacts:** `TraceCorpus`, ingest issues, toolset evidence, lineage, model/runtime metadata.

### Phase 2: Extract tasks, outcomes, and task families

Extract requested work and all available outcome evidence. Group tasks using the existing rule:

> Two tasks belong to one family only when one parameterized verifier contract could correctly
> evaluate both.

Each family states inclusion/exclusion rules and the required outcome shape. Ambiguous and uncovered
traces remain explicit. Split whole lineage and normalized-request groups into fit and held-out
partitions so retries and near-identical tasks cannot leak between training and evaluation.

**Artifacts:** corpus analysis, family contracts, assignments, `TaskSet`, fit/held-out membership.

### Phase 3: Build, attack, calibrate, and review verifiers

Verifiers remain external to the candidate and prefer outcome/state evidence over self-report. They
may inspect real recorded terminal evidence, executable environment state, deterministic invariants,
tool results, and independent rubric judgments where structured evidence is unavailable.

Measure fit and held-out agreement separately, preserve false positives and false negatives, report
scorable coverage, construct gaming attacks, and require explicit review before promotion.

**Artifacts:** verifier draft, labels, validation, gameability assessment, review, and
`ReviewedVerifier`.

### Phase 4: Compile traces into scenarios and checkpoints

Every eligible trace can yield a task-start scenario, reconstructed initial state, intermediate
real-state checkpoints, error/recovery checkpoints, critical-action checkpoints, and a terminal-state
verifier binding.

Checkpoints serve two roles:

1. optional static one-action diagnostics, which stop before a new observation is needed;
2. prefix-started interactive rollouts, which enter TraceWorld after the candidate's first action.

The primary capability claim comes from complete interactive rollouts, not isolated action probes.

### Phase 5: Compile environment specifications

For each selected family, extract the smallest observable world required to execute its tasks. Do not
reproduce all of Slack, Salesforce, an ERP, or an operating system.

Infer and review tool schemas, entities, cross-connector relationships, reads, writes, preconditions,
permissions, success/error transitions, idempotency, temporal effects, valid workflow branches,
terminal conditions, and initial-state facts.

Every field carries provenance such as `observed`, `derived`, `inferred`, `owner_confirmed`, or
`unknown`.

### Phase 6: Build TraceWorld's transition runtime

For the diagnose MVP, retrieval is evidence for the AWM, not a transition engine or policy. Every
candidate action follows the same main path:

```text
task + scenario + complete visible history + current state summary
                            +
                    candidate action
                            |
                            v
           retrieve all relevant real evidence
                            |
                            v
              grounded AWM reasons over it
                            |
                            v
       next observation + state update + terminal flag
                            |
                            v
              validate, accept, or abstain
```

Every transition records its observation, state delta, events, terminal flag, source, support,
uncertainty, and evidence IDs. TraceWorld must never convert unsupported counterfactuals into
confident invented evidence.

Retrieval queries are not based only on the action. They should represent:

```text
task/family intent
+ authentic prefix and current simulated history
+ current state summary and known entities
+ candidate action/tool and canonical arguments
+ offered tool contract and version
+ permissions, errors, and relevant prior events
```

The retriever may return exact, analogous, contrasting, error, and longer-range examples together.
The AWM—not the retriever—decides how those examples apply to the current simulated state. Even when
an exact-looking example is found, the AWM normally produces the response so it can substitute
entities, account for preceding state changes, preserve cross-turn consistency, and combine evidence
from several traces.

Directly returning an empirical observation is only an optional optimization for a separately proven
deterministic, state-equivalent case. It is not the default diagnose architecture. Similarity must be
effect-aware so actions with similar language but different mutations are not treated as
interchangeable. Predictions with little relevant evidence are the lowest-authority path and must be
reported or abstained separately.

### Phase 7: Construct, optimize, and validate the grounded AWM

Extract real action-to-next-observation transitions with complete history, tool identity, canonical
arguments, next observation, inferred state delta, terminal marker, and lineage.

The default path does **not** train world-model weights. Begin with an Experiential-style grounded
world model:

```text
history + action
      -> retrieve fit-only real transitions
      -> render a specialized environment prompt
      -> call a strong existing LLM or Qwen-AgentWorld
      -> strict structured transition
```

Optimize the specialized environment prompt with GEPA or a comparable reflective prompt optimizer:

1. replay real fit transitions under a candidate prompt;
2. compare predicted observations with recorded observations using rules and an external fidelity
   judge;
3. retain the judge's structured critique;
4. let GEPA propose revised environment prompts from those critiques;
5. select on a disjoint validation subset and preserve the base prompt if the winner does not survive
   a paired recheck;
6. open the sealed held-out fidelity split only for final reporting.

Retrieval and GEPA solve different parts of the problem: retrieval supplies enterprise-specific
transition evidence at inference time, while GEPA improves the stable instructions that tell the
existing model how to use that evidence and obey the transition protocol.

The action/reaction machinery proposed in PR #44 supplies two additional inputs, but they must not be
misnamed:

- its turn extractor provides observed `(action, reaction)` records for the transition corpus;
- its next-state judge and accepted deterministic checks estimate whether the observed reaction says
  the action progressed, failed, or is neutral;
- those hints can help GEPA diagnose bad simulated reactions and can provide step-level rollout
  diagnostics;
- they are **not** themselves an environment model or a complete fidelity judge. A fidelity judge
  must compare a predicted reaction with the real recorded reaction, including structured fields and
  state effects.

PR #44's `FamilyVerifier` is not a `VerifierSpec`. Its predicates flag bad `Turn`s, and its trace
“pass” means only “at least one observed turn and no flag”; the fractional score is the unflagged-turn
rate. Diagnose never treats either as task completion or reward. For tau2, accepted predicates are
debug/recovery diagnostics over a rendered rollout view. Terminal success comes from the external
state/event ledger through a reviewed state-based `VerifierSpec` and the rollout-claim adapter.

Only escalate to enterprise transition SFT or other world-model weight training if the optimized
prompt plus retrieval has measured residual errors that additional traces are likely to fix, and the
expected rollout volume justifies maintaining a separate model. Weight training is optional, not a
prerequisite for TraceWorld.

Validate on sealed held-out lineages using tool status/error accuracy, structured-field accuracy,
state-delta accuracy, terminal prediction, invariant violations, multi-step teacher-action replay
drift, confidence calibration, support/abstention coverage, and ensemble disagreement. Text
similarity alone is not a fidelity measure.

Validate the user policy independently. Given an authentic prefix, authentic preceding agent action,
and the hidden tau persona/goal, measure premature disclosure, correct disclosure after elicitation,
response-act/confirmation/refusal behavior, entity and goal consistency, termination, and abstention.
Use deterministic constraint checks where possible and an independent semantic rubric otherwise.
A tool-world fidelity pass cannot compensate for a user-policy fidelity failure.

#### Complete no-weight-training AWM stack

A production-usable prompt-only AWM still needs all of the following:

```text
existing capable model
+ versioned environment prompt
+ lineage-safe transition retrieval
+ externally maintained scenario state/event ledger
+ strict observation and state-delta schema
+ GEPA prompt optimization
+ separate fidelity judge and deterministic field comparison
+ support estimator and abstention
+ invariant/state-delta validator
+ bounded context, cost, retries, and caching
+ immutable artifact/version provenance
```

This is a complete AWM implementation strategy, not merely a prototype prompt. What it deliberately
does not provide is model-owned enterprise knowledge: it pays retrieval/context cost on each turn and
depends more heavily on the chosen foundation model. Weight training is an optimization for measured
fidelity, latency, privacy, or unit-cost needs—not a missing functional requirement.

### Phase 8: Base-model capability bakeoff

For every held-out task-start scenario and selected checkpoint:

1. reset an isolated TraceWorld branch;
2. run the candidate without showing future historical actions;
3. alternate candidate actions with hybrid TraceWorld transitions;
4. terminate on completion, budget, loop, unsupported action, or failure;
5. execute the reviewed external verifier;
6. repeat `n` times.

For `n` samples and `c` verified successes:

\[
pass@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
\]

Also report pass^k/reliability, verifier-unknown rate, unsupported-transition rate, loop rate,
transition-source composition, latency, tokens, cost, and task-family breakdowns.

Static checkpoint pass@k may be used as a cheap preliminary screen. It must be labeled
`next_action_capability`; it is not interchangeable with complete-rollout pass@k.

### Phase 9: Select the training route

| Evidence pattern | Route |
|---|---|
| Near-zero static and rollout pass@k | Reject or choose another base model |
| Static capability exists, interactive capability collapses | SFT on observation reading, planning, and recovery |
| High pass@k but low pass@1/pass^k | Strong OPD/RL candidate; capability exists but is unreliable |
| Correct tools but malformed arguments | Tool-call SFT or constrained decoding before RL |
| Strong only on learned-AWM-heavy rollouts | Block; possible simulator compatibility artifact |
| Strong, reliable, low-cost, high-support rollouts | Minimal adaptation or deployment candidate |

No promotion relies on a single aggregate score.

### Phase 10: Post-train and reevaluate

1. Cold-start SFT on verified fit demonstrations.
2. Rerun sealed trace checkpoints and complete TraceWorld rollouts.
3. Export verified corrective preference pairs.
4. Run agentic OPD inside TraceWorld.
5. Run conservative RL where verifier and transition support are sufficient.
6. Rerun capability-retention and family-level regression suites after every update.

The environment supplies observations. The verifier supplies reward evidence. A teacher may supply
better actions or critiques. These roles must not be conflated.

### Phase 11: Package, deploy, and continue learning

The enterprise handoff includes model weights/adapter, base identity, tokenizer, chat template,
system prompt, tool schemas, inference configuration, supported-family manifest, capability and
fidelity reports, cost/reliability measurements, known unsupported behavior, data lineage, and the
training recipe.

New production traces refresh family distributions, regression coverage, connector contracts, the
AWM transition corpus, environment conformance, and recovery cases. Real, executable-environment,
empirical, and learned-AWM trajectories remain separately labeled.

## Environment architecture

### Agent harness

Owns model invocation, prompts, tool exposure, context management, action parsing, retries, limits,
stopping, and trajectory recording.

### Scenario store

Stores immutable base scenarios and creates copy-on-write rollout branches. Parallel rollouts need
isolated logical state, not separate SaaS deployments.

### Canonical state and connector virtualization

Business-agent environments use shared canonical entities so CRM, ERP, email, calendar, and chat
connectors observe consistent state. Enterprise-specific connectors translate between their public
tool contract and that state.

### Coding backend

Coding tasks require a separate concrete backend: repository/base commit, starting diff, filesystem,
terminal, dependencies, services, network policy, and test verifier in an isolated container or VM.
A textual trace cannot reconstruct a missing repository or dependency state unless those artifacts
were captured or remain addressable.

### External verifier runner

Scores initial/final state, event deltas, invariants, and trajectory evidence. It cannot replace the
environment: a verifier judges consequences but does not generate the next observation.

The existing executor accepts `Evidence` extracted from recorded spans, so it cannot directly score
a simulated rollout. Diagnose adds a provenance-safe adapter:

```text
recorded Evidence -------\
                          > common verifier claim interface -> existing check semantics
rollout state/transcript-/                                  -> Result + world_origin
```

`rollout_claims` materializes the claim names that reviewed checks already consume, while retaining
field-level `recorded | executable | simulated` origin. It does not write simulated claims back as
`Evidence`, `Span`, or `TraceCorpus`. `execute_verifier(spec, evidence)` remains the replay-compatible
entry point; its operator logic is factored behind the common interface rather than duplicated.

Where expected outcomes differ per task, a reviewed `VerifierBinding` resolves placeholders from the
scenario's sealed success contract into a concrete `VerifierSpec`. Neither the candidate nor AWM can
supply expected values. The same template and binding rules are validated across historical labeled
tasks before promotion; the executor still receives an ordinary bound spec.

The sealed contract is a tagged union, not one terminal shape forced over an intent-level family:

```text
refusal       -> forbidden writes/unchanged state + required communication
informational -> operational not_applicable + required communication
mutation      -> required committed effects/events + communication
compound      -> multiple committed effects with multiplicity and optional partial order
```

A reviewed `ToolEffectCatalog` classifies tools as reads or writes; names such as `get_*` are not the
authority. Reads are process evidence only and cannot produce operational success. Required writes
are checked against committed effects, not attempted calls. Binding defaults to argument-aware
multiset matching: multiplicity matters and order is ignored unless the reviewed template declares a
sequence/partial order. Report process, operational, communication, and overall results separately;
never average them.

This also means simulation origin does **not** become a new bottom member of `EvidenceKind`.
Evidence authority and world origin are orthogonal. A verifier can only be calibrated/reviewed on
real held-out evidence; applying that frozen verifier to simulator claims yields a
`simulation_conditioned` verdict, never new promotion evidence.

### Rollout scheduler

Runs repeated isolated attempts with bounded steps, wall time, tokens, model/tool cost, retries, and
parallelism. It persists every attempt, including failures, unsupported transitions, and verifier
unknowns.

## AWM safety and epistemic boundaries

An external verifier reduces direct policy reward hacking, but it does not make learned transitions
true. A policy can exploit the learned AWM, or the AWM can generate a state that a correct verifier
then accepts.

Mitigations:

- prefer executable state for reward-bearing facts;
- validate proposed learned state deltas before committing them;
- penalize learned-AWM uncertainty and unsupported actions;
- cap learned-AWM rollout horizon until fidelity is demonstrated;
- use ensembles or independent models for disagreement;
- run metamorphic invariants and simulator-exploitation probes;
- forbid positive RL reward from unsupported rollouts;
- report results by transition-source composition;
- anchor every continual iteration in fresh real traces.

Passive traces cannot identify arbitrary unseen counterfactuals. TraceWorld therefore represents the
smallest reviewed behavioral world supported by the evidence, not a recovered copy of production.

## Current Bandits implementation audit

Status legend:

- **Built:** implemented, tested, and reachable through the current code/CLI.
- **Partial:** useful foundations exist, but the TraceWorld requirement is not complete.
- **Not built:** no current first-class implementation was found.
- **Open PR:** implemented in PR #44 but not present on the current checked-out branch or merged into
  its base.

### Effect of open PR #44

PR #44, *Verify trajectories by their reactions, not their transcripts*, adds a useful bridge from
Bandits' verifier plane toward TraceWorld:

```text
trace -> observed (action, reaction) turns
      -> next-state progress judge (+1/0/-1 + critique)
      -> RLM-proposed deterministic reaction checks
      -> human review
      -> trace/turn signals
```

It augments the plan in four ways:

1. Its action-to-reaction boundary rule can inform lossless transition extraction; clipped `Turn`
   payloads themselves cannot ground or fidelity-test an AWM.
2. Reaction judgments provide step-level behavior labels and critiques for capability diagnostics.
3. Accepted reaction checks can annotate final-state verifier results, but cannot change terminal
   success or contribute positive reward on AWM-generated reactions.
4. The judge's critiques are candidate feedback that GEPA may consume after a separate fidelity
   comparison establishes how a simulated reaction differs from the recorded one.

It does not yet provide current world state, scenario reset, empirical retrieval, next-observation
generation, connector execution, candidate rollouts, pass@k, or AWM fidelity measurement. Its tau2
result also demonstrates the boundary: when failure lives in hidden goal state and the simulated user
does not react, reaction-only scoring is near chance. TraceWorld still needs state and final-outcome
evidence.

### Trace and task-family plane

| Capability | Status | Current evidence |
|---|---|---|
| OTLP, Chat JSON, and Claude Code ingestion | Built | `bandits/ingest/` and tests |
| Explicit toolset extraction | Built | `bandits/ingest/toolsets.py` |
| Canonical trace/corpus contracts | Built | `bandits/traces.py` |
| Redaction and immutable storage | Built | `bandits/redact.py`, `store.py`, `ledger.py` |
| Task and outcome-evidence analysis | Built | `bandits/analyze/analysis.py`, `tasks.py`, `outcomes.py` |
| RLM task-family discovery and contracts | Built | `bandits/analyze/rlm_mine.py`, `rlm_models.py` |
| User-only/full-trajectory mining views | Built | `bandits/analyze/rlm_corpus.py` |
| Resumable bounded mining and audit sessions | Built | `rlm_session.py`, `rlm_audit_session.py` |
| Advisory adversarial family audit | Built | `bandits/analyze/rlm_audit.py` |
| Lineage/request-safe fit/held-out split | Built | `bandits/analyze/tasksets.py` |
| Fresh independent reassignment/stability gate | Not built | documented as removed in `docs/rlm-task-family-mining-plan.md` |
| Complete environment-state reconstruction | Not built | no scenario/state compiler exists |
| Action/reaction turn extraction | Open PR | PR #44 `bandits/verify/turns.py` |

### Verifier and learning-asset plane

| Capability | Status | Current evidence |
|---|---|---|
| Typed verifier/check contracts | Built | `bandits/verify/models.py` |
| Deterministic replay execution and drafting | Built | `execute.py`, `draft.py` |
| Human/model labels | Built | `bandits/labels.py`, `verify/judge.py` |
| Fit/held-out agreement, errors, and coverage | Built | `bandits/verify/validate.py` |
| Constructed verifier gaming probes | Built | `bandits/verify/validate.py` and tests |
| Human interview, review, promotion | Built | `interview.py`, `review.py`, `interpret.py` |
| Risk acceptance distinct from clean review | Built | `VerifierStatus.RISK_ACCEPTED` |
| Next-state progress judge (+1/0/-1 and hint) | Open PR | PR #44 `bandits/verify/nextstate.py` |
| RLM-proposed deterministic reaction checks | Open PR | PR #44 `bandits/verify/propose.py` |
| Human review and trace scoring for reaction checks | Open PR | PR #44 CLI and proposal artifacts |
| Live-query verifier execution | Partial | mode exists; current execution is trace-evidence oriented |
| Held-out eval export | Built | `bandits/export/eval.py`, `EvalCase` |
| Verifier-gated and direct SFT export | Built | `bandits/export/sft.py`, `direct_sft.py` |
| Tool-call/result pairing and quarantine | Built | export contracts/builders and tests |
| Preference-pair or OPD/RL rollout export | Not built | no first-class artifacts found |

### Capability and TraceWorld plane

| Capability | Status | Current evidence |
|---|---|---|
| Historical human-in-the-loop evaluation workflow | Built | `scripts/evaluate_bandits.py` |
| Static decision-point extraction | Partial via open PR | action/reaction turns exist in PR #44; candidate cases/prefix binding do not |
| Candidate registry/adapters and repeated sampling | Not built | no generic candidate runner found |
| pass@k/pass^k and confidence intervals | Not built | no implementation found |
| Agent harness and scenario/state contracts | Not built | no general interactive runtime found |
| Connector SDK and isolated rollout state | Not built | no connector execution layer found |
| Lossless grounding-transition extraction/retrieval | Not built | PR #44's clipped/string-rendered judge view is insufficient; canonical keys and retrieval do not exist |
| Retrieval-grounded AWM and GEPA prompt optimization | Not built | no current world-model or prompt-optimization package found |
| Optional AWM weight training | Not built | escalation only; not required for the initial system |
| Hybrid transition router and abstention | Not built | no transition runtime found |
| Interactive rollout scheduler/reporting | Not built | no rollout system found |
| Training/checkpoint orchestration | Not built | exports exist; trainers and model lifecycle do not |

## Build plan

### Diagnose vertical slice: what is actually built now

This sequence supersedes the broad destination milestones below for the current work. It targets
one tau2 family (`family-451ae91f975c`) and does not build connectors, an executable SaaS clone, a
universal enterprise schema, automatic environment generation, RL, or AWM weight training.

**Gate 0 — audit and bind real inputs.** Inspect the surviving legacy graph under
`work/tau/run/proj/.bandits/`, then run the current ingester on the same source and compare the
serialized contracts. If ids match, reuse the frozen graph. If they differ, record the field-level
contract delta and re-analyze/re-mine under new ids; do not chase the old hash or mix old and new
parents. Preserve the old target-family draft as provenance, but do not promote it: its checks contain
single-instance constants and were frequency-drafted without labels. Re-draft a parameterized
cancellation/refund `VerifierSpec`, then validate → interview/review → promote through the existing
`review-verifier` lifecycle. Add the narrow scenario-to-spec binding layer needed for task-varying
expected outcomes, and validate those binding rules on the historical labels. PR #44's
`review-checks` is a separate diagnostic lifecycle. Static and rollout scores remain development-only
until this gate closes.

**Gate 1 — static capability lab.** Compile task-start, many middle-prefix, recovery/error, and
end-prefix cases. Run each candidate `n` times at each case and judge action acceptability using tool
contracts, policy/precondition rules, reviewed reaction checks where available, and a bounded rubric
for genuine equivalence. The single recorded next action is a reference, not the unique gold action.
Publish coverage and judge uncertainty with `next_action_capability`. This is immediately useful and
needs no AWM, but is not pass@k.

**Gate 2 — grounded transition/fidelity lab.** Extract lossless structured
`GroundingTransition`s from fit lineages, strip declared control markers in a versioned derived view,
build lineage-safe retrieval, and call a frozen strong model through an injected predictor with a
strict transition contract. First measure the base prompt; then run GEPA on fit-only development
examples and select on a disjoint fit subset. Open the reserved tau tasks only for the final audit.
Gate the tool-world and user policies separately. The tool gate covers structured-field/status
accuracy, replay drift, state effects, invariant violations, and abstention. The user gate covers
premature disclosure, correct elicited disclosure, persona/goal consistency, response acts, and
off-support abstention. Do not proceed to capability pass@k if either role fails its reviewed
thresholds.

**Gate 3 — interactive capability lab.** Add isolated scenario state, a tool-world AWM role, a
separate user-policy role, candidate stepping, budgets/termination, and the rollout-claim verifier
adapter. Run the target OSS model, a strong reference, and a weak control against the exact same
scenario/AWM/user-policy versions. Publish simulation-conditioned pass@k, pass^k, unknown/support,
cost, and source-stratified failures. The reference/weak controls are sanity checks: if the lab cannot
order them sensibly, it cannot select an OSS model. Re-run the bakeoff with a second qualified
user-policy version; material candidate-rank changes are reported as simulator sensitivity and block
selection.

The minimal package is deliberately small:

```text
bandits/diagnose/
  models.py       # tagged success contracts, tool effects, scenarios, state, rollout/report contracts
  compile.py      # lossless transitions + start/middle/end cases + marker transforms
  retrieve.py     # fit-only, lineage-filtered evidence retrieval
  world.py        # tool-world and user-policy predictors; schema validation/abstention
  optimize.py     # GEPA prompt optimization and frozen AWM version
  fidelity.py     # held-out one-step and teacher-action multi-step fidelity
  candidates.py   # injected local/endpoint candidate adapters
  rollout.py      # reset/step loop, budgets, isolation, persistence
  verify.py       # rollout claims -> shared verifier claim interface
  report.py       # static and interactive metrics kept structurally separate
```

The CLI only needs thin commands for `compile`, `fidelity`, `static`, `rollout`, and `report`.
Tests mirror these modules with tiny fake traces and injected fake predictors; no model credential is
required in CI. Start with the base prompt before `optimize.py`: GEPA is an improvement stage, not a
precondition for proving that the contracts, leakage boundary, and scorer work.

**Validity boundary.** This vertical slice can validly rank candidates *on the versioned tau-derived
simulator* once Gates 0–3 pass. It cannot by itself claim production capability, support arbitrary
off-support actions, or provide sound RL reward. Those require later live/executable conformance or
fresh real traces. A low static score can reject a candidate cheaply; a high simulated pass@k is only
evidence to advance it to stronger validation.

### Milestone 1: Contracts and vertical skeleton

Build `CapabilityCase`, `Scenario`, environment state/snapshot, action/observation/transition,
connector/environment/candidate/verifier interfaces, rollout results, and budget contracts. Add one
deterministic candidate and connector for end-to-end tests.

**Exit:** one task can reset, act, transition, terminate, verify, and persist a lineage-bound rollout.

If PR #44 lands first, reuse a lossless span-boundary helper if it exposes one. Do not extend or use
its clipped/string-rendered `Turn` payload as AWM grounding data.

### Milestone 2: Candidate Capability Lab

Build checkpoint compilation, candidate endpoint/local-model adapters, repeated sampling, static
diagnostics, pass@k/pass^k with intervals, cost accounting, model comparison, and route recommendations.

**Exit:** candidates can be ranked on held-out cases, with static results labeled non-interactive.

### Milestone 3: TraceWorld core

Build reset/step/snapshot, immutable scenarios, isolated branches, connector SDK, event ledger,
permissions, clock, loop/limit termination, final-state verification, and bounded parallel rollouts.
Choose one family with two or three interacting tools.

**Exit:** complete-rollout pass@k works without a learned AWM.

### Milestone 4: Trace-to-environment compiler

Build the tool/error catalog, action canonicalization, entity/read/write/precondition hypotheses,
workflow discovery, minimum-consistent state reconstruction, provenance/review, and generated
trace-replay conformance tests.

**Exit:** a reviewed family specification generates scenarios and executable connector tests while
unknown behavior remains explicit.

### Milestone 5: Empirical transitions and retrieval-grounded AWM

Build lineage-safe transition indexes, equivalent empirical transitions, retrieval-grounded LLM and
Qwen-AgentWorld providers, a versioned specialized environment prompt, GEPA optimization against
recorded observations and external fidelity critiques, structured state deltas,
support/disagreement/uncertainty, abstention, fidelity evaluation, and the hybrid router. Do not train
AWM weights in this milestone.

**Exit:** complete rollouts mix executable, empirical, and learned transitions with source-stratified
results and low-support abstention.

### Optional milestone: AWM weight training

Fine-tune world-model weights only when the retrieval-grounded, GEPA-optimized baseline has a
well-characterized fidelity ceiling, sufficient transition data exists, and projected inference
savings or quality gains justify a separately versioned model. Compare it against the frozen
prompt-only baseline on the same sealed fidelity suite before adoption.

### Milestone 6: Interactive model selection

Run task-start and prefix-started candidates. Produce complete-rollout pass@k, reliability, support,
cost, and failure reports.

**Exit:** recommend reject, SFT-first, OPD/RL, or deploy with explicit evidence and limitations.

### Milestone 7: Post-training and regression

Integrate SFT execution, preference exports, OPD/RL collectors, external rewards, model/checkpoint
registry, retention suites, simulator/verifier exploitation tests, and deployment packaging.

**Exit:** train one candidate, reevaluate on sealed tasks, reject or promote it, and package it without
losing artifact lineage.

### Milestone 8: Optional ADWM experiment

Compare ADWM with static checkpoints, the ordinary learned AWM, and executable TraceWorld subsets.
Keep it only if it materially improves held-out rank correlation, calibration, or low-support value
estimation.

## Non-negotiable reporting

Every model report separates:

- static next-action capability from complete-rollout capability;
- task-start from prefix-started rollouts;
- executable, empirical, learned-AWM, live, and unsupported transitions;
- verified failure from verifier unknown;
- pass@k from pass^k/reliability;
- task-family macro results from traffic-weighted results;
- fit from sealed held-out evidence;
- real from generated trajectory provenance;
- model regression from environment/AWM fidelity regression.

No single success percentage is an adequate capability or deployment claim.
