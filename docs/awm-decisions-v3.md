# AWM / traces2rlenv — decision record, v3

Continues `reasoning-awm-decisions.md` (D1–D26) and
`reasoning-awm-decisions-v2.md` (D27–D42). This record starts from the
post-implementation audit in `awm-issues.md`.

## Session log

| # | What | Outcome |
|---|---|---|
| S1 | Audit the implemented emulate modules | I1–I29 recorded in `awm-issues.md` |
| S2 | Establish repair order | Tests first; metric correctness; per-call world semantics; real DSPy GEPA; orchestration |
| S3 | Add numerical-core regression tests | Seven intended failures reproduced before implementation changes |
| S4 | Repair first numerical-core batch | Fidelity lists/abstention, split-safe critiques, task-level reports fixed |
| S5 | Pin actual GEPA API with an injected test | Production path is `dspy.GEPA(...).compile(...)` |
| S6 | Close external-validation P0 implementation | JSON Schema outputs and validator-owned per-call event provenance built; verification intentionally not run yet |

Verification was subsequently run and is recorded in `awm-decisions-v4.md` (S7-S9):
897 passed, 1 failed. Most of S6's claims hold; per-call batch completeness does not
(D70), and the fidelity gate does not apply runtime validation (D71). D43-D67 below
stand except where v4 supersedes them.

## Decisions

### D43. Repair work is test-first. **Decided**

Before changing `fidelity.py`, `optimize.py`, or `report.py`, add adversarial tests that
reproduce the audit findings. A green suite containing no tests for the numerical core is
not evidence that the measurement is correct.

### D44. Abstention is an externally grounded validity outcome. **Decided**

The AWM may not use subjective uncertainty to escape a prediction. Abstention is permitted
only when a versioned support policy, computed from eligible retrieved evidence, says the
requested transition is unsupported. Correct abstention, wrong abstention, and supported
coverage are separate measurements. High abstention blocks capability publication.

### D45. GEPA means the public `dspy.GEPA` optimizer. **Decided**

The existing one-shot `dspy.Predict` prompt rewriter is a reflective baseline, not GEPA.
The production optimizer will call `dspy.GEPA`; the underlying `gepa` package remains an
implementation detail of DSPy. The fidelity metric supplies both a scalar score and
structured textual feedback.

### D46. Capability statistics are task-level statistics. **Decided**

pass@k and pass^k are computed within each scenario and macro-averaged across scenarios.
Rollout pooling is invalid because it weights tasks by attempt count. Uncertainty is
estimated over scenarios, not over correlated attempts.

### D47. Abstention cannot improve the optimization objective by hiding supported cases. **Decided**

Conditional prediction accuracy is multiplied by supported-transition coverage and wrong
abstention is penalized separately. Correct abstention on an unsupported transition does
not reduce supported coverage.

### D48. Tool-world and user-policy compatibility are different relations. **Decided**

Tool-world retrieval ignores task success shape because APIs do not change behavior based
on whether the agent should have called them. User-policy retrieval requires the same
success shape because the person's goal and disclosure behavior do depend on the task.

### D49. Candidate identity pins every inference-affecting setting. **Decided**

Endpoint/provider, model, prompt, temperature, token limit, seed, and prompt version are
part of the digest. An explicitly empty offered-tool set means every attempted tool call is
unoffered; it does not disable validation.

### D50. User-policy support includes the sealed profile, not retrieval alone. **Decided**

Unlike a tool outcome, a user reply is grounded first by the scenario's hidden user profile;
retrieved user turns provide behavioral analogies. An empty retrieval result therefore does
not itself force abstention when the requested reply is determined by the profile. Support
calibration must measure profile coverage and behavioral retrieval separately.

### D51. AWM evidence citations and mutation scope are enforced outside the model. **Decided**

Every cited transition ID must be in the eligible examples retrieved for that step. When a
reviewed tool catalog declares mutation-path patterns, every proposed delta must match one;
the prompt's instruction to stay grounded is not treated as enforcement.

### D52. Fidelity compares complete deltas and accumulates the predicted world. **Decided**

Delta fidelity compares values as well as paths. Multi-step drift teacher-forces recorded
actions but carries the AWM's predicted ledger into the next step; restoring recorded state
would turn it back into unrelated one-step tests.

### D53. A batch is validated and committed per call ID. **Decided**

Every batched call has an independent outcome carrying execution, observation/error,
deltas, events, and evidence IDs. Aggregate effects on a multi-call action are rejected
because they cannot establish which call caused them. Step-level fields are derived from
the validated call outcomes.

### D54. A fidelity gate constrains coverage as well as conditional accuracy. **Decided**

The gate accepts explicit minimum supported coverage and maximum wrong-abstention rate.
Accuracy on the subset the AWM chose to answer cannot establish simulator validity alone.

### D55. Static probes judge one action without requiring action imitation. **Decided**

The candidate receives only the authentic `CandidateView` at the compiled cut. A judge may
use the hidden recorded next transition as evidence, but exact equality with that action is
not required because multiple actions may be valid. No AWM or simulated future participates.

### D56. Production rollouts require a reviewed, calibrated support policy. **Decided**

The campaign overrides development defaults with a content-identified policy naming its
reviewer and fidelity calibration report. `NONE` can never be a commit threshold.

### D57. Campaigns accept candidate factories and fail before candidate spend. **Decided**

Each rollout gets a fresh candidate instance. The fidelity gate and binding completeness
run before construction or model calls. Candidate reports must share the pinned environment,
and a configured competent reference must beat its negative control.

### D58. A campaign artifact contains every attempt, not only aggregates. **Decided**

The content-addressed derived artifact binds its fidelity/support references, candidate
reports, and all pass/fail/unknown/abstained rollouts. Aggregate-only persistence would
make denominator and failure analysis impossible to audit.

### D59. Calling an unavailable tool is candidate failure, never AWM abstention. **Decided**

Tool availability is known deterministically from `CandidateView`. The rollout rejects the
call before retrieval or AWM invocation and keeps it in the capability denominator as a
verified candidate-side failure.

### D60. Candidate-owned terminal failures override partial task effects. **Decided**

Step, token, and time limits, action loops, and unavailable-tool calls always produce an
overall candidate failure. A useful effect performed before looping forever does not turn
the unfinished rollout into a pass.

### D61. User disclosure is compared by normalized fact ID where available. **Decided**

Structured `known_facts` and `unknown_fact_ids` are the authoritative disclosure contract.
Legacy prose retains an explicitly weaker lexical fallback until re-ingestion can normalize
it; lexical tokens are not presented as equivalent fidelity evidence.

### D62. Offered tool schemas validate candidate calls before the AWM. **Decided**

Unavailable tools, missing required arguments, and incompatible primitive argument types
are deterministic candidate failures. They never reach retrieval and cannot turn into AWM
abstentions. Output-schema validation remains unavailable where the source tool contract
does not declare an output schema.

### D63. Declared outputs use standards-compliant JSON Schema validation. **Decided**

`ToolSchema.output_schema` preserves `output_schema` and `outputSchema` declarations from
direct and OpenAI-wrapped tool contracts, including valid boolean JSON Schemas. The emulate extra directly declares
`jsonschema`. Invalid outputs and invalid schemas reject the transition before commit;
missing declarations produce a per-call `unavailable` status and are never described as
validated. An invalid result remains marked `invalid` on its rejected rollout step so the
persisted attempt is auditable. Adding the canonical field changes re-ingested corpus
identities by design.

### D64. The validator owns canonical event provenance. **Decided**

Per-call events are stamped with `_call_id`; a conflicting model-supplied ID rejects the
transition. `RolloutStep` consumes validator-produced events and schema statuses, never raw
AWM events. A required event satisfies an expected effect only when its type and call ID
match the same committed call selected by tool and arguments. Unattributed legacy events
remain compatible only when the event ledger contains no per-call identities. State changes
and emitted events both require eligible retrieved evidence before they can be committed.

### D65. The callable campaign API is the supported execution surface. **Decided**

`run_campaign(...)` plus content-addressed persistence is sufficient for correctness and
automation. A CLI is deferred as convenience work and is not a P0 validity requirement.

### D66. Rejected attempts retain output-validation status. **Decided**

Output status has three explicit values: `validated`, `unavailable`, and `invalid`. An
invalid output or invalid declared schema rejects the transition before commit, but the
`invalid` status remains on the rejected `RolloutStep`. Otherwise the persisted campaign
would record that validation failed while discarding which validation failed.

### D67. Evidence support is per effect-producing call. **Decided**

In a batch, a call proposing a state delta or event must cite its own eligible retrieved
evidence. Evidence cited by another call in the same action cannot support it. Aggregate
single-call effects follow the same rule, and all cited IDs must still belong to the
retrieval result for that exact rollout step.
