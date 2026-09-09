# RLM Task-Family Mining Plan

## Goal

Add an experimental RLM miner that discovers reusable task families from raw agent traces without embedding geometry, predefined domains, task categories, or action schemas.

Two traces belong to one family when the same parameterized verifier contract could correctly evaluate what their users requested.

Keep the existing mutual-kNN miner as a baseline until the RLM path demonstrates better semantic coherence, stability, and downstream verifier transfer.

## Input boundary

The primary experiment gives the RLM only user-role messages from each trajectory:

- Include the initial request and later user clarifications, corrections, choices, and confirmations.
- Exclude assistant messages, tool calls/results, final state, rewards, evaluator labels, and known task/family/lineage labels.
- Keep opaque trace IDs only for evidence references.

This is not semantic preprocessing. The system only recognizes trace boundaries and recorded message roles. If roles are unavailable, mark the trace unreadable instead of guessing.

Run a separate first-user-message-only arm to measure whether later user turns add useful task information or agent-dependent noise.

## Input-view experiment

Run the pure-RLM miner through two primary input paths:

### Path U: user messages only

The RLM receives every user-role message in the trajectory and no assistant messages, tool activity, outcomes, or rewards.

This path tests whether requested work and later user clarification are sufficient to discover reusable task families.

### Path F: full trajectory

The RLM receives the complete raw conversation, including user and assistant messages, tool calls, and tool results. Rewards and evaluator labels remain hidden.

This path tests whether execution context helps the RLM infer the underlying task or instead makes it cluster by agent behavior, tool choice, or success path.

Both paths must use:

- the same traces;
- the same model and prompts except for the declared input view;
- the same chunk size, budgets, seeds, and stop conditions;
- fresh, independent discovery and assignment contexts;
- the same human-review and stability evaluation.

Do not allow either path to see the other path's taxonomy. Compare them only after both runs finish.

Report:

- human-accepted family coherence;
- over-merging and fragmentation;
- assignment stability across repeated runs;
- ambiguous and uncovered coverage;
- correlation between family membership and reward, final tool, tool sequence, model, and trajectory length;
- downstream verifier transfer to unseen requests.

Path F wins only if it improves semantic family coherence and verifier transfer without primarily separating traces by execution behavior or outcome. If it mostly produces categories such as successful runs, handoffs, refusals, or common tool sequences, it is useful for behavior mining but not as the task-family miner.

## End-to-end flow

    Raw trajectories
      -> user-message views
      -> iterative RLM taxonomy discovery
      -> adversarial contract audit
      -> frozen family contracts
      -> fresh RLM assignment of every trace
      -> ambiguity and coverage review
      -> repeated-run stability analysis
      -> optional TaskSet materialization

Embeddings and kNN are excluded from the initial experiment so the semantic hypothesis is tested cleanly.

## Read-only corpus interface

Expose only generic access operations:

    count_traces()
    list_trace_ids(offset, limit)
    get_user_messages(trace_id)
    get_user_message_batch(trace_ids)
    sample_trace_ids(count, seed, exclude=())

The RLM may write working hypotheses and decisions, but cannot mutate source traces or inspect excluded fields.

## Family contracts

Every proposed family must record more than a name:

    {
      "contract_id": "contract-...",
      "name": "Short task-family name",
      "definition": "What user-requested work belongs here",
      "inclusion_rules": ["..."],
      "exclusion_rules": ["..."],
      "required_outcome_shape": [
        "What a verifier must establish for every member"
      ],
      "supporting_trace_ids": ["..."],
      "counterexample_trace_ids": ["..."],
      "revision": 1
    }

Contracts must distinguish requests requiring different mutations or materially different success checks, even when their domain and wording are similar.

## Iterative discovery

For the current 160-trace experiment:

- Shuffle trace IDs with a recorded seed.
- Inspect 20 traces per chunk.
- Process all eight chunks.
- Revisit old assignments after every second chunk.
- Run a complete unresolved-case sweep after every trace has been seen.

The first chunk creates a provisional taxonomy. Every later chunk can:

    KEEP
    CREATE
    REVISE
    SPLIT
    MERGE
    MARK_AMBIGUOUS
    MARK_UNCOVERED

Every operation needs a rationale and supporting trace IDs. Chunk boundaries never become family boundaries.

Each iteration should mix:

- unseen traces;
- ambiguous and uncovered traces;
- traces affected by recent changes;
- random previously assigned traces;
- suspected counterexamples.

## Adversarial audit

Use a fresh RLM context to challenge every provisional contract:

- Find its least-compatible pair of members.
- Identify differences in requested work and required outcomes.
- Find the strongest apparent member currently outside it.
- Detect topical groupings whose members require different verifiers.
- Compare against the sibling contracts for boundaries that were split too narrowly.
- Recommend keep, revise, split, merge (with a target contract ID), or uncertain.

Audit recommendations are advisory and nothing applies them. Discovery may carry
out a MERGE during a later pass; there is no reviewer command that applies an
audit's merge recommendation, and no re-audit of a merged contract, so a merge
finding is discharged today by resolving it or by freezing over it. Both remain
Phase C work.

The discovery RLM must explicitly resolve or preserve every audit finding.

## Stop conditions

Freeze the taxonomy only after two consecutive complete sweeps:

- create no family;
- perform no split or merge;
- make no material contract revision;
- move fewer than 2% of provisional assignments;
- have inspected every trace;
- have audited every family;
- record reasons for all ambiguous or uncovered traces;
- leave no audit recommendation unaddressed.

Enforce hard iteration, model-call, elapsed-time, and monetary budgets. Hitting a limit produces an incomplete artifact, never a successful taxonomy.

## Fresh assignment

Discovery assignments are provisional because their definitions changed during the loop. After freezing:

1. Compute a content-addressed taxonomy ID.
2. Start a fresh RLM context.
3. Classify every trace against the frozen contracts.
4. Forbid taxonomy changes during this pass.

Each result records:

    {
      "trace_id": "...",
      "matching_contract_ids": ["..."],
      "primary_contract_id": "... or null",
      "status": "assigned | ambiguous | uncovered | unreadable",
      "reason": "Concise evidence-based explanation"
    }

Never force a match. Preserve multiple matches and uncertainty for reconciliation and human review.

## Stability

Repeat the complete discovery and assignment process at least five times with different recorded input orders or seeds. Runs must not share taxonomies.

Compare runs by trace co-assignment rather than generated family names. Report:

- stable-assignment fraction;
- pairwise co-assignment agreement;
- recurring split/merge disagreements;
- consistently ambiguous or uncovered traces;
- semantic contracts that recur across runs.

If needed, reconcile the independent taxonomies into a consensus taxonomy, then subject it to a new assignment and audit pass.

## Artifact lineage

Keep stages separate and content-addressed:

    rlm-mining-run
    rlm-taxonomy-draft
    rlm-taxonomy-audit
    rlm-taxonomy
    rlm-assignment-run
    rlm-stability-report
    rlm-taskset

Each artifact records parent IDs, model, prompt digest, sampling seed, budgets, timestamps, and limitations. A draft must never silently become an accepted TaskSet.

Ambiguous, uncovered, and unreadable traces remain visible and are excluded from automatic verifier drafting until reviewed.

## CLI sketch

    bandits mine-rlm ANALYSIS_ID \
      --view user-messages \
      --chunk-size 20 \
      --max-iterations 20 \
      --seed 42 \
      --project PROJECT

Follow-up stages:

    bandits audit-rlm-taxonomy TAXONOMY_DRAFT_ID --project PROJECT
    bandits assign-rlm-taxonomy TAXONOMY_ID ANALYSIS_ID --project PROJECT
    bandits validate-rlm-mining ASSIGNMENT_RUN_ID --project PROJECT
    bandits materialize-rlm-taskset ASSIGNMENT_RUN_ID --project PROJECT

Discovery, assignment, validation, and materialization remain separate operations.

## Evaluation

Compare:

1. Current embedding mutual-kNN mining.
2. One-shot LLM clustering.
3. Iterative RLM over user messages plus fresh assignment.
4. Iterative RLM over full trajectories plus fresh assignment.
5. Iterative RLM over user messages with adversarial audit.
6. Iterative RLM over full trajectories with adversarial audit.

Do not use embedding-mined families as ground truth. Measure:

- blind human acceptance of family coherence;
- over-merging and fragmentation;
- ambiguous and uncovered coverage;
- stability across runs and corpus samples;
- correlation with outcome, tool usage, or trajectory length as leakage diagnostics;
- downstream verifier transfer to independently reviewed requests.

The decisive test is whether a verifier drafted from some family members correctly evaluates unseen, independently requested work assigned to that family.

## Implementation phases

### Phase A: Pure-RLM prototype

- Add user-message trace views.
- Add discovery, contract, iteration, and assignment models.
- Implement the read-only corpus interface.
- Implement iterative discovery and fresh assignment.
- Run on the 160-trace corpus without embeddings.

### Phase B: Reliability

- Add structured response validation.
- Add resumable checkpoints and budgets.
- Add adversarial audits and formal stop conditions.
- Add five-run stability measurement.

### Phase C: Bandits integration

- Add honest RLM provenance to TaskSet without fabricated similarity fields.
- Preserve unresolved traces.
- Connect reviewed families to selection and verifier drafting.
- Add reviewer actions for accept, revise, split, and merge.

### Phase D: Enterprise scaling

- Shard corpora mechanically.
- Recursively reconcile shard-level contract proposals.
- Reassign traces against frozen global contracts.
- Process novel and ambiguous traffic during incremental updates.
- Introduce embedding retrieval only after pure-RLM semantic quality is established.

## Go/no-go criteria

Continue toward production only if the RLM path demonstrates:

- materially fewer human-identified over-merges;
- stable assignments across independent runs;
- explicit and useful ambiguity handling;
- no dominant clustering by agent behavior or outcome;
- auditable evidence for every contract;
- better downstream verifier transfer;
- acceptable measured cost and latency.

If it only creates better names for incoherent groups, the experiment failed. If contracts are coherent but assignment is unstable, improve assignment separately rather than hiding the instability.

The governing principle is:

> The RLM maintains a falsifiable taxonomy, repeatedly tests it against raw user requests, freezes it, and sends every membership decision through an independent pass.
