# Environment experiment loop

This is the operating loop for proving that a reconstructed environment is
useful before attaching a teacher model, OPD, or RL training job.

## The rule

An experiment is only a success when an agent's improvement in the rebuilt
world predicts improvement on data the reconstruction did not use. A high
training reward alone proves nothing.

```text
trace export
  → reconstruction + fidelity diagnosis
  → accepted, reviewed training tasks
  → recorded rollouts
  → held-out environment / production-outcome evaluation
  → diagnose failures and improve telemetry or reconstruction
  ↺
```

## Experiment 0 — baseline reconstruction

Run the deterministic pipeline on a fixed corpus and save every artifact:

```bash
.venv/bin/python -m bandits.cli run \
  tests/fixtures/traces.otlp.jsonl \
  --source otlp \
  --tools tests/fixtures/tools.json \
  -o /tmp/bandits-exp-baseline
```

The bundled retail fixture baseline is intentionally **rejected**:

| Tool | Result | What to do |
|---|---:|---|
| `refund_order` | 3/4 (75%) | Do not train. Inspect the failed trace for missing pre-state or an unmodelled error mode. |
| `get_order` | 7/8 (88%) | Same root trace; do not hide it in the aggregate. |
| all other replayed tools | 100% | Keep as evidence, not as an excuse to waive the failure. |

`ep-double-refund` calls `refund_order` before an observable order row exists.
The source says "already refunded" while the reconstructed store says
"not found". The correct repair is an earlier production read/snapshot or an
explicit customer-provided rule; it is not inventing an already-refunded row.

## Experiment 1 — diagnosis and repair

For every non-matching invocation, record exactly one primary repair class:

| Class | Meaning | Repair |
|---|---|---|
| `missing_pre_state` | The task did not reveal the row/state needed by a call. | Obtain earlier reads or a safe snapshot; otherwise exclude. |
| `missing_error_mode` | The tool's failure response was not modelled. | Add trace coverage or an approved deterministic rule. |
| `write_semantics` | A write changed the wrong fields/rows. | Improve state inference with before/after evidence. |
| `response_shape` | State is right but the tool response projection differs. | Preserve observed response fields/nullable behavior. |
| `unsupported_tool` | No safe implementation exists. | Keep unsupported; never fake success. |

**Exit condition:** overall threshold and every judged per-tool floor pass.

## Experiment 2 — training-ready admission

Build a suite only with all gates enabled:

```text
positive downstream outcome label
AND no unresolved solvability warning
AND human-reviewed verifier
AND accepted per-trace fidelity report
AND membership in the train split
```

Use `TaskSuite.from_pipeline(..., require_fidelity=True)` so a replay-rejected
world cannot enter a training suite accidentally.

## Experiment 3 — rollout capture

Start with deterministic scripted/oracle actions. Capture the complete rollout:

```text
environment/spec digests
task ID and split
actions and observations, in order
terminal reward and every verifier assertion
anti-cheat findings
model identity, prompt version, latency, token/cost data (when a model is used)
```

**Exit condition:** replayed oracle trajectories score as expected and a failed
trajectory is visibly distinguishable from an anti-cheat failure.

## Experiment 4 — held-out evaluation

Split before reconstruction/model tuning. Do not infer a world from a trace and
then claim performance on that same trace is evaluation.

```text
train traces     → reconstruction, teacher rollouts, OPD/RL
held-out traces  → fidelity/evaluation only
production       → downstream outcome measurement, after its label delay
```

Start with trace-level splits. Upgrade to customer/entity/time-window splits
when IDs recur across traces, otherwise the same order/customer can leak from
train to evaluation.

## Experiment 5 — teacher, OPD, then RL

Only after Experiments 0–4 pass:

1. Run a frontier teacher in accepted training environments.
2. Keep only terminally successful, verifier-clean trajectories.
3. Distill those trajectories into an open model (SFT/OPD).
4. Use RL against the same deterministic reward.
5. Compare the resulting policy on the held-out set and downstream outcomes.

The teacher trajectory is training data, not the success metric. A model ships
only when it improves held-out results.
