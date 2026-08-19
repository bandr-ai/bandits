# TASKS — <name>

<!--
TEMPLATE. `scaffold_workspace` writes a populated copy from real TaskCases.
Same three parsed constructs as ENVIRONMENT.md; see that file's header comment.
The `include` column in section 1 is the ONLY place inclusion is decided.
-->

Each task has an instruction taken from a real user turn and a starting state
reconstructed from the reads that happened *before* the first write.

## 0. Where the outcome labels come from — read this before the table

**"Reproduce what production did" is not a reward function.** It trains the model to
copy the old system's mistakes, and it looks like it is working, because agreement is
highest on the easy traces. A trace tells you what the agent *did*. It does not tell
you whether that was right.

The label has to come from outside the trace. Best to worst:

1. **A downstream signal you already have.** Ticket closed and not reopened in 7 days.
   Refund not reversed. No human escalation. Payment settled. CSAT thumbs-up. This
   almost always exists somewhere in your stack and has almost never been joined to
   your traces. It usually lives with a different team than the traces do — start
   asking now.
2. **Human review of a sample.** Expensive, real, and sometimes the only option.
3. **Throw the trace away.** Not every trace becomes a task. Low yield is a correct
   outcome, not a bug.

- **Downstream signal available in your systems:** **TODO(human)** (name the table/field, or say none)
- **Join key from that system to a trace_id:** **TODO(human)**
- **Lag before the signal is trustworthy:** **TODO(human)** (e.g. 7 days for a reopen)

## 1. Task index

Edit the **include** column: `yes` trains on this task, `no` drops it. Anything left as
**TODO(human)** is treated as **excluded** — silence never counts as approval.

| task_id | include | trace | label in export | warnings | instruction |
|---|---|---|---|---|---|
| <task_id> | **TODO(human)** | <trace_id> | unlabeled | - | <instruction> |

### Refused by mining

- `<trace_id>` — <reason>

## 2. Task detail

### <task_id>

- **Instruction:** <instruction>
- **Source trace:** `<trace_id>` (digest `<sha256>`)
- **Tools used:** <tools>
- **Label in export:** <pass/fail/unlabeled> — this is what the export claimed, not a
  downstream outcome
- **Downstream signal for this task:** **TODO(human)** (the real label: which system,
  which field, what value means success)
- **Starting state:** <entity×rows, ...>
- **First write:** <step or none>
- **Post-write reads excluded from the starting state:** <n>
- **Solvability warning:** <warning>
- **Warning resolved:** **TODO(human)** (an unsolvable task makes every rollout fail
  identically, which at the pass@k gate is indistinguishable from "too hard")

## 3. Tasks the traces do not contain

Production is thin exactly where the training signal is: failure paths. Now that the
environment executes, you are no longer limited to what happened. Make the order
non-refundable, the customer unverified, the API rate-limited — situations that never
occurred but are entirely reachable in the real system.

- **Situations worth generating that production never produced:** **TODO(human)**
