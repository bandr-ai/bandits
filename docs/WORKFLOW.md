# The workflow

**How a team actually gets from a trace export to something a trainer can use.**

[docs/PLAN.md](PLAN.md) is the design. This is the loop you run. It is human-in-the-loop
on purpose, and the reason is worth stating once:

> Agents are poor one-shot environment generators, not because they are weak but because
> they are misaligned with what a team actually wants. Good environments come from
> infusing human feedback into the generation process.

That framing comes from LangChain's `eval-engineering` skill and we take it directly. What
we add is that our generation is *deterministic and checkable* — reconstruction from
invocation points, verifiers that are code, and a per-tool fidelity gate — so the human's
attention lands on the handful of judgements that genuinely require a person, instead of
being spent auditing a model's prose.

Three artifacts carry the alignment, and they are ordinary markdown files:

| File | The question it settles |
|---|---|
| `ENVIRONMENT.md` | What world are we rebuilding, and where does the reconstruction admit it does not know? |
| `TASKS.md` | Which real episodes become training tasks, and **where does the outcome label come from?** |
| `VERIFIER.md` | What exactly is being rewarded, and who read it? |

They are diffable and reviewable in a pull request, which is how an ML team will actually
use them: the diff is the record of what a person decided and when.

The agent-facing version of this same loop is `bandits/skill/SKILL.md`. Install that in
Claude Code or Codex and it runs the procedure with you.

---

## The loop

```
    export ──▶ Step 0  feasibility ──── no tool calls ──▶ STOP, honestly
                  │
                  ▼
               Step 1  map the surface ──▶ human corrects tool classes
                  │
                  ▼
               Step 2  ENVIRONMENT.md ──▶ human corrects the world
                  │
                  ▼
               Step 3  TASKS.md ──────▶ human supplies real outcome labels
                  │
                  ▼
               Step 4  VERIFIER.md ───▶ human signs off  (mandatory)
                  │
                  ▼
               Step 5  fidelity gate ──┐
                  │                    │ reject
                  ▼                    │
               Step 6  run an agent ───┘  (loop back to 2/3/4)
                  │
                  ▼
               Step 7  hand off to the trainer
```

---

# Worked example: the golden fixture

Everything below is real output from `tests/fixtures/traces.otlp.jsonl` — a small retail
support agent, 7 episodes, 9 declared tools.

```bash
.venv/bin/python -m bandits.cli run tests/fixtures/traces.otlp.jsonl \
    --source otlp --tools tests/fixtures/tools.json -o work/
```

## Step 0 — Feasibility

```
ingest: 7 traces, 23 invocation points, 0 issue(s)
```

23 invocation points across 7 traces. That is the whole product working or not working:
tool calls present means the action space, the response shapes and the state changes are
all recoverable, and reward can be a state assertion instead of an opinion.

**If that number had been zero, the correct answer is to stop.** An export with prompts
and completions but no `tool_calls` blocks supports exactly one thing — an LLM judging
whether the final answer looks right — and that is the failure mode this project exists to
avoid. The honest response is "your logging is missing the one field that matters, here is
how to turn it on, come back then", not a downgraded product.

At the same time, start asking for the **outcome signal**: ticket reopened, refund
reversed, escalation, settlement. It lives in a different system and usually with a
different team, and it is the long pole. You need it by Step 3.

## Step 1 — Map the surface

```
tool                  class      calls  errors  flags
escalate_to_human     unknown        0       0  declared-only (probing candidate)
get_customer          read           2       0
get_order             read           8       1
get_product           read           2       0
get_store_policy      read           2       0
refund_order          write          4       1
search_orders         read           1       0
send_email            external       3       0
update_order_status   write          1       0
```

Never show a class without its evidence. `refund_order` is a write because:

> observed state change: `refund_order(order_id=7741)` at step 4 of trace `ep-refund-ok`;
> `get_order` returned a different body for that id before (step 2) and after (step 5) the
> call [`status: "delivered" -> "refunded"`]

`send_email` is external because its responses are acknowledgement-shaped — no identifier,
no body, nothing a rebuilt store could have returned — *and* the name contains an
irreversible verb. The verb alone would never have been enough; a name is not behaviour.

`escalate_to_human` is declared but never called, so it stays `unknown`. It is not
reimplemented and calling it raises. It is also never probed to find out what it does —
you do not discover a tool was `charge_card` by charging a card.

### Where the human earns their keep

Every `read` above carries this evidence line:

> no before/after read window was available to test whether it changes state, so 'never
> writes' is unfalsified rather than proven

That is the honest statement of a real blind spot. A tool is classified `write` only when
the corpus shows the same row differing before and after. **A tool that mutates state and
is never followed by a read of what it changed leaves no evidence anywhere in the
corpus** — and gets classified `read`. Our classifier cannot detect this. The information
is not in the data.

Suppose `get_customer` also stamps `customers.last_seen_at`. Nothing in the traces reveals
it. If it ships as `read`, the rebuilt tool returns a row instead of changing one, any
verifier assertion about that field fails on every rollout, and the task looks *impossible*
rather than *mismodelled* — which at the pass@k gate is indistinguishable from "beyond the
model's ability". That would manufacture the exact result the gate exists to test for.

So `ENVIRONMENT.md` §2 asks about every read tool by name, and a `no` becomes a structured
override the pipeline consumes.

## Step 2 — `ENVIRONMENT.md`

Scaffold the workspace from the artifacts:

```python
from pathlib import Path
from bandits.contracts import StateSchema, ToolSurface
from bandits.skill.scaffold import scaffold_workspace

scaffold_workspace(
    "workspace",
    surface=ToolSurface.model_validate_json(Path("work/surface.json").read_text()),
    schema=StateSchema.model_validate_json(Path("work/schema.json").read_text()),
    name="golden-retail",
)
```

`ENVIRONMENT.md` §4 is the reconstructed world:

```markdown
### orders

- **Primary key:** order_id
- **Fields:** order_id, customer_id, sku, status, total_cents, placed_at, refund_amount_cents
- **Written by:** refund_order, update_order_status
- **Read by:** get_order, search_orders
- **Foreign keys:** customer_id -> customers.customer_id (1.00), sku -> products.sku (0.75)
- **Observed write semantics:**
  - `refund_order` key=`order_id` sets={'status': 'refunded'}
    arg_columns={'amount_cents': 'refund_amount_cents'} (confidence 0.80, 3 observation(s))
- **Write semantics correct:** **TODO(human)**
```

Note what is *not* claimed. `status = 'refunded'` is there because it was observed on every
successful call, not because the tool is named `refund_order`. That distinction survives
first contact with `cancel_order -> "cancelled"` and `approve_return -> "authorized"`;
name-guessing does not.

And the honest degradation:

```markdown
### store_policy

- **Primary key:** **TODO(human)** (no key was recoverable)
- **Static snapshot:** yes — the corpus never wrote this entity and nothing
  cross-references it, so no structure could be inferred. Rows are materialized
  verbatim. We refuse to invent a table here.
- **Acceptable as a snapshot:** **TODO(human)**
```

`get_store_policy` returns `{"policy": ...}` with no identifier in it. There is no row
handle, nothing writes it, nothing points at it. Publishing a `store_policy` table would be
a guess dressed up as structure, so we publish the observed rows and say so. The human
answers one question: do agents ever need to *write* this? If yes, the snapshot is wrong
and we need write evidence rather than a plausible schema.

Read the edits back:

```python
from bandits.skill.scaffold import read_back
o = read_back("workspace")
o.tool_classes          # {'get_customer': ToolClass.WRITE}
o.declared_blind_writes # ('get_customer',)
o.issues                # contradictions, e.g. blind write left as 'read' in the table
o.open_questions        # every **TODO(human)** still standing
```

`apply_overrides` folds those back into the `ToolSurface`, and the evidence trail records
who changed what:

```
human override in ENVIRONMENT.md: read -> write
  (declared a blind write: mutates without any follow-up read in the corpus)
```

A changed tool class changes the schema, so re-run `bandits schema` after applying.

## Step 3 — `TASKS.md`, and the label problem

```
tasks: 7 mined, 0 skipped
  warning task-ep-notfound: instruction references '9999' but no pre-state row carries that value
```

The task index, with the decision column:

| task_id | include | trace | label in export | warnings |
|---|---|---|---|---|
| task-ep-refund-ok | yes | ep-refund-ok | pass | - |
| task-ep-refund-ok-2 | yes | ep-refund-ok-2 | pass | - |
| task-ep-notfound | **TODO(human)** | ep-notfound | pass | no pre-state row carries '9999' |
| task-ep-double-refund | no | ep-double-refund | fail | no pre-state row carries '7741' or '88' |
| task-ep-bad-precondition | no | ep-bad-precondition | fail | - |
| task-ep-status-update | yes | ep-status-update | pass | - |
| task-ep-browse | yes | ep-browse | pass | - |

**Undecided reads back as excluded.** Silence never counts as approval, anywhere in this
workflow.

Each task also carries its reconstructed starting state and the reads that were *refused*:

```
task-ep-refund-ok
  Starting state: customers×1, orders×2, store_policy×1
  First write: step 4
  Post-write reads excluded from the starting state: 1
  Partial rows: {'orders': [7742]}
```

Order 7742 appeared in a `search_orders` result but was never read, so only the fields
named in that list are known. It exists in the starting state so search is not trivially
easy, and the verifier asserts only the fields it actually knows. The `get_order(7741)` at
step 5 is *after* the refund and describes the world post-change, so it is excluded — the
starting state is built from reads before the first write and never leaks backward.

### Then the hard part

`TASKS.md` §0 exists to have one argument, once:

> **"Reproduce what production did" is not a reward function.** It trains the model to
> copy the old system's mistakes, and it looks like it is working, because agreement is
> highest on the easy traces.

The `label in export` column is what the export claimed, not an outcome. Every task asks
separately:

```markdown
- **Downstream signal for this task:** **TODO(human)** (the real label: which system,
  which field, what value means success)
```

Push for a signal that already exists in their systems — ticket closed and not reopened in
7 days, refund not reversed, no escalation, payment settled. It nearly always exists and
has nearly never been joined to the traces. Get three things: the field, the join key back
to `trace_id`, and the lag before it is trustworthy.

If there is no such signal, the honest options are human review of a sample or dropping the
task. Low yield is a correct outcome, not a bug.

Also raise, here, the tasks the traces do *not* contain. Production is thin exactly where
the training signal is: failure paths. Because the environment executes, you are no longer
limited to what happened — non-refundable orders, unverified customers, rate-limited APIs.
That is where task volume actually comes from.

## Step 4 — `VERIFIER.md`

```
verify: 5 verifier(s), 2 refused
  no verifier task-ep-double-refund: refusing to synthesize a verifier from trace
  'ep-double-refund' with outcome=False. 'Do what production did' is only a reward
  function when production was labeled correct; a failed or unlabeled trajectory
  would become the target.
```

Those two refusals are the system working. The verifier for `task-ep-refund-ok`:

```markdown
### ver-task-ep-refund-ok

- **Task:** task-ep-refund-ok
- **Assertions:** 5

  - `state_unchanged` · `customers` == `[{'customer_id': 88, ...}]`
  - `state_equals` · `orders[{'order_id': 7741}].status` == `'refunded'`
  - `state_unchanged` · `orders[{'order_id': 7741}]` == `{...customer_id, sku, total_cents, placed_at}`
  - `state_unchanged` · `orders[{'order_id': 7742}]` == `{'order_id': 7742, 'customer_id': 88}`
  - `effect_count` · `effects[send_email]` == `1`

- **Assertions correct:** **TODO(human)**
- **Missing invariants:** **TODO(human)**
- **Unverifiable in code:** **TODO(human)**
- **Reviewed by:** **TODO(human)**
```

Four of the five assertions are about what must **stay the same**. That is deliberate: a
verifier that only checks the field that changed teaches the agent that collateral damage
is free as long as the target field is right. The `effect_count` assertion is how we reward
an email that was never sent — the attempt is in the effect ledger, the effect never
happened.

Read it against four questions:

1. Does it assert what must stay the same?
2. Is the expected value right, or merely what happened? These came from one episode. If
   production refunded the wrong amount, that amount is now the target.
3. Are effects asserted? For many agents success lives entirely in effects.
4. Is anything genuinely unverifiable in code? Then say so in the file. We do not reach for
   a judge to cover the gap — the task gets a narrower reward or gets dropped.

Then **sign it**. `reviewed_by` is unset on every synthesized verifier and
`bandits.verify.evaluate` raises `UnreviewedVerifierError` rather than grading with one.
An unedited workspace reads back as:

```python
o = read_back("workspace")
o.reviewed_by            # {}
o.is_reviewed            # False
o.unreviewed_verifiers   # every verifier id
```

and `apply_overrides` therefore hands the trainer nothing. That is the intended behaviour.
The agent running this loop must not type a name into that field on your behalf, even if
you tell it to — a signature it typed records nothing.

## Step 5 — The fidelity gate

```
tool                  matched  rate   note
get_customer          2/2      100%
get_order             7/8       88%   $status: call status differs
get_product           2/2      100%
get_store_policy      2/2      100%
refund_order          3/4       75%   $error_kind: error kind differs
search_orders         1/1      100%
send_email            3/3      100%
update_order_status   1/1      100%
──────────────────────────────────────
overall              21/23      91%   REJECTED
  refund_order at 75% (3/4) is below the 80% per-tool floor
```

91% overall, and **rejected anyway**. That is the design: one tool at 75% tells you exactly
what to fix, an average tells you nothing. The environment is not reproducing
`refund_order`'s failure mode — it returns a different `error_kind` than production did.
Failure modes are part of the dynamics; an environment that cannot return `not_found` trains
an agent that has never seen adversity.

Fix, re-run `schema → tasks → verify → fidelity`, paste the new table into
`ENVIRONMENT.md` §5 so the history lives in the diff. Exit the loop one of two ways:

- **accepted**, or
- **gaps consciously accepted** — written into §5, naming each failing tool and why the gap
  is tolerable, with a person's name attached.

Do not lower the threshold to pass. The fidelity number is the only claim in this system a
customer can check without trusting us, and it stops meaning anything the moment it is
tuned to pass.

## Step 6 — Put a real agent in it

Serve the environment (`bandits.serve` for HTTP and MCP, `bandits.rl` for `BanditsEnv` and
`TaskSuite`), point a capable agent at the included tasks, and read the transcripts rather
than the score. A lot of what is wrong is only discoverable by running:

- the agent calls an `unknown` tool and it raises — that path is naturally reachable;
- every rollout fails identically — the task is unsolvable, usually a missing pre-state row
  or a blind write from Step 1 that got through;
- one tool call solves it — a search tool returns only the answer; not enough filler;
- the agent wins without doing the work — run `bandits.verify.check_rollout` and add the
  domain-specific exploits to `VERIFIER.md` §3;
- the environment answers something production would have refused — a missing error mode.

Everything found here goes back into the markdown files. Two or three passes is normal.

## Step 7 — Hand off

Ship when all of these hold, and say which one failed if any does:

- [ ] fidelity accepted, or gaps recorded in `ENVIRONMENT.md` §5 with a name attached
- [ ] every included task has a real downstream outcome label in `TASKS.md`
- [ ] every shipped verifier has a human in `reviewed_by`
- [ ] `read_back(...).open_questions` is empty, or every remainder is consciously open
- [ ] anti-cheat run, domain exploits recorded
- [ ] the three markdown files committed — they are the review record

What the trainer gets is the served environment, the task suite, and frozen human-reviewed
verifiers. From there it is the ordinary path: pass@k sweep, routing decision, training,
non-regression check. Nothing downstream needs to know the tasks came from traces.

---

## What a good run looks like

For this fixture: 7 traces in, 7 with recoverable tool calls, 7 tasks mined, 2 refused a
verifier because production failed, 1 flagged unsolvable, 5 verifiers to review, 1 tool
unsupported, 1 entity degraded to a static snapshot, fidelity 91% and rejected on
`refund_order`.

Scaled up, a run that turns 500 traces into 40 solid tasks with real downstream labels is a
success. A run that turns 500 traces into 500 tasks graded by an LLM judge is the failure
this entire workflow exists to prevent.
