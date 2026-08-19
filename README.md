# bandits

**Turn production agent traces into executable, verifiable RL environments.**

Your agent is already generating the raw material for its own training environment.
It's going into a logging bucket.

---

## The problem

Every "traces → evals" pipeline available today ends in the same place: an LLM judge.
Not because anyone wants that, but because it's forced by the data they kept.

If all you retained is prompts and completions, you can recover a task statement and a
final answer — and nothing you can check. So you ask a model whether the answer looks
right, calibrate the judge, and hope. Reward becomes an opinion, and every downstream
number inherits that fuzziness.

## The bet

Keep the **tool calls**, and the picture changes completely.

A single invocation point — `tool name · arguments · response · status · position` —
is enough to reconstruct the world instead of simulating it:

| From the record | You recover |
|---|---|
| Tool name + schema | the action space |
| Argument values | which parameters are actually used (usually ~5 of 38) |
| Response body | the response shape, and — via repeated IDs — the database behind it |
| Error responses | the failure modes the environment must reproduce |
| Position in sequence | reads before the first write = the starting state |
| State change across a write | what the verifier checks |

Rebuild the database, reimplement the tools on top of it, and reward stops being an
opinion. It becomes a state assertion:

```
orders[7741].status == 'refunded'
  AND orders[7741].refund_amount_cents == 4200
  AND effects.emails_to(88) == 1
  AND inventory unchanged
```

That last line matters. Assert what must **stay the same**, not just what must change,
or the agent learns collateral damage is free.

## The part nobody else does

Anyone can generate an environment. The question is whether it's a model of anything.

`bandits fidelity` replays a trace against the environment rebuilt from it and reports
agreement **per tool**:

```
get_order            48/48   100%
search_orders        12/12   100%
refund_order          9/10    90%
send_email            7/7    100%  (effects only, never sent)
update_order_status   3/8     38%   <-- FAIL
─────────────────────────────────
overall              79/85    93%   REJECTED (update_order_status below threshold)
```

An environment that can't reproduce its own source trace isn't a model of anything.
It gets rejected and sent back to schema inference.

Per-tool, never averaged: one tool at 38% tells you exactly what to fix. An aggregate
tells you nothing.

This is also the only claim here a customer can verify without trusting us — it's a
statement about *their* system, not about a model.

## Pipeline

```
bandits triage   traces.jsonl --source langsmith        ->  GO / PARTIAL / NO_GO
bandits ingest   traces.otlp.jsonl --tools tools.json   ->  corpus.json
bandits surface  corpus.json                            ->  surface.json    # action space + read/write/external
bandits schema   corpus.json surface.json               ->  schema.json     # the inferred database
bandits tasks    corpus.json schema.json                ->  tasks.json      # instruction + starting state
bandits verify   tasks.json corpus.json schema.json     ->  verifiers.json  # reward, as code
bandits fidelity corpus.json schema.json tasks.json     ->  fidelity.json   # ACCEPT / REJECT
bandits export   --harbor out/                          ->  Harbor tasks
```

Triage through fidelity is **fully deterministic** — no model calls, no API keys, no
network. That isn't a limitation, it's what makes the fidelity number mean something.

Adapters: `otlp`, `chat-json`, `langsmith`. Declared with `--source`, never sniffed.

## Run triage first

Everything above assumes the tool calls were retained. `bandits triage` asks that
question against a real export before anything is promised, and takes a position:

```
signal              observed   what it buys
invocation_points   yes  7/7   no action space without them; reward falls back to a judge
arguments           yes 21/23  replay, and the write semantics a verifier needs
responses           yes 23/23  the evidence the database is inferred from
identifiers         yes  6/8   recurrence is what turns responses into rows of a table
state_changes       yes 11/12  the change a verifier asserts on
error_modes         yes  2/23  an env that only succeeds trains an agent that never saw adversity

GO — reconstruction can proceed to the fidelity gate.
```

A `NO_GO` says the export is a transcript log, not a tool-call log, and no amount of
downstream work recovers that. It exits nonzero. A `GO` is a claim about the *data* and
never a promise about the gate — the gate is still the only number that means anything.

## Design commitments

**Reward is code.** State assertions plus an effect ledger. If we ever can't verify
something in code, we say so rather than reaching for a judge.

**Irreversible tools never fire.** `send_email` and `charge_card` are recorded to an
effect ledger and stubbed. The verifier asserts the email *would* have been sent.

**Unreimplementable tools raise.** They never return a plausible success. A faked
success silently corrupts every reward computed after it.

**Underdetermined structure is labeled, not invented.** An entity that's only ever read
and never cross-referenced becomes a static snapshot with a note in the manifest — not a
table with made-up fields.

**Shapes, not values.** Filler data is generated from observed types and formats, never
copied from real records. Traces are the most sensitive artifact a company owns.

## Status

v0, under active construction. Deterministic reconstruction (ingest → fidelity) is the
current build. Capability discovery, LoRA/GRPO training and active probing have defined
interfaces and stubs — see [BUILD_PLAN.md](BUILD_PLAN.md) for what's in and what's out.

Full design rationale, including the benchmark-first path and what we expect to break:
[docs/PLAN.md](docs/PLAN.md).

## Prior art

- **[WMO](https://github.com/experientiallabs/world-model-optimizer)** — excellent trace
  ingestion (11 vendor adapters, no format sniffing). Its trace-grounded environment is a
  text world model, so reward lands on a judge.
- **[AWM](https://github.com/Snowflake-Labs/agent-world-model)** (Snowflake, ICML 2026) —
  SQLite-backed executable environments with state-diff verification. Worlds are invented
  from a seed list rather than mined from anyone's traffic.
- **TRACE** ([arXiv 2604.05336](https://arxiv.org/abs/2604.05336), Stanford) — contrastive
  capability discovery from success/failure trajectories, then one targeted environment per
  deficit. Their table is the argument for targeting: generic synthetic envs (AWM) 38.4 on
  τ²-Bench, capability-targeted 48.2.

bandits takes ingestion discipline from the first, executable state-diff verification from
the second, capability targeting from the third — and adds the fidelity gate, which none of
them have, because none of them rebuild a world that something real can be compared against.
