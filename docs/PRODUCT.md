# bandits — the product concept

**Scope of this document:** the environment layer only. The library an MLE (or a coding
agent acting for one) points at a company's traces to get back a running RL environment
with a code verifier. The platform — runs, dashboards, training orchestration — is
deliberately out of scope here and lives in a separate doc. Everything below is written
assuming the platform will exist and will consume what this library produces.

---

## The context we're building into

Enterprises run on frontier APIs today. It works well enough off the shelf, so they build
a harness around it — and the harness gets shaped by the model. The way Sonnet uses a
filesystem is visibly different from the way Opus does. So the harness is the lock-in:
swapping the model breaks the accuracy, and now you're in someone's ecosystem whether you
chose it or not.

Meanwhile two things are happening at exec level:

1. They've been storing agent traces and trajectories for a while. Millions of them, in
   Arize or LangSmith or a bucket. And no private evals — so they cannot say what their
   accuracy actually is. You can't improve what you can't measure.
2. As model companies climb into the application layer, the hypothesis forming is: to keep
   any strategic or economic value, we need **specialized models**, and that requires
   **unique data and unique environments**. Which is the one thing they already have and
   aren't using.

The first move is always testing open-weights models — that's the cost story, and it's the
easy sell. The hard part is what comes after: how do you beat the frontier on your own use
case, continuously?

SFT alone doesn't get there. It's backprop on next-token prediction; it's good for teaching
a model the right tool schema and the right output shape, and that's roughly where it
stops. RL/OPD is where the real improvement lives, because you're optimizing against a
reward rather than imitating a transcript.

Which puts the entire weight of the problem on one thing: **where does the reward come
from?**

---

## Why this isn't already solved

To run RL you need a bespoke environment that mimics production closely enough that the
failure modes are learnable. Building one today needs an AI researcher who understands the
data, understands OPD/RL/SFT, can drive a distributed RL library like slime, *and* can hand-
write the environment and the verifiers. That's a rare person, and every customer needs
them for months before a single reward is computed.

The industry's shortcut is an LLM judge, and it's not a choice — it's forced by what got
retained. If all you kept is prompts and completions, you can recover a task statement and
a final answer and nothing checkable. So you ask a model whether the answer looks right,
calibrate the judge, and hope. Reward becomes an opinion and every number downstream
inherits the fuzziness. For evals that's tolerable. For RL it's a broken reward signal you
will optimize directly into.

## The bet

**Keep the tool calls and the picture changes completely.**

An *invocation point* — `tool name · arguments · response · status · position` — is enough
to reconstruct the world rather than simulate it:

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

That last line is the one people forget. Assert what must **stay the same**, not only what
must change — otherwise the agent learns that collateral damage is free.

---

## What the product is

A library, driven either by an MLE directly or by a coding agent (Claude Code / Codex) that
we ship a skill for. You give it traces plus the tool registry. You get back an executable
environment, tasks, and verifiers.

```
bandits ingest   traces.otlp.jsonl --tools tools.json   ->  corpus.json
bandits surface  corpus.json                            ->  surface.json    # action space
bandits schema   corpus.json surface.json               ->  schema.json     # inferred database
bandits tasks    corpus.json schema.json                ->  tasks.json      # instruction + start state
bandits verify   tasks.json corpus.json schema.json     ->  verifiers.json  # reward, as code
bandits fidelity corpus.json schema.json tasks.json     ->  fidelity.json   # ACCEPT / REJECT
bandits serve    --http --mcp                           ->  reset/step/reward
```

Ingest through fidelity is **fully deterministic** — no model calls, no API keys, no
network. That isn't asceticism; it's what makes the fidelity number mean anything. The
moment an LLM is in the reconstruction path, the gate is grading its own homework.

Two shapes of environment come out, and they do different jobs:

- **Mirror environment** — rebuilds the customer's actual world from their traffic. Fewer
  tasks, but it looks like their system. This is what evals run on and what proves to a
  customer that we understand their agent.
- **Skill environment** — small, isolated, trains one measured deficit. Copies the tool
  interface, not the world. Seeded task generation, so instances are unlimited. This is
  what most of the RL compute actually burns on.

## The part nobody else does

Anyone can generate an environment. The question is whether it's a model of anything.

`bandits fidelity` replays the source trace against the environment rebuilt from it and
reports agreement **per tool**:

```
get_order            48/48   100%
search_orders        12/12   100%
refund_order          9/10    90%
send_email            7/7    100%  (effects only, never sent)
update_order_status   3/8     38%   <-- FAIL
─────────────────────────────────
overall              79/85    93%   REJECTED (update_order_status below threshold)
```

Per-tool, never averaged. One tool at 38% tells you exactly what to fix; an aggregate tells
you nothing. Below threshold the environment is rejected and sent back to schema inference
rather than shipped.

This is also the only claim in the whole stack a customer can verify **without trusting
us** — it's a statement about their system, not about a model. It's the thing you can put
in front of a skeptical MLE lead in the first meeting.

Honest bar: 50–80% reconstruction fidelity on a real customer's traffic is a good outcome
for v1, and it's enough to be useful — you train on the tools that cleared the gate and you
say plainly which ones didn't.

## Commitments that hold under pressure

These exist because each one, violated, silently corrupts every reward computed after it.

- **Reward is code.** State assertions plus an effect ledger. If we can't verify something
  in code, we say so rather than reaching for a judge.
- **Irreversible tools never fire.** `send_email`, `charge_card` land in an effect ledger
  and are stubbed. The verifier asserts the email *would* have been sent.
- **Unreimplementable tools raise.** Never a plausible success. A faked success poisons
  everything downstream of it.
- **Underdetermined structure is labeled, not invented.** An entity only ever read and never
  cross-referenced becomes a static snapshot with a note in the manifest — not a table with
  made-up fields.
- **Shapes, not values.** Filler data is generated from observed types and formats, never
  copied from real records. Traces are the most sensitive artifact a company owns, and this
  is a procurement question long before it's an engineering one.
- **Reproducibility is a digest, not a promise.** Every env ships `schema_digest`,
  `verifier_digest`, `spec_digest`. A run that logs reward without these can't say what it
  trained against.

## The handoff to training

The library's job ends at a reward you can trust. What it hands over:

- `EnvSpec` — task, action space, step budget, reward range, the three digests.
- A served environment over HTTP and MCP: `reset` / `step` / `reward`, so a trainer can
  drive thousands of concurrent episodes without importing our pipeline.
- The fidelity report, which is also the honest scoping document: these tools are
  trustworthy, these are not.

The training recipe — filtering the best traces deterministically then with a judge,
evaluating candidate open-weights models, choosing between OPD and RL — is the second moat
and a separate build. It only becomes tractable once reward is real, which is this library's
entire reason to exist.

---

## Who this is for

A company that:

- is afraid of frontier labs eating their strategic layer, and has said so out loud;
- has already tried open weights, probably already tried SFT, and hit the ceiling;
- has an ML team — we are not the ML team;
- has traces in Arize / LangSmith / anywhere structured, **with tool calls retained**;
- has enough eval to measure a delta, or wants us to build the private eval first;
- has a high-volume task that doesn't need frontier-level intelligence, and is willing to
  host and maintain a model — including paying to retrain it later.

**The disqualifier is one question:** did you keep the tool calls? No invocation points →
no action space → no state → no verifier → judge. Ask it before promising anything.

Why the timing works: open models keep improving, inference costs keep falling, and the
set of tasks where a company wants to own the intelligence — volume, privacy, cost, control
— keeps growing.

## Where we sit against the field

| | What they do | Where the wedge is |
|---|---|---|
| **Applied Compute** ($80M) | Platform for runs/envs/data, FDE into the Fortune 100 | You bring the environment. That's the hard part, and it's the part we automate. |
| **Trajectory.ai** ($44M) | Full harness evolution around SDPO, FDE-heavy | Same — the env is an input, not an output. |
| **Belevidir** (YC, closed alpha) | Prompt evolution and SFT | Doesn't reach RL, so never has to solve reward. |
| **Prime Intellect** | Public env hub; bring data and envs, train | Envs are contributed by hand, not mined from your traffic. |
| **WMO** | Excellent trace ingestion, 11 vendor adapters | Its environment is a text world model, so reward lands on a judge. |
| **AWM** (Snowflake, ICML 2026) | SQLite envs with state-diff verification | Worlds invented from a seed list, not mined from anyone's traffic. |
| **TRACE** (Stanford) | Contrastive capability discovery, targeted envs | Their own numbers are the argument for targeting: generic 38.4 → targeted 48.2 on τ²-Bench. |

We take ingestion discipline from WMO, executable state-diff verification from AWM,
capability targeting from TRACE — and add the fidelity gate, which none of them have,
because none of them rebuild a world that something real can be compared against.

Neolabs / bespokeai / Mercor are plausible indirect entrants.

## How the FDE motion feeds the library

Do the whole thing by hand for the first customers — the ingestion, the schema fixes the
gate rejects, the verifiers, the training run. We've done this for our own use cases and
for papers, never for a production company. The work is to find which parts are actually
repeatable, encode those into the library, and keep the rest as service until it isn't.
Every manual schema fix is a bug report against inference.

## Open questions

**About the customer**
- What fraction of real telemetry actually retains tool calls, arguments *and* responses —
  not just spans and token counts?
- Can traces leave their VPC at all, or does the whole library have to run inside it?
- How stable are their tool schemas over the window of trace history we'd ingest?
- Do they have an eval good enough to detect the delta we produce, or is the private eval
  the actual first deliverable?

**About us**
- What fidelity is the real floor for RL to be worth running — is 80% enough, is 60%?
- How much of schema inference is genuinely deterministic before an LLM becomes necessary,
  and can we keep it out of the reward path when it does?
- Where does the mirror env stop being enough and skill envs have to take over?
- What's the honest failure rate of verifier synthesis, and what happens to tasks we refuse
  to write one for?
- Does the coding-agent-driven path (skill in Claude Code / Codex) genuinely reduce the
  MLE's time, or does it just relocate the debugging?
