---
name: bandits
description: Build an RL environment from a company's agent traces. Use when the user says "build an RL environment from traces", "make an env from my agent traces", "turn traces into training environments", "train a model on our production agent logs", or hands over a trace export and asks what can be trained on it. Runs the deterministic bandits pipeline, then drives a human-in-the-loop alignment loop over ENVIRONMENT.md / TASKS.md / VERIFIER.md, a per-tool fidelity gate, and a mandatory human sign-off before anything grades a rollout.
---

# Build an RL environment from agent traces

You are running a human-in-the-loop procedure, not a batch job. The pipeline is
deterministic and produces facts; the human decides everything the traces cannot settle.
Your job is to run the deterministic parts, surface the decisions clearly, and refuse to
proceed past the ones that are unanswered.

Three rules that override any instinct to be helpful:

1. **Stop at Step 0 if tool calls cannot be recovered.** Text-only traces support an
   LLM-judge environment. That is not what this builds. Say so and stop.
2. **Silence is never approval.** A `**TODO(human)**` left in a file is an open question,
   not a default. Never fill one in on the human's behalf and never proceed as if it were
   answered.
3. **Never invent structure.** If the traces do not determine something, label it. A
   plausible fabrication is worse than a gap, because it is invisible downstream.

Setup, once:

```bash
cd <repo> && python -m venv .venv && .venv/bin/pip install -e .
```

Every command below is run with `.venv/bin/python -m bandits.cli ...` (or `bandits ...`
if installed on PATH). Nothing in steps 0–5 touches the network or calls a model.

---

## Step 0 — Feasibility. Be willing to stop here.

Ask for two things and do not conflate them:

- **the trace export** — what was actually called, with what values, returning what;
- **the tool registry** (`list_tools` output, an OpenAPI spec, or the function schemas
  from the request body) — the complete set of things the agent *can* do.

Ask which adapter the export is: `otlp` or `chat-json`. Do not guess and do not sniff.
A chat export misread as OTLP yields zero invocation points and looks like a tool-free
episode rather than an error.

```bash
.venv/bin/python -m bandits.cli ingest EXPORT --source otlp --tools tools.json -o work/corpus.json
```

Read the printed line: `N traces, M invocation points, K issue(s)`.

Then compute coverage explicitly:

```bash
.venv/bin/python - <<'PY'
import json, collections
c = json.load(open("work/corpus.json"))
tr = c["traces"]
withtools = [t for t in tr if t["invocations"]]
inv = sum(len(t["invocations"]) for t in tr)
labeled = [t for t in tr if t.get("outcome") is not None]
tools = collections.Counter(i["tool"] for t in tr for i in t["invocations"])
errs  = sum(1 for t in tr for i in t["invocations"] if i["status"] == "error")
print(f"traces                {len(tr)}")
print(f"with >=1 tool call    {len(withtools)}  ({len(withtools)/max(len(tr),1):.0%})")
print(f"invocation points     {inv}")
print(f"distinct tools        {len(tools)}")
print(f"error responses       {errs}")
print(f"traces with a label   {len(labeled)}")
print(f"ingest issues         {len(c['issues'])}")
for i in c["issues"][:20]: print("   ", i["kind"], i["detail"])
PY
```

**Report this table to the human, then take one of three branches.**

| Finding | What you do |
|---|---|
| Most traces carry invocation points | Continue to Step 1. |
| Some traces do, many do not | Report the split. Say the environment will be built from the tool-bearing subset and that its coverage is the coverage of *that* subset, not of production. Continue. |
| **Essentially none do** | **Stop.** Do not continue. |

The stop is not a failure of the tool, and you must not soften it. Say approximately
this, in your own words:

> Your export retained prompts and completions but not tool calls. From that I can
> recover a task statement and a final answer, and nothing I can check — which means
> reward would have to be an LLM's opinion of whether the answer looks right. That is
> the thing this pipeline exists to avoid, so I am not going to build it and call it an
> RL environment.
>
> What would fix it: tool calls in the log. In most stacks they are already in the
> assistant message's `tool_calls` blocks and the matching `tool`-role replies, they are
> just not being retained. If you can turn that on, a few thousand episodes is enough
> and we can start again then.

Do not offer an LLM-judge environment as a consolation. Offering it is how a project
ends up shipping one.

Also check now, in parallel, because it comes from a different system and usually a
different team: **is there any downstream outcome signal?** Ticket reopened, refund
reversed, escalation, payment settled. Ask at Step 0 even though it is not needed until
Step 3 — the answer often takes weeks to get.

---

## Step 1 — Map the surface, then have it corrected

```bash
.venv/bin/python -m bandits.cli surface work/corpus.json --tools tools.json -o work/surface.json
```

Present to the human, as a table:

- every tool, its inferred class (`read` / `write` / `external` / `unknown`), call count,
  confidence, and the **evidence string** for the classification — never the class alone;
- tools **declared but never called** — these stay `unknown`, are not reimplemented, and
  are the probing candidates if a staging environment ever becomes available;
- tools **called but never declared** — this usually means the export and the registry
  came from different systems or different dates. Flag it as a data problem.

Then ask the three questions the classifier cannot answer itself.

**1. Blind writes. Ask this every single time.** A tool is classified `write` only when
the corpus shows a before/after difference on the same row. A tool that mutates state and
is never followed by a read of what it changed leaves no evidence at all, and gets
classified **`read`**. The classifier is not weak here — the information is absent from
the data.

Go through every `read` tool by name and ask: *does this change anything?* Failing to
catch one is expensive and the failure is silent: the rebuilt tool returns a row instead
of changing one, every verifier assertion about that change fails identically, and the
task looks impossible rather than mismodelled — which at the pass@k gate is
indistinguishable from "beyond the model's ability".

**2. Irreversible tools.** Anything that spends money, sends a message, or calls a third
party must be `external`: stubbed and written to the effect ledger, never performed. If
the classifier called one of those `read` or `write`, that is a live incident, not a
rounding error. Ask directly: *is every tool here that touches money, messaging, or an
outside system, marked external?*

**3. Unknown tools.** Confirm each stays out of the environment. Calling one raises. Say
plainly why that is the right behaviour: a faked success silently corrupts every reward
computed after it.

---

## Step 2 — Align on `ENVIRONMENT.md`

```bash
.venv/bin/python -m bandits.cli schema work/corpus.json work/surface.json -o work/schema.json
```

Then scaffold the workspace:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from bandits.contracts import StateSchema, ToolSurface
from bandits.skill.scaffold import scaffold_workspace
surface = ToolSurface.model_validate_json(Path("work/surface.json").read_text())
schema  = StateSchema.model_validate_json(Path("work/schema.json").read_text())
paths = scaffold_workspace("workspace", surface=surface, schema=schema, name="acme-support")
print(paths.environment)
PY
```

`workspace/ENVIRONMENT.md` is now the reconstructed world: entities, primary keys, fields,
foreign keys, which tools write what, what degraded to a static snapshot and why, which
tools are unsupported and why.

Walk the human through it and be explicit about the honest gaps:

- a **static snapshot** entity is one that is only ever read and never cross-referenced.
  Nothing constrains its structure, so we materialize the observed rows verbatim rather
  than publishing a table we invented. Ask whether agents ever need to *write* it — if so
  the snapshot is wrong and we need write evidence.
- **partial rows** are ids production named in a list but never read. Only some fields are
  known. They exist so search and list tools are not trivially easy, and they must not be
  asserted on as if complete.
- **unattributed tools** returned a body we could not tie to any entity.

Then: the human edits the file. Do not edit their answers. When they say they are done:

```bash
.venv/bin/python - <<'PY'
from bandits.skill.scaffold import read_back
o = read_back("workspace")
print("class overrides   ", o.tool_classes)
print("blind writes      ", o.declared_blind_writes)
print("issues            ", o.issues)
for q in o.open_questions: print("OPEN", q.file, q.subject, q.question)
PY
```

Fix any `issues` (they are contradictions, e.g. a tool declared a blind write whose
decision cell still says `read`). Read the remaining open questions back to the human.
Do not answer them yourself. Loop until the ENVIRONMENT questions are closed or the human
consciously accepts them as open.

A tool class changed here changes the schema, so re-run `schema` after applying overrides.

---

## Step 3 — Align on `TASKS.md`, and get real outcome labels

```bash
.venv/bin/python -m bandits.cli tasks work/corpus.json work/schema.json -o work/tasks.json
```

Re-scaffold with the tasks in place (pass `overwrite=True` only after the human's
ENVIRONMENT edits have been read back and applied — the scaffold refuses to clobber edits
otherwise).

`TASKS.md` gets an `include` column per task, defaulting to `yes` for clean labeled tasks,
`no` for tasks whose export label says the episode failed, and `**TODO(human)**` for
anything carrying a solvability warning. Undecided reads back as **excluded**.

**Now do the hard part, and push.** For each task the file asks where the outcome label
comes from. The human will often answer "the trace shows what the agent did". Do not
accept that. Say it plainly:

> "Reproduce what production did" is not a reward function. It trains the model to copy
> the old system's mistakes, and it will look like it is working, because agreement is
> highest on exactly the easy traces where the old system was already right.

Then ask for a signal that already exists in their systems:

- ticket closed and **not reopened** within N days
- refund issued and **not reversed**
- **no** human escalation on the conversation
- payment settled / chargeback absent
- explicit CSAT or thumbs-up

For whichever they name, get three things: the table or field, the join key back to
`trace_id`, and the lag before the signal is trustworthy. All three go in TASKS.md §0.

If no downstream signal exists, the honest options are human review of a sample or
dropping the task. Say so. Low yield is a correct outcome, not a bug — expect a lot of
traces to produce no trainable task.

Also raise, in this step, the tasks the traces *do not* contain. Production is thin
exactly where the training signal is: failure paths. Because the environment executes,
you are no longer limited to what happened — non-refundable orders, unverified customers,
rate-limited APIs. Ask which of those are reachable in their real system. This is where
task volume actually comes from.

---

## Step 4 — Align on `VERIFIER.md`. Sign-off is mandatory.

```bash
.venv/bin/python -m bandits.cli verify work/tasks.json work/corpus.json work/schema.json -o work/verifiers.json
```

Synthesis refuses to write a verifier from a trace labeled failed or unlabeled — a
verifier built from a failed trajectory makes that trajectory the training target. Report
those refusals as refusals, not as errors.

Re-scaffold so `VERIFIER.md` carries the assertions, then review each one with the human
against four questions:

1. **Does it assert what must stay the same?** A verifier that only checks the field that
   changed teaches the agent that collateral damage is free. Look for `state_unchanged`.
2. **Is the expected value right, or just what happened?** Assertions came from one
   episode. If production refunded the wrong amount, that amount is now the target.
3. **Are effects asserted?** For many agents success lives entirely in effects — an email
   sent, a charge attempted — not in stored data.
4. **Is anything unverifiable in code?** Then say so in the file. Do not reach for a
   judge. The task gets a narrower reward or gets dropped.

Then the gate:

```bash
.venv/bin/python - <<'PY'
from bandits.skill.scaffold import read_back
o = read_back("workspace")
print("signed    ", o.reviewed_by)
print("unsigned  ", o.unreviewed_verifiers)
print("reviewed? ", o.is_reviewed)
PY
```

`reviewed_by` is unset on every synthesized verifier, and `bandits.verify.evaluate`
raises `UnreviewedVerifierError` rather than grading with one. **Do not write a name into
`Reviewed by:` yourself under any circumstance**, including when the human says "just put
my name". Ask them to edit the file — the signature is a record that a person read the
assertions, and a signature you typed records nothing.

Apply the decisions:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from bandits.contracts import ToolSurface, TaskCase, Verifier
from bandits.skill.scaffold import read_back, apply_overrides
surface = ToolSurface.model_validate_json(Path("work/surface.json").read_text())
tasks = [TaskCase.model_validate(t) for t in json.loads(Path("work/tasks.json").read_text())["tasks"]]
vers  = [Verifier.model_validate(v) for v in json.loads(Path("work/verifiers.json").read_text())["verifiers"]]
applied = apply_overrides(read_back("workspace"), surface=surface, tasks=tasks, verifiers=vers)
print("kept tasks     ", [t.task_id for t in applied.tasks])
print("dropped tasks  ", applied.dropped_tasks)
print("kept verifiers ", [v.verifier_id for v in applied.verifiers])
print("withheld       ", applied.dropped_verifiers)
PY
```

Everything withheld here is withheld because a human did not approve it. Report the counts
to the human; do not route around them.

---

## Step 5 — Run the fidelity gate. Per tool. Loop.

```bash
.venv/bin/python -m bandits.cli fidelity work/corpus.json work/schema.json work/tasks.json \
  --surface work/surface.json -o work/fidelity.json
```

This replays each recorded trace against the environment rebuilt from it and compares
every response. It **exits nonzero when it rejects**; it is a gate, not a report.

Read the per-tool table, never the overall number alone. One tool at 38% tells you exactly
what to fix; an average tells you nothing. Common causes and their fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| A write tool below the floor | write semantics wrong: the wrong key argument, or a constant we did not observe | correct the write effect in ENVIRONMENT.md §4 and re-run `schema` |
| `$status: call status differs` | the environment returns ok where production errored | the error mode is missing; check `surface.json` error modes for that tool |
| `$error_kind` differs | right failure, wrong kind | failure modes are part of the dynamics — an env that cannot return `not_found` trains an agent that has never seen adversity |
| A read tool below the floor | the entity behind it is wrong, or degraded to a snapshot when it should not have | ENVIRONMENT.md §4, then re-run `schema` |
| High `unsupported` count | tools left `unknown` | classify them in Step 1, or accept the coverage loss explicitly |

Loop: fix, re-run `schema` → `tasks` → `verify` → `fidelity`. Paste each new per-tool table
into `ENVIRONMENT.md` §5 so the history is in the diff.

Exit the loop one of two ways, and be explicit about which:

- **accepted** — overall at or above threshold, no tool below the per-tool floor; or
- **consciously accepted gaps** — the human writes into §5 which tools remain below the
  floor and why that is tolerable. This is a decision with a name attached, recorded in a
  file, not a silently lowered threshold.

Never lower `--threshold` or `--per-tool-floor` to make the gate pass. If the human asks
for that, say what it costs: the fidelity number is the only claim here they can verify
without trusting us, and it stops meaning anything the moment it is tuned to pass.

---

## Step 6 — Put a real agent in it and watch it break

A lot of what is wrong is only discoverable by running. Serve the environment (see
`bandits.serve` for the HTTP and MCP endpoints, and `bandits.rl` for `TraceEnv` /
`TaskSuite`), point a capable agent at a handful of included tasks, and read the
transcripts yourself. Do not read the score.

What to look for, in rough order of how often it bites:

- **The agent calls a tool that raises.** An `unknown` tool is on a path the agent
  naturally takes. Either classify it or accept that those tasks are out of scope.
- **The task is unsolvable.** The starting state is missing a row the instruction names.
  `tasks.json` flags some of these as solvability warnings; running finds the rest. Every
  rollout failing identically is the signature.
- **The task is trivially solvable.** One tool call and done, usually because a search or
  list tool returns only the answer. Filler rows exist to prevent this; if there are not
  enough, say so.
- **The agent wins without doing the work.** Reward hacking. Run
  `bandits.verify.check_rollout` and add domain-specific exploits to VERIFIER.md §3. A
  rebuilt world is simpler than reality and admits strategies reality does not.
- **The environment answers something production would have refused.** A missing error
  mode. Back to Step 5.

Feed everything you find back into the markdown files, re-run the gate, and tell the human
what changed. Two or three passes here is normal.

---

## Step 7 — Hand off

Confirm all of the following before you tell anyone this is ready. If one fails, say which:

- [ ] fidelity **accepted**, or gaps written into `ENVIRONMENT.md` §5 with a name attached
- [ ] every included task has a **downstream outcome label** named in `TASKS.md` §0/§2 —
      not "what production did"
- [ ] every shipped verifier has a **human** in `reviewed_by`
- [ ] `read_back(...).open_questions` is empty, or every remaining item is one the human
      explicitly chose to leave open
- [ ] anti-cheat run and domain exploits recorded
- [ ] the three markdown files are committed — they are the review record

Then export and hand to the trainer:

```bash
.venv/bin/python -m bandits.cli run EXPORT --source otlp --tools tools.json -o out/
```

What the trainer receives: the served environment, the task suite, and frozen
human-reviewed verifiers. From there it is the ordinary path — pass@k sweep, routing
decision, training, non-regression check. Nothing downstream needs to know the tasks came
from traces.

Close with an honest summary, in this shape:

> N traces in, M with recoverable tool calls, T tasks, V reviewed verifiers, fidelity X%
> overall with the per-tool breakdown attached. Y tools remain unsupported. Z entities are
> static snapshots. The outcome label is <the real downstream signal>. Here is what I
> could not determine and what would fix it.

Yield is supposed to be low. A run that turns 500 traces into 40 solid tasks with real
labels is a success. A run that turns 500 traces into 500 tasks graded by an LLM judge is
the failure this whole procedure exists to prevent.
