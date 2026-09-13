# Next-state verifiers

A verifier that reads reactions instead of transcripts. This replaces, for the
three archetypes it covers, the per-family deterministic checks in
`bandits/verify/draft.py` — which were drafted from evidence fields, reviewed
through a five-prompt interview, and could not re-enter validation once revised.

## Why

Every attempt to induce success from a whole trajectory failed on the same wall
(`docs/sft-signal-experiment.md`, `docs/rlm-signal-discovery-plan.md`): the
transcript shows the *action*, and whether the action was right lives in a
goal state the transcript never contains. Trajectory-shape signals scored AUC
~0.50 on tau2 and Spearman ~0.05 on TRAIL; a holistic judge scored below the
majority baseline.

What a transcript *does* contain is what happened after each action: the tool
result, the execution log, the user's next message. OpenClaw-RL
(arXiv:2603.10165) trains on exactly this — the reward for turn *t* is the
environment's reaction at *t+1*, judged locally. A judge that sees only
`(action, reaction)` never has to reconstruct the goal; it reads whether the
reaction says the action worked.

## The pipeline

```
ingest  ──►  turns  ──►  judge-turns  ──►  propose-verifier  ──►  review-checks  ──►  score-traces
             (a_t, s_t+1)   +1 / 0 / −1       RLM writes checks     human accepts     flags per turn
                            per turn          re-executed here      one prompt each   pass/score per trace
```

**`bandits/verify/turns.py`** cuts a `Trace` into `Turn`s: each MODEL span is
an action; the TOOL spans and user turns before the next MODEL span are its
reactions. A turn with no reaction is *unobserved* and never counted — the
final turn of most episodes, and every planning call that another model call
follows directly.

**`bandits/verify/nextstate.py`** — the judge. One call per observed turn, shown
the task, the previous action as context, the action, and the reaction. It
returns a `HINT:` line and `\boxed{+1|0|-1}`; the last boxed score wins,
majority over `--votes` samples with ties reading as 0. The **archetype**
(`support`, `coding`, `computer-use`, `generic`) changes only the paragraph
telling the judge what a reaction means in this kind of trace — a write tool
rejecting a call, a traceback the agent's own code caused, a page that did not
change. `TraceSignal` is the count: `score = 1 − negative/scored`, `passes`
when no turn was −1.

**`bandits/verify/propose.py`** — the verifier the RLM proposes. A `dspy.RLM`
is shown a stratified sample of the family's turns *with* the judge's verdicts
and hints, and writes `check(turn) -> True | False | None` predicates in its
REPL, where it can run them against the sample before committing. Nothing it
says about its own predicates is kept: every check is re-executed in an AST
sandbox (no imports, no dunders, no `open`/`eval`, may not read `judge` or
`hint`) over **every** turn of the family, and scored against the judge:

- `precision` — of the judged turns it fired on, the share the judge scored −1
- `recall` — of the judge's −1 turns, the share it fired on

A check *survives* when it fired on at least `--min-fired` judged turns, on
fewer than 80% of all turns, and its precision clears `--min-precision`. A
second round shows the model what survived and why the rest did not.

**`review-checks`** is the human loop, kept deliberately small: hypothesis,
code, the numbers, three example turns it fired on, and one prompt —
`[a]ccept/[r]eject/[s]kip/[q]uit`. Every decision saves a new artifact, so a
review can stop anywhere and resume from the latest id. The decision is the
human's; a check that did not survive can still be accepted with `--all`, and
a survivor can be refused.

**`score-traces`** applies the accepted checks (or `--survivors`) and, unless
`--no-judge`, the judge's −1s. A trace passes when no turn was flagged; its
score is the share of observed turns that were not.

## Running it

```bash
uv run bandits ingest /path/to/trail/data/GAIA --source trail --project work/trail-ns
uv run bandits judge-turns <corpus-id> --archetype computer-use --project work/trail-ns
uv run bandits propose-verifier <turn-judge-id> --project work/trail-ns
uv run bandits review-checks <family-verifier-id> --project work/trail-ns
uv run bandits score-traces <family-verifier-id> --project work/trail-ns
```

Restrict to one mined family with `--task-set <id> --family <id>` on
`judge-turns`; without it the whole corpus is treated as one family, which is
what a benchmark split is.

## Measuring it: TRAIL

`bandits/ingest/trail.py` reads TRAIL's smolagents span trees. Each `Step`
becomes action → tool calls → an `execute` reaction carrying the execution log,
with span ids kept verbatim because TRAIL's human annotations locate every
error by span id. That gives a **turn-level** test — does a −1 land where a
human placed an error? — on top of the trace-level `overall` score, which
nothing before could be measured against.

```bash
uv run python scripts/trail_nextstate_eval.py --project work/trail-ns \
    --judge-run <id> [--scores <verifier-scores-id>] \
    --trail-dir /tmp/trail-benchmark/benchmarking --split gaia --out work/trail-ns/eval-gaia.json
```

Reports precision/recall/F1 of judged-negative (and verifier-flagged) turns
against annotated error locations, the base rate, the "tool span errored"
baseline, and Spearman/AUC of the trace score against `overall`.

## Results on TRAIL (2026-09-13)

Model: `nemotron-lightning-3p5-30b-a3b`, one vote at temperature 0. Turn-level
numbers are against TRAIL's annotated error locations on observed turns; the
base rate is the share of observed turns that carry an annotation. Trace-level
numbers are Spearman of the trace score against the human `overall` (1–5) and
against the number of annotated errors, and AUC of top vs bottom tertile by
`overall`.

TRAIL annotates a few errors per trace, not every misstep, so turn-level
precision is a floor: sampled "false positives" on SWE were all real mistakes
(a regex that matched nothing, a search in the wrong file) that no annotator
marked.

**SWE-bench (31 traces, 466 turns, 465 observed), coding archetype**

| scorer | turn P | turn R | F1 | ρ overall | ρ −errors | AUC |
|---|---|---|---|---|---|---|
| base rate | 0.40 | — | — | — | — | — |
| tool span errored (baseline) | 0.45 | 0.13 | 0.21 | — | — | — |
| judge, prompt v1 | 0.44 | 0.48 | 0.46 | 0.32 | 0.20 | 0.65 |
| judge, prompt v2 (lenient) | 0.47 | 0.29 | 0.36 | 0.15 | 0.04 | 0.53 |
| judge, prompt v3 (strict for coding) | 0.43 | 0.44 | 0.44 | 0.34 | 0.24 | 0.66 |
| **RLM checks only** (7 survivors, anchored to v1) | 0.51 | 0.26 | 0.34 | **0.43** | 0.25 | **0.74** |
| RLM checks only (3 survivors, anchored to v2) | 0.44 | 0.12 | — | 0.11 | — | 0.56 |
| RLM checks only (10 survivors, anchored to v3) | 0.37 | 0.21 | 0.27 | 0.15 | 0.09 | 0.59 |

**GAIA (117 traces, 1508 turns, 751 observed), computer-use archetype**

| scorer | turn P | turn R | F1 | ρ overall | ρ −errors | AUC |
|---|---|---|---|---|---|---|
| base rate | 0.38 | — | — | — | — | — |
| tool span errored (baseline) | 0.58 | 0.39 | 0.47 | — | — | — |
| judge, prompt v1 | 0.42 | 0.64 | 0.51 | 0.11 | 0.60 | 0.56 |
| judge, prompt v2/v3 (lenient) | 0.55 | 0.51 | 0.53 | 0.14 | 0.59 | 0.59 |
| **RLM checks only** (5 survivors) | **0.59** | 0.46 | 0.52 | 0.06 | **0.64** | 0.52 |

**tau2 retail (120-trace subset of 456, 827 turns), support archetype, sealed
truth** — run `turn-judge-d450cf34d97b27a2` in `work/tau2-ns`. The user's
replies are now in the trace (`scripts/tau2_to_otlp.py` carries them, `otlp.py`
reads them), so the reaction to every agent message is the customer's next
line plus any tool results.

| scorer | AUC(score, success) | passes | precision of passes | base rate |
|---|---|---|---|---|
| judge, clean share | 0.51 | 59 / 120 | 0.70 | 0.65 |
| judge, negative count | 0.54 | 59 / 120 | 0.70 | 0.65 |

Chance. This is the predicted failure, not a surprise: tau2's simulated user
never pushes back, and tau2's failures — the wrong variant exchanged, the wrong
order cancelled — live in a goal-state diff the transcript never shows. The
judge's −1s are real friction (the user repeating details, a tool rejecting
an id), but friction and outcome are uncorrelated here. Sibling consensus
(AUC 0.82 on the full set, `docs/sft-signal-experiment.md`) remains the only
signal that works on tau2, and it needs k>1 rollouts. Support needs a corpus
with real users to be tested at all.

What this says:

- The reaction carries the signal the transcript did not. Every earlier
  trajectory-level signal on TRAIL GAIA was Spearman ≤ 0.08; the judge's clean
  share is 0.59–0.60 against annotated error count, and the deterministic
  checks it anchors reach 0.64 with no model call at scoring time.
- Recall splits by category exactly as designed: categories the reaction can
  show (resource abuse 0.81–0.83, tool-related 0.60–0.83, authentication and
  environment errors 1.0, poor retrieval 0.54–0.80) are caught; categories that
  live in the action alone (context handling 0.18 on SWE, language-only 0.18
  on GAIA) are not.
- `overall` is a weak target on GAIA — it averages security and plan scores
  that barely vary — and a decent one on SWE. Read GAIA against error count.
- Leniency is an archetype property. "A page without the answer is not a wrong
  step" fixed GAIA (precision 0.42→0.55) and broke SWE (checks ρ 0.43→0.11),
  because on coding traces a lookup that found nothing after the action assumed
  a path *is* the evidence. Prompt v3 applies the rule to browsing and support
  only.
- The checks the RLM writes are shallow — `error:`, `syntaxerror`, `code
  execution failed`, `TypeError`, `could not find` in the reaction — and
  redundant in pairs. On GAIA they beat the judge that anchored them at both
  turn and trace level, at no model cost. On SWE the two proposal runs
  disagree (ρ 0.43 vs 0.15 against `overall`): 31 traces put roughly ±0.35 on
  a Spearman, so nothing at trace level on SWE separates the scorers, and the
  v1 row should not be read as a win. Review is where a human prunes the
  duplicates and refuses the ones that merely restate "an error happened".
- Proposal quality varies run to run even at temperature 0, because the
  sample the model sees depends on the judge's verdicts. Anchoring to a
  stricter judge (v3) yielded more checks with lower precision against the
  annotations. Whether that is noise or a real effect needs a larger split.

Artifacts in `work/trail-ns`: judge runs `turn-judge-e76d2617aa1fe7b8` (SWE
v1), `turn-judge-36b430856fa11bf2` (SWE v2), `turn-judge-b83480800ad90a25`
(SWE v3), `turn-judge-0044b8f5c041155c` (GAIA v2); verifiers `family-verifier-f0d56785fab2abd6` (SWE, v1),
`family-verifier-0f72ca7ca0065cdb` (SWE, v3), `family-verifier-dadbdfb15ac3d641`
(GAIA); eval reports `eval-*.json`. tau2 lives in `work/tau2-ns` (corpus
`corpus-8071af7bbbade94c`, judge run `turn-judge-d450cf34d97b27a2`,
`eval-judge.json`).

## What it does not do

- **Final turns are unobserved.** Whether the last answer was right is not in
  any reaction. On a benchmark the outcome grader covers it; in production the
  next user message or the post-session state does, when a source records them.
- **The judge is the anchor.** Checks are measured against it, not against
  truth, so a family-specific check that disagrees with the judge is either a
  discovery or a mistake — which is what the review decides, and what the
  TRAIL eval measures.
- **Support has no reaction corpus yet.** tau2's simulated user does not push
  back, so the archetype exists and is untested until a corpus with real user
  turns is ingested.
