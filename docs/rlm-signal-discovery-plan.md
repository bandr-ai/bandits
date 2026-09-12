# Plan: generalized success-signal discovery for SFT selection, via RLM

**Status:** draft for external review. Nothing in the "proposed work" section has
been built. Everything in "what we tried" and "what we found" has been run and
the numbers are real, reproducible from the scripts named.

**Repo:** `vektori-ai/bandits`. **Branch this work lives on:**
`exp/sft-selection-signals` (experiments) plus reads against
`feat/rlm-task-family-mining` (now merged to `main` as PR #42, the existing RLM
family-mining infrastructure this plan extends).

---

## 1. The problem, precisely

`bandits` turns raw agent traces (OTLP, chat-JSON, Claude Code session logs)
into SFT training rows and eval cases. To select a trajectory as a good SFT
demonstration, you need to know it was actually a **good** trajectory — the
agent did what was asked, correctly, without flailing.

The existing design (`bandits/verify/`) answers this with a **verifier**: a
small set of deterministic checks (`final_state_field == X`, `exit_code == 0`,
an invariant between initial and final state), drafted per task family, ranked
against human labels, measured for held-out agreement, probed for gameability,
and promoted only after human review. This is explicitly modeled on how you'd
build an RL reward: it assumes you can induce a reusable, gameable-resistant
checker from historical traces, then defend it against a policy that will try
to exploit it.

**The bet this plan tests, and found wanting:** that a success checker can be
*induced from unlabeled traces alone*, with no reference answer, no external
signal, nothing outside the trace itself. Every experiment below is evidence
about whether that bet holds.

---

## 2. What we tried, and why each attempt was structured that way

### 2.1 Structural / deterministic signals (all four corpora)

**What:** cheap, trace-only predicates — did the agent call a state-changing
tool, did any tool error, does the final message claim completion, is the
trajectory short relative to its peers, did it repeat identical calls, did it
give up / transfer to a human.

**Why this first:** it's free (no model calls), and if trajectory *shape* alone
separated success from failure, everything downstream would be unnecessary.
It's the null hypothesis a real signal has to beat.

**Result — dead everywhere:**

| corpus | n | best structural AUC |
| --- | --- | --- |
| tau2 (retail dialogue benchmark) | 456 | 0.61 |
| AppWorld (multi-app task benchmark) | 30 | 0.58 |
| TRAIL/GAIA (open-web tool use, human-scored) | 116 | 0.08 (Spearman) |

AUC 0.50 is a coin flip. None of these cleared it meaningfully. Concretely on
tau2: successful and failed runs average 14.1 vs 15.4 spans, 1.58 vs 1.77
write-tool calls, 3.6% vs 4.2% give-up rate, 45.0% vs 46.6% "done"-language in
the final message — essentially indistinguishable. **A wrong action and a right
action look the same shape.**

*Script: `scripts/signal_experiment.py` (tau2), `scripts/rederive_signal.py`
(AppWorld), `scripts/trail_signal.py` (TRAIL).*

### 2.2 Holistic LLM judge, no reference (all four corpora)

**What:** show a strong model the whole transcript, ask "did this succeed."

**Why:** the obvious next thing to try once structure fails — maybe a model
can read the semantics even if shape can't carry the signal.

**Result — dead on the two benchmark-graded corpora, real but modest on the
human-graded one:**

| corpus | judge | AUC | accuracy vs baseline |
| --- | --- | --- | --- |
| tau2 | deepseek-v4-flash | 0.58 | 58.9% vs 74.1% baseline (worse) |
| tau2 | deepseek-v4-pro | 0.62 | 62.7% vs 74.1% baseline (worse) |
| AppWorld | decomposed sub-goal judge (deepseek/gpt-oss) | 0.49 | worse than a coin flip |
| TRAIL/GAIA | gpt-oss-120b, same 1-5 rubric humans used | **0.57 (AUC on tertiles), Spearman 0.31** | first real positive result |

The AppWorld judge is the sharpest illustration of *why* this fails: it scored
**seven benchmark-failed runs at 0.9-1.0** ("every sub-goal done") because it
saw a plausible-looking `create_expense(...)` call return OK, and it cannot see
that the expense had the wrong amount or went to the wrong person. The
transcript shows the *action*, not whether the *result* matched a hidden goal
state it was never shown.

**Why TRAIL is different, and this is the load-bearing finding of the whole
plan:** TRAIL's human `overall` reliability score is not a hidden-database
check. It partly reflects things that ARE visible in the trace — did the agent
hallucinate a tool result it never actually got, did it skip a step its own
stated plan named, did it ignore an instruction. Those are **evidence-in-trace**
failures. tau2 and AppWorld's failures — wrong product variant, wrong recipient,
wrong final state — are **evidence-outside-trace**: you need the answer key to
catch them, and the trace doesn't contain it.

*Independent confirmation this isn't an artifact of our prompts:* the published
**TRAIL benchmark** (Patronus AI, arXiv:2505.08638) tests exactly this —
can a frontier model read a trace and localize what went wrong. Best model
(Gemini 2.5 Pro): **18.3% joint category+location accuracy on GAIA, 5.0% on
SWE-Bench**, and specifically **near-zero F1 on "tool selection" and "context
handling" categories** (the semantically-deep ones) versus 0.14-0.59 on
"language-only hallucination" and formatting (the surface ones). Same shape of
result as ours, from a different team, a different benchmark, different models.

*Also confirmed by industry practice, not just research:* Arize's own guide to
building production LLM judges reports **82% agreement with 100 hand-labeled
sessions** as a good outcome for a *calibrated* judge — and even there, the
judge's most common error was marking partially-resolved sessions as fully
resolved. Calibration against real human labels is described as mandatory in
their own framework, never optional. No production system we found operates a
judge with zero reference.

*Scripts and data: `scripts/rederive_signal.py`,
`scripts/trail_signal.py` against `github.com/patronus-ai/trail-benchmark`
(cloned directly — the data is checked into that repo ungated, despite the HF
mirror requiring a gate).*

### 2.3 Sibling consensus (tau2 only — needs data these other corpora lack)

**What:** tau2 runs every task 4 times independently. Fingerprint each run by
its exact set of state-changing calls (tool name + verbatim arguments, mutation
detected by a generic verb classifier — `modify_`, `cancel_`, Claude Code's
`Edit`/`Write` — not a hardcoded tau2 tool list). Score a run by what fraction
of its 4 siblings share its fingerprint.

**Why:** independent, repeated attempts at one task are a free jury. If 3 or 4
of them converge on the identical action, that convergence is hard to fake by
coincidence.

**Result — the strongest signal found anywhere: AUC 0.82.** Unanimous
agreement (4-of-4) → 92.7% of those runs really were successes, covering 48.2%
of the corpus. Relaxing to 3-of-4 → 89-92% precision at ~60-69% coverage.

**Why this doesn't generalize, and this matters for the plan below:** it needs
`k > 1` independent rollouts of the *same task*. Benchmark logs and best-of-N
sampling have this by construction. Production Claude Code sessions do not —
each real session is a unique job, done once. Checked directly: `work/prod-cc`
(48 real sessions) has 31 lineages of size 1, one of size 13, two of size 2.
Sibling consensus is inapplicable there as it stands.

### 2.4 Model-synthesized signals — a sanity check on our own signal-writing

**What:** instead of me hand-writing structural signals, show a model
stratified labelled examples and have it write Python predicate functions
itself, executed under an AST sandbox (no imports, no `eval`/`open`/dunders, a
2-second wall-clock guard per trace), scored the same way as everything else.

**Why:** to rule out that our structural signals failed because *we* wrote bad
ones, not because the signal isn't there.

**Result:** independent confirmation. Best model-invented feature on tau2:
AUC 0.607. Nothing cleared a 0.62 keep bar. Same wall, from a different
author.

*Script: `scripts/signal_synth.py`.*

---

## 3. The reframe: "generalized" means the search is generalized, not the answer

Every dead end above shares one property: **I decided the extraction rules.**
The tau2 write-tool list, AppWorld's mutation-verb classifier, TRAIL's category
names, the judge prompts — all hand-authored, all corpus-specific in some way,
even when I tried to make them generic. That is precisely the complaint that
reframed this plan: *"it need not be so benchmark specific... it could be any
trajectory."*

The fix is not a smarter universal signal (the research above says that
probably doesn't exist in the no-reference regime). The fix is a **procedure**
that writes the corpus-specific extraction code itself, on whatever trace shape
it's actually given, with a discipline that makes the result trustworthy rather
than a black box. That procedure already exists in this codebase, applied to a
different problem: **RLM family mining.**

---

## 4. What RLM means here, concretely (verified against the real code, not assumed)

"RLM" is `dspy.RLM` — a DSPy primitive that gives a language model a Python
REPL sandbox to write and *execute* code as part of answering a typed
`dspy.Signature`, before it commits to a final output. `bandits/analyze/
rlm_mine.py` wraps this primitive to discover task-family taxonomies from raw
agent traces. This is now the **only** family-mining path in `bandits` — the
older embedding/distance-based miner was removed entirely (commit "Remove the
embedding miner, its cache and its audit").

The disciplines this implementation already enforces, each checked directly
against the source rather than assumed from the docstring:

- **Explicit, measured input views, never a silent default.** `TraceView` has
  three arms: `USER_MESSAGES` (every user turn), `FIRST_USER_MESSAGE` (opening
  request only), `FULL_TRAJECTORY` (adds assistant/tool activity, with reward
  and evaluator fields stripped). The reason two user-message arms exist at all:
  "a correction may say what the user actually wanted, or may say only that
  this particular agent went wrong, and which it is cannot be assumed" — so it
  is *measured*, not assumed, and every artifact records which view produced
  it.
- **Chunked, mixed, multi-pass reading.** A pass reshuffles the corpus and
  reads every trace exactly once; a chunk mixes unseen traces with ambiguous
  ones, recently-affected ones, and random settled ones, specifically so a
  family "cannot be an artifact of which twenty traces happened to be read
  together."
- **Honest stopping.** `StopReason` is `PASSES_COMPLETE`, `MAX_ITERATIONS`,
  `MAX_LLM_CALLS`, `MAX_SECONDS`, `MAX_USD`, or `ERROR`. There is **no
  `CONVERGED`** — the code deliberately refuses to let "the schedule finished"
  read as "the taxonomy stopped changing," because those are different claims
  and conflating them is exactly the failure this plan exists to avoid
  repeating with signals.
- **Never fabricate a measurement you didn't take.** `rlm_taskset.py`, which
  turns a finished clustering run into a usable `TaskSet`: the old
  embedding-based family carried a `medoid_trace_id` (a real centrality claim)
  and a coherence diameter (a real distance measurement). RLM mining has no
  distance metric, so instead of inventing plausible numbers, `coherence` stays
  `None`, the medoid is picked by an explicitly-not-a-centrality rule (lexically
  first member), and the family's own limitations say it wasn't measured.
- **A model cannot be trusted to critique its own output.** `rlm_audit.py` runs
  a *separate, fresh-context* model against a finished clustering run,
  specifically because "the reasoning that produced a family is exactly the
  reasoning least able to see what is wrong with it." Purely advisory, no gate,
  saves findings for a human to read.
- **Resumable, auditable sessions.** State is written to disk after every
  chunk, never after every pass, so a run that dies mid-pass resumes from the
  last completed chunk rather than from the start; the scratch session is
  explicitly not evidence and nothing downstream may cite it.

Real operating parameters, read from the code rather than guessed: default
chunk size 20, default seed 42, default max completion tokens per call 24,000,
default backbone `nemotron-lightning-3p5-30b-a3b`. A family-mining chunk
measured against real tau2 data took 5-18 inner model calls.

---

## 5. The blocker found before writing any new code

Before designing a verifier step on top of RLM-mined clusters, I checked
whether clustering is actually *usable* today on the corpora we have data and
labels for. It mostly isn't, and the reason is worth stating precisely because
it is not a bug — it is the miner correctly refusing to trust data it cannot
verify:

- `TraceView.USER_MESSAGES` / `FIRST_USER_MESSAGE` read `trace.user_turns`.
  Checked `bandits/ingest/otlp.py` directly: **it never populates
  `user_turns`, for any OTLP source.** tau2, AppWorld, and TRAIL (if ingested
  through this adapter) would all report every trace **unreadable** under these
  views.
- `TraceView.FULL_TRAJECTORY` falls back to rendering spans when there's no
  `user_turns`, so it doesn't error — but `_render_span` in `rlm_corpus.py`
  reads only `span.arguments` and `span.output`, **never `trace.task`**. For
  tau2 specifically, the actual customer request lives in `trace.task`, which
  the adapter reconstructed rather than recorded as literal per-turn user text.
  `build_view`'s own comment explains why it won't fall back to that field: a
  trace's request "is not readable without inferring one from a field the
  miner may not see" — a deliberate refusal to trust an inferred field as if it
  were a genuine recorded utterance, not an oversight.
- The **only** adapter that builds real `UserTurn` objects today is
  `bandits/ingest/claude_code.py`, for real Claude Code session logs.

**Consequence:** `work/prod-cc` (48 real sessions) is the only corpus where RLM
family mining is legitimate as things stand. It has no sealed truth, which is
the thing we need to measure a verifier step against. tau2 and AppWorld have
sealed truth but are not legitimately mineable today.

---

## 6. Proposed work (not yet started — this is what needs review)

### 6.1 Close the adapter gap (recommended first step)

Extend `bandits/ingest/otlp.py` to extract `UserTurn` objects from
`gen_ai.input.messages` entries whose role is genuinely `user`, mirroring what
`claude_code.py` already does for its format. This is not inventing data — the
user-role messages exist in the raw OTLP for both tau2 and AppWorld (verified
directly while building `rederive_signal.py`'s custom parser, which had to
extract them by hand because the shared adapter doesn't). Doing this properly
gives every OTLP corpus a legitimate `USER_MESSAGES` view, which:

- makes tau2 and AppWorld honestly mineable, giving us RLM family mining
  **and** sealed truth on the same corpus, which is what a real end-to-end test
  of a verifier step needs;
- is a real fix to a real gap, not scoped to unblock this experiment alone —
  every future use of RLM mining on OTLP data benefits.

*Risk to flag for review: this must not smuggle `trace.task` in through the
back door disguised as a "user turn" — it has to be an actual per-turn message
with a real role, or it reproduces the exact problem `build_view` is refusing
to create.*

### 6.2 Run `mine-rlm` for real, on a corpus that can now legitimately produce it

Once 6.1 lands: run the existing, unmodified `mine-rlm` CLI command against
tau2 (456 traces, sealed truth, now-readable user turns). Budget-capped first
run (a `--max-usd` ceiling, small `--passes`) purely to see real families
before committing to anything downstream. This has never actually been run in
this investigation — everything about RLM family mining so far was read from
source, not executed.

### 6.3 Design and build RLM-based signal/verifier discovery, per family

This is the actual new module, applying the same primitive to a different
target:

| RLM family mining (exists) | RLM signal mining (proposed) |
| --- | --- |
| Input: chunks of raw user requests | Input: chunks of raw `Trace` objects + a weak label, drawn from one family's traces |
| Output: a revised `FamilyContract` taxonomy | Output: a revised library of `SignalContract` objects — hypothesis, code, evidence kind, provenance |
| Model reasons about topical/verifier-contract similarity | Model's own REPL loop: parse whatever shape these traces have, write a candidate `signal(trace)`, execute it against the traces it can see, check its own hit rate, revise, repeat, before finalizing |
| `TraceView` arm: how much of the trajectory the miner may read | Same arm, explicit and measured, not a silent default — a signal built from the full trajectory risks learning "how this agent behaves" instead of "whether it succeeded," the identical risk `TraceView` already exists to catch |
| Chunk priority: unseen, ambiguous, recently-affected | Same priority, plus: traces the *current signal library disagrees on* — the single most informative example to show next, same logic verifier drafting's disagreement-first labeling already uses |
| `StopReason`: no invented `CONVERGED` | Identical enum, reused, not re-derived |
| `rlm_audit.py`: fresh-context adversarial pass, advisory only | Mirrored: after a signal survives mining, a separate fresh-context model tries to argue it's spurious or overfit to the chunks it happened to see |
| **Independent, honest re-scoring is new here** — families aren't executable, so nothing in the existing pipeline re-checks a family's own self-report against held-out data the way `score_signal()` already does for every signal in this investigation | Every surviving signal is re-executed by the harness against the **full** corpus and the **held-out** split — never trusted from the model's own chunk-level self-report, which only ever saw a slice |

### 6.4 The test that would actually validate this

Run the **identical** signal-mining script, unmodified, against **at least two
corpora with different label types** — tau2 (binary sealed truth) and TRAIL
(continuous human score) — without touching the code in between runs. If it
discovers something real on both without per-corpus tuning, that is the actual
proof the generalization worked, not an AUC number on one dataset. If it needs
a knob turned between them, that knob becomes something the *model* decides
next chunk, not something I hand-tune.

### 6.5 What this does *not* solve, stated up front so it isn't oversold

RLM generalizes the **search** for signal-extraction code. It does not remove
the need for a real label to score candidates against per chunk. Every corpus
still needs one of: sealed benchmark truth, a human-scored calibration set, or
sibling consensus where repeated rollouts exist. For `prod-cc` — the corpus
that matters most for the actual SFT-selection goal — none of these exist yet.
The next-user-turn signal and post-session git-state signal (proposed earlier,
not yet built) are what would eventually supply that label there; RLM signal
mining and that labeling problem are separate pieces of work that both have to
land before `prod-cc` is usable end to end.

---

## 7. Summary table: every signal tried, for quick reference

| signal | corpus | mechanism | result |
| --- | --- | --- | --- |
| 8 structural predicates | tau2 | rule-based | AUC 0.50-0.61, dead |
| holistic judge, 2 models | tau2 | LLM, no reference | AUC 0.58-0.62, worse than baseline accuracy |
| sibling consensus | tau2 | compare repeated rollouts | **AUC 0.82** — the strongest result, needs k>1 rollouts |
| model-synthesized predicates | tau2 | LLM writes code, sandboxed exec | best AUC 0.607, confirms the wall independently |
| 4 structural predicates | AppWorld | rule-based | AUC 0.45-0.58, dead |
| decomposed sub-goal judge | AppWorld | LLM, grounded in tool results | AUC 0.49, worse than chance |
| reverse task reconstruction | AppWorld | LLM, infers task from actions | AUC 0.66, ties baseline, 10% coverage |
| 3 structural predicates | TRAIL/GAIA | rule-based | Spearman 0.01-0.08, dead |
| holistic 1-5 reliability judge | TRAIL/GAIA | LLM, same rubric humans used | **Spearman 0.31, AUC 0.57 — real, modest, first positive result** |

---

## 8. Open questions for review

1. Is the adapter fix in 6.1 scoped correctly, or does extracting `UserTurn`
   from `gen_ai.input.messages` risk reproducing the exact "inferred field
   masquerading as recorded text" problem `build_view` already refuses?
2. Should signal mining run per-family (requires 6.2 first) or corpus-wide
   (decouples from family mining, but loses the family-level label most other
   parts of the pipeline assume)?
3. Is a fresh-context signal audit (mirroring `rlm_audit.py`) worth building
   before or after the first real mining run, given it adds cost per signal?
4. Budget for a first real run: what dollar ceiling is acceptable, given family
   mining measured 5-18 calls per chunk on tau2 and a signal-mining chunk has
   to additionally execute and debug code inside the same call?
