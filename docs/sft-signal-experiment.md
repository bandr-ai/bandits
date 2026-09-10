# SFT selection signals vs. sealed truth — first results

**Question.** Can cheap per-trajectory signals stand in for a reviewed verifier
when selecting SFT demonstrations? The bar is the majority-class baseline:
predicting "success" for every trace.

**Corpus.** `work/tau2-run/tau2.otlp.jsonl`, 456 τ²-bench retail runs with sealed
`success` labels. 338 success / 118 failure, so the baseline accuracy is **74.1%**.
The normalized corpus carries the opening task, the agent's model turns, and the
tool calls with results. It carries **no user turns, no tool errors, no tool
schema** — the OTLP export dropped them — so the trajectory is close to
outcome-blind on its face.

τ²'s own reward compares the final database state against a per-task goal state.
None of the signals here see that; they see only the transcript.

Reproduce:

```bash
uv run python scripts/signal_experiment.py \
  --corpus work/tau2-hitl-eval/.bandits/artifacts/corpus-85e4bd83c00ff7df \
  --truth  work/tau2-run/tau2.labels.json --out work/signal-experiment \
  --judge  deepseek-v4-flash-0731
```

## What each signal scored

`AUC` is threshold-free rank separation. `prec` / `cover` are the positive-set
precision and the share of the corpus admitted at the highest-precision operating
point the sweep found.

| signal | AUC | acc@0.5 | prec | cover | note |
| --- | --- | --- | --- | --- | --- |
| baseline (all success) | — | 74.1% | 74.1% | 100% | |
| **sibling consensus (share of 4 task-siblings with identical mutations)** | **0.82** | **78–83%** | **92%** | **48–60%** | the only signal that clears the bar |
| sibling consensus (is-modal fingerprint) | 0.73 | 82% | 85% | 74–80% | looser, broader |
| LLM judge, `deepseek-v4-pro-0813`, policy-anchored prompt | 0.62 | 63% | 83% | 58% | below baseline accuracy |
| short trajectory | 0.61 | 74% | 90% | 13% | weak ranker, tiny coverage |
| LLM judge, `deepseek-v4-flash-0731`, same prompt | 0.58 | 59% | 81% | 55% | below baseline accuracy |
| `no_repeat_calls`, `made_a_write`, `no_giveup`, `final_claims_done`, `no_tool_error`, `read_before_write` | 0.50 | ~74% | ~74% | — | no separation at all |

`consensus>=4 else judge` (trust a unanimous jury, spend the pro judge only on the
runs it splits on) reaches AUC 0.76 and 89% precision at **66%** coverage — the
judge buys ~18 points of coverage over consensus alone for ~3 points of precision.

Model-synthesized signals (`scripts/signal_synth.py`, a model writes Python
predicates, they are executed over the corpus and scored) land in the same place:
best invented feature `final_confirmation_phrase` at **AUC 0.607**, nothing clears
the 0.62 keep bar, whether scored against sealed truth or against the
sibling-consensus proxy label.

## Reading

1. **Structural signals are noise on this corpus.** Success and failure runs have
   near-identical shape: 14 vs 15 spans, 1.6 vs 1.8 mutations, 3.6% vs 4.2%
   transfer-to-human, 45% vs 47% "done"-phrasing in the final message. τ²
   failures are semantic — wrong variant, wrong order, partial completion — and a
   wrong exchange looks exactly like a right one in the trace.

2. **The LLM judge does not rescue it.** A policy-anchored prompt reaches AUC
   0.58 on `deepseek-v4-flash` and 0.62 on `deepseek-v4-pro`, both at
   *below-baseline* accuracy, over-calling failure. Flash catches 57% of real
   failures at the cost of 46 false alarms on 239 admits. This matches the
   earlier dogfood number (the `gpt-oss-120b` auto-labeler scored 40.6%). A
   bigger model moves the needle a few points; it does not reconstruct a hidden
   goal-state check from the transcript.

3. **Repeated independent rollouts of one task are a free jury, and it works.**
   τ² ships 4 runs per task. When all 4 converge on the same set of
   state-changing calls (same tool, same verbatim arguments), 93% of those runs
   really are successes, and that covers ~48% of the corpus; relaxing to 3-of-4
   gives 89–92% precision at ~60%. The run whose mutation set is unique among its
   siblings is very often the one that went wrong. AUC 0.82.

   This is domain-agnostic in the signal — it needs only "which runs share a
   task" (`lineage_id`) and "what did each change" (tool name classified
   read/write by verb, no τ² tool list). Swapping the hardcoded τ² tool list for
   the generic verb classifier left AUC unchanged at 0.82.

## Limits

- **One corpus.** Sealed truth exists only for τ². Every number above is a τ²
  number.
- **Consensus needs k > 1 rollouts per task.** `work/prod-cc` (48 real Claude
  Code sessions) is almost all singletons — 31 lineages of size 1 — so the
  consensus signal is N/A there. It applies to benchmark logs, eval-harness
  output, and best-of-n sampling logs, not to one-shot production traces.
- The synthesis loop executes model-authored Python. It runs under an AST
  allowlist (no imports, no dunders, no `open`/`eval`/`getattr`), restricted
  builtins, and a 2s per-trace alarm. Still: read `discovered_signals.py` before
  reuse.

## Second corpus: AppWorld (`scripts/rederive_signal.py`)

To check whether the wall was a τ² artefact, the same question was run on
**AppWorld** — 30 sealed-truth runs from `Exgentic/agent-llm-traces-v2`, 7-67
steps each, real multi-app tasks (pay a debt list over Venmo/Splitwise, archive
songs across every Spotify playlist), and the tool responses in the trace *do*
show what came back, errors included. 10 success / 20 failure, baseline 66.7%.

New signal tried here, the "solve it yourself and diff" idea: a **grounded
decomposed judge**. A model first breaks the request into atomic sub-goals, then
for each one is shown the agent's calls and results and asked whether the
evidence shows it was done. Score is the fraction of sub-goals with positive
evidence. Also a **reverse reconstruction**: infer the task from only the
state-changing calls, then rate how well that matches the real request.

| signal | AUC | vs baseline 66.7% |
| --- | --- | --- |
| reverse_reconstruction | 0.66 | ties baseline; 10% coverage at its precision point |
| made_mutations | 0.58 | no |
| no_error_results | 0.54 | no |
| **grounded decomposed judge** | **0.49** | worse than a coin flip |
| final_claims_complete | 0.48 | no |
| not_errored | 0.45 | no |

The decomposed judge scored seven benchmark-failed runs at 0.9-1.0 ("every
sub-goal done") and three benchmark-passed runs at 0.2-0.33. It sees a
plausible-looking `create_expense` call return OK and calls the sub-goal done; it
cannot see that the expense had the wrong amount or the wrong person. Same wall
as τ², now on long trajectories with full tool output.

## The conclusion

Across two corpora that share almost nothing — short vs long, retail vs
multi-app, stripped vs full tool output — **every method that reads only the
trajectory lands at chance.** Structural signals, a holistic judge, a grounded
decomposed judge, reverse reconstruction, model-synthesized features: all AUC
0.45-0.62.

The reason is the same both times: success is defined by a per-task checker
against a hidden goal state, and the trajectory does not contain the goal state.
You cannot re-derive a grade you were never shown the answer key for. The
verifier pipeline's central bet — that a success check can be induced from traces
— does not hold for benchmark data of this kind.

The one thing that beat chance, sibling consensus at AUC 0.82, works by not
judging a trajectory at all: it compares repeated rollouts of one task to each
other. It needs k > 1 attempts per task.

## What follows

1. **Labelled benchmark and eval-harness data already carries the outcome.** Use
   it directly for SFT. Exgentic ships `success` per run; τ² ships it. The blind
   export was a choice for the verifier dogfood, not a constraint.
2. **For unlabelled production traces the signal must come from outside the
   trajectory** — the user's next message, whether the diff was kept or reverted,
   whether a follow-up session was needed. Not from re-reading the transcript,
   which these experiments show carries almost nothing.
3. **Sibling consensus** is worth wiring into `bandits/export/` as a
   precision-first per-trajectory signal for any corpus with repeated rollouts,
   kept separate from `ReviewedVerifier`.
