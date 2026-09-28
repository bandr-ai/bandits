# Jev / RLCD post-training — learnings from all repos

Date: 2026-09-22. Research notes from reading the public Jev reproductions (Nimble, decision-head-rlcd, jev-on-a-laptop, AutoJev, JevBench entrants). Background only; the build plan is [decision-models-plan.md](decision-models-plan.md) and wins where they differ (e.g. SFT only, no RL).

---

## 0. TL;DR (first principles)

A "Jev-style" model = **classifier that reuses an LLM's forward pass**. Three separable layers:

| Layer | What it is | Who owns it |
|---|---|---|
| **Inference shape** | prefill state+questions once → read logits at one position per question → softmax over *allowed* answers only → Python assembles typed output. No generate(). | pure engineering, zero training (jev-on-a-laptop, nimble scorer) |
| **Decision readout** | which logits: (a) letter tokens A..Z from lm_head, (b) sliced lm_head rows as separate `readout` layer, (c) pointer/attention head query(`<decide>`)·key(`</opt>`) | architecture choice |
| **Post-training** | make those probs *accurate + calibrated*. CE / RLCD / temp-scaling. **Data is the whole game.** | this is what "Jev post-training" means |

Everyone who tried: **data coverage predicts everything**; algorithm (CE vs RLCD) is secondary. TypeSafe itself: "we're a data research lab, 100% synthetic data".

Nothing public from TypeSafe: no paper, no weights, no RLCD details. Everything below is community reconstruction.

---

## 1. What Jev actually is (TypeSafe, per docs + black-box studies)

- API: `state` (str/JSON) + list of `questions` → typed answers w/ probabilities + confidence. One call, questions evaluated **independently** (no cross-question context).
- 3 primitives: **Choice** (≤255 options → `choice`, `probabilities`, `confidence`), **Score** (ordered rubric → expected index), **Noul** (bool → p∈[0,1]).
- Doctrine: atomic questions, compose in code ("smart if-statements"). Model = per-question semantic judgment, code = logic.
- Claims: 70–500 ms, $0.042/MTok in, output free; "RLCD" training; "new architecture + parallel sampler". 0% hallucination is *by construction*.
- Black-box (agrogov study): latency ~flat 99→167 ms from 1→128 questions ⇒ shared prefill / amortized compute. Binary prob MAE 0.079 vs 0.25–0.29 for open competitors ⇒ probs genuinely sharper/calibrated. ~21K ctx.
- Diogo on X: "extremely diverse input distribution to make a non-jagged model"; 100% synthetic.
- Name collision: 2023 Meta "RLCD" (contrastive distillation, PPO on LLaMA) is unrelated.
- Ecosystem: JevBench (fstandhartinger/jevbench, 231 items, easy/standard/hard; leaderboard benchmarkheaven.com/jev-models, 21 systems), TypeSafe workflow evals (evals.typesafe.ai; ref = avg GPT-6 Astra + Fable 5.1), `typesafe-ai/system-one-adapter-python`.

---

## 2. Repo-by-repo

### 2.1 `jev-on-a-laptop` (rorshopping) — inference-only reproduction, honest audit
- Engine `harshatheg/Qwen-2.5-1B-RLCD` on HF = **zero weights, zero training**; stock Qwen2.5-1.5B-4bit + parallel constrained decoding. "RLCD" was just an alias.
- Mechanism: prefill once → `mx.repeat` KV cache ×N fields → one batched forward with row i = `  "field_i": ` → slice logits to candidate first-tokens → softmax → argmax. JSON assembled programmatically.
- Collision path (choices share first token) → sequential fallback, **fabricated confidence** `clamp(Πp, .75, .9999)`. Avoid: use single-token codes (A/B/C) instead of choice text. Everyone else learned this.
- Measured: 4.4–7.9× vs naive JSON gen locally; 1.5B→7B = +37 pts accuracy; **confidence ≠ correctness** (7B >0.90 on 13/20 wrong; ECE≈0.094, 28/40 wrong at ≥0.90).
- Head-to-head on TypeSafe public cases (343 pairs): Opus 89.8, DeepSeek 89.5, Sol 89.2, **Jev 86.6**, local Qwen2.5-7B 73.8.
- Memory: KV broadcast = fields × context; 48 fields × 8.5k tok doesn't fit 16 GB → chunk.
- Roadmap priorities: temp-scaling, calibration eval, more field types (Score/Noul), batched collision resolution.

### 2.2 `nimble` (Bespoke Labs) — cleanest open recipe: data + LoRA + eval
- Base Qwen3.5-9B, **LoRA r16, lr 5e-5, eff. batch 8, 1 epoch (335 steps), 2,676 examples**. H100. Result: 66.4% → **90.1%** on 324 holdout (Jev 93.2%, untuned Qwen3.8-27B 84.9%).
- **Loss = CE over candidate-code logits only** (`logits.gather(candidate_ids)` → CE). Not full-vocab LM loss. Answer = one-letter code A..Z (≤26 choices). Gold label never in prompt.
- Prompt: system "Classify the context using the supplied schema… return only one-letter code" + user JSON `{context, schema:[{name, description, choices:[{code,value,description}]}]}` + `Requested field: "<name>"`. One prompt per field; MLX scorer shares prefix KV; CUDA scorer reprocesses full prompt per field. Tokenizer boundary checks: code must be exactly one token after prompt.
- Prompt contract hashed into `schema_config.json` shipped w/ model ("the prompt format is the model").
- **Contrastive data curation** (from MiniCheck): pairs differing in ≤8 words / one focus fact, label flips. Pipeline: (1) audit rules from schema+policy → basic facts, (2) write 2 evidence sentences both needed, edit one, (3) separate model calls verify facts, policy consistency, no answer leakage; **deletion test**: removing either sentence → focus fact "unknown", (4) label = code applying rules to verified facts (labels never from model directly). Keep pair only if both pass and labels differ. Full request/response cache → offline replay. Pairs + source families stay in same split.
- Deletion variants are checks only, not training rows (missing evidence ≠ false).
- 10 domains × 3 primitives (Choice 856 / Noul 888 / Score 932). Score = ordered levels, argmax==ref counts as match.
- Limits acknowledged: probs not calibrated (T=1, untuned), narrow domains ("don't expect a lot of generalization"), 2048 tok prompt, ≤26 choices, no cross-field dependency.
- Curation generators: GPT-5.6 Terra/Sol/Luna, Claude Sonnet 5 / Haiku 4.5 — same prompts, JSON-schema outputs, refusals never become data. Adaptive source selection (favor high acceptance rate, every 5th explore).
- Has: playground (browser), Modal/SGLang serving, public benchmark suite (13 subsets: VitaminC, Banking77, …), comparison app vs Jev.

### 2.3 `decision-head-rlcd` (Astro-Han) — the actual RL experiment + generalization study
- Base Qwen3.5-4B, **LoRA r8 attention-only (4.9M params)**, 32K decisions, 1 pass, 3 seeds, lr 5e-5. 9 GPU-h H200 ≈ $42.
- Decision head: chat prompt `Material / Question / Options (A. … B. …)` → last-position logits → gather letter tokens (A..Z then AA..; up to 80) → mask → softmax. Also reports `candidate_token_mass` (how much of full-vocab mass lands on letters).
- **RLCD objective (their reconstruction)**: action = *reported distribution* = softmax(centre_logits + N(0,σ=1)); 8 samples; reward = **strictly proper scoring rule**: `log p_gold + p_gold/‖p‖` (log + spherical); for ordered (Score) questions add mean over thresholds of log cumulative prob (so near-miss ≠ far-miss); **leave-one-out baseline** (no std-normalization, no self-reward); loss = −adv · logπ(Gaussian). Caveat in their docstring: noisy-train / clean-infer makes the centre *sharper* than truth ⇒ RLCD doesn't guarantee calibration; measure ECE/Brier/NLL.
- Also implements `ce`, `rl` (0/1 reward REINFORCE), `exact` (−E[p_gold]) for comparison.
- **Headline result**: generalization tracks training coverage in *both* directions. Covered structure → ties Jev on JevBench easy+standard (120 items). Uncovered → **below untrained base** (pubmedqa 70.8→65, vitaminc 73.8→66, long_policy 0.47→0.32), all seeds same sign. "Narrow data isn't a no-op, it's a liability."
- Calibration *degraded* w/ RL (Brier 0.342→0.378): more accurate + more overconfident.
- OOD accuracy still rising 16K→32K (+2.1); public panel peaked at 8K then drifted. Gap to Jev entirely in hard tier: long material w/ many interacting rules (data gap), tradeoffs w/o rule (objective gap—all labels rule-derived), arithmetic (Jev also fails; one pass can't compute).
- BFCL abstain: RL moved the act/abstain **threshold**, not discriminative power.
- Unresolved confound: rank 8 attn-only may be too small; control arm = r64 + MLP not run.
- Synthetic data design (`synth.py`): families evidence / rules / boolean / routing / width-sweep; **domain packs** as structural unit — held-out packs never in training (true OOD); labels always from program, never model; models only write nouns/templates, validated + blind-checked by other model family; programmatic unit conversions. Prompt-order shuffles, sibling variants (flip one fact).
- Data plan (v3): 5 layers — public human-labeled (30%), constructive synthetic (50%), program-verified (10%), multi-model-vote soft labels (10%), human multimodal (eval/calib). **Evaluate only on human labels; never synth-on-synth.** Acceptance: no template >5% of family, 30-group blind audit ≥95%, exact+near-dup (Jaccard .85) = 0 vs all evals, structural IDs on every row, manifest at freeze.
- Pre-registered predictions w/ failures published — good methodology template.

### 2.4 `autojev` (denis-pplx) — biggest open model, full SFT, agent-built
- Base **Qwen3.8-27B, full-weight SFT**, lr 2e-6, wd .01, eff. batch 256, 1 epoch over 73K rows, **286 updates; released ckpt = step 200** (selected on 4,700-row monitored panel). One H200 (CPU-offload AdamW).
- Result: 69.8 → **84.6%** overall; ECE .043, Brier .220 — **beats Jev (82.8%, ECE .053)** on their panel.
- Architecture: separate `readout` Linear(hidden→255) **initialized from lm_head rows of 255 single-token codes**; mask to option count; `logits.masked_fill(-1e9)`. Multimodal (images via Qwen3.5 VL processor) but no image accuracy claim.
- **Calibration = post-hoc scalar temperature** fitted on separate 3,500-row fold (T≈2.21!). CE training makes it very overconfident; temp fixes it. Distinct folds: train / temperature / development / public / reference(Jev preds).
- Prompt: `State:\n… Question:\n… Options:\nA: … Return only the letter code`. Score options = "0".."k-1" indices; Noul = ["false","true"]. `confidence` for choice = (p_max − 1/n)/(1 − 1/n); score = expected index, confidence = 1 − expected distance/baseline.
- Data: public HF sets (boolq, squad2, paws, civil_comments, aegis2, pubmedqa…) converted to decision rows + Sol-generated synthetic across ~10 task types × conditions (clean/noisy/negated/long/shifted_prior) + programmatic numeric comparison (Fractions, unit conversions). Independent relabel of synthetic. Everything hashed (data, code, revisions); immutable run dirs; events log.
- Serves TypeSafe-compatible `POST /v1/systemone` + playground HTML.

### 2.5 Other community reconstructions (web)
- **kev-0.5b** (jaredpalmer): pointer head — `<opt>…</opt>` delimiters, query from `<decide>` token, key from each `</opt>` hidden state, dot-product softmax ⇒ 255 options w/o letter tokens; attention mask lets each question see state + own branch only, position IDs restart per branch ⇒ true parallel questions in one sequence. 9K records/6 datasets, CE + option shuffle + distractors; T=1.47 → ECE .065→.031.
- **Open-Jev** (Zefan-Cai): scalar decision head + LoRA r8, 2B/9B/27B, 80K rows 1 pass, temp on 512-row calib set; 9B ≈ Jev on hard coverage; 85 ms vs Jev 295 ms.
- Prompt-logprob zero-train variants (ekzhang openjev-sglang, so1, bnsd55). Small heads on Gemma 270M / ModernBERT (~50 ms, 64%). SemIf (Qwen3.5-4B) strongest public 4B.
- Niels Rogge X post is the canonical "how Jev decodes" explainer everyone cites.

---

## 3. Cross-cutting lessons (what to actually build)

**Architecture**
1. Single-token answer codes (A..Z / 255 reserved tokens) or pointer head. Never softmax over choice *text* first-tokens (collisions → fake confidence).
2. Report `candidate_token_mass` — tells you whether the model even "wanted" to answer with a code.
3. Prompt format IS the model. Hash it into the checkpoint contract (nimble `schema_config.json`, autojev `decision_config.json`).
4. Independence of questions is a feature (no context rot) and a limit (no cross-field consistency; code must check).
5. One forward pass cannot compute → arithmetic/dates: precompute in code, feed as facts.

**Objective**
6. CE over candidate logits works (nimble, autojev, kev, Open-Jev). RLCD-as-proper-scoring-rule (decision-head) works too but didn't beat CE on calibration; sharpening bias documented.
7. **Calibrate post-hoc with temperature on a held-out fold** (autojev T=2.2, kev 1.47). Always report ECE/Brier/NLL, not just accuracy. Every single repo found raw softmax overconfident.
8. LoRA r8–16 enough for in-distribution; possible capacity limit for holding multiple strategies (open question). Full SFT at lr 2e-6 also fine at 27B.
9. Tiny compute: 2.7K rows / 335 steps (nimble) or 32K / 2000 steps ($42) already reach 90%+ of Jev. **Data quality/coverage >> steps.**
10. Checkpoint selection on a monitored dev panel; frozen holdout touched once. Autojev picked step 200/286.

**Data (the actual moat)**
11. Labels from **construction/programs**, never from a model's opinion: contrastive pairs (nimble), domain packs + program rules (decision-head), converters over public human-labeled sets (autojev, kev).
12. Contrastive/counterfactual pairs: same context, one fact flipped, label flips. Deletion test to guarantee evidence necessity. Keep pairs in same split.
13. Structural held-out (whole domain / tool catalog / rule combination / phrasing), not random. Random split lies.
14. Diversity gates: no template >5%, option-count sweep (5→77), shuffle option order (measure order agreement), negated/noisy/long/shifted-prior conditions.
15. Narrow data **regresses OOD below base**. Mix in broad public sets (NLI, QA, safety, relevance, routing) as anchor ~30%.
16. Multi-model vote soft labels for genuinely ambiguous items; use as soft CE targets / calibration set.
17. Immutable, content-addressed, hashed everything: data files, code, base revision, request caches → offline replay. (Same philosophy as bandits `.bandits/` store.)

**Eval**
18. Eval sets: (a) in-dist held-out families, (b) structural OOD, (c) public human-labeled panel (nimble 13-subset), (d) JevBench, (e) Jev itself on same items as ceiling. Report per-seed; pre-register criteria.
19. Confidence/abstain analysis: recall vs false-abstain; RL shifts threshold not discriminability.

---

## 4. Where `bandits` fits

Bandits today: traces → families → next-state judge → RLM-proposed deterministic `check(turn)` predicates → reviewed → scored → SFT rows (pos/neg) w/ full lineage. Its **judge decisions are exactly Jev-shaped questions**: "did this turn's reaction indicate success?" (Noul), "which failure mode?" (Choice), "how good?" (Score), "which archetype/family?" (Choice).

Natural integration = **bandits produces decision-training data + gets a fast local decision model back**:

- **Data source**: every judged turn = `(state = turn + reaction, question, label)` with provenance. `export-decisions` → Jev-format rows (`state, question{type,criteria}, label`). Counterfactual pairs come free-ish: same turn, reaction swapped from a sibling trace (contrastive!). Accepted `check()` predicates = program-derived labels (layer 3 in decision-head plan) — highest-trust labels.
- **Consumer**: train small decision model → replace/pre-filter the LLM judge (`judge-turns --judge local-decision`) at ~50–100 ms/turn, with probabilities → abstain-to-LLM below threshold. Calibration fold from human `review-checks` decisions.
- Also: AWM/emulate `support estimation` + `abstain` is literally a Noul decision.

---

## 5. Open questions to settle before building

- Head: letter-token readout (simple, nimble/autojev proven) vs pointer head (255 opts, true parallel branches, kev)? Recommend letter/readout v1, pointer later.
- Objective v1: CE + temp scaling (proven) ; RLCD as optional flag (decision-head code portable).
- Base: Qwen3.5-4B (cheap, 4 public baselines to compare) vs 9B.
- Which hardware do we have for training? (LoRA 4B fits 24 GB; 9B needs ~40+; 27B full SFT needs H200.)
- Data: start from bandits traces only, or bandits + public panel + synthetic packs?

---

## 6. JevBench v1.3.0 + top entrants (added 2026-09-22; repos cloned: `openjev`(SemIf), `reflex`, `jqv`, `simple-jev`, `jevbench`, `kev`, `system-one-open`, `open-alternative-jev`)

**Scoring**: geo-mean of Intelligence (chance-corrected; hard 30/easy 14/std 28/judge 28%), Calibration (hard tier ECE + TV to exact gold dists), Speed (0.1s=100, −20/decade), Cost ($/1k *decisions*, $0.001=100, −30/decade). Below 50 Intel → ×(I/50)². 534 decisions, 220 hard (109 held-out). Self-hosted latency ×2+0.15s penalty. Harness MIT: `jevbench/` (adapters/, composite_v13.py, datasets/public/hard.jsonl).

**Board (top)**: Jev 74.4 (Intel 85.7, Cal 82.7, hard 74.1%) · **SemIf 73.1 — frozen Qwen3.5-4B, NO training** · djev 73.0 — stock DiffusionGemma 26B-A4B, one-step, no weights · Winnow-12B Q8 71.2 · reflex 4B 70.3 (frozen, two-order avg) · jqv 68.6 (stock Qwen3-32B + one temp T=3.02). Trained models: decider-35b 67.6, decider-2b 61.7, Nimble 60.5, kev 4B 59.7, Open-Jev 9B 55.0, system-one-open 66.6.
→ **#2, #3, #5, #6 are untrained.** Best trained model (decider-35b) is #8. Stock 27B readouts (SimpleJev 75.0%, reflex-27b 75.9%, LitJev 73.2%) already beat Jev on hard-tier accuracy; they lose on Speed/Cost only.

**Where Jev still wins vs SemIf (all tiers)**: rules&law 83.6 vs 64.2, finance 73.4 vs 60.9, safety 100 vs 75. SemIf wins coding 96.4 vs 83.9.

### SemIf (#2) — TheoLeeCJ/openjev
- Frozen Qwen3.5-4B, one forward pass, softmax over fixed uppercase letter logits at last position. Compact JSON prompt, letters for every option. Prompts frozen + `prompt_sha256` per row. Modes: direct / serial prefix reuse / parallel suffixes (2.3 → 10.8 → 20 decisions/s on 3090; BF16 reuse flips 5–6/777 argmaxes). Per-workload temp calibration added (PR #19). MLX/MPS/llama.cpp/WebGPU backends. Interpretation rule: "softmax over allowed tokens is conditional on supplied alternatives; not calibrated operational confidence."

### reflex (#5) — kshetrajna12/reflex — **the most important negative result**
- `stable` = frozen Qwen3.5-4B, default prompt, **no adapter, no calibration file**, every question read in two option orders and averaged.
- Tried: 4 LoRA mixes on public datasets, 27B→4B distillation, GEPA prompt opt, wording ensembles, >2 orders, reasoning cascade. **All rejected.** Every adapter won on training-shaped data, lost on hard external items (raw 0.640 hard vs mix1 .595 / mix2 .541 / mix3 .604) and on 3/4 external domains (MNLI .840→.807, toxic .787→.710, Yelp .650→.640; only same-task vendor intents improved .910→.937).
- Distillation from 27B: student→teacher agreement .65→.85, no collapse, still didn't beat frozen on hard (.613 vs .658; teacher .703).
- GEPA prompt opt: +2 on val, −2..−5 everywhere else. "Any search driven by our mix fits our mix."
- **Post-hoc temperature also doesn't transfer**: fitted on their mix (ECE .108→.028) → external ECE got *worse* (support .035→.085, MNLI .054→.123, public standard .055→.214). "A temperature is a property of a distribution, not of a model. Fit on a few hundred rows from the workload you'll serve."
- Prompt wording worth ±8 pts on same frozen model. Two one-liners that DID transfer: lettered yes/no ("A. yes / B. no", read letter logits) + "# Evidence / # Criterion" headings → toxic .787→.843, ECE .104→.046, hard .640→.658.
- Two-order averaging ≈ halves external ECE for free. Self-distilling the two-order readout into one pass: agreement .83→.94.
- Their rule: "Fine-tune only for a specific deployment with in-domain labels, keep it light (one epoch, attention-only), always measure on held-out sets from *other* domains. Report external-set numbers first."

### jqv (#6) — Octalab-Inc/jqv
- Stock Qwen3-32B, zero-shot. Engines: naive / kvcache / packed (block attn mask `[state|q1|q2..]`) / shared (Hydragen-style, mask-free). fp32 logits agree across all; leakage 0.000 vs control .996. 8k state × 100 q: 53–73× faster than per-question forward. `readout="rows"` = only lm_head rows for letters. One temperature: MMLU ECE .137→.023. Hard: 1.7B .423 / 14B .550 / 32B .622 / Jev .741. Has JP report.

### kev — jaredpalmer/kev
- 0.8B/4B/9B Qwen3.5 + LoRA + pointer head (Hume architecture). Trainable, TypeSafe API, playground w/ order-sensitivity check. Date preprocessing (appends day-count sentences) raised temporal-numeric .133→.200 — arithmetic outside forward pass.

### system-one-open — mithalouni/system-one-open (closest to a "short training demo")
- Gemma 4 E2B attention-LoRA + Gemma3 270M, all on Modal. **CE + Brier loss**, temp scaling, cosine LR. Data: 92 public HF decision datasets (23 held-out never trained) + rule-based synthetic demo families (doom, smart home, support, security, invoice, agent-trace, catalog). Prompt: `<state>…</state>` + `Answer k: (` slot per question; A–Z,a–z = 52 single-token options, chunk larger sets w/ "none of the above". Results: TypeSafe eval 76.7% (Jev 86.9, stock Qwen7B 73.8), held-out task types 74.8%, demo families 98.8% ECE .003. 8 live demos (Doom, browser agent, emails 74/s). Tiers: smoke(6 steps)/lite/full.

### Archer Hume, "Jev's Architecture Unmasked" (10k API probes) — the reference architecture everyone copies
- Causal transformer, state prefilled once, isolated question branches (secret-code leak test: 0.00 across questions, .90 when in state). 23k state × 5,000 q fits 65k limit. Question tokens cost ~2× state tokens.
- **Options interact**: adding irrelevant option shifted log-odds customer/unknown +.38→+.11 ⇒ listwise readout at final position (or pointer head), not independent per-option scoring.
- **Order sensitivity**: correct card last 16/16, middle 11/16, first 12/16.
- `confidence = (p_max − 1/K)/(1 − 1/K)` — arithmetic from adapter code, not learned.
- ECE .031 on 1,200 MMLU. Latency: 360 tok 57ms → 30k tok 218ms; 1→100 q ≈ flat 80–85ms; 1,500 q 610ms. 200-option answer as fast as 2-option ⇒ no decode loop.
- Guess: sparse MoE backbone; tokenizer ≈ o200k-like but matches none of 192 public.

### djev / OpenJev(razorback16) — DiffusionGemma 26B-A4B, one denoising step, bidirectional attention, no training; 73.0. Shows a *different backbone class* works zero-shot.

### Meta-lessons this adds to §3
20. **On general benchmarks, training has not beaten the frozen readout at equal size.** Trained 4B (reflex mixes, kev 4B, distill) < frozen 4B on hard/OOD. Training pays only in-domain (nimble 66→90 on own families; system-one-open 98.8% on own demo families; decider on classification-shaped tasks). ⇒ Train for *a workload*, not for *the world*.
21. Prompt wording + readout choice (letters, Evidence/Criterion, two-order avg) are worth as much as a LoRA and can't regress capabilities. Do these first.
22. Temperature is per-workload. Never ship a global T.
23. Report external/OOD numbers *first*; in-dist held-out of training sources looks great for the worst adapters.
24. Contract compliance (Choice/Noul/Score + structured/null over `/v1/systemone`) is a hard gate for JevBench and for swapping Jev ↔ ours.
25. Option-order shuffle in training + two-order avg at serve; listwise readout (options interact, like Jev).

---

## 7. Deep-dive on entrant research logs (kev PLAN.md, reflex results/, jqv report, decider card, jevbench HARD-TIER) — 2026-09-22

**What moved the needle (kev, 89 trials, $400):** backbone size (0.6B→4B +14–19pp; 4B→9B +2) >> low LR (2e-4→5e-5, +4.7pp; "less drift from base") >> none-of-the-above minimal pairs >> compositional rule-tree data (+4–5pp at 4B, 0 at 0.6B). Config knobs (rank, targets, epochs, head dim, perm-KL) all within ~1pp noise: **"config space exhausted for this data."**
**What didn't (kev):** more public data (dev↑ transfer↓), synthetic oversampling, option isolation (−5.8pp), WiSE-FT interpolation, bigger MoE base (35B-A3B: +knowledge, −noisy-label classification, net +1pp, 8× memory, not shipped), label smoothing / CE+Brier / focal (kev round 3: none beat recalibrated parent on selective coverage; smoothing *destroyed* ranking).
**LoRA erodes latent arithmetic** (dates 0.82→0.72) on Base ckpts but not on post-trained ones; fix is `date_facts` preprocessor in serving (0.72→0.90). Arithmetic = code, not weights. Confirmed by kev diagnostic: 14/14 flips when day-count stated.
**Uniform-target "unknowable" rows** → evidence-free items at ≥0.9 confidence 0.19→0.00. Cheap, works.
**Global temperature T≈2.0–2.3 baked into checkpoint**: ECE .105→.039, confident errors 7.5→3.2% (Jev 3.7%). Per-(type,K) temps worse. But reflex + jqv: **T fitted on one distribution can hurt another** (jqv bridge ECE .05→.24). Fit on your workload.
**jqv:** at 14B+ 4.8k-example decision training adds ~0 accuracy; scale dominates (1.7B .55 → 14B .75 → 32B .81 MMLU; hard .42→.55→.62). CE+λ·Brier ≡ CE for λ≤1; λ=2 just bakes in a temperature at −3pp acc. Cyclic-rotation averaging (K=4) +3pp, raw ECE .41→.20, free. 5-shot fixed examples in shared state −13pp (letter prior hijack). Slot head init from lm_head letter rows == vocab readout at step 0 (clean upgrade path). Pointer head order-equivariant, no gain vs slot.
**decider-2b (best trained recipe on board):** ~95 public datasets, ≤10 options/example subsampled (gold kept, shuffled), abstain option in 10% (¼ of those correct), CE; then 384 RL steps lr 1e-6: PPO on real outcomes + log-score on stated belief + rendering-consistency + **hard KL cap to SFT weights (.01 mean/.05 row)** → browser success 83→93%, general benchmarks unchanged. Still only .459 JevBench hard at 2B. CUDA graphs+compile: 49→4ms.
**JevBench hard tier construction** = a reusable data recipe: two frontier models author (Opus 5 / Sol), cross-review blind → gold → one discussion round, drop unaccepted; families adversarial/ambiguous/judge_hard/long_policy/multi_hop/probability/routing_hard/temporal_numeric/tradeoff/trap; schema `{state, question{type,instructions,criteria}, labels, expected, provenance{rationale, surface_answer, why_hard, gold_probs}}`. **`surface_answer` (the tempting wrong label) is the key field** — hard items are built around a distractor shortcut.
**jqv synth for weak families**: labels from solvers only (rule engine / `datetime`+`zoneinfo` / `fractions`), LLM only paraphrases fact paragraphs (kept only if all numbers/dates/names/negations survive), "wrong-computation note" distractor in state, 0 shared 8-grams w/ JevBench, difficulty gated to ±10pt of real hard-tier accuracy.
**Cascades/committees (jevbench combos):** fast→Jev cascade at τ=.42 kept 99.6% acc at 1/9 cost; committees improve calibration but cost axis collapses. Cascade is the only combination that's operationally attractive.

### Net-net for us
- Zero-training frozen readout + good prompt + two-order avg + per-workload T is the floor, and it's #2 on the board. **Build this first (1 day).**
- Training pays only on *your* workload's families; keep LR ≤5e-5, 1 epoch, attention-only LoRA, anchor to base (KL or replay), measure external-first.
- Data recipe that works: program/solver-derived labels, contrastive minimal pairs, none-of-the-above pairs, uniform-target unknowables, option shuffle, ≤10 options, surface_answer distractors, 8k ctx.
- A 2–3 hr demo run = nimble-scale (2–3k rows, ~300 steps, 4B LoRA) on one workload family; show frozen vs trained on held-out *same family* (+20–25pp expected) AND on an external set (expect ≈0 / slightly −) — that honest side-by-side *is* the demo.
