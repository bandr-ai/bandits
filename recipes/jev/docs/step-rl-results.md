# Scoped step-reward RL results (seed 1)

Status 2026-10-06. One seed per arm, 10 Dr. GRPO updates of 16 training-pool
questions x 8 rollouts, LoRA rank 32, lr 1e-4, no KL; Qwen3-4B-Instruct-2507
policy; BrowseComp-Plus Qwen3-Embedding-8B retriever. Both arms saw the same
questions in the same order. The step arm adds 0.3 x (Jev step score - 0.20
baseline fixed on training-pool rollouts), capped at +/-1 per rollout, repeats
never rewarded. Held-out evaluation: 60 never-trained questions x 8 rollouts.
Accuracy is a strict string-match proxy, not the official LLM grader.

| Metric | No RL | Outcome-only | Step-wise | Step - outcome (95% CI over questions) |
|---|---|---|---|---|
| Gold evidence seen (primary) | 0.218 | 0.200 | 0.224 | +0.024 (-0.016, +0.068) |
| Gold evidence opened | 0.058 | 0.100 | 0.136 | +0.036 (+0.009, +0.064) |
| Accuracy (proxy) | 15.6% | 22.9% | 20.6% | -2.3 pts (-8.5, +4.4) |
| Tool calls per rollout | 3.6 | 4.1 | 7.4 | +3.3 |
| Rollouts repeating an action | 4% | 8% | 26% | +18 pts (+10, +26) |

Outcome-only vs no RL: accuracy +7.3 pts (+1.7, +13.5).

- Training signal: the step arm had learning signal in 14-16 of 16 groups per
  update; outcome-only in 0-7.
- The step arm searched and opened documents much more (the base policy's main
  weakness was stopping early) at no accuracy cost, but gold evidence found and
  accuracy did not beat outcome-only within 10 updates.
- Side effect: repeats rose to 26%. Repeats are only denied reward, not
  penalised, so extra searching came with padding.
- Mid-training the step arm first searched less (3.9 -> 2.2 calls) while its
  judge reward rose, then swung to more (6.2 calls by update 10).

Verdict: directional, not a win. One seed; the next run penalises repeats.

Engineering failures on the way (fixed): catastrophic regexes in `find` froze
the trainer; a per-update HTTP client to vLLM left requests hanging until its
timeout; a laptop suspend ended a non-detached run.
