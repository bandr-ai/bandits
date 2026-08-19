# VERIFIER — <name>

<!--
TEMPLATE. `scaffold_workspace` writes a populated copy from real Verifiers.
Same three parsed constructs as ENVIRONMENT.md; see that file's header comment.
`- **Reviewed by:**` under a `### <verifier_id>` heading is the sign-off gate.
Leaving it as **TODO(human)** means the verifier is withheld, not accepted.
-->

Reward is code: assertions over the final state and the effect ledger. Never a judge.
Each verifier was synthesized from the state diff of one trace labeled correct.

**A generated reward function nobody has read is an unexamined reward function.**
`reviewed_by` is unset on every synthesized verifier, and `bandits.verify.evaluate`
raises `UnreviewedVerifierError` rather than grading with it. Signing is not a
formality — it is the only thing standing between a wrong assertion and a model
trained on it.

## 0. What to look for

- **Does it assert what must stay the same?** If a verifier only checks the field that
  changed, the agent learns that collateral damage is free.
- **Is the expected value right, or just what happened?** These came from one episode.
  If production refunded the wrong amount, the assertion now encodes that.
- **Are effects asserted?** For many agents success lives entirely in effects — an
  email sent, a charge attempted — not in stored data.
- **Is anything unverifiable?** If a real success condition cannot be written as an
  assertion, say so here. We do not reach for a judge to cover the gap; the task either
  gets a narrower reward or gets dropped.

## 1. Verifiers

### <verifier_id>

- **Task:** <task_id>
- **Assertions:** <n>

  - `state_equals` · `<entity>[<key>].<field>` == `<value>`
    - <description>

- **Assertions correct:** **TODO(human)** (yes / no + which one is wrong)
- **Missing invariants:** **TODO(human)** (what must stay the same and is not asserted)
- **Unverifiable in code:** **TODO(human)** (say so rather than adding a judge)
- **Reviewed by:** **TODO(human)** — put your name here. Until you do, this verifier
  cannot grade anything.

## 2. Tasks we refused to write a verifier for

- `<task_id>` — <reason>

These are refusals, not failures. A verifier synthesized from a failed or unlabeled
trajectory would make that trajectory the training target.

## 3. Anti-cheat

A rebuilt world is simpler than reality, so it admits strategies reality does not.
`bandits.verify.check_rollout` fails a rollout that writes the store directly, reads
the verifier or the effect ledger from inside the episode, or touches the network.

- **Exploits specific to this domain:** **TODO(human)** (a shortcut in your world that a
  generic check would miss)
