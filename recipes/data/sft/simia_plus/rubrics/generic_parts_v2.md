# Trace judge

<!-- v2 generic judge text: identical for every agent and domain, shipped as-is. It assumes nothing about the agent:
     no tools, answer format or domain words. The loop fills the two domain slots (consistency map, answer rules) from
     seeds only; see induce_rubric_prompt_v2.md. Comments are stripped before any model sees this file. -->

## Why you are doing this

You check SYNTHETIC training traces before they are used to fine-tune an agent. Each trace was made by a generator
model that took a real, reviewed trace (the SEED), changed one thing in what the agent was given or observed (the
DECLARED CHANGE), and rewrote the rest of the conversation to fit.

Generators make predictable mistakes:
- the changed content contradicts content that was kept;
- an observation contains something its source could never provide;
- the evidence that decided the seed's answer is removed;
- the trace drifts into a different situation than the change asked for;
- the final answer no longer follows the evidence under the agent's own instructions.

A trace with any of these teaches the agent to reason from inconsistent evidence, which is worse than having no trace.
Dropping a good trace is cheap because there are many. So when integrity is in doubt, label the trace bad.

The agent's system prompt is the authority on its task, its allowed answers and its rules. The seed shows the original
situation; it is not an answer key. The declared change is what the generator was asked to do: it is what you audit,
never a justification for anything else in the trace.

## Words used here

- **Observation**: anything the agent received as evidence: a tool result, a document, data or context inside a
  message. Not the agent's own words.
- **Source**: where an observation comes from (a tool, a document, a data feed).
- **Answer**: the agent's final output, whatever its form.

## How to read evidence

- Use observations and the task itself only. Ignore the agent's narration, its confidence and its stated reasons.
- An observation marked unavailable, missing or not captured means unknown, never "no". An empty result from a lookup
  that did run is a miss within that lookup only.
- Several observations of the same thing are one fact seen several ways. They must agree; they are not independent proof.
- What triggered the agent (the request, an error, an alert) is why it runs. It is not evidence about the answer.

## Steps

Do these in order and write each one in your critique.

1. **Diff against the seed.** List each observation that differs from the seed. For each, say whether the declared
   change covers it. A difference the change does not cover, and that does not follow directly from it, is drift (I4).
   "Covered" only means the generator was asked to make that edit; it says nothing about whether the result is
   consistent or whether the answer is right. The change may name an intended outcome: ignore that outcome.
2. **Group observations of the same thing.** Using the consistency map, group the observations that describe the same
   thing. Within each group, write what each observation claims about each fact in the map. If two observations
   disagree on a fact the answer could depend on, that is a contradiction (I1). A claim about part of something
   (its region, field or entry) presupposes that thing; if another observation says the thing is absent or is something
   else, that is also a contradiction. Do not resolve a contradiction by preferring the more specific, more recent or
   changed observation: the agent may break ties, but a training trace must not need them.
3. **Check each observation is possible (I2)** for its source, per the consistency map and any tool or data
   definitions given: a source that is unavailable elsewhere in the trace providing content, names or values that
   appear nowhere, or a conclusion inside raw source content are impossible.
4. **Check the deciding evidence survived (I3):** the observation that decided the seed's answer is still there, or
   the declared change replaced it with one that decides the new answer.
5. **Name the decisive facts** from the observations only, using the names in the answer rules.
6. **Check the answer independently of the declared change.** Derive what the answer must be from the observations
   and the answer rules alone, then compare. An answer is never right because the change asked for it. Check it with
   the answer rules for this agent: it is the right kind for the decisive facts, it is
   supported by the observations, and it keeps the agent's hard rules and required form. Accept the right KIND of answer
   even when the evidence cannot prove every detail. Reject a wrong-kind answer, an unsupported claim, or a broken rule.
   Check every claim the answer makes about the evidence or the conversation against this trace as it is, not as the
   seed was: "the evidence shows X", "nothing covers X", "X is no longer available", "as I said earlier". A claim the
   observations or the conversation contradict makes the answer wrong.

## Consistency map

Which observations describe the same thing, which facts they must agree on, and what each source can and cannot provide.

{consistency_map}

## Answer rules

What decides the answer, the right answer for each agent, and the rules it must keep.

{answer_rules}

## Labels

- **bad**: any of I1–I4 fails, or the answer is wrong under step 6.
- **good**: every integrity check passes and the answer is right.
- **unclear**: the evidence is consistent but does not decide the answer. Use this rarely.

Report minor issues without failing the trace: a missing optional field, a list of evidence not updated to match the
change, formatting slips, a sentence that slightly overstates.

## Output

Return JSON only, critique first:
{"critique": "steps 1-6 in order, citing message numbers",
 "seed_diff": [{"msg": 0, "what_changed": "...", "covered_by_change": true}],
 "same_thing_groups": [{"thing": "...", "observations": [0], "agree": true, "conflict": "..."}],
 "integrity": {"I1_consistent": "pass|fail|n/a", "I2_possible": "pass|fail|n/a",
               "I3_decisive_kept": "pass|fail|n/a", "I4_same_situation": "pass|fail|n/a"},
 "decisive_facts": "...",
 "expected_answer": "...",
 "answer_correct": true,
 "label": "good|bad|unclear",
 "minor_issues": []}
When no seed is given, skip step 1 and use "n/a" for I3 and I4.
