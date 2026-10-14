# Rubric induction prompts (seed-only)

Two prompts. INDUCE writes the domain part (Part B) of a judge rubric from reviewed seeds. REVISE improves it from
measured judge failures. The generic parts (A: evidence integrity, C: minor issues) are fixed text and are not induced.
Placeholders in {braces}.

---

## INDUCE

<agent_system_prompts note="every distinct system prompt in the seeds, with its id">
{system_prompts}
</agent_system_prompts>

<tool_definitions>
{tool_definitions}
</tool_definitions>

<seeds note="reviewed real traces; each ends with the agent's accepted final answer">
{seeds}
</seeds>

<request>
You write the answer-correctness part of a rubric that a judge model will use to decide whether a trace's final answer
is right for its evidence. The judge will see one trace and its agent's system prompt, never these seeds.

Use only the material above. Never invent a rule, an answer value or a tool behaviour. Every rule you write must cite
its source: a quote from a system prompt (prompt:<id>) or seed ids (seed:<id>). If a rule is implied by a prompt but
no seed shows it, mark it "unverified".

For each distinct agent (group prompts that ask the same question with the same allowed answers):

1. Question: the one question the agent answers, in one sentence.
2. Allowed answers: the exact values its output contract permits, and which fields must appear with which answer.
3. Situations: the observable facts that decide the answer. Find them by contrast: compare seeds of this agent whose
   answers differ and name the fact in the tool results that differs between them. Describe each situation by what a
   tool result would show, not by the answer. Add situations the prompt names ("conclude X when Y") even if no seed
   shows them, marked unverified. Keep the list to what actually changes the answer.
4. Situation -> answer table: for each situation, the answer(s) that are right. List more than one when the prompt or
   the seeds allow more than one. Say which answers are the same KIND (they recommend the same action or describe
   the same type of problem), because the judge accepts the right kind even when the logs cannot prove the specific
   detail.
5. Hard rules: the prompt's prohibitions and priorities that override the table ("never", "only after", "when in doubt",
   "X is not Y", tie-breaks), quoted.
6. Evidence traps: readings of tool results that the prompt or seeds show to be misleading (an unavailable result is
   not a negative one; a record-side result says nothing about the live side; a generic control resembling the target
   on another page is not the target).
7. Unknowns: what the material does not settle. Do not resolve them.

Additional constraints (from measured judge failures):

- Table rows state only evidence a trace can actually show (what a tool result says). Never copy lines from a cause
  catalog or system prompt into table cells; paraphrase the observable fact. A copied "detecting evidence" line makes the
  judge accept causes more specific than the logs support.
- Smallest relevant rule set: only rules that change the answer for these seeds. No per-agent rows that repeat a rule
  already stated once; shared rules go in one place.
- Put cross-read contradiction checks BEFORE the tables: first compare all reads of the same screenshot or snapshot
  (a crop saying the target is visible plus another read saying login/different screen is a contradiction); only then
  name the situation and consult the tables.
- Leniency: accept the right KIND of answer. When an answer's kind fits the situation, do not demand proof of its
  specific cause (restart, form factor, session expiry, variant); reject only wrong-kind answers or broken hard rules.

Write it as a compact markdown section per agent: a short header, the question, a two-column situation table, a
hard-rules list, evidence traps, unknowns. No prose beyond that. Aim for the shortest rubric that would let a stranger
grade these seeds correctly.
</request>

---

## REVISE

<rubric note="current version">
{rubric}
</rubric>

<results note="judge verdicts on the self-test set, with the expected label and how the case was made">
{failures}
</results>

<request>
You improve the rubric so the judge agrees with the expected labels, without overfitting.

The self-test cases were built from reviewed seeds: unchanged held-out seeds (expected good), benign changes that must
not change the label (expected good), and deliberate defects with a known type (expected bad). For each disagreement:

1. Say why the judge was wrong: it misread the situation, the table gave the wrong answer, a hard rule was missing or
   too broad, a minor issue was treated as a defect, or the case itself is ambiguous.
2. Propose the smallest edit that fixes that cause in general (a clearer situation description, a table row, a rule,
   moving an item to the minor list). Never add a rule that names one case, one app or one value.
3. Cite the system-prompt quote or seed ids that justify the edit. An edit with no source is rejected.

Then output the full revised rubric and a changelog: one line per edit with its cause and source. If a disagreement
comes from an ambiguous case, list it under "needs a human" instead of editing.
</request>
