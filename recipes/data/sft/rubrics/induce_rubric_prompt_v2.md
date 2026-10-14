# Rubric induction prompts v2 (seeds only, any domain)

Two prompts. INDUCE writes the two domain sections of the judge (consistency map, answer rules) from seeds. REVISE
improves them from the judge's measured mistakes on a self-test set. The generic judge text
(rubrics/generic_parts_v2.md) is fixed and shown to both prompts so they do not repeat it. Placeholders in {braces}.
The only input is the seeds. A seed is a reviewed trace: the agent's system prompt, any tool definitions, and the
conversation (which may contain tool calls and results, documents or data in messages, or neither). Nothing here names
a domain, a tool or an answer format: every domain fact must come from the seeds.

---

## INDUCE

<generic_judge note="fixed text the judge already gets; your sections fill its two slots">
{generic_judge}
</generic_judge>

<agent_system_prompts note="every distinct system prompt in the seeds, with its id">
{system_prompts}
</agent_system_prompts>

<tool_definitions note="may be empty">
{tool_definitions}
</tool_definitions>

<seeds note="reviewed real traces; each ends with the agent's accepted final answer">
{seeds}
</seeds>

<request>
You write the two domain sections of the judge above. The judge checks SYNTHETIC traces made from these seeds: a
generator changes one thing the agent was given or observed and rewrites the rest. The judge must catch the generator's
mistakes (contradictions, impossible content, removed deciding evidence, drift, wrong or unsupported answers) while
keeping correct variations. It will see one trace, its seed and the agent's system prompt, never these seeds as a set.

**First, work out what the evidence is in these seeds.** It may be tool results, documents or data inside messages,
or both. Use the judge's words: an observation is anything the agent received as evidence; a source is where it came
from. Write everything below in terms of the observations and sources these seeds actually have.

**Take the judge's perspective, not the agent's.** An agent's prompt teaches it to reach an answer despite noisy or
conflicting evidence ("X takes precedence", "when in doubt, Y"). The judge does the opposite job: in a synthetic trace,
two observations that disagree are a generator defect to flag, not noise to resolve. Never turn an agent's tie-break
into a consistency rule. Tie-breaks belong only in the answer rules, and only for evidence that is consistent.

**Sources of your rules.** Use only the material above. Cite every line: a quote from a system prompt (prompt:<id>) or
seed ids (seed:<id>). Write nothing the material does not show. State rules that hold across seeds; never describe a
single seed.

### Section 1: consistency map

1. **Same-thing groups.** Which observations describe the same thing in these seeds: one artifact read or queried
   several ways, one entity or record mentioned by several sources or documents, a value that appears in more than one
   place. For each group, how to recognise its members (source, arguments, titles, identifiers).
2. **Facts per group.** The facts those observations state that an answer can depend on, and how they word them.
3. **Contradiction patterns.** Cross the facts of each group and write the pairs of claims that cannot both be true,
   as concretely as the seeds' wording allows ("observation A says <fact = x> while observation B of the same thing
   says <fact = y>"). Where the seeds show that differing observations are legitimately different things (another
   variant, another scope, another time), say how to tell them apart. These are the most important lines you write.
4. **Possible and impossible content.** For each source: what the seeds show it providing, including unavailable or
   empty forms. Then what would be impossible: a source unavailable in one place providing content in another place of
   the same trace, names or values that appear nowhere else, a conclusion inside raw source content.
5. **Readings that are not evidence**, only where the seeds or prompts show them.

### Section 2: answer rules

1. **Decisive facts.** One shared list for all agents: the facts in the observations that decide what the answer must
   be, each named in one or two words and described only by what an observation states.
2. **Per agent** (group prompts that set the same task with the same answer form):
   - the task, in one sentence;
   - the answer form: fixed values (list them exactly, with required fields) or free text (its required parts and
     format rules, quoted from the prompt);
   - a two-column table: decisive facts → what the answer must be. For fixed values, the right value(s); for free text,
     what it must state, include or leave out. List several where the prompt or seeds allow several;
   - right-kind groups: answers that describe the same type of outcome. The judge accepts the right kind even when the
     evidence cannot prove every detail;
   - wrong answers to reject for each decisive fact, including claims the observations do not support;
   - hard rules: the prompt's prohibitions, priorities and tie-breaks, quoted.
3. **Constraints.**
   - Rows name facts that an observation in these seeds actually states. An answer whose evidence no seed shows gets no
     row. Never paste a prompt's lists of "evidence that indicates X" into the table; they make the judge accept answers
     more specific than the evidence. Instead, add a wrong-answer line where it applies: "<answer> is never supported by
     <fact> alone".
   - State each rule once. No unknowns section; no rows marked unverified; no worked examples.

### Output

Return exactly these two blocks with markdown inside and nothing else:
<consistency_map>...</consistency_map>
<answer_rules>...</answer_rules>

Aim for the shortest text that would let a stranger catch these generator mistakes: under 1,500 words in total.
</request>

---

## REVISE

<generic_judge note="fixed text, not to be edited">
{generic_judge}
</generic_judge>

<current note="the two domain sections now in use">
{sections}
</current>

<results note="the judge's verdicts on the self-test set, with the expected label and how each case was made">
{failures}
</results>

<request>
Improve the two domain sections so the judge agrees with the expected labels without overfitting.

The self-test cases are made from seeds only: unchanged held-out seeds and harmless variations (expected good), and
deliberate generator mistakes of a known type (expected bad). For each disagreement:
1. Say why the judge was wrong: a contradiction pattern or impossible content missing from the consistency map; a
   decisive fact misdescribed; a table row or wrong-answer line missing or wrong; a hard rule missing or too broad; a
   minor issue treated as a defect; or the case itself is ambiguous.
2. Propose the smallest edit that fixes that cause in general. Never add a rule that names one case or one value, and
   never add a worked example.
3. Cite the system-prompt quote or seed ids that justify the edit. An edit with no source is rejected.

Keep the judge's perspective: a contradiction in a trace is a defect to flag, never a tie to break. Keep the total under
1,500 words; if an edit adds text, remove a line that no case needed.

Return both blocks in full (<consistency_map>, <answer_rules>), then <changelog> with one line per edit (cause and
source), then <needs_human> listing ambiguous cases you did not edit for.
</request>
