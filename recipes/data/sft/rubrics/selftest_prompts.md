# Self-test prompts (seeds only, any domain)

Build a labelled self-test for the judge from good seeds: CORRUPT plants one known generator mistake in a seed (expected
bad), HARMLESS makes a meaning-preserving edit (expected good). Both answer with find/replace edits that code applies,
so they work for any trace format. Nothing here names a domain. Placeholders in {braces}.

---

## CORRUPT

<agent_system_prompt>
{system}
</agent_system_prompt>

<trace note="a real, reviewed, correct trace; messages numbered">
{trace}
</trace>

<request>
You help test a judge that screens synthetic training traces. Plant exactly ONE mistake of this type in the trace:

**{defect}**: {definition}

Make it the kind of mistake a generator model makes while editing one thing: subtle and plausible, not absurd. Keep
everything else as it is. The trace's evidence is whatever the agent received (tool results, documents or data in
messages); use the trace's own wording and formats.

Return JSON only:
{"edits": [{"message": <message number>, "find": "<exact text currently in that message>", "replace": "<new text>"}],
 "declared_change": "<the visible edit, worded as the instruction a generator would have been given; never mention or hint at the mistake>",
 "defect": "<one sentence: what is now wrong and where>"}
Each "find" must be copied exactly from the message content and be unique in it. Use as few edits as possible. If this
mistake type cannot be planted in this trace, return {"edits": [], "reason": "..."}.
</request>

---

## HARMLESS

<agent_system_prompt>
{system}
</agent_system_prompt>

<trace note="a real, reviewed, correct trace; messages numbered">
{trace}
</trace>

<request>
You help test that a judge does not reject correct traces. Make ONE harmless edit to an observation (anything the
agent received as evidence): reword it, or reorder independent parts of it, without changing any fact it states, any
value, name or identifier, or anything the answer depends on. The final answer must stay correct as it is.

Return JSON only:
{"edits": [{"message": <message number>, "find": "<exact text currently in that message>", "replace": "<new text>"}],
 "declared_change": "<the edit, worded as the instruction a generator would have been given>",
 "defect": "none"}
Each "find" must be copied exactly from the message content and be unique in it.
</request>
