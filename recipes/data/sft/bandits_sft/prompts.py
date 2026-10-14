"""Seed-conditioned generation and reference simulation prompts.

Adapted upstream prompts and modifications are documented in ../THIRD_PARTY.md.
"""
from __future__ import annotations

BASELINE_GEN = """You are an AI assistant that generates multi-turn conversation data for agent training. Your task is to create new agent trajectories based on existing examples.

## Example Trajectory:
{sample_text}

## Available Tools:
{available_tools}

CRITICAL FORMAT PRESERVATION REQUIREMENTS:
1. **STRICTLY PRESERVE ORIGINAL FORMAT**: You MUST maintain the EXACT format structure from the example trajectory (EXCEPTION: function_call turns may include <think> tags when reasoning is needed)
2. **NO SYSTEM PROMPT GENERATION**: Do NOT generate any SYSTEM messages - follow the system prompt from the original example and that will be preserved separately
3. **TOOL CONSTRAINT ADHERENCE**: You MUST STRICTLY use ONLY the tools listed in the "Available Tools" section above. DO NOT use any tools outside this specified allowed tool set. This is MANDATORY.
4. **FORMAT CONSISTENCY**: Maintain identical conversation structure, role naming conventions, and response patterns as shown in the example (EXCEPTION: function_call turns may include <think> tags when reasoning is added)
5. **TURN COUNT MATCHING**: Generate approximately the SAME NUMBER of conversation turns as the example trajectory - the generated conversation should have a comparable length and depth to the sample data

## CRITICAL COMPLIANCE RULES:
1. **FOLLOW THE SYSTEM POLICY**: Every action must respect the rules and procedures in the example's system prompt.
2. **ALWAYS COMMUNICATE KEY RESULTS**: State the concrete outcome values (amounts, identifiers, statuses) to the user explicitly.
3. **ID FORMAT CONSISTENCY**: When generating IDs, carefully observe the format and pattern used in the example trajectory and generate IDs that follow the same style. Avoid obvious fake patterns like "112233", "123456", etc. Use varied, realistic-looking combinations similar to those in the sample.

## FUNCTION_CALL TURN REQUIREMENTS:
1. **REASONING IN THINK TAGS**: When making function calls, add brief reasoning (1-3 sentences) inside `<think> </think>` tags ONLY in FUNCTION_CALL turns after you output 'FUNCTION_CALL:'
2. **SELECTIVE REASONING**: Not every function call needs reasoning. Only include it when it helps explain the complex decision-making process
3. **STRICT TURN CONSTRAINT**: Reasoning in `<think> </think>` tags should ONLY appear in FUNCTION_CALL turns, NEVER in HUMAN, GPT, OBSERVATION or additional turns
4. **FORMAT REQUIREMENT**: If reasoning is included, the FUNCTION_CALL turn are allowed to add the thinking sentences instead of only JSON format. You should follow this format:
   ```
   FUNCTION_CALL:
   <think>
   Brief reasoning about why this function call is needed (1-3 sentences). Ended with: I will call the function <function_name>.
   </think>
   {{"name": "function_name", "arguments": {{...}}}}
   ```
   Each FUNCTION_CALL is followed by an OBSERVATION turn holding the tool result.

## ABSOLUTE PROHIBITIONS:
- DO NOT use ANY tools that are not explicitly listed in the "Available Tools" section above
- DO NOT change the conversation format structure (human/gpt roles, value formatting, etc.) - EXCEPTION: function_call turns may include <think> tags when reasoning is needed
- DO NOT violate any fixed formatting elements, tool specifications, or requirements in system instructions from the example - EXCEPTION: function_call turns may include <think> tags when reasoning is added
- DO NOT generate significantly fewer or more turns than the example trajectory
- DO NOT invent or create new tools - use ONLY the provided tools

## Requirements:
1. Generate a completely NEW scenario/task that is different from the example but requires similar problem-solving patterns
2. Create a multi-turn conversation between Human and Assistant that demonstrates systematic problem-solving
3. The conversation should show the agent's reasoning process and step-by-step approach
4. **Start directly with a HUMAN message - do not include the SYSTEM content**
5. **CRITICAL: Ensure the generated conversation has approximately the same number of turns as the example trajectory**

## Agent Behavior Guidelines:
- Think step by step and explain reasoning
- Use ONLY the tools listed in the "Available Tools" section above - no exceptions
- Provide clear and helpful responses
- Maintain conversation flow and context
- Generate a conversation with comparable depth and turn count to the example
- Strictly adhere to the provided tool constraints without deviation
{extra_blocks}
## Output Format:
Generate the conversation:
HUMAN: [user message content]
ASSISTANT: [assistant reply content]
FUNCTION_CALL: [function call JSON, optionally preceded by <think>]
OBSERVATION: [tool result]
...(until the task is finished and the conversation is complete)
"""

# ---- format fixes (config baseline_prompt="fixed", the default; "original" = baseline's prompt as published) ----
# Found on real failure-analysis seeds: the model skipped the first HUMAN turn when it is a data blob,
# wrapped calls in <tool_call> tags or JSON lists (which baseline's filter deletes), and reused the example's case.
BLOCK_FORMAT = """
## Format Rules (common errors to avoid):
1. The conversation MUST start with a HUMAN turn. It is the task input: write it in exactly the same form as the example's first HUMAN turn (same structure and fields), with NEW values for a NEW case.
2. One tool call per FUNCTION_CALL turn, as one JSON object {"name": ..., "arguments": {...}}. For calls made in parallel, write consecutive FUNCTION_CALL turns, then their OBSERVATION turns in the same order.
3. Never write <tool_call>, <tool_response> or similar tags, and never put a JSON list in a FUNCTION_CALL turn.
4. Every OBSERVATION must match the format of the example's results exactly (same prefixes, field names and JSON shape); only the values change.
5. The final ASSISTANT turn must follow the output format the system prompt requires (if it requires a JSON object, output only that object).
6. A NEW case means different concrete inputs (entities, identifiers, elements, values). Let the evidence lead to whatever outcome it supports; do not copy the example's outcome by default.
"""

# ---- optional blocks for BASELINE_GEN (each tied to a feature flag) ----

BLOCK_SPEC = """
## Scenario To Realize (overrides "completely NEW scenario" above):
Write the conversation for exactly this scenario. The tool results (OBSERVATION) must be consistent with the
initial state, and the conversation must end in the expected final state if the agent behaves correctly.
{spec_json}
"""

BLOCK_PERSONA = """
## The Human's Behavior:
Persona "{persona}": {persona_desc}
Rules for the human: {universal_rules}
Style examples of how real users of this system write (match their register, not their content):
{user_examples}
"""

BLOCK_FAILURE = """
## Injected Tool Failure:
The tool result of FUNCTION_CALL number {at_call} (counting from 1) must be a realistic "{failure_type}" error,
in the same format real errors of this system use. The assistant must notice it and recover sensibly
(retry, use another route, or explain to the user). Do not let the failure make the task unrecoverable.
"""

BLOCK_RETRIEVAL = """
## Real Tool Results From This System (format reference):
These are real recorded results. Make every OBSERVATION match their structure, field names, value styles and
terseness exactly; only the values should change to fit the scenario.
{obs_examples}
"""

# ---- strategy instructions (BeyondWeb: many grounded transformation strategies beat one) ----

STRATEGIES: dict[str, str] = {
    "rephrase": "Same task and same solution path as the example, but a different user: new wording, tone and order of information. Keep the example's entities and tool results.",
    "entity_swap": "Same kind of task and solution path as the example, but with different entities and values (new people, ids, items, amounts, dates) that follow the example's formats.",
    "extend": "The example's task followed by a natural follow-up request from the same user that needs further tool use (a deeper version of the task).",
    "compose": "A single conversation in which the user needs both the example's task and the second example's task, handled in a sensible order.",
    "variant_outcome": "A similar request whose correct outcome differs from the example: e.g. the policy forbids it, an entity does not exist, a precondition fails, or the user must be told no and offered an alternative.",
    "new_scenario": "A completely new task in the same domain that requires similar problem-solving patterns (baseline's default).",
}

SPEC_BATCH = """You design scenarios for training a tool-using agent. Below is a real example conversation from the
system, the tools, and the system policy. Write {n} scenario specs, one per requested strategy, in order.

Rules:
- Each spec must follow its strategy, respect the system policy, and use only the listed tools.
- Specs must differ from the example and from each other in task, entities and details.
- Must not be: {avoid}
- initial_state: a JSON object holding every entity the tools will read (records keyed by id), with values in the
  same formats as the example's tool results. expected_final_state: the same object after a correct agent has
  finished (include unchanged entities only if relevant).
- user_facts: everything the user knows and can tell the agent (ids, names, preferences). The agent must be able
  to finish the task using only these facts and the tools.
- expected_behavior: 1-3 sentences on what a correct agent does, including any policy-required step
  (verification, confirmation, refusal).
- difficulty: easy | medium | hard.

SYSTEM POLICY:
{system}

TOOLS:
{tools}

EXAMPLE CONVERSATION:
{sample_text}
{second_example}
STRATEGIES (one spec each, in this order):
{strategy_list}

Return JSON: {{"specs": [{{"strategy": "...", "goal": "...", "user_facts": {{...}}, "initial_state": {{...}},
"expected_final_state": {{...}}, "expected_behavior": "...", "difficulty": "..."}}, ...]}}"""

SEED_CHECK = """Check this agent trajectory for use as a training seed (baseline pre-filter). Answer three checks:
completeness (task finished, all needed reasoning/tool calls/results present), logic (actions consistent with the
reasoning, results and policy), format (proper roles, valid tool-call JSON, results follow calls).

SYSTEM POLICY:
{system}

TRAJECTORY:
{transcript}

Return JSON: {{"complete": true/false, "logical": true/false, "well_formatted": true/false, "issues": ["..."]}}"""

# ---- mode B (reference simulation style per-step simulation) ----

USER_SIM = """You are role-playing a human user talking to an AI assistant of this service.

Your goal: {goal}
Facts you know (only these): {user_facts}
Persona "{persona}": {persona_desc}
Rules: {universal_rules}
Style examples of how real users write (match register, not content):
{user_examples}

Write only your next message to the assistant. When your goal is fully handled, or the assistant has clearly
and correctly said it cannot be done, reply with exactly [DONE]."""

TOOL_SIM = """You are the backend of a software system, simulating the result of one tool call.

Tool specification:
{tool_spec}

Call: {call_name}({call_args})

Current backend state (the only source of truth):
{state_json}

Real recorded results of this tool (copy their structure, field names, value styles and terseness exactly):
{obs_examples}

Rules:
1. FIRST validate the call against the tool specification (missing/invalid arguments -> an error result).
2. Answer ONLY from the current state. If a requested entity is not in the state, return a realistic not-found error.
3. If the call changes data, describe each change as an op on the state: {{"op": "set"|"delete"|"append", "path": "dotted.path", "value": ...}}.
   Read-only calls have no ops.
4. Never reveal anything not in the state.
{failure_rule}
Return JSON: {{"result": <the tool result exactly as the system would return it: string or JSON>, "ops": [...]}}"""

FAILURE_RULE = """5. THIS CALL FAILS: return a realistic "{failure_type}" error in the system's error format, and no ops."""

JUDGE = """You audit a synthetic agent training conversation. The SCENARIO says what should happen. Judge only
substance; do not prefer longer or more verbose answers.

SYSTEM POLICY:
{system}

SCENARIO:
{spec_json}

CONVERSATION (numbered messages):
{transcript}

CHECKLIST (for each item decide if it applies; if it applies, whether it is met):
{checklist}

Return JSON:
{{"task_success": true/false,
  "final_state": <the backend state after the conversation, reconstructed from the scenario's initial_state and the tool results>,
  "obs_consistent": true/false,      // every tool result is consistent with the state and with earlier results
  "user_realistic": true/false,      // the user only used facts they had and behaved like a real person
  "policy_followed": true/false,
  "hallucinations": ["..."],         // facts stated by the assistant or tools that no source supports
  "checklist_failed": ["..."],       // applicable checklist items that were NOT met (exact item text)
  "bad_steps": [<message indices of assistant messages that were mistakes>],
  "notes": "..."}}"""

DEFAULT_CHECKLIST = [
    "The assistant looks up the relevant record before changing it.",
    "The assistant obtains the user's explicit confirmation before any irreversible or state-changing action, when the policy requires it.",
    "The assistant does not claim an action succeeded unless a tool result shows it.",
    "The assistant does not state facts that no tool result, user message or policy provides.",
    "The assistant's final reply answers what the user asked.",
]


# ---- reference simulation environment simulator (verbatim from vendored simulated_general_env.py) ----
# One model plays both the human and the tools, using the seed conversation as reference data and
# replaying its recorded results when the agent's call matches. Used by loop_style "reference_env".
REFERENCE_ENV = """You are a simulation environment. Based on the RL model's response, you need to simulate the human or tool response.

System prompt (task description and rules):
{system_prompt}

Reference conversation examples (use this as reference data):
{ref_conv_text}

Current conversation history:
{history_text}

RL model's latest response:
{agent_message}

Requirements:
1. Based on the reference conversation and system prompt, determine how to respond
2. **Important for tool call format checking**: 
   - If the RL model is attempting to call a tool, check if it is properly wrapped in <tool_call></tool_call> tags
   - If the tool call is NOT in <tool_call> tags or has incorrect format (e.g., malformed JSON, wrong structure), return an error message like: "Error: Tool call must be wrapped in <tool_call></tool_call> tags with proper JSON format"
   - Only if the tool call format is correct, proceed to generate the tool response
3. **Important for tool responses**: 
   - If the RL model called a tool correctly, check if related results exist in the reference conversation
   - If YES: Include the related results from the reference conversation in the tool response
   - If NO: Generate reasonable results for the tool response based on the query parameters
   - If the format is incorrect, return an error message explaining why
3. If the RL model is waiting for human input, simulate the human's next message based on the reference conversation flow
4. Follow the pattern in the reference conversation examples, but adjust appropriately to fit the current conversation
5. If the task is completed or the conversation should end, output the "[TERMINATE]" marker

Output format:
- If continuing the conversation, output the message content directly
- If should end, output "[TERMINATE]"

Please directly generate the next message without the prefix "User/Tool response":"""


# ---- Analyzer (ANALYZER_SPEC.md): an agent writes the whole generation prompt from the seeds ----
# Seed expansion, not invention: each job keeps one real seed's situation and applies one assigned variation.
# Code fills the per-job slots and appends OUTPUT_FORMAT, the one part the analyzer cannot change.
ANALYZER_SLOTS = ("seed_trace", "variation", "tools", "system_prompt", "obs_examples", "generation_id")
ANALYZER_REQUIRED_SLOTS = ("seed_trace", "variation", "system_prompt", "tools", "generation_id")
VARIATION_KEYS = ("name", "applies_to", "requires_tools", "teaches", "change", "preserve", "dependencies", "correct_when")

# Why this prompt looks the way it does: it is the
# pre-baa7d37 seed-expansion prompt (concrete, one change per job, 6-12 variations) minus the harness source, which
# is replaced by contracts the analyzer reads off the seeds. No signals layer, no case-first, no "write from scratch".
# KEEP IT GENERAL: no tool, field, label or domain wording from any one agent (any one customer's agent).
# Everything specific must come from the seeds the analyzer is shown; examples only as clearly generic illustrations.
ANALYZER = """You will write the generator prompt for one agent's synthetic-data pipeline. In short: study the agent's
reviewed trajectories (seeds), their embedded system prompts and tool definitions below, then write (1) a factual
description of the harness as far as the seeds show it, (2) a set of controlled variations and (3) the generator
prompt. The generator EXPANDS one seed at a time: it keeps that seed's real situation and applies one assigned
variation; it never invents a new world. These seeds are the ONLY source: no harness code, catalog or simulator
exists, so anything the seeds do not show stays unknown. The full request follows the material.

<system_prompts>
{system_prompts}
</system_prompts>

<seeds count="{n_seeds}" note="reviewed, good trajectories">
{seeds}
</seeds>

<tool_schemas>
{tools}
</tool_schemas>

<real_tool_results note="sample">
{obs}
</real_tool_results>
{revision}
<request>
Why this matters: the generated trajectories become supervised fine-tuning data. The trained model learns every
assistant turn: which tool call to make given what it has seen so far, and the final answer given the evidence. A
trajectory teaches the right behaviour only if its situation is real and its decisions follow from its evidence under
the rules in the system prompt. Invented worlds, impossible tool outputs and answers the evidence does not support
teach the wrong thing.

Seed expansion: each generated trajectory starts from ONE seed and keeps the seed's situation (who or what the request
is about, the environment, the material the agent works from, the wording style of the user turns) except what the ONE assigned variation changes. Everything that
depends on the change is updated: later calls, their results, the final answer. A seed tool result is reused only
when its call and the evidence behind it remain valid after the variation; otherwise it is updated or omitted. The
outcome follows from the evidence; never choose the outcome first.

The final answer is derived again from what the generated conversation shows, under the system prompt; it is not
copied. Keep the seed's answer, and its wording, where the evidence it relied on is unchanged and still supports it.
Where the seed's own answer says something the visible evidence does not support, do not carry that forward: correct
or drop it. An answer may claim only what each observation establishes: one kind of observation says nothing about
another kind of state (an image does not show internal or network state; an unavailable source shows nothing about
that source; a failed search is not proof of absence unless the tool's contract says it is exhaustive).

Produce:

1. "observed_contracts" (markdown): facts only, in three headed sections.
   - Explicit: rules stated in the system prompts and tool definitions (task, input and output contracts, closed
     vocabularies, evidence-to-answer rules).
   - Observed: formats and behaviours the seeds actually show: the user-message shape, any fixed wrapper or numbering on
     tool results, each tool's result shape for each outcome seen (success, empty, unavailable, error), whether the
     agent uses tools at all or answers from material given in the user turn, batching, how confidence and
     causes relate to evidence. A seen maximum (such as 3 calls) is not a limit. An unavailable result does not
     establish what a successful result looks like.
   - Unknown: tools, result branches and fields no seed demonstrates.
   Do not turn unknowns into rules and do not import outside knowledge.

2. "variations": 4-6 controlled transformations of the seeds that are not tied to one seed's literal values (one short call per seed adds more
   seed-specific ones afterwards, so keep these general and few). Each changes what the agent must recognize, handle or
   justify: an evidence source becomes unavailable, an observation supports a different reading, a fact the
   agent relied on is removed or contradicted, two sources disagree, the same task on a different part of the same
   material, and so
   on. The correct answer may stay the same when the evidence the agent handles changed. Nothing cosmetic. Rules:
   - Change ONE thing. Keep every evidence channel the seed lacks unavailable unless the variation is exactly about
     making it available. Do not replace the seed's whole situation (a different subject, request, material or environment).
   - A change must fit the seed's situation. Do not attach a fact from one context to another it does not belong to
     (if the seed's evidence says two contexts differ, for example a reference and a current one, that difference
     stays), and do not drop a call or observation that established such a fact unless the change makes it unnecessary.
   - Do not invent specifics the seeds do not contain (names, identifiers, URLs, quoted text, document content). Describe
     new observations with words already in the seed, or choose a different variation.
   - Use only result shapes listed under Explicit or Observed. If a variation would need a branch listed under
     Unknown, pick a different variation instead of guessing its shape.
   - Write each variation about a CLASS of situation, not one seed's literal values: state the evidence condition
     ("a source becomes unavailable", "the deciding observation changes") and let the generator map it onto its
     assigned seed's own calls and values. Set eligible_seeds to EVERY shown seed that meets the preconditions
     (seeds under the same label usually do); a variation fit to a single seed is acceptable only when no other seed fits.
   - At most two variations may share the same decision_type and answer_transition; vary both across the set
   - Spread the variations over the seeds and over different decisions (tool choice, evidence reading,
     corroboration, uncertainty, when to stop). Do not write several variations that teach the same lesson.
   Fields:
   - name: short_snake_case
   - applies_to: system prompt labels (["S1", ...]) or ["all"]. Prompts under one label are variants of the same task
     (they differ in context notes), so a label's seeds all ask the same question; list ONLY labels whose seeds make the change
     meaningful, and say in "change" which kind of seed it needs (tools called, evidence present)
   - requires_tools: tools the seed must already call for the variation to make sense ([] if none)
   - eligible_seeds: ids of the shown seeds (each seed's id attribute) whose situation actually meets this
     variation's preconditions; the variation is only ever assigned to these. Check the evidence the change needs, not
     just the system label. A generator may still refuse a pairing; that is allowed
   - teaches: the behaviour it demonstrates
   - change: what changes vs the seed
   - preserve: what must stay as in the seed
   - dependencies: observations AND assistant actions, arguments and stopping that must change as a consequence
   - correct_when: the evidence that makes the final answer correct under the system prompt; conditional, never a
     predetermined answer
   - decision_type: one or two words for the kind of decision the variation exercises (for example tool choice,
     evidence reading, scope of a claim, handling a missing source, stopping); use your own labels, consistently
   - answer_transition: "unchanged", "changes" or "null": what the correct answer does relative to the seed's. It is a
     planning label for spreading the set; the generator never sees it and it never fixes an answer
   - basis: "seed" when the variation only reuses result shapes already present in the seeds; otherwise a verbatim
     quote from your own Explicit or Observed section showing the result shape or branch the variation needs. Code
     checks the quote. A shape you list under Unknown cannot be a basis: pick a different variation instead

3. "template": the whole instruction the generator sees for one job. Code fills these slots, written as {{name}}:
   {{seed_trace}} the seed as {{"messages": [...]}} JSON; {{variation}} the one assigned variation as JSON;
   {{system_prompt}} the seed agent's system prompt (its question and answer contract); {{tools}} its tool schemas;
   {{generation_id}} a unique job id; optionally {{obs_examples}} real tool results from other seeds. All except
   obs_examples are required; use no other {{word}} slots (other literal braces are fine). Layout: each filled input
   in its own clearly delimited block first, then the instructions, then a short restatement of the variation's
   change and the rule to change nothing else. Keep the instruction part under about 1,200 words. It must say:
   expand THIS seed with ONLY the assigned change; keep the world; and number the concrete contracts you read off the
   seeds (any result wrapper or counter the seeds show, each tool's result shapes, how unavailable differs from
   empty where the seeds distinguish them, the answer format, keys and closed vocabularies the system prompt
   states, and when the system prompt allows a conclusion or cause versus requiring a cautious or null one).
   It must also say: keep every call whose result established a fact the conversation still relies on; take specific
   names, identifiers and content from the seed, never invent them; leave confidence and numeric fields to the system
   prompt's conventions rather than raising or lowering them without cause.
   It must also say: when an observation changes or a source becomes unavailable, the final answer's explanatory text
   (detail, reason, notes) must say what the changed evidence does and does not establish, in the seed's own style;
   do not leave the seed's explanation word for word when the evidence it cites has changed.
   It must also say: before answering, check every observation you kept from the seed against the observations you
   changed. If a kept observation contradicts a changed one (two observations that cannot both be true of the same thing),
   rewrite the kept one consistently or return invalid_pairing; never leave both and answer from one of them. A changed
   observation must also describe something that can exist in the seed's situation (do not place an item in a place or
   context that the kept observations say is a different one).
   It must also say: derive the final answer again from the visible evidence (keep the seed's answer where its
   evidence is unchanged and still supports it; do not carry forward a claim the evidence does not support), and
   claim only what each observation establishes. The generator maps the variation's condition onto this seed's own
   calls and values.
   The generator writes actions from what the agent has seen so far; it never uses the
   variation's intent as evidence. Do not describe the output format: code appends a fixed block that specifies
   the one JSON object to return.

4. "combinations": [] unless two variations interact in a way the agent must handle and neither removes evidence the
   other needs; then pairs of variation names.

5. "rationale" (markdown): what you learned and why the contracts, variations and template look the way they do{rationale_extra}.

Return exactly one JSON object with these five keys, shaped like:
{{"observed_contracts": "## Explicit\\n...", "variations": [{{"name": "source_unavailable", "applies_to": ["S1"],
"requires_tools": ["tool_a"], "eligible_seeds": ["<seed id>"], "basis": "seed", "decision_type": "missing source", "answer_transition": "unchanged", "teaches": "...", "change": "...", "preserve": "...", "dependencies": "...",
"correct_when": "..."}}], "combinations": [],
"template": "<seed>\\n{{seed_trace}}\\n</seed>\\n...", "rationale": "..."}}
</request>"""

ANALYZER_REVISION = """
<previous_version number="{version}">
<observed_contracts>
{observed_contracts}
</observed_contracts>
<variations>
{variations}
</variations>
<template>
{template}
</template>
</previous_version>

<covered_so_far note="cells already generated in earlier rounds; a finished (seed, variation) cell is never run again">
{covered}
</covered_so_far>

<diagnosis note="from a separate reviewer of the previous version's pilot">
{notes}
</diagnosis>

Revising: the next round only runs cells that are not finished, so the revision must open new ones. Do this by (a) widening
eligible_seeds of variations that worked to further seeds whose situation fits them (a variation name keeps its finished
cells, new seeds under it are new cells), and (b) adding variations for evidence conditions not yet covered, especially
for seeds that have few or no untried cells. Do not re-propose a finished cell under a new name. Write the next version
of all of them. Fix what the diagnosis found without breaking what worked. Drop or
repair variations that kept failing; keep the ones that worked. Consolidate rather than accumulate: replace rules
that did not help, merge overlapping ones, and resolve instructions that conflict. In the rationale, list what you
removed or merged and why.
"""

_FORMAT_RULES = """Rules: roles are only user, assistant, tool. "arguments" is a JSON object. Every tool call gets exactly one tool
message with its id, right after the assistant message that made the call. Tool "content" is a string. The last
message is an assistant reply with no tool calls."""

OUTPUT_FORMAT = """

## Output format (fixed)
Return ONE JSON object and nothing else (no prose, no markdown fences, no planning text):
{"messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "<tool>", "arguments": {...}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "<the tool's result as a string>"},
  {"role": "assistant", "content": "<final reply>"}
]}

If the assigned variation cannot be applied to this seed without replacing its situation or inventing facts the seed
does not support, return instead exactly {"invalid_pairing": true, "reason": "<why it does not fit this seed>"}.
Refusing is a normal, valued outcome; a forced conversation is not.
""" + _FORMAT_RULES

# Case-first variant: the generator settles the changed situation before writing the conversation. The case block is
# kept in the trace's meta for the judge and for review; it is never part of the training messages.
CASE_KEYS = ("assigned_change", "kept", "dependent_observations", "unavailable_or_uncertain")

OUTPUT_FORMAT_CASE = """

## Output format (fixed)
Return ONE JSON object and nothing else (no prose, no markdown fences, no planning text). Write "case" first:
{"case": {
   "assigned_change": "<the assigned variation's change, as applied to this seed>",
   "kept": "<what stays exactly as in the seed: subject, environment, material, request context, ...>",
   "dependent_observations": "<which tool calls and results change because of the change, and how>",
   "unavailable_or_uncertain": "<evidence that is unavailable, missing or ambiguous in the changed situation>"},
 "messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "<tool>", "arguments": {...}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "<the tool's result as a string>"},
  {"role": "assistant", "content": "<final reply>"}
]}
The case describes the situation only: it states no answer, and it cannot widen the assigned change. The agent in
"messages" never sees the case: every assistant turn must follow from the messages before it.
If the assigned variation cannot be applied to this seed without replacing its situation or inventing facts the seed
does not support, return instead exactly {"invalid_pairing": true, "reason": "<why it does not fit this seed>"}.
Refusing is a normal, valued outcome; a forced conversation is not.
""" + _FORMAT_RULES

# Split mode: the generator plays only the world (user and observations); the agent model writes the final answer in a
# separate call that never sees the variation. Appended after the output format, so it overrides its last rule.
ENV_ONLY = """

## Split mode (overrides the rule on the last message)
You write everything the agent receives, and the agent's own steps up to its final answer, but NOT the final answer:
the agent itself answers later from these messages alone. End "messages" with the last observation the agent sees
before answering (the last tool result, or the last user message when there are no tools). Do not write a final
assistant reply.
Every observation states only what its source returns, in the style of real outputs of that source shown above. No
observation states what the agent should conclude, what changed compared to before, or why.
"""


# Split mode: when the agent calls a tool after the generator's part, this simulates the result. The conversation so far
# is the truth; the simulator knows what is true in the varied situation but not what the agent should conclude.
TOOL_SIM_TRACE = """You simulate the result of one tool call for an agent, inside a situation that is already partly
recorded. Everything in the conversation so far, including earlier tool results, is true and must stay true.

<situation note="what is true in this situation, beyond the conversation">
{situation}
</situation>

<tool_spec>
{tool_spec}
</tool_spec>

<real_results note="real recorded results of this tool: copy their structure, field names, value styles and terseness">
{obs_examples}
</real_results>

<conversation>
{conversation}
</conversation>

<call>{call_name}({call_args})</call>

Rules: if the call is invalid for the tool specification, return the error a real system would return. Otherwise
return only what this source would report, consistent with every earlier observation and the situation; for anything
neither establishes, return what a real system plausibly holds, plain and terse. A result never states a conclusion,
what changed, or what the agent should do.
Return JSON only: {{"result": "<the result exactly as the system returns it, as one string>"}}"""


DIAGNOSE = """<signals>
{signals}
</signals>

<observed_contracts>
{observed_contracts}
</observed_contracts>

<variations>
{variations}
</variations>

<generator_template version="{version}">
{template}
</generator_template>

<pilot_report>
{report}
</pilot_report>

<rejected_traces shown="{n_rejected}" total="{total_rejected}">
{rejected}
</rejected_traces>

<kept_traces shown="{n_kept}" total="{total_kept}">
{kept}
</kept_traces>

<request>
You review the pilot of a seed-expansion generator. Each job took ONE real, reviewed seed and ONE assigned variation;
the output should be the seed's real situation with exactly that change, every dependent detail updated, and the
answer following from the evidence. Each sampled trace above is shown next to its full seed. Find what makes outputs
rejected, wrong, or weak as training data, and what the generator template or the variations should change. Look
especially for:
- refusals (invalid_pairing): decide whether each is justified. A justified refusal means the ASSIGNMENT was wrong:
  fix the variation's eligible_seeds or preconditions, not the generator template. An unjustified one is a template problem;
- drift: changes the variation did not ask for (new subject, environment, material, story), or a made-up world;
- dependencies not updated: a changed observation whose later calls or final answer still follow the seed;
- tool results that break the tool's supplied contract or demonstrated shape (see the demonstrated formats and supplied system prompt), or state a verdict the
  tool cannot know;
- final answers the evidence does not support, especially named causes without their detecting evidence;
- a named signal that is not actually exercised, or an answer shortcut that bypasses the intended evidence;
- variations that are cosmetic (nothing changes in what the agent must recognize, handle or justify) or keep
  failing. A variation whose correct answer stays the same is fine when the evidence the agent handles changed. Judge
  distinctness by the evidence condition a trace exercises, never by whether the answer or the action sequence differs;
- claims in a final answer that the cited observation cannot establish, including claims inherited from the seed's
  answer: inheriting is not an excuse;
- variations that name one seed's literal values and so fit only that seed (they cap how many distinct examples exist);
- template instructions that conflict with each other, are obsolete, or never mattered (candidates to remove).
Do not recommend banning or blacklisting seed content: staying close to the seed is the goal. Judge from the traces,
and cite trace ids and counts.

Return JSON only: {{"notes": "<markdown: concrete failure patterns with trace ids and counts, a verdict per variation
(keep / fix / drop), and specific template changes, including what to remove; most important first>"}}
</request>"""

# Judge for seed-expansion traces: sees the seed, the variation and observed contracts; gives a grounded verdict per check.
# Code verifies that every quoted excerpt really occurs in the generated trace or the seed.
JUDGE_CHECKS = ("only_declared_change", "dependencies_updated", "tool_contracts_respected", "observations_consistent",
                "case_consistent", "policy_followed", "task_success")

JUDGE_EXPANSION = """<observed_contracts>
{observed_contracts}
</observed_contracts>

<system_policy>
{system}
</system_policy>

<seed note="the real trajectory before the variation; messages cited as seed:N">
{seed}
</seed>

<variation>
{variation}
</variation>

{case}
<generated note="the trajectory to judge; messages cited by number">
{transcript}
</generated>

<checklist>
{checklist}
</checklist>

<request>
You audit one synthetic training trajectory (GENERATED) made by applying the VARIATION to the real, reviewed SEED.
The seed shows the situation before the change; it is not an answer key, because after the variation the correct
answer may differ. Judge substance only; do not prefer longer answers.

Distinguish failed execution, unavailable evidence, an observed negative result, and positive evidence. State exactly
what the operation tested. Do not infer broader absence or inactivity than the result supports. Attribute arguments
to the call and observations to the returned result.

Two kinds of checks. CONSTRUCTION: was the variation applied to the seed correctly? Compare the seed, the variation,
the case (when given; the generator's own description of the changed situation) and the generated observations.
BEHAVIOUR: is each assistant action justified by what the agent had seen at that point? The agent never sees the
variation or the case: for behaviour, only messages before the turn count as evidence. Observed formats are examples, not an exhaustive implementation specification; unknown rules cannot justify rejection.

For each check, give a verdict: "pass", "fail", "cannot_determine" (the trajectory lacks what you need to decide) or
"not_applicable". Back it with evidence: the message numbers (GENERATED as N, SEED as "seed:N") with short excerpts
copied exactly from those messages (each excerpt one contiguous span of at most about 20 words from the one message
it cites; inside JSON, quote a single key and its value), then a one-sentence reason. When the problem is something missing (an
unsupported claim), quote the claim and say what support is missing.

Construction checks:
- only_declared_change: everything that differs from the seed is what the variation's "change" asks for, or a
  consequence covered by its "dependencies"; nothing else (subject, environment, material, story) drifted. When the variation
  combines several changes, each is applied and none removes evidence another needs. A case cannot authorize a change
  the variation does not ask for.
- dependencies_updated: every later call and tool result is consistent with the change; no stale seed observation
  the change invalidated. The final answer follows the generated observations; keeping the seed's answer is valid
  when those observations still support it. Test it explicitly: for each seed observation kept unchanged, ask
  whether it still holds once the changed observation is true. If an unchanged observation asserts something the
  changed one now contradicts, verdict "fail" with excerpts from BOTH messages. Do not reconcile the two by
  inventing a distinction the tool's contract does not make. Unchanged observations that do not bear on the change
  are fine and need no comment.
- tool_contracts_respected: every tool result is something that tool can return per the supplied definitions and demonstrated formats (shape,
  content, no verdict the tool cannot know).
- observations_consistent: tool results agree with each other and with the failure context.
- case_consistent: the observations express the case, and the case stays within the variation. "not_applicable" when
  no case is given.
Behaviour checks:
- policy_followed: the agent follows the system policy.
- task_success: the final answer is correct for the generated evidence, using only what the conversation shows. The
  system policy and the rules explicitly present in the supplied system prompt decide what is correct; the variation's "correct_when" is a
  hypothesis to check against them, never a rule that overrides them.
A claim the visible evidence does not support is a hallucination even when the seed's answer made the same claim: the
seed is a reference, not proof.
Then judge every checklist item the same way.

Turns: one entry for EVERY assistant message in GENERATED, in order: its number, "justified" or "unjustified" (or
"cannot_determine"), "relies_on": the numbers of the earlier messages that justify it (use "system" when the system
policy alone justifies it), and a short reason. A call or answer that relies on something shown only later, or only
in the case or variation, is unjustified.

Return JSON only. "checks" must contain exactly these seven keys, each with a verdict: only_declared_change,
dependencies_updated, tool_contracts_respected, observations_consistent, case_consistent, policy_followed,
task_success.
{{"checks": {{"<check name>": {{"verdict": "pass", "evidence": [{{"message": "3", "excerpt": "..."}}], "reason": "..."}}}},
  "checklist": [{{"item": "<exact item text>", "verdict": "...", "evidence": [...], "reason": "..."}}],
  "turns": [{{"message": 1, "verdict": "justified", "relies_on": [0], "reason": "..."}}],
  "hallucinations": ["<claims no source supports>"], "notes": "..."}}
</request>"""



SEED_VARIATIONS = """<observed_contracts>
{observed_contracts}
</observed_contracts>

<system_prompt label="{label}">
{system}
</system_prompt>

<seed id="{seed_id}">
{seed}
</seed>

<siblings note="other seeds under the same label, for contrast only; never copy their observations">
{siblings}
</siblings>

<already_covered note="cells finished on this seed, and variation names that already exist anywhere">
{covered}
</already_covered>

<request>
Propose {n} NEW controlled variations of THIS seed, each changing ONE thing about the evidence the agent handles,
so that the correct answer follows the changed evidence under the system prompt. Describe an evidence CONDITION (not
literal strings), keep the seed's situation, use only result shapes listed under Explicit or Observed in the
contracts (quote the supporting line in "basis", or "seed" when you only reuse shapes in the seed), invent no names,
ids, URLs or quoted text, and never attach a fact to a situation the seed's other evidence says it does not belong to.
Prefer conditions that change what the assistant says or does, not only a tool result. The answer your condition
implies must be reachable with the calls the seed already makes plus at most one more; if the system prompt only allows
that answer after an exhaustive or multi-step process, or needs evidence the tools cannot give, do not propose it. Do not reuse a name or a
condition from already_covered. eligible_seeds must contain this seed id and may add sibling ids the condition truly fits.

Return JSON only: {{"variations": [{{"name": "...", "applies_to": ["{label}"], "requires_tools": [], "eligible_seeds": ["{seed_id}"],
"basis": "seed", "decision_type": "...", "answer_transition": "unchanged|changes|null", "teaches": "...", "change": "...",
"preserve": "...", "dependencies": "...", "correct_when": "..."}}]}}
</request>"""


# Wider variation (ab-vary): besides one evidence change, a different user goal in the same world ("task"). Coverage
# lists what the seeds and existing variations already have, so proposals go where little exists. (A "cross_seed" kind,
# this seed's request over a sibling's material, was tried and dropped: 0/2 correct and 0/2 kept on two test sets.)
VARIATION_KINDS = ("task", "evidence")
SEED_VARIATIONS_WIDE = """<observed_contracts>
{observed_contracts}
</observed_contracts>

<system_prompt label="{label}">
{system}
</system_prompt>

<seed id="{seed_id}">
{seed}
</seed>

<seed_tools note="the only tools this agent has in this seed; every variation uses only these">
{tools}
</seed_tools>

<siblings note="other real seeds under the same label, for contrast only; never copy their observations">
{siblings}
</siblings>

<coverage note="what the seeds and existing variations already cover, most common first">
{coverage}
</coverage>

<already_covered note="variation names that already exist anywhere">
{covered}
</already_covered>

<request>
Purpose: these variations become supervised training conversations for the agent above. Changing one fact in a seed
teaches little, because the new conversation repeats the seed's request, steps and reasoning. Each variation should
make the agent handle something it has not handled in the seeds, while staying a conversation that could really
happen with this agent, its tools and its sources.

Propose {n} NEW variations of THIS seed, spread over the two kinds below. Use a kind only when the system prompt and
the seed allow it: an agent that always receives the same kind of request has no realistic "task" variation, so give
it none and use the other kinds.
- "task": the same world (system prompt, tools, material, what the sources hold) with a different user goal that this
  agent handles under its system prompt: another question about the same material, another action, or another target.
  The calls, the evidence used and the answer all follow from the new request.
- "evidence": one changed condition in the evidence, so that the correct answer follows the changed evidence.

Make the {n} variations differ from each other in what the agent must do or conclude, and aim them at what coverage
shows as rare or missing; leave the most common decision types and answers alone.

Every variation stays realistic and answerable:
- The answer follows from the visible evidence under the system prompt, and is reachable with at most two calls more
  than the seed makes, using only the tools listed in seed_tools.
- Results use only shapes listed under Explicit or Observed in the contracts: quote the supporting line in "basis", or
  write "seed" when you reuse shapes shown in the seed.
- Names, ids, values and quoted text come from this seed. Describe conditions and requests in words; the
  generator writes the literal content later.
- The request is one the system prompt allows, and the evidence it needs is something the tools can return.
- The name is new (not in already_covered).


Return JSON only, every field filled: {{"variations": [{{"name": "...", "kind": "task|evidence", "applies_to": ["{label}"], "requires_tools": [], "eligible_seeds":
["{seed_id}"], "basis": "seed", "decision_type": "...", "answer_transition": "unchanged|changes|null", "teaches": "...",
"change": "...", "preserve": "...", "dependencies": "...", "correct_when": "..."}}]}}
</request>"""

# Appended to the generator prompt for a "task" variation: the user's goal is what changes.
TASK_NOTE = """

## A different request
This variation changes what the user asks (see "change" in the <variation> block). Replacing the seed's request with
that one is the assigned change, not a new situation: keep the world, the sources, the material and the style, and let
the calls, the evidence used and the final answer follow the new request. Return invalid_pairing only when the new
request cannot be served with the seed's world and tools.
"""


# Consistency check (one call, any domain): the model writes out what each observation states, then looks for facts
# that cannot all be true, results a source could not return for the call that produced them, and answer claims no
# observation supports. Code decides the verdict from the lists, so the label cannot disagree with the findings.
CONSISTENCY_CHECK = """<agent_system_prompt>
{system}
</agent_system_prompt>
{reference}
<tools note="the tools this agent can call, with their parameters">
{tools}
</tools>

<conversation note="messages are numbered; observations are tool results and any documents or data inside user messages">
{conversation}
</conversation>

<request>
Purpose: this conversation is a candidate training example. A conversation whose observations contradict each other,
contain a result its source could not have returned, or whose final answer claims something no observation shows would
teach the agent to reason from impossible evidence. Your job is to find such problems, carefully and literally.

Work in four steps.
1. facts: for every observation, list the concrete facts it states (values, states, counts, names, dates, what is
   present or absent), each tied to its message number. Include facts in the user's request and in documents.
2. contradictions: pairs of facts that cannot both be true of the same thing at the same time. Two observations of
   different things, at different times, or with different scopes are not a contradiction; neither is different
   wording for the same fact.
3. impossible_results: results that could not come from the call that produced them, judged from the call's own
   arguments and the tool's description.
4. unsupported_claims: statements in assistant messages (especially the final answer) that present as fact something
   no observation shows.

When a reference_seed is given, it is a real recorded conversation of this same agent: the conclusions it draws from a
kind of observation are this agent's established conventions, so the same conclusion drawn from the same kind of
observation here is supported, not a problem.

List only problems you can point to with message numbers. Leave a list empty when there is none.

Return JSON only: {{"facts": [{{"msg": <n>, "facts": ["..."]}}], "contradictions": [{{"msgs": [<n>, <m>], "why": "..."}}],
"impossible_results": [{{"msg": <n>, "why": "..."}}], "unsupported_claims": [{{"msg": <n>, "claim": "...", "why": "..."}}]}}
</request>"""
