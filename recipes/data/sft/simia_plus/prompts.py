"""Prompts.

SIMIA_GEN is Simia's trajectory-synthesis prompt (arXiv 2511.01824, Fig 9; microsoft/Simia-Agent-Training,
Simia_SFT/Tau2/utils/conversation_generator.py, MIT). The domain-specific retail/airline rule blocks are
replaced by one generic block so it runs on any domain. Everything else is kept, including the
HUMAN/ASSISTANT/FUNCTION_CALL/OBSERVATION text format and its parser. Improvement blocks are appended only
when their feature is switched on, so the baseline stays Simia.
"""
from __future__ import annotations

SIMIA_GEN = """You are an AI assistant that generates multi-turn conversation data for agent training. Your task is to create new agent trajectories based on existing examples.

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

# ---- format fixes (config simia_prompt="fixed", the default; "original" = Simia's prompt as published) ----
# Found on real failure-analysis seeds: the model skipped the first HUMAN turn when it is a data blob,
# wrapped calls in <tool_call> tags or JSON lists (which Simia's filter deletes), and reused the example's case.
BLOCK_FORMAT = """
## Format Rules (common errors to avoid):
1. The conversation MUST start with a HUMAN turn. It is the task input: write it in exactly the same form as the example's first HUMAN turn (same structure and fields), with NEW values for a NEW case.
2. One tool call per FUNCTION_CALL turn, as one JSON object {"name": ..., "arguments": {...}}. For calls made in parallel, write consecutive FUNCTION_CALL turns, then their OBSERVATION turns in the same order.
3. Never write <tool_call>, <tool_response> or similar tags, and never put a JSON list in a FUNCTION_CALL turn.
4. Every OBSERVATION must match the format of the example's results exactly (same prefixes, field names and JSON shape); only the values change.
5. The final ASSISTANT turn must follow the output format the system prompt requires (if it requires a JSON object, output only that object).
6. A NEW case means different concrete inputs (entities, identifiers, elements, values). Let the evidence lead to whatever outcome it supports; do not copy the example's outcome by default.
"""

# ---- optional blocks for SIMIA_GEN (each tied to a feature flag) ----

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
    "new_scenario": "A completely new task in the same domain that requires similar problem-solving patterns (Simia's default).",
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

SEED_CHECK = """Check this agent trajectory for use as a training seed (Simia pre-filter). Answer three checks:
completeness (task finished, all needed reasoning/tool calls/results present), logic (actions consistent with the
reasoning, results and policy), format (proper roles, valid tool-call JSON, results follow calls).

SYSTEM POLICY:
{system}

TRAJECTORY:
{transcript}

Return JSON: {{"complete": true/false, "logical": true/false, "well_formatted": true/false, "issues": ["..."]}}"""

# ---- mode B (Simia-RL style per-step simulation) ----

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


# ---- Simia-RL environment simulator (verbatim from vendored simulated_general_env.py) ----
# One model plays both the human and the tools, using the seed conversation as reference data and
# replaying its recorded results when the agent's call matches. Used by loop_style "simia_env".
SIMIA_ENV = """You are a simulation environment. Based on the RL model's response, you need to simulate the human or tool response.

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

ANALYZER = """You will design a synthetic-data generator for one agent. In short: study the agent's real harness and its
reviewed trajectories (seeds) below, then write (1) a description of the harness, (2) a set of controlled variations
and (3) the generator prompt. The generator EXPANDS one seed at a time: it keeps that seed's real situation and
applies one assigned variation; it never invents a new world. The full request follows the material.

<harness_source note="the agent's real implementation: established fact">
{harness}
</harness_source>

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
the harness rules. Invented worlds, impossible tool outputs and answers the evidence does not support teach the
wrong thing.

Seed expansion: each generated trajectory starts from ONE seed and keeps the seed's situation (platform, app or site,
page, element, failure context, wording style) except what the ONE assigned variation changes. Everything that
depends on the change is updated: later calls, their results, the final answer. A seed tool result is reused only
when its call and the evidence behind it remain valid after the variation; otherwise it is updated or omitted. The
outcome follows from the evidence; never choose the outcome first.

Produce:

1. "harness" (markdown): the agent's task; its input and output contracts; for every tool, what it can and cannot
   observe and its exact result shape for every outcome (success, empty, unavailable, error), including the
   "[obs:N] " prefix; the evidence-to-answer rules (what supports each conclusion or cause, when the answer must
   stay cautious); the unknowns. Tag every claim [established: <file>] (harness source or schemas), [inferred]
   (seen in seeds only) or [unknown]. Do not turn unknowns into rules.

2. "variations": 6-12 controlled transformations. Each changes what the agent must recognize, handle or justify: an
   evidence source becomes unavailable, an observation supports a different reading, an obstruction covers or
   clears the target, two sources disagree, a different element or step on the same page, and so on. The correct
   answer may stay the same when the evidence the agent handles changed. Nothing cosmetic (renaming, rewording).
   Mostly ordinary, realistic changes, plus a few decision-boundary cases taken from rules and incidents the
   harness source documents; never invent rules. Fields:
   - name: short_snake_case
   - applies_to: system prompt labels (["S1", ...]) or ["all"]
   - requires_tools: tools the seed must already call for the variation to make sense ([] if none)
   - teaches: the behaviour it demonstrates
   - change: what changes vs the seed
   - preserve: what must stay as in the seed
   - dependencies: what must be updated to stay consistent
   - correct_when: the evidence that makes the final answer correct under the harness rules

3. "template": the whole instruction the generator sees for one job. Code fills these slots, written as {{name}}:
   {{seed_trace}} the seed as {{"messages": [...]}} JSON; {{variation}} the one assigned variation as JSON;
   {{system_prompt}} the seed agent's system prompt (its question and answer contract); {{tools}} its tool schemas;
   {{generation_id}} a unique job id; optionally {{obs_examples}} real tool results from other seeds. All except
   obs_examples are required; use no other {{word}} slots (other literal braces are fine). Layout: each filled input
   in its own clearly delimited block first, then the instructions, then a short restatement of the variation's
   change and the rule to change nothing else. Keep the instruction part under about 1,200 words: state principles
   and the exact contracts, not long if-then rule lists. Do not describe the output format: code appends a fixed
   block that specifies the one JSON object to return.

4. "combinations": pairs of variation names that can be applied to the same seed together, where the two changes
   interact in a way the agent must handle (for example one source becomes unavailable while another shows an
   obstruction). Only pairs whose changes do not remove evidence the other needs and do not contradict each other;
   [] if none qualify.

5. "rationale" (markdown): what you learned and why the harness description, variations, combinations and template
   look the way they do{rationale_extra}.

Return exactly one JSON object with these five keys, shaped like:
{{"harness": "## Task\\n...", "variations": [{{"name": "har_unavailable", "applies_to": ["S1"], "requires_tools":
["query_har"], "teaches": "...", "change": "...", "preserve": "...", "dependencies": "...", "correct_when": "..."}}],
"combinations": [["har_unavailable", "overlay_covers_target"]],
"template": "<seed>\\n{{seed_trace}}\\n</seed>\\n...", "rationale": "..."}}
</request>"""

ANALYZER_REVISION = """
<previous_version number="{version}">
<harness>
{harness}
</harness>
<variations>
{variations}
</variations>
<template>
{template}
</template>
</previous_version>

<diagnosis note="from a separate reviewer of the previous version's pilot">
{notes}
</diagnosis>

Revising: write the next version of all of them. Fix what the diagnosis found without breaking what worked. Drop or
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
""" + _FORMAT_RULES

# Case-first variant: the generator settles the changed situation before writing the conversation. The case block is
# kept in the trace's meta for the judge and for review; it is never part of the training messages.
CASE_KEYS = ("assigned_change", "kept", "dependent_observations", "unavailable_or_uncertain")

OUTPUT_FORMAT_CASE = """

## Output format (fixed)
Return ONE JSON object and nothing else (no prose, no markdown fences, no planning text). Write "case" first:
{"case": {
   "assigned_change": "<the assigned variation's change, as applied to this seed>",
   "kept": "<what stays exactly as in the seed: platform, app or site, page, element, failure context, ...>",
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
""" + _FORMAT_RULES

DIAGNOSE = """<harness_description>
{harness}
</harness_description>

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
- drift: changes the variation did not ask for (new site, platform, element, story), or a made-up world;
- dependencies not updated: a changed observation whose later calls or final answer still follow the seed;
- tool results that break the tool's real contract or shape (see the harness description), or state a verdict the
  tool cannot know;
- final answers the evidence does not support, especially named causes without their detecting evidence;
- variations that are cosmetic (nothing changes in what the agent must recognize, handle or justify) or keep
  failing. A variation whose correct answer stays the same is fine when the evidence the agent handles changed;
- template instructions that conflict with each other, are obsolete, or never mattered (candidates to remove).
Do not recommend banning or blacklisting seed content: staying close to the seed is the goal. Judge from the traces,
and cite trace ids and counts.

Return JSON only: {{"notes": "<markdown: concrete failure patterns with trace ids and counts, a verdict per variation
(keep / fix / drop), and specific template changes, including what to remove; most important first>"}}
</request>"""

# Judge for seed-expansion traces: sees the seed, the variation and the harness; gives a grounded verdict per check.
# Code verifies that every quoted excerpt really occurs in the generated trace or the seed.
JUDGE_CHECKS = ("only_declared_change", "dependencies_updated", "tool_contracts_respected", "observations_consistent",
                "case_consistent", "policy_followed", "task_success")

JUDGE_EXPANSION = """<harness_description>
{harness}
</harness_description>

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

Two kinds of checks. CONSTRUCTION: was the variation applied to the seed correctly? Compare the seed, the variation,
the case (when given; the generator's own description of the changed situation) and the generated observations.
BEHAVIOUR: is each assistant action justified by what the agent had seen at that point? The agent never sees the
variation or the case: for behaviour, only messages before the turn count as evidence.

For each check, give a verdict: "pass", "fail", "cannot_determine" (the trajectory lacks what you need to decide) or
"not_applicable". Back it with evidence: the message numbers (GENERATED as N, SEED as "seed:N") with short excerpts
copied exactly from those messages, then a one-sentence reason. When the problem is something missing (an
unsupported claim), quote the claim and say what support is missing.

Construction checks:
- only_declared_change: everything that differs from the seed is what the variation's "change" asks for, or a
  consequence covered by its "dependencies"; nothing else (site, platform, element, story) drifted. When the variation
  combines several changes, each is applied and none removes evidence another needs. A case cannot authorize a change
  the variation does not ask for.
- dependencies_updated: every later call and tool result is consistent with the change; no stale seed observation
  the change invalidated. The final answer follows the generated observations; keeping the seed's answer is valid
  when those observations still support it.
- tool_contracts_respected: every tool result is something that tool can return per the harness description (shape,
  content, no verdict the tool cannot know).
- observations_consistent: tool results agree with each other and with the failure context.
- case_consistent: the observations express the case, and the case stays within the variation. "not_applicable" when
  no case is given.
Behaviour checks:
- policy_followed: the agent follows the system policy.
- task_success: the final answer is correct for the generated evidence, using only what the conversation shows. The
  system policy and the established harness rules decide what is correct; the variation's "correct_when" is a
  hypothesis to check against them, never a rule that overrides them.
Then judge every checklist item the same way.

Turns: one entry for EVERY assistant message in GENERATED, in order: its number, "justified" or "unjustified" (or
"cannot_determine"), "relies_on": the numbers of the earlier messages that justify it (use "system" when the system
policy alone justifies it), and a short reason. A call or answer that relies on something shown only later, or only
in the case or variation, is unjustified.

Return JSON only:
{{"checks": {{"<check name>": {{"verdict": "pass", "evidence": [{{"message": "3", "excerpt": "..."}}], "reason": "..."}}}},
  "checklist": [{{"item": "<exact item text>", "verdict": "...", "evidence": [...], "reason": "..."}}],
  "turns": [{{"message": 1, "verdict": "justified", "relies_on": [0], "reason": "..."}}],
  "hallucinations": ["<claims no source supports>"], "notes": "..."}}
</request>"""


# Blind answer check: a separate call answers from the observations alone (no seed, variation, case or judge verdict).
# Agreement on the contract's categorical fields corroborates the generated answer; disagreement marks it unresolved.
BLIND_ANSWER = """<system_prompt>
{system}
</system_prompt>

<conversation note="everything the agent has observed; its final answer is withheld">
{transcript}
</conversation>

<request>
You are the agent described in the system prompt. Using only the conversation above, give the final answer the system
prompt's output contract requires, as that JSON object. If the observations support none of the answers the contract
allows, return {{"insufficient_evidence": true, "reason": "<one sentence>"}} instead.
</request>"""
