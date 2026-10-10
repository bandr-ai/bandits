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
