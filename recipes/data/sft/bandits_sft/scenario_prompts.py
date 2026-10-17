"""Scenario recipe prompts. Generic: no domain, tool, field or label names live here.

Everything specific to an agent reaches these prompts as data derived from its seeds: the system prompt, tool
definitions, observed contracts (what each tool reads and returns, the output format, unknown branches),
tool-result rules learned by code, and reference traces. Each prompt states its purpose, what matters most, the
reason for each rule, and the exact output.
"""

COVERAGE = """Derive a coverage catalogue ONLY from the supplied correct seeds and their inferred contracts.

Purpose: list decisions, evidence routes or permitted outcomes that the seeds show rarely or not at all, so new
training situations can exercise them. These are coverage gaps in the examples, not measured model failures, so
never describe them as failures. There are no labels, predictions, application code or environment.

What matters most: every capability must be demonstrable with what the seeds actually show. A capability that
needs a tool the agent lacks, a tool result never seen, or an answer value the output contract does not allow
cannot become a valid training example, and code rejects it.

Rules:
- Prefer substantive gaps (a different decision or evidence route) over surface changes such as new names.
- Use only result shapes the seeds demonstrate or the contracts state explicitly. A branch listed as Unknown
  stays unknown: do not build a capability on it.
- basis: write "seed", or copy words verbatim from the Explicit/Observed contracts or a seed system prompt.
  Code checks the copy, so do not paraphrase or prefix it.
- seed_ids: supplied seed ids whose agent has every tool the capability needs. Never invent ids.
- needs: what the capability depends on, so code can check it is demonstrable:
  tools = tool names it calls; tools_with_data = tools whose result must carry data (not an empty result);
  answer = final-answer field values it expects, copied exactly from the seeds or contracts ({} for none).
- An empty list is fine when no defensible gap exists. The catalogue guides planning; it never sets final labels.

Return {"capabilities":[{"name":"short_name","capability":"decision/evidence condition to exercise",
"basis":"...","seed_ids":["..."],"needs":{"tools":[],"tools_with_data":[],"answer":{}}}]}.
"""

PLAN = """Design ONE candidate training situation from the supplied agent contract and recorded examples.

Purpose: a new situation for the same agent (same system prompt, tools and output format) whose correct answer
follows from evidence the agent gathers. The references show formats and what each tool can return; they are not a
list of situations to copy.

What matters most, in order:
1. Every observation must be something the recorded tools could return. tool_result_rules were learned from the
   real examples and are enforced by code: a trace that breaks one is discarded. Use only result branches and
   shapes shown in the references or stated in the contracts; when a tool's success shape is unknown, that tool
   only returns its unavailable/empty shape.
2. The answer follows from the evidence, never from the job. If job.outcome is set (a broad job), design a
   different situation whose evidence genuinely leads to exactly those answer field values. If a capability is
   supplied (a gap job), exercise it without forcing an answer. If the evidence cannot support the job, or explicit
   contract rules conflict or leave the answer undecided, return {"unsupported": true, "reason": "..."}: an
   honest refusal costs less than a wrong example.
3. Novelty: change what the agent faces (the task's target, its inputs, the evidence, or the history), not only
   names or identifiers. already_planned lists situations made earlier in this run for the same agent: choose a
   clearly different situation, decision and evidence route, never a renamed repeat.

How to build it:
- Declare the episode state FIRST and derive every observation from it. episode_state = {"context": the exact
  list-valued fields of the first user input, shaped like the references ({} when the input has none); "facts":
  short statements true in this episode}. A tool reading a source absent from the context returns its
  unavailable/empty shape. Facts must agree with each other.
- Each tool reports only what it can observe (the contracts and references show what each tool reads). A fact no
  tool can see stays unknown to the agent and cannot decide the answer.
- When the contract offers several tools, let the decisive evidence come from at least two independent sources,
  so the answer never rests on one observation that simply states it. Observations report what is seen or
  counted; the agent draws the conclusion.
- Sparse evidence: keep enough to justify the answer, or use the contract's own uncertain outcome.
  Distractors: observations about other times or scopes, never unresolved contradictions.
- Apply job.dimensions (evidence availability, temporal reasoning, composition) so each is visible in the trace;
  the final check verifies them.
- Complexity is not length. Vary evidence routes and the reasons behind outcomes across jobs.

Return JSON with nonempty strings situation, workflow, decisive_evidence, dependencies, novelty_reason; the object
episode_state; a variation object with nonempty strings name, change, preserve, dependencies, correct_when,
teaches, basis (basis = "seed" or words copied from the contracts or system prompt); evidence_sources as a list of
strings; and unsupported=false. The variation describes evidence conditions, not an instruction to emit a label.
If pair=true, also describe in counterfactual_flip ONE causal fact to flip and how both cases stay consistent.
"""

GENERATE = """Write a new candidate training conversation for this agent using the supplied scenario card.

Purpose: one realistic, internally consistent conversation that teaches the card's situation. Code adds the
system prompt and tool definitions; output only the messages.

What matters most, in order:
1. Follow card.episode_state exactly. Copy its context lists into the first user input unchanged. Every tool result
   agrees with its facts; a call on a source absent from the context returns that tool's unavailable/empty shape.
2. Obey tool_result_rules exactly. They were learned from the real examples and any violation discards the trace.
   Use only result keys and branches shown in the references or contracts, and keep their wrappers, nested JSON
   string conventions and value formats. Never add a field.
3. Consistency: user facts, history, timestamps, tool arguments, results and the agent's decisions all agree. Each
   assistant step uses only information already visible at that point, and each tool reports only what it can
   observe. Evidence that is unavailable establishes nothing.
4. The final assistant message follows THIS system prompt's output format exactly: a bare JSON object when JSON is
   required, otherwise the format the system prompt asks for. Keep exactly the answer fields it asks for and add
   none. If job.outcome is set, those answer fields keep exactly those values. Any score or confidence the format
   asks for reflects the evidence actually seen.

Also:
- Use the references for style, answer conventions and tool-result formats, not for their situation.
- Names and content may be fictional; each result must be a possible output of its call.
- Include the reference wrap-up instruction only when stop_nudge is true.
- Keep the card, target, generation dimensions and answer out of user inputs and tool results: tool results report
  what was seen or counted, never the final answer's label values or the conclusion itself.
- For pair=true, write two complete conversations that share a situation, flip the one causal fact from the card,
  update everything that depends on it, and reach different final answers. No unrelated edits.

Return {"traces": [{"messages": [user, assistant with tool_calls, tool, ..., assistant final]}]} with calls in the
canonical form {"id":"call_1","name":"tool_name","arguments":{...}}: one trace normally, two for a pair.
If the card cannot be written consistently, return {"unsupported":true,"reason":"..."}.
"""

AUDIT = """Check whether these candidate conversations actually implement the requested generation dimensions.

Purpose: a final yes/no gate. The references are a style and contract reference, not an answer key. Apply the
explicit system rules; do not invent policy, and do not repair or relabel anything.

Answer each check with true only when the trace clearly passes; false or uncertain rejects the candidate.
- supported: every answer the conversation gives follows from visible evidence and the explicit contract.
  For targeting=coverage there is no required label (job.outcome, if set, is fixed); still check support.
- novelty_ok: the situation differs from the references in structure (target, inputs, evidence or history), not
  just names. Novelty is relative to the supplied references only.
- dimensions_ok: the requested evidence availability, temporal reasoning and composition are visible in the trace;
  when the job names a capability, visible evidence and actions actually exercise it.
- pair_ok: true for a single conversation. For a pair: one causal fact flips, dependent facts agree, and the final
  answer's label fields change under the contract. Surface edits or confidence changes alone do not count.

Return {"supported": boolean, "novelty_ok": boolean, "dimensions_ok": boolean, "pair_ok": boolean,
"reason": "specific evidence with message numbers"}.
"""
