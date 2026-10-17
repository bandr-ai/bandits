"""Opt-in scenario synthesis. No domain labels or evidence rules live here."""

COVERAGE = """Derive a coverage catalogue ONLY from the supplied correct seeds and their inferred contracts.
There are no student predictions, external labels, application code or environment. Never describe a
coverage gap as a measured student failure. Identify decisions, evidence routes or explicitly permitted
outcomes that are rare/missing in the examples. Prefer substantive gaps over superficial name changes.
Use only demonstrated result shapes or explicit tool contracts. Unknown branches remain unknown.
Return {"capabilities":[{"name":"short_name","capability":"decision/evidence condition to exercise",
"basis":"seed, or words copied verbatim from the Explicit/Observed contracts or a seed system prompt",
"seed_ids":["compatible supplied seed ids"]}]}. basis is checked against those texts: copy, do not paraphrase or prefix.
Each capability needs at least one compatible reference. Do not invent ids, rules or output enums.
An empty list is allowed if no defensible gap exists. This catalogue guides planning, not final labels.
"""

PLAN = """Design ONE candidate training situation from the supplied agent contract and recorded examples.
The examples demonstrate formats and tool semantics; they are not a catalog of situations to copy.
Read the requested dimensions. A targeted job should exercise the supplied weak capability, but the
answer must follow from evidence, never from the request to target it. A coverage job has no forced label.
If job.outcome is set (control job), the final answer must have exactly those field values, taken from a real
example: design a different situation whose evidence genuinely leads to that outcome.
If a capability is supplied, exercise that seed-derived evidence/decision gap without forcing an answer.
For controlled novelty, preserve the reference situation and change one condition, updating all dependencies.
For scenario novelty, change the workflow/page/target/history, not merely names or URLs. Keep the same
agent task, system instructions, tool semantics, and output contract. Invent plausible fictional content.
Use only demonstrated tool-result branches or shapes explicitly specified by the supplied contracts.
Do not invent a successful branch just because a tool name suggests it exists. If the capability cannot
be demonstrated with supported observations, return {"unsupported": true, "reason": "..."}.
If explicit contract rules conflict or do not resolve the required label, return unsupported; do not invent policy.
For sparse evidence, keep sufficient evidence to justify the answer or use the contract's uncertain outcome.
Distractors must be observations about different times/scopes, never unresolved impossible facts.
Do not equate complexity with length. Vary evidence routes and failure mechanisms across jobs.
tool_result_rules were learned from the real examples and are enforced by code: a trace breaking any is discarded.
already_planned lists situations made earlier in this run for the same agent: choose a clearly different
situation, decision and evidence route; never repeat one of them with renamed entities.
Declare the episode state FIRST, then derive every observation from it (observations never invent state):
episode_state = {"context": the exact list-valued fields of the first user input (for example which evidence
sources exist), shaped like the references; "facts": short statements true in this episode}. A tool reading a
source not in the context returns that tool's unavailable shape from the references or contracts. Facts must
agree with each other (an item absent from a captured source cannot also be present in it).
Vary evidence routes: when the contract offers several tools, the decisive evidence comes from at least two
independent sources, and the answer never rests on one observation whose text simply states the conclusion.
Observations report what is seen or counted, not the diagnosis.
Return JSON with nonempty strings: situation, workflow, decisive_evidence, dependencies, novelty_reason, episode_state,
and a variation object with name, change, preserve, dependencies, correct_when, teaches, basis.
The variation describes evidence conditions, not an instruction to emit a particular label.
Also return evidence_sources as a list of strings and unsupported=false.
If pair=true, describe ONE causal fact to flip and how both cases remain internally consistent;
put that description in counterfactual_flip. No human labeling is available.
"""

GENERATE = """Write a new candidate training conversation for this agent using the supplied scenario card.
Use references for style, answer conventions, and tool-result formats, not for copying their situation.
Preserve the supplied system prompt and tool definitions (code adds them; do not output replacements).
Names and content may be fictional, but observations must be possible outputs of their calls.
All user input facts, history, timestamps, tool arguments, results and assistant decisions must agree.
Each assistant action must use only information already visible at that turn. A screenshot observation
cannot establish invisible process or network state. Unavailable evidence establishes no facts.
The final assistant content follows THIS system prompt's own output contract. When JSON is required,
return a bare JSON object; when free text is required, preserve that format instead.
If job.outcome is set, the final answer keeps exactly those field values. Some agents have no conclusion field: do not add one. Confidence reflects evidence; never force a mean.
Obey tool_result_rules exactly (learned from real examples; any violation discards the trace).
Follow card.episode_state exactly: copy its context lists into the first user input unchanged; every tool
result must agree with its facts; a call on a source absent from the context returns that tool's unavailable
shape. Use only result keys and branches shown in the references or contracts; never add new fields.
Use the supplied stop_nudge requirement exactly: include the reference stop instruction when true,
otherwise omit it. Do not expose the scenario card, target label, generation dimensions, or answer key
in user inputs or tool outputs. Tool observations describe facts, not diagnostic root-cause enums.
Keep tool-result wrappers, nested JSON string conventions and shapes from contracts/examples.
For pair=true, output two complete conversations sharing a scenario. Flip one decisive causal fact,
update its dependent observations/actions, and derive different final labels. No unrelated edits.
Return {"traces": [{"messages": [user, assistant with tool_calls, tool, ..., assistant final]}]}.
Use the existing canonical call format: {"id":"call_1","name":"tool_name","arguments":{...}}.
Return one trace normally, two for a pair. If impossible, return {"unsupported":true,"reason":"..."}.
"""

AUDIT = """Check whether these candidate conversations actually implement the requested generation dimensions.
The reference is a style/contract reference, not an answer key. Apply explicit system rules; do not invent policy.
Return {"supported": boolean, "novelty_ok": boolean, "dimensions_ok": boolean,
"pair_ok": boolean, "reason": "specific evidence with message numbers"}.
supported means every intended target answer follows from visible evidence and the explicit contract.
For targeting=coverage there is no required target label (job.outcome, if set, is fixed); still check answer support.
novelty_ok: controlled keeps the original situation and applies its declared change; scenario changes
workflow/page/target/history structurally, not just names. Novelty is relative to supplied references,
not proof of population-level OOD generalization.
dimensions_ok: verify requested evidence availability, temporal reasoning and composition from the trace.
When a capability is supplied in the job, check it is actually exercised by visible evidence and actions.
pair_ok: true for a singleton; for a pair, one causal fact flips, dependent facts agree, and conclusions
or root causes change under the contract. Surface edits and confidence changes alone are insufficient.
False or uncertain on any requirement rejects the candidate. Do not repair or relabel silently.
"""
