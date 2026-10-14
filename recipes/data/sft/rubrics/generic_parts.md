# Trace rubric

<!-- Generic parts of a judge rubric (A: evidence integrity, C: minor issues, labels). Domain-agnostic: the same text for
     every agent. The domain part (B) is induced from seeds by rubric_loop.py and inserted below, under Step 2. -->

## Task

You check one trace of an agent: a model that answers a question using tools and ends with a final answer. Decide two things:
1. **Evidence integrity:** are the tool results coherent with each other, with the task context and with what each tool can return?
2. **Answer correctness:** given that evidence, is the final answer right under the agent's own rules?

The agent's system prompt (given to you) is the authority on its question, its allowed answers and its rules. When a SEED is given, the trace is a synthetic variation of that real trace: the seed shows the original situation, it is not an answer key. When a DECLARED CHANGE is given, it is the one thing the variation was meant to change.

## Evidence rules

- Use tool results and the task context only. Ignore the agent's own narration and confidence.
- A result marked unavailable or not captured means unknown, never "no". An empty result from a search that did run is a miss within that search only.
- Repeated reads of the same input are one observation, not independent proof.
- A failure or error message in the task context is why the agent runs; it is not evidence about the answer.
- A variation may make evidence available that the seed lacked. A context or evidence list that was not updated to match is a minor issue, not a contradiction.

## Step 1: evidence integrity (any failure = bad)

| Check | Fails when |
|---|---|
| I1 Consistent results | Two results disagree on a fact the answer depends on, with nothing reconciling them. |
| I2 Possible results only | A result shows something its tool cannot return per the tool definitions and the other traces: data from a source that is unavailable elsewhere, names or values that appear nowhere, a verdict inside a raw tool result. |
| I3 Decisive evidence kept (synthetic) | The variation removed or changed the observation that established the situation, so the answer rests on less than it claims. |
| I4 Same situation as the seed (synthetic) | The trace replaces the seed's situation with a different one beyond what the declared change asks for. The declared change itself is not drift. |

## Step 2: answer correctness

Grade whether the answer is the right KIND for what the evidence shows: accept an answer whose kind fits the situation even when the logs cannot prove every detail of its specific cause; reject an answer whose kind does not fit, or one that breaks a hard rule below.

{domain}

## Step 3: label

- **bad**: any integrity check fails, or the answer is incorrect under Step 2.
- **good**: integrity holds and the answer matches the agent's rules for this situation.
- **unclear**: the situation itself cannot be decided from the evidence.

Report minor issues separately without failing the trace: a missing optional field or argument, an evidence list not updated, quoting or formatting slips in arguments, a detail sentence that slightly overstates.
