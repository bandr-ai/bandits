# Plan: issue #14 — free-text, LLM-interpreted verifier interview

Six stages, each landing green with its own tests. Stages 1-2 are the core and
are independently shippable; 3-6 build on them in order.

---

## Stage 0 — facts this plan is built on

- `Completion = Callable[[str, str, float], str]` (`judge.py:207`) is the
  existing LLM seam. `judge_trace` takes it as a parameter, so tests inject a
  fake and never hit the network. The interpreter reuses this type exactly.
- `DerivedStore.write` is content-addressed and raises `ArtifactConflict` on a
  same-id different-payload write (`store.py:140-156`). Every per-answer save
  therefore mints a *new* interview id.
- `Contract` is frozen with `extra="forbid"`; `.replace()` re-runs validators
  (`traces.py:17-29`). No in-place mutation anywhere.
- `VerifierInterview.validate_progress` currently enforces
  `next_question_index == len(answers)` and answers matching questions in
  order — a question-indexed model that the decision model replaces.
- The CLI interview loops in memory and saves once at the end
  (`cli.py:540-545`). Resume does not exist today.
- `_show_draft_run` (`cli.py:~500-534`) already renders per-family
  disagreements; the summary work reshapes this per-check.
- **DSPy is not used here.** It is an optional extra (`uv sync --extra audit`),
  deliberately kept out of core so the pipeline holds three runtime
  dependencies and CI needs no model or REPL sandbox (`pyproject.toml`). The
  interview is core pipeline, and `dspy.RLM` solves a context-window problem
  (#15: dozens of traces per family) that one reply about one check does not
  have. What *is* borrowed from `analyze/audit.py`: the injected-predictor
  seam, `resolve_api_key`, and the `_clean_ids` tolerance rule below.

---

## Stage 1 — the decision model

**Why first:** everything else writes into these types. Nothing can be built
against `InterviewQuestion`, which is field-oriented
(`Literal["expected","blind_spots","gaming_hypotheses"]`) and cannot express a
decision.

New contracts in `bandits/verify/models.py`:

- `InterviewDecision(str, Enum)`: `ACCEPT`, `REJECT`, `REVISE`, `COMBINE`.
- `Interpretation` — the model's structured read of one reply:
  `decision`, `rationale`, `revised_expected: Any = None`,
  `revised_operator: CheckOperator | None = None`,
  `combine_with: str | None = None`,
  `blind_spots: tuple[str, ...] = ()`,
  `gaming_hypotheses: tuple[str, ...] = ()`.
  Validators: `revised_*` only with `REVISE`; `combine_with` required for and
  exclusive to `COMBINE`.
- `CheckReview` — one completed review of one check:
  `verifier_id`, `check_id`, `reply` (raw human text),
  `authoritative: bool`, `authoritative_why: str`,
  `interpretation: Interpretation | None` (None when the model failed and the
  human chose manually), `decision` (the confirmed decision — the human's, not
  the model's), `model: str`, `prompt: str`, `response: str`,
  `failure: str | None`, `superseded_by: str | None = None`.
- `VerifierInterview` gains `reviews: tuple[CheckReview, ...]` and
  `pending: tuple[tuple[str, str], ...]` (verifier_id, check_id still to
  review). Keep `questions`/`answers` fields untouched so existing artifacts
  still load; the new flow simply leaves them empty.

**Tests** (`models_test.py`): each validator rejects its bad case —
`combine_with` on a non-combine, `revised_expected` on an accept, a combine
with no target.

---

## Stage 2 — the interpreter

New module `bandits/verify/interpret.py`.

- `INTERPRETER_MODEL` default, mirroring `judge.py::DEFAULT_MODEL`.
- `render_interview_prompt(check, spec, summary, reply, prior_reviews) -> str`.

  Sends everything the human is looking at when they answer, because the
  model's job is to map what they meant and it cannot do that against a
  thinner view:

  - check definition and `evidence_kind`;
  - pass/fail/unscorable counts and any `Disagreement` for the check;
  - **prior validation results** (round 2+): per-split label agreement and any
    passing gameability result. Strictly stronger discrimination signal than
    the fit counts, and the thing a round-2 reply is usually responding to;
  - **prior decisions on this check**: what was decided in earlier rounds, the
    rationale, and whether the check has been revised since. Without it the
    model cannot tell a considered "still fine" about a known-weak check from
    a lazy repeat of round one;
  - for a reply naming another check (`combine`), that check's definition and
    current decision, so the target can be validated;
  - the reply.

  **Not** sent: trace ids (opaque, human-only), and other checks' decision
  history (context bloat with no bearing on this reply).

  Trace-derived content is fenced and labelled as data.

  The failure this prevents: the human sees "51% agreement, forgery passed"
  and types "still fine, that forgery isn't realistic here" — a considered
  accept of a known-weak check. A model blind to both numbers records a plain
  accept whose rationale never mentions the weakness, and the audit trail this
  issue exists to build is worthless at exactly the point it mattered.
- `interpret_reply(..., predict: Interpreter) -> Interpretation` — parses
  strict JSON from the reply.
- `Interpreter = Callable[..., str]` — the injected seam, following
  `audit.py::build_predictor` rather than being a second transport. A default
  built on `judge.py::fireworks_completion` + `resolve_api_key`; tests inject a
  fake and never touch the network.
- `InterpretationFailure` exception carrying `kind` and the raw response.

Failure handling, exactly as the issue specifies:

| kind | behavior |
|---|---|
| transport/timeout | retry once, then raise for manual entry |
| unparseable JSON / missing field | no retry, raise |
| operator outside `CheckOperator` | no retry, raise with what was proposed |
| `combine_with` not a real check id | **not** a hard failure — drop the reference, keep the rest of the interpretation, report the drop |
| `revised_expected` unparseable | not a failure — resolve via `_parse_expected` (JSON first, raw string otherwise) and surface for confirmation |

The dropped-reference rule follows `audit.py::_clean_ids`: a model that
hallucinates an id "would otherwise fail the contract's validator and lose the
whole audit — including the parts it got right". A reply whose decision is
right but whose combine target is fumbled should not discard the
interpretation; the human is shown the decision, told the named target did not
resolve, and picks a real one or falls back.

**Tests** (`interpret_test.py`), all with a fake `Completion`: each decision
type parses; one retry then success; two failures raise; bad JSON raises with
no retry; an unknown combine target is dropped with the rest of the
interpretation preserved and the drop reported; unparseable revised value
resolves.
Prompt-content tests: validation agreement and gameability appear in the
rendered prompt when a validation is supplied and are absent when it is not; a
prior decision on the check appears; trace ids never appear.

---

## Stage 3 — applying a decision

New functions in `interview.py`, replacing the `answer_question` path for the
new flow (the old function stays for the legacy question model).

- `start_review(draft, source_draft_id, validation=None) -> VerifierInterview`
  — populates `pending` with every (verifier, check).
- `apply_decision(interview, review: CheckReview) -> VerifierInterview` —
  dispatches on `review.decision`.

Per decision:

- **accept** — record the review, drop from `pending`. Spec untouched.
- **reject** — record with rationale, mark the spec `VerifierStatus.REJECTED`.
- **revise** — mint a new verifier id via a new
  `revised_verifier_id(spec, check)` helper reusing `draft.py::_spec_id`'s
  hashing over (family, claim, expected). Apply the new expected/operator.
  Clear `supporting_evidence_ids` to `()` and record the check as pending
  re-evaluation. Set `provenance="human"`. Record the interview id + review.
- **combine** — see Stage 4.

Blind spots / gaming hypotheses extracted by the interpreter are appended to
the spec's tuples.

**Tests** (`interview_test.py`): accept leaves the spec identical; reject sets
status and keeps rationale; revise mints a different id, clears evidence, sets
provenance human; extracted hypotheses land on the spec.

---

## Stage 4 — combine

The largest piece, and the reason it gets its own stage.

`combine_checks(interview, source_ids: tuple[str, str], review) -> VerifierInterview`:

- Mints a new verifier identity (same rule as revision).
- `checks` = concatenation of both specs' checks, each keeping its own weight
  (no renormalizing — `Result` aggregation already weights).
- `unknown_when` = union. Fail-closed: unknown if either input is unknown,
  consistent with `Result.unknown_is_consistent`.
- `blind_spots`, `gaming_hypotheses` = union.
- `evidence_kind` per check is untouched; the combined spec's standing is read
  through the existing `weakest_evidence_kind` property — nothing new to store.
- `supporting_evidence_ids` cleared on every check, pending re-evaluation.
- `inputs` = union.
- Records `source_verifier_ids` — **new field on `VerifierSpec`**,
  `tuple[str, ...] = ()`, so a combination says what it came from.
- Both source specs are removed from the draft and replaced by the combined
  one.

**Reaching backwards.** If the combine target was already reviewed earlier in
this interview, that `CheckReview` gets `superseded_by` set to the new
review's id, and the combined result is confirmed by the human like any other
decision. A prior decision is never silently overwritten.

**Tests**: two checks combine into one new id; `unknown_when` is the union;
weights survive; evidence cleared on both; `source_verifier_ids` recorded;
combining with an already-accepted check marks that review superseded;
`VerifierDraft.validate_references` still passes (no duplicate ids left
behind).

---

## Stage 5 — the evidence summary

New `build_check_summary(check, spec, run: DraftRun, validation: Validation | None)`
in `run.py` (it already owns `DraftRun` and `Disagreement`).

Returns a `CheckSummary` contract: `passed`, `failed`, `unscorable`,
`example_trace_ids`, `evidence_kind`, `evidence_source`, `blind_spots`,
`gaming_hypotheses`, and — when a validation is supplied —
`agreements: tuple[Agreement, ...]` and `gameability: tuple[GameabilityResult, ...]`
filtered to this verifier.

This is the "round two" half: with no validation it is the round-one summary;
with one it carries what validation measured. The human sees all fields. The
interpreter prompt (Stage 2) receives all of it **except** `example_trace_ids`
— including the agreements and gameability, which are the point of running a
second round at all.

**Tests** (`run_test.py`): counts match the draft run; disagreements surface;
with a validation the agreements and gameability for that verifier appear and
another verifier's do not.

---

## Stage 6 — CLI, resume, and gaming probes

**Per-answer persistence.** Because ids are content-hashed, each save mints a
new id. `save_interview` is called after every decision and the CLI prints the
latest id. Resume takes the id of the last save.

- `interview-verifier <draft-id> [--validation <id>] [--resume <interview-id>]`
- Per check: render the summary, prompt `"what do you think?"`, prompt the
  authoritativeness question, call the interpreter, **display the structured
  interpretation and require confirmation** (y/n; n falls through to manual
  decision entry), then apply and save.
- On `InterpretationFailure`: show the raw reply + the failure, offer the
  four decisions directly. Answer preserved, interview resumable at the same
  check.

**Gaming probe wiring.** For each extracted gaming hypothesis, build a
single-check `VerifierSpec` from the check under review and pass it to the
existing `probe_gameability`. If `_attack()` returns `None` for that operator,
say so plainly — the hypothesis is recorded but no template exists to test it
(the issue's stated limit). Show any passing attack back to the human before
they confirm.

**Tests** (`cli_test.py`, typer runner + fake `Completion`): a full accept
run; a reject; a revise; a resume from a partial interview id; a failed
interpretation falling back to manual entry; an affirmative reply carrying a
combine ("fine, but combine with X") producing a combine, not an accept; a
round-two interview whose prompt carries the prior round's decision and
validation numbers.

---

## Order and shippability

1 → 2 → 3 are the minimum coherent change (accept/reject/revise via LLM, no
combine, no summary). 4, 5, 6 each land independently on top. Stage 6 is what
makes it usable; stages 1-5 are testable without a terminal.

**Not in this plan** (issue says out of scope): prompt/model digest pinning,
length caps, attack synthesis for unmatched templates, label-filtered
drafting, loop stop conditions.
