"""Iterative RLM discovery of task families from raw user requests.

The loop, and why it is a loop. A single call over a whole corpus either
truncates it or reads it once with no chance to revise; both produce a taxonomy
whose earliest guesses are never tested against what came later. This instead
shows the model a chunk at a time and lets it change its mind, then requires the
changes to stop before the result counts as finished.

What "stop" means is the load-bearing decision here. A loop that ran until the
model stopped objecting would be optimising for the model's agreement, so
convergence is defined mechanically over recorded operations — two consecutive
complete sweeps that create nothing, split and merge nothing, revise nothing
materially, and move under 2% of assignments — and every other way of stopping
is a budget limit that yields an explicitly *incomplete* artifact. A taxonomy
that stopped because the money ran out must never read as one that stopped
changing, so :class:`~bandits.analyze.rlm_models.StopReason` travels with the
run for the rest of its life.

Chunk boundaries are never family boundaries. Each chunk deliberately mixes
unseen traces with ambiguous ones, traces touched by recent changes, and random
previously-assigned traces, so that a family cannot be an artifact of which
twenty traces happened to be read together.

Nothing here is deterministic — the root model writes its own code each run —
which is why the plan calls for five independent runs and compares them by trace
co-assignment rather than by the names they generate.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import Field

from bandits import ledger, providers
from bandits.analyze.rlm_account import (
    AccountIdentity,
    Completion,
    ProposedAccount,
    RunAccount,
    coerce_account,
    family_row,
    latest_by_run,
    serialize_candidate,
    validate_account,
)
from bandits.analyze.rlm_budget import BudgetExhausted
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_models import (
    VIEW_PREAMBLES,
    Budget,
    ChunkResult,
    FamilyContract,
    Operation,
    PassResult,
    ProposedContract,
    ProposedOperation,
    RLMClusteringRun,
    StopReason,
    TaxonomyOperation,
    TraceView,
    coverage_of,
)
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract

DEFAULT_MODEL = providers.default_model(providers.RLM_FIREWORKS_DEFAULT)
"""Matches the family audit's default, so one credential covers both paths."""

DEFAULT_CHUNK_SIZE = 20
DEFAULT_SEED = 42
PROMPT_VERSION = 9

CLEAN_SWEEPS_TO_FREEZE = 2
"""Consecutive clean sweeps required before a taxonomy may freeze.

Two rather than one: a single quiet sweep is as easily explained by an
unrepresentative chunk as by convergence, and the cost of one more pass is far
below the cost of freezing a taxonomy that was still moving.
"""

MAX_CONTRACT_REPAIRS = 1
"""How many times one chunk may be asked to fix contracts it got wrong.

Capped at one rather than looped. Returning the validation error to the model is
the standard repair for structured extraction, and it is right here: a contract
silently dropped costs the whole chunk's work, while a contract corrected costs
one call. But a model that fails validation once will often fail it the same way
again, and an uncapped loop would spend a chunk's entire call budget arguing
with itself. One retry, then the drop is recorded with its evidence.
"""

MAX_ASSIGNMENT_CHURN = 0.02
"""Fraction of provisional assignments that may move in a sweep still called clean.

Not zero. A model rereading a borderline trace and placing it differently is
noise in the classifier, not evidence that the taxonomy is unsettled, and
demanding exact repetition would make convergence depend on sampling luck.
"""


_INSTRUCTION_HEAD = """You are discovering reusable task families from recorded runs.

{view}

The variable `chunk` is a list of dicts, each with keys: trace_id, the run as you \
may read it, and status (one of "unseen", "ambiguous", "uncovered", "assigned", \
"affected"). The variable `taxonomy` is the list of family contracts you have \
built so far, possibly empty.

Parse `chunk` and `taxonomy` programmatically. Do not print either variable in \
full. Print only trace ids, short excerpts, contract ids, and outcome shapes \
needed for the decision in front of you. Use a sub-LLM call for semantic \
comparison of specific traces rather than reasoning over the whole chunk again \
in text.

A family is defined by the work requested and its required completion \
conditions — not by the particular values in one request, and not by what \
happened. Two traces belong to the same family when ONE verifier template, \
parameterized with each trace's requested values, could evaluate both.

Treat request-specific values as parameters unless they materially change how \
completion is checked: identifiers, names, inputs, targets, environments, \
platforms and similar values. Different objects, parameters, internal steps or \
observed failures do not by themselves make different families. A successful \
and a failed attempt at the same work belong together.

One family is a correct answer when the requested work is equivalent across \
runs; do not manufacture families from different inputs, scenarios or reported \
findings to increase the count. Create a separate family only when the evidence \
shows materially different requested work or completion conditions. When the \
requested work is not supported by evidence, mark the trace ambiguous or \
uncovered rather than guessing a family. Field names or values that happen to \
appear in a run's data do not establish what was requested.

Before creating a contract:
1. Check whether an existing contract already fits after substituting parameters.
2. If its core outcome is correct but its wording is too narrow, REVISE it.
3. CREATE only when completion requires materially different verification.
4. Mark uncovered only when neither reuse nor safe revision works.

REVISE rules:
- Preserve the existing contract_id exactly.
- Increase revision by exactly one.
- Broaden only enough to cover the old members and the new evidence.
- Do not create a differently named sibling for a parameter variation.
- Do not remove an old required outcome merely to admit an incompatible trace.

Naming rules:
- Use a short human-readable imperative phrase with spaces.
- Never include trace ids, people, identifiers, dates, amounts, or other \
request-specific values in the name.

Contract wording:
- Definitions describe reusable work using parameter language ("the requested \
target", "the stated input").
- required_outcome_shape states invariant checks that refer to the trace's \
requested values.
- Do not copy concrete values from a supporting trace unless that value \
fundamentally changes the kind of work.

For every contract you propose or keep, state:
- name: a short imperative task-family name, in prose.
- definition: what requested work belongs here, in parameter language.
- inclusion_rules / exclusion_rules: what admits and excludes a member.
- required_outcome_shape: what a verifier must establish for EVERY member.
- supporting_trace_ids / counterexample_trace_ids: evidence from what you read.

Operation rules. Valid operations: KEEP, CREATE, REVISE, SPLIT, MERGE, \
MARK_AMBIGUOUS, MARK_UNCOVERED.
- Emit operations only for contracts actually created, revised, split, merged, \
or reconsidered because of the current chunk.
- Do not emit KEEP for unrelated contracts; omitted contracts remain unchanged.
- KEEP means the taxonomy contract remains unchanged; its subject is always a \
contract, never something in a run.
- Every operation must name the affected contract_ids and motivating trace_ids.

For example, retaining contract c1 unchanged uses {{"operation": "KEEP", \
"contract_ids": ["c1"], "trace_ids": ["t1"], "rationale": "..."}}. Merging c1 \
and c2 into c3 uses contract_ids ["c1", "c2", "c3"], with the produced contract \
last.

For every trace in the chunk, assign one matching contract or explicitly mark it \
ambiguous or uncovered. After creating or revising a contract, reconsider any \
trace from this chunk that you previously left unplaced.

Assign every trace in this chunk, including ones you have seen before: re-reading \
an old trace against changed definitions is the point of the loop."""


_ACCOUNT_FAMILY_NOTE = """

In this mode each chunk row carries a run's ACCOUNT (its intent reading and \
execution milestones), produced earlier from the run's evidence; results and \
evaluator claims are deliberately absent. Group on intent.candidate_goal, \
intent.required_outcome, parameters and constraints. An intent with status \
"inferred" is an interpretation, not a declared request. To settle a disputed \
boundary you may read evidence with inspect_run(run_id) and get_evidence(run_id, \
ref) for runs in this chunk; that evidence withholds outcome-bearing fields, \
because families group requested work, not results. Do not reread runs that are \
not in dispute."""


_ACCOUNT_INSTRUCTION = """You are writing an evidence-backed ACCOUNT of one recorded run.

{view}

The variable `run_index` is a JSON object: run_id, total_events, \
candidate_instructions (refs, origins and short excerpts of where a request \
may have been recorded), distinct_system_prompts and limitations. Read the run \
with the two helpers, not by guessing:
- inspect_run(run_id, cursor=0, limit=20) -> dict: a bounded page of rows \
(candidate instructions, system prompts and limitations first, then chronological \
events with input/output/tool-call refs) and next_cursor (null at the end).
- get_evidence(run_id, ref, start=0, limit=4096) -> dict: exact text of a ref \
(max limit 8192); continue with next_start when it is not null.
Keep what you read in Python variables; print only short excerpts.

Answer three questions, citing refs returned by the helpers:
1. Intent: what work was this run invoked to perform, with what parameters, \
constraints and required completion conditions? status is "declared" only when a \
ref records the request as such (a declared task field or a recorded user turn); \
a parsed payload, an internal model prompt or a tentative clue supports \
"inferred" at most. Use "unknown" when evidence does not establish it — that is a \
valid answer. Keep the run-level job separate from internal subtask \
instructions, and from the scenario or failure described in its inputs.
2. Execution: the consequential actions, decisions and interactions, as \
milestones citing event span ids or refs. Not every event.
3. Result: what was observed and whether evidence supports completion \
(supported_complete), non-completion (supported_incomplete) or neither \
(unknown). Evidence marked analysis_claims "evaluator_assertion:..." is somebody's \
judgement of the run: put it in evaluator_claims, never adopt it as your \
assessment. A reported diagnosis is likewise a claim to assess, not automatically \
the truth. Fields named like "expected" or "outcome" are ordinary data: read them \
for what they say.

Record missing or conflicting evidence in limitations (kind \
"missing_requirement", "contradictory_evidence", "unavailable_source" or \
another short kind). List refs you know matter but did not read in \
unresolved_refs. Every ref you cite must come from inspect_run or get_evidence \
for this run_id; never invent one.

SUBMIT(account=...) with a dict matching the account fields. If SUBMIT reports a \
type error, fix exactly what it names and SUBMIT again; do not restart the \
investigation."""


def instruction_for(view: TraceView) -> str:
    """The mining prompt as this arm's miner actually receives it.

    Passed as the signature's instructions rather than as an input field. The
    two are not interchangeable: an input field becomes a REPL variable, and
    ``REPLVariable.from_value`` shows the root model a 1000-character peek —
    the first 500 characters and the last 500 — of any value larger than that.
    This prompt is 2.8k characters, so a third of it was visible and the
    schema, the outcome requirement and the worked examples all sat in the
    elided middle. The model could have printed the variable to read the rest;
    across 42 recorded calls it referenced it seven times and printed it in
    full none.

    Signature instructions are interpolated straight into the action prompt and
    never wrapped in a variable, so there is nothing to elide and nothing the
    model has to think to retrieve. In the RLM's own terms this is the query,
    which belongs in the token window, while the traces are the context, which
    belongs in the REPL.
    """
    return _INSTRUCTION_HEAD.format(view=VIEW_PREAMBLES[view])


def account_family_instruction(view: TraceView) -> str:
    """The family instruction when chunks carry accounts rather than raw runs."""
    return instruction_for(view) + _ACCOUNT_FAMILY_NOTE


def account_instruction(view: TraceView) -> str:
    return _ACCOUNT_INSTRUCTION.format(view=VIEW_PREAMBLES[view])


class MiningError(RuntimeError):
    """The miner could not be built or returned nothing usable."""


class _Predictor(Protocol):
    """The one call this module makes, so tests need no model and no sandbox."""

    def __call__(self, *, chunk: str, taxonomy: str, question: str) -> Any: ...


def prompt_digest(model: str) -> str:
    """Pins wording, model and version onto every run they produced."""
    payload = json.dumps(
        {
            "instruction": _INSTRUCTION_HEAD,
            "account_instruction": _ACCOUNT_INSTRUCTION,
            "account_family_note": _ACCOUNT_FAMILY_NOTE,
            "views": {v.value: text for v, text in VIEW_PREAMBLES.items()},
            "model": model,
            "version": PROMPT_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


DEFAULT_MAX_TOKENS = 24000
"""Per-call ceiling for the default Nemotron + ChatAdapter path.

Every recorded Nemotron completion above 12000 tokens was inspected rather
than classified from token counts alone. The set was mixed: several were
hallucinated multi-turn transcripts, but valid final actions and extraction
responses also reached 13845, 19179, 20395 and 22806 tokens. A 12000 default
would therefore cut off work known to complete successfully. 24000 leaves
measured headroom for those calls while remaining below the confirmed 32768
token runaway. This is still an experiment-specific default, so callers can
lower or raise it explicitly and the ledger records the value actually used.
"""


class GenerationSettings(Contract):
    """Every knob that changes what a mining invocation sends or may do.

    Recorded on the run and the session and hashed into account identity: two
    runs that differ here are different experiments, and resume refuses to mix
    them. Defaults preserve the previous behaviour except where noted.
    """

    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float | None = 0.0
    """0.0 preserves the previous default. Vendor-recommended sampling is passed
    explicitly, never substituted silently."""

    top_p: float | None = None
    reasoning_effort: str | int | None = None
    """Sent as LiteLLM's ``reasoning_effort``: a level (``none``, ``low``…) or, on
    providers that document it (Fireworks), a positive integer hard cap on
    reasoning tokens. Preflight checks it reaches the request body, not that the
    provider honours it for this model."""

    root_max_iterations: int = Field(default=25, ge=1)
    max_subcalls: int = Field(default=60, ge=1)
    """Per invocation: ``llm_query``/``llm_query_batched`` calls."""

    max_output_chars: int = Field(default=10_000, ge=100)
    """REPL output shown back to the root per observation. Display only."""

    inspect_page_chars: int = Field(default=6000, ge=1000, le=50_000)
    """Ceiling on one ``inspect_run`` page. Changes what the root reads per
    call, so it is part of the settings identity."""

    subcall_workers: int = Field(default=8, ge=1)
    """Concurrency of ``llm_query_batched``; 1 runs subcalls sequentially."""

    provider_retries: int = Field(default=0, ge=0)
    """LiteLLM transport retries beneath one admitted, recorded call. Zero by
    default (a reviewed change from DSPy's 3): a retry is a provider attempt the
    call count, the recording and the cost figures cannot see individually."""

    lm_cache: bool = False
    """DSPy's response cache. Off by default (a reviewed change): with it on, a
    retry of an identical invocation at temperature zero replays the cached
    failure for free and is recorded as a fresh attempt."""

    contract_repairs: int = Field(default=MAX_CONTRACT_REPAIRS, ge=0)
    """Full extra RLM runs allowed per chunk to fix rejected contracts."""

    def request_settings(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class CompletionTracker:
    """How the last invocation ended, recorded as it happens.

    Set before DSPy's fallback extraction starts, so an extraction that raises
    is still known to have been one. Reading ``final_reasoning`` afterwards
    only works when extraction succeeded.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mode = "error"
        self.iterations = 0
        self.iterations_to_submit: int | None = None
        self.submit_rejections: list[str] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "iterations_used": self.iterations,
            "iterations_to_submit": self.iterations_to_submit,
            "submit_rejections": tuple(self.submit_rejections),
        }


def instrument_completion(rlm: Any) -> CompletionTracker:
    """Attach a :class:`CompletionTracker` to a ``dspy.RLM`` instance.

    Wraps three of the installed DSPy 3.3.1 RLM's methods on the instance:
    each iteration, each SUBMIT's validation (so type errors DSPy feeds back are
    kept), and the extraction fallback (marked before it runs).
    """
    tracker = CompletionTracker()
    execute_iteration = rlm._execute_iteration
    process_final = rlm._process_final_output
    extract_fallback = rlm._extract_fallback

    def iteration(*args, **kwargs):
        tracker.iterations += 1
        # A stage per iteration: every model call, subcall and REPL step inside
        # carries the iteration and the invocation's stage ids.
        with ledger.stage("rlm_iteration", iteration=tracker.iterations):
            return execute_iteration(*args, **kwargs)

    def final_output(*args, **kwargs):
        parsed, error = process_final(*args, **kwargs)
        if error:
            tracker.submit_rejections.append(error)
            ledger.record(
                {"event_type": "submit_rejected", "iteration": tracker.iterations, "error": error}
            )
        else:
            tracker.mode = "submit"
            tracker.iterations_to_submit = tracker.iterations
            ledger.record({"event_type": "submit_accepted", "iteration": tracker.iterations})
        return parsed, error

    def extract(*args, **kwargs):
        tracker.mode = "extract"
        ledger.record({"event_type": "extract_start", "iterations_used": tracker.iterations})
        try:
            with ledger.stage("rlm_extract"):
                result = extract_fallback(*args, **kwargs)
        except BaseException as exc:
            ledger.record(
                {"event_type": "extract_error", "error": str(exc), "error_type": type(exc).__name__}
            )
            raise
        ledger.record({"event_type": "extract_end"})
        return result

    rlm._execute_iteration = iteration
    rlm._process_final_output = final_output
    rlm._extract_fallback = extract
    return tracker


def _language_model(model: str, api_key: str | None, settings: GenerationSettings, guard: Any):
    kwargs = {key: value for key, value in settings.request_settings().items() if value is not None}
    return providers.dspy_lm(
        model,
        api_key=api_key,
        call_guard=guard,
        num_retries=settings.provider_retries,
        cache=settings.lm_cache,
        **kwargs,
    )


def _rlm(
    dspy: Any, signature: Any, *, settings: GenerationSettings, language_model: Any, tools=None
):
    from bandits.analyze.rlm_history import record_repl

    rlm = record_repl(
        dspy.RLM(
            signature,
            max_iters=settings.root_max_iterations,
            max_llm_calls=settings.max_subcalls,
            max_output_chars=settings.max_output_chars,
            sub_lm=language_model,
            tools=tools or None,
        )
    )
    # DSPy builds fresh subcall tools per invocation with its own worker count;
    # binding the count here is the one change, the tools themselves are DSPy's.
    rlm._make_llm_tools = functools.partial(
        rlm._make_llm_tools, max_workers=settings.subcall_workers
    )
    return rlm, instrument_completion(rlm)


def _import_dspy() -> Any:
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise MiningError("RLM mining needs the 'audit' extra: uv sync --extra audit") from exc
    return dspy


def build_predictor(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    view: TraceView = TraceView.USER_MESSAGES,
    max_iterations: int | None = None,
    max_llm_calls: int | None = None,
    max_tokens: int | None = None,
    settings: GenerationSettings | None = None,
    guard: Any = None,
    catalog: Any = None,
    accounts_mode: bool = False,
) -> _Predictor:
    """A ``dspy.RLM`` over one chunk, imported only when mining actually runs.

    The per-chunk ceilings are what the real model needed rather than what
    seemed reasonable: measured against tau2 airline requests, one chunk of
    three traces took 5 to 18 calls, and a chunk that hit the old 12-iteration
    limit fell back to DSPy's ``extract`` and returned a partial answer.

    ``max_iterations``/``max_llm_calls``/``max_tokens`` are the older spellings
    of ``settings.root_max_iterations``/``max_subcalls``/``max_tokens`` and win
    when given. With ``catalog``, the evidence helpers are registered as tools.

    DSPy and its REPL sandbox are an optional extra: the core install stays at
    three runtime dependencies, and the tests below run against an injected
    predictor rather than a model.
    """
    settings = settings or GenerationSettings()
    overrides = {
        key: value
        for key, value in (
            ("root_max_iterations", max_iterations),
            ("max_subcalls", max_llm_calls),
            ("max_tokens", max_tokens),
        )
        if value is not None
    }
    if overrides:
        settings = settings.replace(**overrides)
    dspy = _import_dspy()
    from bandits.analyze.rlm_history import scoped_to_history

    language_model = _language_model(model, api_key, settings, guard)

    # Typed rather than list[dict]: the decoder then enforces the fields, and a
    # topic object cannot be submitted as a valid contract in the first place.
    # Built from real classes rather than a signature string, because DSPy
    # resolves names in a string against its own namespace and cannot see these.
    class _Mine(dspy.Signature):
        chunk: str = dspy.InputField(desc="the traces to read this round")
        taxonomy: str = dspy.InputField(desc="contracts built so far, possibly empty")
        correction: str = dspy.InputField(
            desc="empty on a first attempt; otherwise what was wrong with your last answer"
        )
        contracts: list[ProposedContract] = dspy.OutputField()
        operations: list[ProposedOperation] = dspy.OutputField()
        assignments: dict[str, str] = dspy.OutputField()
        ambiguous_trace_ids: list[str] = dspy.OutputField()
        uncovered_trace_ids: list[str] = dspy.OutputField()

    # The query, in the RLM's sense: it belongs in the token window, not in a
    # REPL variable the model has to remember to print.
    _Mine.__doc__ = account_family_instruction(view) if accounts_mode else instruction_for(view)
    rlm, tracker = _rlm(
        dspy,
        _Mine,
        settings=settings,
        language_model=language_model,
        tools=catalog.tools() if catalog is not None else None,
    )

    def predict(*, chunk: str, taxonomy: str, question: str = "") -> Any:
        # ``question`` carries a correction when one is being asked for, and is
        # empty otherwise. It gets its own input field rather than being
        # appended to the taxonomy: concatenating it there made that variable
        # invalid JSON, and the model's first move is to parse it.
        #
        # Forcing JSONAdapter was tried and reverted: a controlled Nemotron run
        # made more calls per chunk and stalled on the JSON schema syntax
        # itself. ChatAdapter's automatic fallback is cheaper.
        tracker.reset()
        with dspy.context(lm=language_model):
            return rlm(chunk=chunk, taxonomy=taxonomy, correction=question)

    wrapped = with_cost(scoped_to_history(predict, language_model))
    wrapped.completion = tracker  # type: ignore[attr-defined]
    return wrapped


def build_account_predictor(
    *,
    catalog: Any,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    view: TraceView = TraceView.FULL_TRAJECTORY,
    settings: GenerationSettings | None = None,
    guard: Any = None,
) -> Any:
    """One RLM invocation per run, writing a typed :class:`ProposedAccount`.

    The account is the single output field, typed, so a malformed SUBMIT is
    refused inside the loop with DSPy's own error and the model can fix it
    while its working variables still exist.
    """
    settings = settings or GenerationSettings()
    dspy = _import_dspy()
    from bandits.analyze.rlm_history import scoped_to_history

    language_model = _language_model(model, api_key, settings, guard)

    class _Account(dspy.Signature):
        run_index: str = dspy.InputField(
            desc="the run to account for, with candidate instruction refs"
        )
        correction: str = dspy.InputField(
            desc="empty on a first attempt; otherwise why the last account was rejected"
        )
        account: ProposedAccount = dspy.OutputField()

    _Account.__doc__ = account_instruction(view)
    rlm, tracker = _rlm(
        dspy, _Account, settings=settings, language_model=language_model, tools=catalog.tools()
    )

    def predict(*, run_index: str, question: str = "") -> Any:
        tracker.reset()
        with dspy.context(lm=language_model):
            return rlm(run_index=run_index, correction=question)

    wrapped = with_cost(scoped_to_history(predict, language_model))
    wrapped.completion = tracker  # type: ignore[attr-defined]
    return wrapped


def with_cost(predict: Any) -> Any:
    """Attach a ``.cost()`` reporting what the last prediction was billed.

    ``scoped_to_history`` already isolates the history entries one prediction
    added; this reads the provider's own ``cost`` field off those same entries,
    so the figure a budget is enforced against is the one being charged rather
    than one derived from token counts here.
    """
    spend = getattr(predict, "spend", None)

    def cost() -> float | None:
        entries = getattr(spend, "entries", None) or ()
        total = 0.0
        seen = False
        for entry in entries:
            value = entry.get("cost") if isinstance(entry, dict) else None
            if isinstance(value, (int, float)):
                total += float(value)
                seen = True
        return total if seen else None

    predict.cost = cost  # type: ignore[attr-defined]
    return predict


def _last_call_was_truncated(predict: Any) -> bool:
    """Whether the call that produced this prediction was cut off mid-output.

    DSPy's RLM loop returns the instant one iteration's action calls FINAL
    successfully, so the last entry in ``predict.spend.entries`` is always the
    exact call whose parsed code produced this prediction — never an earlier,
    abandoned attempt. A response Fireworks or DeepSeek cut off at the
    provider's token ceiling is not a plan that happened to finish early: two
    real runs each produced one response that ran to its full ceiling (32768
    and 65536 tokens) — one a hallucinated multi-turn transcript, the other a
    repeated sentence — and neither was output a taxonomy should be built from,
    even where DSPy's parser managed to salvage something that looked valid.
    """
    spend = getattr(predict, "spend", None)
    entries = getattr(spend, "entries", None) if spend is not None else None
    if not entries:
        return False
    last = entries[-1]
    if not isinstance(last, dict):
        return False
    response_obj = last.get("response")
    choices = getattr(response_obj, "choices", None)
    if not choices:
        return False
    return getattr(choices[0], "finish_reason", None) == "length"


def _spend_of(predict: Any) -> tuple[int | None, dict[str, int]]:
    """What the predictor says its last prediction cost, if it says anything.

    Optional by design: the injected predictors the tests use are plain
    functions, and a backend that cannot report its own history should leave the
    count unknown rather than have one invented for it.
    """
    spend = getattr(predict, "spend", None)
    if spend is None:
        return None, {}
    try:
        return spend()
    except Exception:  # noqa: BLE001 - a bookkeeping failure must not lose the chunk
        return None, {}


def _prompt_tokens(predict: Any) -> tuple[int | None, ...]:
    """Each call's reported prompt tokens, in order: how the root's history grew.

    A page limit bounds one observation, not the conversation: everything printed
    is resent on every later root call. This is the measurement that shows
    whether that accumulation matters. ``None`` where a call reported no usage.
    """
    spend = getattr(predict, "spend", None)
    entries = getattr(spend, "entries", None) or ()
    growth: list[int | None] = []
    for entry in entries:
        usage = entry.get("usage") if isinstance(entry, dict) else None
        value = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        growth.append(value if isinstance(value, int) else None)
    return tuple(growth)


def _added_spend(
    calls: int | None,
    tokens: dict[str, int],
    cost: float | None,
    more_calls: int | None,
    more_tokens: dict[str, int],
    more_cost: float | None,
) -> tuple[int | None, dict[str, int], float | None]:
    """Sum two spend readings taken from the same predictor at different times.

    ``scoped_to_history`` reports only the calls made since the wrapper was
    last invoked, so a repair's reading replaces rather than extends the
    original attempt's — summing here is what makes the two attempts add up
    to what the chunk actually spent.
    """
    total_calls = None if calls is None and more_calls is None else (calls or 0) + (more_calls or 0)
    total_tokens = dict(tokens)
    for field, value in more_tokens.items():
        total_tokens[field] = total_tokens.get(field, 0) + value
    total_cost = None if cost is None and more_cost is None else (cost or 0.0) + (more_cost or 0.0)
    return total_calls, total_tokens, total_cost


def _cost_of(predict: Any) -> float | None:
    """What the provider itself charged for the last prediction, in USD.

    Read from the provider's own figure rather than derived from token counts:
    a price computed here would be a guess about the current rate card, and a
    monetary ceiling enforced against a guess is not a ceiling.

    ``None`` means nothing reported a cost, which is not the same as free. The
    budget check treats it as unknown and says so rather than counting zero,
    because a run whose backend reports no cost would otherwise have its
    ``--max-usd`` silently disabled — the exact failure this fixes.
    """
    reporter = getattr(predict, "cost", None)
    if reporter is None:
        return None
    try:
        return reporter()
    except Exception:  # noqa: BLE001 - bookkeeping must not lose the chunk
        return None


def _decoded(value: Any) -> Any:
    """Parse a field the backend handed back as JSON text rather than a value.

    DSPy usually returns the declared types, but not always: when the root model
    runs out of iterations it falls back to an ``extract`` pass whose fields
    arrive as strings, and a model writing JSON into a string field does the
    same. Every parser below type-checks its input, so an undecoded string was
    silently dropped — a chunk that recorded three CREATE operations could
    contribute no contracts at all, and the run ended reporting an empty
    taxonomy it had actually paid to build.

    Returns the value untouched when it is not a string or does not parse, so a
    genuinely malformed reply is still handled by the parsers rather than here.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    # Fenced blocks are common when a model writes JSON into a text field.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except ValueError:
        return value


def _rows(value: Any) -> list[Any]:
    """Decoded model output as a list of rows, or nothing.

    Guards the two shapes ``_decoded`` can produce that are not collections of
    rows. A field holding ``"1"`` decodes to an int, which raises on iteration;
    a field holding ``'"text"'`` decodes to a string, which iterates one
    character at a time and would feed the parsers a stream of single letters
    that each fail quietly. Both are model output, so neither may crash a run
    nor be mistaken for data.

    A bare dict is wrapped: a model returning one object where a list of one was
    asked for has answered the question, and rejecting it would discard a real
    contract on a formatting technicality.
    """
    decoded = _decoded(value)
    if isinstance(decoded, dict):
        return [decoded]
    if isinstance(decoded, (list, tuple)):
        return list(decoded)
    return []


def _raw_reply(prediction: Any) -> str:
    """Everything the backend returned, serialized for a human to read.

    Never truncated. It was capped at twenty thousand characters, which is
    smaller than a reply proposing a dozen contracts over long requests — so the
    one case this exists for, a big chunk that came back unusable, was the case
    it silently cut in half. A run costs money and a disk does not.

    Best effort in the other direction only: an unserializable reply costs the
    record, never the chunk.
    """
    fields = (
        "contracts",
        "operations",
        "assignments",
        "ambiguous_trace_ids",
        "uncovered_trace_ids",
    )
    try:
        return json.dumps(
            {name: getattr(prediction, name, None) for name in fields},
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        return str(prediction)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _field(raw: dict, *names: str) -> Any:
    """The first of several spellings a model might use for one field.

    A model writes these keys, and it does not read the schema as strictly as
    the schema is checked. Asked for ``required_outcome_shape`` it may return
    ``required_outcome``, ``outcome_shape``, or the camelCase form, and asked
    for ``definition`` it may write ``description``. Every one of those is the
    field that was asked for, and rejecting the contract over the spelling
    discards work the run already paid for.
    """
    for name in names:
        if name in raw and raw[name] not in (None, "", [], {}):
            return raw[name]
    return None


def _lines(raw: Any) -> tuple[str, ...]:
    """A list of statements, however the model chose to express one.

    A single statement arrives as a bare string far more often than as a list
    of one — "the reservation is cancelled" rather than ``["..."]`` — and
    treating that as absent was rejecting well-formed contracts for their
    punctuation.
    """
    decoded = _decoded(raw)
    if isinstance(decoded, str):
        text = decoded.strip()
        return (text,) if text else ()
    if isinstance(decoded, dict):
        # e.g. {"outcome": "..."} — take the values, which are the statements.
        return tuple(str(v).strip() for v in decoded.values() if str(v).strip())
    if isinstance(decoded, (list, tuple)):
        return tuple(str(item).strip() for item in decoded if str(item).strip())
    return ()


def _string_tuple(raw: Any, *, allowed: set[str] | None = None) -> tuple[str, ...]:
    """Clean a model-written list of strings, dropping repeats and unknown ids.

    A model writes these. One that hallucinates a trace id, or repeats one,
    would otherwise fail a contract validator and lose the whole chunk —
    including the parts it got right — so unknown values are dropped here and
    the drop is reported as a limitation by the caller.
    """
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        # A decoded scalar or bare string is not a list of ids. A string would
        # otherwise iterate character by character and yield nothing useful.
        return ()
    seen: dict[str, None] = {}
    for item in raw:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or (allowed is not None and value not in allowed):
            continue
        seen.setdefault(value, None)
    return tuple(seen)


def _as_mapping(raw: Any) -> dict[str, Any] | None:
    """One model-written record as a plain dict, however the backend returned it.

    Typing the signature was supposed to stop topics being submitted, and it
    does — but it also changes what comes back: DSPy hands over
    ``ProposedContract`` instances rather than dicts, and every parser here
    tested ``isinstance(raw, dict)`` and dropped them. The fix for the empty
    taxonomy would have reproduced the empty taxonomy.

    Both shapes are accepted rather than only the typed one, because the
    backend is not the only caller: an untyped predictor, a repaired reply and
    every injected test double still return dicts.
    """
    if isinstance(raw, dict):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001 - a odd model must not lose the record
            return None
    return None


def _parse_contract(raw: Any, *, known_traces: set[str]) -> FamilyContract | None:
    """Build one contract from a model-written dict, or nothing.

    Returns ``None`` rather than raising for anything malformed. A contract the
    model failed to argue for — no outcome shape, no definition — is dropped and
    counted, because keeping it would put a topic into a taxonomy that is
    supposed to hold only verifiable claims.
    """
    raw = _as_mapping(raw)
    if raw is None:
        return None
    contract_id = _text(_field(raw, "contract_id", "id", "family_id"))
    name = _text(_field(raw, "name", "family_name", "title"))
    definition = _text(_field(raw, "definition", "description", "definition_text"))
    outcome = _lines(
        _field(
            raw,
            "required_outcome_shape",
            "required_outcome",
            "outcome_shape",
            "requiredOutcomeShape",
            "required_outcomes",
            "outcome",
        )
    )
    if not name and definition:
        # A contract that argued its claim but skipped the label is worth more
        # than its missing name; the definition's first clause stands in.
        name = definition.split(".")[0][:60]
    if not (name and definition and outcome):
        return None
    if not contract_id:
        # Derived from the claim rather than invented as a counter, so the same
        # contract proposed twice in one run lands on one id instead of two.
        contract_id = f"contract-{hashlib.sha256(definition.lower().encode()).hexdigest()[:12]}"

    supporting = _string_tuple(
        _field(raw, "supporting_trace_ids", "supporting_traces", "members", "trace_ids"),
        allowed=known_traces,
    )
    counter = _string_tuple(
        _field(raw, "counterexample_trace_ids", "counterexamples", "counterexample_traces"),
        allowed=known_traces,
    )
    revision = raw.get("revision")
    try:
        return FamilyContract(
            contract_id=contract_id,
            name=name,
            definition=definition,
            inclusion_rules=_lines(_field(raw, "inclusion_rules", "includes", "inclusion")),
            exclusion_rules=_lines(_field(raw, "exclusion_rules", "excludes", "exclusion")),
            required_outcome_shape=outcome,
            supporting_trace_ids=supporting,
            # A trace cannot both support and refute one contract; support wins
            # because it is the claim the model is making, and the validator
            # would otherwise reject the contract outright.
            counterexample_trace_ids=tuple(t for t in counter if t not in set(supporting)),
            revision=revision if isinstance(revision, int) and revision >= 1 else 1,
        )
    except ValueError:
        return None


def _parse_operation(raw: Any, *, known_traces: set[str]) -> TaxonomyOperation | None:
    raw = _as_mapping(raw)
    if raw is None:
        return None
    try:
        operation = Operation(_text(raw.get("operation")).upper())
    except ValueError:
        return None
    rationale = _text(raw.get("rationale"))
    if not rationale:
        # An unjustified operation is not a recorded decision. Dropping it is
        # safer than keeping it: the stop condition counts mutations, and a
        # mutation nobody argued for would either block convergence forever or
        # license a change with no evidence behind it.
        return None
    material = raw.get("material")
    return TaxonomyOperation(
        operation=operation,
        contract_ids=_string_tuple(raw.get("contract_ids")),
        trace_ids=_string_tuple(raw.get("trace_ids"), allowed=known_traces),
        rationale=rationale,
        material=True if material is None else bool(material),
    )


def _parse_assignments(raw: Any, *, chunk_ids: set[str], contract_ids: set[str]) -> dict[str, str]:
    """Keep only assignments naming a real chunk trace and a real contract."""
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, str] = {}
    for trace_id, contract_id in raw.items():
        if not isinstance(trace_id, str) or not isinstance(contract_id, str):
            continue
        if trace_id.strip() in chunk_ids and contract_id.strip() in contract_ids:
            cleaned[trace_id.strip()] = contract_id.strip()
    return cleaned


class _TaxonomyState:
    """The working taxonomy as the loop mutates it.

    Kept out of the contracts deliberately: every model in ``rlm_models`` is
    frozen, because an artifact that could be edited after the fact is not
    evidence. This is the mutable scratch space that produces one, and it never
    leaves this module.
    """

    def __init__(self) -> None:
        self.contracts: dict[str, FamilyContract] = {}
        self.assignments: dict[str, str] = {}
        self.ambiguous: set[str] = set()
        self.uncovered: set[str] = set()
        self.seen: set[str] = set()
        self.recently_affected: set[str] = set()
        self.revision_aliases = {}

    revision_aliases: dict[str, str]
    """Ids the model invented on a REVISE, mapped to the contract they belong to."""

    def enforce_revisions(
        self, operations: Sequence[TaxonomyOperation], contracts: list[FamilyContract]
    ) -> list[str]:
        """Make a REVISE keep the contract it revises.

        The model reasons about this correctly and still gets it wrong. In a
        recorded run it decided to "revise to remove the earlier-date
        constraint", then wrote the broadened wording under a *new*
        ``contract_id`` derived from that wording. The original was never
        retired, so one lineage ended up split across two contracts that said
        almost the same thing.

        A REVISE that names a live contract therefore rewrites that contract in
        place: the id it named wins, the revision counter advances, and the
        model's invented id is discarded. Nothing about the semantics is
        guessed at — the operation says which contract it is revising, and this
        only enforces that the returned body lands on it.
        """
        notes: list[str] = []
        self.revision_aliases: dict[str, str] = {}
        proposed = {c.contract_id: c for c in contracts}
        for op in operations:
            if op.operation is not Operation.REVISE or not op.contract_ids:
                continue
            target = op.contract_ids[0]
            existing = self.contracts.get(target)
            if existing is None:
                continue
            # The body it returned, whatever it chose to call it. A revision
            # that reused the right id needs no repair.
            body = proposed.get(target)
            if body is not None:
                # Reusing the right id is the good case, and it still has to be
                # taken out of the list: the revised copy is appended below, and
                # leaving the original in place put the same contract in twice.
                contracts.remove(body)
            else:
                strays = [c for c in contracts if c.contract_id not in self.contracts]
                if len(strays) != 1:
                    continue
                body = strays[0]
                # Assignments in this chunk still point at the invented id, so
                # the rename has to be remembered for them too.
                self.revision_aliases[body.contract_id] = target
                notes.append(
                    f"a REVISE of {target} returned its new wording under "
                    f"{body.contract_id!r}; the wording was applied to {target} and the "
                    "invented id discarded"
                )
                contracts.remove(body)
            contracts.append(
                body.replace(
                    contract_id=target,
                    revision=max(existing.revision + 1, body.revision),
                    # Evidence accumulates across a revision: the traces that
                    # motivated the original still support the broadened form.
                    supporting_trace_ids=tuple(
                        dict.fromkeys(existing.supporting_trace_ids + body.supporting_trace_ids)
                    ),
                )
            )
        return notes

    def apply_operations(
        self, operations: Sequence[TaxonomyOperation], contracts: Sequence[FamilyContract]
    ) -> None:
        """Make SPLIT, MERGE and REVISE actually change the taxonomy.

        These used to be recorded and nothing more: the loop counted them for
        the stop condition while the taxonomy changed only if the model happened
        to also restate every contract correctly in ``contracts``. A MERGE the
        model declared but did not carry out left both originals standing, and a
        run could then "converge" over a taxonomy that never matched what its
        own operation log said had been done.

        So the operation log is now authoritative for removal. A MERGE consumes
        the contracts it names except the one it produced; a SPLIT consumes the
        contract it split when replacements were supplied. Creating and
        rewording still arrive as contract bodies, because only the model can
        write those — but whether an old contract survives is decided here.
        """
        proposed = {c.contract_id for c in contracts}
        for op in operations:
            if op.operation is Operation.MERGE and len(op.contract_ids) >= 2:
                # By the plan's convention the produced contract is named last.
                *consumed, produced = op.contract_ids
                if produced in proposed or produced in self.contracts:
                    for contract_id in consumed:
                        if contract_id != produced:
                            self._retire(contract_id, into=produced)
            elif op.operation is Operation.SPLIT and op.contract_ids:
                original, *replacements = op.contract_ids
                live = [r for r in replacements if r in proposed or r in self.contracts]
                # Only when the split actually produced something. A SPLIT that
                # named no replacement would otherwise delete a family and leave
                # its members pointing at nothing.
                if live and original not in live:
                    self._retire(original, into=None)

    def _retire(self, contract_id: str, *, into: str | None) -> None:
        """Drop a contract, moving or unplacing whatever was assigned to it."""
        if contract_id not in self.contracts:
            return
        del self.contracts[contract_id]
        for trace_id, assigned in list(self.assignments.items()):
            if assigned != contract_id:
                continue
            if into is not None:
                self.assignments[trace_id] = into
            else:
                # A split whose members were not reassigned in this chunk goes
                # back to being an open question rather than silently vanishing.
                del self.assignments[trace_id]
                self.uncovered.add(trace_id)

    def apply(
        self,
        result: ChunkResult,
        contracts: Sequence[FamilyContract],
        *,
        limitations: list[str] | None = None,
    ) -> int:
        """Fold one chunk's output in, and report how many assignments moved.

        The churn figure is the count of traces whose *previous* placement
        changed — not the count of assignments made. A trace placed for the
        first time has not moved; counting it as movement would make an early
        sweep over unseen traces look unstable by construction.

        A trace that already had a placement may only move to a *different*
        contract when some operation in this chunk actually names it. Without
        that check a chunk could return ``operations: []`` and still move an
        already-assigned trace's ``assignments`` entry to an unrelated
        contract — recorded in a real run: a trace assigned to a contract
        about Requests' Unicode handling in pass 1 came back in pass 2 under a
        sparse-SVM contract, with no MERGE, SPLIT or REVISE naming either
        contract or that trace to explain the move. ``apply_operations``
        already retires a contract's members correctly for a real MERGE or
        SPLIT — first, so the trace lands on the survivor — and REVISE never
        moves a trace to a different contract at all. So a bare reassignment
        that isn't the target of any operation here is not a second, quieter
        way to do the same thing; it is rejected and the trace stays where it
        was, with the rejection recorded rather than silently applied.
        """
        self.apply_operations(result.operations, contracts)
        for contract in contracts:
            self.contracts[contract.contract_id] = contract

        # KEEP names a trace to affirm its *current* contract, not to license
        # moving it elsewhere - a response that says "KEEP t1" for t1's
        # existing family while separately assigning t1 to a different
        # contract must not count as an operation justifying that move.
        justified_moves = {
            trace_id
            for op in result.operations
            if op.operation is not Operation.KEEP
            for trace_id in op.trace_ids
        }

        moved = 0
        for trace_id, contract_id in result.assignments.items():
            previous = self.assignments.get(trace_id)
            if previous is not None and previous != contract_id and trace_id not in justified_moves:
                if limitations is not None:
                    limitations.append(
                        f"{trace_id} was already assigned to {previous!r}; a chunk tried "
                        f"to move it to {contract_id!r} with no operation naming it, so "
                        "the move was rejected and the trace stays on its previous contract"
                    )
                continue
            if previous is not None and previous != contract_id:
                moved += 1
            self.assignments[trace_id] = contract_id
            self.ambiguous.discard(trace_id)
            self.uncovered.discard(trace_id)

        for trace_id in result.ambiguous_trace_ids:
            if self.assignments.pop(trace_id, None) is not None:
                moved += 1
            self.uncovered.discard(trace_id)
            self.ambiguous.add(trace_id)

        for trace_id in result.uncovered_trace_ids:
            if self.assignments.pop(trace_id, None) is not None:
                moved += 1
            self.ambiguous.discard(trace_id)
            self.uncovered.add(trace_id)

        self.seen.update(result.trace_ids)

        # Traces the operations touched are pulled into the next chunk, so a
        # change is immediately tested against the evidence that motivated it.
        self.recently_affected = {trace_id for op in result.operations for trace_id in op.trace_ids}
        touched = {cid for op in result.operations if op.mutating for cid in op.contract_ids}
        if touched:
            self.recently_affected.update(
                trace_id for trace_id, cid in self.assignments.items() if cid in touched
            )
        return moved


def _compose_chunk(
    state: _TaxonomyState,
    *,
    unseen: list[str],
    chunk_size: int,
    seen_this_pass: set[str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Pick the next chunk from what this pass has not yet read.

    Every trace comes from ``unseen``, which is what remains of *this pass*, so
    a pass is guaranteed to terminate having read each eligible trace exactly
    once. That is the property the stopping rule rests on, and mixing in
    already-read traces to make the chunk more interesting would quietly break
    it — a chunk that re-read ten settled traces would leave ten of this pass's
    traces for a later chunk that may never come.

    The mixing the plan asks for still happens, but between passes rather than
    within one: each pass reshuffles, so a trace's neighbours differ every time,
    and the status label tells the model which traces it has seen before and
    what became of them.
    """
    statuses: dict[str, str] = {}
    picked: list[str] = []

    def status_of(trace_id: str) -> str:
        if trace_id in state.ambiguous:
            return "ambiguous"
        if trace_id in state.uncovered:
            return "uncovered"
        if trace_id in state.recently_affected:
            return "affected"
        if trace_id in state.assignments:
            return "assigned"
        return "unseen"

    # Unresolved and recently-affected traces first, so a chunk that can settle
    # something does. They are still drawn only from this pass's remainder.
    def priority(trace_id: str) -> int:
        return {"ambiguous": 0, "uncovered": 0, "affected": 1, "assigned": 2, "unseen": 2}[
            status_of(trace_id)
        ]

    for trace_id in sorted(unseen, key=lambda t: (priority(t), unseen.index(t))):
        if len(picked) >= chunk_size:
            break
        if trace_id in seen_this_pass:
            continue
        statuses[trace_id] = status_of(trace_id)
        picked.append(trace_id)
    return tuple(picked), statuses


def _chunk_payload(
    corpus: ReadOnlyCorpus, trace_ids: Sequence[str], statuses: dict[str, str]
) -> str:
    rows = [
        {
            "trace_id": view.trace_id,
            "messages": list(view.messages),
            "status": statuses.get(view.trace_id, "unseen"),
        }
        for view in corpus.get_user_message_batch(trace_ids)
        if view.readable
    ]
    return json.dumps(rows, indent=2, sort_keys=True, default=str)


def _taxonomy_payload(state: _TaxonomyState) -> str:
    return json.dumps(
        [
            contract.model_dump(mode="json")
            for contract in sorted(state.contracts.values(), key=lambda c: c.contract_id)
        ],
        indent=2,
        sort_keys=True,
        default=str,
    )


def _budget_stop(
    budget: Budget,
    *,
    iterations: int,
    calls: int,
    started: float,
    usd: float,
    guard: Any = None,
) -> StopReason | None:
    """Which ceiling, if any, this run has hit.

    Checked before each invocation, so a run never starts one past a limit it
    has already reached. With a ``guard``, its call/time/money figures — counted
    before every model call, not after every chunk — are the authority; the
    figures here remain for runs built without one.
    """
    if iterations >= budget.max_iterations:
        return StopReason.MAX_ITERATIONS
    if guard is not None:
        try:
            guard.check()
        except BudgetExhausted as exc:
            return exc.reason
        return None
    if calls >= budget.max_llm_calls:
        return StopReason.MAX_LLM_CALLS
    if time.monotonic() - started >= budget.max_seconds:
        return StopReason.MAX_SECONDS
    if budget.max_usd is not None and usd >= budget.max_usd:
        return StopReason.MAX_USD
    return None


class _Scheduler:
    """Fresh work first, then bounded retries.

    The incident's failure: a failed batch was never marked read, the retry
    guard fired only when *everything* left had failed, and with thirty fresh
    traces still waiting the same twenty were chosen five times running. Here a
    failed trace waits until every fresh trace has been tried; then the
    failures get another sweep, until each has used ``max_attempts``.
    """

    def __init__(
        self, order: Sequence[str], *, max_attempts: int, done: Sequence[str] = ()
    ) -> None:
        self.order = list(order)
        self.max_attempts = max_attempts
        self.done: set[str] = set(done)
        self.attempts: dict[str, int] = {}
        self.waiting: set[str] = set()
        self.exhausted: set[str] = set()

    def available(self) -> list[str] | None:
        remaining = [t for t in self.order if t not in self.done and t not in self.exhausted]
        if not remaining:
            return None
        fresh = [t for t in remaining if t not in self.waiting]
        if fresh:
            return fresh
        retryable = [t for t in remaining if self.attempts.get(t, 0) < self.max_attempts]
        newly = [t for t in remaining if t not in retryable and t not in self.exhausted]
        self.exhausted.update(newly)
        if newly:
            ledger.record(
                {
                    "event_type": "schedule_exhausted",
                    "trace_ids": newly,
                    "max_attempts": self.max_attempts,
                }
            )
        if not retryable:
            return None
        self.waiting.clear()
        ledger.record(
            {
                "event_type": "schedule_retry_sweep",
                "trace_ids": retryable,
                "attempts": {t: self.attempts.get(t, 0) for t in retryable},
            }
        )
        return retryable

    def attempt_of(self, trace_ids: Sequence[str]) -> int:
        return max((self.attempts.get(t, 0) for t in trace_ids), default=0) + 1

    def record(self, trace_ids: Sequence[str], *, ok: bool) -> None:
        for trace_id in trace_ids:
            self.attempts[trace_id] = self.attempts.get(trace_id, 0) + 1
            if ok:
                self.done.add(trace_id)
                self.waiting.discard(trace_id)
            else:
                self.waiting.add(trace_id)


def _stop_of(failure_kind: str) -> StopReason | None:
    """The ceiling a budget refusal names (``budget:<stop reason>``), else None.

    A refusal is never retried: the ceiling that refused one call refuses the next.
    """
    kind, _, reason = failure_kind.partition(":")
    return StopReason(reason) if kind == "budget" and reason else None


def _run_account(
    corpus: ReadOnlyCorpus,
    run_id: str,
    *,
    predict: Any,
    identity: AccountIdentity,
    attempt: int,
    correction: str = "",
    invocation_id: str = "",
) -> RunAccount:
    """One account invocation for one run, validated by the host.

    Every outcome is kept: an accepted account, a SUBMIT the host rejected (with
    its candidate and errors), a fallback extraction (quarantined with whatever
    it returned), or a failure. What was actually retrieved is read from the
    helpers' access log, not from what the model says it read.
    """
    catalog = corpus.evidence("account")
    catalog.set_scope((run_id,))
    mark = len(catalog.accesses)
    prediction = None
    error = ""
    failure = ""
    try:
        with ledger.stage(
            "rlm_account", run_id=run_id, attempt=attempt, invocation_id=invocation_id
        ):
            prediction = predict(
                run_index=json.dumps(catalog.overview_for(run_id)), question=correction
            )
    except ledger.LedgerWriteError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded on the account, not fatal
        error = str(exc)
        failure = _failure_kind(exc)
    finally:
        catalog.set_scope(None)
    calls, tokens = _spend_of(predict)
    cost = _cost_of(predict)
    snapshot = _completion_of(predict)
    refs, ranges = catalog.retrieved(run_id, since=mark)
    raw = getattr(prediction, "account", None) if prediction is not None else None
    proposed = coerce_account(raw) if raw is not None else None
    mode = snapshot.get("mode", "unknown")
    if error and mode != "extract":
        mode = "error"
    completion = Completion(
        mode=mode if mode in ("submit", "extract", "error") else "unknown",
        iterations_used=snapshot.get("iterations_used"),
        iterations_to_submit=snapshot.get("iterations_to_submit"),
        submit_rejections=tuple(snapshot.get("submit_rejections", ())),
        inspected_refs=refs,
        inspected_ranges=ranges,
        prompt_tokens_per_call=_prompt_tokens(predict),
        unresolved_refs=tuple(proposed.unresolved_refs) if proposed else (),
        error=error,
    )
    common = {
        "identity": identity,
        "attempt": attempt,
        "candidate": serialize_candidate(raw) if raw is not None else "",
        "completion": completion,
        "llm_calls": calls,
        "cost_usd": cost,
        "tokens": tokens,
    }
    if error:
        # An invocation that raised returned nothing to keep; whether it raised
        # during extraction is still visible in ``completion.mode``.
        return RunAccount(status="failed", failure_kind=failure, **common)
    if mode == "extract":
        return RunAccount(
            status="quarantined", account=proposed, failure_kind="no_submit", **common
        )
    if _last_call_was_truncated(predict):
        return RunAccount(
            status="rejected",
            account=proposed,
            failure_kind="truncated",
            validation_errors=("the call that produced this account was truncated",),
            **common,
        )
    if proposed is None:
        return RunAccount(
            status="rejected",
            failure_kind="host_rejected",
            validation_errors=("the returned value is not an account",),
            **common,
        )
    errors = validate_account(proposed, run_id=run_id, catalog=catalog)
    if errors:
        return RunAccount(
            status="rejected",
            account=proposed,
            failure_kind="host_rejected",
            validation_errors=errors,
            **common,
        )
    return RunAccount(status="accepted", account=proposed, **common)


def _correction_for(account: RunAccount) -> str:
    """What a retry is told, so a rejected SUBMIT is fixed rather than redone."""
    if account.status == "rejected" and account.validation_errors:
        return (
            "Your last account was rejected by host validation. Fix exactly these and SUBMIT "
            "again; keep everything else:\n- " + "\n- ".join(account.validation_errors[:8])
        )
    if account.status == "quarantined":
        return (
            "Your last attempt reached the iteration limit without SUBMIT. Read less, then "
            "SUBMIT an account with unknowns and unresolved_refs rather than running out."
        )
    return ""


def mine_accounts(
    corpus: ReadOnlyCorpus,
    run_ids: Sequence[str],
    *,
    predict: Any,
    identity_for: Callable[[str], AccountIdentity],
    budget: Budget,
    guard: Any = None,
    existing: Sequence[RunAccount] = (),
    on_account: Callable[[RunAccount, tuple[RunAccount, ...]], None] | None = None,
    started: float | None = None,
) -> tuple[tuple[RunAccount, ...], StopReason | None, list[str]]:
    """Produce one account per run, fresh runs before any retry.

    Runs already holding an accepted account are skipped: that is the reuse
    and resume path, and only accounts whose identity matches were passed in.
    Returns every attempt, the stop reason if a ceiling fired, and limitations.
    """
    accounts = list(existing)
    accepted = {a.run_id for a in existing if a.status == "accepted"}
    scheduler = _Scheduler(run_ids, max_attempts=budget.max_attempts, done=tuple(accepted))
    for account in existing:
        if account.status != "accepted":
            scheduler.attempts[account.run_id] = max(
                scheduler.attempts.get(account.run_id, 0), account.attempt
            )
    # The latest earlier attempt per run, so a resumed retry is still told why
    # its previous account was rejected.
    last: dict[str, RunAccount] = {
        run_id: account
        for run_id, account in latest_by_run(existing).items()
        if account.status != "accepted"
    }
    limitations: list[str] = []
    started = time.monotonic() if started is None else started
    calls = sum(a.llm_calls or 0 for a in accounts)
    usd = sum(a.cost_usd or 0.0 for a in accounts)
    stop: StopReason | None = None
    while True:
        available = scheduler.available()
        if not available:
            break
        stop = _budget_stop(
            budget, iterations=0, calls=calls, started=started, usd=usd, guard=guard
        )
        if stop is not None:
            break
        run_id = available[0]
        attempt = scheduler.attempt_of((run_id,))
        previous = last.get(run_id)
        invocation_id = uuid.uuid4().hex[:16]
        ledger.record(
            {
                "event_type": "invocation_start",
                "invocation_id": invocation_id,
                "kind": "account",
                "trace_ids": [run_id],
                "attempt": attempt,
                "correction": bool(previous),
            }
        )
        account = _run_account(
            corpus,
            run_id,
            predict=predict,
            identity=identity_for(run_id),
            attempt=attempt,
            correction=_correction_for(previous) if previous else "",
            invocation_id=invocation_id,
        )
        ledger.record(
            {
                "event_type": "invocation_end",
                "invocation_id": invocation_id,
                "kind": "account",
                "trace_ids": [run_id],
                "attempt": attempt,
                "status": account.status,
                "completion_mode": account.completion.mode,
                "failure_kind": account.failure_kind,
                "validation_errors": list(account.validation_errors),
                "llm_calls": account.llm_calls,
                "cost_usd": account.cost_usd,
            }
        )
        accounts.append(account)
        last[run_id] = account
        calls += account.llm_calls or 0
        usd += account.cost_usd or 0.0
        scheduler.record((run_id,), ok=account.status == "accepted")
        if on_account is not None:
            on_account(account, tuple(accounts))
        stop = _stop_of(account.failure_kind)
        if stop is not None:
            break
    if scheduler.exhausted:
        limitations.append(
            f"{len(scheduler.exhausted)} run(s) produced no accepted account after "
            f"{budget.max_attempts} attempt(s): {', '.join(sorted(scheduler.exhausted))}"
        )
    return tuple(accounts), stop, limitations


def mine_taxonomy(
    corpus: ReadOnlyCorpus,
    analysis_id: str,
    *,
    predict: _Predictor,
    analysis: Any = None,
    model: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = DEFAULT_SEED,
    budget: Budget | None = None,
    on_chunk: Callable[[ChunkResult], None] | None = None,
    session: Any = None,
    resume: Any = None,
    account_predict: Any = None,
    identity_for: Callable[[str], AccountIdentity] | None = None,
    reuse_accounts: Sequence[RunAccount] = (),
    on_account: Callable[[RunAccount], None] | None = None,
    guard: Any = None,
    contract_repairs: int = MAX_CONTRACT_REPAIRS,
    settings: dict[str, Any] | None = None,
) -> RLMClusteringRun:
    """Read the corpus through complete passes, then pause for review.

    Each pass reads every eligible trace exactly once, in its own shuffled
    order. A pass counts toward the schedule only when it finished; a pass cut
    short by a budget guard counts for nothing, however much of the corpus it
    got through.

    With ``account_predict`` (the full-trajectory path), every selected run
    first gets an account — one invocation per run — and families are formed
    over the accounts that are accepted and carry an intent reading. Runs that
    cannot be placed keep their reason: missing intent, contradictory evidence,
    processing failure or quarantined extraction.

    Nothing here decides the taxonomy has converged, because nothing here can.
    ``session``, when given, is checkpointed after every invocation so a run
    that dies mid-pass is resumable and a run in flight is observable.
    """
    budget = budget or Budget()
    state = _TaxonomyState()
    started = time.monotonic()

    resume_pass = 0
    resume_seen: set[str] = set()
    resume_order: tuple[str, ...] = ()
    resumed_accounts: tuple[RunAccount, ...] = ()
    if resume is not None:
        state.contracts = {c.contract_id: c for c in resume.contracts}
        state.assignments = dict(resume.assignments)
        state.ambiguous = set(resume.ambiguous_trace_ids)
        state.uncovered = set(resume.uncovered_trace_ids)
        state.seen = set(resume.assignments) | state.ambiguous | state.uncovered
        resume_pass = resume.pass_index
        resume_seen = set(resume.seen_this_pass)
        resume_order = resume.pass_order
        resumed_accounts = tuple(getattr(resume, "accounts", ()) or ())

    readable = list(corpus.readable_trace_ids())
    unreadable = corpus.unreadable_trace_ids()
    if not readable:
        raise MiningError(
            "no trace in this corpus has readable user messages; there is nothing to mine"
        )

    chunks: list[ChunkResult] = []
    passes: list[PassResult] = []
    limitations: list[str] = []
    calls = 0
    usd = 0.0
    cost_reported = False
    stop_reason: StopReason | None = None
    completed_passes = resume.completed_passes if resume is not None else 0
    accounts: tuple[RunAccount, ...] = tuple(resumed_accounts)

    accounts_mode = account_predict is not None
    if accounts_mode and identity_for is None:
        raise MiningError("account mining needs an identity for each account")

    if session is not None:
        session.begin(traces_total=len(readable), requested_passes=budget.passes, seed=seed)
    ledger.record(
        {
            "event_type": "run_started",
            "analysis_id": analysis_id,
            "view": corpus.view.value,
            "seed": seed,
            "traces": readable,
            "selection": list(corpus.list_trace_ids()) if corpus.selected else [],
            "accounts_mode": account_predict is not None,
            "budget": budget.model_dump(mode="json"),
            "settings": settings or {},
            "resumed": resume is not None,
            "reused_accounts": [a.run_id for a in reuse_accounts],
        }
    )

    if accounts_mode:
        existing = list(resumed_accounts)
        have = {a.run_id for a in existing if a.status == "accepted"}
        existing += [
            a for a in reuse_accounts if a.run_id not in have and a.run_id in set(readable)
        ]

        def checkpoint_account(account: RunAccount, so_far: tuple[RunAccount, ...]) -> None:
            if on_account is not None:
                on_account(account)
            if session is not None:
                session.checkpoint_account(account, so_far, elapsed=time.monotonic() - started)

        accounts, stop_reason, account_limits = mine_accounts(
            corpus,
            readable,
            predict=account_predict,
            identity_for=identity_for,
            budget=budget,
            guard=guard,
            existing=existing,
            on_account=checkpoint_account,
            started=started,
        )
        limitations.extend(account_limits)
        new = accounts[len(existing) :]
        calls += sum(a.llm_calls or 0 for a in new)
        for account in new:
            if account.cost_usd is not None:
                usd += account.cost_usd
                cost_reported = True
        latest = latest_by_run(accounts)
        eligible = [tid for tid in readable if tid in latest and latest[tid].eligible_for_families]
    else:
        latest = {}
        eligible = readable

    def account_payload(trace_ids: Sequence[str], statuses: dict[str, str]) -> str:
        rows = [family_row(latest[tid], statuses.get(tid, "unseen")) for tid in trace_ids]
        return json.dumps(rows, indent=2, sort_keys=True, default=str)

    payload = account_payload if accounts_mode else None

    if accounts_mode and not eligible and stop_reason is None:
        stop_reason = StopReason.NO_ELIGIBLE_ACCOUNTS
        limitations.append(
            "no run has an accepted account with an intent reading, so no family could be "
            "formed; see each account's status and unassigned reason"
        )

    for pass_index in range(resume_pass, budget.passes):
        if stop_reason is not None:
            break
        pass_seed = seed + pass_index
        if pass_index == resume_pass and resume_order:
            order = [tid for tid in resume_order if tid in set(eligible)]
        else:
            order = list(eligible)
            random.Random(pass_seed).shuffle(order)

        previous_placement = dict(state.assignments)
        contracts_before = tuple(sorted(state.contracts))
        pass_operations: list[TaxonomyOperation] = []
        pass_chunk_indices: list[int] = []
        seen_this_pass: set[str] = set(resume_seen) if pass_index == resume_pass else set()
        pass_complete = True
        scheduler = _Scheduler(order, max_attempts=budget.max_attempts, done=tuple(seen_this_pass))

        while True:
            available = scheduler.available()
            if available is None:
                break
            stop_reason = _budget_stop(
                budget, iterations=len(chunks), calls=calls, started=started, usd=usd, guard=guard
            )
            if stop_reason is not None:
                pass_complete = False
                break

            trace_ids, statuses = _compose_chunk(
                state, unseen=available, chunk_size=chunk_size, seen_this_pass=seen_this_pass
            )
            if not trace_ids:  # pragma: no cover - available is non-empty here
                pass_complete = False
                break

            result = _run_chunk(
                corpus,
                state,
                trace_ids=trace_ids,
                statuses=statuses,
                index=len(chunks),
                pass_index=pass_index,
                predict=predict,
                limitations=limitations,
                session_id=getattr(session, "session_id", "") if session else "",
                repairs_left=contract_repairs,
                payload=payload,
                attempt=scheduler.attempt_of(trace_ids),
            )
            chunks.append(result)
            pass_chunk_indices.append(result.index)
            ok = result.status == "success"
            scheduler.record(trace_ids, ok=ok)
            if ok:
                # Only a successful chunk read anything. An error or a
                # quarantined extraction leaves its traces owed to this pass.
                seen_this_pass.update(trace_ids)
            calls += result.llm_calls or 0
            if result.cost_usd is not None:
                usd += result.cost_usd
                cost_reported = True

            if ok:
                contracts = [state.contracts[cid] for cid in result.assignments.values()]
                state.apply(result, contracts, limitations=limitations)
                pass_operations.extend(result.operations)
            elif result.status == "quarantined":
                limitations.append(
                    f"chunk {result.index} reached its iteration cap without SUBMIT; its "
                    "extracted answer is quarantined in raw_reply and changed nothing"
                )

            if on_chunk is not None:
                on_chunk(result)
            if session is not None:
                session.checkpoint(
                    state,
                    pass_index=pass_index,
                    completed_passes=completed_passes,
                    chunk=result,
                    chunks=chunks,
                    passes=passes,
                    seen_this_pass=seen_this_pass,
                    pass_order=tuple(order),
                    calls=calls,
                    usd=usd,
                    elapsed=time.monotonic() - started,
                )
            budget_stop = _stop_of(result.failure_kind)
            if budget_stop is not None:
                stop_reason = budget_stop
                pass_complete = False
                break

        if scheduler.exhausted:
            pass_complete = False
            limitations.append(
                f"pass {pass_index} could not read {len(scheduler.exhausted)} trace(s) "
                f"after {budget.max_attempts} attempt(s) (a retry only after every fresh "
                "trace was tried), so it did not cover the corpus and does not count "
                "toward the requested passes"
            )

        # One last look at whatever this pass could not place. A trace read
        # before the family that fits it existed is not uncovered — it is a
        # trace that arrived early.
        unplaced = sorted(state.ambiguous | state.uncovered)
        if pass_complete and unplaced and state.contracts and stop_reason is None:
            reconcile_stop = _budget_stop(
                budget, iterations=len(chunks), calls=calls, started=started, usd=usd, guard=guard
            )
            if reconcile_stop is None:
                reconciled = _run_chunk(
                    corpus,
                    state,
                    trace_ids=tuple(unplaced[:chunk_size]),
                    statuses={t: "unresolved" for t in unplaced[:chunk_size]},
                    index=len(chunks),
                    pass_index=pass_index,
                    predict=predict,
                    limitations=limitations,
                    session_id=getattr(session, "session_id", "") if session else "",
                    repairs_left=contract_repairs,
                    payload=payload,
                )
                chunks.append(reconciled)
                pass_chunk_indices.append(reconciled.index)
                calls += reconciled.llm_calls or 0
                if reconciled.cost_usd is not None:
                    usd += reconciled.cost_usd
                    cost_reported = True
                if reconciled.status == "success":
                    state.apply(
                        reconciled,
                        [state.contracts[cid] for cid in reconciled.assignments.values()],
                        limitations=limitations,
                    )
                    pass_operations.extend(reconciled.operations)
                    recovered = len(unplaced) - len(state.ambiguous | state.uncovered)
                    if recovered > 0:
                        limitations.append(
                            f"pass {pass_index} placed {recovered} trace(s) that an earlier "
                            "chunk left unresolved, on a reconciliation sweep against the "
                            "finished taxonomy"
                        )
                if on_chunk is not None:
                    on_chunk(reconciled)

        reassigned = tuple(
            sorted(
                trace_id
                for trace_id, contract_id in state.assignments.items()
                if trace_id in previous_placement and previous_placement[trace_id] != contract_id
            )
        )
        passes.append(
            PassResult(
                pass_index=pass_index,
                seed=pass_seed,
                trace_ids=tuple(order if pass_complete else sorted(seen_this_pass)),
                chunk_indices=tuple(pass_chunk_indices),
                operations=tuple(pass_operations),
                contracts_before=contracts_before,
                contracts_after=tuple(sorted(state.contracts)),
                reassigned_trace_ids=reassigned,
                complete=pass_complete,
            )
        )
        if pass_complete:
            completed_passes += 1
        else:
            break

    if stop_reason is None:
        stop_reason = (
            StopReason.PASSES_COMPLETE if completed_passes >= budget.passes else StopReason.ERROR
        )

    if stop_reason is not StopReason.PASSES_COMPLETE:
        limitations.append(
            f"discovery stopped on {stop_reason.value} before completing its "
            f"{budget.passes} requested pass(es); {completed_passes} pass(es) read every "
            "eligible trace, so the rest of the corpus was seen fewer times than planned"
        )
    else:
        limitations.append(
            f"{completed_passes} complete pass(es) were run and the session paused for "
            "review; this is not a convergence test and the taxonomy is not final"
        )
    if state.ambiguous:
        limitations.append(
            f"{len(state.ambiguous)} trace(s) matched several contracts equally and were "
            "left ambiguous rather than forced into one"
        )
    if state.uncovered:
        limitations.append(
            f"{len(state.uncovered)} trace(s) matched no contract; the taxonomy does not reach them"
        )
    if unreadable:
        limitations.append(
            f"{len(unreadable)} trace(s) recorded no user messages and were never shown "
            "to the miner"
        )
    if not state.contracts:
        limitations.append("discovery produced no contracts at all")
    if budget.max_usd is not None and not cost_reported and guard is None:
        limitations.append(
            f"a monetary ceiling of ${budget.max_usd:.2f} was requested but no backend "
            "reported a cost, so the run was bounded only by its call, iteration and "
            "time limits"
        )
    if guard is not None and budget.max_usd is not None:
        limitations.append(
            "the monetary ceiling reserved an estimated worst case per call (request bytes "
            "as input tokens plus the output allowance); provider template tokens are not "
            "in that estimate"
            if guard.can_reserve
            else "the monetary ceiling was enforced on reported spend before each call; "
            "without per-token prices no call could be reserved in advance, so the last "
            "admitted call may overshoot it by its own cost"
        )
    if guard is not None and guard.attempts_per_call > 1:
        limitations.append(
            f"provider retries were allowed ({guard.attempts_per_call - 1} per call): the "
            "call count and recording see one call per admitted request, not each "
            "transport attempt beneath it"
        )
    if corpus.view.reads_agent_behavior:
        limitations.append(
            "this taxonomy was mined from full trajectories, so the miner could see "
            "what the agent did; its families must be checked against tool usage, "
            "trajectory length and outcome before being read as task families"
        )
        leaked = corpus.leakage_report(analysis)
        if leaked:
            limitations.append(
                f"OUTCOME LEAKAGE: {len(leaked)} trace(s) still show a recorded outcome "
                "value in the trajectory the miner read, so this taxonomy may be grouping "
                f"by outcome: {'; '.join(leaked[:5])}"
            )
        elif analysis is None:
            limitations.append(
                "no analysis was supplied, so the trajectory view was never checked for "
                "outcome values surviving redaction under unforeseen field names"
            )

        withheld = corpus.withheld_fields()
        if withheld:
            limitations.append(
                "outcome-bearing fields were stripped from the trajectory view before "
                f"mining: {', '.join(withheld)}"
            )
        else:
            limitations.append(
                "no outcome-bearing field was found to strip from the trajectory view; "
                "if this corpus records rewards under other names they reached the miner"
            )

    members_by_contract: dict[str, list[str]] = {}
    for trace_id, contract_id in state.assignments.items():
        members_by_contract.setdefault(contract_id, []).append(trace_id)
    final_contracts = tuple(
        contract.replace(
            supporting_trace_ids=tuple(sorted(members_by_contract.get(contract.contract_id, ())))
        )
        for contract in sorted(state.contracts.values(), key=lambda c: c.contract_id)
    )

    unassigned, coverage = _coverage(
        corpus,
        readable=readable,
        chunks=chunks,
        accounts=accounts,
        state=state,
        accounts_mode=accounts_mode,
    )

    ledger.record(
        {
            "event_type": "run_finished",
            "stop_reason": stop_reason.value,
            "completed_passes": completed_passes,
            "contracts": sorted(state.contracts),
            "assignments": dict(state.assignments),
            "unassigned_reasons": unassigned,
            "coverage": coverage,
            "budget_usage": guard.summary() if guard is not None else None,
        }
    )
    return RLMClusteringRun(
        analysis_id=analysis_id,
        view=corpus.view,
        seed=seed,
        contracts=final_contracts,
        chunks=tuple(chunks),
        passes=tuple(passes),
        completed_passes=completed_passes,
        requested_passes=budget.passes,
        assignments=dict(state.assignments),
        ambiguous_trace_ids=tuple(sorted(state.ambiguous)),
        uncovered_trace_ids=tuple(sorted(state.uncovered)),
        unreadable_trace_ids=tuple(sorted(unreadable)),
        stop_reason=stop_reason,
        budget=budget,
        model=model,
        prompt_digest=prompt_digest(model),
        limitations=tuple(dict.fromkeys(limitations)),
        accounts=accounts,
        selection=tuple(corpus.list_trace_ids()) if corpus.selected else (),
        settings=dict(settings or {}),
        unassigned_reasons=unassigned,
        coverage=coverage,
    )


def _coverage(
    corpus: ReadOnlyCorpus,
    *,
    readable: Sequence[str],
    chunks: Sequence[ChunkResult],
    accounts: Sequence[RunAccount],
    state: _TaxonomyState,
    accounts_mode: bool,
) -> tuple[dict[str, str], dict[str, int]]:
    """Why each unplaced run is unplaced, and coverage counted separately.

    "Read" never means that an id appeared in a batch: ``source_accessed``
    counts runs whose evidence a helper actually returned, and
    ``account_complete`` counts accepted accounts, unknowns included.
    """
    latest = latest_by_run(accounts)
    in_chunks = {t for c in chunks for t in c.trace_ids}
    read = {t for c in chunks if c.status == "success" for t in c.trace_ids}
    quarantined_chunks = {
        t for c in chunks if c.status == "quarantined" for t in c.trace_ids
    } - read
    failed_chunks = in_chunks - read - quarantined_chunks
    accessed: set[str] = set()
    if corpus.view.reads_agent_behavior:
        accessed = {
            run_id
            for catalog in corpus.evidence_catalogs()
            for run_id in readable
            if catalog.retrieved(run_id)[0]
        }

    reasons: dict[str, str] = {}
    for trace_id in readable:
        if trace_id in state.assignments:
            continue
        if accounts_mode:
            account = latest.get(trace_id)
            if account is None:
                reasons[trace_id] = "not_attempted"
                continue
            if account.unassigned_reason():
                reasons[trace_id] = account.unassigned_reason()
                continue
        if trace_id in state.ambiguous or trace_id in state.uncovered:
            reasons[trace_id] = "uncertain_boundary"
        elif trace_id in quarantined_chunks:
            reasons[trace_id] = "quarantined_extract"
        elif trace_id in failed_chunks:
            reasons[trace_id] = "processing_failure"
        else:
            # Eligible, but no family invocation reached it (a ceiling fired first).
            reasons[trace_id] = "not_attempted"
    coverage = coverage_of(
        selected=len(readable),
        chunks=chunks,
        accounts=accounts,
        assignments=state.assignments,
        unresolved=state.ambiguous | state.uncovered,
        accessed=accessed,
    )
    return reasons, coverage


def _completion_of(predict: Any) -> dict[str, Any]:
    """The invocation's completion provenance, or ``unknown`` for a bare predictor.

    Injected test predictors and older wrappers carry no tracker; their answers
    are treated as submitted *only* when they say so, never by default.
    """
    tracker = getattr(predict, "completion", None)
    if tracker is None:
        return {"mode": "unknown"}
    snapshot = tracker.snapshot() if hasattr(tracker, "snapshot") else dict(tracker)
    return snapshot


def _failure_kind(exc: BaseException) -> str:
    if isinstance(exc, BudgetExhausted):
        return f"budget:{exc.reason.value}"
    return "provider_error"


def _run_chunk(
    corpus: ReadOnlyCorpus,
    state: _TaxonomyState,
    *,
    trace_ids: tuple[str, ...],
    statuses: dict[str, str],
    index: int,
    pass_index: int,
    predict: _Predictor,
    limitations: list[str],
    session_id: str = "",
    repairs_left: int = MAX_CONTRACT_REPAIRS,
    payload: Callable[[Sequence[str], dict[str, str]], str] | None = None,
    attempt: int = 1,
) -> ChunkResult:
    """One invocation over one chunk, bracketed by invocation start/end events."""
    invocation_id = uuid.uuid4().hex[:16]
    ledger.record(
        {
            "event_type": "invocation_start",
            "invocation_id": invocation_id,
            "kind": "family",
            "chunk_index": index,
            "pass_index": pass_index,
            "trace_ids": list(trace_ids),
            "attempt": attempt,
        }
    )
    result = _run_chunk_inner(
        corpus,
        state,
        trace_ids=trace_ids,
        statuses=statuses,
        index=index,
        pass_index=pass_index,
        predict=predict,
        limitations=limitations,
        session_id=session_id,
        repairs_left=repairs_left,
        payload=payload,
        attempt=attempt,
        invocation_id=invocation_id,
    )
    ledger.record(
        {
            "event_type": "invocation_end",
            "invocation_id": invocation_id,
            "kind": "family",
            "chunk_index": index,
            "trace_ids": list(trace_ids),
            "attempt": attempt,
            "status": result.status,
            "completion_mode": result.completion_mode,
            "failure_kind": result.failure_kind,
            "operations": [op.operation.value for op in result.operations],
            "assignments": result.assignments,
            "llm_calls": result.llm_calls,
            "cost_usd": result.cost_usd,
        }
    )
    return result


def _run_chunk_inner(
    corpus: ReadOnlyCorpus,
    state: _TaxonomyState,
    *,
    trace_ids: tuple[str, ...],
    statuses: dict[str, str],
    index: int,
    pass_index: int,
    predict: _Predictor,
    limitations: list[str],
    session_id: str,
    repairs_left: int,
    payload: Callable[[Sequence[str], dict[str, str]], str] | None,
    attempt: int,
    invocation_id: str,
) -> ChunkResult:
    chunk_json = (
        payload(trace_ids, statuses)
        if payload is not None
        else _chunk_payload(corpus, trace_ids, statuses)
    )
    taxonomy_json = _taxonomy_payload(state)
    started = time.monotonic()
    catalog = corpus.evidence("grouping") if corpus.view.reads_agent_behavior else None
    if catalog is not None:
        catalog.set_scope(trace_ids)
    try:
        with ledger.stage(
            "rlm_chunk",
            chunk_index=index,
            pass_index=pass_index,
            session_id=session_id,
            traces=len(trace_ids),
            invocation_id=invocation_id,
        ):
            prediction = predict(
                chunk=chunk_json,
                taxonomy=taxonomy_json,
                question="",
            )
    except ledger.LedgerWriteError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed chunk is recorded, not fatal
        calls, tokens = _spend_of(predict)
        completion = _completion_of(predict)
        return ChunkResult(
            index=index,
            pass_index=pass_index,
            trace_ids=trace_ids,
            llm_calls=calls,
            tokens=tokens,
            cost_usd=_cost_of(predict),
            duration_seconds=time.monotonic() - started,
            status="error",
            error=str(exc),
            completion_mode="extract" if completion.get("mode") == "extract" else "error",
            prompt_tokens_per_call=_prompt_tokens(predict),
            iterations_used=completion.get("iterations_used"),
            submit_rejections=tuple(completion.get("submit_rejections", ())),
            failure_kind=_failure_kind(exc),
            attempt=attempt,
        )
    finally:
        if catalog is not None:
            catalog.set_scope(None)
    completion = _completion_of(predict)
    provenance = {
        "prompt_tokens_per_call": _prompt_tokens(predict),
        "completion_mode": completion.get("mode", "unknown"),
        "iterations_used": completion.get("iterations_used"),
        "iterations_to_submit": completion.get("iterations_to_submit"),
        "submit_rejections": tuple(completion.get("submit_rejections", ())),
        "attempt": attempt,
    }
    if completion.get("mode") == "extract":
        # Fallback extraction after the iteration cap. Its answer is preserved
        # whole and quarantined: it never mutates the taxonomy and never counts
        # as reading these traces, because nothing validated it inside the loop.
        calls, tokens = _spend_of(predict)
        return ChunkResult(
            index=index,
            pass_index=pass_index,
            trace_ids=trace_ids,
            llm_calls=calls,
            tokens=tokens,
            cost_usd=_cost_of(predict),
            duration_seconds=time.monotonic() - started,
            status="quarantined",
            error="the iteration cap was reached without SUBMIT; the extracted answer is "
            "kept in raw_reply and was not applied",
            raw_reply=_raw_reply(prediction),
            failure_kind="no_submit",
            **provenance,
        )
    if _last_call_was_truncated(predict):
        # Discards the chunk's output when the call that actually produced it
        # was cut off at the provider's token ceiling, however plausible the
        # salvaged parse looks. Only the last history entry is checked: an
        # earlier truncated iteration may already have executed in the sandbox.
        calls, tokens = _spend_of(predict)
        return ChunkResult(
            index=index,
            pass_index=pass_index,
            trace_ids=trace_ids,
            llm_calls=calls,
            tokens=tokens,
            cost_usd=_cost_of(predict),
            duration_seconds=time.monotonic() - started,
            status="error",
            error="the call that produced this chunk's output was truncated at the "
            "provider's token ceiling; discarded rather than parsed",
            raw_reply=_raw_reply(prediction),
            failure_kind="truncated",
            **provenance,
        )
    calls, tokens = _spend_of(predict)
    cost = _cost_of(predict)

    known = set(corpus.list_trace_ids())
    raw_contracts = _rows(getattr(prediction, "contracts", ()))
    contracts = []
    dropped_contracts: list[str] = []
    for raw in raw_contracts:
        parsed = _parse_contract(raw, known_traces=known)
        if parsed is None:
            # Kept verbatim, because a count cannot say what was wrong with it.
            dropped_contracts.append(json.dumps(raw, default=str))
        else:
            contracts.append(parsed)
    if dropped_contracts and repairs_left > 0:
        # Hand the model its own invalid output and the reason, rather than
        # discarding work it already paid to produce. One whole extra RLM run:
        # set ``contract_repairs`` to 0 to forbid it.
        with ledger.stage("rlm_repair", chunk_index=index, invocation_id=invocation_id):
            repaired = _repair_contracts(
                dropped_contracts,
                chunk_json=chunk_json,
                taxonomy_json=taxonomy_json,
                predict=predict,
                known=known,
            )
        if repaired:
            contracts.extend(repaired)
            limitations.append(
                f"chunk {index} returned {len(dropped_contracts)} contract(s) that failed "
                f"validation; {len(repaired)} were recovered on a second attempt and every "
                "original is kept in dropped_contracts"
            )
        # Added to, not re-read from: scoped_to_history reports only the calls
        # made since it was last invoked, so the repair's own reading replaces
        # rather than includes the original attempt's unless summed here.
        repair_calls, repair_tokens = _spend_of(predict)
        repair_cost = _cost_of(predict)
        calls, tokens, cost = _added_spend(
            calls, tokens, cost, repair_calls, repair_tokens, repair_cost
        )
    if dropped_contracts:
        limitations.append(
            f"chunk {index} proposed {len(dropped_contracts)} contract(s) the parser "
            "refused; see the chunk's dropped_contracts for exactly what was returned"
        )

    # Merged with what already exists so an assignment may name a contract from
    # an earlier chunk that this one did not restate.
    raw_operations = _rows(getattr(prediction, "operations", ()))
    operations = [
        op
        for op in (_parse_operation(raw, known_traces=known) for raw in raw_operations)
        if op is not None
    ]

    # Enforced once, and before the assignment ids are validated. A REVISE that
    # renamed its contract leaves this chunk's assignments pointing at an id
    # that enforcement is about to discard, so the raw ids are rewritten through
    # the alias map first.
    limitations.extend(state.enforce_revisions(operations, contracts))

    available = {**state.contracts, **{c.contract_id: c for c in contracts}}
    chunk_ids = set(trace_ids)
    raw_assignments = _decoded(getattr(prediction, "assignments", None))
    if state.revision_aliases and isinstance(raw_assignments, dict):
        raw_assignments = {
            trace_id: state.revision_aliases.get(contract_id, contract_id)
            if isinstance(contract_id, str)
            else contract_id
            for trace_id, contract_id in raw_assignments.items()
        }
    assignments = _parse_assignments(
        raw_assignments,
        chunk_ids=chunk_ids,
        contract_ids=set(available),
    )
    ambiguous = _string_tuple(
        _rows(getattr(prediction, "ambiguous_trace_ids", ())), allowed=chunk_ids
    )
    uncovered = _string_tuple(
        _rows(getattr(prediction, "uncovered_trace_ids", ())), allowed=chunk_ids
    )
    # A trace cannot be both assigned and unplaced. The unplaced claim wins: it
    # is the more conservative reading, and forcing a match is the one thing
    # this stage must never do.
    unplaced = set(ambiguous) | set(uncovered)
    assignments = {t: c for t, c in assignments.items() if t not in unplaced}

    # A trace this chunk read but named in none of assignments, ambiguous or
    # uncovered did not vanish - the model omitted it. Silence must not read as
    # coverage: an omitted trace goes to uncovered instead of disappearing.
    omitted = [t for t in trace_ids if t not in assignments and t not in unplaced]
    if omitted:
        limitations.append(
            f"chunk {index} did not report {len(omitted)} trace(s) it read "
            f"({', '.join(omitted)}); marked uncovered rather than dropped"
        )
        uncovered = uncovered + tuple(omitted)

    # An operation naming a contract nobody proposed and nobody already holds is
    # a claim the model did not carry out. Recorded rather than executed.
    holdable = set(available)
    for op in operations:
        unknown = [cid for cid in op.contract_ids if cid not in holdable]
        if unknown:
            limitations.append(
                f"chunk {index} recorded a {op.operation.value} naming contract(s) "
                f"{', '.join(sorted(unknown))} that it never proposed and the taxonomy "
                "does not hold; the operation was logged but could not be applied"
            )

    state.contracts.update({c.contract_id: c for c in contracts})
    duration = time.monotonic() - started
    return ChunkResult(
        index=index,
        pass_index=pass_index,
        trace_ids=trace_ids,
        operations=tuple(operations),
        assignments=assignments,
        ambiguous_trace_ids=ambiguous,
        uncovered_trace_ids=tuple(t for t in uncovered if t not in set(ambiguous)),
        llm_calls=calls,
        tokens=tokens,
        cost_usd=cost,
        duration_seconds=duration,
        raw_reply=_raw_reply(prediction),
        dropped_contracts=tuple(dropped_contracts),
        **provenance,
    )


_REPAIR_INSTRUCTION = """Your last answer was rejected. Every contract MUST have a \
non-empty `definition` and a non-empty `required_outcome_shape`: a list of \
statements someone could check once the work is done. A name plus a description \
is a topic, and a topic cannot be verified.

Return corrected contracts only. Do not reconsider the taxonomy operations or \
the trace assignments from your last answer; those were accepted and anything \
you return for them is discarded. Keep what was right about each contract and \
add what was missing. Omit any that genuinely cannot be given a checkable \
outcome. Rejected:"""
"""Deliberately short. This rides in its own input field, and a field under a
thousand characters is shown to the root model whole rather than as a peek."""


def _repair_contracts(
    rejected: list[str],
    *,
    chunk_json: str,
    taxonomy_json: str,
    predict: _Predictor,
    known: set[str],
) -> list[FamilyContract]:
    """Ask the model to fix contracts that failed validation.

    Best effort by design: a repair that fails costs one attempt and the
    originals are still recorded verbatim, so nothing is lost that was not
    already lost.

    One *attempt*, not one model call: the repair is a whole RLM run, so it may
    make several. Its spend is re-read into the chunk afterwards rather than
    being left out of the budget.
    """
    try:
        prediction = predict(
            chunk=chunk_json,
            taxonomy=taxonomy_json,
            # Truncated per contract: the point is to show the model the shape
            # it got wrong, and a long payload would push this past the peek.
            question=_REPAIR_INSTRUCTION + "\n" + "\n".join(item[:300] for item in rejected[:4]),
        )
    except ledger.LedgerWriteError:
        raise
    except Exception:  # noqa: BLE001 - a failed repair must not lose the chunk
        return []
    if _completion_of(predict).get("mode") == "extract":
        # A repair that ran out of iterations is fallback output too.
        return []
    repaired = []
    for raw in _rows(getattr(prediction, "contracts", ())):
        parsed = _parse_contract(raw, known_traces=known)
        if parsed is not None:
            repaired.append(parsed)
    return repaired


def compute_run_id(run: RLMClusteringRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-clustering-run-{digest[:16]}"


def save_clustering_run(run: RLMClusteringRun, store: DerivedStore) -> DerivedEnvelope:
    """Persist beside the analysis it was mined from, never onto it."""
    return store.write(
        compute_run_id(run),
        kind="rlm_clustering_run",
        parent_artifact_id=run.analysis_id,
        payload=run.model_dump_json().encode("utf-8"),
        summary={
            "contracts": len(run.contracts),
            "chunks": len(run.chunks),
            "ambiguous": len(run.ambiguous_trace_ids),
            "uncovered": len(run.uncovered_trace_ids),
            "complete": int(run.complete),
        },
    )


def load_clustering_run(run_id: str, store: DerivedStore) -> RLMClusteringRun:
    return RLMClusteringRun.model_validate_json(store.read_payload(run_id))
