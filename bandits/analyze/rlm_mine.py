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
draft for the rest of its life.

Chunk boundaries are never family boundaries. Each chunk deliberately mixes
unseen traces with ambiguous ones, traces touched by recent changes, and random
previously-assigned traces, so that a family cannot be an artifact of which
twenty traces happened to be read together.

Nothing here is deterministic — the root model writes its own code each run —
which is why the plan calls for five independent runs and compares them by trace
co-assignment rather than by the names they generate.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from bandits import ledger
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
    StopReason,
    TaxonomyDraft,
    TaxonomyOperation,
    TraceView,
)
from bandits.store import DerivedEnvelope, DerivedStore

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
"""Matches the family audit's default, so one credential covers both paths."""

DEFAULT_CHUNK_SIZE = 20
DEFAULT_SEED = 42
PROMPT_VERSION = 2

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


_INSTRUCTION_HEAD = """You are discovering reusable task families from raw agent traces.

{view}

The variable `chunk` is a list of dicts, each with keys: trace_id, messages (the \
episode as you may read it, in order), and status (one of "unseen", "ambiguous", \
"uncovered", "assigned", "affected"). The variable `taxonomy` is the list of \
family contracts you have built so far, possibly empty.

Two traces belong to the SAME family when one parameterized verifier contract \
could correctly evaluate what both users requested. They belong to DIFFERENT \
families when checking success would require materially different work — even \
if the domain, the wording, and the verb are the same. Sharing a topic is not \
sharing a family.

Decision examples (the examples illustrate the test, not domain categories):
- "Reset my password" and "I forgot my password" belong together: both can be \
verified by establishing that the requested account has a usable new password.
- "Reset my password" and "Change my account email" do not belong together: \
both concern an account, but their required outcomes are different.
- "Summarize this report in three bullets" and "Summarize this report in one \
paragraph" may share one contract parameterized by requested output format.

For every contract you propose or keep, state:
- name: a short imperative task-family name.
- definition: what user-requested work belongs here.
- inclusion_rules / exclusion_rules: what admits and excludes a member.
- required_outcome_shape: what a verifier must establish for EVERY member. A \
contract with no stated outcome shape is a topic, not a family; do not propose one.
- supporting_trace_ids / counterexample_trace_ids: evidence from what you have read.

Record every change as an operation with a rationale and the trace_ids that \
motivated it. Valid operations: KEEP, CREATE, REVISE, SPLIT, MERGE, \
MARK_AMBIGUOUS, MARK_UNCOVERED. Mark a trace ambiguous when several contracts \
fit it equally and uncovered when none does. Never force a trace into a family \
to avoid leaving it unplaced; an unplaced trace is a finding about the taxonomy.

For example, retaining contract c1 unchanged uses {{"operation": "KEEP", \
"contract_ids": ["c1"], "trace_ids": ["t1"], "rationale": "..."}}. Merging \
c1 and c2 into c3 uses contract_ids ["c1", "c2", "c3"], with the produced \
contract last.

Assign every trace in this chunk, including ones you have seen before: re-reading \
an old trace against changed definitions is the point of the loop."""


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


class MiningError(RuntimeError):
    """The miner could not be built or returned nothing usable."""


class _Predictor(Protocol):
    """The one call this module makes, so tests need no model and no sandbox."""

    def __call__(self, *, chunk: str, taxonomy: str, question: str) -> Any: ...


def prompt_digest(model: str) -> str:
    """Pins wording, model and version onto every draft they produced."""
    payload = json.dumps(
        {
            "instruction": _INSTRUCTION_HEAD,
            "views": {v.value: text for v, text in VIEW_PREAMBLES.items()},
            "model": model,
            "version": PROMPT_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_predictor(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    view: TraceView = TraceView.USER_MESSAGES,
    max_iterations: int = 25,
    max_llm_calls: int = 60,
) -> _Predictor:
    """A ``dspy.RLM`` over one chunk, imported only when mining actually runs.

    The per-chunk ceilings are what the real model needed rather than what
    seemed reasonable: measured against tau2 airline requests, one chunk of
    three traces took 5 to 18 calls, and a chunk that hit the old 12-iteration
    limit fell back to DSPy's ``extract`` and returned a partial answer. Real
    customer requests are long and ask for several things at once, so the root
    model slices them more than a short instruction would.

    DSPy and its REPL sandbox are an optional extra: the core install stays at
    three runtime dependencies, and the tests below run against an injected
    predictor rather than a model.
    """
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise MiningError("RLM mining needs the 'audit' extra: uv sync --extra audit") from exc

    from bandits.analyze.audit import scoped_to_history
    from bandits.verify.judge import resolve_api_key

    key = api_key or resolve_api_key()
    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=key,
        # The root model writes code rather than prose, and a sampled plan
        # rereads the same chunk differently for no gain a reviewer can use.
        # Run-to-run variation is measured across seeds, not sampled per call.
        temperature=0.0,
    )

    # Typed rather than list[dict]: the decoder then enforces the fields, and a
    # topic object cannot be submitted as a valid contract in the first place.
    # Built from real classes rather than a signature string, because DSPy
    # resolves names in a string against its own namespace and cannot see these.
    class _Mine(dspy.Signature):
        chunk: str = dspy.InputField(desc="the traces to read this round")
        taxonomy: str = dspy.InputField(desc="contracts built so far, possibly empty")
        contracts: list[ProposedContract] = dspy.OutputField()
        operations: list[ProposedOperation] = dspy.OutputField()
        assignments: dict[str, str] = dspy.OutputField()
        ambiguous_trace_ids: list[str] = dspy.OutputField()
        uncovered_trace_ids: list[str] = dspy.OutputField()

    # The query, in the RLM's sense: it belongs in the token window, not in a
    # REPL variable the model has to remember to print.
    _Mine.__doc__ = instruction_for(view)
    rlm = dspy.RLM(
        _Mine,
        max_iters=max_iterations,
        max_llm_calls=max_llm_calls,
        sub_lm=language_model,
    )

    def predict(*, chunk: str, taxonomy: str, question: str = "") -> Any:
        # ``question`` carries a correction when one is being asked for, and is
        # empty on an ordinary call. It is appended to the taxonomy variable
        # rather than dropped: the standing instructions live on the signature
        # now, so there is no input field left to put it in, and a repair
        # request that reached nothing would silently re-send the original
        # question and be charged for the same answer twice.
        with dspy.context(lm=language_model):
            if question:
                return rlm(chunk=chunk, taxonomy=f"{taxonomy}\n\n{question}")
            return rlm(chunk=chunk, taxonomy=taxonomy)

    return with_cost(scoped_to_history(predict, language_model))


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

    def apply(self, result: ChunkResult, contracts: Sequence[FamilyContract]) -> int:
        """Fold one chunk's output in, and report how many assignments moved.

        The churn figure is the count of traces whose *previous* placement
        changed — not the count of assignments made. A trace placed for the
        first time has not moved; counting it as movement would make an early
        sweep over unseen traces look unstable by construction.
        """
        self.apply_operations(result.operations, contracts)
        for contract in contracts:
            self.contracts[contract.contract_id] = contract

        moved = 0
        for trace_id, contract_id in result.assignments.items():
            previous = self.assignments.get(trace_id)
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
    budget: Budget, *, iterations: int, calls: int, started: float, usd: float
) -> StopReason | None:
    """Which ceiling, if any, this run has hit.

    Checked before each chunk rather than after, so a run never spends past a
    limit it has already reached.
    """
    if iterations >= budget.max_iterations:
        return StopReason.MAX_ITERATIONS
    if calls >= budget.max_llm_calls:
        return StopReason.MAX_LLM_CALLS
    if time.monotonic() - started >= budget.max_seconds:
        return StopReason.MAX_SECONDS
    if budget.max_usd is not None and usd >= budget.max_usd:
        return StopReason.MAX_USD
    return None


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
) -> TaxonomyDraft:
    """Read the corpus through complete passes, then pause for review.

    Each pass reads every eligible trace exactly once, in its own shuffled
    order. A pass counts toward the schedule only when it finished; a pass cut
    short by a budget guard counts for nothing, however much of the corpus it
    got through.

    Nothing here decides the taxonomy has converged, because nothing here can.
    The run stops when it has done the passes it was asked for, and hands back a
    draft plus the diff between those passes for a person to judge. ``session``,
    when given, is checkpointed after every chunk so a run that dies mid-pass is
    resumable and a run in flight is observable.
    """
    budget = budget or Budget()
    state = _TaxonomyState()

    # A resumed run starts from the workspace the crash left rather than from
    # nothing. Restoring the taxonomy and the placements is most of it; the rest
    # is knowing which pass was in flight, which traces it had already read, and
    # the order it was reading them in — without that last one a resume would
    # reshuffle, reread some traces and never reach others.
    resume_pass = 0
    resume_seen: set[str] = set()
    resume_order: tuple[str, ...] = ()
    if resume is not None:
        state.contracts = {c.contract_id: c for c in resume.contracts}
        state.assignments = dict(resume.assignments)
        state.ambiguous = set(resume.ambiguous_trace_ids)
        state.uncovered = set(resume.uncovered_trace_ids)
        state.seen = set(resume.assignments) | state.ambiguous | state.uncovered
        resume_pass = resume.pass_index
        resume_seen = set(resume.seen_this_pass)
        resume_order = resume.pass_order

    eligible = list(corpus.readable_trace_ids())
    unreadable = corpus.unreadable_trace_ids()
    if not eligible:
        raise MiningError(
            "no trace in this corpus has readable user messages; there is nothing to mine"
        )

    chunks: list[ChunkResult] = []
    passes: list[PassResult] = []
    limitations: list[str] = []
    calls = 0
    usd = 0.0
    cost_reported = False
    started = time.monotonic()
    stop_reason: StopReason | None = None
    completed_passes = resume.completed_passes if resume is not None else 0

    if session is not None:
        session.begin(
            traces_total=len(eligible), requested_passes=budget.passes, seed=seed
        )

    for pass_index in range(resume_pass, budget.passes):
        # Reshuffled per pass with a seed derived from the run's, so each pass
        # reads the corpus in a different order — chunk composition cannot
        # become family structure — while staying reproducible from the draft.
        pass_seed = seed + pass_index
        if pass_index == resume_pass and resume_order:
            # The order the interrupted pass was working through, so what it had
            # already read stays read and what it had not is what gets finished.
            order = [tid for tid in resume_order if tid in set(eligible)]
        else:
            order = list(eligible)
            random.Random(pass_seed).shuffle(order)

        previous_placement = dict(state.assignments)
        contracts_before = tuple(sorted(state.contracts))
        pass_operations: list[TaxonomyOperation] = []
        pass_chunk_indices: list[int] = []
        # The pass's own progress, reset here. This is the fix: the old loop
        # tracked traces seen across the whole run, so once every trace had been
        # read once the "have we swept" test was permanently true and two quiet
        # chunks could end the run.
        seen_this_pass: set[str] = set(resume_seen) if pass_index == resume_pass else set()
        pass_complete = True

        failed_this_pass: set[str] = set()
        retried = False
        while True:
            remaining = [tid for tid in order if tid not in seen_this_pass]
            if not remaining:
                break
            if remaining and set(remaining) == failed_this_pass:
                # Everything left has already failed once. Retry the batch a
                # single time, then stop: a provider that is down stays down,
                # and looping would burn the budget without reading anything.
                if retried:
                    pass_complete = False
                    limitations.append(
                        f"pass {pass_index} could not read {len(failed_this_pass)} trace(s) "
                        "after a retry, so it did not cover the corpus and does not count "
                        "toward the requested passes"
                    )
                    break
                retried = True
                failed_this_pass.clear()

            stop_reason = _budget_stop(
                budget, iterations=len(chunks), calls=calls, started=started, usd=usd
            )
            if stop_reason is not None:
                # A guard fired mid-pass. The pass is partial and must not count
                # toward the schedule, however many traces it happened to read.
                pass_complete = False
                break

            trace_ids, statuses = _compose_chunk(
                state, unseen=remaining, chunk_size=chunk_size, seen_this_pass=seen_this_pass
            )
            if not trace_ids:  # pragma: no cover - remaining is non-empty here
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
            )
            chunks.append(result)
            pass_chunk_indices.append(result.index)
            if result.status == "error":
                # Nothing was read, so these traces are still owed to this pass.
                # Counting them would let a provider outage silently shrink a
                # pass's coverage while the run still called the pass complete.
                failed_this_pass.update(trace_ids)
            else:
                seen_this_pass.update(trace_ids)
            calls += result.llm_calls or 0
            if result.cost_usd is not None:
                usd += result.cost_usd
                cost_reported = True

            if result.status != "error":
                contracts = [state.contracts[cid] for cid in result.assignments.values()]
                state.apply(result, contracts)
                pass_operations.extend(result.operations)

            if on_chunk is not None:
                on_chunk(result)
            # Written after every chunk, not every pass: a run that dies at
            # chunk three of pass two must be resumable from chunk three, and a
            # person watching must never be more than one model call behind.
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

        reassigned = tuple(
            sorted(
                trace_id
                for trace_id, contract_id in state.assignments.items()
                if trace_id in previous_placement
                and previous_placement[trace_id] != contract_id
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
            # The only place this increments, and only after every eligible
            # trace has appeared in this pass. Two clean chunks are not a pass.
            completed_passes += 1
        else:
            break

    if stop_reason is None:
        # The schedule finished only if every requested pass actually completed.
        # A pass cut short by repeated chunk failures leaves the run short of
        # what it was asked for, and must not report the same stop reason as a
        # run that read the whole corpus the requested number of times.
        stop_reason = (
            StopReason.PASSES_COMPLETE
            if completed_passes >= budget.passes
            else StopReason.ERROR
        )

    if stop_reason is not StopReason.PASSES_COMPLETE:
        limitations.append(
            f"discovery stopped on {stop_reason.value} before completing its "
            f"{budget.passes} requested pass(es); {completed_passes} pass(es) read every "
            "eligible trace, so the rest of the corpus was seen fewer times than planned"
        )
    else:
        # Said on every complete run, because the previous stop reason was named
        # "converged" and invited exactly the inference this denies.
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
    if budget.max_usd is not None and not cost_reported:
        # Loud, because the user asked for a spending limit and did not get one.
        limitations.append(
            f"a monetary ceiling of ${budget.max_usd:.2f} was requested but no backend "
            "reported a cost, so the run was bounded only by its call, iteration and "
            "time limits"
        )
    if corpus.view.reads_agent_behavior:
        # Stated on every artifact this arm produces, because the families it
        # finds are exactly the ones that cannot be taken at face value: a group
        # that reads as a task family may be a group of episodes the agent
        # handled alike. Nothing here can tell the difference, which is why the
        # plan measures membership against tool usage and outcome afterwards.
        limitations.append(
            "this taxonomy was mined from full trajectories, so the miner could see "
            "what the agent did; its families must be checked against tool usage, "
            "trajectory length and outcome before being read as task families"
        )
        # The check that does not depend on guessing field names. Empty is the
        # only acceptable result; anything here means the arm was not blind.
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
            # Not reassuring. A scored corpus that yielded nothing to strip is
            # more likely to be naming its rewards something unforeseen than to
            # be carrying none.
            limitations.append(
                "no outcome-bearing field was found to strip from the trajectory view; "
                "if this corpus records rewards under other names they reached the miner"
            )

    return TaxonomyDraft(
        analysis_id=analysis_id,
        view=corpus.view,
        seed=seed,
        contracts=tuple(sorted(state.contracts.values(), key=lambda c: c.contract_id)),
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
    )


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
) -> ChunkResult:
    """One call over one chunk, with its output cleaned at the boundary."""
    chunk_json = _chunk_payload(corpus, trace_ids, statuses)
    taxonomy_json = _taxonomy_payload(state)
    started = time.monotonic()
    try:
        with ledger.stage(
            "rlm_chunk",
            chunk_index=index,
            pass_index=pass_index,
            session_id=session_id,
            traces=len(trace_ids),
        ):
            prediction = predict(
                chunk=chunk_json,
                taxonomy=taxonomy_json,
                question=instruction_for(corpus.view),
            )
    except Exception as exc:  # noqa: BLE001 - a failed chunk is recorded, not fatal
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
            error=str(exc),
        )
    duration = time.monotonic() - started
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
        # discarding work it already paid to produce.
        repaired = _repair_contracts(
            dropped_contracts,
            chunk_json=chunk_json,
            taxonomy_json=taxonomy_json,
            corpus=corpus,
            predict=predict,
            known=known,
        )
        if repaired:
            contracts.extend(repaired)
            dropped_contracts = dropped_contracts[len(repaired) :]
            limitations.append(
                f"chunk {index} returned {len(repaired)} contract(s) that failed "
                "validation and were corrected on a second attempt"
            )
    if dropped_contracts:
        limitations.append(
            f"chunk {index} proposed {len(dropped_contracts)} contract(s) the parser "
            "refused; see the chunk's dropped_contracts for exactly what was returned"
        )

    # Merged with what already exists so an assignment may name a contract from
    # an earlier chunk that this one did not restate.
    available = {**state.contracts, **{c.contract_id: c for c in contracts}}
    raw_operations = _rows(getattr(prediction, "operations", ()))
    operations = [
        op
        for op in (_parse_operation(raw, known_traces=known) for raw in raw_operations)
        if op is not None
    ]
    chunk_ids = set(trace_ids)
    assignments = _parse_assignments(
        _decoded(getattr(prediction, "assignments", None)),
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

    # An operation naming a contract nobody proposed and nobody already holds is
    # a claim the model did not carry out. Recorded rather than executed: it
    # would otherwise count toward convergence while changing nothing.
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
    )


_REPAIR_INSTRUCTION = """Some contracts you returned were rejected because they \
were missing a required field. Every contract MUST carry a non-empty `definition` \
and a non-empty `required_outcome_shape` — a list of statements a verifier could \
check once the work is done. A contract with only a name and a description names \
a topic, and a topic cannot be verified.

Return corrected versions of the rejected contracts below. Keep what was right \
about them and add what was missing. Return nothing for any that genuinely cannot \
be given a checkable outcome."""


def _repair_contracts(
    rejected: list[str],
    *,
    chunk_json: str,
    taxonomy_json: str,
    corpus: ReadOnlyCorpus,
    predict: _Predictor,
    known: set[str],
) -> list[FamilyContract]:
    """Ask the model to fix contracts that failed validation.

    Best effort by design: a repair that fails costs one call and the originals
    are still recorded verbatim, so nothing is lost that was not already lost.
    """
    try:
        prediction = predict(
            chunk=chunk_json,
            taxonomy=taxonomy_json,
            question=_REPAIR_INSTRUCTION + "\n\nRejected contracts:\n" + "\n".join(rejected),
        )
    except Exception:  # noqa: BLE001 - a failed repair must not lose the chunk
        return []
    repaired = []
    for raw in _rows(getattr(prediction, "contracts", ())):
        parsed = _parse_contract(raw, known_traces=known)
        if parsed is not None:
            repaired.append(parsed)
    return repaired


def compute_draft_id(draft: TaxonomyDraft) -> str:
    digest = hashlib.sha256(draft.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-taxonomy-draft-{digest[:16]}"


def save_draft(draft: TaxonomyDraft, store: DerivedStore) -> DerivedEnvelope:
    """Persist beside the analysis it was mined from, never onto it."""
    return store.write(
        compute_draft_id(draft),
        kind="rlm_taxonomy_draft",
        parent_artifact_id=draft.analysis_id,
        payload=draft.model_dump_json().encode("utf-8"),
        summary={
            "contracts": len(draft.contracts),
            "chunks": len(draft.chunks),
            "ambiguous": len(draft.ambiguous_trace_ids),
            "uncovered": len(draft.uncovered_trace_ids),
            "complete": int(draft.complete),
        },
    )


def load_draft(draft_id: str, store: DerivedStore) -> TaxonomyDraft:
    return TaxonomyDraft.model_validate_json(store.read_payload(draft_id))
