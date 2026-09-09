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
PROMPT_VERSION = 1

CLEAN_SWEEPS_TO_FREEZE = 2
"""Consecutive clean sweeps required before a taxonomy may freeze.

Two rather than one: a single quiet sweep is as easily explained by an
unrepresentative chunk as by convergence, and the cost of one more pass is far
below the cost of freezing a taxonomy that was still moving.
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

Assign every trace in this chunk, including ones you have seen before: re-reading \
an old trace against changed definitions is the point of the loop."""


def instruction_for(view: TraceView) -> str:
    """The mining prompt as this arm's miner actually receives it."""
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
    max_iterations: int = 12,
    max_llm_calls: int = 40,
) -> _Predictor:
    """A ``dspy.RLM`` over one chunk, imported only when mining actually runs.

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

    signature = (
        "chunk: str, taxonomy: str, question: str -> contracts: list[dict], "
        "operations: list[dict], assignments: dict[str, str], "
        "ambiguous_trace_ids: list[str], uncovered_trace_ids: list[str]"
    )
    rlm = dspy.RLM(
        signature,
        max_iters=max_iterations,
        max_llm_calls=max_llm_calls,
        sub_lm=language_model,
    )

    def predict(*, chunk: str, taxonomy: str, question: str) -> Any:
        with dspy.context(lm=language_model):
            return rlm(chunk=chunk, taxonomy=taxonomy, question=question)

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


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_tuple(raw: Any, *, allowed: set[str] | None = None) -> tuple[str, ...]:
    """Clean a model-written list of strings, dropping repeats and unknown ids.

    A model writes these. One that hallucinates a trace id, or repeats one,
    would otherwise fail a contract validator and lose the whole chunk —
    including the parts it got right — so unknown values are dropped here and
    the drop is reported as a limitation by the caller.
    """
    if not isinstance(raw, (list, tuple)):
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


def _parse_contract(raw: Any, *, known_traces: set[str]) -> FamilyContract | None:
    """Build one contract from a model-written dict, or nothing.

    Returns ``None`` rather than raising for anything malformed. A contract the
    model failed to argue for — no outcome shape, no definition — is dropped and
    counted, because keeping it would put a topic into a taxonomy that is
    supposed to hold only verifiable claims.
    """
    if not isinstance(raw, dict):
        return None
    contract_id = _text(raw.get("contract_id")) or _text(raw.get("id"))
    name = _text(raw.get("name"))
    definition = _text(raw.get("definition"))
    outcome = _string_tuple(raw.get("required_outcome_shape"))
    if not (name and definition and outcome):
        return None
    if not contract_id:
        # Derived from the claim rather than invented as a counter, so the same
        # contract proposed twice in one run lands on one id instead of two.
        contract_id = f"contract-{hashlib.sha256(definition.lower().encode()).hexdigest()[:12]}"

    supporting = _string_tuple(raw.get("supporting_trace_ids"), allowed=known_traces)
    counter = _string_tuple(raw.get("counterexample_trace_ids"), allowed=known_traces)
    revision = raw.get("revision")
    try:
        return FamilyContract(
            contract_id=contract_id,
            name=name,
            definition=definition,
            inclusion_rules=_string_tuple(raw.get("inclusion_rules")),
            exclusion_rules=_string_tuple(raw.get("exclusion_rules")),
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
    if not isinstance(raw, dict):
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
    rng: random.Random,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Pick the next chunk, mixing the five categories the plan requires.

    Unseen traces come first and take at least half the chunk while any remain,
    so a corpus is actually covered rather than endlessly re-litigated. The rest
    is filled with unresolved cases, traces the last operations touched, and a
    random sample of settled ones — the last being the only way a definition
    that drifted is caught against a trace nobody was worried about.
    """
    statuses: dict[str, str] = {}
    picked: list[str] = []

    def take(candidates: Sequence[str], status: str, room: int) -> None:
        for trace_id in candidates:
            if room <= 0:
                return
            if trace_id in statuses:
                continue
            statuses[trace_id] = status
            picked.append(trace_id)
            room -= 1

    if unseen:
        take(unseen[: max(1, chunk_size // 2)], "unseen", chunk_size)

    take(sorted(state.ambiguous), "ambiguous", chunk_size - len(picked))
    take(sorted(state.uncovered), "uncovered", chunk_size - len(picked))
    take(sorted(state.recently_affected), "affected", chunk_size - len(picked))

    # Any remaining room goes back to unseen traces before it goes to review, so
    # a corpus with many unresolved cases still finishes its first pass.
    take(unseen, "unseen", chunk_size - len(picked))

    settled = sorted(set(state.assignments) - set(statuses))
    rng.shuffle(settled)
    take(settled, "assigned", chunk_size - len(picked))
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
) -> TaxonomyDraft:
    """Run the discovery loop until it converges or a budget stops it.

    Returns a draft either way. A draft is never a taxonomy and never a task
    set: it records how it stopped, and only a run that went two full sweeps
    without changing anything is marked complete.
    """
    budget = budget or Budget()
    rng = random.Random(seed)
    state = _TaxonomyState()

    readable = list(corpus.readable_trace_ids())
    unreadable = corpus.unreadable_trace_ids()
    if not readable:
        raise MiningError(
            "no trace in this corpus has readable user messages; there is nothing to mine"
        )
    # Shuffled with the recorded seed so chunk composition is reproducible from
    # the draft alone, and so corpus order cannot become family structure.
    rng.shuffle(readable)

    chunks: list[ChunkResult] = []
    limitations: list[str] = []
    clean_sweeps = 0
    calls = 0
    usd = 0.0
    cost_reported = False
    """Whether any chunk's backend reported a price at all.

    Tracked because an unreported cost and a zero cost are different facts, and
    only one of them means ``--max-usd`` was actually enforced.
    """
    started = time.monotonic()
    stop_reason: StopReason | None = None

    while True:
        stop_reason = _budget_stop(
            budget, iterations=len(chunks), calls=calls, started=started, usd=usd
        )
        if stop_reason is not None:
            break

        unseen = [trace_id for trace_id in readable if trace_id not in state.seen]
        if not unseen and clean_sweeps >= CLEAN_SWEEPS_TO_FREEZE:
            stop_reason = StopReason.CONVERGED
            break

        trace_ids, statuses = _compose_chunk(state, unseen=unseen, chunk_size=chunk_size, rng=rng)
        if not trace_ids:
            # Nothing left to look at and convergence not yet earned: the sweep
            # requirement cannot be met, so this is a limit, not a success.
            stop_reason = StopReason.CONVERGED if not unseen else StopReason.ERROR
            break

        result = _run_chunk(
            corpus,
            state,
            trace_ids=trace_ids,
            statuses=statuses,
            index=len(chunks),
            predict=predict,
            limitations=limitations,
        )
        chunks.append(result)
        if on_chunk is not None:
            on_chunk(result)

        if result.cost_usd is not None:
            usd += result.cost_usd
            cost_reported = True

        if result.status == "error":
            # A failed chunk is recorded and the loop continues: one bad call
            # must not lose a taxonomy that is otherwise progressing. It cannot
            # count toward convergence, since nothing was read. Its spend still
            # counts: a failed call was billed like any other.
            clean_sweeps = 0
            calls += result.llm_calls or 0
            continue

        contracts = [state.contracts[cid] for cid in result.assignments.values()]
        moved = state.apply(result, contracts)
        calls += result.llm_calls or 0

        swept = not [tid for tid in readable if tid not in state.seen]
        churn = moved / max(len(state.assignments), 1)
        if swept and not result.mutated and churn < MAX_ASSIGNMENT_CHURN:
            clean_sweeps += 1
        else:
            clean_sweeps = 0

    if stop_reason is not StopReason.CONVERGED:
        limitations.append(
            f"discovery stopped on {stop_reason.value} rather than converging; this "
            "taxonomy was still changing when the run ended and must not be read as final"
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
    predict: _Predictor,
    limitations: list[str],
) -> ChunkResult:
    """One call over one chunk, with its output cleaned at the boundary."""
    chunk_json = _chunk_payload(corpus, trace_ids, statuses)
    taxonomy_json = _taxonomy_payload(state)
    started = time.monotonic()
    try:
        with ledger.stage("rlm_chunk", chunk_index=index, traces=len(trace_ids)):
            prediction = predict(
                chunk=chunk_json,
                taxonomy=taxonomy_json,
                question=instruction_for(corpus.view),
            )
    except Exception as exc:  # noqa: BLE001 - a failed chunk is recorded, not fatal
        calls, tokens = _spend_of(predict)
        return ChunkResult(
            index=index,
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
    contracts = [
        contract
        for contract in (
            _parse_contract(raw, known_traces=known)
            for raw in getattr(prediction, "contracts", ()) or ()
        )
        if contract is not None
    ]
    dropped = len(list(getattr(prediction, "contracts", ()) or ())) - len(contracts)
    if dropped > 0:
        limitations.append(
            f"chunk {index} proposed {dropped} contract(s) with no definition or no "
            "required outcome shape; they name topics rather than families and were dropped"
        )

    # Merged with what already exists so an assignment may name a contract from
    # an earlier chunk that this one did not restate.
    available = {**state.contracts, **{c.contract_id: c for c in contracts}}
    operations = [
        op
        for op in (
            _parse_operation(raw, known_traces=known)
            for raw in getattr(prediction, "operations", ()) or ()
        )
        if op is not None
    ]
    chunk_ids = set(trace_ids)
    assignments = _parse_assignments(
        getattr(prediction, "assignments", None),
        chunk_ids=chunk_ids,
        contract_ids=set(available),
    )
    ambiguous = _string_tuple(getattr(prediction, "ambiguous_trace_ids", ()), allowed=chunk_ids)
    uncovered = _string_tuple(getattr(prediction, "uncovered_trace_ids", ()), allowed=chunk_ids)
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
        trace_ids=trace_ids,
        operations=tuple(operations),
        assignments=assignments,
        ambiguous_trace_ids=ambiguous,
        uncovered_trace_ids=tuple(t for t in uncovered if t not in set(ambiguous)),
        llm_calls=calls,
        tokens=tokens,
        cost_usd=cost,
        duration_seconds=duration,
    )


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
