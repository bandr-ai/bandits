"""Classify every trace against a frozen taxonomy, in a context that never mined it.

Why this exists as a separate stage. Assignments made during discovery were made
against definitions that kept changing for the rest of the loop: a trace placed
in chunk one was judged against wording that chunk six revised, and nobody went
back. Those assignments are provisional by construction, and treating them as
the result would report a membership no single version of the taxonomy ever
actually claimed.

So the taxonomy is frozen, content-addressed, and handed to a fresh context that
never watched it being argued into existence. That context may not change it:
there is no operation here that creates, splits or revises a contract, so a
trace that fits nothing produces an ``uncovered`` row rather than a new family
invented to house it.

Never force a match. A trace matching two contracts stays ambiguous and a trace
matching none stays uncovered — both are findings about the taxonomy, and a
tiebreak here would convert a real gap into a confident-looking placement.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from bandits import ledger
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_models import (
    VIEW_PREAMBLES,
    AssignmentRun,
    AssignmentStatus,
    FrozenTaxonomy,
    TraceAssignment,
    TraceView,
)
from bandits.store import DerivedEnvelope, DerivedStore

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
DEFAULT_BATCH_SIZE = 20
PROMPT_VERSION = 1

_INSTRUCTION_HEAD = """You are classifying agent traces against a FIXED set of task-family \
contracts. You may not create, split, merge, or reword any contract. If nothing \
fits, say nothing fits.

{view}

The variable `taxonomy` is the fixed list of contracts. The variable `batch` is a \
list of dicts with keys trace_id and messages.

A trace matches a contract when that contract's parameterized verifier could \
correctly evaluate what this user requested — check it against the contract's \
required_outcome_shape, inclusion_rules and exclusion_rules, not against its name.

For every trace in the batch return one result with:
- trace_id
- matching_contract_ids: every contract that genuinely matches. Often exactly one. \
Empty if none does.
- primary_contract_id: the single best match, ONLY when exactly one contract \
matches. Null when zero or several match.
- status: "assigned" (exactly one match), "ambiguous" (two or more match equally), \
or "uncovered" (none matches).
- reason: one or two sentences citing what in the user's request decided it.

Do not force a trace into a family to avoid leaving it unplaced. An uncovered \
trace is useful information about the taxonomy; a wrong assignment is not."""


def instruction_for(view: TraceView) -> str:
    """The assignment prompt as this arm's classifier actually receives it."""
    return _INSTRUCTION_HEAD.format(view=VIEW_PREAMBLES[view])


class AssignmentError(RuntimeError):
    """The assigner could not be built or returned nothing usable."""


class _Predictor(Protocol):
    def __call__(self, *, taxonomy: str, batch: str, question: str) -> Any: ...


def prompt_digest(model: str) -> str:
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
    max_iterations: int = 12,
    max_llm_calls: int = 30,
) -> _Predictor:
    """A ``dspy.RLM`` over one batch, imported only when assignment runs."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise AssignmentError(
            "RLM assignment needs the 'audit' extra: uv sync --extra audit"
        ) from exc

    from bandits.analyze.audit import scoped_to_history
    from bandits.verify.judge import resolve_api_key

    key = api_key or resolve_api_key()
    language_model = dspy.LM(f"fireworks_ai/{model}", api_key=key, temperature=0.0)
    # Instructions on the signature rather than in an input field, so the whole
    # prompt reaches the root model. See rlm_mine.instruction_for for why.
    class _Assign(dspy.Signature):
        taxonomy: str = dspy.InputField(desc="the fixed contracts to classify against")
        batch: str = dspy.InputField(desc="the traces to classify")
        results: list[dict] = dspy.OutputField()

    _Assign.__doc__ = instruction_for(view)
    rlm = dspy.RLM(
        _Assign,
        max_iters=max_iterations,
        max_llm_calls=max_llm_calls,
        sub_lm=language_model,
    )

    def predict(*, taxonomy: str, batch: str, question: str = "") -> Any:
        with dspy.context(lm=language_model):
            return rlm(taxonomy=taxonomy, batch=batch)

    return scoped_to_history(predict, language_model)


def _spend_of(predict: Any) -> tuple[int | None, dict[str, int]]:
    spend = getattr(predict, "spend", None)
    if spend is None:
        return None, {}
    try:
        return spend()
    except Exception:  # noqa: BLE001 - bookkeeping must not lose the assignment
        return None, {}


def _raw_reply(prediction: Any) -> str:
    """Everything the classifier returned for one batch, never truncated."""
    try:
        return json.dumps(
            {"results": getattr(prediction, "results", None)}, indent=2, default=str
        )
    except (TypeError, ValueError):
        return str(prediction)


def _rows(value: Any) -> list[Any]:
    """Decoded model output as a list of rows. See rlm_mine._rows."""
    from bandits.analyze.rlm_mine import _rows as rows

    return rows(value)


def _parse_result(raw: Any, *, known_contracts: set[str]) -> TraceAssignment | None:
    """Build one assignment from a model-written dict, or nothing.

    The status is recomputed from the matches rather than trusted, because the
    two disagree often and the match list is the load-bearing claim: a model
    that says "assigned" while naming two contracts has found an ambiguity and
    mislabelled it, and taking its word would erase exactly the uncertainty this
    stage exists to preserve.
    """
    if not isinstance(raw, dict):
        return None
    trace_id = str(raw.get("trace_id") or "").strip()
    if not trace_id:
        return None

    matches: list[str] = []
    for item in raw.get("matching_contract_ids") or ():
        if isinstance(item, str) and item.strip() in known_contracts:
            if item.strip() not in matches:
                matches.append(item.strip())

    reason = str(raw.get("reason") or "").strip()
    if not matches:
        return TraceAssignment(
            trace_id=trace_id,
            status=AssignmentStatus.UNCOVERED,
            reason=reason or "no contract in the taxonomy matched this request",
        )
    if len(matches) > 1:
        return TraceAssignment(
            trace_id=trace_id,
            matching_contract_ids=tuple(matches),
            status=AssignmentStatus.AMBIGUOUS,
            reason=reason or "several contracts matched this request equally",
        )

    primary = str(raw.get("primary_contract_id") or "").strip()
    return TraceAssignment(
        trace_id=trace_id,
        matching_contract_ids=(matches[0],),
        # A primary naming something outside the single match is a contradiction;
        # the match is what was argued for, so it wins.
        primary_contract_id=matches[0] if primary != matches[0] else primary,
        status=AssignmentStatus.ASSIGNED,
        reason=reason or "one contract matched this request",
    )


def assign_traces(
    taxonomy: FrozenTaxonomy,
    taxonomy_id: str,
    corpus: ReadOnlyCorpus,
    *,
    predict: _Predictor,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> AssignmentRun:
    """Classify every trace in the corpus against the frozen contracts.

    Every trace gets exactly one row. A trace the model never returned a result
    for becomes ``uncovered`` with a reason saying so, because a trace missing
    from the run would silently shrink the denominator that coverage is measured
    against.
    """
    if corpus.view is not taxonomy.view:
        # The two arms are different experiments. Assigning first-message views
        # against contracts mined from full conversations would measure neither.
        raise AssignmentError(
            f"taxonomy was mined from the {taxonomy.view.value} view but this corpus "
            f"exposes {corpus.view.value}; they are different experiments"
        )

    taxonomy_json = json.dumps(
        [c.model_dump(mode="json") for c in taxonomy.contracts], indent=2, sort_keys=True
    )
    known_contracts = {c.contract_id for c in taxonomy.contracts}
    readable = corpus.readable_trace_ids()
    results: dict[str, TraceAssignment] = {}
    raw_replies: list[str] = []
    dropped_results: list[str] = []
    limitations: list[str] = []
    calls: int | None = None
    tokens: dict[str, int] = {}
    started = time.monotonic()

    for offset in range(0, len(readable), batch_size):
        window = readable[offset : offset + batch_size]
        payload = json.dumps(
            [
                {"trace_id": view.trace_id, "messages": list(view.messages)}
                for view in corpus.get_user_message_batch(window)
            ],
            indent=2,
            sort_keys=True,
            default=str,
        )
        try:
            with ledger.stage(
                "rlm_assign",
                taxonomy_id=taxonomy_id,
                batch_offset=offset,
                traces=len(window),
            ):
                prediction = predict(
                    taxonomy=taxonomy_json,
                    batch=payload,
                    question=instruction_for(corpus.view),
                )
        except Exception as exc:  # noqa: BLE001 - a failed batch is recorded, not fatal
            limitations.append(
                f"the batch at offset {offset} failed and its {len(window)} trace(s) are "
                f"recorded as uncovered rather than dropped: {exc}"
            )
            prediction = None

        batch_calls, batch_tokens = _spend_of(predict)
        if batch_calls is not None:
            calls = (calls or 0) + batch_calls
        for field, value in batch_tokens.items():
            tokens[field] = tokens.get(field, 0) + value

        if prediction is not None:
            raw_replies.append(_raw_reply(prediction))
        raw_results = _rows(getattr(prediction, "results", ())) if prediction else []
        for raw in raw_results:
            parsed = _parse_result(raw, known_contracts=known_contracts)
            # Only traces in this batch: a model naming a trace from another
            # batch would overwrite a decision made with different context.
            if parsed is not None and parsed.trace_id in set(window):
                results.setdefault(parsed.trace_id, parsed)
            else:
                # A row the parser refused, or one naming a trace outside this
                # batch. Kept verbatim: a trace reported uncovered because its
                # row failed to parse is not the same finding as one the
                # taxonomy genuinely does not reach.
                dropped_results.append(json.dumps(raw, default=str))

        if on_batch is not None:
            on_batch(offset, len(window))

    missing = [trace_id for trace_id in readable if trace_id not in results]
    for trace_id in missing:
        results[trace_id] = TraceAssignment(
            trace_id=trace_id,
            status=AssignmentStatus.UNCOVERED,
            reason="the assignment pass returned no result for this trace",
        )
    if missing:
        limitations.append(
            f"{len(missing)} trace(s) got no result from the model and are recorded as "
            "uncovered; that is a gap in the run, not evidence the taxonomy misses them"
        )

    for trace_id in corpus.unreadable_trace_ids():
        results[trace_id] = TraceAssignment(
            trace_id=trace_id,
            status=AssignmentStatus.UNREADABLE,
            reason="the source recorded no user messages for this trace",
        )

    ambiguous = sum(1 for a in results.values() if a.status is AssignmentStatus.AMBIGUOUS)
    uncovered = sum(1 for a in results.values() if a.status is AssignmentStatus.UNCOVERED)
    if ambiguous:
        limitations.append(
            f"{ambiguous} trace(s) matched several contracts and were left ambiguous; "
            "they are excluded from automatic verifier drafting until reviewed"
        )
    if dropped_results:
        limitations.append(
            f"{len(dropped_results)} result row(s) could not be read and are kept "
            "verbatim in dropped_results; any trace they named is reported uncovered "
            "because of that, not because the taxonomy misses it"
        )
    if uncovered:
        limitations.append(
            f"{uncovered} trace(s) matched no contract; they are excluded from automatic "
            "verifier drafting until reviewed"
        )

    return AssignmentRun(
        taxonomy_id=taxonomy_id,
        analysis_id=taxonomy.analysis_id,
        view=corpus.view,
        assignments=tuple(sorted(results.values(), key=lambda a: a.trace_id)),
        model=model,
        prompt_digest=prompt_digest(model),
        llm_calls=calls,
        tokens=tokens,
        duration_seconds=time.monotonic() - started,
        raw_replies=tuple(raw_replies),
        dropped_results=tuple(dropped_results),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def compute_assignment_run_id(run: AssignmentRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-assignment-{digest[:16]}"


def save_assignment_run(run: AssignmentRun, store: DerivedStore) -> DerivedEnvelope:
    """Persist beside the taxonomy it classified against, never onto it."""
    return store.write(
        compute_assignment_run_id(run),
        kind="rlm_assignment_run",
        parent_artifact_id=run.taxonomy_id,
        payload=run.model_dump_json().encode("utf-8"),
        summary={
            "assigned": len(run.by_status(AssignmentStatus.ASSIGNED)),
            "ambiguous": len(run.by_status(AssignmentStatus.AMBIGUOUS)),
            "uncovered": len(run.by_status(AssignmentStatus.UNCOVERED)),
            "unreadable": len(run.by_status(AssignmentStatus.UNREADABLE)),
        },
    )


def load_assignment_run(run_id: str, store: DerivedStore) -> AssignmentRun:
    return AssignmentRun.model_validate_json(store.read_payload(run_id))
