"""Read a whole family with a model and report whether it hangs together.

Advisory, and deliberately so. Clustering groups episodes deterministically and
nothing checks whether the grouping was any good; a reviewer can split or merge
a family but has nothing telling them *which* family is worth the look. This
supplies that, and stops there: no output here changes ``similarity``,
``neighbors``, ``family_id``, ``fingerprint()`` or the fit/held-out split, and
re-mining without this pass reproduces the same families byte for byte.

Only splits are proposed. The two errors are not symmetric — a family split that
should not have been yields two coherent families that each draft a valid
verifier, costing some redundancy, while a family merged that should not have
been yields one family whose evidence disagrees with itself, and
``draft_verifiers`` will key a check to whichever value happened to be most
common. That is a wrong verifier that looks fine, so merging stays a human call.

Why a recursive scaffold rather than one call: a family can hold dozens of
members. Feeding all of them to a single call either truncates or overruns the
window, and chunking by hand loses the cross-chunk comparison that "which of
these forty belong together" is entirely made of. A Recursive Language Model
(https://arxiv.org/abs/2512.24601) loads the members into a REPL as a variable
the root model never reads directly, then writes code to slice them and spawns
sub-calls over the pieces.

The scaffold is not deterministic — the root model writes different code each
run — which is exactly why nothing here feeds back into grouping. One pass per
family, output annotated for a human, no loop and therefore no stopping
criterion to get wrong. A loop that ran until the model stopped objecting would
be optimising for the model's agreement rather than for correct grouping, the
same failure ``assess_promotion`` exists to prevent one layer down. Anything
beyond advisory output waits on a labelled benchmark (#16).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from bandits import ledger
from bandits.analyze.families import normalize_instruction
from bandits.analyze.models import (
    CorpusAnalysis,
    Evidence,
    FamilyAudit,
    FamilyAuditRun,
    SkippedAudit,
    TaskFamily,
    TaskSet,
)
from bandits.store import DerivedEnvelope, DerivedStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
"""Matches the rubric judge's default, so one credential covers both passes."""

DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_LLM_CALLS = 30
"""Hard ceilings on one family's audit. The root model decides how to slice a
family, so cost is bounded here rather than by trusting it to be brief."""

PROMPT_VERSION = 1

_INSTRUCTION = """You are auditing one group of agent episodes that an automatic \
clustering step claims are all the same repeatable task.

The variable `members` is a list of dicts, each with keys: trace_id, instruction, \
normalized (the value-preserving form clustering compared), tool_names, span_count.

Decide whether these episodes are genuinely one task. Judge by what the human \
was asking for, not by surface wording: two differently-phrased requests for the \
same work are one task, while one shared verb over different goals is not.

Report:
- coherent: true only if every member is the same task.
- outlier_trace_ids: trace_ids that do not belong. Empty if coherent.
- proposed_subgroups: if the group is really several tasks, the split, as lists \
of trace_ids. Every listed trace_id must be a member, and none may appear twice. \
Leave empty unless you are proposing a split; never return a single subgroup.
- generated_name: a short imperative name for the dominant task, e.g. "Refund an \
eligible order". This is used for reports only.
- rationale: two or three sentences on what decided it.

Split when in doubt; never propose merging this family with anything else."""


class AuditError(RuntimeError):
    """The auditor could not be built or returned nothing usable."""


class AuditFailed(AuditError):
    """One family's audit failed, carrying the record of the attempt.

    An error string says what went wrong and nothing about what it cost. The
    attached audit is a real ``FamilyAudit`` with ``status="error"``, so a
    failure is stored the same way a success is and a run that lost a family
    can still say what that family was asked and what the attempt spent.
    """

    def __init__(self, audit: FamilyAudit) -> None:
        super().__init__(audit.error)
        self.audit = audit


def _spend_of(predict: Any) -> tuple[int | None, dict[str, int]]:
    """What the predictor says its last prediction cost, if it says anything.

    Optional by design: the injected predictors the tests use are plain
    functions, and a backend that cannot report its own history should leave
    the count unknown rather than have one invented for it.
    """
    spend = getattr(predict, "spend", None)
    if spend is None:
        return None, {}
    try:
        return spend()
    except Exception:  # noqa: BLE001 - a bookkeeping failure must not lose the audit
        return None, {}


def _failed_audit(
    family: TaskFamily,
    *,
    model: str,
    inputs: dict[str, str],
    error: str,
    duration: float,
    llm_calls: int | None,
    tokens: dict[str, int],
) -> FamilyAudit:
    """The record of an attempt that produced no verdict.

    ``coherent`` is true and the outlier and subgroup fields are empty, because
    the contract has no third state and a failed audit must not read as a
    finding: nothing was concluded. ``status`` is what distinguishes it, and
    every caller that counts findings filters on that.
    """
    return FamilyAudit(
        family_id=family.family_id,
        coherent=True,
        rationale=f"the audit failed and reached no verdict: {error}",
        model=model,
        prompt_digest=prompt_digest(model),
        inputs=inputs,
        llm_calls=llm_calls,
        tokens=tokens,
        duration_seconds=duration,
        status="error",
        error=error,
    )


class _Predictor(Protocol):
    """The one call this module makes, so tests need no model and no sandbox."""

    def __call__(self, *, members: str, question: str) -> Any: ...


def _member_view(family: TaskFamily, analysis: CorpusAnalysis) -> list[dict[str, Any]]:
    """What the model reads: one row per member, ordered like the family.

    The instruction, the normalized form clustering compared, and structural shape.
    No outcome and no label: either would let the audit call a family incoherent
    because its episodes *ended* differently, which is a fact about the runs and
    not about the grouping. The tools called and the span count are shape rather
    than outcome, and are what ``families.py`` groups on.
    """
    by_trace = {task.trace_id: task for task in analysis.tasks}
    evidence_by_trace: dict[str, list[Evidence]] = {}
    for item in analysis.evidence:
        evidence_by_trace.setdefault(item.trace_id, []).append(item)

    rows: list[dict[str, Any]] = []
    for trace_id in family.trace_ids:
        task = by_trace.get(trace_id)
        if task is None:
            continue
        evidence = evidence_by_trace.get(trace_id, [])
        rows.append(
            {
                "trace_id": trace_id,
                "instruction": task.instruction,
                "normalized": normalize_instruction(task.instruction),
                # Both read from evidence, the way `families.py` reads them to
                # group in the first place. Span ids are `span-{n}` or
                # `{trace_id}:span-{n}` and never name a tool, so deriving these
                # from them reported no tools at all on one ingest path and the
                # trace's own id as a tool name on the other.
                "tool_names": next(
                    (sorted(e.value) for e in evidence if e.claim == "tools_called"), []
                ),
                "span_count": next(
                    (int(e.value) for e in evidence if e.claim == "episode_span_count"), 0
                ),
            }
        )
    return rows


def prompt_digest(model: str) -> str:
    """Pins wording, model and version onto every audit they produced."""
    payload = json.dumps(
        {"instruction": _INSTRUCTION, "model": model, "version": PROMPT_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_predictor(
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
) -> _Predictor:
    """A ``dspy.RLM`` over one family, imported only when an audit actually runs.

    DSPy and its REPL sandbox are an optional extra: the core install stays at
    three runtime dependencies, and CI runs the tests below against an injected
    predictor rather than a model.
    """
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise AuditError("the family audit needs the 'audit' extra: uv sync --extra audit") from exc

    from bandits.verify.judge import resolve_api_key

    key = api_key or resolve_api_key()
    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=key,
        # The root model writes code rather than prose, and a sampled plan
        # rereads the family differently for no gain a reviewer can use.
        temperature=0.0,
    )

    signature = (
        "members: str, question: str -> coherent: bool, outlier_trace_ids: list[str], "
        "proposed_subgroups: list[list[str]], generated_name: str, rationale: str"
    )
    rlm = dspy.RLM(
        signature,
        max_iters=max_iterations,
        max_llm_calls=max_llm_calls,
        sub_lm=language_model,
    )

    def predict(*, members: str, question: str) -> Any:
        with dspy.context(lm=language_model):
            return rlm(members=members, question=question)

    return scoped_to_history(predict, language_model)


class _Spend:
    """The calls the last prediction added to a shared history, and their cost."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def __call__(self) -> tuple[int | None, dict[str, int]]:
        return _summarize_history(self.entries)


def scoped_to_history(predict: _Predictor, language_model: Any) -> _Predictor:
    """Wrap a predictor so it reports only the calls *it* made.

    ``lm.history`` is one mutable list the language model appends to for the
    life of the process, shared across every family audited in a run. Reading
    it whole would charge each family for all its predecessors, so the length
    is marked before the prediction and only the entries added after that point
    are attributed to it.

    The slice is copied immediately rather than held as a reference, because
    the list keeps growing: a reference read after the next family started
    would describe that family's calls too.
    """
    spend = _Spend()

    def wrapped(*, members: str, question: str) -> Any:
        before = len(getattr(language_model, "history", ()) or ())
        try:
            return predict(members=members, question=question)
        finally:
            # In `finally` because a failed audit still spent calls, and those
            # are exactly the ones a rerun that behaved differently needs.
            history = getattr(language_model, "history", None)
            spend.entries = list(history[before:]) if isinstance(history, list) else []
            _record_history(spend.entries)

    wrapped.spend = spend  # type: ignore[attr-defined]
    return wrapped


_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _record_history(entries: Sequence[Any]) -> None:
    """Write each RLM subcall to the ledger as the physical call it was.

    The audit reaches Fireworks through DSPy rather than through
    ``transport.request_with_retry``, so ``ledger.model_call`` never sees these
    and the stage would otherwise record one event for what is really a dozen
    requests. Verified against a real run: each entry carries the messages
    sent, the text returned, the code the root model wrote, per-call token
    usage and the provider's own cost figure.

    What is still not visible here: the REPL's stdout appears only inside the
    *next* entry's prompt, as the ``repl_history`` the model is shown, so a
    final iteration's output is unrecoverable from history alone. Retries and
    failures inside litellm are invisible too — a call that failed and was
    retried appears as one entry, or as none.
    """
    if not ledger.enabled():
        return
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        outputs = entry.get("outputs") or []
        text = ""
        if outputs:
            first = outputs[0]
            text = first.get("text", "") if isinstance(first, dict) else str(first)
        code = _CODE_BLOCK.findall(text)
        ledger.record(
            {
                "event_type": "model_call",
                "provider": "dspy",
                "model": entry.get("model"),
                "iteration": index + 1,
                "request": {"messages": entry.get("messages")},
                "response": {"text": text},
                # Pulled out of the reply rather than left inside it: the code
                # is what the root model actually did, and grepping a ledger
                # for it should not mean parsing markdown fences back out.
                "generated_code": code,
                "usage": entry.get("usage"),
                "cost_usd": entry.get("cost"),
                "provider_request_id": entry.get("uuid"),
                "timestamp": entry.get("timestamp"),
                "status": "success",
            }
        )


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
"""What a provider reports. Never derived: a total computed here from a sum
that is missing a call reads as authoritative and is not."""


def _summarize_history(entries: Sequence[Any]) -> tuple[int | None, dict[str, int]]:
    """How many DSPy history entries those are, and what they reported.

    One entry is one request DSPy issued, verified against a real run. It is
    not necessarily one *physical* HTTP request: litellm retries below this
    layer, so a call that failed and succeeded on retry appears once, and a
    call that failed permanently may not appear at all. The count is therefore
    a floor on what was spent, which is the honest reading of it.

    Tokens are summed only over the entries that actually reported them. A call
    whose backend said nothing contributes nothing rather than a zero, because
    a zero would silently understate the bill; when no call reported at all the
    result is empty, which reads as unknown.
    """
    totals: dict[str, int] = {}
    for entry in entries:
        usage = entry.get("usage") if isinstance(entry, dict) else getattr(entry, "usage", None)
        if not isinstance(usage, dict):
            continue
        for field in _USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, int):
                totals[field] = totals.get(field, 0) + value
    return len(entries), totals


def _rendered(prediction: Any) -> str:
    """The auditor's reply as text, for a reader comparing it to what was stored.

    Read off the fields the signature declares rather than by serializing the
    prediction, whose backend type carries trace state a reviewer has no use
    for. Best effort: this exists to be read, so a backend that returns
    something unreadable costs the record, never the audit.
    """
    fields = ("coherent", "outlier_trace_ids", "proposed_subgroups", "generated_name", "rationale")
    try:
        return json.dumps(
            {name: getattr(prediction, name, None) for name in fields},
            indent=2,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return str(prediction)


def _clean_ids(raw: Any, members: set[str]) -> tuple[str, ...]:
    """Keep only ids that are really members, in a stable order.

    A model writes these. One that hallucinates a trace_id, or repeats one,
    would otherwise fail the contract's validator and lose the whole audit —
    including the parts it got right — so unknown ids are dropped here and the
    drop is reported as a limitation by the caller.
    """
    if not isinstance(raw, (list, tuple)):
        return ()
    seen: dict[str, None] = {}
    for item in raw:
        if isinstance(item, str) and item in members:
            seen.setdefault(item, None)
    return tuple(seen)


def audit_family(
    family: TaskFamily,
    analysis: CorpusAnalysis,
    *,
    predict: _Predictor,
    model: str = DEFAULT_MODEL,
) -> FamilyAudit:
    """Read one family and report whether its members are the same task."""
    rows = _member_view(family, analysis)
    if not rows:
        raise AuditError(f"family {family.family_id} has no readable members to audit")

    members_json = json.dumps(rows, indent=2, sort_keys=True, default=str)
    inputs = {"members": members_json, "question": _INSTRUCTION}
    started = time.monotonic()
    try:
        with ledger.stage("family_audit", family_id=family.family_id, model=model):
            prediction = predict(members=members_json, question=_INSTRUCTION)
    except Exception as exc:
        # The failure gets the same record as a success, minus a verdict. An
        # audit that vanished into a skip reason lost what it was asked, what
        # it spent before dying and how long it ran — and a failed call has
        # already cost rate-limit budget and possibly tokens.
        calls, tokens = _spend_of(predict)
        raise AuditFailed(
            _failed_audit(
                family,
                model=model,
                inputs=inputs,
                error=str(exc),
                duration=time.monotonic() - started,
                llm_calls=calls,
                tokens=tokens,
            )
        ) from exc
    duration = time.monotonic() - started
    llm_calls, tokens = _spend_of(predict)

    members = {row["trace_id"] for row in rows}
    outliers = _clean_ids(getattr(prediction, "outlier_trace_ids", ()), members)

    subgroups: list[tuple[str, ...]] = []
    placed: set[str] = set()
    for group in getattr(prediction, "proposed_subgroups", ()) or ():
        # A trace claimed by two subgroups is not a split; the first claim wins
        # so the proposal stays something `split-family` could act on.
        cleaned = tuple(t for t in _clean_ids(group, members) if t not in placed)
        if cleaned:
            subgroups.append(cleaned)
            placed.update(cleaned)
    if len(subgroups) == 1:
        # One subgroup is the family it already is, not a proposal.
        subgroups = []

    coherent = bool(getattr(prediction, "coherent", False))
    if subgroups:
        # A split proposal is itself the claim that this is not one task; taking
        # the boolean over the proposal would emit a contract the validator rejects.
        coherent = False

    name = getattr(prediction, "generated_name", None)
    rationale = str(getattr(prediction, "rationale", "") or "").strip()
    return FamilyAudit(
        family_id=family.family_id,
        coherent=coherent,
        outlier_trace_ids=() if coherent else outliers,
        proposed_subgroups=tuple(subgroups),
        generated_name=(str(name).strip() or None) if name else None,
        rationale=rationale or "the auditor returned no rationale",
        model=model,
        prompt_digest=prompt_digest(model),
        inputs=inputs,
        rendered_prediction=_rendered(prediction),
        llm_calls=llm_calls,
        tokens=tokens,
        duration_seconds=duration,
    )


def audit_task_set(
    task_set: TaskSet,
    task_set_id: str,
    analysis: CorpusAnalysis,
    *,
    predict: _Predictor,
    model: str = DEFAULT_MODEL,
    family_ids: Sequence[str] | None = None,
    on_error: Callable[[str, str], None] | None = None,
) -> FamilyAuditRun:
    """Audit every multi-member family, one pass each, and report what was skipped.

    One family failing does not lose the rest: the failure is recorded as a skip
    with its reason, since an audit that silently covered less than it claimed
    would be read as an all-clear.
    """
    wanted = set(family_ids) if family_ids is not None else None
    unknown = sorted(wanted - {f.family_id for f in task_set.families}) if wanted else []
    if unknown:
        raise ValueError(f"unknown family id(s): {unknown}")

    audits: list[FamilyAudit] = []
    skipped: list[SkippedAudit] = []
    limitations: list[str] = []

    for family in task_set.families:
        if wanted is not None and family.family_id not in wanted:
            continue
        if len(family.trace_ids) < 2:
            # Nothing to split and no internal disagreement to find; a call here
            # buys nothing.
            skipped.append(
                SkippedAudit(
                    family_id=family.family_id,
                    reason="single-member family: nothing to split",
                )
            )
            continue
        try:
            audit = audit_family(family, analysis, predict=predict, model=model)
        except AuditFailed as exc:
            # Both, and deliberately: the skip is what stops this family being
            # read as audited, and the record beside it is what the attempt
            # cost. Storing only the reason string was how a failure became
            # indistinguishable from a family nobody tried.
            audits.append(exc.audit)
            skipped.append(SkippedAudit(family_id=family.family_id, reason=str(exc), failed=True))
            if on_error is not None:
                on_error(family.family_id, str(exc))
            continue
        except AuditError as exc:
            skipped.append(SkippedAudit(family_id=family.family_id, reason=str(exc)))
            if on_error is not None:
                on_error(family.family_id, str(exc))
            continue
        audits.append(audit)
        dropped = set(audit.outlier_trace_ids) - set(family.trace_ids)
        if dropped:  # pragma: no cover - _clean_ids already filters these
            limitations.append(f"audit of {family.family_id} named traces it does not contain")

    if any(a.status == "success" for a in audits):
        limitations.append(
            "audit output is advisory and uncalibrated: it proposes splits for a "
            "human to apply and never changes grouping, and it has not been "
            "measured against labelled same-family pairs"
        )
    if skipped:
        limitations.append(
            f"{len(skipped)} family(ies) were not audited; see the skipped list for why"
        )

    return FamilyAuditRun(
        task_set_id=task_set_id,
        audits=tuple(audits),
        skipped=tuple(skipped),
        model=model,
        limitations=tuple(limitations),
    )


def compute_audit_run_id(run: FamilyAuditRun) -> str:
    digest = hashlib.sha256(run.model_dump_json().encode("utf-8")).hexdigest()
    return f"family-audit-{digest[:16]}"


def save_audit_run(run: FamilyAuditRun, store: DerivedStore) -> DerivedEnvelope:
    """Persist beside the task set, never onto it."""
    return store.write(
        compute_audit_run_id(run),
        kind="family_audit",
        parent_artifact_id=run.task_set_id,
        payload=run.model_dump_json().encode("utf-8"),
        summary={
            "audited": len(run.audits),
            "incoherent": len(run.incoherent()),
            "skipped": len(run.skipped),
        },
    )


def load_audit_run(run_id: str, store: DerivedStore) -> FamilyAuditRun:
    return FamilyAuditRun.model_validate_json(store.read_payload(run_id))
