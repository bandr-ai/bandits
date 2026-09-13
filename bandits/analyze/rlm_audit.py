"""Attack a clustering run with a context that never watched it being built.

The discovery loop argues itself into a taxonomy, and the reasoning that
produced a family is exactly the reasoning least able to see what is wrong with
it. So the audit runs in a fresh context holding the contract, its siblings, and their
members: it did not see the chunk that motivated a split, so it cannot inherit
the assumption behind it.

Five questions per contract, each aimed at a distinct failure. The least
compatible pair finds internal disagreement. The strongest outsider finds a
boundary drawn too tight. The topical-grouping check finds the specific failure
this whole path exists to detect — members that share a subject while needing
different verifiers, which is what embedding distance cannot tell apart and what
a plausible family name actively hides.

Purely advisory, and with no gate behind it. Nothing here edits a contract,
changes a placement, or blocks materialization; it saves findings for a person
to read. An audit that is never run costs a task set nothing but the reading,
which is the honest position until the findings are shown to be worth acting on.
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
    AuditBudget,
    AuditFinding,
    AuditStopReason,
    FamilyContract,
    RLMClusteringAudit,
    RLMClusteringRun,
    TraceView,
)
from bandits.store import DerivedEnvelope, DerivedStore

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
DEFAULT_MAX_TOKENS = 24000
PROMPT_VERSION = 2

_INSTRUCTION_HEAD = """You are auditing one proposed task-family contract by trying to \
break it. Be adversarial: your job is to find what is wrong with it, not to agree.

{view}

The variable `contract` is the proposed family. The variable `sibling_contracts` \
lists every other proposed contract, so you can detect an unnecessarily split \
boundary. The variable `members` lists the \
traces it claims, each with trace_id and messages. The variable `outsiders` lists \
traces it does not claim, in the same shape.

Two traces belong together only when ONE parameterized verifier could correctly \
evaluate what both users requested. Sharing a domain, a verb, or a phrasing is \
not enough.

Report:
- least_compatible_pair: the two member trace_ids whose requested work differs \
most, and say how their required outcomes differ. Empty list only if there are \
under two members.
- strongest_outsider_trace_id: the outsider that most looks like it belongs. \
Empty string if none is close.
- topical_only: true if the members share a subject but would need materially \
different verifiers. This is the failure you are most looking for.
- recommendation: "keep", "revise", "split", "merge", or "uncertain".
- merge_with_contract_id: for "merge", the sibling contract_id whose requested \
work and required outcome can use the same parameterized verifier; otherwise an \
empty string.
- rationale: two or three sentences on what decided it.

When the evidence does not settle it, answer "uncertain" rather than guessing at \
split or keep. Split when you can name the outcome check that would have to \
differ; keep when every member answers to the same one.

Check the sibling contracts for a boundary that exists only because two \
contracts fixed different parameter values — a different route, amount, product \
or account — while requiring the same outcome check. That is a merge, and it is \
the failure most easily missed. Recommend merge only when the two contracts can \
genuinely share one verifier contract, not merely because their topics overlap. \
Recommendations are advisory; never rewrite or combine contracts yourself."""


def instruction_for(view: TraceView) -> str:
    """The audit prompt as this arm's auditor actually receives it.

    An auditor told it sees only user messages, while being handed trajectories,
    would report differences in how agents behaved as differences in requested
    work — manufacturing exactly the finding Path F must be tested for.
    """
    return _INSTRUCTION_HEAD.format(view=VIEW_PREAMBLES[view])


class ClusteringAuditError(RuntimeError):
    """The auditor could not be built or returned nothing usable."""


class _Predictor(Protocol):
    def __call__(
        self, *, contract: str, sibling_contracts: str, members: str, outsiders: str, question: str
    ) -> Any: ...


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
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> _Predictor:
    """A ``dspy.RLM`` over one contract, imported only when an audit runs."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ClusteringAuditError(
            "the taxonomy audit needs the 'audit' extra: uv sync --extra audit"
        ) from exc

    from bandits.analyze.rlm_history import scoped_to_history
    from bandits.analyze.rlm_mine import with_cost
    from bandits.verify.judge import resolve_api_key

    key = api_key or resolve_api_key()
    language_model = dspy.LM(
        f"fireworks_ai/{model}",
        api_key=key,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    # Instructions on the signature, not in an input field: an input field
    # becomes a REPL variable and is shown as a 1000-character peek, which is
    # how the mining prompt's schema went unread. See rlm_mine.instruction_for.
    class _Audit(dspy.Signature):
        contract: str = dspy.InputField(desc="the family under attack")
        sibling_contracts: str = dspy.InputField(desc="every other proposed contract")
        members: str = dspy.InputField(desc="traces this contract claims")
        outsiders: str = dspy.InputField(desc="traces it does not claim")
        recommendation: str = dspy.OutputField()
        least_compatible_pair: list[str] = dspy.OutputField()
        strongest_outsider_trace_id: str = dspy.OutputField()
        merge_with_contract_id: str = dspy.OutputField()
        topical_only: bool = dspy.OutputField()
        rationale: str = dspy.OutputField()

    _Audit.__doc__ = instruction_for(view)
    rlm = dspy.RLM(
        _Audit, max_iters=max_iterations, max_llm_calls=max_llm_calls, sub_lm=language_model
    )

    def predict(
        *,
        contract: str,
        sibling_contracts: str,
        members: str,
        outsiders: str,
        question: str = "",
    ) -> Any:
        with dspy.context(lm=language_model):
            return rlm(
                contract=contract,
                sibling_contracts=sibling_contracts,
                members=members,
                outsiders=outsiders,
            )

    return with_cost(scoped_to_history(predict, language_model))


_OUTSIDER_SAMPLE = 12
"""How many non-members one audit is shown.

Bounded because "the strongest apparent member currently outside" is a search
over the whole corpus, and handing a whole corpus to every contract's audit
costs more than the question is worth. Drawn deterministically from the run
seed so the sample is reproducible from the artifact.
"""


def _rows(corpus: ReadOnlyCorpus, trace_ids: tuple[str, ...]) -> str:
    return json.dumps(
        [
            {"trace_id": view.trace_id, "messages": list(view.messages)}
            for view in corpus.get_user_message_batch(trace_ids)
            if view.readable
        ],
        indent=2,
        sort_keys=True,
        default=str,
    )


def _raw_reply(prediction: Any) -> str:
    """Everything the auditor returned, serialized and never truncated."""
    fields = (
        "recommendation",
        "least_compatible_pair",
        "strongest_outsider_trace_id",
        "merge_with_contract_id",
        "topical_only",
        "rationale",
    )
    try:
        return json.dumps(
            {name: getattr(prediction, name, None) for name in fields},
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        return str(prediction)


_PEEK_LIMIT = 1000
"""What ``REPLVariable.from_value`` shows of a variable before eliding its middle."""


def _compact_siblings(contracts: tuple[FamilyContract, ...]) -> str:
    """Sibling contracts, trimmed to what a merge judgement actually needs.

    Compact because this is a REPL variable, and anything past a thousand
    characters is shown to the root model as a peek: the first five hundred
    characters, an ellipsis, then the last five hundred. Six contracts dumped in
    full serialize to roughly twice that, so a model reading only the peek would
    see some ids and not others — and could name a merge target it had never
    been shown, or miss the one that mattered.

    Identity, the claim, and the outcome. The rules and the evidence are what a
    reviewer reads; they are not what decides whether one verifier covers two
    contracts.
    """
    if not contracts:
        return "[]"

    # Measured rather than estimated. A per-row character budget guessed from
    # the key names was still overflowing on long definitions, and an overflow
    # here is not cosmetic: past the peek the root model sees an elided middle
    # and can name a merge target it was never shown. So the text is trimmed
    # until the serialized result actually fits.
    def render(budget: int) -> str:
        def clip(text: str) -> str:
            flat = " ".join(text.split())
            return flat if len(flat) <= budget else flat[: max(budget - 1, 1)] + "…"

        return json.dumps(
            [
                {
                    "contract_id": item.contract_id,
                    "name": clip(item.name),
                    "definition": clip(item.definition),
                    "required_outcome_shape": [
                        clip(line) for line in item.required_outcome_shape[:1]
                    ],
                }
                for item in contracts
            ],
            separators=(",", ":"),
            default=str,
        )

    for budget in (160, 110, 70, 45, 25, 12):
        rendered = render(budget)
        if len(rendered) <= _PEEK_LIMIT:
            return rendered
    # Nothing fits: ids alone still answer "is this target real", which is the
    # one question a merge recommendation has to be checked against.
    return json.dumps(
        [{"contract_id": item.contract_id} for item in contracts], separators=(",", ":")
    )


def _clean_pair(raw: Any, members: set[str]) -> tuple[str, str] | None:
    """Keep a pair only if it names two distinct real members."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    left, right = (str(item).strip() for item in raw)
    if left == right or left not in members or right not in members:
        return None
    return (left, right)


def audit_contract(
    contract: FamilyContract,
    corpus: ReadOnlyCorpus,
    *,
    members: tuple[str, ...],
    outsiders: tuple[str, ...],
    sibling_contracts: tuple[FamilyContract, ...],
    predict: _Predictor,
) -> AuditFinding:
    """Challenge one contract and report what the attack found."""
    prediction = predict(
        contract=contract.model_dump_json(indent=2),
        sibling_contracts=_compact_siblings(sibling_contracts),
        members=_rows(corpus, members),
        outsiders=_rows(corpus, outsiders),
        question=instruction_for(corpus.view),
    )

    raw_recommendation = str(getattr(prediction, "recommendation", "") or "").strip().lower()
    if raw_recommendation not in ("keep", "revise", "split", "merge", "uncertain"):
        # An unreadable recommendation becomes "uncertain", never "keep". A
        # parse failure that defaulted to keep would silently clear the freeze
        # gate, which is the one outcome an audit must never produce by accident.
        raw_recommendation = "uncertain"

    raw_reply = _raw_reply(prediction)
    outsider = str(getattr(prediction, "strongest_outsider_trace_id", "") or "").strip()
    merge_with = str(getattr(prediction, "merge_with_contract_id", "") or "").strip()
    sibling_ids = {item.contract_id for item in sibling_contracts}
    if raw_recommendation == "merge" and merge_with not in sibling_ids:
        # A target that names nothing real cannot be acted on, and downgrading
        # is safer than dropping the field and keeping the recommendation.
        raw_recommendation = "uncertain"
        merge_with = ""
    elif raw_recommendation != "merge":
        # A target on a keep or a split is a contradiction the contract rejects
        # outright, which would lose the whole finding over a stray field.
        merge_with = ""
    rationale = str(getattr(prediction, "rationale", "") or "").strip()
    return AuditFinding(
        contract_id=contract.contract_id,
        recommendation=raw_recommendation,  # type: ignore[arg-type]
        least_compatible_pair=_clean_pair(
            getattr(prediction, "least_compatible_pair", None), set(members)
        ),
        strongest_outsider_trace_id=outsider if outsider in set(outsiders) else None,
        merge_with_contract_id=merge_with or None,
        topical_only=bool(getattr(prediction, "topical_only", False)),
        rationale=rationale or "the auditor returned no rationale",
        raw_reply=raw_reply,
    )


def _audit_budget_stop(
    budget: AuditBudget,
    *,
    calls: int,
    started: float,
    baseline_elapsed: float,
    usd: float,
    contracts_done: int,
) -> AuditStopReason | None:
    """Which ceiling, if any, this audit session has hit.

    Checked before each contract rather than after, so a session never spends
    past a limit it has already reached. ``baseline_elapsed`` is the time a
    resumed session had already spent before this process started — without
    adding it in, --max-seconds would restart its clock on every resume, so a
    session interrupted and resumed five times could run five times as long as
    the ceiling actually allows.
    """
    if calls >= budget.max_llm_calls:
        return AuditStopReason.MAX_LLM_CALLS
    if baseline_elapsed + (time.monotonic() - started) >= budget.max_seconds:
        return AuditStopReason.MAX_SECONDS
    if budget.max_usd is not None and usd >= budget.max_usd:
        return AuditStopReason.MAX_USD
    if budget.max_contracts is not None and contracts_done >= budget.max_contracts:
        return AuditStopReason.MAX_CONTRACTS
    return None


def audit_clustering(
    run: RLMClusteringRun,
    run_id: str,
    corpus: ReadOnlyCorpus,
    *,
    predict: _Predictor,
    model: str = DEFAULT_MODEL,
    budget: AuditBudget | None = None,
    session: Any = None,
    resume: Any = None,
    should_stop: Callable[[], bool] | None = None,
) -> RLMClusteringAudit:
    """Challenge every contract in a run, one pass each.

    A contract whose audit fails is recorded as ``uncertain`` rather than
    skipped: an absent finding reads as a contract nobody objected to, and the
    freeze gate would then pass on the strength of a call that never happened.

    ``session``, when given, is checkpointed after every contract so a run
    that dies mid-audit is resumable and a run in flight is observable.
    ``resume``, when given (an ``AuditSessionState``), skips contracts it
    already has a finding for and carries its cumulative calls/cost/elapsed
    forward. ``should_stop``, when given, is polled before each contract so a
    SIGINT handler can ask the loop to stop cleanly and save what it has.
    """
    budget = budget or AuditBudget()
    digest = prompt_digest(model)

    if resume is not None:
        if resume.run_id != run_id:
            raise ClusteringAuditError(
                f"session was auditing run {resume.run_id!r}, not {run_id!r}"
            )
        if resume.model != model:
            raise ClusteringAuditError(
                f"session used model {resume.model!r}, this call uses {model!r}"
            )
        if resume.view is not corpus.view:
            raise ClusteringAuditError(
                f"session used the {resume.view.value} view, this call uses "
                f"{corpus.view.value}"
            )
        if resume.prompt_digest != digest:
            raise ClusteringAuditError(
                "the audit prompt has changed since this session started; its "
                "existing findings do not answer the same question a fresh "
                "contract would be asked"
            )

    already_done = set(resume.completed_contract_ids) if resume is not None else set()
    findings: list[AuditFinding] = list(resume.findings) if resume is not None else []
    limitations: list[str] = []

    assigned = _run_members(run)
    all_readable = set(corpus.readable_trace_ids())
    contract_order = tuple(c.contract_id for c in run.contracts)

    if session is not None:
        session.begin(contract_order=contract_order)

    calls = resume.llm_calls if resume is not None else 0
    usd = resume.cost_usd if resume is not None else 0.0
    started = time.monotonic()
    baseline_elapsed = resume.elapsed_seconds if resume is not None else 0.0
    stop_reason = AuditStopReason.ALL_CONTRACTS_AUDITED
    interrupted = False

    for contract in run.contracts:
        if contract.contract_id in already_done:
            continue

        if should_stop is not None and should_stop():
            interrupted = True
            stop_reason = AuditStopReason.INTERRUPTED
            break

        budget_hit = _audit_budget_stop(
            budget,
            calls=calls,
            started=started,
            baseline_elapsed=baseline_elapsed,
            usd=usd,
            contracts_done=len(already_done),
        )
        if budget_hit is not None:
            stop_reason = budget_hit
            break

        members = assigned.get(contract.contract_id, ())
        pool = sorted(all_readable - set(members))
        outsiders = (
            corpus.sample_trace_ids(_OUTSIDER_SAMPLE, seed=run.seed, exclude=tuple(members))
            if pool
            else ()
        )
        try:
            with ledger.stage("rlm_taxonomy_audit", contract_id=contract.contract_id, model=model):
                finding = audit_contract(
                    contract,
                    corpus,
                    members=members,
                    outsiders=outsiders,
                    sibling_contracts=tuple(
                        sibling
                        for sibling in run.contracts
                        if sibling.contract_id != contract.contract_id
                    ),
                    predict=predict,
                )
        except Exception as exc:  # noqa: BLE001 - one failure must not lose the session
            if should_stop is not None and should_stop():
                # A SIGINT during this exact call propagates to the RLM
                # sandbox subprocess and kills it, which surfaces here as an
                # ordinary exception — but the call did not fail on its own
                # merits, it was cut off by the same interrupt this loop is
                # about to honor. Recording an "uncertain" finding for it
                # would manufacture a permanent verdict out of an interrupt
                # and make resume skip a contract that was never actually
                # audited. So this contract is left untouched: not appended
                # to findings, not added to already_done, and the loop stops
                # here rather than checkpointing anything for it.
                interrupted = True
                stop_reason = AuditStopReason.INTERRUPTED
                break
            finding = AuditFinding(
                contract_id=contract.contract_id,
                recommendation="uncertain",
                rationale=f"the audit of this contract failed and reached no verdict: {exc}",
            )
            limitations.append(f"the audit of {contract.contract_id} failed: {exc}")

        findings.append(finding)
        already_done.add(contract.contract_id)
        # One contract's audit is a whole dspy.RLM sub-loop, not one physical
        # call: it can issue anywhere from one call up to that predictor's own
        # max_llm_calls before returning. `predict.spend.entries`, set by
        # `scoped_to_history`, holds exactly the calls *this* contract's
        # predict() just made, so its length is the real count — a flat +1
        # here would silently make --max-llm-calls (and every calls figure
        # shown to a person watching) count contracts instead of the thing its
        # name promises.
        spend = getattr(predict, "spend", None)
        calls += len(getattr(spend, "entries", None) or ()) or 1
        call_cost = getattr(predict, "cost", lambda: None)()
        if call_cost is not None:
            usd += call_cost

        if session is not None:
            session.checkpoint(
                finding,
                calls=calls,
                usd=usd,
                elapsed=baseline_elapsed + (time.monotonic() - started),
            )

    findings = _normalize_reciprocal_merges(findings)
    complete = stop_reason == AuditStopReason.ALL_CONTRACTS_AUDITED
    if complete and findings:
        limitations.append(
            "audit output is advisory and uncalibrated: it never edits a contract, and "
            "it has not been measured against labelled same-family pairs"
        )
    if not complete:
        limitations.append(
            f"this audit stopped early ({stop_reason.value}) and covers "
            f"{len(already_done)}/{len(contract_order)} contract(s); contracts with no "
            "finding here were never checked, not passed"
        )

    if session is not None:
        if complete:
            session_status = "awaiting_review"
        elif interrupted:
            session_status = "interrupted"
        else:
            session_status = "incomplete"
        session.finish(status=session_status, stop_reason=stop_reason.value)

    return RLMClusteringAudit(
        run_id=run_id,
        findings=tuple(findings),
        model=model,
        prompt_digest=digest,
        limitations=tuple(dict.fromkeys(limitations)),
        status="complete" if complete else "incomplete",
        stop_reason=stop_reason.value,
    )


def _normalize_reciprocal_merges(findings: list[AuditFinding]) -> list[AuditFinding]:
    """Collapse a mutual merge proposal into one.

    Two contracts that each nominate the other are one proposal about the pair,
    not two independent findings. Left as two, a reviewer sees the same merge
    twice and the freeze gate counts it twice, so accepting it once would still
    leave the taxonomy blocked by its own mirror image.

    The finding on the lexically first contract is kept, because a stable choice
    keeps the artifact reproducible; the other becomes an ordinary keep whose
    rationale records where the proposal went.
    """
    proposals = {
        f.contract_id: f.merge_with_contract_id
        for f in findings
        if f.recommendation == "merge" and f.merge_with_contract_id
    }
    superseded = {
        other for one, other in proposals.items() if proposals.get(other) == one and other > one
    }
    if not superseded:
        return findings
    return [
        f.replace(
            recommendation="keep",
            merge_with_contract_id=None,
            rationale=(
                f"{f.rationale} (this merge is recorded once, on {f.merge_with_contract_id})"
            ),
        )
        if f.contract_id in superseded
        else f
        for f in findings
    ]


def _run_members(run: RLMClusteringRun) -> dict[str, tuple[str, ...]]:
    """The membership the discovery loop actually ended with.

    Read from ``run.assignments``, which is the placement the run held when it
    stopped. The obvious alternative — replaying the chunk assignments in order —
    is wrong, and quietly so: a MERGE moves the consumed family's members onto
    the survivor without re-listing them in any later chunk, so a replay shows
    them still under a contract that no longer exists. A reviewer then sees a
    ghost family beside the real one and a survivor missing the members it
    absorbed, which is the opposite of what the merge decided.

    Falls back to the replay only for drafts written before ``assignments`` was
    stored, where it is the sole record there is; such a run predates merges
    being applied at all, so the replay is accurate for it.
    """
    if run.assignments:
        placement = dict(run.assignments)
    else:  # pragma: no cover - only reachable for drafts written before this field
        placement = {}
        for chunk in run.chunks:
            placement.update(chunk.assignments)
            for trace_id in chunk.ambiguous_trace_ids + chunk.uncovered_trace_ids:
                placement.pop(trace_id, None)

    # A contract that no longer exists cannot have members. Anything pointing at
    # one is a bug rather than a finding, so it is dropped here instead of being
    # rendered as a family a reviewer might try to act on.
    live = {contract.contract_id for contract in run.contracts}
    grouped: dict[str, list[str]] = {}
    for trace_id, contract_id in placement.items():
        if contract_id in live:
            grouped.setdefault(contract_id, []).append(trace_id)
    return {cid: tuple(sorted(traces)) for cid, traces in grouped.items()}


def compute_audit_id(audit: RLMClusteringAudit) -> str:
    digest = hashlib.sha256(audit.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-taxonomy-audit-{digest[:16]}"


def save_audit(audit: RLMClusteringAudit, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_audit_id(audit),
        kind="rlm_clustering_audit",
        parent_artifact_id=audit.run_id,
        payload=audit.model_dump_json().encode("utf-8"),
        summary={
            "findings": len(audit.findings),
            "actionable": sum(1 for f in audit.findings if f.demands_action),
            "topical_only": sum(1 for f in audit.findings if f.topical_only),
        },
    )


def load_audit(audit_id: str, store: DerivedStore) -> RLMClusteringAudit:
    return RLMClusteringAudit.model_validate_json(store.read_payload(audit_id))
