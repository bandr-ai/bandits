"""Attack a draft taxonomy with a context that never watched it being built.

The discovery loop argues itself into a taxonomy, and the reasoning that
produced a family is exactly the reasoning least able to see what is wrong with
it. So the audit runs in a fresh context holding only the contracts and their
members: it did not see the chunk that motivated a split, so it cannot inherit
the assumption behind it.

Five questions per contract, each aimed at a distinct failure. The least
compatible pair finds internal disagreement. The strongest outsider finds a
boundary drawn too tight. The topical-grouping check finds the specific failure
this whole path exists to detect — members that share a subject while needing
different verifiers, which is what embedding distance cannot tell apart and what
a plausible family name actively hides.

Advisory, like the family audit it parallels: nothing here edits a contract.
What it does have is teeth at the freeze — :func:`freeze_taxonomy` refuses while
an actionable finding is unresolved, so an audit cannot be run and ignored.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from bandits import ledger
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_models import (
    VIEW_PREAMBLES,
    AuditFinding,
    FamilyContract,
    FrozenTaxonomy,
    TaxonomyAudit,
    TaxonomyDraft,
    TraceView,
)
from bandits.store import DerivedEnvelope, DerivedStore

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
PROMPT_VERSION = 1

_INSTRUCTION_HEAD = """You are auditing one proposed task-family contract by trying to \
break it. Be adversarial: your job is to find what is wrong with it, not to agree.

{view}

The variable `contract` is the proposed family. The variable `members` lists the \
traces it claims, each with trace_id and messages. The variable `outsiders` lists \
traces it does not claim, in the same shape.

Two traces belong together only when ONE parameterized verifier could correctly \
evaluate what both users requested. Sharing a domain, a verb, or a phrasing is \
not enough.

Report:
- least_compatible_pair: the two member trace_ids whose requested work differs \
most, and say how their required outcomes differ. Null only if there are under \
two members.
- strongest_outsider_trace_id: the outsider that most looks like it belongs. \
Null if none is close.
- topical_only: true if the members share a subject but would need materially \
different verifiers. This is the failure you are most looking for.
- recommendation: "keep", "revise", "split", or "uncertain".
- rationale: two or three sentences on what decided it.

Prefer split over keep when in doubt. Never recommend merging two contracts."""


def instruction_for(view: TraceView) -> str:
    """The audit prompt as this arm's auditor actually receives it.

    An auditor told it sees only user messages, while being handed trajectories,
    would report differences in how agents behaved as differences in requested
    work — manufacturing exactly the finding Path F must be tested for.
    """
    return _INSTRUCTION_HEAD.format(view=VIEW_PREAMBLES[view])


class TaxonomyAuditError(RuntimeError):
    """The auditor could not be built or returned nothing usable."""


class FreezeRefused(RuntimeError):
    """A taxonomy was asked to freeze while something was still unresolved.

    Raised rather than warned about. Freezing is what lets an assignment name
    fixed wording, and a taxonomy frozen over an open split recommendation would
    carry that unresolved finding into every artifact derived from it, with
    nothing downstream able to tell.
    """


class _Predictor(Protocol):
    def __call__(self, *, contract: str, members: str, outsiders: str, question: str) -> Any: ...


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
    max_iterations: int = 12,
    max_llm_calls: int = 30,
) -> _Predictor:
    """A ``dspy.RLM`` over one contract, imported only when an audit runs."""
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise TaxonomyAuditError(
            "the taxonomy audit needs the 'audit' extra: uv sync --extra audit"
        ) from exc

    from bandits.analyze.audit import scoped_to_history
    from bandits.verify.judge import resolve_api_key

    key = api_key or resolve_api_key()
    language_model = dspy.LM(f"fireworks_ai/{model}", api_key=key, temperature=0.0)
    signature = (
        "contract: str, members: str, outsiders: str, question: str -> "
        "recommendation: str, least_compatible_pair: list[str], "
        "strongest_outsider_trace_id: str, topical_only: bool, rationale: str"
    )
    rlm = dspy.RLM(
        signature, max_iters=max_iterations, max_llm_calls=max_llm_calls, sub_lm=language_model
    )

    def predict(*, contract: str, members: str, outsiders: str, question: str) -> Any:
        with dspy.context(lm=language_model):
            return rlm(contract=contract, members=members, outsiders=outsiders, question=question)

    return scoped_to_history(predict, language_model)


_OUTSIDER_SAMPLE = 12
"""How many non-members one audit is shown.

Bounded because "the strongest apparent member currently outside" is a search
over the whole corpus, and handing a whole corpus to every contract's audit
costs more than the question is worth. Drawn deterministically from the draft
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
    predict: _Predictor,
) -> AuditFinding:
    """Challenge one contract and report what the attack found."""
    prediction = predict(
        contract=contract.model_dump_json(indent=2),
        members=_rows(corpus, members),
        outsiders=_rows(corpus, outsiders),
        question=instruction_for(corpus.view),
    )

    raw_recommendation = str(getattr(prediction, "recommendation", "") or "").strip().lower()
    if raw_recommendation not in ("keep", "revise", "split", "uncertain"):
        # An unreadable recommendation becomes "uncertain", never "keep". A
        # parse failure that defaulted to keep would silently clear the freeze
        # gate, which is the one outcome an audit must never produce by accident.
        raw_recommendation = "uncertain"

    raw_reply = _raw_reply(prediction)
    outsider = str(getattr(prediction, "strongest_outsider_trace_id", "") or "").strip()
    rationale = str(getattr(prediction, "rationale", "") or "").strip()
    return AuditFinding(
        contract_id=contract.contract_id,
        recommendation=raw_recommendation,  # type: ignore[arg-type]
        least_compatible_pair=_clean_pair(
            getattr(prediction, "least_compatible_pair", None), set(members)
        ),
        strongest_outsider_trace_id=outsider if outsider in set(outsiders) else None,
        topical_only=bool(getattr(prediction, "topical_only", False)),
        rationale=rationale or "the auditor returned no rationale",
        raw_reply=raw_reply,
    )


def audit_taxonomy(
    draft: TaxonomyDraft,
    draft_id: str,
    corpus: ReadOnlyCorpus,
    *,
    predict: _Predictor,
    model: str = DEFAULT_MODEL,
) -> TaxonomyAudit:
    """Challenge every contract in a draft, one pass each.

    A contract whose audit fails is recorded as ``uncertain`` rather than
    skipped: an absent finding reads as a contract nobody objected to, and the
    freeze gate would then pass on the strength of a call that never happened.
    """
    assigned = _draft_members(draft)
    all_readable = set(corpus.readable_trace_ids())
    findings: list[AuditFinding] = []
    limitations: list[str] = []

    for contract in draft.contracts:
        members = assigned.get(contract.contract_id, ())
        pool = sorted(all_readable - set(members))
        outsiders = (
            corpus.sample_trace_ids(_OUTSIDER_SAMPLE, seed=draft.seed, exclude=tuple(members))
            if pool
            else ()
        )
        try:
            with ledger.stage("rlm_taxonomy_audit", contract_id=contract.contract_id, model=model):
                findings.append(
                    audit_contract(
                        contract,
                        corpus,
                        members=members,
                        outsiders=outsiders,
                        predict=predict,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - one failure must not lose the pass
            findings.append(
                AuditFinding(
                    contract_id=contract.contract_id,
                    recommendation="uncertain",
                    rationale=f"the audit of this contract failed and reached no verdict: {exc}",
                )
            )
            limitations.append(f"the audit of {contract.contract_id} failed: {exc}")

    if findings:
        limitations.append(
            "audit output is advisory and uncalibrated: it never edits a contract, and "
            "it has not been measured against labelled same-family pairs"
        )
    return TaxonomyAudit(
        draft_id=draft_id,
        findings=tuple(findings),
        model=model,
        prompt_digest=prompt_digest(model),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def _draft_members(draft: TaxonomyDraft) -> dict[str, tuple[str, ...]]:
    """The membership the discovery loop actually ended with.

    Read from ``draft.assignments``, which is the placement the run held when it
    stopped. The obvious alternative — replaying the chunk assignments in order —
    is wrong, and quietly so: a MERGE moves the consumed family's members onto
    the survivor without re-listing them in any later chunk, so a replay shows
    them still under a contract that no longer exists. A reviewer then sees a
    ghost family beside the real one and a survivor missing the members it
    absorbed, which is the opposite of what the merge decided.

    Falls back to the replay only for drafts written before ``assignments`` was
    stored, where it is the sole record there is; such a draft predates merges
    being applied at all, so the replay is accurate for it.
    """
    if draft.assignments:
        placement = dict(draft.assignments)
    else:  # pragma: no cover - only reachable for drafts written before this field
        placement = {}
        for chunk in draft.chunks:
            placement.update(chunk.assignments)
            for trace_id in chunk.ambiguous_trace_ids + chunk.uncovered_trace_ids:
                placement.pop(trace_id, None)

    # A contract that no longer exists cannot have members. Anything pointing at
    # one is a bug rather than a finding, so it is dropped here instead of being
    # rendered as a family a reviewer might try to act on.
    live = {contract.contract_id for contract in draft.contracts}
    grouped: dict[str, list[str]] = {}
    for trace_id, contract_id in placement.items():
        if contract_id in live:
            grouped.setdefault(contract_id, []).append(trace_id)
    return {cid: tuple(sorted(traces)) for cid, traces in grouped.items()}


def resolve_findings(audit: TaxonomyAudit, resolutions: dict[str, str]) -> TaxonomyAudit:
    """Record how discovery answered each actionable finding.

    Separate from the audit that raised them so the original verdict is never
    overwritten: what was found and what was done about it are two facts, and a
    resolution that quietly edited the finding would leave no way to tell an
    addressed objection from one that was argued away.
    """
    updated = []
    for finding in audit.findings:
        resolution = resolutions.get(finding.contract_id, "").strip()
        if resolution and finding.demands_action:
            updated.append(finding.replace(resolved=True, resolution=resolution))
        else:
            updated.append(finding)
    return audit.replace(findings=tuple(updated))


def freeze_taxonomy(
    draft: TaxonomyDraft,
    draft_id: str,
    *,
    audit: TaxonomyAudit | None = None,
    audit_id: str | None = None,
    force: bool = False,
) -> FrozenTaxonomy:
    """Freeze a draft into a taxonomy an assignment can be made against.

    Refuses while an actionable audit finding is unresolved. ``force`` exists
    for the case where a reviewer has decided to freeze anyway — an incomplete
    artifact is still worth inspecting — and it records that decision as a
    limitation rather than hiding it.
    """
    if audit is not None:
        unresolved = audit.unresolved()
        if unresolved and not force:
            raise FreezeRefused(
                f"{len(unresolved)} audit finding(s) recommend revising or splitting a "
                f"contract and have not been resolved: "
                f"{', '.join(f.contract_id for f in unresolved)}"
            )

    limitations = list(draft.limitations)
    if not draft.complete:
        limitations.append(
            f"frozen from a draft that stopped on {draft.stop_reason.value} rather than "
            "converging; its contracts were still changing"
        )
    if audit is None:
        limitations.append(
            "no adversarial audit was run against this taxonomy; its contracts have not "
            "been challenged by an independent context"
        )
    elif audit.unresolved():
        limitations.append(
            f"frozen over {len(audit.unresolved())} unresolved audit finding(s) "
            "recommending revision or a split"
        )

    return FrozenTaxonomy(
        draft_id=draft_id,
        audit_id=audit_id,
        analysis_id=draft.analysis_id,
        view=draft.view,
        contracts=draft.contracts,
        complete=draft.complete,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def compute_audit_id(audit: TaxonomyAudit) -> str:
    digest = hashlib.sha256(audit.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-taxonomy-audit-{digest[:16]}"


def compute_taxonomy_id(taxonomy: FrozenTaxonomy) -> str:
    """Content-addressed over the contracts, so an id names exact wording.

    Derived from the semantic claims rather than the whole model: an assignment
    naming a taxonomy id must be naming the text it classified against, and the
    draft and audit ids that produced it are lineage, not content.
    """
    payload = json.dumps(
        {
            "view": taxonomy.view.value,
            "contracts": [
                contract.model_dump(mode="json")
                for contract in sorted(taxonomy.contracts, key=lambda c: c.contract_id)
            ],
        },
        sort_keys=True,
    )
    return f"rlm-taxonomy-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def save_audit(audit: TaxonomyAudit, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_audit_id(audit),
        kind="rlm_taxonomy_audit",
        parent_artifact_id=audit.draft_id,
        payload=audit.model_dump_json().encode("utf-8"),
        summary={
            "findings": len(audit.findings),
            "actionable": sum(1 for f in audit.findings if f.demands_action),
            "topical_only": sum(1 for f in audit.findings if f.topical_only),
        },
    )


def save_taxonomy(taxonomy: FrozenTaxonomy, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_taxonomy_id(taxonomy),
        kind="rlm_taxonomy",
        parent_artifact_id=taxonomy.draft_id,
        payload=taxonomy.model_dump_json().encode("utf-8"),
        summary={"contracts": len(taxonomy.contracts), "complete": int(taxonomy.complete)},
    )


def load_audit(audit_id: str, store: DerivedStore) -> TaxonomyAudit:
    return TaxonomyAudit.model_validate_json(store.read_payload(audit_id))


def load_taxonomy(taxonomy_id: str, store: DerivedStore) -> FrozenTaxonomy:
    return FrozenTaxonomy.model_validate_json(store.read_payload(taxonomy_id))
