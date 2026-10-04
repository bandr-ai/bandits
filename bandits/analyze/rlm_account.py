"""Per-run accounts: what was requested, what happened, and what supports it.

A family is only as good as the readings of the runs it groups, and the old
loop never wrote those readings down: it went straight from raw lines to
contracts, so a run's interpretation existed only inside one model call and
was lost with it. An account is that reading made explicit and checkable —
intent, execution and result, each citing index-issued evidence references,
with unknowns allowed as answers rather than treated as schema failures.

Two shapes, deliberately. :class:`ProposedAccount` is exactly what the model
must SUBMIT, typed so DSPy's own SUBMIT validation feeds errors back inside the
loop. :class:`RunAccount` is what the host persists: it adds identity (run,
corpus, index, view, prompt and settings versions) and completion provenance
that only the host may set — how the answer was produced, which ranges were
actually retrieved, and whether host validation accepted it. The model supplies
interpretations and references; it never supplies its own provenance.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ConfigDict, Field

from bandits.traces import Contract

if TYPE_CHECKING:
    from bandits.analyze.rlm_evidence import EvidenceCatalog

ACCOUNT_SCHEMA_VERSION = 1


class _Proposed(Contract):
    # Extra keys are ignored rather than refused: a model adding a field it was
    # not asked for has still answered, and the verbatim candidate keeps it.
    model_config = ConfigDict(frozen=True, extra="ignore")


class ProposedIntent(_Proposed):
    statement: str | None = None
    """The request as recorded, verbatim or nearly so, when one was recorded."""

    candidate_goal: str | None = None
    """Your reading of the work this run was invoked to perform, at the scope the
    evidence supports. Tentative unless status says otherwise."""

    status: Literal["declared", "inferred", "unknown"]
    parameters: dict[str, str] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    required_outcome: str | None = None
    """What completion would have to establish, when the evidence states it."""

    evidence_refs: list[str] = Field(default_factory=list)


class ProposedMilestone(_Proposed):
    description: str = Field(min_length=1)
    span_refs: list[str] = Field(default_factory=list)
    """Event span ids (from inspect_run) or evidence refs this milestone rests on."""


class ProposedExecution(_Proposed):
    milestones: list[ProposedMilestone] = Field(default_factory=list)


class ProposedResult(_Proposed):
    observations: list[str] = Field(default_factory=list)
    assessment: Literal["supported_complete", "supported_incomplete", "unknown"]
    evidence_refs: list[str] = Field(default_factory=list)
    evaluator_claims: list[str] = Field(default_factory=list)
    """Claims some component made about success — kept apart from your assessment."""


class ProposedLimitation(_Proposed):
    kind: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ProposedAccount(_Proposed):
    """One run's account exactly as the model must SUBMIT it."""

    run_id: str
    intent: ProposedIntent
    execution: ProposedExecution
    result: ProposedResult
    limitations: list[ProposedLimitation] = Field(default_factory=list)
    unresolved_refs: list[str] = Field(default_factory=list)
    """Refs you know matter and did not read, or could not resolve."""


class AccountIdentity(Contract):
    """Everything an account's validity depends on. A mismatch forbids reuse."""

    run_id: str
    corpus_version: str
    index_version: int
    view_policy: str
    schema_version: int = ACCOUNT_SCHEMA_VERSION
    prompt_version: int
    prompt_digest: str
    model: str
    settings_digest: str

    def key(self) -> str:
        payload = json.dumps(self.model_dump(mode="json", exclude={"run_id"}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Completion(Contract):
    """How this account was produced. Set by the host, never by the model."""

    mode: Literal["submit", "extract", "error", "unknown"]
    """``submit``: the model called SUBMIT and DSPy accepted the types.
    ``extract``: the iteration cap forced DSPy's fallback extraction.
    ``error``: the invocation raised. ``unknown``: recorded before provenance existed."""

    iterations_used: int | None = None
    iterations_to_submit: int | None = None
    """Set only when the model submitted; a capped attempt has no convergence time."""

    submit_rejections: tuple[str, ...] = ()
    """Type errors DSPy fed back after a SUBMIT it refused, in order."""

    inspected_refs: tuple[str, ...] = ()
    inspected_ranges: tuple[str, ...] = ()
    """``ref[start:end]`` actually retrieved through get_evidence."""

    unresolved_refs: tuple[str, ...] = ()
    error: str = ""


class RunAccount(Contract):
    """A persisted account. ``status`` says whether it may be used."""

    identity: AccountIdentity
    attempt: int = Field(default=1, ge=1)
    status: Literal["accepted", "rejected", "quarantined", "failed"]
    """``accepted``: SUBMITted and host-validated, still a model interpretation.
    ``rejected``: SUBMITted but failed host validation; candidate preserved.
    ``quarantined``: produced by fallback extraction; inspectable, never applied.
    ``failed``: no candidate."""

    account: ProposedAccount | None = None
    candidate: str = ""
    """Exactly what was returned, serialized, whatever its status."""

    validation_errors: tuple[str, ...] = ()
    completion: Completion
    failure_kind: str = ""
    """For non-accepted accounts: ``provider_error``, ``budget:<stop reason>``,
    ``no_submit`` (fallback extraction), ``truncated`` or ``host_rejected``."""

    llm_calls: int | None = None
    cost_usd: float | None = None
    tokens: dict[str, int] = Field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def eligible_for_families(self) -> bool:
        """Accepted and with an intent reading to group on.

        An accepted account with unknown intent is a successful account — the
        honest answer — and still not a basis for a task assignment.
        """
        return (
            self.status == "accepted"
            and self.account is not None
            and self.account.intent.status != "unknown"
        )

    def unassigned_reason(self) -> str | None:
        """Why this run cannot be placed in a family, when it cannot."""
        if self.status in ("failed", "rejected"):
            return "processing_failure"
        if self.status == "quarantined":
            return "quarantined_extract"
        if self.account is not None and self.account.intent.status == "unknown":
            if any(item.kind == "contradictory_evidence" for item in self.account.limitations):
                return "contradictory_evidence"
            return "missing_intent"
        return None


def _refs_of(account: ProposedAccount) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = [
        ("intent.evidence_refs", r) for r in account.intent.evidence_refs
    ]
    found += [("result.evidence_refs", r) for r in account.result.evidence_refs]
    for i, limitation in enumerate(account.limitations):
        found += [(f"limitations[{i}].evidence_refs", r) for r in limitation.evidence_refs]
    found += [("unresolved_refs", r) for r in account.unresolved_refs]
    return found


def validate_account(
    account: ProposedAccount, *, run_id: str, catalog: EvidenceCatalog
) -> tuple[str, ...]:
    """Host checks a typed SUBMIT cannot make: identity, references, provenance.

    Every cited ref must be one the index issued for *this* run; every
    milestone ref must be an event span id or an evidence ref of this run. A
    ``declared`` intent must cite a source that records a request as such — a
    parsed context payload, an internal prompt or a clue can support an
    inferred intent, never a declared one. Passing here makes an account
    structurally sound; it says nothing about whether its reading is right.
    """
    from bandits.analyze.rlm_evidence import declared_origin

    errors: list[str] = []
    if account.run_id != run_id:
        errors.append(f"run_id is {account.run_id!r}; this invocation is for {run_id!r}")
        return tuple(errors)
    index = catalog.index(run_id)
    span_ids = index.span_ids()
    for where, ref in _refs_of(account):
        if not catalog.resolves(run_id, ref):
            errors.append(f"{where}: {ref!r} is not a ref the index issued for this run")
    for i, milestone in enumerate(account.execution.milestones):
        for ref in milestone.span_refs:
            if ref not in span_ids and not catalog.resolves(run_id, ref):
                errors.append(
                    f"execution.milestones[{i}].span_refs: {ref!r} is neither an event span id "
                    "nor an evidence ref of this run"
                )
    if account.intent.status == "declared":
        if not any(
            declared_origin(catalog.origin_of(run_id, r)) for r in account.intent.evidence_refs
        ):
            errors.append(
                "intent.status is 'declared' but no intent evidence ref records a request as "
                "such (a declared task field or a recorded user turn); parsed payloads, internal "
                "prompts and clues support 'inferred' at most"
            )
    if account.intent.status != "unknown" and not account.intent.evidence_refs:
        errors.append("intent.status is not 'unknown' but cites no evidence_refs")
    if account.result.assessment != "unknown" and not account.result.evidence_refs:
        errors.append("result.assessment is not 'unknown' but cites no evidence_refs")
    return tuple(errors)


def coerce_account(raw: Any) -> ProposedAccount | None:
    """A model-returned account as the typed model, or None when it is not one."""
    if isinstance(raw, ProposedAccount):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        raw = dump(mode="json")
    if not isinstance(raw, dict):
        return None
    try:
        return ProposedAccount.model_validate(raw)
    except ValueError:
        return None


def serialize_candidate(raw: Any) -> str:
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            raw = dump(mode="json")
        except Exception:  # noqa: BLE001 - the record matters more than its shape
            return str(raw)
    try:
        return json.dumps(raw, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(raw)


class AccountSet(Contract):
    """Reusable accounts for one identity, persisted as a small derived artifact.

    References the corpus by version; never copies it. Every account here
    shares ``identity_key``: an artifact mixing identities could not say which
    prompt or settings produced which reading.
    """

    schema_version: int = 1
    analysis_id: str
    identity_key: str
    accounts: tuple[RunAccount, ...] = ()
    superseded_by: str | None = None

    def accepted(self) -> dict[str, RunAccount]:
        return {a.run_id: a for a in self.accounts if a.status == "accepted"}


def account_set_id(accounts: AccountSet) -> str:
    digest = hashlib.sha256(accounts.model_dump_json().encode("utf-8")).hexdigest()
    return f"rlm-account-set-{digest[:16]}"


def family_row(account: RunAccount, status: str) -> dict[str, Any]:
    """What family formation reads for one run: intent and execution, not result.

    Result and evaluator claims are left out on purpose: a successful and a
    failed attempt at the same task belong together, and a grouping that could
    see the outcome could group by it.
    """
    if account.account is None:
        raise ValueError(f"run {account.run_id} has no account to group on")
    intent = account.account.intent
    return {
        "trace_id": account.run_id,
        "status": status,
        "intent": intent.model_dump(mode="json"),
        "milestones": [m.description for m in account.account.execution.milestones],
        "limitations": [f"{item.kind}: {item.detail}" for item in account.account.limitations],
    }


def latest_by_run(accounts: Sequence[RunAccount]) -> dict[str, RunAccount]:
    """The latest attempt per run; an accepted attempt is never displaced by a later failure."""
    chosen: dict[str, RunAccount] = {}
    for account in accounts:
        current = chosen.get(account.run_id)
        if current is None or (current.status != "accepted" and account.attempt >= current.attempt):
            chosen[account.run_id] = account
    return chosen
