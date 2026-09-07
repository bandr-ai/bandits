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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

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

DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
"""Matches the rubric judge's default, so one credential covers both passes."""

DEFAULT_MAX_ITERATIONS = 12
DEFAULT_MAX_LLM_CALLS = 30
"""Hard ceilings on one family's audit. The root model decides how to slice a
family, so cost is bounded here rather than by trusting it to be brief."""

PROMPT_VERSION = 1

_INSTRUCTION = """You are auditing one group of agent episodes that an automatic \
clustering step claims are all the same repeatable task.

The variable `members` is a list of dicts, each with keys: trace_id, instruction, \
normalized (the masked form clustering compared), tool_names, span_count.

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


class _Predictor(Protocol):
    """The one call this module makes, so tests need no model and no sandbox."""

    def __call__(self, *, members: str, question: str) -> Any: ...


def _member_view(family: TaskFamily, analysis: CorpusAnalysis) -> list[dict[str, Any]]:
    """What the model reads: one row per member, ordered like the family.

    The instruction, the masked form clustering compared, and structural shape.
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

    return predict


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

    try:
        prediction = predict(
            members=json.dumps(rows, indent=2, sort_keys=True, default=str),
            question=_INSTRUCTION,
        )
    except AuditError:
        raise
    except Exception as exc:  # noqa: BLE001 - any backend failure is one failure here
        raise AuditError(f"audit of {family.family_id} failed: {exc}") from exc

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
        except AuditError as exc:
            skipped.append(SkippedAudit(family_id=family.family_id, reason=str(exc)))
            if on_error is not None:
                on_error(family.family_id, str(exc))
            continue
        audits.append(audit)
        dropped = set(audit.outlier_trace_ids) - set(family.trace_ids)
        if dropped:  # pragma: no cover - _clean_ids already filters these
            limitations.append(f"audit of {family.family_id} named traces it does not contain")

    if audits:
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
