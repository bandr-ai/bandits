"""results.jsonl rows and the status rules.

Status is always derived, never assigned freely:
  blocked   - structure/leak/exec failed, or an explicit blocked_reason is set
  validated - ONLY if exec gates ran and passed in a real Harbor run (and
              structure + leak passed). The host-local runner is a fast
              verifier check, so its passes stay `generated`.
  generated - structure + leak passed, exec gates not run / errored / local only
"""

from __future__ import annotations

from .models import ContractError


def new_row(task_id: str, ttype: str, sources: list, generator: str, gold: str) -> dict:
    return {
        "task_id": task_id,
        "type": ttype,
        "sources": sources,
        "generator": generator,
        "gold": gold,
        "gates": {"structure": "not_run", "leak": "not_run", "exec": "not_run"},
        "status": "generated",
    }


def has_harbor_evidence(g: dict) -> bool:
    """Only a real Harbor run leaves trial ids; a runner's name alone proves nothing."""
    return g.get("runner") == "harbor" and bool(g.get("trials"))


def derive_status(row: dict) -> tuple[str, str]:
    g = row["gates"]
    if row.get("blocked_reason"):
        return "blocked", row["blocked_reason"]
    if g["structure"] == "fail":
        return "blocked", "structure gate failed"
    if g["leak"] == "fail":
        return "blocked", "leak scan failed"
    if g["exec"] == "fail":
        return "blocked", "exec gate failed"
    if (
        g["exec"] == "pass"
        and g["structure"] == "pass"
        and g["leak"] == "pass"
        and has_harbor_evidence(g)
    ):
        return "validated", ""
    return "generated", ""


def finalize(row: dict) -> dict:
    status, reason = derive_status(row)
    row["status"] = status
    if status == "blocked":
        row["reason"] = reason
    else:
        row.pop("reason", None)
    return row


def check_invariants(rows: list[dict]) -> None:
    for r in rows:
        g = r["gates"]
        if r["status"] == "validated" and not (
            g["exec"] == "pass"
            and g["structure"] == "pass"
            and g["leak"] == "pass"
            and has_harbor_evidence(g)
        ):
            raise ContractError(
                f"task {r['task_id']} would be 'validated' without its execution gates having run and passed"
            )
        if r["status"] == "generated" and (g["structure"] != "pass" or g["leak"] != "pass"):
            raise ContractError(
                f"task {r['task_id']} is 'generated' but structure/leak did not pass"
            )
