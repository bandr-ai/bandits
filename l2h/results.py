"""results.jsonl rows and the status rules.

Status is always derived, never assigned freely:
  blocked     - leak or secret scan failed: the package must not be used as is
  exec_failed - the package is malformed (structure) or its exec gates failed, or its
                type failed sampled validation (blocked_reason): it does not work yet
  validated   - ONLY if exec gates ran and passed in a real Harbor run, oracle included
                (and structure + leak + secrets passed). The host-local runner is a fast
                verifier check, so its passes stay `generated`.
  generated   - structure + leak + secrets passed; exec gates not run / errored / local
                only / passed without an oracle (`no_oracle`)
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
        "gates": {
            "structure": "not_run",
            "leak": "not_run",
            "secrets": "not_run",
            "exec": "not_run",
        },
        "status": "generated",
    }


def has_harbor_evidence(g: dict) -> bool:
    """Only a real Harbor run leaves trial ids; a runner's name alone proves nothing."""
    return g.get("runner") == "harbor" and bool(g.get("trials"))


def derive_status(row: dict) -> tuple[str, str]:
    g = row["gates"]
    if g["leak"] == "fail":
        return "blocked", "leak scan failed"
    if g.get("secrets") == "fail":
        return "blocked", "secret scan failed"
    if row.get("blocked_reason"):
        return "exec_failed", row["blocked_reason"]
    if g["structure"] == "fail":
        return "exec_failed", "structure gate failed"
    if g["exec"] == "fail":
        return "exec_failed", "exec gate failed"
    if (
        g["exec"] == "pass"
        and g["structure"] == "pass"
        and g["leak"] == "pass"
        and g.get("secrets") == "pass"
        and has_harbor_evidence(g)
    ):
        return "validated", ""
    return "generated", ""


def finalize(row: dict) -> dict:
    status, reason = derive_status(row)
    row["status"] = status
    if status in ("blocked", "exec_failed"):
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
            and g.get("secrets") == "pass"
            and has_harbor_evidence(g)
        ):
            raise ContractError(
                f"task {r['task_id']} would be 'validated' without its execution gates having run and passed"
            )
        if r["status"] == "generated" and not (
            g["structure"] == "pass" and g["leak"] == "pass" and g.get("secrets") == "pass"
        ):
            raise ContractError(
                f"task {r['task_id']} is 'generated' but structure/leak/secrets did not pass"
            )
