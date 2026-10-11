"""World state for mode B: a JSON object the tool simulator must answer from and may only change via ops.

StateGen (2606.16307): an authoritative state object ("backend is truth") removes most tool hallucinations.
The simulator proposes ops; this module applies them, so the model never rewrites state wholesale.
"""
from __future__ import annotations

import copy
from typing import Any


def _split(path: str) -> list[str]:
    return [p for p in path.split(".") if p != ""]


def _child(node: Any, key: str, create: bool) -> Any:
    if isinstance(node, list):
        return node[int(key)] if key.lstrip("-").isdigit() and abs(int(key)) <= len(node) else None
    if isinstance(node, dict):
        if key not in node and create:
            node[key] = {}
        return node.get(key)
    return None


def apply_ops(state: dict, ops: list[dict]) -> tuple[dict, list[str]]:
    """Apply set/delete/append ops on dotted paths. Returns (new_state, errors). Bad ops are skipped."""
    new = copy.deepcopy(state)
    errors = []
    for op in ops or []:
        kind, path = op.get("op"), op.get("path", "")
        parts = _split(path)
        if kind not in ("set", "delete", "append") or not parts:
            errors.append(f"invalid op: {op}")
            continue
        parent = new
        for key in parts[:-1]:
            parent = _child(parent, key, create=(kind == "set"))
            if parent is None:
                break
        last = parts[-1]
        if parent is None or not isinstance(parent, (dict, list)):
            errors.append(f"missing parent for {path}")
            continue
        try:
            if kind == "set":
                if isinstance(parent, list):
                    parent[int(last)] = op.get("value")
                else:
                    parent[last] = op.get("value")
            elif kind == "delete":
                if isinstance(parent, list):
                    parent.pop(int(last))
                elif last in parent:
                    del parent[last]
                else:
                    errors.append(f"delete of missing {path}")
            else:  # append
                target = parent[int(last)] if isinstance(parent, list) else parent.setdefault(last, [])
                if not isinstance(target, list):
                    errors.append(f"append to non-list {path}")
                else:
                    target.append(op.get("value"))
        except (ValueError, IndexError, KeyError) as e:
            errors.append(f"{kind} {path}: {e}")
    return new, errors


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out or ({prefix: {}} if prefix else {})
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}.{i}" if prefix else str(i)))
        return out or ({prefix: []} if prefix else {})
    return {prefix: obj}


def _norm(v: Any) -> str:
    return str(v).strip().lower()


def state_match(expected: dict, actual: dict) -> float | None:
    """Fraction of expected leaf values reproduced in actual. None when nothing is expected."""
    exp = flatten(expected or {})
    if not exp:
        return None
    act = flatten(actual or {})
    hits = sum(1 for k, v in exp.items() if k in act and _norm(act[k]) == _norm(v))
    return hits / len(exp)
