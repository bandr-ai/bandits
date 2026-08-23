"""Build a bandits tool registry (tools.json) from a τ²-bench domain's tools.py.

τ²-bench tags every tool method with @is_tool(ToolType.READ|WRITE|GENERIC) --
ground truth we should feed straight into bandits' trusted-registry override
(ToolClass) instead of re-deriving it empirically. GENERIC tools (calculate,
transfer_to_human_agents) don't map onto read/write/external/unknown, so we
leave tool_class undeclared for them and let bandits' evidence-based
classifier decide, same as it does for escalate_to_human in the fixture.

Usage:
    python scripts/tau2/extract_tools.py \
        /tmp/tau2-bench/src/tau2/domains/retail/tools.py \
        -o work/tau2-retail/tools.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TOOL_RE = re.compile(
    r"@is_tool\(ToolType\.(?P<type>\w+)\)\s*\n\s*def (?P<name>\w+)\(",
)

_TYPE_MAP = {"READ": "read", "WRITE": "write"}  # GENERIC/THINK left undeclared


def extract(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    entries = []
    for match in _TOOL_RE.finditer(text):
        name = match.group("name")
        if name.startswith("_"):
            continue
        entry: dict = {"name": name, "input_schema": {"type": "object", "properties": {}}}
        tool_class = _TYPE_MAP.get(match.group("type"))
        if tool_class:
            entry["tool_class"] = tool_class
        entries.append(entry)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tools_py", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    entries = extract(args.tools_py)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    declared = sum(1 for e in entries if "tool_class" in e)
    print(f"wrote {len(entries)} tools ({declared} with a declared tool_class) -> {args.out}")


if __name__ == "__main__":
    main()
