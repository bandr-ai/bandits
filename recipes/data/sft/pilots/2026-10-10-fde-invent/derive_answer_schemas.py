"""Answer schemas for the FAA inspector seeds, keyed by sha256(system prompt).

Source of truth: agentic-test src/failure_analyzer/prompts.py (_OUTPUT_CONTRACT, extra_fields, SWEEP_SYSTEM,
ESCALATION_SYSTEM; fde-work/references/agentic-test @ 3fd5e445). Each seed's system prompt embeds its own
instance of that contract, so the schema is parsed from the "Return ONLY a JSON object" block of each prompt:
  "<one of: a, b>"                      -> enum (", or null ..." adds null)
  "<one of the candidate causes ...>"   -> enum of the "- CAUSE — ..." rows in the prompt (+ null)
  "<catalog root cause ...>"            -> same, from the cause catalog rows (+ null)
  "<0.0-1.0 ..."                        -> number in [0, 1]
  "<true only when ..." / "<true ..."   -> boolean
  "<one short ... sentence ...>"        -> string
Every key in the template is required; extra keys (e.g. divergence's step_id) are allowed.
usage: python derive_answer_schemas.py out/seeds.jsonl out/heldout.jsonl > answer_schemas.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

KEY = re.compile(r'^\s*"(\w+)":\s*(.*?),?\s*$')
ROW = re.compile(r"^- ([A-Z][A-Z0-9_]+) — ", re.MULTILINE)


def schema_for(system: str) -> dict | None:
    i = system.rfind("Return ONLY a JSON object")
    if i < 0:
        return None
    block = system[i:]
    start, end = block.index("{"), block.rindex("}")
    rows = ROW.findall(system)
    props = {}
    for line in block[start + 1:end].splitlines():
        m = KEY.match(line)
        if not m:
            continue
        key, spec = m.groups()
        spec = spec.strip().strip('"')
        nullable = "or null" in spec
        if spec.startswith("<one of:"):
            vals = spec[len("<one of:"):].split(", or null")[0].rstrip(">").strip()
            enum = [v.strip() for v in vals.split(",") if v.strip() and v.strip() != "null"]
            props[key] = {"enum": enum + ([None] if nullable or not enum else [])}
        elif spec.startswith(("<one of the candidate causes", "<catalog root cause")):
            if not rows:
                raise ValueError(f"no cause rows in prompt for {key}")
            props[key] = {"enum": sorted(set(rows)) + [None]}
        elif spec.startswith("<0.0-1.0"):
            props[key] = {"type": "number", "minimum": 0, "maximum": 1}
        elif spec.startswith("<true"):
            props[key] = {"type": "boolean"}
        elif "sentence" in spec:
            props[key] = {"type": "string"}
        else:
            props[key] = {}
    return {"type": "object", "required": list(props), "properties": props}


out = {}
for path in sys.argv[1:]:
    for line in Path(path).read_text().splitlines():
        s = json.loads(line)
        sch = schema_for(s.get("system", ""))
        if sch is None:
            print(f"no contract in system prompt of {s['id']}", file=sys.stderr)
            continue
        out[hashlib.sha256(s["system"].encode()).hexdigest()] = sch
print(json.dumps(out, indent=1))
