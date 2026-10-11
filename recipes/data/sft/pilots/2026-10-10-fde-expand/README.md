# Pilot 2026-10-10: seed expansion on FAA seeds (v1–v3)

The first run of the seed-expansion analyzer. Same 12 dev / 5 held-out split as `../2026-10-10-fde-invent`. Config: `../next-seed-expansion/pilot_expand.json`. The analyzer gets the harness source (agentic-test `failure_analyzer` tools, cause catalog, `check_specs.py`, `shaping.py`, `tool_defs.json`). The judge sees the seed, the variation and the harness. deepseek-v4.1-flash on Relace. $0.57 over 130 calls. Traces stay in `fde-work/sft-analyzer-pilot/out_expand/`; held-out seeds not used.

Each `vN/` holds `harness.md`, `variations.json`, `prompt.txt`, `rationale.md` and `report.json`, plus `diagnosis.md` for v1 and v2.

| | v1 | v2 | v3 |
|---|---|---|---|
| kept / 20 | 14 | 18 | 17 |
| code rejects | 7 (missing `[obs:N]` prefix, wrong-inspector answer) | 0 | 0 |
| judge rejects | 2 | 2 | 3 |
| distinct decision cells (not in seeds) | 24 (7) | 30 (13) | 31 (15) |
| root_cause null among kept | 12/14 | 14/18 | 10/17 |
| template chars | 4.1k | 5.3k | 6.1k |

- **Context fidelity:** 0 failure-context fields changed and no platform changes across all three versions. The invent pilot rewrote whole worlds.
- **Diagnosis worked.** It caught the missing prefix, an answer value outside the element-presence inspector's vocabulary, and the missing `page_kind`. v2 fixed all three.
- **Chosen: v2** (provisional until spot-check). The run first picked v3 because the old dedup collapsed distinct expansions; fixed in `022027c`, and the reports here are recomputed.
- **Concentration:** one seed+variation pair appears 8 times across rounds, and one decision cell 11 times (keyboard seeds × `keyboard_shown_target_above`). Step 2 needs a per-pair cap, and 17 seeds limit variety.
- **`page_kind`:** the harness asks for it on `different_page` but tolerates it missing, and the reviewed seed omits it. It is not enforced in code.
