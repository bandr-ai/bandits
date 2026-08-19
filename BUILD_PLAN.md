# Build plan

Design rationale lives in [docs/PLAN.md](docs/PLAN.md). This file is the build:
what gets written, by whom, and the rules that keep parallel work from colliding.

## Rules for every contributor (human or agent)

1. **`bandits/contracts.py` is frozen.** Every stage speaks these types and only
   these types. If you believe a contract is wrong, say so — do not edit it.
2. **Own your directory, touch nothing else.** Do not edit `pyproject.toml`,
   another module's files, or the fixtures.
3. **Tests live beside the code** as `*_test.py` inside your own directory, plus
   integration tests in `tests/` only if you own that file name.
4. **Develop against the golden corpus** in `tests/fixtures/`. `expected.json`
   is the ground truth your stage must reproduce. Regenerate with
   `python tests/fixtures/make_corpus.py` — never hand-edit the generated files.
5. **No network, no API keys, no LLM calls** anywhere in stages 1–7. The whole
   reconstruction half is deterministic. That is the point: it is what makes the
   fidelity gate meaningful.
6. **Fail loudly.** A record you cannot handle becomes an explicit issue or an
   exception. Never silently skip, never invent a plausible default.
7. Run `.venv/bin/python -m pytest` and `.venv/bin/ruff check .` before you finish.

## Module ownership

| # | Directory | Owns | Consumes | Produces |
|---|---|---|---|---|
| 1 | `bandits/ingest/` | OTLP + chat-json adapters, tool-call recovery, registry loading | raw files | `TraceCorpus`, `ToolProfile.declared_schema` |
| 2 | `bandits/surface/` | argument/response profiling, error modes, read/write/external classification | `TraceCorpus` | `ToolSurface` |
| 3 | `bandits/state/` | entity discovery by ID recurrence, keys, foreign keys, write→read relations | `TraceCorpus` + `ToolSurface` | `StateSchema` |
| 4 | `bandits/env/` | SQLite materialization, tool reimplementation, effect ledger, session API | `StateSchema` + `TaskCase` | live env + `EnvManifest` |
| 5 | `bandits/task/` + `bandits/verify/` | pre-state reconstruction, task mining, verifier synthesis, anti-cheat | `TraceCorpus` + `StateSchema` | `TaskCase`, `Verifier` |
| 6 | `bandits/fidelity/` + `bandits/cli.py` | trace replay, per-tool divergence, accept/reject, CLI wiring | everything | `FidelityReport` |

Stage 4 defines the environment session interface that stage 6 replays against.
That contract is `bandits/env/interface.py` and stage 4 owns it.

## The loop, end to end

```
bandits ingest   traces.otlp.jsonl --tools tools.json   ->  corpus.json
bandits surface  corpus.json                            ->  surface.json
bandits schema   corpus.json surface.json               ->  schema.json
bandits tasks    corpus.json schema.json                ->  tasks.json
bandits verify   tasks.json corpus.json schema.json     ->  verifiers.json
bandits fidelity corpus.json schema.json tasks.json     ->  fidelity.json   # ACCEPT/REJECT
bandits export   --harbor out/                          ->  Harbor tasks
```

## Definition of done for v0

- `bandits fidelity` on the golden corpus reports **per-tool** rates and an
  overall figure, and rejects environments below threshold.
- Tool classes, entities, primary keys and error modes match `expected.json`.
- `store_policy` comes out as a **static snapshot**, not an invented table.
- `escalate_to_human` stays `UNKNOWN` and is never probed.
- `send_email` never "sends" — it lands in the effect ledger.
- A verifier for `ep-refund-ok` passes on a correct rollout and fails on a
  rollout that writes the store directly.

## Not in v0

Capability discovery (needs an LLM), LoRA/GRPO training and the MoE gate (need
GPUs), and active probing (needs a customer's staging environment). Each gets an
interface and a stub, clearly marked, so the shape is right when they land.
