# SFT data recipe: seed traces → more traces (Simia+)

Give it N good agent traces and get M new ones back for SFT (e.g. 100 → 1,000). No environment is needed:
nothing is executed. Based on [Simia](https://arxiv.org/abs/2511.01824), whose code and prompts are copied in
(`simia_plus/simia.py`, `simia_plus/prompts.py`, MIT, `LICENSE-SIMIA`). Design and sources: [`SPEC.md`](SPEC.md).

Standalone: it does not import Bandits core.

## Install
```bash
cd recipes/data/sft
uv sync --extra dev
```

## Run
```bash
export OPENAI_API_KEY=...         # or any OpenAI-compatible endpoint via base_url / api_key_env
uv run simia-plus run --config configs/simia.json      # plain Simia (baseline)
uv run simia-plus run --config configs/simia_plus.json # Simia + improvements
```
Stages can also run one at a time: `ingest`, `plan`, `specs`, `generate`, `verify`, `select`. Every stage resumes.

Outputs in `out_dir`:
- `final/synthetic.jsonl`: canonical traces, each with `meta` (seed, strategy, persona, failure, spec, verify, bad_steps)
- `final/synthetic_sharegpt.jsonl`, `final/seeds_plus_synthetic_sharegpt.jsonl`: Simia / LLaMA-Factory format
- `final/report.json`: counts, reject reasons, diversity vs the seeds

## Seed formats
- Simia/ShareGPT: `{"system", "tools", "conversations": [{"from": "human|gpt|function_call|observation", "value"}]}`
- Canonical or OpenAI/LangChain-style `messages`, including content blocks and `tool_calls` with `args`
- LLM-gateway logs: `final_messages` plus `turns` (the last turn's response is appended)

Tools may be full schemas, or names only together with `tool_defs_path`.

## Models
One OpenAI-compatible model per role, with `default` as the fallback:
- `generator` (mode simia)
- `spec`
- `agent`, `user_sim`, `tool_sim` (loop modes)
- `judge`, `seed_check`

For reasoning models, set `"temperature": null` and `"max_tokens_param": "max_completion_tokens"`.

## Data handling
Seeds and generated traces are sent to the configured model endpoints. Scrub customer data first, or point
`base_url` at a model you host.

## Tests
```bash
uv run pytest      # offline: a fake LLM plays every role
uv run ruff check .
```
