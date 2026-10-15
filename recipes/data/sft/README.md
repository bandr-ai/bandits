# Bandits trace generation

Give it N good agent traces and get M new ones back for SFT (e.g. 100 → 1,000). No environment is needed:
nothing is executed. Design and sources: [`SPEC.md`](SPEC.md).

Standalone: it does not import Bandits core.

Current seed-only flow: [ANALYZER_SPEC.md](ANALYZER_SPEC.md). End to end:
`bandits-sft pipeline --seeds <seeds.jsonl> --out <dir> --budget 5` with the shipped defaults in
[`bandits_sft/defaults/pipeline_fireworks.json`](bandits_sft/defaults/pipeline_fireworks.json).

## Install
```bash
cd recipes/data/sft
uv sync --extra dev
```

## Run
```bash
export OPENAI_API_KEY=...         # or any OpenAI-compatible endpoint via base_url / api_key_env
uv run bandits-sft run --config configs/baseline.json      # baseline configuration
uv run bandits-sft run --config configs/enhanced.json # enhanced configuration
```
Stages can also run one at a time: `ingest`, `plan`, `specs`, `generate`, `verify`, `select`. Every stage resumes.

Outputs in `out_dir`:
- `final/synthetic.jsonl`: canonical traces, each with `meta` (seed, strategy, persona, failure, spec, verify, bad_steps)
- `final/synthetic_sharegpt.jsonl`, `final/seeds_plus_synthetic_sharegpt.jsonl`: ShareGPT / LLaMA-Factory format
- `final/report.json`: counts, reject reasons, diversity vs the seeds

## Seed formats
- ShareGPT: `{"system", "tools", "conversations": [{"from": "human|gpt|function_call|observation", "value"}]}`
- Canonical or OpenAI/LangChain-style `messages`, including content blocks and `tool_calls` with `args`
- LLM-gateway logs: `final_messages` plus `turns` (the last turn's response is appended)

Tools may be full schemas, or names only together with `tool_defs_path`.

## Models
One OpenAI-compatible model per role, with `default` as the fallback:
- `generator` (mode baseline)
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

## Version 0.2 configuration

Use `bandits-sft` or `python -m bandits_sft`. The baseline configuration is `configs/baseline.json`; optional improvements are in `configs/enhanced.json`.

Configuration uses `baseline_prompt`, `prompt_source: "baseline"`, and `loop_style: "reference_env"` (or `"split"`). Generation modes are `baseline`, `reference_env`, and `split`.

When upgrading an earlier recipe installation, update imports and configuration keys and start with a new output directory. Saved jobs and checkpoints from earlier versions are not a supported migration path.
