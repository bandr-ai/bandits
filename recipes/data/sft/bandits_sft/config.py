"""Run configuration, loaded from a JSON file.

With every feature off, the run is plain baseline: each seed is the one-shot example of a single LLM call that
writes a whole new trajectory, followed by baseline's rule-based post-processing. Each feature adds one
improvement so it can be ablated against that baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_PERSONA_WEIGHTS = {
    "cooperative": 0.30, "vague": 0.10, "impatient": 0.10, "withholding": 0.10,
    "goalpost_shift": 0.10, "informal_typos": 0.10, "frustrated": 0.10, "multi_request": 0.10,
}
DEFAULT_STRATEGY_WEIGHTS = {
    "rephrase": 0.15, "entity_swap": 0.20, "extend": 0.15, "compose": 0.10,
    "variant_outcome": 0.15, "new_scenario": 0.25,
}


@dataclass
class Features:
    check_seeds: bool = False  # baseline LLM pre-filter on seeds (completeness / logic / format)
    strategies: bool = False   # mix of grounded generation strategies instead of only "new scenario" (BeyondWeb)
    spec: bool = False         # scenario spec (goal, facts, initial/expected state) written first, in batches
    persona: bool = False      # realistic user personas + real user-turn style examples; applied only to
                               # seeds that are real conversations (>= 2 user turns) unless persona_all_seeds
    failure: bool = False      # inject one tool failure at a sampled call
    retrieval: bool = False    # real recorded tool results as format reference
    loop: bool = False         # per-step generation instead of one call; style set by Config.loop_style
    provenance: bool = False   # drop traces whose ID-like tool arguments come from nowhere
    judge: bool = False        # LLM audit against the spec + checklist; OFF by default
    judge_gate: bool = False   # False (default): the judge only writes critique (meta.verify.judge) for the analyzer's
                               # diagnosis; it never changes `kept` and its failures are never eval errors. Rule checks
                               # decide. True restores the old behaviour where the judge's verdict gates `kept`.
    rubric_judge: bool = False  # seed expansion: the rubric judge (Config.rubric_path) gates `kept`: only label "good" is kept.
                               # Runs after the rule checks; a failed call keeps nothing (fail closed). See rubric_judge.py.
    change_gate: bool = True   # seed expansion: drop a trace whose inputs equal the seed's after cosmetic normalization
                               # (nothing was varied) or whose assistant side is near-identical to the seed's (nothing to
                               # learn). Code only, before any judge call. Threshold: Config.min_assistant_change.
    lesson_dedup: bool = True  # seed expansion: after generation, one judge-model call groups conversations of one seed that teach the same lesson
    near_dedup: bool = False   # near-duplicate removal on top of baseline's exact dedup
    input_json_gate: bool = True  # seed expansion: reject a trace whose user-turn JSON parsed in the seed but no longer parses
                               # (damaged while copying; code, before any judge call). Off: only a soft flag the judges see.
    consistency_check: bool = True  # with the rubric judge, a second call at the same time (consistency.check): lists each
                               # observation's facts and rejects contradictions, impossible results, unsupported claims
    split_answer: bool = False  # seed expansion: the generator writes the trace up to the last observation, without the
                               # intended outcome; the agent model then continues from the real system prompt and tools,
                               # its further tool calls answered by a simulator (rollout.agent_rollout).


@dataclass
class ModelCfg:
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 1.0  # None = omit (some reasoning models reject it)
    max_tokens: int = 16000
    max_tokens_param: str = "max_tokens"  # or "max_completion_tokens"
    json_mode: bool = True  # send response_format=json_object when JSON is expected
    no_cache: bool = True   # ask a LiteLLM gateway not to serve cached responses (identical prompts must still vary)
    extra: dict = field(default_factory=dict)  # passed through to the API call
    stream: bool = False  # stream the response: keeps the connection alive through long reasoning calls that a gateway would otherwise cut
    # USD per million tokens. Used to compute cost when the provider returns none (direct Fireworks/OpenAI-style APIs);
    # a gateway-reported cost always wins. Leave unset to log cost as unknown.
    price_input_per_mtok: float | None = None
    price_cached_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None


@dataclass
class Config:
    seeds_path: str
    out_dir: str = "out"
    seed_format: str = "auto"  # auto | canonical (incl. OpenAI/LangChain-style messages) | sharegpt
    tool_defs_path: str | None = None  # JSON with tool schemas, for traces that list tools by name only
    baseline_prompt: str = "fixed"  # "fixed": baseline's prompt + format rules; "original": baseline's prompt only
    prompt_source: str = "baseline"  # "baseline": baseline's prompt; "analyzer": the prompt written by `analyze` (ANALYZER_SPEC.md)
    generation_attempts: int = 1  # mode baseline: regenerate when the output cannot be used (no user turn,
                                  # baseline's markup filter, no final reply); baseline itself makes one attempt
    target_count: int = 1000   # traces wanted back (e.g. 100 golden traces -> 1000)
    overgen: float = 1.5       # generate target_count * overgen, then filter and select down (Datology: curate from a bigger pool)
    features: Features = field(default_factory=Features)
    loop_frac: float = 0.3     # share of jobs generated per step when features.loop is on
    loop_style: str = "split"  # "reference_env": reference simulation's simulator (one model plays user and tools, seed as reference)
                               # "split": separate user-sim, agent, and stateful tool-sim (needs features.spec)
    failure_rate: float = 0.25
    persona_all_seeds: bool = False
    persona_weights: dict = field(default_factory=lambda: dict(DEFAULT_PERSONA_WEIGHTS))
    strategy_weights: dict = field(default_factory=lambda: dict(DEFAULT_STRATEGY_WEIGHTS))
    checklist: list[str] | None = None  # None = prompts.DEFAULT_CHECKLIST
    retrieved_obs_per_tool: int = 3
    real_user_examples: int = 4
    max_user_turns: int = 8    # mode B
    max_tool_rounds: int = 8   # mode B, agent rounds per user turn
    min_assistant_turns: int = 1  # 1: single-turn agents (one answer, no tools) are valid; the judge checks the rest
    keep_failures: bool = True  # keep judged failures (with bad_steps) for loss masking
    decontam_paths: list[str] = field(default_factory=list)  # eval traces; seeds and outputs overlapping them are dropped
    holdout_frac: float = 0.0  # share of seeds (by id) held out at ingest: kept out of seeds.jsonl, banks and the analyzer
    answer_schemas_path: str | None = None  # JSON {sha256(system prompt): JSON schema}; the final reply must be JSON matching it
    outcome_fields: list[str] = field(default_factory=list)  # final-answer keys whose values the report counts
    tool_optional_args: dict = field(default_factory=dict)  # {tool: [arg]}: required in the schema but omitted by real seeds; set from the seeds
    tool_loose_type_args: dict = field(default_factory=dict)  # {tool: [arg]}: real seeds pass another type than the schema; set from the seeds
    tool_arg_defaults: dict = field(default_factory=dict)  # {tool: {arg: default}}: the harness fills these before validation
    tool_result_shapes: dict = field(default_factory=dict)  # {tool or "*": [[top-level keys], ...]}: allowed real result shapes
    tool_result_prefix: str | None = None  # regex every tool result starts with; group 1 = 1-based result index (e.g. "[r:N] ")
    min_assistant_change: float = 0.02  # change_gate: the assistant side must differ from the seed's by at least this
                                        # (1 - difflib ratio on normalized text); 0 disables the assistant check
    rubric_path: str | None = None  # rubric for features.rubric_judge (required: rubric_loop induce --v2, or a hand-written one)
    analyzer_seeds: int = 20   # legacy config compatibility; analyzer now sees all supplied dev seeds
    analyzer_rounds: int = 2   # analyzer versions at most (v1 + revisions from diagnosis)
    analyzer_round2_below: float = 0.5  # run a revision round only when the previous version's usable rate is below this
                                        # (0 = never revise, >1 = always run every round)
    analyzer_stop_below: float = 0.3  # pipeline: stop before production when the chosen version's usable rate is below this
    pilot_jobs: int = 20       # jobs per round, over dev seeds
    per_seed_variations: bool = True  # after the main analyzer call, one short call per seed proposes more variations
    variations_per_seed: int = 6
    max_jobs_per_seed: int = 3  # per round
    split_max_rounds: int = 3  # split_answer: tool rounds the agent may take (simulated results) before its final reply
    max_variation_share: float = 1.0  # batch: at most this share of jobs per variation family (name before "__")
    case_block: bool = False   # analyzer generator writes a "case" (the changed situation, no answer) before the messages
    combo_share: float = 0.0   # share of analyzer jobs that apply one of the analyzer's variation combinations
    quarantine_seeds: dict[str, str] = field(default_factory=dict)  # seed id (or unique prefix) -> reason; kept on disk, never used as a seed
    heldout_jobs_per_seed: int = 3  # final comparison on held-out seeds: chosen analyzer prompt vs baseline_prompt=fixed
    # pipeline (bandits-sft pipeline): see pipeline.py
    judge_revise_rounds: int = 1  # REVISE rounds of the judge self-test loop (round 1 gives nearly all the gain)
    selftest_seeds: int = 8       # seeds the self-test dev cases are built from (7 cases each)
    selftest_test_seeds: int = 0  # other seeds for held-out self-test cases; report only, so off by default
    variation_mix: dict = field(default_factory=lambda: {"old": 0.5, "task": 1 / 6, "evidence": 1 / 6, "stack": 1 / 6})
    proposals_per_call: int = 8   # variations asked per proposal call (bigger calls get slow)
    breaker_window: int = 20      # circuit breaker: last N production jobs ...
    breaker_max_bad: float = 0.5  # ... stop when more than this share failed (execution error, judge or check call error)
    # scenario recipe (scenario_pipeline.py)
    scenario_attempts_per_target: float = 2.0  # planned jobs per wanted trace; rejection never buys more than this
    coverage_max_chars: int = 1_000_000  # the coverage call sees at most this much seed text (round robin over agents, long traces cut)
    reference_max_chars: int = 20_000  # each reference trace shown to the planner / generator is cut to this
    workers: int = 16
    random_seed: int = 0
    models: dict[str, ModelCfg] = field(default_factory=dict)

    def model_for(self, role: str) -> ModelCfg:
        if role in self.models:
            return self.models[role]
        if "default" in self.models:
            return self.models["default"]
        raise KeyError(f"no model configured for role {role!r} and no 'default'")

    @property
    def out(self) -> Path:
        return Path(self.out_dir)


REMOVED_KEYS = {"blind_check", "blind_gate", "answer_samples"}  # removed checks; older configs still name them


def load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text())
    models = {k: ModelCfg(**v) for k, v in raw.pop("models", {}).items()}
    feats = Features(**{k: v for k, v in raw.pop("features", {}).items() if k not in REMOVED_KEYS})
    raw = {k: v for k, v in raw.items() if k not in REMOVED_KEYS}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**raw, features=feats, models=models)
    if cfg.baseline_prompt not in ("fixed", "original"):
        raise ValueError("baseline_prompt must be 'fixed' or 'original'")
    if cfg.prompt_source not in ("baseline", "analyzer"):
        raise ValueError("prompt_source must be 'baseline' or 'analyzer'")
    if cfg.loop_style not in ("split", "reference_env"):
        raise ValueError("loop_style must be 'split' or 'reference_env'")
    if feats.loop and cfg.loop_style == "split" and not feats.spec:
        raise ValueError("features.loop with loop_style 'split' needs features.spec (the tool simulator answers from the spec's state)")
    bad_mix = set(cfg.variation_mix) - {"old", "task", "evidence", "stack"}
    if bad_mix or sum(cfg.variation_mix.values()) <= 0:
        raise ValueError(f"variation_mix: keys from old/task/evidence/stack with positive total (unknown: {sorted(bad_mix)})")
    cfg.variation_mix = {k: v / sum(cfg.variation_mix.values()) for k, v in cfg.variation_mix.items()}
    for name in ("persona_weights", "strategy_weights"):
        w = getattr(cfg, name)
        total = sum(w.values())
        if total <= 0:
            raise ValueError(f"{name} must have positive total weight")
        setattr(cfg, name, {k: v / total for k, v in w.items()})
    return cfg
