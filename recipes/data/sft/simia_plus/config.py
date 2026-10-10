"""Run configuration, loaded from a JSON file.

With every feature off, the run is plain Simia: each seed is the one-shot example of a single LLM call that
writes a whole new trajectory, followed by Simia's rule-based post-processing. Each feature adds one
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
    check_seeds: bool = False  # Simia LLM pre-filter on seeds (completeness / logic / format)
    strategies: bool = False   # mix of grounded generation strategies instead of only "new scenario" (BeyondWeb)
    spec: bool = False         # scenario spec (goal, facts, initial/expected state) written first, in batches
    persona: bool = False      # realistic user personas + real user-turn style examples; applied only to
                               # seeds that are real conversations (>= 2 user turns) unless persona_all_seeds
    failure: bool = False      # inject one tool failure at a sampled call
    retrieval: bool = False    # real recorded tool results as format reference
    loop: bool = False         # per-step generation instead of one call; style set by Config.loop_style
    provenance: bool = False   # drop traces whose ID-like tool arguments come from nowhere
    judge: bool = False        # LLM audit against the spec + checklist
    near_dedup: bool = False   # near-duplicate removal on top of Simia's exact dedup


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


@dataclass
class Config:
    seeds_path: str
    out_dir: str = "out"
    seed_format: str = "auto"  # auto | canonical (incl. OpenAI/LangChain-style messages) | sharegpt
    tool_defs_path: str | None = None  # JSON with tool schemas, for traces that list tools by name only
    simia_prompt: str = "fixed"  # "fixed": Simia's prompt + format rules; "original": Simia's prompt only
    generation_attempts: int = 1  # mode simia: regenerate when the output cannot be used (no user turn,
                                  # Simia's markup filter, no final reply); Simia itself makes one attempt
    target_count: int = 1000   # traces wanted back (e.g. 100 golden traces -> 1000)
    overgen: float = 1.5       # generate target_count * overgen, then filter and select down (Datology: curate from a bigger pool)
    features: Features = field(default_factory=Features)
    loop_frac: float = 0.3     # share of jobs generated per step when features.loop is on
    loop_style: str = "split"  # "simia_env": Simia-RL's simulator (one model plays user and tools, seed as reference)
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
    min_assistant_turns: int = 2
    keep_failures: bool = True  # keep judged failures (with bad_steps) for loss masking
    decontam_paths: list[str] = field(default_factory=list)  # eval traces; seeds and outputs overlapping them are dropped
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


def load_config(path: str | Path) -> Config:
    raw = json.loads(Path(path).read_text())
    models = {k: ModelCfg(**v) for k, v in raw.pop("models", {}).items()}
    feats = Features(**raw.pop("features", {}))
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**raw, features=feats, models=models)
    if cfg.simia_prompt not in ("fixed", "original"):
        raise ValueError("simia_prompt must be 'fixed' or 'original'")
    if cfg.loop_style not in ("split", "simia_env"):
        raise ValueError("loop_style must be 'split' or 'simia_env'")
    if feats.loop and cfg.loop_style == "split" and not feats.spec:
        raise ValueError("features.loop with loop_style 'split' needs features.spec (the tool simulator answers from the spec's state)")
    for name in ("persona_weights", "strategy_weights"):
        w = getattr(cfg, name)
        total = sum(w.values())
        if total <= 0:
            raise ValueError(f"{name} must have positive total weight")
        setattr(cfg, name, {k: v / total for k, v in w.items()})
    return cfg
