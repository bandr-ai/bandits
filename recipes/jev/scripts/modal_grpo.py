"""Scoped Dr. GRPO run on one Modal H100: outcome-only or outcome + Jev step reward.

One container holds everything that would otherwise sit idle on its own GPU:
vLLM serves the policy (LoRA hot-swapped after each update), the policy is
trained with PEFT on the same GPU, and the frozen Jev judge scores steps
in-process. Only the retriever stays remote (`jev-retriever` must be deployed).
Both arms use the same questions per step for a given seed, the same
hyperparameters and the same evaluation; they differ only in the reward.

    uvx modal run recipes/jev/scripts/modal_grpo.py --arm outcome --seed 1 --steps 10
    uvx modal run recipes/jev/scripts/modal_grpo.py --arm step --seed 1 --steps 10

Metrics are appended per step to /runs/step-rl/grpo/<run>/metrics.jsonl on the
`jev-runs` volume, so an interrupted run keeps what it finished.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")
POLICY = "Qwen/Qwen3-4B-Instruct-2507"
POLICY_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
JUDGE = "Qwen/Qwen3.5-4B-Base"
JUDGE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
JUDGE_ADAPTER = "/runs/search-judge-pilot/checkpoints/step-50"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    # flash-linear-attention is Triton only; without it the judge's linear-attention layers
    # fall back to slow PyTorch code.
    .pip_install("vllm==0.31.0", "openai", "huggingface_hub", "peft", "typer", "rich", "flash-linear-attention")
    .env(
        {
            "PYTHONPATH": "/app",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        }
    )
    .add_local_dir(str(REPO / "recipes/jev/bandits_jev"), "/app/bandits_jev")
    .add_local_dir(str(REPO / "bandits"), "/app/bandits")
)
app = modal.App("jev-grpo", image=image)
runs = modal.Volume.from_name("jev-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("jev-hf-cache", create_if_missing=True)


@app.cls(
    gpu="H100",
    timeout=4 * 3600,
    scaledown_window=30,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
)
class Trainer:
    @modal.enter()
    def start(self) -> None:
        import subprocess
        import time
        import urllib.request

        self.server = subprocess.Popen(
            [
                "vllm", "serve", POLICY, "--revision", POLICY_REVISION, "--served-model-name", "base",
                "--dtype", "bfloat16", "--max-model-len", "32768", "--gpu-memory-utilization", "0.5",
                "--enable-lora", "--max-lora-rank", "32", "--max-loras", "2",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(PORT),
            ]
        )
        deadline = time.time() + 20 * 60
        while True:
            if self.server.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.server.returncode}")
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError("vllm did not become ready in 20 minutes") from None
                time.sleep(5)

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        from bandits_jev.hf_predictor import HFPredictor

        model = AutoModelForCausalLM.from_pretrained(POLICY, revision=POLICY_REVISION, dtype=torch.bfloat16).to("cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        lora = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(model, lora)
        self.judge = HFPredictor(JUDGE, revision=JUDGE_REVISION, adapter_path=JUDGE_ADAPTER)

    def _judge_score(self, steps: list[dict]) -> dict:
        from bandits_jev.step_judge import judge_step

        results = []
        for step in steps:
            judged = judge_step(
                self.judge, step["question"], [tuple(p) for p in step["previous"]],
                step["tool"], step["action"], step["observation"],
            )
            results.append({"probabilities": judged.probabilities, "score": judged.score, "reason": judged.reason})
        return {"judge": {"model": JUDGE, "revision": JUDGE_REVISION, "adapter": JUDGE_ADAPTER}, "results": results}

    async def _rollouts(self, tasks: list[dict], n: int, model_name: str, keep_tokens: bool, salt: str) -> list[dict]:
        import asyncio
        import hashlib

        import openai

        from bandits_jev.modal_searcher import ModalSearcher
        from bandits_jev.policy_client import judge_rollout, make_chat
        from bandits_jev.rollout import run_rollout

        client = openai.AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="unused", timeout=300)
        searcher, gate = ModalSearcher(), asyncio.Semaphore(64)

        async def judge_async(steps: list[dict]) -> dict:
            return self._judge_score(steps)

        async def one(task: dict, sample: int) -> dict:
            seed = int(hashlib.sha256(f"{salt}:{task['query_id']}:{sample}".encode()).hexdigest()[:8], 16)
            async with gate:
                result = await run_rollout(task, make_chat(client, model_name, seed, keep_tokens=keep_tokens), searcher)
            result["sample"] = sample
            return result

        from concurrent.futures import ThreadPoolExecutor

        # Retriever calls block in threads; the default pool is small enough that a few slow
        # calls queue every other rollout behind them.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=128))
        try:
            # Watchdog: a stalled batch ends the run instead of holding an idle GPU.
            results = await asyncio.wait_for(
                asyncio.gather(*(one(t, s) for t in tasks for s in range(n))), 60 * self.batch_minutes
            )
        except TimeoutError:
            raise RuntimeError(f"rollout batch stalled for {self.batch_minutes} minutes; stopping") from None
        # Judge after generation: the judge shares this GPU, so it runs while vLLM is idle.
        by_id = {t["query_id"]: t for t in tasks}
        for result in results:
            await judge_rollout(judge_async, by_id[result["query_id"]]["query"], result)
        return results

    def _load_adapter(self, name: str, path: str, previous: str | None) -> None:
        import urllib.request

        def post(endpoint: str, payload: dict) -> None:
            request = urllib.request.Request(
                f"http://localhost:{PORT}/v1/{endpoint}", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=120).read()

        post("load_lora_adapter", {"lora_name": name, "lora_path": path})
        if previous:
            post("unload_lora_adapter", {"lora_name": previous})

    @modal.method()
    async def train(self, config: dict, train_pool: list[dict], eval_tasks: list[dict]) -> dict:
        import random
        import time

        import torch

        from bandits_jev.grpo import (
            group_advantages,
            policy_gradient_step,
            rollout_reward,
            training_segments,
        )
        from bandits_jev.rollout import summarize

        run_dir = Path(f"/runs/step-rl/grpo/{config['run']}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.batch_minutes = config["batch_timeout_minutes"]
        check = config.pop("judge_check", [])
        if check:
            got = self._judge_score([c["step"] for c in check])["results"]
            diffs = [
                max(abs(g["probabilities"][k] - v) for k, v in c["probabilities"].items())
                for g, c in zip(got, check, strict=True)
            ]
            agree = sum(
                max(g["probabilities"], key=g["probabilities"].get) == c["predicted"]
                for g, c in zip(got, check, strict=True)
            )
            config["judge_check_result"] = {"steps": len(check), "class_agreement": agree, "max_abs_diff": max(diffs)}
            print(json.dumps(config["judge_check_result"]), flush=True)
            if agree < len(check) - 1 or max(diffs) > 0.05:
                raise RuntimeError(f"in-container judge does not reproduce the audit: {config['judge_check_result']}")
        deadline = time.time() + 60 * config["max_train_minutes"]
        (run_dir / "config.json").write_text(json.dumps(config, indent=1))
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.0
        )
        model_name, history = "base", []
        for step in range(config["steps"]):
            if time.time() > deadline:
                print(json.dumps({"stopped_early": f"time guard after {step} steps"}), flush=True)
                config["stopped_early_after_steps"] = step
                break
            started = time.time()
            rng = random.Random(f"{config['seed']}:{step}")  # same questions per step for both arms
            tasks = rng.sample(train_pool, config["questions_per_step"])
            results = await self._rollouts(tasks, config["n"], model_name, True, f"train:{config['seed']}:{step}")
            generated = time.time()

            groups: dict[str, list[dict]] = {}
            for result in results:
                result["reward_parts"] = rollout_reward(
                    result, config["arm"], step_weight=config["step_weight"],
                    step_cap=config["step_cap"], baseline=config["baseline"],
                )
                groups.setdefault(result["query_id"], []).append(result)
            items, sampled_tokens, vllm_logprobs = [], 0, []
            for group in groups.values():
                for result, advantage in zip(
                    group, group_advantages([r["reward_parts"]["reward"] for r in group]), strict=True
                ):
                    result["advantage"] = advantage
                    for call in result["model_calls"]:
                        sampled_tokens += len(call["token_ids"])
                        if advantage != 0:
                            vllm_logprobs += call["logprobs"] or []
                    if advantage != 0:
                        items += [(ids, mask, advantage) for ids, mask in training_segments(result["model_calls"])]

            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = policy_gradient_step(self.model, items, normalizer=max(1, sampled_tokens), device="cuda")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)) if stats["trained_tokens"] else 0.0
            if stats["trained_tokens"]:
                optimizer.step()
            path = str(run_dir / f"adapter-step{step + 1}")
            self.model.save_pretrained(path)
            runs.commit()
            new_name = f"policy-step{step + 1}"
            self._load_adapter(new_name, path, model_name if model_name != "base" else None)
            model_name = new_name

            summary = summarize(results)
            record = {
                "step": step + 1,
                "questions": [t["query_id"] for t in tasks],
                "reward_mean": sum(r["reward_parts"]["reward"] for r in results) / len(results),
                "outcome_mean": sum(r["reward_parts"]["outcome"] for r in results) / len(results),
                "step_sum_mean": sum(r["reward_parts"]["step_sum"] for r in results) / len(results),
                "groups_with_signal": sum(any(r["advantage"] != 0 for r in g) for g in groups.values()),
                "groups": len(groups),
                "trained_tokens": stats["trained_tokens"],
                "sampled_tokens": sampled_tokens,
                "hf_mean_logprob": stats["mean_logprob"],
                "vllm_mean_logprob": sum(vllm_logprobs) / len(vllm_logprobs) if vllm_logprobs else None,
                "grad_norm": grad_norm,
                "generate_seconds": round(generated - started, 1),
                "step_seconds": round(time.time() - started, 1),
                **{k: summary[k] for k in (
                    "accuracy_proxy", "mean_tool_calls", "evidence_seen_recall", "evidence_opened_recall",
                    "rollouts_with_repeated_action", "judge_mean_score_on_evidence_steps",
                    "judge_mean_score_on_other_steps",
                )},
            }
            history.append(record)
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            runs.commit()
            print(json.dumps(record), flush=True)

        evaluation = await self._rollouts(eval_tasks, config["eval_n"], model_name, False, "eval")
        for result in evaluation:
            result.pop("model_calls", None)
        out = {"config": config, "history": history, "eval_summary": summarize(evaluation), "eval": evaluation}
        (run_dir / "eval.json").write_text(json.dumps(out))
        runs.commit()
        return {k: out[k] for k in ("config", "history", "eval_summary")}


def judge_check(work: Path, count: int = 20) -> list[dict]:
    """The first audited steps with their saved probabilities, to confirm the in-container judge."""
    raw = {r["id"]: r for r in map(json.loads, (work / "openresearcher-v3-audit-candidates.jsonl").read_text().splitlines())}
    saved = json.loads((work / "search-judge-v3-audit.json").read_text())["new_judge"]["predictions"][:count]
    return [
        {
            "step": {
                "question": raw[row["id"]]["question"],
                "previous": [[p["tool"], p["action"], p["observation"]] for p in raw[row["id"]]["previous"]],
                "tool": raw[row["id"]]["tool"], "action": raw[row["id"]]["action"],
                "observation": raw[row["id"]]["observation"],
            },
            "probabilities": row["probabilities"],
            "predicted": row["predicted"],
        }
        for row in saved
    ]


@app.local_entrypoint()
def main(
    arm: str = "outcome", seed: int = 1, steps: int = 10, questions_per_step: int = 16, n: int = 8,
    lr: float = 1e-4, step_weight: float = 0.3, step_cap: float = 1.0, baseline: float = 0.0,
    eval_n: int = 8, eval_limit: int = 60, tag: str = "", max_train_minutes: int = 60,
    batch_timeout_minutes: int = 12,
) -> None:
    work = REPO / "work/step-rl"
    tasks = [json.loads(line) for line in (work / "tasks.jsonl").read_text().splitlines()]
    train_pool = [t for t in tasks if t["split"] == "train_pool"]
    eval_tasks = [t for t in tasks if t["split"] == "held_out_eval"][:eval_limit]
    assert not {t["query_id"] for t in train_pool} & {t["query_id"] for t in eval_tasks}
    run = f"{arm}-seed{seed}{'-' + tag if tag else ''}"
    config = {
        "run": run, "arm": arm, "seed": seed, "steps": steps, "questions_per_step": questions_per_step, "n": n,
        "lr": lr, "step_weight": step_weight, "step_cap": step_cap, "baseline": baseline,
        "eval_n": eval_n, "eval_questions": len(eval_tasks), "policy": POLICY, "policy_revision": POLICY_REVISION,
        "judge_adapter": JUDGE_ADAPTER, "lora_rank": 32, "algorithm": "Dr. GRPO, no KL, one update per batch",
        "max_train_minutes": max_train_minutes, "batch_timeout_minutes": batch_timeout_minutes,
        "judge_check": judge_check(work),
    }
    out = Trainer().train.remote(config, train_pool, eval_tasks)
    target = work / "grpo" / f"{run}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps({"run": run, "eval_summary": out["eval_summary"]}, indent=2))
