"""Scoped Dr. GRPO run on one Modal H100: outcome-only or outcome + Jev step reward.

One H200 holds everything that would otherwise sit idle on its own GPU or
stall on the network: vLLM serves the policy (LoRA hot-swapped after each
update), the policy is trained with PEFT, the frozen Jev judge scores steps,
and the BrowseComp-Plus retriever answers searches, all in one process. Before
training it must reproduce saved judge and retriever outputs.
Both arms use the same questions per step for a given seed, the same
hyperparameters and the same evaluation; they differ only in the reward.

    uvx modal run --detach recipes/jev/scripts/modal_grpo.py --arm outcome --seed 1 --steps 10
    uvx modal run --detach recipes/jev/scripts/modal_grpo.py --arm step --seed 1 --steps 10
    uvx modal run --detach recipes/jev/scripts/modal_grpo.py --arm step2 --seed 1 --steps 10

Metrics are appended per step to /runs/step-rl/grpo/<run>/metrics.jsonl on the
`jev-runs` volume, so an interrupted run keeps what it finished.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")
POLICY = "Qwen/Qwen3-4B-Instruct-2507"
POLICY_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
JUDGE = "Qwen/Qwen3.5-4B-Base"
JUDGE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
JUDGE_ADAPTER = "/runs/search-judge-pilot/checkpoints/step-50"
OUTCOME_ADAPTER = "/runs/outcome-judge-v1/checkpoints/step-150"
"""The outcome Jev (held-out AUC 0.92 on final answers): reward for the Jev-only arms."""
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    # flash-linear-attention is Triton only; without it the judge's linear-attention layers
    # fall back to slow PyTorch code.
    .pip_install(
        "vllm==0.31.0", "openai", "huggingface_hub", "peft", "typer", "rich", "flash-linear-attention", "regex", "pyarrow"
    )
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
corpus = modal.Volume.from_name("jev-browsecomp")
hf_cache = modal.Volume.from_name("jev-hf-cache", create_if_missing=True)


def start_vllm():
    """Start vLLM for the policy with runtime LoRA loading and wait until it answers."""
    import subprocess
    import urllib.request

    server = subprocess.Popen(
        [
            "vllm", "serve", POLICY, "--revision", POLICY_REVISION, "--served-model-name", "base",
            "--dtype", "bfloat16", "--max-model-len", "32768", "--gpu-memory-utilization", "0.4",
            "--enable-lora", "--max-lora-rank", "32", "--max-loras", "2",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(PORT),
        ]
    )
    deadline = time.time() + 20 * 60
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"vllm exited with code {server.returncode}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            return server
        except OSError:
            if time.time() > deadline:
                raise RuntimeError("vllm did not become ready in 20 minutes") from None
            time.sleep(5)


class Harness:
    """Rollout generation, judging and adapter loading shared by Trainer and Evaluator."""

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

    async def _rollouts(
        self, tasks: list[dict], n: int, model_name: str, keep_tokens: bool, salt: str, judge: bool = True
    ) -> list[dict]:
        import asyncio
        import hashlib

        import openai

        from bandits_jev.policy_client import judge_rollout, make_chat
        from bandits_jev.rollout import run_rollout

        if getattr(self, "client", None) is None:
            import httpx

            # One client for the whole run, with a short per-request timeout and retries: in the
            # step arm some requests hung on the client side until the timeout fired (15 min at
            # 900 s, 2 to 5 min at 300 s) while vLLM itself answered new requests instantly.
            self.client = openai.AsyncOpenAI(
                base_url=f"http://localhost:{PORT}/v1", api_key="unused", max_retries=3,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
                    timeout=httpx.Timeout(90.0, connect=10.0),
                ),
            )
        client = self.client
        searcher, gate = self.retriever, asyncio.Semaphore(64)
        finished = [0]

        async def judge_async(steps: list[dict]) -> dict:
            return self._judge_score(steps)

        async def one(task: dict, sample: int) -> dict:
            seed = int(hashlib.sha256(f"{salt}:{task['query_id']}:{sample}".encode()).hexdigest()[:8], 16)
            async with gate:
                result = await run_rollout(task, make_chat(client, model_name, seed, keep_tokens=keep_tokens), searcher)
            result["sample"] = sample
            finished[0] += 1
            return result

        def awaiting(task: asyncio.Task) -> str:
            """The innermost frames of a pending task's await chain."""
            names, coro = [], task.get_coro()
            while coro is not None:
                frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
                if frame is not None:
                    names.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
            return " > ".join(names[-4:])

        async def report_stalls() -> None:
            from collections import Counter

            last, since = 0, time.monotonic()
            while True:
                await asyncio.sleep(30)
                if finished[0] != last:
                    last, since = finished[0], time.monotonic()
                elif time.monotonic() - since > 120:
                    pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
                    where = Counter(awaiting(t) for t in pending).most_common(6)
                    print(json.dumps({"stall_seconds": round(time.monotonic() - since), "finished": finished[0],
                                      "pending_tasks": len(pending), "awaiting": where}), flush=True)

        from concurrent.futures import ThreadPoolExecutor

        # Retriever calls block in threads; the default pool is small enough that a few slow
        # calls queue every other rollout behind them.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=128))
        import faulthandler
        import sys

        # Diagnosis: if generation runs long, dump every thread's stack each minute. It runs on
        # a C thread, so it shows where Python is stuck even when the interpreter lock is held.
        faulthandler.dump_traceback_later(150, repeat=True, exit=False, file=sys.stderr)
        reporter = asyncio.create_task(report_stalls())
        try:
            # Watchdog: a stalled batch ends the run instead of holding an idle GPU.
            results = await asyncio.wait_for(
                asyncio.gather(*(one(t, s) for t in tasks for s in range(n))), 60 * self.batch_minutes
            )
        except TimeoutError:
            raise RuntimeError(f"rollout batch stalled for {self.batch_minutes} minutes; stopping") from None
        finally:
            faulthandler.cancel_dump_traceback_later()
            reporter.cancel()
        slow = sorted(
            ((e["seconds"], e["tool"], e["action"][:200]) for r in results for e in r["events"] if e.get("seconds", 0) > 5),
            reverse=True,
        )
        print(json.dumps({"slow_tool_calls": len(slow), "slowest": slow[:5]}), flush=True)
        if not judge:
            return results
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


@app.cls(
    # H200 (141 GB) so the retriever fits beside vLLM, the policy and the judge: a remote
    # retriever stalled training batches for minutes at a time between updates.
    gpu="H200",
    # Runs are launched detached (they must survive the launching laptop sleeping), so this
    # is the hard ceiling on what a forgotten run can bill.
    timeout=2 * 3600,
    scaledown_window=30,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache, "/data": corpus},
)
class Trainer(Harness):
    @modal.enter()
    def start(self) -> None:
        self.server = start_vllm()

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
        self.outcome_judge = HFPredictor(JUDGE, revision=JUDGE_REVISION, adapter_path=OUTCOME_ADAPTER)

        from bandits_jev.dense_retriever import DenseRetriever

        self.retriever = DenseRetriever.load("/data/index/corpus.shard*.pkl", "/data/corpus/*.parquet")

    @modal.method()
    async def train(self, config: dict, train_pool: list[dict], eval_tasks: list[dict]) -> dict:
        import random
        import time

        import torch

        from bandits_jev.grpo import (
            group_advantages,
            group_rewards,
            policy_gradient_step,
            training_segments,
        )
        from bandits_jev.rollout import summarize

        def auc(positive: list[float], negative: list[float]) -> float | None:
            if not positive or not negative:
                return None
            wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
            return wins / (len(positive) * len(negative))

        run_dir = Path(f"/runs/step-rl/grpo/{config['run']}")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.batch_minutes = config["batch_timeout_minutes"]
        reference = config.pop("retrieval_reference", None)
        if reference:
            import hashlib

            got = self.retriever.search_many(reference["queries"], 10)
            overlaps = [
                len({h["docid"] for h in g} & {d for d, _, _ in r}) / 10 for g, r in zip(got, reference["top10"], strict=True)
            ]
            same_top = sum(g[0]["docid"] == r[0][0] for g, r in zip(got, reference["top10"], strict=True))
            same_docs = all(
                hashlib.sha256(self.retriever.texts[d].encode()).hexdigest() == h for d, h in reference["documents"].items()
            )
            config["retrieval_check_result"] = {
                "queries": len(overlaps), "mean_top10_overlap": sum(overlaps) / len(overlaps),
                "min_top10_overlap": min(overlaps), "same_top1": same_top, "documents_match": same_docs,
            }
            print(json.dumps(config["retrieval_check_result"]), flush=True)
            if min(overlaps) < 0.8 or same_top < len(overlaps) - 1 or not same_docs:
                raise RuntimeError(f"in-container retriever does not match the service: {config['retrieval_check_result']}")
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
        import faulthandler
        import sys

        def arm_watchdog() -> None:
            # Runs on a C thread, so it fires even if Python code holds the interpreter lock:
            # dumps every thread's stack to the log, then exits so the GPU is released.
            faulthandler.dump_traceback_later(60 * (self.batch_minutes + 8), exit=True, file=sys.stderr)

        for step in range(config["steps"]):
            arm_watchdog()
            if time.time() > deadline:
                print(json.dumps({"stopped_early": f"time guard after {step} steps"}), flush=True)
                config["stopped_early_after_steps"] = step
                break
            started = time.time()
            rng = random.Random(f"{config['seed']}:{step}")  # same questions per step for both arms
            tasks = rng.sample(train_pool, config["questions_per_step"])
            results = await self._rollouts(tasks, config["n"], model_name, True, f"train:{config['seed']}:{step}")
            arm_watchdog()
            generated = time.time()
            (run_dir / f"rollouts-step{step + 1}.json").write_text(
                json.dumps([{k: v for k, v in r.items() if k != "model_calls"} for r in results])
            )

            from bandits_jev.outcome_judge import judge_outcome

            by_id = {t["query_id"]: t for t in tasks}
            for result in results:
                # Scored in every arm: the reward in the Jev-only arms, and in all arms a check
                # that the verifier still tracks ground truth as the policy changes.
                result["jev_outcome"] = judge_outcome(
                    self.outcome_judge, by_id[result["query_id"]]["query"], result["events"], result["final"]
                )
            groups: dict[str, list[dict]] = {}
            for result in results:
                groups.setdefault(result["query_id"], []).append(result)
            for group in groups.values():
                parts = group_rewards(
                    group, config["arm"], step_weight=config["step_weight"], step_cap=config["step_cap"],
                    baseline=config["baseline"], repeat_penalty=config["repeat_penalty"],
                )
                for result, part in zip(group, parts, strict=True):
                    result["reward_parts"] = part
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
                "jev_outcome_mean": sum(r["jev_outcome"] or 0.0 for r in results) / len(results),
                "jev_outcome_auc_vs_truth": auc(
                    [r["jev_outcome"] or 0.0 for r in results if r["correct"]],
                    [r["jev_outcome"] or 0.0 for r in results if not r["correct"]],
                ),
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

        arm_watchdog()
        evaluation = await self._rollouts(eval_tasks, config["eval_n"], model_name, False, "eval")
        arm_watchdog()
        for result in evaluation:
            result.pop("model_calls", None)
        out = {"config": config, "history": history, "eval_summary": summarize(evaluation), "eval": evaluation}
        (run_dir / "eval.json").write_text(json.dumps(out))
        runs.commit()
        faulthandler.cancel_dump_traceback_later()
        return {k: out[k] for k in ("config", "history", "eval_summary")}


@app.cls(
    gpu="H100",
    timeout=3600,
    scaledown_window=30,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache, "/data": corpus},
)
class Evaluator(Harness):
    """Held-out evaluation of a saved adapter without training or judging: vLLM and the
    retriever only. Deploy the app and spawn this so it does not depend on the launching
    machine staying awake (scripts/spawn_eval.py)."""

    @modal.enter()
    def start(self) -> None:
        from bandits_jev.dense_retriever import DenseRetriever

        self.server = start_vllm()
        self.retriever = DenseRetriever.load("/data/index/corpus.shard*.pkl", "/data/corpus/*.parquet")

    @modal.method()
    async def evaluate(self, run: str, adapter: str, eval_tasks: list[dict], eval_n: int) -> dict:
        from bandits_jev.rollout import summarize

        self.batch_minutes = 40
        self._load_adapter("evaluated", adapter, None)
        evaluation = await self._rollouts(eval_tasks, eval_n, "evaluated", False, "eval", judge=False)
        out = {"config": {"run": run, "adapter": adapter, "eval_n": eval_n, "judged": False},
               "eval_summary": summarize(evaluation), "eval": evaluation}
        path = Path(f"/runs/step-rl/grpo/{run}/eval.json")
        path.write_text(json.dumps(out))
        runs.commit()
        return out["eval_summary"]


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


def make_job(
    arm: str = "outcome", seed: int = 1, steps: int = 10, questions_per_step: int = 16, n: int = 8,
    lr: float = 1e-4, step_weight: float = 0.3, step_cap: float = 1.0, baseline: float = 0.0,
    eval_n: int = 8, eval_limit: int = 60, tag: str = "", max_train_minutes: int = 60,
    batch_timeout_minutes: int = 12, repeat_penalty: float = 0.5,
) -> tuple[dict, list[dict], list[dict]]:
    """The training config and task lists for one run, shared by the launcher and spawn_train.py."""
    work = REPO / "work/step-rl"
    tasks = [json.loads(line) for line in (work / "tasks.jsonl").read_text().splitlines()]
    train_pool = [t for t in tasks if t["split"] == "train_pool"]
    eval_tasks = [t for t in tasks if t["split"] == "held_out_eval"][:eval_limit]
    assert not {t["query_id"] for t in train_pool} & {t["query_id"] for t in eval_tasks}
    run = f"{arm}-seed{seed}{'-' + tag if tag else ''}"
    config = {
        "run": run, "arm": arm, "seed": seed, "steps": steps, "questions_per_step": questions_per_step, "n": n,
        "lr": lr, "step_weight": step_weight, "step_cap": step_cap, "baseline": baseline,
        "repeat_penalty": repeat_penalty,
        "eval_n": eval_n, "eval_questions": len(eval_tasks), "policy": POLICY, "policy_revision": POLICY_REVISION,
        "judge_adapter": JUDGE_ADAPTER, "lora_rank": 32, "algorithm": "Dr. GRPO, no KL, one update per batch",
        "max_train_minutes": max_train_minutes, "batch_timeout_minutes": batch_timeout_minutes,
        "judge_check": judge_check(work),
        "retrieval_reference": json.loads((work / "retrieval_reference.json").read_text()),
    }
    return config, train_pool, eval_tasks


@app.local_entrypoint()
def main(
    arm: str = "outcome", seed: int = 1, steps: int = 10, questions_per_step: int = 16, n: int = 8,
    lr: float = 1e-4, step_weight: float = 0.3, step_cap: float = 1.0, baseline: float = 0.0,
    eval_n: int = 8, eval_limit: int = 60, tag: str = "", max_train_minutes: int = 60,
    batch_timeout_minutes: int = 12, repeat_penalty: float = 0.5,
) -> None:
    config, train_pool, eval_tasks = make_job(
        arm, seed, steps, questions_per_step, n, lr, step_weight, step_cap, baseline,
        eval_n, eval_limit, tag, max_train_minutes, batch_timeout_minutes, repeat_penalty,
    )
    run, work = config["run"], REPO / "work/step-rl"
    out = Trainer().train.remote(config, train_pool, eval_tasks)
    target = work / "grpo" / f"{run}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps({"run": run, "eval_summary": out["eval_summary"]}, indent=2))
