"""Run the policy on BrowseComp-Plus questions on one Modal GPU and score every step.

Serves Qwen3-4B-Instruct-2507 with vLLM, runs rollouts through the real search
tools (the deployed `jev-retriever`), has the deployed `jev-step-judge` score
each observed step, and compares everything with ground truth (answer proxy,
gold evidence found). Needs both apps deployed:

    uvx modal deploy recipes/jev/scripts/modal_retriever.py
    uvx modal deploy recipes/jev/scripts/modal_jev_judge.py

Smoke test (10 training-pool questions, one rollout each):

    uvx modal run recipes/jev/scripts/modal_rollouts.py --which train_pool --limit 10 --n 1 --tag smoke

Base policy on the held-out evaluation set (60 questions, 8 rollouts each):

    uvx modal run recipes/jev/scripts/modal_rollouts.py --which held_out_eval --n 8 --tag base
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "openai", "huggingface_hub")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(str(REPO / "recipes/jev/bandits_jev"), "/app/bandits_jev")
)
app = modal.App("jev-rollouts", image=image)
runs = modal.Volume.from_name("jev-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("jev-hf-cache", create_if_missing=True)


@app.cls(
    gpu="L40S",
    timeout=3 * 3600,
    scaledown_window=60,
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
)
class Rollouts:
    @modal.enter()
    def start_server(self) -> None:
        import subprocess
        import time
        import urllib.request

        self.server = subprocess.Popen(
            [
                "vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "policy",
                "--dtype", "bfloat16", "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
                "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(PORT),
            ]
        )
        deadline = time.time() + 20 * 60
        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.server.returncode}")
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
                return
            except OSError:
                time.sleep(5)
        raise RuntimeError("vllm did not become ready in 20 minutes")

    @modal.method()
    async def run(self, tasks: list[dict], n: int, tag: str, max_turns: int = 12, concurrency: int = 48) -> dict:
        import asyncio
        import hashlib

        import modal as modal_client
        import openai

        from bandits_jev.modal_searcher import ModalSearcher
        from bandits_jev.rollout import TOOLS, ContextOverflow, run_rollout, summarize

        client = openai.AsyncOpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="unused", timeout=900)
        searcher = ModalSearcher()
        judge = modal_client.Cls.from_name("jev-step-judge", "JevJudge")()
        gate, judge_gate = asyncio.Semaphore(concurrency), asyncio.Semaphore(16)
        provenance: dict = {}

        async def one(task: dict, sample: int) -> dict:
            seed = int(hashlib.sha256(f"{task['query_id']}:{sample}".encode()).hexdigest()[:8], 16)

            async def chat(messages, tool_choice):
                try:
                    reply = await client.chat.completions.create(
                        model="policy", messages=messages, tools=TOOLS, tool_choice=tool_choice,
                        temperature=1.0, max_tokens=1024, seed=seed,
                    )
                except openai.BadRequestError as exc:
                    if "context length" in str(exc) or "maximum" in str(exc):
                        raise ContextOverflow(str(exc)[:200]) from exc
                    raise
                message = reply.choices[0].message
                return {
                    "content": message.content,
                    "tool_calls": [
                        {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                        for c in message.tool_calls or []
                    ],
                }

            async with gate:
                result = await run_rollout(task, chat, searcher, max_turns=max_turns)
            result["sample"] = sample
            history, steps = [], []
            for event in result["events"]:
                steps.append(
                    {
                        "question": task["query"], "previous": [list(h) for h in history[-2:]],
                        "tool": event["tool"], "action": event["action"], "observation": event["observation"],
                    }
                )
                history.append((event["tool"], event["action"], event["observation"]))
            if steps:
                async with judge_gate:
                    reply = await judge.score.remote.aio(steps)
                provenance.update(reply["judge"])
                for event, scored in zip(result["events"], reply["results"], strict=True):
                    event["judge"] = scored
            return result

        results = await asyncio.gather(*(one(t, s) for t in tasks for s in range(n)))
        out = {
            "tag": tag, "policy": {"model": MODEL, "revision": REVISION}, "judge": provenance,
            "max_turns": max_turns, "summary": summarize(results), "results": results,
        }
        path = Path(f"/runs/step-rl/rollouts/{tag}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        runs.commit()
        return out


@app.local_entrypoint()
def main(which: str = "train_pool", limit: int = 10, n: int = 1, tag: str = "smoke", max_turns: int = 12) -> None:
    work = REPO / "work/step-rl"
    tasks = [json.loads(line) for line in (work / "tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if t["split"] == which][:limit]
    assert tasks, f"no tasks in split {which!r}"
    out = Rollouts().run.remote(tasks, n, tag, max_turns)
    target = work / "rollouts" / f"{tag}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out))
    print(json.dumps({"tag": tag, "tasks": len(tasks), "n": n, "judge": out["judge"], "summary": out["summary"]}, indent=2))
