"""Small, resumable client for scoring decision datasets with the Jev API."""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bandits_jev.dataset import DecisionExample

DEFAULT_BASE_URL = "https://api.typesafe.ai"


class JevAPIError(RuntimeError):
    pass


def _request(
    example: DecisionExample,
    *,
    api_key: str,
    model: str,
    base_url: str,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "state": example.state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": example.question,
                "criteria": dict(example.options),
            }
        },
    }
    data = json.dumps(body, ensure_ascii=False).encode()
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/v1/systemone",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            latency = time.perf_counter() - started
            answer = payload["answers"]["decision"]
            probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
            if set(probabilities) != set(example.options):
                raise JevAPIError(
                    f"Jev returned options {sorted(probabilities)}, expected {sorted(example.options)}"
                )
            total = sum(probabilities.values())
            if total <= 0:
                raise JevAPIError("Jev returned probabilities with a non-positive total")
            return {
                "decision_id": example.decision_id,
                # The API returns two-decimal probabilities, whose rounded
                # values can total 0.99. Store a proper distribution for the
                # generic prediction importer while preserving their ratios.
                "probabilities": {key: value / total for key, value in probabilities.items()},
                "latency_seconds": latency,
                "model": payload.get("model", model),
                "usage": payload.get("usage", {}),
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            last_error = JevAPIError(f"HTTP {exc.code}: {detail}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504, 529}:
                break
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(30.0, 2**attempt + random.random())
        except (OSError, TimeoutError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            delay = min(30.0, 2**attempt + random.random())
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise JevAPIError(f"{example.decision_id}: {last_error}")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            probabilities = row.get("probabilities")
            if isinstance(probabilities, dict):
                total = sum(float(value) for value in probabilities.values())
                if total > 0:
                    row["probabilities"] = {
                        key: float(value) / total for key, value in probabilities.items()
                    }
            rows[row["decision_id"]] = row
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid Jev cache row") from exc
    return rows


def score_jev_api(
    examples: list[DecisionExample],
    output: Path,
    *,
    api_key: str | None = None,
    model: str = "jev-latest",
    base_url: str = DEFAULT_BASE_URL,
    workers: int = 8,
    timeout: float = 90,
    attempts: int = 5,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Score missing rows and append each success immediately for safe resume."""
    api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError("set JEV_API_KEY (or TYPESAFE_API_KEY)")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cache(output)
    wanted = {example.decision_id for example in examples}
    foreign = set(cached) - wanted
    if foreign:
        raise ValueError(f"cache has {len(foreign)} decision(s) outside this dataset split")
    todo = [example for example in examples if example.decision_id not in cached]
    errors: list[str] = []
    lock = threading.Lock()
    completed = len(cached)

    def run(example: DecisionExample) -> dict[str, Any]:
        return _request(
            example,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            attempts=attempts,
        )

    with output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, example): example for example in todo}
        for future in as_completed(futures):
            example = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # keep other paid calls and make the cache resumable
                errors.append(str(exc))
                continue
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                cached[example.decision_id] = row
                completed += 1
                if progress is not None:
                    progress(completed, len(examples))
    # Canonicalize order and normalize cached rounded probabilities too. Use
    # an atomic replace so interruption cannot destroy the resumable cache.
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(cached[example.decision_id], ensure_ascii=False) + "\n" for example in examples if example.decision_id in cached),
        encoding="utf-8",
    )
    temporary.replace(output)
    return cached, errors
