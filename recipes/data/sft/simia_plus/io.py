"""JSONL helpers and a resumable parallel map."""
from __future__ import annotations

import json
import threading
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def parallel_map(fn: Callable[[dict], dict | list[dict] | None], items: list[dict], key: Callable[[dict], str],
                 out_path: str | Path, workers: int, done_key: Callable[[dict], str] | None = None,
                 desc: str = "") -> list[dict]:
    """Run fn over items not already present in out_path (by key), appending results as they finish.

    fn may return one row, several rows, or None (skipped). Exceptions are logged to <out>.errors.jsonl
    and the item is retried on the next run.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_key = done_key or key
    done = {done_key(r) for r in read_jsonl(out_path)}
    todo = [it for it in items if key(it) not in done]
    err_path = out_path.with_suffix(".errors.jsonl")
    lock = threading.Lock()
    ok = failed = 0
    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, it): it for it in todo}
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad item must not stop the run; logged and retried next run
                failed += 1
                with lock, err_path.open("a") as ef:
                    ef.write(json.dumps({"key": key(it), "error": repr(e),
                                         "traceback": traceback.format_exc()[-2000:]}) + "\n")
                continue
            rows = res if isinstance(res, list) else ([res] if res else [])
            with lock:
                for r in rows:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
            ok += 1
    if desc:
        print(f"{desc}: {ok} done, {failed} failed, {len(items) - len(todo)} already present")
    return read_jsonl(out_path)
