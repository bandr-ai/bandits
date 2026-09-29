"""Local web front end for the existing Jev CLI pipeline."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bandits.store import DerivedStore
from bandits_jev.importer import import_jsonl, save_imported_dataset

_PAGE = Path(__file__).with_name("web.html")
_ID = re.compile(r"^[a-zA-Z0-9._-]+$")


class WebState:
    def __init__(self, project: Path):
        self.project = project.resolve()
        self.runs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, data: dict) -> str:
        sources = data.get("sources", [])
        if not isinstance(sources, list) or not sources or len(sources) > 20 or any(not isinstance(s, str) or not _ID.fullmatch(s) for s in sources):
            raise ValueError("Select at least one dataset or judge run")
        model, revision = data.get("model"), data.get("revision")
        if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not revision.strip():
            raise ValueError("Model and pinned revision are required")
        seed = int(data.get("seed", 1))
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        split = data.get("eval_split", "dev")
        if split not in ("dev", "test"):
            raise ValueError("Invalid evaluation split")
        if split == "test" and data.get("allow_test") is not True:
            raise ValueError("Confirm the locked test evaluation")
        device = data.get("device", "cuda")
        dtype = data.get("dtype", "bfloat16")
        if device not in ("cuda", "cpu") or dtype not in ("bfloat16", "float32"):
            raise ValueError("Invalid device or precision")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU training requires float32 precision")
        run_id = uuid.uuid4().hex[:12]
        root = self.project / ".bandits" / "jev-web" / run_id
        root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-c", "from bandits_jev.cli import app; app()", "run", *sources,
                   "--model", model.strip(), "--revision", revision.strip(), "--seed", str(seed),
                   "--eval-split", split, "--checkpoint-dir", str(root / "checkpoints"),
                   "--output", str(root / "report"), "--project", str(self.project),
                   "--device", device, "--dtype", dtype]
        if split == "test":
            command.append("--allow-test")
        with self.lock:
            self.runs[run_id] = {"id": run_id, "status": "running", "log": "", "report": None}
        threading.Thread(target=self._execute, args=(run_id, command, root), daemon=True).start()
        return run_id

    def _execute(self, run_id: str, command: list[str], root: Path) -> None:
        try:
            with subprocess.Popen(command, cwd=self.project, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    with self.lock:
                        self.runs[run_id]["log"] = (self.runs[run_id]["log"] + line)[-100000:]
                code = process.wait()
            with self.lock:
                self.runs[run_id]["status"] = "complete" if code == 0 else "failed"
                if code == 0 and (root / "report" / "report.md").exists():
                    self.runs[run_id]["report"] = f"/api/report/{run_id}"
        except Exception as exc:
            with self.lock:
                self.runs[run_id]["status"] = "failed"
                self.runs[run_id]["log"] += f"\n{exc}\n"


def serve(project: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    state = WebState(project)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: object) -> None:
            self._send(status, json.dumps(value).encode())

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/sources":
                store = DerivedStore(state.project / ".bandits")
                sources = [{"id": e.artifact_id, "kind": e.kind} for kind in ("decision_dataset", "turn_judge_run") for e in store.list(kind=kind)]
                self._json(200, sources)
            elif path.startswith("/api/runs/"):
                run_id = path.rsplit("/", 1)[-1]
                with state.lock:
                    run = state.runs.get(run_id)
                    self._json(200, dict(run)) if run else self._json(404, {"error": "Run not found"})
            elif path.startswith("/api/report/"):
                run_id = path.rsplit("/", 1)[-1]
                with state.lock:
                    run = state.runs.get(run_id)
                report = state.project / ".bandits" / "jev-web" / run_id / "report" / "report.md"
                if run and run["status"] == "complete" and report.is_file():
                    self._send(200, report.read_bytes(), "text/plain; charset=utf-8")
                else:
                    self._json(404, {"error": "Report not ready"})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 10_000_000:
                    raise ValueError("Request must be under 10 MB")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                path = urlparse(self.path).path
                if path == "/api/import":
                    content = data.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Upload a nonempty JSONL file")
                    dataset = import_jsonl(content, source_file=data.get("name", "upload.jsonl"))
                    if not dataset.examples:
                        raise ValueError("No valid decisions found in file")
                    envelope = save_imported_dataset(dataset, DerivedStore(state.project / ".bandits"), source_file="web upload")
                    self._json(200, {"id": envelope.artifact_id, "counts": dataset.counts.model_dump()})
                elif path == "/api/runs":
                    self._json(200, {"id": state.start(data)})
                else:
                    self._json(404, {"error": "Not found"})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Jev UI: http://{host}:{port}", flush=True)
    server.serve_forever()
