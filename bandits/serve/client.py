"""A synchronous client for a served environment. ``urllib`` only.

    from bandits.serve import EnvClient

    with EnvClient("http://127.0.0.1:8080") as env:
        spec = env.spec()
        ep = env.reset(seed=0)
        out = env.step(ep.episode_id, "get_order", {"order_id": 7741})
        env.close(ep.episode_id)

One client object is *not* one episode. It is a handle on a server, and every
call carries its ``episode_id`` explicitly -- the same reason the wire protocol
has no ambient current episode. A rollout worker that wants the ergonomic form
uses :meth:`EnvClient.episode`, which is a context manager that closes its own
episode on the way out (including on exception, which is where rollouts leak).

The client is thread-safe: it holds no per-request state and opens one
connection per call. That is slower than a pooled session and completely
adequate -- a step is a SQLite write, not a network-bound operation.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from .protocol import (
    PROTOCOL_VERSION,
    Action,
    CloseResponse,
    EnvSpec,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    ResetResponse,
    StepResponse,
    protocol_major,
)

__all__ = ["EnvClient", "EnvClientError", "ProtocolMismatch"]


class EnvClientError(RuntimeError):
    """A structured failure from the server.

    Carries the machine-readable :class:`ErrorCode` so a trainer can branch:
    retry on ``episode_limit``, re-reset on ``episode_closed``, and treat
    ``unsupported_tool`` as a bug in the reconstruction rather than a bad
    rollout.
    """

    def __init__(self, status: int, response: ErrorResponse) -> None:
        super().__init__(f"[{status} {response.error.value}] {response.message}")
        self.status = status
        self.code: ErrorCode = response.error
        self.response = response
        self.episode_id = response.episode_id
        self.detail = response.detail


class ProtocolMismatch(RuntimeError):
    """The server speaks a different major protocol version than this client."""


class EnvClient:
    """Drives one served environment over HTTP.

    Parameters
    ----------
    base_url:
        e.g. ``http://127.0.0.1:8080``. Trailing slash optional.
    timeout:
        Per-request timeout in seconds. A step is local work; if it takes longer
        than this the server is wedged and a rollout worker should fail fast
        rather than hang a training step.
    check_version:
        Verify the server's protocol major matches this client's on first call.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        check_version: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._check_version = check_version
        self._version_checked = False

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            raise self._error(exc) from None
        except urllib.error.URLError as exc:
            raise EnvClientError(
                0,
                ErrorResponse(
                    error=ErrorCode.INTERNAL,
                    message=f"cannot reach {url}: {exc.reason}",
                ),
            ) from None
        return json.loads(body) if body else {}

    @staticmethod
    def _error(exc: urllib.error.HTTPError) -> EnvClientError:
        """Turn an HTTP failure into a structured error, even if the body is junk."""
        raw = exc.read() or b""
        try:
            parsed = ErrorResponse.model_validate_json(raw)
        except Exception:
            parsed = ErrorResponse(
                error=ErrorCode.INTERNAL,
                message=(raw.decode("utf-8", "replace") or exc.reason or "unknown error")[:500],
            )
        return EnvClientError(exc.code, parsed)

    def _verify_version(self, server_version: str) -> None:
        if not self._check_version or self._version_checked:
            return
        self._version_checked = True
        if protocol_major(server_version) != protocol_major(PROTOCOL_VERSION):
            raise ProtocolMismatch(
                f"server speaks protocol {server_version}, client speaks {PROTOCOL_VERSION}"
            )

    # -- endpoints ---------------------------------------------------------

    def health(self) -> HealthResponse:
        """Liveness and current capacity."""
        health = HealthResponse.model_validate(self._request("GET", "/health"))
        self._verify_version(health.protocol_version)
        return health

    def spec(self, task_id: str | None = None) -> EnvSpec:
        """The environment description: instruction, tools, caps, digests."""
        path = "/spec" if task_id is None else f"/spec?task_id={urllib.parse.quote(task_id)}"
        spec = EnvSpec.model_validate(self._request("GET", path))
        self._verify_version(spec.protocol_version)
        return spec

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> ResetResponse:
        """Start a fresh, isolated episode."""
        return ResetResponse.model_validate(
            self._request("POST", "/reset", {"task_id": task_id, "seed": seed})
        )

    def step(
        self,
        episode_id: str,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> StepResponse:
        """Take one action inside ``episode_id`` and nothing else."""
        action = Action(name=name, arguments=dict(arguments or {}))
        return StepResponse.model_validate(
            self._request(
                "POST",
                "/step",
                {"episode_id": episode_id, "action": action.model_dump(mode="json")},
            )
        )

    def close(self, episode_id: str) -> CloseResponse:
        """Release the episode's session. Idempotent server-side."""
        return CloseResponse.model_validate(
            self._request("POST", "/close", {"episode_id": episode_id})
        )

    # -- ergonomics --------------------------------------------------------

    @contextmanager
    def episode(
        self, *, task_id: str | None = None, seed: int | None = None
    ) -> Iterator[ResetResponse]:
        """Reset, yield the episode, and always close it.

        The ``finally`` is the point: a rollout that raises mid-trajectory still
        releases its SQLite handle, so a crash loop in the policy does not walk
        the server into its episode cap.
        """
        started = self.reset(task_id=task_id, seed=seed)
        try:
            yield started
        finally:
            try:
                self.close(started.episode_id)
            except EnvClientError:
                pass

    def rpc(self, method: str, params: Mapping[str, Any] | None = None, rid: int = 1) -> Any:
        """Call the server's MCP JSON-RPC endpoint (``POST /mcp``)."""
        return self._request(
            "POST", "/mcp", {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        )

    def __enter__(self) -> EnvClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
