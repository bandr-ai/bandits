"""HTTP transport for a reconstructed environment. Stdlib only.

    from bandits.serve import SessionBackend, serve

    backend = SessionBackend(schema, tasks, tool_classes, surface=surface,
                             verifiers=verifiers)
    server = serve(backend, port=8080)   # 127.0.0.1 only, see below
    server.start()
    ...
    server.shutdown()

No FastAPI, no uvicorn, nothing to install. A training container that has
bandits has the server, which is the point: an RL environment that needs a
dependency tree is an environment that does not get deployed.

Binding
-------

**The default bind address is 127.0.0.1 and that is deliberate.** A served
environment is reconstructed from a customer's traces: their entity names,
their tool surface, their state. There is no authentication in this protocol.
Binding to ``0.0.0.0`` publishes that to the network, so it has to be typed out
by hand (``host="0.0.0.0"``), and should only ever happen inside a trusted
training network.

Episode isolation
-----------------

This is the property the whole server exists to protect. A distributed trainer
runs many rollouts against one server; if two of them touched one world, the
reward for both would be wrong and *nothing would report an error*.

* ``/reset`` builds a new :class:`~bandits.serve.backend.Episode` with its own
  session, its own SQLite store, its own effect ledger, and mints a
  cryptographically random ``episode_id`` (``secrets.token_hex``) -- ids are
  never sequential, so a fabricated or stale id cannot collide with a live one.
* ``/step`` resolves ``episode_id`` in the registry and acts on that episode
  only. There is no ambient "current episode" to fall back to. An unknown id is
  a 404 and a closed id is a 409 -- never a step into somebody else's world.
* Each episode carries its own lock. Two threads stepping *different* episodes
  never contend; two threads stepping the *same* episode are serialized, so a
  write is applied whole.
* The registry lock is held only for dict operations, never across a step, so
  parallel rollouts run parallel.

Capacity
--------

Live episodes are capped (``max_episodes``). Past the cap the server either
refuses with 429 (default) or evicts the least-recently-used episode, per
``on_full``. Refusing is the default because silently evicting a rollout a
trainer still believes in is worse than a retryable error.

Sessions are closed deterministically: on ``/close``, on eviction, on cap-driven
teardown, and on ``shutdown()`` -- which closes every remaining episode before
the socket goes away, so no SQLite handle outlives the process.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError

from .backend import (
    DEFAULT_MAX_STEPS,
    EnvBackend,
    Episode,
    EpisodeClosed,
    EpisodeFinished,
    UnsupportedToolCall,
)
from .protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    CloseRequest,
    CloseResponse,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MAX_EPISODES",
    "EnvHTTPServer",
    "EpisodeRegistry",
    "serve",
]

DEFAULT_HOST = "127.0.0.1"
"""Loopback. Reconstructed customer data does not go on the network by accident."""

DEFAULT_MAX_EPISODES = 64
"""Live episodes allowed at once. Each holds a SQLite connection open."""

_STATUS_FOR = {
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.EPISODE_CLOSED: 409,
    ErrorCode.EPISODE_DONE: 409,
    ErrorCode.EPISODE_LIMIT: 429,
    ErrorCode.UNSUPPORTED_TOOL: 422,
    ErrorCode.INTERNAL: 500,
}

MAX_BODY_BYTES = 4 * 1024 * 1024
"""Refuse absurd bodies rather than buffering them. A tool call is small."""


class ServeError(Exception):
    """An error with a wire representation. Anything else becomes a 500."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        episode_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.episode_id = episode_id
        self.detail = detail or {}

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=self.code,
            message=self.message,
            episode_id=self.episode_id,
            detail=self.detail,
        )


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


class EpisodeRegistry:
    """Bounded, thread-safe pool of live episodes.

    Insertion-ordered so ``evict_lru`` is O(1). The lock protects the mapping
    only -- it is never held while an episode is being stepped, because a global
    lock across steps would serialize every rollout on the box and quietly turn
    a parallel trainer into a sequential one.
    """

    def __init__(self, max_episodes: int = DEFAULT_MAX_EPISODES, on_full: str = "refuse") -> None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be >= 1")
        if on_full not in ("refuse", "evict_lru"):
            raise ValueError("on_full must be 'refuse' or 'evict_lru'")
        self.max_episodes = max_episodes
        self.on_full = on_full
        self._lock = threading.Lock()
        self._episodes: OrderedDict[str, Episode] = OrderedDict()
        # Bounded memory of retired ids, so a stale step gets 409 ("closed")
        # rather than 404 ("never existed"). Bounded because a long-running
        # trainer retires millions of episodes and this must not be a leak; an
        # id evicted from this set degrades to a 404, which is still correct.
        self._closed_ids: OrderedDict[str, None] = OrderedDict()
        self._started = 0
        self._evicted = 0

    _CLOSED_MEMORY = 8192

    def _remember_closed(self, episode_id: str) -> None:
        """Record a retired id under the registry lock, oldest dropped first."""
        self._closed_ids.pop(episode_id, None)
        self._closed_ids[episode_id] = None
        while len(self._closed_ids) > self._CLOSED_MEMORY:
            self._closed_ids.popitem(last=False)

    # -- accounting --------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    @property
    def started(self) -> int:
        with self._lock:
            return self._started

    @property
    def evicted(self) -> int:
        with self._lock:
            return self._evicted

    def live_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._episodes)

    # -- admission ---------------------------------------------------------

    def reserve(self) -> str:
        """Claim capacity and an id *before* an expensive session build.

        Returning the id first means the cap is enforced against episodes being
        built, not just episodes already built -- otherwise N concurrent resets
        could all pass the check and then all materialize.
        """
        with self._lock:
            if len(self._episodes) >= self.max_episodes:
                if self.on_full == "refuse":
                    raise ServeError(
                        ErrorCode.EPISODE_LIMIT,
                        f"live episode cap reached ({self.max_episodes}); close an episode "
                        f"or retry",
                        detail={"limit": self.max_episodes, "live": len(self._episodes)},
                    )
                victim_id, victim = self._episodes.popitem(last=False)
                self._remember_closed(victim_id)
                self._evicted += 1
            else:
                victim = None
            episode_id = f"ep-{secrets.token_hex(12)}"
            self._episodes[episode_id] = _PENDING
            self._started += 1
        if victim is not None:
            # Outside the registry lock: teardown must not block admission.
            victim.close()
        return episode_id

    def commit(self, episode_id: str, episode: Episode) -> None:
        """Install the built episode against its reserved id."""
        with self._lock:
            if self._episodes.get(episode_id) is not _PENDING:
                # Reservation was evicted while the session was building.
                stale = True
            else:
                self._episodes[episode_id] = episode
                stale = False
        if stale:
            episode.close()
            raise ServeError(
                ErrorCode.EPISODE_LIMIT,
                "episode was evicted before it finished starting; retry",
                episode_id=episode_id,
            )

    def release(self, episode_id: str) -> None:
        """Give back a reservation whose session failed to build."""
        with self._lock:
            if self._episodes.get(episode_id) is _PENDING:
                del self._episodes[episode_id]

    # -- routing -----------------------------------------------------------

    def get(self, episode_id: str) -> Episode:
        """Resolve an id to its episode, or raise the right wire error.

        Unknown -> 404. Known but closed/evicted -> 409. A step never falls
        through to some other episode.
        """
        with self._lock:
            episode = self._episodes.get(episode_id)
            known_closed = episode_id in self._closed_ids
        if episode is None or episode is _PENDING:
            if known_closed:
                raise ServeError(
                    ErrorCode.EPISODE_CLOSED,
                    f"episode {episode_id!r} has been closed",
                    episode_id=episode_id,
                )
            raise ServeError(
                ErrorCode.NOT_FOUND,
                f"unknown episode_id {episode_id!r}",
                episode_id=episode_id,
            )
        if episode.closed:
            raise ServeError(
                ErrorCode.EPISODE_CLOSED,
                f"episode {episode_id!r} has been closed",
                episode_id=episode_id,
            )
        return episode

    # -- teardown ----------------------------------------------------------

    def close(self, episode_id: str) -> tuple[Episode | None, bool]:
        """Remove and close one episode. Returns ``(episode, already_closed)``.

        Idempotent: closing an id twice is not an error, which matters because a
        trainer that crashes mid-rollout will re-issue closes on restart.
        """
        with self._lock:
            episode = self._episodes.pop(episode_id, None)
            already = episode_id in self._closed_ids
            self._remember_closed(episode_id)
        if episode is None or episode is _PENDING:
            if already:
                return None, True
            raise ServeError(
                ErrorCode.NOT_FOUND,
                f"unknown episode_id {episode_id!r}",
                episode_id=episode_id,
            )
        episode.close()
        return episode, False

    def close_all(self) -> int:
        """Close every live episode. Called on shutdown; leaks nothing."""
        with self._lock:
            episodes = [e for e in self._episodes.values() if e is not _PENDING]
            for retired in self._episodes:
                self._remember_closed(retired)
            self._episodes.clear()
        for episode in episodes:
            episode.close()
        return len(episodes)


class _Pending:
    """Placeholder occupying a reserved slot while its session is built."""

    closed = True

    def close(self) -> None:  # pragma: no cover - never actually closed
        return


_PENDING: Any = _Pending()


# --------------------------------------------------------------------------
# the handler
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """Routes five endpoints and turns every failure into structured JSON."""

    protocol_version = "HTTP/1.1"
    server_version = f"{PROTOCOL_NAME}/{PROTOCOL_VERSION}"
    sys_version = ""

    # The owning EnvHTTPServer injects these onto the socket server.
    @property
    def _env(self) -> EnvHTTPServer:
        return self.server._env  # type: ignore[attr-defined]

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silent by default. A trainer's stderr is not a request log."""
        if self._env.access_log:
            self._env.access_log(fmt % args)

    def _send(self, status: int, model: BaseModel) -> None:
        body = model.model_dump_json().encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, err: ServeError) -> None:
        self._send(_STATUS_FOR.get(err.code, 500), err.to_response())

    def _read_json(self) -> dict[str, Any]:
        length = self.headers.get("Content-Length")
        try:
            size = int(length or 0)
        except ValueError:
            raise ServeError(ErrorCode.BAD_REQUEST, "invalid Content-Length") from None
        if size > MAX_BODY_BYTES:
            raise ServeError(
                ErrorCode.BAD_REQUEST,
                f"request body too large ({size} bytes > {MAX_BODY_BYTES})",
            )
        raw = self.rfile.read(size) if size else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ServeError(ErrorCode.BAD_REQUEST, f"body is not valid JSON: {exc.msg}") from None
        if not isinstance(payload, dict):
            raise ServeError(ErrorCode.BAD_REQUEST, "body must be a JSON object")
        return payload

    def _parse(self, model: type[BaseModel], payload: dict[str, Any]) -> Any:
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise ServeError(
                ErrorCode.BAD_REQUEST,
                f"invalid {model.__name__}: {loc or '<body>'}: {first.get('msg', 'invalid')}",
                detail={"errors": json.loads(exc.json())},
            ) from None

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        try:
            if method == "GET" and route == "/health":
                return self._health()
            if method == "GET" and route == "/spec":
                return self._spec(parse_qs(parsed.query))
            if method == "POST" and route == "/reset":
                return self._reset()
            if method == "POST" and route == "/step":
                return self._step()
            if method == "POST" and route == "/close":
                return self._close()
            if method == "POST" and route == "/mcp":
                return self._mcp()
            raise ServeError(ErrorCode.NOT_FOUND, f"no route for {method} {route}")
        except ServeError as err:
            self._fail(err)
        except Exception as exc:  # never leak a traceback to the wire
            self._env.on_internal_error(exc)
            self._fail(
                ServeError(
                    ErrorCode.INTERNAL,
                    f"{type(exc).__name__}: {exc}",
                )
            )

    # -- endpoints ---------------------------------------------------------

    def _health(self) -> None:
        env = self._env
        self._send(
            200,
            HealthResponse(
                ok=True,
                live_episodes=len(env.registry),
                max_episodes=env.registry.max_episodes,
                episodes_started=env.registry.started,
                uptime_seconds=round(time.monotonic() - env.started_at, 3),
            ),
        )

    def _spec(self, query: dict[str, list[str]]) -> None:
        task_id = (query.get("task_id") or [None])[0]
        try:
            spec = self._env.backend.spec(task_id)
        except KeyError:
            raise ServeError(ErrorCode.NOT_FOUND, f"unknown task_id {task_id!r}") from None
        self._send(200, spec)

    def _reset(self) -> None:
        req: ResetRequest = self._parse(ResetRequest, self._read_json())
        env = self._env
        episode_id = env.registry.reserve()
        try:
            episode = env.backend.new_episode(episode_id, req.task_id, req.seed)
        except KeyError:
            env.registry.release(episode_id)
            raise ServeError(ErrorCode.NOT_FOUND, f"unknown task_id {req.task_id!r}") from None
        except Exception:
            env.registry.release(episode_id)
            raise
        env.registry.commit(episode_id, episode)
        self._send(
            200,
            ResetResponse(
                episode_id=episode_id,
                task_id=episode.task_id,
                instruction=episode.task.instruction,
                observation=episode.opening_observation(),
                tools=episode.tools,
                max_steps=episode.max_steps,
                step=0,
                seed=req.seed,
            ),
        )

    def _step(self) -> None:
        req: StepRequest = self._parse(StepRequest, self._read_json())
        episode = self._env.registry.get(req.episode_id)
        try:
            outcome = episode.step(req.action.name, req.action.arguments)
        except UnsupportedToolCall as exc:
            raise ServeError(
                ErrorCode.UNSUPPORTED_TOOL,
                f"environment cannot reimplement {exc.tool!r}: {exc.reason}",
                episode_id=req.episode_id,
                detail={"tool": exc.tool, "reason": exc.reason},
            ) from None
        except EpisodeFinished as exc:
            raise ServeError(
                ErrorCode.EPISODE_DONE, str(exc), episode_id=req.episode_id
            ) from None
        except EpisodeClosed:
            raise ServeError(
                ErrorCode.EPISODE_CLOSED,
                f"episode {req.episode_id!r} has been closed",
                episode_id=req.episode_id,
            ) from None
        self._send(
            200,
            StepResponse(
                episode_id=req.episode_id,
                observation=outcome.observation,
                reward=outcome.reward,
                done=outcome.done,
                truncated=outcome.truncated,
                step=outcome.step,
                info=outcome.info,
            ),
        )

    def _close(self) -> None:
        req: CloseRequest = self._parse(CloseRequest, self._read_json())
        episode, already = self._env.registry.close(req.episode_id)
        self._send(
            200,
            CloseResponse(
                episode_id=req.episode_id,
                closed=True,
                already_closed=already,
                steps=episode.steps if episode is not None else 0,
                final_digest=episode.final_digest if episode is not None else None,
            ),
        )

    def _mcp(self) -> None:
        """JSON-RPC endpoint, same sessions, MCP shapes. See serve.mcp."""
        payload = self._read_json()
        response = self._env.mcp.handle(payload)
        if response is None:  # a notification: acknowledged, no body
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# --------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------


class EnvHTTPServer:
    """Owns the socket, the thread and the episode registry.

    ``shutdown()`` is the only teardown path and it is idempotent: it stops the
    accept loop, closes every live episode, and then closes the socket. Closing
    episodes *before* the socket goes away means an in-flight request either
    completed or sees a closed episode -- never a half-torn-down store.
    """

    def __init__(
        self,
        backend: EnvBackend,
        *,
        host: str = DEFAULT_HOST,
        port: int = 0,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        on_full: str = "refuse",
        access_log: Any = None,
        on_internal_error: Any = None,
    ) -> None:
        self.backend = backend
        self.registry = EpisodeRegistry(max_episodes=max_episodes, on_full=on_full)
        self.access_log = access_log
        self._on_internal_error = on_internal_error
        self.started_at = time.monotonic()
        self._thread: threading.Thread | None = None
        self._mcp: Any = None
        self._mcp_lock = threading.Lock()
        self._shutdown = False
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd._env = self  # type: ignore[attr-defined]

    @property
    def mcp(self) -> Any:
        """The MCP view of this server, sharing its registry.

        One instance per server, not per request: an MCP client has no episode
        id, so the implicit episode has to survive between calls -- a fresh
        server per request would open (and strand) a new world every tool call.
        """
        with self._mcp_lock:
            if self._mcp is None:
                from .mcp import McpEnvServer

                self._mcp = McpEnvServer(self.backend, self.registry)
            return self._mcp

    # -- addressing --------------------------------------------------------

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> EnvHTTPServer:
        """Serve in a background thread. Returns self so it chains."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="bandits-serve",
            daemon=True,
        )
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Serve in the calling thread. For `python -m` style entry points."""
        try:
            self._httpd.serve_forever()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Stop accepting, close every episode, release the socket. Idempotent."""
        if self._shutdown:
            return
        self._shutdown = True
        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self.registry.close_all()
        self._httpd.server_close()

    def on_internal_error(self, exc: BaseException) -> None:
        if self._on_internal_error is not None:
            self._on_internal_error(exc)

    def __enter__(self) -> EnvHTTPServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()


def serve(
    backend: EnvBackend,
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    on_full: str = "refuse",
    max_steps: int = DEFAULT_MAX_STEPS,
) -> EnvHTTPServer:
    """Build a server bound to ``host:port`` (loopback, ephemeral port by default).

    Not started: call ``.start()`` for a background thread or ``.serve_forever()``
    to block. ``max_steps`` is accepted for symmetry and applies only when the
    backend has not set its own.
    """
    if getattr(backend, "max_steps", None) is None:  # pragma: no cover - duck typing
        backend.max_steps = max_steps  # type: ignore[attr-defined]
    return EnvHTTPServer(
        backend,
        host=host,
        port=port,
        max_episodes=max_episodes,
        on_full=on_full,
    )
