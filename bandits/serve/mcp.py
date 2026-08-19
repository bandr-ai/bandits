"""MCP-shaped access to the same reconstructed environment.

AWM (Snowflake, ICML 2026) exposes its synthetic environments as MCP servers,
and that is the right shape for the *agent* side of this system: an MCP-capable
client -- Claude Code, an IDE agent, a customer's own harness -- can act inside
a reconstructed world with no bandits-specific integration at all. The HTTP
protocol in :mod:`bandits.serve.protocol` is for trainers, which need rewards,
episode ids and step caps. This module is for agents, which need
``tools/list`` and ``tools/call``.

Both drive the *same* :class:`~bandits.serve.backend.Episode` through the same
:class:`~bandits.serve.server.EpisodeRegistry`. There is one implementation of
the environment and two doors into it; an MCP call and an HTTP step against the
same ``episode_id`` are serialized by that episode's lock and see the same
world.

JSON-RPC, no SDK
----------------

The shapes are implemented directly (JSON-RPC 2.0 over a dict in / dict out),
so this adds no dependency. :meth:`McpEnvServer.handle` takes a decoded request
object and returns the response object, or ``None`` for a notification. Wire it
to stdio, to ``POST /mcp``, or call :meth:`list_tools` / :meth:`call_tool` in
process.

Methods
-------

``initialize``            handshake; reports server info and capabilities
``ping``                  liveness
``tools/list``            the reconstructed action space, as MCP tool descriptors
``tools/call``            dispatch one action into an episode
``env/spec``              the full :class:`~bandits.serve.protocol.EnvSpec`
``env/reset``             open a fresh episode, returns its ``episode_id``
``env/close``             release an episode

Episode binding
---------------

An MCP client has no notion of an episode, so one is opened lazily on first use
and reused (``auto_open``). A caller that *does* know about episodes -- a
trainer, or a harness running several agents at once -- passes
``params.episode_id`` and gets exactly that episode. Never both: an explicit id
always wins, and a wrong one is an error rather than a silent fallback to the
implicit episode, because falling back is how two agents end up in one world.

Refusals stay refusals
----------------------

A tool the environment could not reimplement returns ``isError: true`` with the
reason. It never returns a plausible success. An MCP agent must be able to tell
"this world cannot do that" from "that action failed", because the first is our
bug and the second is the environment working.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from typing import Any

from .backend import EnvBackend, Episode, EpisodeClosed, EpisodeFinished, UnsupportedToolCall
from .protocol import PROTOCOL_NAME, PROTOCOL_VERSION

__all__ = ["MCP_PROTOCOL_VERSION", "JsonRpcError", "McpEnvServer"]

MCP_PROTOCOL_VERSION = "2025-06-18"
"""The MCP revision whose ``tools/*`` shapes this implements."""

SERVER_INFO = {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION}

# JSON-RPC 2.0 reserved codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """An error with a JSON-RPC representation."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_obj(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


class McpEnvServer:
    """An MCP server bound to one backend, and to one episode at a time.

    Parameters
    ----------
    backend:
        Where episodes come from.
    registry:
        The shared :class:`~bandits.serve.server.EpisodeRegistry`. Sharing it
        with the HTTP server is what makes the two transports one environment
        rather than two. Omit it and a private registry is created, which is the
        right thing for a standalone stdio server.
    episode_id:
        Bind to an already-open episode instead of opening one.
    task_id:
        Which task ``auto_open`` instantiates. Defaults to the backend's first.
    auto_open:
        Open an episode on first ``tools/call`` when none is bound. True is
        correct for an agent-facing server (the agent never says "reset"); pass
        False when episodes are managed by a trainer.
    """

    def __init__(
        self,
        backend: EnvBackend,
        registry: Any = None,
        *,
        episode_id: str | None = None,
        task_id: str | None = None,
        auto_open: bool = True,
    ) -> None:
        if registry is None:
            from .server import EpisodeRegistry

            registry = EpisodeRegistry()
        self.backend = backend
        self.registry = registry
        self.task_id = task_id
        self.auto_open = auto_open
        self._episode_id = episode_id
        self._lock = threading.Lock()

    # -- episode plumbing --------------------------------------------------

    @property
    def episode_id(self) -> str | None:
        """The implicitly bound episode, if one has been opened."""
        return self._episode_id

    def open_episode(self, task_id: str | None = None, seed: int | None = None) -> str:
        """Open a fresh episode and bind this server to it."""
        episode_id = self.registry.reserve()
        try:
            episode = self.backend.new_episode(episode_id, task_id or self.task_id, seed)
        except KeyError:
            self.registry.release(episode_id)
            raise JsonRpcError(INVALID_PARAMS, f"unknown task_id {task_id!r}") from None
        except Exception:
            self.registry.release(episode_id)
            raise
        self.registry.commit(episode_id, episode)
        with self._lock:
            self._episode_id = episode_id
        return episode_id

    def close_episode(self, episode_id: str | None = None) -> bool:
        """Release an episode. Returns False if it was already gone."""
        target = episode_id or self._episode_id
        if target is None:
            return False
        try:
            self.registry.close(target)
        except Exception:
            return False
        with self._lock:
            if self._episode_id == target:
                self._episode_id = None
        return True

    def _episode(self, episode_id: str | None) -> Episode:
        """Resolve the episode for a call. Explicit id wins; never falls back.

        A wrong explicit id raises. It must not quietly land in the implicit
        episode -- that is precisely the cross-episode leak this whole layer
        exists to prevent.
        """
        from .server import ServeError

        if episode_id is not None:
            try:
                return self.registry.get(episode_id)
            except ServeError as exc:
                raise JsonRpcError(INVALID_PARAMS, exc.message, {"error": exc.code.value}) from None
        with self._lock:
            bound = self._episode_id
        if bound is None:
            if not self.auto_open:
                raise JsonRpcError(INVALID_PARAMS, "no episode bound; call env/reset first")
            bound = self.open_episode()
        try:
            return self.registry.get(bound)
        except ServeError as exc:
            raise JsonRpcError(INVALID_PARAMS, exc.message, {"error": exc.code.value}) from None

    # -- the two MCP operations -------------------------------------------

    def list_tools(self, task_id: str | None = None) -> list[dict[str, Any]]:
        """The reconstructed action space as MCP tool descriptors.

        Names and input schemas come straight from the environment spec, which
        took them from the declared tool registry when the corpus had one. An
        agent should not be able to tell this from the real tool list -- that is
        the requirement, since a policy trained here has to transfer.

        Tools the environment could not reimplement are still listed, with the
        refusal in the description. Hiding them would silently shrink the action
        space relative to production.
        """
        spec = self.backend.spec(task_id or self.task_id)
        tools: list[dict[str, Any]] = []
        for tool in spec.tools:
            description = tool.description or f"{tool.tool_class.value} tool ({tool.name})"
            if not tool.supported:
                description = (
                    f"UNSUPPORTED in this reconstructed environment: "
                    f"{tool.unsupported_reason}. Calling it returns an error."
                )
            tools.append(
                {
                    "name": tool.name,
                    "description": description,
                    "inputSchema": tool.input_schema or {"type": "object", "properties": {}},
                    "_meta": {
                        "bandits/tool_class": tool.tool_class.value,
                        "bandits/supported": tool.supported,
                    },
                }
            )
        return tools

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Dispatch one action into exactly one episode's session.

        Returns an MCP ``CallToolResult``: a text content block carrying the
        tool's JSON response, ``structuredContent`` with the same data plus the
        episode's step accounting, and ``isError`` set for both refusals and
        real environment errors -- distinguishable via
        ``structuredContent.error``.
        """
        episode = self._episode(episode_id)
        try:
            outcome = episode.step(name, dict(arguments or {}))
        except UnsupportedToolCall as exc:
            return _error_result(
                f"{exc.tool} is not supported by this environment: {exc.reason}",
                {"error": "unsupported_tool", "tool": exc.tool, "reason": exc.reason},
            )
        except EpisodeFinished as exc:
            return _error_result(str(exc), {"error": "episode_done"})
        except EpisodeClosed as exc:
            return _error_result(str(exc), {"error": "episode_closed"})

        obs = outcome.observation
        structured: dict[str, Any] = {
            "response": obs.response,
            "status": obs.status.value,
            "error_kind": obs.error_kind,
            "episode_id": episode.episode_id,
            "step": outcome.step,
            "reward": outcome.reward,
            "done": outcome.done,
            "truncated": outcome.truncated,
        }
        return {
            "content": [{"type": "text", "text": obs.text or json.dumps(obs.response, default=str)}],
            "structuredContent": structured,
            "isError": obs.status.value == "error",
        }

    # -- JSON-RPC ----------------------------------------------------------

    def handle(self, request: Any) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC request. ``None`` for a notification."""
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
            return _rpc_error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 request object")
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            return _rpc_error(rid, INVALID_REQUEST, "missing method")
        if not isinstance(params, dict):
            return _rpc_error(rid, INVALID_PARAMS, "params must be an object")
        if rid is None and method.startswith("notifications/"):
            return None
        try:
            result = self._dispatch(method, params)
        except JsonRpcError as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": exc.to_obj()}
        except Exception as exc:  # structured, never a traceback
            return _rpc_error(rid, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if rid is None:
            return None
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": self.backend.spec(self.task_id).instruction,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.list_tools(params.get("task_id"))}
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise JsonRpcError(INVALID_PARAMS, "tools/call requires a string 'name'")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")
            return self.call_tool(name, arguments, episode_id=params.get("episode_id"))
        if method == "env/spec":
            return self.backend.spec(params.get("task_id") or self.task_id).model_dump(mode="json")
        if method == "env/reset":
            episode_id = self.open_episode(params.get("task_id"), params.get("seed"))
            episode = self.registry.get(episode_id)
            return {
                "episode_id": episode_id,
                "task_id": episode.task_id,
                "instruction": episode.task.instruction,
                "max_steps": episode.max_steps,
            }
        if method == "env/close":
            return {"closed": self.close_episode(params.get("episode_id"))}
        raise JsonRpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")


def _error_result(text: str, structured: dict[str, Any]) -> dict[str, Any]:
    """An MCP tool result that is loudly a failure.

    Tool failures are reported in the *result* (``isError``), not as a protocol
    error, so the agent sees them and can react -- which is the whole point of
    reproducing production's failure modes.
    """
    return {"content": [{"type": "text", "text": text}], "structuredContent": structured, "isError": True}


def _rpc_error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
