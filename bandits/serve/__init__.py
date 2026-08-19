"""Serving layer: make a reconstructed environment reachable over the wire.

Everything upstream of this package runs in-process, which means nothing can
train on it. A distributed RL trainer (slime, verl) drives environments over
HTTP from many workers at once; an agent drives them over MCP. This package is
both doors, on stdlib only -- no FastAPI, no uvicorn -- so a training container
that can `pip install bandits` can serve environments.

    from bandits.serve import EnvClient, SessionBackend, serve

    backend = SessionBackend(schema, tasks, tool_classes, surface=surface,
                             verifiers=verifiers)
    with serve(backend, port=8080) as server:      # loopback by default
        with EnvClient(server.base_url) as env:
            ep = env.reset(seed=0)
            env.step(ep.episode_id, "refund_order", {"order_id": 7741})

Read :mod:`bandits.serve.protocol` for the wire contract,
:mod:`bandits.serve.server` for the episode-isolation guarantees.
"""

from __future__ import annotations

from .backend import (
    DEFAULT_MAX_STEPS,
    EnvBackend,
    Episode,
    EpisodeClosed,
    EpisodeFinished,
    SessionBackend,
    StepOutcome,
    UnsupportedToolCall,
)
from .client import EnvClient, EnvClientError, ProtocolMismatch
from .mcp import McpEnvServer
from .protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Action,
    CloseRequest,
    CloseResponse,
    EnvSpec,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    Observation,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    ToolSchema,
)
from .server import (
    DEFAULT_HOST,
    DEFAULT_MAX_EPISODES,
    EnvHTTPServer,
    EpisodeRegistry,
    ServeError,
    serve,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MAX_EPISODES",
    "DEFAULT_MAX_STEPS",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "Action",
    "CloseRequest",
    "CloseResponse",
    "EnvBackend",
    "EnvClient",
    "EnvClientError",
    "EnvHTTPServer",
    "EnvSpec",
    "Episode",
    "EpisodeClosed",
    "EpisodeFinished",
    "EpisodeRegistry",
    "ErrorCode",
    "ErrorResponse",
    "HealthResponse",
    "McpEnvServer",
    "Observation",
    "ProtocolMismatch",
    "ResetRequest",
    "ResetResponse",
    "ServeError",
    "SessionBackend",
    "StepOutcome",
    "StepRequest",
    "StepResponse",
    "ToolSchema",
    "UnsupportedToolCall",
    "serve",
]
