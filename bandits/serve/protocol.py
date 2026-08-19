"""The wire contract for a served bandits environment.

This module is the *only* thing a client needs to agree with. It is pure
pydantic and has no imports from the runtime, so a trainer can vendor it,
generate types from it, or reimplement it in another language without pulling
in SQLite or the reconstruction pipeline.

Endpoints
---------

=========================  ======================================  =========================
Method / path              Request body                            Response body
=========================  ======================================  =========================
``GET  /health``           --                                      :class:`HealthResponse`
``GET  /spec``             -- (optional ``?task_id=``)              :class:`EnvSpec`
``POST /reset``            :class:`ResetRequest`                   :class:`ResetResponse`
``POST /step``             :class:`StepRequest`                    :class:`StepResponse`
``POST /close``            :class:`CloseRequest`                   :class:`CloseResponse`
=========================  ======================================  =========================

Every non-2xx response is an :class:`ErrorResponse` -- structured JSON with a
machine-readable :class:`ErrorCode`, never an HTML traceback. A trainer that
loses a rollout needs to know *which* failure it was (episode gone? cap hit?
tool unsupported?) without scraping a stack trace.

Episode model
-------------

``/reset`` mints a fresh ``episode_id`` backed by its own materialized
environment session. ``/step`` routes by ``episode_id`` and touches nothing
else; there is no implicit "current episode" anywhere in this protocol, because
an implicit current episode is exactly how parallel rollouts corrupt each
other. ``/close`` tears the session down. See :mod:`bandits.serve.server`.

Versioning
----------

:data:`PROTOCOL_VERSION` is bumped on any change a client could notice. It is
returned by ``/health`` and ``/spec``; a client should refuse a server whose
major version differs from its own.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from bandits.contracts import CallStatus, JsonObject, JsonValue, ToolClass

__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "Action",
    "CloseRequest",
    "CloseResponse",
    "EnvSpec",
    "ErrorCode",
    "ErrorResponse",
    "HealthResponse",
    "Observation",
    "ResetRequest",
    "ResetResponse",
    "RewardRange",
    "StepRequest",
    "StepResponse",
    "ToolSchema",
    "protocol_major",
]

PROTOCOL_NAME = "bandits-env"
"""Identifies this wire protocol. Sent in ``/health`` and ``/spec``."""

PROTOCOL_VERSION = "1.0.0"
"""Semantic version of the wire contract. Bump on any client-visible change."""


def protocol_major(version: str) -> int:
    """Major component of a protocol version string, for compatibility checks."""
    try:
        return int(version.split(".", 1)[0])
    except ValueError:  # pragma: no cover - malformed peer version
        return -1


class Wire(BaseModel):
    """Base for every wire type: unknown fields are rejected, not ignored.

    Silently dropping an unknown field is how a trainer ends up believing it
    passed a seed that the server never saw.
    """

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class ErrorCode(str, Enum):
    """Machine-readable failure classes. The HTTP status is a coarse echo of these."""

    BAD_REQUEST = "bad_request"
    """Body was not valid JSON, or failed model validation. 400."""

    NOT_FOUND = "not_found"
    """Unknown route, unknown ``task_id``, or unknown ``episode_id``. 404."""

    EPISODE_CLOSED = "episode_closed"
    """The ``episode_id`` existed but has been closed or evicted. 409."""

    EPISODE_LIMIT = "episode_limit"
    """The live-episode cap is full and the server refuses to admit another. 429."""

    UNSUPPORTED_TOOL = "unsupported_tool"
    """The environment refuses to fake this tool (see ``UnsupportedToolError``). 422.

    This is *not* an environment error observation. An observed failure the real
    tool would also have returned comes back as a normal 200 ``StepResponse``
    with ``observation.status == "error"``. This code means the reconstruction
    could not model the tool at all, so no reward computed after it would be
    trustworthy.
    """

    EPISODE_DONE = "episode_done"
    """A step was sent to an episode that already returned ``done`` or ``truncated``. 409."""

    INTERNAL = "internal"
    """Anything unhandled. 500. The detail is a message, never a traceback."""


class ErrorResponse(Wire):
    """The body of every non-2xx response."""

    error: ErrorCode
    """Machine-readable failure class. Branch on this, not on the message."""

    message: str
    """Human-readable one-liner. Safe to log; contains no traceback."""

    episode_id: str | None = None
    """The episode the failure concerned, when the request named one."""

    detail: JsonObject = Field(default_factory=dict)
    """Extra structured context, e.g. ``{"limit": 64}`` for ``episode_limit``."""

    protocol_version: str = PROTOCOL_VERSION


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------


class ToolSchema(Wire):
    """One action in the reconstructed action space."""

    name: str
    description: str = ""
    input_schema: JsonObject = Field(default_factory=dict)
    """JSON Schema for ``arguments``.

    Taken verbatim from the declared tool registry when the corpus had one
    (``ToolProfile.declared_schema``); otherwise synthesized from the argument
    fields actually observed in traces, which is a narrower but honest schema.
    """

    tool_class: ToolClass = ToolClass.UNKNOWN
    """read / write / external / unknown, from stage 2. Carried so an agent
    harness can, for instance, surface which actions are irreversible."""

    supported: bool = True
    """False when the environment could not reimplement this tool.

    An unsupported tool is still advertised -- hiding it would make the action
    space differ from production -- but calling it fails loudly with
    :attr:`ErrorCode.UNSUPPORTED_TOOL` rather than returning a fake success.
    """

    unsupported_reason: str | None = None
    """Why, when ``supported`` is False. Operator-facing."""


class RewardRange(Wire):
    """Closed interval the reward is guaranteed to fall in."""

    low: float = 0.0
    high: float = 1.0


class EnvSpec(Wire):
    """Everything needed to construct a rollout worker for this environment.

    Fetched once at startup by a trainer, then cached. The digests are the
    reproducibility anchor: two runs that report the same ``env_digest`` and
    ``verifier_digest`` graded against the same world with the same reward code.
    """

    protocol: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION

    env_id: str
    """Identity of the materialized environment (``EnvManifest.env_id``)."""

    task_id: str
    """The task this spec describes. ``/reset`` without a ``task_id`` uses it."""

    task_ids: tuple[str, ...] = ()
    """Every task this server can reset into. A ``TaskSuite``, flattened."""

    instruction: str
    """The task statement given to the agent, verbatim from the mined trace."""

    tools: tuple[ToolSchema, ...] = ()
    """The action space."""

    max_steps: int
    """Hard cap on actions per episode. Hitting it sets ``truncated``."""

    reward_range: RewardRange = Field(default_factory=RewardRange)

    env_digest: str
    """Digest of the reconstructed world (schema digest of the store)."""

    verifier_digest: str | None = None
    """Digest of the reward code (assertions) grading this task.

    ``None`` when the server has no verifier for the task, in which case every
    step returns reward 0.0 and ``done`` only ever comes from the step cap. A
    trainer should treat a ``None`` here as a misconfiguration, not as a task.
    """

    static_entities: tuple[str, ...] = ()
    """Entities materialized as verbatim snapshots because their structure was
    never determined. Documented rather than invented."""

    unsupported_tools: tuple[str, ...] = ()
    """Convenience mirror of the tools with ``supported=False``."""


# --------------------------------------------------------------------------
# reset / step / close
# --------------------------------------------------------------------------


class ResetRequest(Wire):
    """Ask for a fresh episode."""

    task_id: str | None = None
    """Which task to instantiate. ``None`` means the server's default task."""

    seed: int | None = None
    """Recorded on the episode and echoed back in ``info``.

    Reconstruction is deterministic (no clocks, no randomness), so the seed does
    not change the world today. It is carried because a trainer needs the
    rollout it logged to be identifiable, and because task *generation* will use
    it later.
    """


class Observation(Wire):
    """What the environment returned for one action, plus enough framing to
    render it to a policy."""

    response: JsonValue = None
    """The tool's response body, exactly as the reconstructed tool produced it."""

    status: CallStatus = CallStatus.OK
    """``ok`` or ``error``. An ``error`` here is real environment dynamics --
    the failure the production tool would also have returned."""

    error_kind: str | None = None
    """Coarse label when ``status`` is ``error``, e.g. ``already_refunded``."""

    text: str = ""
    """A flat rendering of ``response`` for text-in/text-out policies. JSON."""


class ResetResponse(Wire):
    """A new episode."""

    episode_id: str
    """Opaque handle. Every subsequent ``/step`` and ``/close`` must carry it."""

    task_id: str
    instruction: str
    """The task statement. Repeated here so a rollout worker needs only /reset."""

    observation: Observation
    """The opening observation. There is no action yet, so ``response`` carries
    the task framing: instruction and the tool names available."""

    tools: tuple[ToolSchema, ...] = ()
    """The action space for this episode, identical to ``/spec``'s."""

    max_steps: int
    step: int = 0
    seed: int | None = None


class Action(Wire):
    """One tool call."""

    name: str
    arguments: JsonObject = Field(default_factory=dict)


class StepRequest(Wire):
    episode_id: str
    action: Action


class StepResponse(Wire):
    """The result of one action. Gym semantics.

    ``done`` and ``truncated`` are disjoint in meaning: ``done`` is the task's
    own termination (the verifier passed), ``truncated`` is the harness cutting
    the episode off at ``max_steps``. A trainer bootstraps value past a
    truncation and not past a termination, so conflating them is a real bug.
    """

    episode_id: str
    observation: Observation
    reward: float = 0.0
    """Reward for the state *after* this action. Computed by
    :func:`bandits.verify.run.evaluate` -- assertions over state and effects,
    never a judge. 0.0 when the server has no verifier for the task."""

    done: bool = False
    """The task's own success condition was met."""

    truncated: bool = False
    """The step cap was reached. Not a failure of the policy."""

    step: int
    """Number of actions taken in this episode, including this one."""

    info: JsonObject = Field(default_factory=dict)
    """Diagnostics. Keys currently emitted: ``env_digest`` (state digest after
    the action), ``effects`` (count of recorded external effects), ``seed``,
    ``task_id``, and ``assertions`` (per-assertion pass/fail) when a verifier
    graded the step."""


class CloseRequest(Wire):
    episode_id: str


class CloseResponse(Wire):
    """Acknowledgement of teardown. Idempotent: closing twice is not an error,
    ``already_closed`` just becomes True."""

    episode_id: str
    closed: bool = True
    already_closed: bool = False
    steps: int = 0
    final_digest: str | None = None
    """State digest at teardown, for post-hoc grading and audit."""


class HealthResponse(Wire):
    """Liveness plus enough capacity information to schedule against."""

    ok: bool = True
    protocol: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION
    live_episodes: int = 0
    max_episodes: int = 0
    episodes_started: int = 0
    uptime_seconds: float = 0.0
