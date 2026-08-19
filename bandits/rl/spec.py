"""``EnvSpec`` -- the machine-readable description a trainer needs.

A trainer never sees a :class:`~bandits.contracts.StateSchema` or a
:class:`~bandits.contracts.Verifier`. What it needs is much smaller and much
more stable: what the task says, which tools may be called and with what
arguments, how long an episode may run, what range the reward lives in, and
enough identity to prove two runs graded against the same world.

Everything here is derivable from artifacts the deterministic half of the
pipeline already produced. Nothing is invented: a tool with no declared JSON
Schema gets a schema reconstructed from its *observed* argument fields, marked
``"x-bandits-source": "observed"``, rather than a plausible-looking one.

Reproducibility identity
------------------------
Three digests travel with every spec:

``schema_digest``
    the inferred database. Two envs with the same digest have the same tables.
``verifier_digest``
    the reward function, byte-for-byte. If this changes, rewards are not
    comparable across runs, no matter how similar the numbers look.
``spec_digest``
    everything above plus the action space and the step budget.

A training run that logs only the reward and not these three cannot say what it
trained against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from bandits.contracts import (
    JsonObject,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolProfile,
    ToolSurface,
    Verifier,
)

__all__ = [
    "EnvSpec",
    "ToolSpec",
    "canonical_json",
    "declared_description",
    "digest_of",
    "json_schema_for",
]


def canonical_json(value: Any) -> str:
    """Stable rendering used for every digest in this package."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_JSON_TYPE_ORDER = ("null", "boolean", "integer", "number", "string", "array", "object")


def _types_of(profile_types: tuple[str, ...]) -> list[str]:
    """Observed JSON type names, ordered so the schema is stable across runs."""
    seen = [t for t in _JSON_TYPE_ORDER if t in profile_types]
    extra = sorted(t for t in profile_types if t not in _JSON_TYPE_ORDER)
    return seen + extra


def _unwrap_declared(declared: JsonObject | None) -> JsonObject | None:
    """Pull the argument schema out of a registry record.

    ``ToolProfile.declared_schema`` is whatever stage 2 was handed: sometimes the
    JSON Schema itself, sometimes the whole registry record with the schema under
    ``input_schema`` (Anthropic) or ``parameters`` (OpenAI). Both spellings are
    unwrapped here rather than in the caller, so a spec never publishes
    ``{"description": ..., "input_schema": {...}}`` as if it were an argument
    schema.
    """
    if not declared:
        return None
    if "properties" in declared or declared.get("type") == "object":
        return declared
    for key in ("input_schema", "parameters", "schema"):
        inner = declared.get(key)
        if isinstance(inner, dict):
            return inner
    return declared


def declared_description(profile: ToolProfile) -> str:
    """The registry's own description of a tool, when it published one."""
    declared = profile.declared_schema or {}
    value = declared.get("description")
    return str(value) if isinstance(value, str) else ""


def json_schema_for(profile: ToolProfile) -> JsonObject:
    """The JSON Schema for one tool's arguments.

    Preference order, and the reason for it:

    1. ``declared_schema`` -- the registry the customer publishes to their own
       agent. Authoritative for the action space (PLAN.md Step 6).
    2. reconstructed from ``argument_fields`` -- what production actually sent.
       Marked ``"x-bandits-source": "observed"`` so nobody mistakes it for the
       real contract, and ``required`` holds only fields present in *every*
       observed call, which is evidence rather than a guess.
    3. an open object -- when there is neither. Empty ``properties`` says "we do
       not know", which is honest; a fabricated property list would not be.
    """
    declared = _unwrap_declared(profile.declared_schema)
    if declared:
        schema = dict(declared)
        schema.setdefault("x-bandits-source", "declared")
        return schema
    if not profile.argument_fields:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
            "x-bandits-source": "unknown",
        }
    properties: JsonObject = {}
    required: list[str] = []
    for f in sorted(profile.argument_fields, key=lambda p: p.name):
        types = _types_of(f.json_types)
        prop: JsonObject = {"type": types[0] if len(types) == 1 else types}
        if f.sample_values:
            prop["examples"] = list(f.sample_values[:3])
        if f.looks_like_identifier:
            prop["x-bandits-identifier"] = True
        properties[f.name] = prop
        if profile.call_count and f.occurrences == profile.call_count:
            required.append(f.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "x-bandits-source": "observed",
    }


@dataclass(frozen=True)
class ToolSpec:
    """One callable action, as a trainer sees it."""

    name: str
    tool_class: str
    parameters: JsonObject = field(default_factory=dict)
    """JSON Schema for ``arguments``."""

    description: str = ""

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "tool_class": self.tool_class,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_json(cls, payload: JsonObject) -> ToolSpec:
        return cls(
            name=payload["name"],
            tool_class=payload.get("tool_class", ToolClass.UNKNOWN.value),
            parameters=dict(payload.get("parameters") or {}),
            description=payload.get("description", ""),
        )


@dataclass(frozen=True)
class EnvSpec:
    """Everything a trainer needs to run and reproduce one task's environment."""

    task_id: str
    instruction: str
    tools: tuple[ToolSpec, ...] = ()
    """Tools the agent may call. Unsupported tools are deliberately absent."""

    finish_tool: str = "respond"
    """The terminal action. Carries the agent's final message and ends the episode."""

    max_steps: int = 30
    reward_range: tuple[float, float] = (0.0, 1.0)
    reward_mode: str = "all_or_nothing"
    env_id: str = ""
    trace_id: str = ""
    schema_digest: str = ""
    verifier_id: str = ""
    verifier_digest: str = ""
    verifier_reviewed_by: str | None = None
    outcome_label: bool | None = None
    unsupported_tools: tuple[str, ...] = ()
    """Declared in the surface but not reimplementable. Calling one is an error,
    never a success. Listed so a trainer can explain the error to the policy."""

    static_entities: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # -- action space ------------------------------------------------------

    @property
    def action_space(self) -> JsonObject:
        """A JSON Schema for a single action.

        One shape for every step: ``{"name": ..., "arguments": {...}}``. The
        terminal action uses the same shape with ``name == finish_tool``, so a
        policy never has to learn two action encodings.
        """
        names = [t.name for t in self.tools] + [self.finish_tool]
        return {
            "type": "object",
            "description": (
                "One tool call. Every step has this shape, including the terminal "
                f"action {self.finish_tool!r}, whose arguments carry the final "
                "message as {'message': str}."
            ),
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": {"type": "object"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    @property
    def spec_digest(self) -> str:
        payload = self.to_json()
        payload.pop("spec_digest", None)
        return digest_of(payload)

    # -- serialization -----------------------------------------------------

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "tools": [t.to_json() for t in self.tools],
            "finish_tool": self.finish_tool,
            "action_space": self.action_space,
            "max_steps": self.max_steps,
            "reward_range": list(self.reward_range),
            "reward_mode": self.reward_mode,
            "env_id": self.env_id,
            "trace_id": self.trace_id,
            "schema_digest": self.schema_digest,
            "verifier_id": self.verifier_id,
            "verifier_digest": self.verifier_digest,
            "verifier_reviewed_by": self.verifier_reviewed_by,
            "outcome_label": self.outcome_label,
            "unsupported_tools": list(self.unsupported_tools),
            "static_entities": list(self.static_entities),
            "notes": list(self.notes),
        }
        payload["spec_digest"] = digest_of(payload)
        return payload

    def to_str(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_json(), indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, payload: JsonObject) -> EnvSpec:
        reward_range = tuple(payload.get("reward_range") or (0.0, 1.0))
        return cls(
            task_id=payload["task_id"],
            instruction=payload.get("instruction", ""),
            tools=tuple(ToolSpec.from_json(t) for t in payload.get("tools") or ()),
            finish_tool=payload.get("finish_tool", "respond"),
            max_steps=int(payload.get("max_steps", 30)),
            reward_range=(float(reward_range[0]), float(reward_range[1])),
            reward_mode=payload.get("reward_mode", "all_or_nothing"),
            env_id=payload.get("env_id", ""),
            trace_id=payload.get("trace_id", ""),
            schema_digest=payload.get("schema_digest", ""),
            verifier_id=payload.get("verifier_id", ""),
            verifier_digest=payload.get("verifier_digest", ""),
            verifier_reviewed_by=payload.get("verifier_reviewed_by"),
            outcome_label=payload.get("outcome_label"),
            unsupported_tools=tuple(payload.get("unsupported_tools") or ()),
            static_entities=tuple(payload.get("static_entities") or ()),
            notes=tuple(payload.get("notes") or ()),
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        task: TaskCase,
        verifier: Verifier,
        schema: StateSchema,
        tool_classes: dict[str, ToolClass],
        surface: ToolSurface | None = None,
        unsupported_tools: tuple[str, ...] = (),
        env_id: str = "",
        schema_digest: str = "",
        max_steps: int = 30,
        reward_mode: str = "all_or_nothing",
        finish_tool: str = "respond",
        notes: tuple[str, ...] = (),
    ) -> EnvSpec:
        """Assemble a spec from the pipeline artifacts for one task."""
        tools: list[ToolSpec] = []
        for name in sorted(tool_classes):
            if name in unsupported_tools:
                continue
            klass = tool_classes[name]
            if klass is ToolClass.UNKNOWN:
                # Never in the action space: an UNKNOWN tool cannot be
                # reimplemented, and offering it invites the policy to spend
                # steps on an action that can only error.
                continue
            profile = surface.by_name(name) if surface else None
            tools.append(
                ToolSpec(
                    name=name,
                    tool_class=klass.value,
                    parameters=json_schema_for(profile) if profile else {
                        "type": "object", "properties": {},
                        "additionalProperties": True, "x-bandits-source": "unknown",
                    },
                    description=_describe(name, klass, profile),
                )
            )
        return cls(
            task_id=task.task_id,
            instruction=task.instruction,
            tools=tuple(tools),
            finish_tool=finish_tool,
            max_steps=max_steps,
            reward_range=(0.0, 1.0),
            reward_mode=reward_mode,
            env_id=env_id,
            trace_id=task.trace_id,
            schema_digest=schema_digest,
            verifier_id=verifier.verifier_id,
            verifier_digest=digest_of(verifier.model_dump(mode="json")),
            verifier_reviewed_by=verifier.reviewed_by,
            outcome_label=task.outcome,
            unsupported_tools=tuple(unsupported_tools),
            static_entities=tuple(
                sorted(e.name for e in schema.entities if e.static_snapshot)
            ),
            notes=notes,
        )


def _describe(name: str, klass: ToolClass, profile: ToolProfile | None = None) -> str:
    declared = declared_description(profile) if profile else ""
    prefix = f"{declared} " if declared else ""
    if klass is ToolClass.READ:
        return prefix + f"{name}: reads the environment's state. Does not change anything."
    if klass is ToolClass.WRITE:
        return prefix + f"{name}: changes the environment's state. This is what is graded."
    if klass is ToolClass.EXTERNAL:
        return prefix + (
            f"{name}: an external side effect. It is recorded but never performed; "
            f"the acknowledgement you get back does not mean anything left the system."
        )
    return prefix + name
