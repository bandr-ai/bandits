"""Standard OTLP/JSON adapter.

Reads what an OpenTelemetry exporter actually writes: ``ExportTraceServiceRequest``
objects in the OTLP/JSON encoding (``resourceSpans`` → ``scopeSpans`` → ``spans``,
attributes as ``{key, value: AnyValue}`` lists, nanosecond timestamps). One per
line (the collector's file exporter), one per file, or an array of them; a
directory is read recursively as one export, since a batching exporter splits a
single trace across requests.

OTLP itself says nothing about models or tools. That meaning comes from an
instrumentation convention, and each one is read only for what it declares:

=====================  ==========================  ==================================
convention             kind attribute              messages
=====================  ==========================  ==================================
OTel GenAI semconv     ``gen_ai.operation.name``   ``gen_ai.input/output.messages``;
                                                   legacy ``gen_ai.prompt.N.*`` and
                                                   ``gen_ai.*.message`` span events
OpenInference          ``openinference.span.kind`` ``llm.input/output_messages.N.*``,
                                                   ``input.value``/``output.value``
OpenLLMetry            ``traceloop.span.kind``,    ``gen_ai.prompt/completion.N.*``,
                       ``llm.request.type``        ``traceloop.entity.input/output``
Langfuse               ``langfuse.observation      ``langfuse.observation.input/
                       .type``                     output``, ``input/output.value``
=====================  ==========================  ==================================

Every span is classified as a model call, a tool call, a declared workflow
*step* (a chain, agent, task or retriever node), or something with no model or
tool meaning (an embedding, an evaluator, an HTTP span nobody labeled). Nothing
is classified from its name or its content.

A step that contains no model or tool call anywhere beneath it is work a fixed
pipeline did between model calls — a retrieval, a rerank, a permission filter.
Its input and output are the evidence the next model call reacted to, so by
default it becomes a TOOL span marked ``call_recorded=False``: kept for
analysis and judging, but never replayed as a call the model decided to make.
``pipeline_steps=False`` leaves such steps out. A step that does contain
model or tool calls is structure; its children represent it.

Messages found under any convention are written to ``gen_ai.input.messages`` /
``gen_ai.output.messages`` in the GenAI parts form only when the span did not
already declare them, with ``bandits.*_messages_from`` naming where they came
from. Every attribute the span declared is kept unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bandits.genai import PIPELINE_STEP
from bandits.ingest.otlp import (
    _LINEAGE_KEYS,
    _declared_completion,
    _declared_task,
    assemble_corpus,
)
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_source
from bandits.traces import (
    Span,
    SpanKind,
    SpanStatus,
    TraceCorpus,
    TraceIssue,
    UserTurn,
    WorkflowDeclaration,
    WorkflowNode,
)

SOURCE = "otlp-std"

_MODEL = "model"
_TOOL = "tool"
_STEP = "step"
_NONE = "none"
_EXCLUDED = "excluded"
"""Classifications. ``_NONE`` is a span with no model or tool meaning;
``_EXCLUDED`` one that must not reach the corpus, nor anything beneath it."""

_TRANSPORT_RESPONSE_REPR = re.compile(r"^<[^<>]+ \[[1-5]\d\d [A-Za-z ]+\] type=<class '[^']+'>>$")

_CONVENTIONS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "gen_ai.operation.name",
        {
            "chat": _MODEL,
            "text_completion": _MODEL,
            "generate_content": _MODEL,
            "response": _MODEL,
            "execute_tool": _TOOL,
            "invoke_agent": _STEP,
            "invoke_workflow": _STEP,
            "create_agent": _NONE,
            "embeddings": _NONE,
        },
    ),
    (
        "openinference.span.kind",
        {
            "LLM": _MODEL,
            "TOOL": _TOOL,
            "CHAIN": _STEP,
            "AGENT": _STEP,
            "RETRIEVER": _STEP,
            "RERANKER": _STEP,
            "GUARDRAIL": _STEP,
            "EMBEDDING": _NONE,
            "EVALUATOR": _EXCLUDED,
            "PROMPT": _NONE,
            "UNKNOWN": _NONE,
        },
    ),
    (
        "traceloop.span.kind",
        {"tool": _TOOL, "task": _STEP, "workflow": _STEP, "agent": _STEP},
    ),
    ("llm.request.type", {"chat": _MODEL, "completion": _MODEL}),
    (
        "langfuse.observation.type",
        {
            "GENERATION": _MODEL,
            "TOOL": _TOOL,
            "CHAIN": _STEP,
            "AGENT": _STEP,
            "SPAN": _STEP,
            "RETRIEVER": _STEP,
            "GUARDRAIL": _STEP,
            "EMBEDDING": _NONE,
            "EVALUATOR": _EXCLUDED,
            "EVENT": _NONE,
        },
    ),
    # Vercel AI SDK telemetry: outer generateText/streamText spans enclose
    # individual provider calls. Only doGenerate/doStream is a model action.
    # Attribute names and operations follow MLflow's VercelAITranslator and
    # https://ai-sdk.dev/docs/ai-sdk-core/telemetry.
    (
        "ai.operationId",
        {
            "ai.generateText": _STEP,
            "ai.generateText.doGenerate": _MODEL,
            "ai.streamText": _STEP,
            "ai.streamText.doStream": _MODEL,
            "ai.generateObject": _STEP,
            "ai.generateObject.doGenerate": _MODEL,
            "ai.streamObject": _STEP,
            "ai.streamObject.doStream": _MODEL,
            "ai.toolCall": _TOOL,
            "ai.embed": _NONE,
            "ai.embed.doEmbed": _NONE,
            "ai.embedMany": _NONE,
            "ai.embedMany.doEmbed": _NONE,
        },
    ),
)
"""Checked in order; the first convention that declares a recognized value wins,
except that an exclusion declared by any convention wins over all of them.

An evaluator is a label on the episode attached after the fact, often by a
model call of its own. Letting it, or the LLM-as-judge call inside it, into the
trajectory would leak the grade into whatever is judged or trained from it. A
span that some library also stamped ``chat`` is still that evaluator."""

_LINEAGE = (*_LINEAGE_KEYS, "langfuse.session.id", "langfuse.session_id")

_INPUT_VALUE_KEYS = (
    "input.value",
    "langfuse.observation.input",
    "traceloop.entity.input",
    "gen_ai.prompt",
    "ai.prompt.messages",
    "ai.prompt",
    "ai.toolCall.args",
    "braintrust.input_json",
)
_OUTPUT_VALUE_KEYS = (
    "output.value",
    "langfuse.observation.output",
    "traceloop.entity.output",
    "gen_ai.completion",
    "ai.response.text",
    "ai.response.object",
    "ai.response.toolCalls",
    "ai.toolCall.result",
    "braintrust.output_json",
)
_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name", "traceloop.entity.name", "ai.toolCall.name")

# Key candidates verified against the OTel/OpenInference conventions and the
# MLflow translator tables. Every original key remains in attributes; this
# small index is for callers that need a convention-independent view.
_SCALAR_KEYS: dict[str, tuple[str, ...]] = {
    "model": (
        "gen_ai.response.model",
        "gen_ai.request.model",
        "llm.response.model_name",
        "llm.model_name",
        "llm.request.model_name",
        "embedding.model_name",
        "ai.response.model",
        "ai.model.id",
    ),
    "provider": ("gen_ai.provider.name", "llm.provider", "gen_ai.system", "ai.model.provider"),
    "input_tokens": (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.prompt_tokens",
        "llm.token_count.prompt",
        "ai.usage.promptTokens",
    ),
    "output_tokens": (
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.completion_tokens",
        "llm.token_count.completion",
        "ai.usage.completionTokens",
    ),
    "total_tokens": (
        "gen_ai.usage.total_tokens",
        "llm.token_count.total",
        "llm.usage.total_tokens",
    ),
    "cache_read_tokens": (
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cache_read_input_tokens",
        "llm.token_count.prompt_details.cache_read",
    ),
    "cache_write_tokens": (
        "gen_ai.usage.cache_creation.input_tokens",
        "gen_ai.usage.cache_creation_input_tokens",
        "llm.token_count.prompt_details.cache_write",
    ),
    "reasoning_tokens": (
        "gen_ai.usage.reasoning_tokens",
        "llm.token_count.completion_details.reasoning",
    ),
    "finish_reasons": (
        "gen_ai.response.finish_reasons",
        "llm.finish_reason",
        "ai.response.finishReason",
    ),
}


def _scalar_index(attributes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {"source": key, "value": attributes[key]}
        for name, keys in _SCALAR_KEYS.items()
        if (found := _first_value(attributes, keys)) is not None
        for key in (found[0],)
    }


_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "developer": "system",
    "function": "tool",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "AIMessageChunk": "assistant",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "FunctionMessage": "tool",
}
_GENAI_PART_TYPES = {"text", "tool_call", "tool_call_response", "reasoning", "blob", "file", "uri"}
_EVENT_ROLES = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}


# ---------- OTLP/JSON decoding ----------


def _any_value(value: object) -> object:
    """An OTLP ``AnyValue`` as a plain Python value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "arrayValue" in value:
        return [_any_value(item) for item in (value["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in value:
        return _attributes((value["kvlistValue"] or {}).get("values"))
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _attributes(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):  # some SDKs emit a plain map; accept it as declared
        return dict(raw)
    if not isinstance(raw, list):
        return {}
    return {
        item["key"]: _any_value(item.get("value"))
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _timestamp(value: object) -> datetime | None:
    """Unix nanoseconds (string or int, as OTLP/JSON allows) to a UTC datetime."""
    if isinstance(value, bool):
        return None
    try:
        nanos = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    if nanos <= 0:
        return None
    seconds, remainder = divmod(nanos, 1_000_000_000)
    try:
        return datetime.fromtimestamp(seconds, tz=UTC) + timedelta(microseconds=remainder // 1000)
    except (OverflowError, OSError, ValueError):
        return None  # past what a datetime can hold: not a time this span ran at


def _is_error(status: object, attributes: dict[str, Any]) -> bool:
    code = status.get("code") if isinstance(status, dict) else None
    return (
        code in (2, "2", "STATUS_CODE_ERROR")
        or attributes.get("error.type") not in (None, "")
        or attributes.get("langfuse.observation.level") == "ERROR"
    )


# ---------- message normalization ----------


_UNPARSED: Any = object()
"""A value that opens as a JSON object or array and does not parse: in practice
an exporter that cut a long value off. Read as text it would put half a
serialized message list in the transcript as if someone had typed it."""


def _json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in '[{"':
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Prose can open with a bracket; only a JSON-looking opening is cut JSON.
        return _UNPARSED if stripped.startswith(('{"', "[{", '["', "[[")) else value


def _unflatten(attributes: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """``prefix.0.role``, ``prefix.0.tool_calls.1.name`` … as a list of nested dicts."""
    root: dict[Any, Any] = {}
    marker = prefix + "."
    for key, value in attributes.items():
        if not key.startswith(marker):
            continue
        node = root
        parts = [int(p) if p.isdigit() else p for p in key[len(marker) :].split(".")]
        if not isinstance(parts[0], int):
            continue
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {}
            node = child
        node[parts[-1]] = value
    return [m for m in _listify(root) if isinstance(m, dict)] if root else []


def _listify(node: object) -> Any:
    if not isinstance(node, dict):
        return node
    if node and all(isinstance(key, int) for key in node):
        return [_listify(node[key]) for key in sorted(node)]
    return {key: _listify(value) for key, value in node.items()}


def _openinference_message(item: dict[str, Any]) -> dict[str, Any]:
    """``{message: {role, content, contents, tool_calls}}`` in OpenAI shape."""
    message = item.get("message") if isinstance(item.get("message"), dict) else item
    content = message.get("content")
    if content is None and isinstance(message.get("contents"), list):
        content = [
            {"type": "text", "text": c["message_content"].get("text")}
            for c in message["contents"]
            if isinstance(c, dict)
            and isinstance(c.get("message_content"), dict)
            and c["message_content"].get("text") is not None
        ]
    calls = [
        call["tool_call"] if isinstance(call.get("tool_call"), dict) else call
        for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    return {
        "role": message.get("role"),
        "content": content,
        "tool_calls": calls,
        "tool_call_id": message.get("tool_call_id"),
        "name": message.get("name"),
    }


def _call_part(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = function.get("arguments", function.get("args", function.get("input")))
    part: dict[str, Any] = {"type": "tool_call", "name": name, "arguments": arguments}
    if isinstance(call.get("id"), str):
        part["id"] = call["id"]
    return part


def _content_parts(content: object) -> list[dict[str, Any]]:
    """Text and tool parts from a message's content, across provider shapes."""
    if isinstance(content, str):
        encoded = _json_value(content) if content.lstrip().startswith("[{") else None
        if (
            isinstance(encoded, list)
            and encoded
            and all(isinstance(p, dict) and ("text" in p or "type" in p) for p in encoded)
        ):
            content = encoded  # a parts list serialized into the content string
        else:
            return [{"type": "text", "content": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in _GENAI_PART_TYPES and ("content" in item or kind != "text"):
            parts.append(item)
        elif isinstance(item.get("text"), str):  # OpenAI, Anthropic, Gemini text
            parts.append({"type": "text", "content": item["text"]})
        elif kind == "tool_use":  # Anthropic
            part = _call_part(item)
            if part:
                parts.append(part)
        elif kind == "tool_result":  # Anthropic
            parts.append(
                {
                    "type": "tool_call_response",
                    "id": item.get("tool_use_id"),
                    "result": item.get("content"),
                }
            )
        elif isinstance(item.get("functionCall") or item.get("function_call"), dict):  # Gemini
            part = _call_part(item.get("functionCall") or item["function_call"])
            if part:
                parts.append(part)
        elif isinstance(item.get("functionResponse") or item.get("function_response"), dict):
            response = item.get("functionResponse") or item["function_response"]  # Gemini
            parts.append(
                {
                    "type": "tool_call_response",
                    "id": response.get("id"),
                    "name": response.get("name"),
                    "result": response.get("response"),
                }
            )
    return parts


def _message(raw: object) -> dict[str, Any] | None:
    """One chat message, in any provider's shape, in the GenAI parts form."""
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("kwargs"), dict) and raw.get("lc") is not None:  # LangChain dump
        ids = raw.get("id")
        raw = {**raw["kwargs"], "role": raw["kwargs"].get("type") or (ids[-1] if ids else None)}
    role = raw.get("role") or raw.get("type")
    role = _ROLE_ALIASES.get(role, role) if isinstance(role, str) else None
    if role not in ("system", "user", "assistant", "tool"):
        return None

    source = raw.get("parts") if isinstance(raw.get("parts"), list) else raw.get("content")
    parts = _content_parts(source)
    if role == "tool":
        call_id = raw.get("tool_call_id") or raw.get("id")
        if isinstance(call_id, str) and not any(p["type"] == "tool_call_response" for p in parts):
            text = "\n".join(p["content"] for p in parts if p["type"] == "text")
            parts = [
                {"type": "tool_call_response", "id": call_id, "result": text if text else source}
            ]
    # LangChain keeps the provider's calls in additional_kwargs when its own
    # tool_calls field is empty.
    extra = raw.get("additional_kwargs") if isinstance(raw.get("additional_kwargs"), dict) else {}
    calls = _json_value(raw.get("tool_calls") or extra.get("tool_calls"))
    calls = list(calls) if isinstance(calls, list) else []
    function_call = _json_value(raw.get("function_call") or extra.get("function_call"))
    if isinstance(function_call, dict):  # OpenAI legacy
        calls.append(function_call)
    for call in calls:
        call = _json_value(call)  # some exporters serialize each call separately
        if isinstance(call, dict) and (part := _call_part(call)):
            parts.append(part)
    message: dict[str, Any] = {"role": role, "parts": parts}
    if isinstance(raw.get("name"), str):
        message["name"] = raw["name"]
    return message


def _gemini_system(request: dict[str, Any]) -> list[dict[str, Any]]:
    config = request.get("config") if isinstance(request.get("config"), dict) else {}
    system = next(
        (
            v
            for v in (
                request.get("system_instruction"),
                request.get("systemInstruction"),
                config.get("system_instruction"),
                config.get("systemInstruction"),
            )
            if v
        ),
        None,
    )
    if isinstance(system, dict):  # Content object
        system = system.get("parts")
    parts = _content_parts(system)
    return [{"role": "system", "parts": parts}] if parts else []


def _messages(value: object, default_role: str | None = None) -> list[dict[str, Any]] | None:
    """A message list, when *value* is one in any common shape; else None.

    ``default_role`` applies only when no item in a list names a role: a
    role-less ``[{"content": ...}]`` recorded as a call's input or output.
    """
    value = _json_value(value)
    if value is _UNPARSED:
        return None
    system: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            value = value["messages"]
        elif default_role == "user" and _prompt_strings(value) is not None:
            # A completion request: OpenAI ``prompt``, LangChain LLM ``prompts``.
            value = [{"role": "user", "content": text} for text in _prompt_strings(value)]
        elif isinstance(value.get("contents"), (list, str)):  # Gemini request
            system = _gemini_system(value)
            contents = value["contents"]
            # The SDK accepts a bare string as one user turn.
            value = (
                [{"role": "user", "content": contents}] if isinstance(contents, str) else contents
            )
        elif isinstance(value.get("generations"), list):  # LangChain chat model result
            generations = value["generations"]
            value = [
                generation.get("message")
                or {"role": "assistant", "content": generation.get("text")}
                for batch in generations
                for generation in (batch if isinstance(batch, list) else [batch])
                if isinstance(generation, dict)
            ]
        elif isinstance(value.get("choices"), list):  # OpenAI response
            value = [c.get("message") for c in value["choices"] if isinstance(c, dict)]
        elif isinstance(value.get("candidates"), list):  # Gemini response
            value = [c.get("content") for c in value["candidates"] if isinstance(c, dict)]
        else:
            value = [value]
    if not isinstance(value, list):
        return None
    if value and all(isinstance(item, list) for item in value):  # LangChain batches
        value = [message for batch in value for message in batch]
    if (
        default_role
        and value
        and all(
            isinstance(item, dict)
            and "content" in item
            and item.get("role") is None
            and item.get("type") is None
            for item in value
        )
    ):
        value = [{**item, "role": default_role} for item in value]
    messages = [
        m for m in (_message(item) for item in value if not _tool_definition(item)) if m is not None
    ]
    return [*system, *messages] or None


def _prompt_strings(request: dict[str, Any]) -> list[str] | None:
    for key in ("prompt", "prompts"):
        prompt = request.get(key)
        if isinstance(prompt, str) and prompt:
            return [prompt]
        if isinstance(prompt, list) and prompt and all(isinstance(p, str) for p in prompt):
            return prompt
    return None


def _tool_definition(item: object) -> bool:
    """A tool schema some integrations list among messages as ``role: tool``.

    It describes what could be called; it is not a turn. The raw value keeps it.
    """
    if not isinstance(item, dict) or item.get("role") != "tool":
        return False
    content = _json_value(item.get("content"))
    if not isinstance(content, dict):
        return False
    if isinstance(content.get("name"), str) and (
        "input_schema" in content or "parameters" in content
    ):
        return True  # Anthropic / bare function schema
    function = content.get("function")
    return (
        content.get("type") == "function"
        and isinstance(function, dict)
        and isinstance(function.get("name"), str)
        and "arguments" not in function
    )


def _recorded_attribute_value(value: object, direction: str) -> object:
    """The message value inside an OTLP key/value list stored as an I/O value.

    Langfuse keeps span-event attributes it received over OTLP this way, e.g.
    ``[{"key": "gen_ai.prompt", "value": {"stringValue": ...}}]``. Anything else
    is returned unchanged.
    """
    if not (
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) and set(item) == {"key", "value"} for item in value)
    ):
        return value
    decoded = _attributes(value)
    keys = (
        ("gen_ai.input.messages", "gen_ai.prompt")
        if direction == "input"
        else ("gen_ai.output.messages", "gen_ai.completion")
    )
    found = _first_value(decoded, keys)
    return _json_value(found[1]) if found is not None else value


def _text_message(role: str, value: object) -> list[dict[str, Any]] | None:
    return (
        [{"role": role, "parts": [{"type": "text", "content": value}]}]
        if (isinstance(value, str) and value)
        else None
    )


def _vercel_output_messages(attributes: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Provider reply from the AI SDK's recorded text and tool-call fields."""
    parts: list[dict[str, Any]] = []
    text = attributes.get("ai.response.text")
    if isinstance(text, str) and text:
        parts.append({"type": "text", "content": text})
    calls = _json_value(attributes.get("ai.response.toolCalls"))
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("toolName"), str):
                continue
            part: dict[str, Any] = {
                "type": "tool_call",
                "name": call["toolName"],
                "arguments": _json_value(call.get("input")),
            }
            if isinstance(call.get("toolCallId"), str):
                part["id"] = call["toolCallId"]
            parts.append(part)
    return [{"role": "assistant", "parts": parts}] if parts else None


def _event_messages(events: list[dict[str, Any]]) -> tuple[list | None, list | None]:
    """Messages carried on span events, in the legacy GenAI event conventions."""
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for event in events:
        name, attrs = event["name"], event["attributes"]
        if name == "gen_ai.content.prompt":
            inputs.extend(_messages(attrs.get("gen_ai.prompt")) or [])
        elif name == "gen_ai.content.completion":
            outputs.extend(_messages(attrs.get("gen_ai.completion")) or [])
        elif name in _EVENT_ROLES:
            body = _json_value(attrs.get("gen_ai.event.content", attrs.get("content")))
            if body is _UNPARSED:
                continue
            body = body if isinstance(body, dict) else {"content": body}
            message = _message({**attrs, **body, "role": _EVENT_ROLES[name]})
            if message:
                inputs.append(message)
        elif name == "gen_ai.choice":
            body = _json_value(attrs.get("gen_ai.event.content", attrs.get("message")))
            if body is _UNPARSED:
                continue
            if isinstance(body, dict) and isinstance(body.get("message"), dict):
                body = body["message"]
            message = _message(
                {"role": "assistant", **body}
                if isinstance(body, dict)
                else {"role": "assistant", "content": body}
            )
            if message:
                outputs.append(message)
    return inputs or None, outputs or None


def _first_value(attributes: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, object] | None:
    return next(((key, attributes[key]) for key in keys if key in attributes), None)


def _normalized_messages(
    attributes: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    prompt_is_text: bool,
    unparsed: Counter[str] | None = None,
) -> dict[str, Any]:
    """GenAI message attributes a span did not declare, read from other conventions.

    A value that looks like cut-off JSON is counted into *unparsed* and read as
    nothing, never as text.
    """
    added: dict[str, Any] = {}
    details = next(
        (
            e["attributes"]
            for e in events
            if e["name"] == "gen_ai.client.inference.operation.details"
        ),
        {},
    )
    event_in, event_out = _event_messages(events)
    for direction, role, flat_legacy, flat_oi, event_messages, value_keys in (
        ("input", "user", "gen_ai.prompt", "llm.input_messages", event_in, _INPUT_VALUE_KEYS),
        (
            "output",
            "assistant",
            "gen_ai.completion",
            "llm.output_messages",
            event_out,
            _OUTPUT_VALUE_KEYS,
        ),
    ):
        key = f"gen_ai.{direction}.messages"
        if attributes.get(key) is not None:
            if unparsed is not None and _json_value(attributes[key]) is _UNPARSED:
                unparsed[key] += 1
            declared = _messages(attributes[key])
            # A producer can declare an empty assistant message while also
            # recording a real function call in flattened legacy attributes.
            # Keep the declared bytes in source_context, but let substantive
            # recorded content fill the normalized view.
            if declared and any(message.get("parts") for message in declared):
                continue
        candidates: list[tuple[str, Any]] = [
            (f"event:{key}", _messages(details.get(key))),
            (
                f"{flat_legacy}.N",
                _messages(
                    [
                        {**item, "role": item.get("role") or role}
                        for item in _unflatten(attributes, flat_legacy)
                    ]
                ),
            ),
            (
                f"{flat_oi}.N",
                _messages([_openinference_message(m) for m in _unflatten(attributes, flat_oi)]),
            ),
            ("events", event_messages),
        ]
        if direction == "output" and attributes.get("ai.operationId") is not None:
            candidates.append(("ai.response", _vercel_output_messages(attributes)))
        declared = _first_value(attributes, value_keys)
        if declared is not None:
            parsed = _json_value(declared[1])
            if parsed is _UNPARSED:
                if unparsed is not None:
                    unparsed[declared[0]] += 1
                found = None
            else:
                parsed = _recorded_attribute_value(parsed, direction)
                found = _messages(parsed, default_role=role)
                if found is None and prompt_is_text:
                    found = _text_message(role, parsed)
            candidates.append((declared[0], found))
        origin, messages = next(
            ((o, m) for o, m in candidates if m and any(message.get("parts") for message in m)),
            (None, None),
        )
        if messages:
            added[key] = messages
            added[f"bandits.{direction}_messages_from"] = origin
    if attributes.get("gen_ai.system_instructions") is None and isinstance(
        details.get("gen_ai.system_instructions"), (str, list)
    ):
        added["gen_ai.system_instructions"] = details["gen_ai.system_instructions"]
    return added


# ---------- reading ----------


class _Decoded:
    """One OTLP span, decoded but not yet classified into the canonical model."""

    __slots__ = (
        "index",
        "location",
        "trace_id",
        "span_id",
        "parent_id",
        "name",
        "started_at",
        "ended_at",
        "attributes",
        "events",
        "source_context",
        "error",
        "role",
        "label",
    )

    def __init__(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self, key, value)


def _classify(attributes: dict[str, Any]) -> tuple[str, str]:
    found: tuple[str, str] | None = None
    first_declared: str | None = None
    for key, table in _CONVENTIONS:
        value = attributes.get(key)
        if value is None:
            continue
        label = f"{key}={value}"
        role = table.get(value) if isinstance(value, str) else None
        if role == _EXCLUDED:
            return role, label
        if role is None:
            first_declared = first_declared or label
        elif found is None:
            found = (role, label)
    braintrust = _json_value(attributes.get("braintrust.span_attributes"))
    if isinstance(braintrust, dict):
        kind = braintrust.get("type")
        role = {"llm": _MODEL, "tool": _TOOL, "task": _STEP, "score": _EXCLUDED}.get(kind)
        if role == _EXCLUDED:
            return role, f"braintrust.span_attributes.type={kind}"
        if found is None and role is not None:
            return role, f"braintrust.span_attributes.type={kind}"
    if (
        found is not None
        and found[0] == _MODEL
        and attributes.get("gen_ai.operation.name") is None
        and _embedding_model(attributes)
    ):
        # A generic "generation" kind (Langfuse GENERATION, OpenInference LLM)
        # also covers embedding requests; an embedding has no reply to learn.
        return _NONE, f"{found[1]} with embedding model"
    return found or (_NONE, first_declared or "no declared kind")


_MODEL_NAME_KEYS = ("gen_ai.request.model", "llm.model_name", "langfuse.observation.model.name")


def _embedding_model(attributes: dict[str, Any]) -> bool:
    return any(
        isinstance(attributes.get(key), str) and "embed" in attributes[key].lower()
        for key in _MODEL_NAME_KEYS
    )


def _exclude_subtrees(spans: dict[str, _Decoded]) -> None:
    """Mark everything beneath an excluded span excluded, under its label."""
    for span in spans.values():
        if span.role == _EXCLUDED:
            continue
        parent, seen = span.parent_id, {span.span_id}
        while parent in spans and parent not in seen:
            if spans[parent].role == _EXCLUDED:
                span.role, span.label = _EXCLUDED, spans[parent].label
                break
            seen.add(parent)
            parent = spans[parent].parent_id


def _requests(data: bytes, location: str) -> Iterator[tuple[str, object]]:
    """Each ``ExportTraceServiceRequest`` in a file, with where it was found.

    A whole-file document (one request, or an array of them) first; otherwise
    one request per line, where a line that is not UTF-8 JSON is its own error
    and never costs its neighbors. Bytes are never replaced to make them decode.
    """
    try:
        whole = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        whole = None
    if whole is not None:
        for i, item in enumerate(whole if isinstance(whole, list) else [whole]):
            yield f"{location}[{i}]" if isinstance(whole, list) else location, item
        return
    for number, line in enumerate(data.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            yield f"{location}:{number}", json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            yield f"{location}:{number}", exc


def _decode_file(
    path: Path, data: bytes, issues: list[TraceIssue], counter: list[int]
) -> Iterator[_Decoded]:
    for location, request in _requests(data, str(path)):
        if isinstance(request, (json.JSONDecodeError, UnicodeDecodeError)):
            issues.append(TraceIssue(kind="malformed_json", detail=str(request), location=location))
            continue
        resource_spans = (
            (request.get("resourceSpans") or request.get("batches"))
            if isinstance(request, dict)
            else None
        )
        if not isinstance(resource_spans, list):
            issues.append(
                TraceIssue(
                    kind="malformed_record",
                    detail="expected an OTLP ExportTraceServiceRequest with 'resourceSpans'",
                    location=location,
                )
            )
            continue
        for r, resource_span in enumerate(resource_spans):
            if not isinstance(resource_span, dict):
                continue
            resource = _attributes((resource_span.get("resource") or {}).get("attributes"))
            scopes = (
                resource_span.get("scopeSpans")
                or resource_span.get("instrumentationLibrarySpans")
                or []
            )
            for s, scope_span in enumerate(scopes):
                if not isinstance(scope_span, dict):
                    continue
                scope = scope_span.get("scope") or scope_span.get("instrumentationLibrary") or {}
                for k, raw in enumerate(scope_span.get("spans") or []):
                    where = f"{location}#resourceSpans[{r}].scopeSpans[{s}].spans[{k}]"
                    counter[0] += 1
                    decoded = _decode_span(raw, resource, scope, where, counter[0], issues)
                    if decoded is not None:
                        yield decoded


def _decode_span(
    raw: object,
    resource: dict[str, Any],
    scope: object,
    location: str,
    index: int,
    issues: list[TraceIssue],
) -> _Decoded | None:
    if not isinstance(raw, dict):
        issues.append(
            TraceIssue(kind="malformed_span", detail="span is not an object", location=location)
        )
        return None
    trace_id, span_id, name = raw.get("traceId"), raw.get("spanId"), raw.get("name")
    missing = [
        field
        for field, value in (("traceId", trace_id), ("spanId", span_id), ("name", name))
        if not isinstance(value, str) or not value
    ]
    if missing:
        issues.append(
            TraceIssue(
                kind="malformed_span",
                detail=f"missing or invalid field(s): {', '.join(missing)}",
                location=location,
            )
        )
        return None
    started_at = _timestamp(raw.get("startTimeUnixNano"))
    ended_at = _timestamp(raw.get("endTimeUnixNano"))
    if started_at is None or ended_at is None:
        issues.append(
            TraceIssue(
                kind="malformed_span",
                detail="startTimeUnixNano/endTimeUnixNano must be positive unix "
                "nanoseconds within the representable calendar",
                location=location,
            )
        )
        return None

    # A resource attribute describes every span it carries (service, session),
    # but a span's own declaration is the more specific claim.
    span_attributes = _attributes(raw.get("attributes"))
    attributes = {**resource, **span_attributes}
    if isinstance(scope, dict) and isinstance(scope.get("name"), str):
        attributes.setdefault("otel.scope.name", scope["name"])
    events = [
        {
            "name": event.get("name"),
            "attributes": _attributes(event.get("attributes")),
            "timeUnixNano": event.get("timeUnixNano"),
            "droppedAttributesCount": event.get("droppedAttributesCount"),
        }
        for event in raw.get("events") or []
        if isinstance(event, dict) and isinstance(event.get("name"), str)
    ]
    role, label = _classify(attributes)
    return _Decoded(
        index=index,
        location=location,
        trace_id=trace_id,
        span_id=span_id,
        parent_id=raw.get("parentSpanId") or None,
        name=name,
        started_at=started_at,
        ended_at=max(started_at, ended_at),
        attributes=attributes,
        events=events,
        source_context={
            "resource": resource,
            "scope": scope,
            "span_attributes": span_attributes,
            "links": raw.get("links", []),
            "events": raw.get("events", []),
            "status": raw.get("status"),
        },
        error=_is_error(raw.get("status"), attributes),
        role=role,
        label=label,
    )


def _cyclic(spans: dict[str, _Decoded]) -> set[str]:
    """Spans whose parent chain loops back on itself.

    Each span has one parent, so walking up from every span once finds every
    loop: a walk that meets a span already on its own path has closed one.
    """
    state: dict[str, int] = {}  # 1: on the current walk, 2: settled
    looped: set[str] = set()
    for start in spans:
        path: list[str] = []
        node: str | None = start
        while node in spans and node not in state:
            state[node] = 1
            path.append(node)
            node = spans[node].parent_id
        if node in spans and state.get(node) == 1:
            looped.update(path[path.index(node) :])
        for visited in path:
            state[visited] = 2
    return looped


def _pipeline_steps(spans: dict[str, _Decoded]) -> set[str]:
    """Declared steps with no model or tool call beneath them, outermost only."""
    children: dict[str, list[str]] = {}
    for span in spans.values():
        if span.parent_id is not None:
            children.setdefault(span.parent_id, []).append(span.span_id)

    has_action: dict[str, bool] = {}
    for root in spans:  # iterative post-order; exported trees can be deep
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in has_action:
                continue
            if not expanded:
                stack.append((node, True))
                stack.extend((c, False) for c in children.get(node, ()) if c not in has_action)
                continue
            has_action[node] = spans[node].role in (_MODEL, _TOOL) or any(
                has_action.get(c, False) for c in children.get(node, ())
            )

    def eligible(span_id: str) -> bool:
        span = spans[span_id]
        # The root is the episode itself, not a step inside it.
        return span.role == _STEP and span.parent_id is not None and not has_action[span_id]

    selected = set()
    for span_id in spans:
        if not eligible(span_id):
            continue
        parent, seen = spans[span_id].parent_id, {span_id}
        while parent in spans and parent not in seen and not eligible(parent):
            seen.add(parent)
            parent = spans[parent].parent_id
        if parent not in spans or parent in seen:
            selected.add(span_id)
    return selected


def _collapse_model_wrappers(spans: dict[str, _Decoded]) -> list[tuple[_Decoded, _Decoded]]:
    """Treat near-identical enclosing instrumentation as structure, not a second call.

    A parent/child relation or shared model name alone does not establish this:
    agents may genuinely nest model calls. We require a direct child, matching
    model and status, nearly identical timing, and compatible recorded replies.
    Both original records remain available (as a workflow node and raw source).
    """
    children: dict[str, list[_Decoded]] = {}
    for span in spans.values():
        if span.parent_id in spans:
            children.setdefault(span.parent_id, []).append(span)

    def model(span: _Decoded) -> str | None:
        return next(
            (
                value
                for key in (
                    "gen_ai.request.model",
                    "gen_ai.response.model",
                    "llm.model_name",
                    "llm.request.model_name",
                )
                if isinstance(value := span.attributes.get(key), str) and value
            ),
            None,
        )

    def reply(span: _Decoded) -> object:
        attrs = {
            **span.attributes,
            **_normalized_messages(span.attributes, span.events, prompt_is_text=True),
        }
        return _declared_completion(attrs)

    collapsed: list[tuple[_Decoded, _Decoded]] = []
    for parent in spans.values():
        if parent.role != _MODEL:
            continue
        model_children = [
            child for child in children.get(parent.span_id, ()) if child.role == _MODEL
        ]
        if len(model_children) != 1:
            continue
        child = model_children[0]
        if parent.error != child.error or model(parent) is None or model(parent) != model(child):
            continue
        duration = (parent.ended_at - parent.started_at).total_seconds()
        tolerance = min(0.05, duration * 0.05)
        if duration <= 0 or tolerance <= 0:
            continue
        start_gap = (child.started_at - parent.started_at).total_seconds()
        end_gap = (parent.ended_at - child.ended_at).total_seconds()
        if not (0 <= start_gap <= tolerance and 0 <= end_gap <= tolerance):
            continue
        parent_reply, child_reply = reply(parent), reply(child)
        if parent_reply is not None and child_reply is not None and parent_reply != child_reply:
            continue
        if (parent_reply is None) != (child_reply is None) and child_reply != "None":
            continue
        # Two blank replies provide no independent evidence that the operations
        # are the same. An enclosing tool call and an empty provider text are
        # compatible: provider instrumentors often render a tool-only reply as
        # the literal string "None".
        if parent_reply is None and child_reply in (None, "None"):
            parent_messages = _normalized_messages(
                parent.attributes, parent.events, prompt_is_text=True
            ).get("gen_ai.output.messages", parent.attributes.get("gen_ai.output.messages"))
            parsed = _json_value(parent_messages)
            if not (
                isinstance(parsed, list)
                and any(
                    part.get("type") == "tool_call"
                    for message in parsed
                    if isinstance(message, dict)
                    for part in message.get("parts", ())
                    if isinstance(part, dict)
                )
            ):
                continue
        parent.role = _STEP
        parent.attributes["bandits.duplicate_model_of"] = child.span_id
        collapsed.append((parent, child))
    return collapsed


def _covered(spans: dict[str, _Decoded], selected: set[str]) -> set[str]:
    """Spans inside a selected step, which that step already represents."""
    covered = set()
    for span_id in spans:
        parent, seen = spans[span_id].parent_id, {span_id}
        while parent in spans and parent not in seen:
            if parent in selected:
                covered.add(span_id)
                break
            seen.add(parent)
            parent = spans[parent].parent_id
    return covered


def _io_value(attributes: dict[str, Any], keys: tuple[str, ...], unparsed: Counter[str]) -> object:
    """The first declared value under *keys*, parsed when it is JSON.

    Cut-off JSON is counted and kept as the string it was: a tool's recorded
    result is still what the tool returned, however much of it survived.
    """
    found = _first_value(attributes, keys)
    if found is None:
        return None
    return _counted(found[0], found[1], unparsed)


def _counted(key: str, value: object, unparsed: Counter[str]) -> object:
    parsed = _json_value(value)
    if parsed is _UNPARSED:
        unparsed[key] += 1
        return value
    return parsed


def _to_span(decoded: _Decoded, *, as_step: bool, unparsed: Counter[str]) -> Span:
    attributes = dict(decoded.attributes)
    # Keep namespaces distinct. The flattened attributes above remain for
    # existing convention translators; these are the recorded source facts.
    attributes["bandits.otlp.source_context"] = decoded.source_context
    if decoded.events:
        attributes["otel.events"] = decoded.events
    attributes["bandits.declared_kind"] = decoded.label
    status = SpanStatus.ERROR if decoded.error else SpanStatus.OK

    if decoded.role == _MODEL:
        # Only a model call's input is a conversation. A step's input is
        # whatever state the pipeline passed it, and reading messages out of
        # that would invent user turns the model never received.
        attributes.update(
            _normalized_messages(
                decoded.attributes, decoded.events, prompt_is_text=True, unparsed=unparsed
            )
        )
        attributes["bandits.normalized_scalars"] = _scalar_index(decoded.attributes)
        output = _declared_completion(attributes)
        if output is None and attributes.get("gen_ai.output.messages") is None:
            # Only when no response message was found at all: a structured
            # answer the model returned as data. A response that was messages
            # but held no text made tool calls, which are recovered as tool
            # spans, and dumping it here would read as the model saying JSON.
            output = _first_value(attributes, _OUTPUT_VALUE_KEYS)
            output = _json_value(output[1]) if output is not None else None
            if output is _UNPARSED:
                output = None
        if isinstance(output, str) and _TRANSPORT_RESPONSE_REPR.fullmatch(output):
            # Some instrumentors record a Python HTTP response object rather
            # than its model completion. Keep the source field, but do not let
            # this placeholder become a training target or a judge answer.
            attributes["bandits.output_unusable_reason"] = "transport_response_object_repr"
            output = None
        return Span(
            span_id=decoded.span_id,
            parent_span_id=decoded.parent_id,
            kind=SpanKind.MODEL,
            name=decoded.name,
            started_at=decoded.started_at,
            ended_at=decoded.ended_at,
            status=status,
            output=output,
            attributes=attributes,
        )

    arguments = attributes.get("gen_ai.tool.call.arguments")
    if arguments is not None:
        arguments = _counted("gen_ai.tool.call.arguments", arguments, unparsed)
    else:
        arguments = _io_value(attributes, _INPUT_VALUE_KEYS, unparsed)
    output = attributes.get("gen_ai.tool.call.result")
    if output is not None:
        output = _counted("gen_ai.tool.call.result", output, unparsed)
    else:
        output = _io_value(attributes, _OUTPUT_VALUE_KEYS, unparsed)
    name = (
        next(
            (
                attributes[k]
                for k in _TOOL_NAME_KEYS
                if isinstance(attributes.get(k), str) and attributes[k]
            ),
            decoded.name,
        )
        if not as_step
        else decoded.name
    )
    if as_step:
        attributes[PIPELINE_STEP] = True
    return Span(
        span_id=decoded.span_id,
        parent_span_id=decoded.parent_id,
        kind=SpanKind.TOOL,
        name=name,
        started_at=decoded.started_at,
        ended_at=decoded.ended_at,
        status=status,
        arguments=arguments
        if isinstance(arguments, dict)
        else ({} if arguments is None else {"input": arguments}),
        output=output,
        # A step ran because the pipeline is written that way, not because a
        # model asked for it: nothing recorded a decision to call it.
        call_recorded=not as_step,
        attributes=attributes,
    )


def _task(spans: list[Span], decoded: dict[str, _Decoded]) -> str | None:
    """The root's declared ``task``, else the first user instruction recorded.

    Never read from an excluded span: an evaluator's prompt is a grading
    rubric, and taking it for the task would group and judge by the grade.
    """
    candidates = [s for s in decoded.values() if s.role != _EXCLUDED]
    for span in candidates:
        if span.parent_id is None and isinstance(span.attributes.get("task"), str):
            return span.attributes["task"]
    for span in spans:
        task = _declared_task(span.attributes)
        if task:
            return task
    # A wrapper span (a LangGraph root, an agent invocation) often carries the
    # conversation it was started with; only a message-shaped input counts.
    for span in sorted(candidates, key=lambda s: (s.started_at, s.index)):
        messages = _normalized_messages(span.attributes, span.events, prompt_is_text=False)
        task = _declared_task({**span.attributes, **messages})
        if task:
            return task
    return None


def _episode_root(spans: dict[str, _Decoded]) -> _Decoded | None:
    """The span the episode started from, whether or not it was kept.

    An agent or workflow wrapper is where GenAI instrumentation declares the
    system instructions and tool definitions for the whole run; filtering it
    out as structure must not take that context with it. A root whose parent
    was not exported counts, so a partial export still has one.
    """
    roots = [
        s
        for s in spans.values()
        if (s.parent_id is None or s.parent_id not in spans) and s.role != _EXCLUDED
    ]
    return min(roots, key=lambda s: (s.parent_id is not None, s.started_at, s.index), default=None)


_CONTAINER_MARKERS = (("langfuse.synthetic_span", "trace_record"),)
"""Attributes by which a converter declares a span it made itself to hold
trace-level fields. Such a span records nothing the application did: it is a
structural root, never the invocation."""


def _is_container(span: _Decoded) -> bool:
    return any(span.attributes.get(key) == value for key, value in _CONTAINER_MARKERS)


def _framework(attributes: dict[str, Any]) -> dict[str, Any]:
    """Framework metadata a node recorded (``langgraph_step`` and its kind)."""
    return {
        key.rsplit(".", 1)[-1]: value
        for key, value in attributes.items()
        if key.rsplit(".", 1)[-1].startswith("langgraph_")
    }


def _invocation_candidates(spans: dict[str, _Decoded]) -> list[str]:
    """Outermost spans once converter containers are looked through.

    A model or tool call is never a candidate: a lone call surviving at the top
    of a partial export is not the application run that made it.
    """
    out = []
    for span in sorted(spans.values(), key=lambda s: (s.started_at, s.index)):
        if _is_container(span) or span.role in (_MODEL, _TOOL, _EXCLUDED):
            continue
        parent, seen, outermost = span.parent_id, {span.span_id}, True
        while parent in spans and parent not in seen:
            if not _is_container(spans[parent]):
                outermost = False
                break
            seen.add(parent)
            parent = spans[parent].parent_id
        if outermost:
            out.append(span.span_id)
    return out


def _files(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))


def load_otlp_standard(
    path: Path,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    pipeline_steps: bool = True,
    workflow: WorkflowDeclaration | None = None,
) -> TraceCorpus:
    """Read a standard OTLP/JSON export (a file or a directory) into one corpus.

    ``workflow`` declares the export a program-driven workflow. Then no model
    input becomes a user turn, the task comes only from the declared fields of
    the invocation record, steps containing model calls are kept as structure,
    and each model call gets evidence links (``bandits.ingest.workflow``).
    """
    from bandits.ingest.workflow import build_request

    workflow_extras: dict[str, dict[str, Any]] = {}
    workflow_counts: Counter[str] = Counter()
    files = _files(path)
    if not files:
        raise FileNotFoundError(f"no .json or .jsonl files under {path}")

    issues: list[TraceIssue] = []
    digests: list[str] = []
    by_trace: dict[str, dict[str, _Decoded]] = {}
    counter = [0]
    ruleset_name = ruleset.name
    for file in files:
        source = redact_source(file, ruleset)
        ruleset_name = source.ruleset
        issues.extend(source.issues)
        digests.append(
            f"{file.relative_to(path) if path.is_dir() else file.name}\0{source.source_digest}"
        )
        for decoded in _decode_file(file, source.data, issues, counter):
            spans = by_trace.setdefault(decoded.trace_id, {})
            if decoded.span_id in spans:
                issues.append(
                    TraceIssue(
                        kind="duplicate_span",
                        detail=f"span {decoded.span_id} of trace {decoded.trace_id} was exported "
                        "more than once; the first copy is kept",
                        location=decoded.location,
                    )
                )
                continue
            spans[decoded.span_id] = decoded

    # One file is its own digest. A directory is one export split into parts,
    # so its digest covers every part's exact bytes and its relative path.
    source_digest = (
        digests[0].split("\0", 1)[1]
        if not path.is_dir()
        else hashlib.sha256("\n".join(digests).encode()).hexdigest()
    )

    unrepresented: Counter[str] = Counter()
    containers: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    unparsed: Counter[str] = Counter()
    spans_by_trace: dict[str, list[tuple[int, Span]]] = {}
    task_by_trace: dict[str, str] = {}
    lineage_by_trace: dict[str, str] = {}
    episode_attributes: dict[str, dict[str, Any]] = {}
    for trace_id, decoded in by_trace.items():
        looped = _cyclic(decoded)
        for span_id in sorted(looped):
            issues.append(
                TraceIssue(
                    kind="malformed_span",
                    detail=f"span {span_id} of trace {trace_id} is its own ancestor through "
                    "parentSpanId; a trace is a tree, so it cannot be placed",
                    location=decoded[span_id].location,
                )
            )
        decoded = {k: v for k, v in decoded.items() if k not in looped}
        if not decoded:
            continue
        _exclude_subtrees(decoded)
        for wrapper, call in _collapse_model_wrappers(decoded):
            issues.append(
                TraceIssue(
                    kind="duplicate_model_instrumentation",
                    detail=f"trace {trace_id}: enclosing model span {wrapper.span_id} and "
                    f"provider span {call.span_id} appear to record one call; "
                    "the enclosing record is retained as structure",
                    location=wrapper.location,
                )
            )
        steps = _pipeline_steps(decoded)
        covered = _covered(decoded, steps)
        collected: list[tuple[int, Span]] = []
        for span in sorted(decoded.values(), key=lambda s: s.index):
            if span.role in (_MODEL, _TOOL):
                collected.append((span.index, _to_span(span, as_step=False, unparsed=unparsed)))
            elif span.span_id in steps:
                if pipeline_steps:
                    collected.append((span.index, _to_span(span, as_step=True, unparsed=unparsed)))
                else:
                    unrepresented[span.label] += 1
            elif _is_container(span):
                containers[span.label] += 1
            elif span.role == _EXCLUDED:
                excluded[span.label] += 1
            elif (
                span.role == _NONE
                and span.span_id not in covered
                # A workflow keeps these as structure (below).
                and workflow is None
            ):
                unrepresented[span.label] += 1
            # Otherwise a step with calls beneath it (represented by them) or a
            # span inside a selected step (represented by the step).
            lineage = next(
                (
                    span.attributes[k]
                    for k in _LINEAGE
                    if isinstance(span.attributes.get(k), str) and span.attributes[k]
                ),
                None,
            )
            if lineage is not None:
                lineage_by_trace.setdefault(trace_id, lineage)
        if not collected:
            if all(span.role == _EXCLUDED for span in decoded.values()):
                issues.append(
                    TraceIssue(
                        kind="excluded_evaluator_trace",
                        detail=f"trace {trace_id} contains only evaluator spans",
                    )
                )
            else:
                issues.append(
                    TraceIssue(
                        kind="empty_trace",
                        detail=f"trace {trace_id} has no span declared as a model call, tool call "
                        "or pipeline step",
                    )
                )
            lineage_by_trace.pop(trace_id, None)
            continue
        spans_by_trace[trace_id] = collected
        root = _episode_root(decoded)
        if workflow is not None:
            candidates = _invocation_candidates(decoded)
            scratch: Counter[str] = Counter()
            records = {
                span_id: {
                    "input": _io_value(decoded[span_id].attributes, _INPUT_VALUE_KEYS, scratch),
                    "output": _io_value(decoded[span_id].attributes, _OUTPUT_VALUE_KEYS, scratch),
                    "status": SpanStatus.ERROR if decoded[span_id].error else SpanStatus.OK,
                }
                for span_id in candidates
            }
            request = build_request(candidates=candidates, records=records, declaration=workflow)
            if request.source_span_id is None:
                issues.append(
                    TraceIssue(
                        kind="ambiguous_invocation",
                        detail=f"trace {trace_id}: {request.invocation_basis}; candidates: "
                        f"{', '.join(request.candidate_span_ids) or 'none'}",
                    )
                )
            else:
                # The invocation, not the converter's container, is the episode's context.
                root = decoded[request.source_span_id]
                # It is the episode record, never one of its actions: a code-only
                # invocation would otherwise also be kept as a pipeline step.
                spans_by_trace[trace_id] = [
                    pair for pair in collected if pair[1].span_id != request.source_span_id
                ]
            workflow_counts[request.task_status] += 1
            # Every recorded step not already kept as a span is structure: steps
            # containing calls, and the code-only records inside a kept pipeline
            # step, whose output is often its own (a scoring, a search backend).
            kept_as_span = steps if pipeline_steps else set()
            nodes = tuple(
                WorkflowNode(
                    span_id=span.span_id,
                    parent_span_id=span.parent_id,
                    name=span.name,
                    started_at=span.started_at,
                    ended_at=span.ended_at,
                    input=_io_value(span.attributes, _INPUT_VALUE_KEYS, unparsed),
                    output=_io_value(span.attributes, _OUTPUT_VALUE_KEYS, unparsed),
                    status=SpanStatus.ERROR if span.error else SpanStatus.OK,
                    framework=_framework(span.attributes),
                    attributes={
                        key: value
                        for key, value in span.attributes.items()
                        if key not in _INPUT_VALUE_KEYS and key not in _OUTPUT_VALUE_KEYS
                    }
                    | {"bandits.otlp.source_context": span.source_context},
                )
                for span in sorted(decoded.values(), key=lambda s: (s.started_at, s.index))
                if span.role in (_STEP, _NONE)
                and span.span_id not in kept_as_span
                and span.span_id != request.source_span_id
                and not _is_container(span)
            )
            turns: tuple[UserTurn, ...] = ()
            if request.origin == "human" and request.task is not None:
                turns = (UserTurn(text=request.task, after_span_id=None, origin="declared"),)
            workflow_extras[trace_id] = {
                "interaction": "workflow",
                "request": request,
                "workflow_nodes": nodes,
                "user_turns": turns,
                "task_source": (
                    f"declared field {request.task_path} of span {request.source_span_id}"
                    if request.task_path
                    else None
                ),
            }
            if request.task is not None:
                task_by_trace[trace_id] = request.task
            if root is not None:
                episode_attributes[trace_id] = {
                    **root.attributes,
                    **_normalized_messages(root.attributes, root.events, prompt_is_text=False),
                    "bandits.otlp.source_context": root.source_context,
                }
            continue
        if root is not None:
            episode_attributes[trace_id] = {
                **root.attributes,
                **_normalized_messages(root.attributes, root.events, prompt_is_text=False),
                "bandits.otlp.source_context": root.source_context,
            }
        ordered = [s for _, s in sorted(collected, key=lambda p: (p[1].started_at, p[0]))]
        task = _task(ordered, decoded)
        if task is not None:
            task_by_trace[trace_id] = task

    for key, count in sorted(unparsed.items()):
        issues.append(
            TraceIssue(
                kind="unparsed_value",
                detail=f"{count} value(s) under {key} open as JSON but do not parse, as an "
                "exporter that cuts long values off leaves them; none was read as message text",
                location=str(path),
            )
        )
    for label, count in sorted(unrepresented.items()):
        issues.append(
            TraceIssue(
                kind="unrepresented_span",
                detail=f"{count} span(s) with {label} carry no model or tool call and are not "
                "in the corpus",
                location=str(path),
            )
        )
    for label, count in sorted(containers.items()):
        issues.append(
            TraceIssue(
                kind="source_container",
                detail=f"{count} converter container span(s) with {label} are kept in the "
                "source archive, not as application actions",
                location=str(path),
            )
        )
    for label, count in sorted(excluded.items()):
        issues.append(
            TraceIssue(
                kind="excluded_evaluator",
                detail=f"{count} evaluator span(s) with {label} excluded from the action corpus",
                location=str(path),
            )
        )

    if workflow is not None:
        for status in ("unresolved", "conflict"):
            if workflow_counts[status]:
                issues.append(
                    TraceIssue(
                        kind=f"task_{status}",
                        detail=f"{workflow_counts[status]} workflow trace(s) have their task "
                        f"{status}; see each trace's request.task_reason",
                        location=str(path),
                    )
                )

    corpus = assemble_corpus(
        spans_by_trace,
        task_by_trace=task_by_trace,
        lineage_by_trace=lineage_by_trace,
        source=SOURCE,
        source_digest=source_digest,
        issues=issues,
        redaction_ruleset=ruleset_name,
        episode_attributes=episode_attributes,
        trace_extras=workflow_extras,
    )
    if workflow is None:
        return corpus
    from bandits.ingest.workflow import build_evidence

    traces = tuple(
        trace.replace(evidence=build_evidence(trace.spans, trace.workflow_nodes, trace.request))
        for trace in corpus.traces
    )
    return corpus.replace(traces=traces, workflow=workflow)
