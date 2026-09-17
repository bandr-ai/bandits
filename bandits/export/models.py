"""Shared contracts for rebuilding an episode as a chat-format transcript."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from bandits.traces import Contract


class ToolFunction(Contract):
    name: str
    arguments: str
    """The call arguments as a JSON string, per the chat-completions convention.

    A string rather than an object because that is what trainers parse. Traces
    that recorded arguments as text bandits could not parse are serialized back
    out unchanged rather than being guessed at.
    """


class ToolCall(Contract):
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class TrainingMessage(Contract):
    """One message in a training transcript, in chat-completions shape.

    The action the agent chose belongs on an ``assistant`` message as a tool
    call, never on the ``tool`` message carrying the result. A ``tool`` message
    is what was handed back to the model — context, not target — so an argument
    recorded there is on the wrong side of the loss and teaches nothing.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def shape_matches_role(self) -> TrainingMessage:
        if self.role == "system" and not self.content:
            raise ValueError("a system message requires the instruction it carries")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only an assistant message may announce tool calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message must name the call it answers")
        if self.role != "tool" and self.tool_call_id:
            raise ValueError("only a tool message may carry a tool_call_id")
        if self.content is None and not self.tool_calls:
            # An assistant turn that only calls a tool legitimately has no text.
            # Anything else with no content is an empty message, not a message.
            raise ValueError(f"a {self.role} message requires content")
        ids = [call.id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("an assistant message announces a tool call id twice")
        return self

    def as_chat_message(self) -> dict[str, Any]:
        """Render only the keys this message actually carries.

        The stored artifact stays lossless; the emitted file does not. A trainer
        reading ``tool_calls: []`` on a user message either rejects the row or
        quietly treats it as a turn that called nothing.
        """
        row: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            row["content"] = self.content
        if self.name is not None:
            row["name"] = self.name
        if self.tool_calls:
            row["tool_calls"] = [call.model_dump(mode="json") for call in self.tool_calls]
        if self.tool_call_id is not None:
            row["tool_call_id"] = self.tool_call_id
        return row


class RejectedTrace(Contract):
    trace_id: str
    family_id: str | None = None
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def has_reason(self) -> RejectedTrace:
        if not self.reasons:
            raise ValueError("a rejected trace must explain why")
        return self

    def jsonl_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
