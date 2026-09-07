"""Shared contracts and deterministic persistence for learning-asset exports."""

from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract


class ExportKind(str, Enum):
    EVAL = "eval"
    SFT = "sft"


class Partition(str, Enum):
    """Which side of a family's split an export may draw from.

    Demonstrations and evaluation cases must not come from the same traces. A
    task trained on and then scored is not a measurement of anything, and the
    split is already grouped by lineage, so drawing the two exports from
    opposite sides is leak-safe by construction rather than by convention.
    """

    FIT = "fit"
    HELD_OUT = "held_out"
    ALL = "all"
    """Every trace in the family. Recorded with a warning: whatever is exported
    twice this way overlaps whatever was exported from the other side."""


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


class EvalCase(Contract):
    case_id: str
    instruction: str
    system_prompt: str | None = None
    """The system instruction the recorded episode ran under, when recorded."""

    tools: tuple[dict[str, Any], ...] | None = None
    """The toolset the recorded episode was offered. None means the source did
    not declare one, so a harness replaying this case cannot reproduce it."""

    grader: dict[str, Any]
    corpus_id: str
    task_set_id: str
    family_id: str
    trace_id: str
    verifier_id: str
    validation_id: str

    def jsonl_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SFTExample(Contract):
    example_id: str
    messages: tuple[TrainingMessage, ...]
    tools: tuple[dict[str, Any], ...] | None = None
    """The toolset the episode was offered, when its source declared one.

    A row showing a call teaches the call's shape; which tool to reach for, out
    of what was on offer, is the decision behind it, and without the schemas the
    row may not be reproducible against a harness at all. None means the source
    declared no toolset — never that the agent had none.
    """

    generating_policy: dict[str, Any]
    warnings: tuple[str, ...] = ()
    """What is imperfect about this demonstration, carried rather than hidden.

    A row good enough to train on is not automatically a row with nothing wrong
    with it, and the difference belongs where whoever trains on it will see it.
    """

    corpus_id: str
    task_set_id: str
    family_id: str
    trace_id: str
    verifier_id: str
    validation_id: str

    @model_validator(mode="after")
    def has_prompt_and_target(self) -> SFTExample:
        # A system instruction may come first, and only first: a row whose
        # instructions arrive after the agent has already acted describes a run
        # nobody had.
        body = (
            self.messages[1:]
            if self.messages[:1] and self.messages[0].role == "system"
            else self.messages
        )
        if any(message.role == "system" for message in body):
            raise ValueError("a system message may only open an SFT example")
        if not body or body[0].role != "user":
            raise ValueError("an SFT example must begin with a user message")
        if not any(message.role == "assistant" for message in body[1:]):
            raise ValueError("an SFT example requires an assistant target")
        if not self.generating_policy.get("models"):
            raise ValueError("an SFT example must record its generating model(s)")
        return self

    @model_validator(mode="after")
    def tool_calls_and_results_pair_up(self) -> SFTExample:
        """Every result answers a call the assistant already made, and every call is answered.

        A transcript that fails this is not merely untidy: a result with no
        announced call trains the model on an observation it was never shown
        asking for, and a call with no result trains it to stop mid-action.
        """
        announced: dict[str, str] = {}
        answered: set[str] = set()
        for message in self.messages:
            for call in message.tool_calls:
                announced[call.id] = call.function.name
            if message.role != "tool":
                continue
            call_id = message.tool_call_id or ""
            if call_id not in announced:
                raise ValueError(f"tool result {call_id!r} answers no announced call")
            if call_id in answered:
                raise ValueError(f"tool call {call_id!r} is answered twice")
            answered.add(call_id)
        unanswered = sorted(set(announced) - answered)
        if unanswered:
            raise ValueError(f"tool call(s) with no recorded result: {unanswered}")
        return self

    def jsonl_row(self) -> dict[str, Any]:
        row = self.model_dump(mode="json")
        row["messages"] = [message.as_chat_message() for message in self.messages]
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


class LengthSummary(Contract):
    """How one per-row measurement is spread across a dataset.

    Message and character counts, never token counts. No tokenizer is
    configured anywhere in this pipeline, and a character count converted to
    tokens by a rule of thumb is a number that reads as measured and is not.
    These are tokenizer-independent approximations of what a row costs, and a
    caller that needs exact tokens must count them with the tokenizer it will
    actually train under.
    """

    rows: int
    minimum: int = 0
    median: float = 0.0
    p90: float = 0.0
    maximum: int = 0
    total: int = 0


class PartitionComposition(Contract):
    """What one side of an export looks like in aggregate rather than row by row.

    Every gate in the exporter asks whether a single trace is good enough. None
    of them can see that four fifths of what passed came from one lineage, one
    source, or one tool — which is the difference between a dataset and a
    transcript of the busiest week in the corpus.
    """

    rows: int
    rows_by_family: dict[str, int] = {}
    rows_by_source: dict[str, int] = {}
    rows_by_model: dict[str, int] = {}
    """Generating model, as the trace recorded it. A row may count under several."""

    rows_by_scaffold: dict[str, int] = {}
    rows_by_tool: dict[str, int] = {}
    """How many rows call each tool at least once — the share, not the volume."""

    tool_calls_by_tool: dict[str, int] = {}
    """How many calls each tool receives in total, which a row share hides."""

    lineages: int = 0
    repeated_lineages: dict[str, int] = {}
    """Lineages contributing more than one row, and how many each contributes.

    A lineage is a retry chain or a session. Several rows from one is not a
    defect on its own, and it is the shape most likely to be mistaken for
    independent evidence of the same behavior.
    """

    messages_per_row: LengthSummary
    characters_per_row: LengthSummary


class DuplicateGroup(Contract):
    """Traces whose transcripts normalize to the same thing."""

    signature: str
    trace_ids: tuple[str, ...]


class CompositionReport(Contract):
    """A deterministic description of what an export selected, and out of what.

    Reports; never decides. Nothing here changes which rows are exported — that
    is what :class:`SamplingCaps` is for, and the caps exist because this report
    is what tells a caller which limits are worth setting.
    """

    schema_version: int = 1
    offered_traces: int
    """Traces the partition offered, including any the corpus could not supply."""

    offered: PartitionComposition
    """Every offered trace whose transcript could be rebuilt, gated or not."""

    selected: PartitionComposition
    unresolved_reasons: dict[str, int] = {}
    warning_reasons: dict[str, int] = {}
    duplicate_groups: tuple[DuplicateGroup, ...] = ()
    """Normalized-duplicate groups across the offered partition, largest first."""


class SamplingCaps(Contract):
    """Explicit limits on the composition of the selected dataset.

    Deliberately opt-in and deliberately blunt. An unconfigured cap is not a
    default of infinity chosen by this module; it is the caller declining to
    limit that dimension, which keeps an export backward compatible with one
    run before any of this existed. A row a cap removes is quarantined with the
    cap that removed it, never dropped quietly, because a cap is a curation
    decision and the rows it costs are the evidence for revisiting it.
    """

    max_rows_per_family: int | None = None
    max_rows_per_lineage: int | None = None
    max_messages_per_row: int | None = None
    max_characters_per_row: int | None = None

    @model_validator(mode="after")
    def caps_admit_something(self) -> SamplingCaps:
        for field, value in self.model_dump().items():
            if value is not None and value < 1:
                raise ValueError(f"{field} must be at least 1, or unset to mean no limit")
        return self

    @property
    def configured(self) -> bool:
        return any(value is not None for value in self.model_dump().values())


class ExportManifest(Contract):
    schema_version: int = 1
    format: ExportKind
    corpus_id: str
    task_set_id: str
    reviewed_verifier_id: str
    verifier_id: str
    validation_id: str
    human_acceptance_id: str

    verifier_status: str
    """``reviewed`` or ``risk_accepted``. What authorized this export, exactly."""

    accepted_risks: tuple[str, ...] = ()
    """Codes the owner promoted the verifier over, carried to where the data lands."""

    partition: Partition
    partition_trace_count: int
    """How many traces the partition offered, before any gate ran."""

    success_threshold: float
    """Frozen at review. Never re-defaulted by whoever ran the export."""

    max_median_multiplier: float | None = None
    """The step-count bound applied, for exports where trajectory quality matters."""

    caps: SamplingCaps | None = None
    """The sampling caps this export ran under. None means none were configured."""

    selection_policy: str | None = None
    """The order rows were considered in, and how ties in it were broken.

    Recorded because caps make order load-bearing: which rows a cap keeps is
    decided entirely by which ones it reached first.
    """

    composition_schema_version: int | None = None
    """Which version of the composition report this export carries, if any."""

    rows: int
    unresolved: int
    warnings: tuple[str, ...] = ()


class ExportBundle(Contract):
    manifest: ExportManifest
    rows: tuple[EvalCase | SFTExample, ...]
    unresolved: tuple[RejectedTrace, ...] = ()
    composition: CompositionReport | None = None

    @model_validator(mode="after")
    def counts_and_types_match(self) -> ExportBundle:
        if self.manifest.rows != len(self.rows):
            raise ValueError("manifest row count does not match export")
        if self.manifest.unresolved != len(self.unresolved):
            raise ValueError("manifest unresolved count does not match export")
        expected = EvalCase if self.manifest.format is ExportKind.EVAL else SFTExample
        if any(not isinstance(row, expected) for row in self.rows):
            raise ValueError("export contains a row of the wrong format")
        if self.composition is not None:
            if self.manifest.format is not ExportKind.SFT:
                raise ValueError("a composition report describes a training set, not an eval set")
            if self.manifest.composition_schema_version != self.composition.schema_version:
                raise ValueError("manifest and composition report disagree about schema version")
            if self.composition.selected.rows != len(self.rows):
                raise ValueError("composition report does not describe the rows exported")
        return self


def compute_export_id(bundle: ExportBundle) -> str:
    digest = hashlib.sha256(bundle.model_dump_json().encode()).hexdigest()
    return f"export-{bundle.manifest.format.value}-{digest[:16]}"


def save_export(bundle: ExportBundle, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_export_id(bundle),
        kind=f"{bundle.manifest.format.value}_export",
        parent_artifact_id=bundle.manifest.reviewed_verifier_id,
        payload=bundle.model_dump_json().encode(),
        summary={"rows": len(bundle.rows), "unresolved": len(bundle.unresolved)},
    )


def load_export(export_id: str, store: DerivedStore) -> ExportBundle:
    return ExportBundle.model_validate_json(store.read_payload(export_id))


def _atomic_jsonl(path: Path, values: tuple[EvalCase | SFTExample | RejectedTrace, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    lines = (
        json.dumps(value.jsonl_row(), sort_keys=True, separators=(",", ":")) for value in values
    )
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(bundle: ExportBundle, output: Path) -> tuple[Path, Path, Path | None]:
    """Write accepted rows, a sibling quarantine file, and the composition report.

    The first two are written even when empty: an export that quarantined
    everything and one that ran over an empty partition look identical if the
    file is simply absent. The third is written only when the bundle carries a
    report, and its path is returned as None when it does not.
    """
    unresolved = output.with_name(f"{output.stem}.unresolved.jsonl")
    _atomic_jsonl(output, bundle.rows)
    _atomic_jsonl(unresolved, bundle.unresolved)
    if bundle.composition is None:
        return output, unresolved, None

    composition = output.with_name(f"{output.stem}.composition.json")
    temporary = composition.with_suffix(composition.suffix + ".tmp")
    temporary.write_text(
        json.dumps(bundle.composition.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, composition)
    return output, unresolved, composition
