"""Import a user's own labeled decisions into the ``DecisionDataset`` contract.

An importer only translates and validates into that one contract; it never
grows a parallel schema. Every row becomes a ``DecisionExample`` with its own
question and options (the judge compiler's one-shared-schema check does not
apply here -- see ``DecisionSchema``), or is quarantined with the line number
and reason a trainer needs to fix its source file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from typing import Any

from bandits.decide.dataset import (
    DecisionDataset,
    DecisionDatasetCounts,
    DecisionExample,
    DecisionLineage,
    DecisionSplit,
    DecisionTarget,
    RejectedDecision,
)
from bandits.decide.prompt import MAX_OPTIONS
from bandits.store import DerivedEnvelope, DerivedStore

_VALID_SPLITS: frozenset[str] = frozenset({"train", "dev", "calibration", "test"})


def _quarantine(
    *, source_file: str, source_line: int, reasons: tuple[str, ...]
) -> RejectedDecision:
    return RejectedDecision(source_file=source_file, source_line=source_line, reasons=reasons)


def _row_identity_key(question: str, options: dict[str, str], state: str) -> str:
    """A digest of a row's own content -- state, question and options --
    never its file path or line number. Used both to derive a stable
    ``decision_id``/``record_id`` and, for an ungrouped row, to pick its
    split: identical content always lands on the same id and the same
    split, so a rename or a reordered file can never reshuffle
    train/dev/calibration/test or leak the locked test split."""
    payload = json.dumps({"question": question, "options": options, "state": state}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _deterministic_split(key: str) -> DecisionSplit:
    """Assign a split from a stable hash of a content key (a row's own id,
    its group id, or a hash of its question+options -- never a file path or
    line number). Roughly 70/10/10/10 train/dev/calibration/test."""
    digest = hashlib.sha256(key.encode()).digest()
    bucket = digest[0] / 256.0
    if bucket < 0.70:
        return "train"
    if bucket < 0.80:
        return "dev"
    if bucket < 0.90:
        return "calibration"
    return "test"


def _parse_target(raw: Any, options: dict[str, str], reasons: list[str]) -> DecisionTarget | None:
    if isinstance(raw, str):
        if raw not in options:
            reasons.append(f"target option {raw!r} is not among this row's options")
            return None
        return DecisionTarget(kind="hard", probabilities={o: (1.0 if o == raw else 0.0) for o in options})
    if isinstance(raw, dict):
        missing = set(options) - set(raw)
        extra = set(raw) - set(options)
        if missing or extra:
            reasons.append(f"target probabilities do not cover exactly this row's options (missing={sorted(missing)}, extra={sorted(extra)})")
            return None
        try:
            return DecisionTarget(kind="soft", probabilities={k: float(v) for k, v in raw.items()})
        except (ValueError, TypeError) as exc:
            reasons.append(f"invalid target probabilities: {exc}")
            return None
    reasons.append(f"target must be an option id (hard) or a probability map (soft), got {type(raw).__name__}")
    return None


class _ParsedRow:
    __slots__ = ("example", "rejection")

    def __init__(self, *, example: DecisionExample | None, rejection: RejectedDecision | None) -> None:
        self.example = example
        self.rejection = rejection


def _parse_row(
    raw: dict[str, Any], *, source_file: str, line_number: int, dataset_source: str | None
) -> _ParsedRow:
    reasons: list[str] = []

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        reasons.append("missing or empty 'question'")

    state = raw.get("state", "")
    if not isinstance(state, str):
        reasons.append("'state' must be a string")
        state = ""

    options = raw.get("options")
    if not isinstance(options, dict) or not options:
        reasons.append("'options' must be a non-empty object of option id -> description")
        options = {}
    else:
        empty_options = [k for k, v in options.items() if not str(k).strip() or not str(v).strip()]
        if empty_options:
            reasons.append(f"empty option id or description: {empty_options}")
        if len(options) < 2:
            reasons.append("needs at least 2 options")
        if len(options) > MAX_OPTIONS:
            reasons.append(f"at most {MAX_OPTIONS} options allowed, got {len(options)}")

    target: DecisionTarget | None = None
    if options and not reasons:
        raw_target = raw.get("target")
        if raw_target is None:
            reasons.append("missing 'target'")
        else:
            target = _parse_target(raw_target, options, reasons)

    group_id = raw.get("group_id")
    split = raw.get("split")
    if split is not None and split not in _VALID_SPLITS:
        reasons.append(f"invalid split {split!r}, must be one of {sorted(_VALID_SPLITS)}")

    if reasons:
        return _ParsedRow(
            example=None,
            rejection=_quarantine(source_file=source_file, source_line=line_number, reasons=tuple(reasons)),
        )

    identity_key = str(raw.get("id")) if raw.get("id") else _row_identity_key(question, options, state)
    record_id = str(raw.get("id") or identity_key[:16])
    split_key = str(group_id) if group_id else identity_key
    digest = hashlib.sha256(f"decision:{identity_key}".encode()).hexdigest()
    example = DecisionExample(
        decision_id=f"decision-import-{digest[:16]}",
        family_id=str(group_id) if group_id else record_id,
        group_id=str(group_id) if group_id else None,
        state=state,
        question=question,
        primitive="choice",
        options={str(k): str(v) for k, v in options.items()},
        target=target,
        label_source=str(raw.get("label_source", "user_import")),
        split=split or _deterministic_split(split_key),
        lineage=DecisionLineage(
            source_kind="jsonl_import",
            record_id=record_id,
            source_file=source_file,
            source_line=line_number,
        ),
        source=str(raw.get("source")) if raw.get("source") is not None else dataset_source,
        license=str(raw.get("license")) if raw.get("license") is not None else None,
    )
    return _ParsedRow(example=example, rejection=None)


def _iter_lines(path_text: str, source_file: str) -> Iterator[tuple[int, dict[str, Any] | None, RejectedDecision | None]]:
    for line_number, line in enumerate(path_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            yield line_number, None, _quarantine(
                source_file=source_file, source_line=line_number, reasons=(f"invalid JSON: {exc}",)
            )
            continue
        if not isinstance(row, dict):
            yield line_number, None, _quarantine(
                source_file=source_file,
                source_line=line_number,
                reasons=(f"each line must be a JSON object, got {type(row).__name__}",),
            )
            continue
        yield line_number, row, None


def _resolve_group_conflicts(
    examples: list[DecisionExample], quarantined: list[RejectedDecision]
) -> tuple[list[DecisionExample], list[RejectedDecision]]:
    """Rows sharing a ``group_id`` must land in one split (mixing explicit
    and hashed splits, or two different explicit splits, would otherwise
    make ``DecisionDataset`` reject the whole import). Resolve per group
    before construction: if every row in a group already agrees, keep them;
    otherwise quarantine every row of that group with a reason, rather than
    silently picking a winner or failing the entire import."""
    split_by_group: dict[str, set[str]] = {}
    for e in examples:
        if e.group_id is None:
            continue
        split_by_group.setdefault(e.group_id, set()).add(e.split)

    conflicted_groups = {group for group, splits in split_by_group.items() if len(splits) > 1}
    if not conflicted_groups:
        return examples, quarantined

    kept: list[DecisionExample] = []
    newly_quarantined: list[RejectedDecision] = list(quarantined)
    for e in examples:
        if e.group_id in conflicted_groups:
            newly_quarantined.append(
                RejectedDecision(
                    trace_id=None,
                    source_file=e.lineage.source_file,
                    source_line=e.lineage.source_line,
                    reasons=(
                        f"group {e.group_id!r} has rows with conflicting splits "
                        f"{sorted(split_by_group[e.group_id])}; all rows sharing a "
                        "group_id must agree on one split",
                    ),
                )
            )
        else:
            kept.append(e)
    return kept, newly_quarantined


def import_jsonl(
    path_text: str,
    *,
    source_file: str,
    dataset_source: str | None = None,
    producer: str = "jsonl_import",
) -> DecisionDataset:
    """Parse a JSONL file of labeled decisions into a ``DecisionDataset``.

    Each line is a JSON object with (at minimum) ``question`` and ``options``
    (id -> description) and a ``target`` (either an option id for a hard
    label, or an id -> probability map for a soft label). Optional per-row
    fields: ``state``, ``split``, ``group_id``, ``source``, ``license``,
    ``label_source``, ``id``.

    A row missing an explicit ``split`` gets one deterministically from a
    hash of its own content: ``group_id`` when set, otherwise a hash of its
    question+options (never the source file path or line number, so a
    rename or a reordered file can never reshuffle splits or leak the locked
    test split -- see ``_deterministic_split``). Rows that share a
    ``group_id`` but disagree on their split (whether explicit or hashed)
    are quarantined as a group, never silently resolved by picking one.

    Malformed lines are quarantined with their line number and reason, never
    coerced or silently dropped.
    """
    examples: list[DecisionExample] = []
    quarantined: list[RejectedDecision] = []

    for line_number, row, rejection in _iter_lines(path_text, source_file):
        if rejection is not None:
            quarantined.append(rejection)
            continue
        assert row is not None
        parsed = _parse_row(row, source_file=source_file, line_number=line_number, dataset_source=dataset_source)
        if parsed.rejection is not None:
            quarantined.append(parsed.rejection)
        else:
            assert parsed.example is not None
            examples.append(parsed.example)

    examples, quarantined = _resolve_group_conflicts(examples, quarantined)

    by_split: dict[str, int] = {}
    for e in examples:
        by_split[e.split] = by_split.get(e.split, 0) + 1

    return DecisionDataset(
        producer=producer,
        source_artifact_ids=(),
        decision_schema=None,
        examples=tuple(examples),
        quarantined=tuple(quarantined),
        counts=DecisionDatasetCounts(
            examples=len(examples),
            train=by_split.get("train", 0),
            dev=by_split.get("dev", 0),
            calibration=by_split.get("calibration", 0),
            test=by_split.get("test", 0),
            quarantined=len(quarantined),
        ),
    )


def import_jevbench(
    items: Iterable[dict[str, Any]],
    *,
    source_file: str,
    dataset_source: str = "jevbench",
    default_license: str | None = None,
) -> DecisionDataset:
    """Import real JevBench items through the same JSONL-row contract.

    A JevBench item looks like::

        {"id", "state",
         "question": {"type": "choice", "instructions": "...", "criteria": {option_id: description}},
         "labels": [option_ids in display order], "expected": option_id,
         "provenance": {"license", "source", ...}, "family", "group", "split"}

    Mapping: ``question`` text = ``question.instructions``; ``options`` =
    ``question.criteria``, presented in ``labels`` order when given;
    ``target`` = ``expected`` (a hard label -- ``labels`` is the option
    *order*, not a target, and is never used as one); ``license``/``source``
    come from ``provenance``; ``group_id`` = ``group``, falling back to
    ``family`` when ``group`` is null (JevBench sets ``group`` on only some
    items).

    Only ``question.type == "choice"`` items are imported -- Noul and Score
    items are quarantined with a reason, since this dataset contract's
    ``primitive`` is choice-only (see ``DecisionExample.primitive``). Each
    item's own source and license are kept per row so redistribution stays
    honest.
    """
    lines = []
    quarantined_lines: list[tuple[int, str]] = []
    for index, item in enumerate(items, start=1):
        question = item.get("question")
        if not isinstance(question, dict) or question.get("type") != "choice":
            got = question.get("type") if isinstance(question, dict) else type(question).__name__
            quarantined_lines.append((index, f"not a choice item (question.type={got!r}), skipped"))
            continue

        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            quarantined_lines.append((index, "question.criteria is missing or empty"))
            continue

        labels = item.get("labels")
        if isinstance(labels, list) and set(labels) == set(criteria):
            options = {option_id: criteria[option_id] for option_id in labels}
        else:
            options = dict(criteria)

        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        row = {
            "id": item.get("id"),
            "state": item.get("state", ""),
            "question": question.get("instructions", ""),
            "options": options,
            "target": item.get("expected"),
            "group_id": item.get("group") or item.get("family"),
            "split": item.get("split") if item.get("split") in _VALID_SPLITS else None,
            "source": provenance.get("source", dataset_source),
            "license": provenance.get("license", default_license),
            "label_source": "jevbench",
        }
        lines.append(json.dumps(row))

    dataset = import_jsonl(
        "\n".join(lines), source_file=source_file, dataset_source=dataset_source, producer="jevbench_import"
    )
    if not quarantined_lines:
        return dataset

    extra_quarantined = tuple(dataset.quarantined) + tuple(
        RejectedDecision(source_file=source_file, source_line=line, reasons=(reason,))
        for line, reason in quarantined_lines
    )
    by_split: dict[str, int] = {}
    for e in dataset.examples:
        by_split[e.split] = by_split.get(e.split, 0) + 1
    return DecisionDataset(
        producer=dataset.producer,
        source_artifact_ids=dataset.source_artifact_ids,
        decision_schema=dataset.decision_schema,
        examples=dataset.examples,
        quarantined=extra_quarantined,
        counts=DecisionDatasetCounts(
            examples=len(dataset.examples),
            train=by_split.get("train", 0),
            dev=by_split.get("dev", 0),
            calibration=by_split.get("calibration", 0),
            test=by_split.get("test", 0),
            quarantined=len(extra_quarantined),
        ),
    )


def compute_import_dataset_id(dataset: DecisionDataset) -> str:
    digest = hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()
    return f"decision-dataset-{digest[:16]}"


def save_imported_dataset(dataset: DecisionDataset, store: DerivedStore, *, source_file: str) -> DerivedEnvelope:
    return store.write(
        compute_import_dataset_id(dataset),
        kind="decision_dataset",
        parent_artifact_id=source_file,
        payload=dataset.model_dump_json().encode(),
        summary={
            "examples": dataset.counts.examples,
            "train": dataset.counts.train,
            "dev": dataset.counts.dev,
            "calibration": dataset.counts.calibration,
            "test": dataset.counts.test,
            "quarantined": dataset.counts.quarantined,
        },
    )
