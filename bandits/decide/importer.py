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
    __slots__ = ("example", "rejection", "explicit_id")

    def __init__(
        self,
        *,
        example: DecisionExample | None,
        rejection: RejectedDecision | None,
        explicit_id: str | None = None,
    ) -> None:
        self.example = example
        self.rejection = rejection
        self.explicit_id = explicit_id
        """The row's own supplied ``id``, if any -- distinct from an
        auto-derived content-hash record_id, which two unrelated identical
        rows may legitimately share without that being a data-entry mistake
        (see ``_resolve_duplicate_ids``, which only checks this field)."""


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
        empty_options = [
            k
            for k, v in options.items()
            if not str(k).strip() or not isinstance(v, str) or not v.strip()
        ]
        if empty_options:
            reasons.append(f"empty or non-string option id or description: {empty_options}")
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
    return _ParsedRow(example=example, rejection=None, explicit_id=str(raw.get("id")) if raw.get("id") else None)


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


def _resolve_duplicate_ids(
    examples_with_ids: list[tuple[DecisionExample, str | None]], quarantined: list[RejectedDecision]
) -> tuple[list[DecisionExample], list[RejectedDecision]]:
    """Two different rows that happen to supply the same *explicit* ``id``
    would otherwise collide on ``decision_id`` (derived from that id alone)
    and both be silently accepted as if they were one record. Quarantine
    every row sharing a duplicated explicit id instead -- a collision here
    almost always means a data-entry mistake, and accepting either row would
    train or evaluate on it as though its id were unique.

    Only *explicit* ids are checked here: two rows with no supplied id that
    happen to have identical content share an auto-derived record_id, which
    is handled separately by ``_resolve_duplicate_content`` (identical
    content with identical targets is a harmless duplicate; identical
    content with conflicting targets is not)."""
    explicit_ids = [explicit_id for _e, explicit_id in examples_with_ids if explicit_id is not None]
    duplicated = {i for i in explicit_ids if explicit_ids.count(i) > 1}
    if not duplicated:
        return [e for e, _ in examples_with_ids], quarantined

    kept: list[DecisionExample] = []
    newly_quarantined: list[RejectedDecision] = list(quarantined)
    for e, explicit_id in examples_with_ids:
        if explicit_id in duplicated:
            newly_quarantined.append(
                RejectedDecision(
                    trace_id=None,
                    source_file=e.lineage.source_file,
                    source_line=e.lineage.source_line,
                    reasons=(
                        f"id {explicit_id!r} is used by more than one row in this "
                        "import; every explicit id must be unique",
                    ),
                )
            )
        else:
            kept.append(e)
    return kept, newly_quarantined


def _resolve_duplicate_content(
    examples: list[DecisionExample], quarantined: list[RejectedDecision]
) -> tuple[list[DecisionExample], list[RejectedDecision]]:
    """Rows with no explicit id share their ``decision_id`` when their
    content (state+question+options) is identical, since that id is a hash
    of content alone -- it says nothing about the target. Two such rows
    with the *same* target are a harmless duplicate (keep the first, drop
    the redundant copy so downstream decision_id-keyed consumers never see
    two rows under one id). Two such rows with *different* targets, splits
    or group_ids are a real conflict -- contradictory labels, or the same
    decision placed on both sides of a split -- and every row sharing that
    decision_id is quarantined rather than silently picking a winner."""
    by_decision_id: dict[str, list[DecisionExample]] = {}
    for e in examples:
        by_decision_id.setdefault(e.decision_id, []).append(e)

    kept: list[DecisionExample] = []
    newly_quarantined: list[RejectedDecision] = list(quarantined)
    for decision_id, group in by_decision_id.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        targets = {tuple(sorted(e.target.probabilities.items())) for e in group}
        placements = {(e.split, e.group_id) for e in group}
        if len(targets) == 1 and len(placements) == 1:
            # Identical content, target, split and group: a harmless
            # duplicate. Keep the first occurrence only, so the decision_id
            # stays unique among accepted rows.
            kept.append(group[0])
            continue
        # Same content placed in different splits/groups would leak (e.g. a
        # test row silently dropped while its twin stays in train).
        conflict = "targets" if len(targets) > 1 else "split/group_id"
        for e in group:
            newly_quarantined.append(
                RejectedDecision(
                    trace_id=None,
                    source_file=e.lineage.source_file,
                    source_line=e.lineage.source_line,
                    reasons=(
                        f"decision_id {decision_id!r} (same state+question+options, no "
                        f"explicit id) is given conflicting {conflict} by different rows in "
                        "this import",
                    ),
                )
            )
    return kept, newly_quarantined


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
    examples_with_ids: list[tuple[DecisionExample, str | None]] = []
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
            examples_with_ids.append((parsed.example, parsed.explicit_id))

    examples, quarantined = _resolve_duplicate_ids(examples_with_ids, quarantined)
    examples, quarantined = _resolve_duplicate_content(examples, quarantined)
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

    Every item is evaluation-only: it lands in split ``"test"``
    unconditionally, regardless of its own ``split`` field -- even when that
    field happens to spell one of our four split names exactly. JevBench's
    own split names ("public", "hard_holdout", ...) are an external
    benchmark's own partitioning, not ours, and are never reinterpreted as
    one of ours. A JevBench item must never enter train/dev/calibration,
    where it could be fit or tuned against and stop being an honest external
    check.

    Only ``question.type == "choice"`` items are imported -- Noul and Score
    items are quarantined with a reason, since this dataset contract's
    ``primitive`` is choice-only (see ``DecisionExample.primitive``). Each
    item's own source and license are kept per row so redistribution stays
    honest.
    """
    # Every item -- accepted or skipped -- contributes exactly one line, so
    # an accepted row's line number always equals its original position in
    # `items`. A skipped item emits a blank placeholder line rather than
    # being omitted: `_iter_lines` silently skips blank lines without
    # affecting the line count, which is exactly what keeps later accepted
    # rows aligned (omitting the line entirely would shift every later
    # row's recorded source_line backward by the number of prior skips).
    lines: list[str] = []
    quarantined_lines: list[tuple[int, str]] = []
    for index, item in enumerate(items, start=1):
        question = item.get("question")
        if not isinstance(question, dict) or question.get("type") != "choice":
            got = question.get("type") if isinstance(question, dict) else type(question).__name__
            quarantined_lines.append((index, f"not a choice item (question.type={got!r}), skipped"))
            lines.append("")
            continue

        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            quarantined_lines.append((index, "question.criteria is missing or empty"))
            lines.append("")
            continue

        labels = item.get("labels")
        if isinstance(labels, list) and set(labels) == set(criteria):
            options = {option_id: criteria[option_id] for option_id in labels}
        else:
            options = dict(criteria)

        # Every JevBench item is evaluation-only: it always lands in "test",
        # unconditionally, regardless of what its own "split" field says.
        # JevBench's own split names ("public", "hard_holdout", ...) are an
        # external benchmark's partitioning, not ours, and are never
        # reinterpreted as one of ours -- not even when a JevBench item
        # happens to spell one of our four names exactly ("train", "dev",
        # "calibration"). Honoring that coincidence was a real leak: nothing
        # then stopped a JevBench-shaped file from writing `"split": "train"`
        # and bypassing this importer's entire test-only guarantee. There is
        # no override; a caller who genuinely needs one adds it explicitly
        # after this function returns.
        resolved_split = "test"

        provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else {}
        row = {
            "id": item.get("id"),
            "state": item.get("state", ""),
            "question": question.get("instructions", ""),
            "options": options,
            "target": item.get("expected"),
            "group_id": item.get("group") or item.get("family"),
            "split": resolved_split,
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
    """Identifies this dataset as a whole. Unlike a single row's
    ``decision_id``/``split`` (stable under a rename -- see
    ``_row_identity_key``), this id legitimately changes if ``source_file``
    changes: it hashes the full payload, and every example's
    ``lineage.source_file`` is part of that payload. That is intentional --
    the dataset-level id is provenance for "this exact file, imported", not
    a claim that renaming a file produces the same dataset artifact."""
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
