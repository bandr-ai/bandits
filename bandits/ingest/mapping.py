"""A confirmed ingest mapping: the choices one export needs, saved once and applied every time.

Profile → propose → the user confirms → saved → applied on every ingest →
what the mapping does not cover is flagged. A mapping is fully explicit: once
it is applied nothing is discovered from the data, so the corpus is built from
exactly what the user confirmed, and a field that is missing in new data
shows up as the usual unresolved task instead of quietly changing.

Files live at ``<project>/.bandits/mappings/<name>.json``. Confirming records a
digest of the user's choices; a file edited afterwards no longer matches it
and is refused until confirmed again.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator

import bandits
from bandits.traces import Contract, WorkflowDeclaration

NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
CONFIRMATION_FIELDS = frozenset(
    {"confirmed", "confirmed_digest", "confirmed_at", "bandits_version", "derivation_version"}
)
"""Written by ``confirm``; everything else is what the user decided, and digested."""

LABEL_PREFIX = "mapping:"
"""Declared-kind label given to a span the mapping excluded, so its exclusion is
reported apart from evaluators excluded by their own declaration."""


class MappingError(ValueError):
    """A mapping that cannot be used as it stands; the message says how to fix it."""


class Identity(Contract):
    """A kind of span across traces: its declared kind label and its name."""

    kind_label: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.kind_label}|{self.name}"

    @classmethod
    def parse(cls, text: str) -> Identity:
        label, sep, name = text.partition("|")
        if not sep or not label or not name:
            # ValueError, so pydantic reports it as a validation error in a file.
            raise MappingError(f"{text!r} is not KIND_LABEL|NAME (e.g. 'kind=SPAN|run')")
        return cls(kind_label=label, name=name)


class ShapeRef(Contract):
    shape_id: str
    example_trace_id: str
    trace_count: int


class IngestMapping(Contract):
    format_version: Literal[1] = 1
    source: str
    """The ``--source`` it was made for; applying it to another is an error."""

    task_fields: tuple[str, ...] = ()
    delivered_field: str | None = None
    invocation: tuple[Identity, ...] = ()
    """Identities allowed as the invocation; several, so each shape can name its own."""

    candidate_identities: tuple[Identity, ...] = ()
    """The identities discovery offered as options when it was proposed."""

    step_kinds: dict[str, Literal["tool", "step", "exclude"]] = {}
    """``"<kind_label>|<name>"`` → how to treat spans of that identity."""

    shapes: tuple[ShapeRef, ...] = ()
    """The trace shapes the user saw when confirming; others are flagged."""

    @field_validator("step_kinds")
    @classmethod
    def _identity_keys(cls, kinds: dict[str, str]) -> dict[str, str]:
        # A key that cannot name an identity would match nothing and silently
        # do nothing; refuse it instead.
        for key in kinds:
            Identity.parse(key)
        return kinds

    confirmed: bool = False
    confirmed_digest: str | None = None
    confirmed_at: str | None = None
    bandits_version: str | None = None
    derivation_version: int | None = None

    def digest(self) -> str:
        decided = self.model_dump(mode="json", exclude=set(CONFIRMATION_FIELDS))
        canonical = json.dumps(decided, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def unmodified(self) -> bool:
        """Confirmed, and not edited since."""
        return self.confirmed and self.confirmed_digest == self.digest()

    @property
    def invocation_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset((i.kind_label, i.name) for i in self.invocation)

    @property
    def shape_ids(self) -> frozenset[str]:
        return frozenset(shape.shape_id for shape in self.shapes)


def mapping_path(project: Path, name: str) -> Path:
    if not NAME.fullmatch(name):
        raise MappingError(
            f"mapping name {name!r} must be lowercase letters, digits, '-' or '_' "
            "(at most 64, starting with a letter or digit)"
        )
    return project / ".bandits" / "mappings" / f"{name}.json"


def load_mapping(project: Path, name: str) -> IngestMapping:
    path = mapping_path(project, name)
    if not path.exists():
        raise MappingError(f"no mapping {name!r} at {path}")
    try:
        return IngestMapping.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise MappingError(f"mapping {name!r} is not valid, nothing was applied: {exc}") from exc


def save_mapping(project: Path, name: str, mapping: IngestMapping, *, overwrite: bool) -> Path:
    path = mapping_path(project, name)
    if path.exists() and not overwrite:
        raise MappingError(f"mapping {name!r} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mapping.model_dump_json(indent=2) + "\n")
    return path


def confirm(mapping: IngestMapping) -> IngestMapping:
    if not mapping.task_fields:
        listed = ", ".join(f'"{i.key}"' for i in mapping.candidate_identities) or "none found"
        raise MappingError(
            "discovery was ambiguous; rerun propose with --invocation "
            f"(candidate identities: {listed})"
        )
    return mapping.replace(
        confirmed=True,
        confirmed_digest=mapping.digest(),
        confirmed_at=datetime.now(UTC).isoformat(),
        bandits_version=bandits.__version__,
        derivation_version=WorkflowDeclaration().derivation_version,
    )


def applicable(mapping: IngestMapping, name: str, source: str) -> IngestMapping:
    """The mapping, if it may be applied to an ingest of *source*."""
    if mapping.source != source:
        raise MappingError(f"mapping {name!r} was made for --source {mapping.source}, not {source}")
    if not mapping.confirmed:
        raise MappingError(
            f"mapping {name!r} is not confirmed; run `bandits mapping confirm {name}`"
        )
    if not mapping.unmodified:
        raise MappingError(
            f"mapping {name} changed since it was confirmed; run `bandits mapping confirm {name}`"
        )
    return mapping
