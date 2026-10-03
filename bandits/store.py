"""Local content-addressed artifact store.

Every ingested corpus is written once under an id derived from its own content
and never mutated afterward. Ingesting the same file twice lands on the same id
and is a no-op; two different corpora landing on the same id — unreachable in
practice, since ids are content hashes, but not trusted blindly — raises instead
of silently overwriting.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import json
import os
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

import bandits
from bandits.jsonarray import iter_array
from bandits.redact import redact_bytes, ruleset_by_name
from bandits.traces import Trace, TraceCorpus


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArtifactEnvelope(Contract):
    schema_version: int = 2
    """Informational. 1: written before the fields below existed (all None)."""

    artifact_id: str
    created_at: str
    source_path: str
    source: str
    trace_count: int
    span_count: int
    issue_count: int
    """Every corpus issue, redactions and notices included; see ``problem_count``."""

    derivation_version: int | None = None
    """The workflow declaration's derivation version; None for non-workflow corpora."""

    bandits_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    """The bandits checkout that wrote this corpus. None when bandits does not run
    from its own git checkout (an installed package), never another repo's commit."""

    problem_count: int | None = None
    """Health warnings plus fatal problems at ingest, as printed then."""

    redaction_count: int | None = None


@functools.cache
def code_version() -> tuple[str, str | None, bool | None]:
    """``(bandits_version, git_commit, git_dirty)`` of the running code.

    The commit is recorded only when the enclosing checkout is bandits' own:
    an installed copy inside some project's virtualenv must not report that
    project's commit. Dirty means tracked changes under the package or
    ``pyproject.toml``; untracked work elsewhere does not mark every corpus.
    """
    package = Path(bandits.__file__).resolve().parent

    def git(*args: str, cwd: Path = package) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    try:
        declared = tomllib.loads((Path(top) / "pyproject.toml").read_text())  # type: ignore[arg-type]
    except (TypeError, OSError, tomllib.TOMLDecodeError):
        declared = {}
    if top is None or declared.get("project", {}).get("name") != "bandits":
        return bandits.__version__, None, None
    commit = git("rev-parse", "HEAD")
    # Pathspecs resolve against the working directory, so run from the top.
    status = git("status", "--porcelain", "-uno", "--", "bandits", "pyproject.toml", cwd=Path(top))
    return bandits.__version__, commit, (None if status is None else bool(status))


_WRAPPERS = ("observations", "spans", "data", "runs")
_NESTED = (*_WRAPPERS, "children", "child_runs")
_ID_KEYS = ("id", "run_id", "span_id")


_LINE_LIMIT = 64 << 20
"""Matches ``bandits.ingest.native``: a longer first line starting with ``[``
is a single-line array, archived as a stream."""


def resolve_record(data: bytes, pointer: dict | str) -> dict:
    """The native record a span's ``bandits.source.record`` pointer names in *data*.

    *data* is the archived (redacted) source file. ``line`` is the physical
    1-based line, ``index`` a position in a JSON array (on that line, or the
    whole file), ``document`` the whole file. The record found there may wrap
    the observation (a trace with ``observations``, a ``spans``/``runs``
    list), so it is searched for the object whose ``id``, ``run_id``,
    ``span_id`` or ``context.span_id`` is the pointer's ``observation_id``.

    Lines are split with ``bytes.splitlines``, as the archive's whole-file
    redaction path does, so a file using a bare ``\r`` as a line break would
    shift line numbers; such exports have not been seen.
    """
    if isinstance(pointer, str):
        pointer = json.loads(pointer)
    if "line" in pointer:
        record = json.loads(data.splitlines()[pointer["line"] - 1])
        if "index" in pointer:
            record = record[pointer["index"]]
    else:
        record = json.loads(data)
        if "index" in pointer:
            record = record[pointer["index"]]
    wanted = str(pointer["observation_id"])
    stack: list[object] = [record]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(reversed(node))
            continue
        if not isinstance(node, dict):
            continue
        wrapper = any(isinstance(node.get(key), list) for key in _WRAPPERS)
        context = node.get("context") if isinstance(node.get("context"), dict) else {}
        ids = [node.get(key) for key in _ID_KEYS] + [context.get("span_id")]
        if not wrapper and wanted in {str(i) for i in ids if i is not None}:
            return node
        stack.extend(node[key] for key in reversed(_NESTED) if isinstance(node.get(key), list))
    raise LookupError(f"no observation {wanted} at {pointer} in the source archive")


class ArtifactConflict(ValueError):
    """An existing artifact at this id has different content than what was just written."""


def compute_artifact_id(corpus: TraceCorpus) -> str:
    return _id_of(corpus.model_dump_json().encode("utf-8"))


def _id_of(corpus_bytes: bytes) -> str:
    return f"corpus-{hashlib.sha256(corpus_bytes).hexdigest()[:16]}"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


class ArtifactStore:
    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._project_dir = Path(project_dir)
        self._artifacts_dir = self._project_dir / "artifacts"

    def _dir(self, artifact_id: str) -> Path:
        return self._artifacts_dir / artifact_id

    def write(
        self,
        corpus: TraceCorpus,
        *,
        source_path: str,
        problem_count: int | None = None,
        report: dict | None = None,
    ) -> ArtifactEnvelope:
        """Store *corpus* under its content id.

        ``report`` (what the read saw, as plain JSON data) is saved as
        ``report.json`` beside it, outside the id: the same corpus read twice
        is the same artifact, and an existing report is never replaced.
        """
        # Serialized once: a large corpus is the biggest object an ingest holds,
        # and its id is the digest of exactly these bytes.
        corpus_bytes = corpus.model_dump_json().encode("utf-8")
        artifact_id = _id_of(corpus_bytes)
        artifact_dir = self._dir(artifact_id)

        if artifact_dir.exists():
            existing_bytes = (artifact_dir / "corpus.json").read_bytes()
            if existing_bytes != corpus_bytes:
                raise ArtifactConflict(
                    f"artifact {artifact_id} already exists with different content"
                )
            self._archive_source(artifact_dir, source_path, corpus)
            self._write_report(artifact_dir, report)
            return self.read_envelope(artifact_id)

        artifact_dir.mkdir(parents=True)
        version, commit, dirty = code_version()
        envelope = ArtifactEnvelope(
            artifact_id=artifact_id,
            created_at=datetime.now(UTC).isoformat(),
            source_path=source_path,
            source=corpus.source,
            trace_count=len(corpus.traces),
            span_count=sum(len(t.spans) for t in corpus.traces),
            issue_count=len(corpus.issues),
            derivation_version=(
                corpus.workflow.derivation_version if corpus.workflow is not None else None
            ),
            bandits_version=version,
            git_commit=commit,
            git_dirty=dirty,
            problem_count=problem_count,
            redaction_count=sum(issue.kind == "redaction" for issue in corpus.issues),
        )
        _atomic_write(artifact_dir / "corpus.json", corpus_bytes)
        _atomic_write(artifact_dir / "envelope.json", envelope.model_dump_json().encode("utf-8"))
        self._archive_source(artifact_dir, source_path, corpus)
        self._write_report(artifact_dir, report)
        return envelope

    @staticmethod
    def _write_report(artifact_dir: Path, report: dict | None) -> None:
        path = artifact_dir / "report.json"
        if report is not None and not path.exists():
            _atomic_write(path, json.dumps(report, ensure_ascii=False, indent=1).encode("utf-8"))

    def read_report(self, artifact_id: str) -> dict | None:
        path = self._dir(artifact_id) / "report.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _archive_source(self, artifact_dir: Path, source_path: str, corpus: TraceCorpus) -> None:
        """Store redacted source bytes next to the normalized corpus.

        A digest alone cannot recover fields a reader did not map. The archive
        keeps them available without reintroducing secrets removed at ingest.
        """
        manifest_path = artifact_dir / "source-manifest.json"
        if manifest_path.exists():
            return
        source = Path(source_path)
        if not source.exists():
            # Some derived/test corpora have a logical source name, not a file.
            # CLI ingest passes an existing path and receives an archive.
            return
        if source.is_file():
            files = [source]
        elif corpus.source == "trail":
            files = sorted(source.glob("*.json"))
        elif corpus.source == "claude-code":
            files = sorted(source.rglob("*.jsonl"))
        else:
            files = sorted(
                p for p in source.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl")
            )
        ruleset = ruleset_by_name(corpus.redaction_ruleset or "default-v1")
        archive = artifact_dir / "source"
        archive.mkdir(exist_ok=True)
        manifest: list[dict[str, str | int]] = []
        for index, file in enumerate(files):
            name = f"{index:06d}.json"
            archive_path = archive / name
            source_hash = hashlib.sha256()
            redacted_hash = hashlib.sha256()
            redacted_bytes = 0
            with file.open("rb") as stream:
                first = stream.readline(_LINE_LIMIT)
                while first and not first.strip():
                    first = stream.readline(_LINE_LIMIT)
                # A first line cut at the limit is a single-line array: stream it.
                cut = len(first) == _LINE_LIMIT and not first.endswith(b"\n")
                array = first.lstrip().startswith(b"[")
                if cut and not array:
                    first += stream.readline()
                    cut = False
                if array and not cut:
                    # A complete first line is JSONL (or a one-line array) to
                    # the native reader too; only a multi-line array streams.
                    try:
                        array = not isinstance(json.loads(first), list)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass
                jsonl = (
                    not cut
                    and file.suffix == ".jsonl"
                    and (
                        (first.lstrip().startswith(b"{") and first.rstrip().endswith(b"}"))
                        or (first.lstrip().startswith(b"[") and first.rstrip().endswith(b"]"))
                        or not first.lstrip().startswith((b"{", b"["))
                    )
                )
                stream.seek(0)
                if jsonl:
                    with archive_path.with_suffix(".tmp").open("wb") as output:
                        for line in stream:
                            source_hash.update(line)
                            data = redact_bytes(line, str(file), ruleset).data
                            output.write(data)
                            redacted_hash.update(data)
                            redacted_bytes += len(data)
                    os.replace(archive_path.with_suffix(".tmp"), archive_path)
                elif array:
                    # Elements redacted one at a time, everything between them
                    # copied verbatim, so the array's index pointers still resolve.
                    with archive_path.with_suffix(".tmp").open("wb") as output:
                        for gap, element, line in iter_array(stream, source_hash):
                            data = gap
                            if element is not None:
                                data += redact_bytes(
                                    element, str(file), ruleset, first_line=line
                                ).data
                            output.write(data)
                            redacted_hash.update(data)
                            redacted_bytes += len(data)
                    os.replace(archive_path.with_suffix(".tmp"), archive_path)
                else:
                    original = stream.read()
                    source_hash.update(original)
                    redacted = redact_bytes(original, str(file), ruleset)
                    _atomic_write(archive_path, redacted.data)
                    redacted_hash.update(redacted.data)
                    redacted_bytes = len(redacted.data)
            if (
                source.is_file()
                and corpus.traces
                and any(trace.source_digest != source_hash.hexdigest() for trace in corpus.traces)
            ):
                raise ValueError(f"source file changed after ingest: {file}")
            manifest.append(
                {
                    "path": str(file.relative_to(source)) if source.is_dir() else file.name,
                    "archive": name,
                    "source_sha256": source_hash.hexdigest(),
                    "redacted_sha256": redacted_hash.hexdigest(),
                    "redacted_bytes": redacted_bytes,
                }
            )
        _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False).encode("utf-8"))

    def source_manifest(self, artifact_id: str) -> list[dict[str, str | int]]:
        return json.loads((self._dir(artifact_id) / "source-manifest.json").read_text())

    def read_source(self, artifact_id: str, archive_name: str) -> bytes:
        names = {str(item["archive"]) for item in self.source_manifest(artifact_id)}
        if archive_name not in names:
            raise ValueError(f"unknown archived source {archive_name!r}")
        return (self._dir(artifact_id) / "source" / archive_name).read_bytes()

    def read_native_record(self, artifact_id: str, pointer: dict | str) -> dict:
        """The archived native record a span's ``bandits.source.record`` points to."""
        manifest = self.source_manifest(artifact_id)
        if len(manifest) != 1:
            raise ValueError("a native record pointer needs a single-file source archive")
        return resolve_record(self.read_source(artifact_id, str(manifest[0]["archive"])), pointer)

    def read(self, artifact_id: str) -> TraceCorpus:
        return TraceCorpus.model_validate_json(
            (self._dir(artifact_id) / "corpus.json").read_bytes()
        )

    def read_envelope(self, artifact_id: str) -> ArtifactEnvelope:
        return ArtifactEnvelope.model_validate_json(
            (self._dir(artifact_id) / "envelope.json").read_bytes()
        )

    def list(self) -> list[ArtifactEnvelope]:
        if not self._artifacts_dir.exists():
            return []
        envelopes = [
            self.read_envelope(entry.name)
            for entry in self._artifacts_dir.iterdir()
            if entry.is_dir()
        ]
        return sorted(envelopes, key=lambda e: e.created_at, reverse=True)


class StreamingWrite:
    """Stage canonical corpus bytes; publish only after the caller's health gate.

    Temporary files live beside the store, on the same filesystem as artifacts.
    No complete corpus or serialized corpus is retained in memory.
    """

    def __init__(self, store: ArtifactStore, source: str):
        self.store = store
        store._project_dir.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix=".bandits-ingest-", dir=store._project_dir.parent
        )
        self.directory = Path(self.temp.name)
        self.output = (self.directory / "corpus.json").open("wb")
        self.digest = hashlib.sha256()
        self.trace_count = self.span_count = 0
        self.source_digest: str | None = None
        self.mixed_source_digests = False
        prefix = TraceCorpus(source=source, traces=()).model_dump_json().split('"traces":[]', 1)[0]
        self._write((prefix + '"traces":[').encode())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.output.close()
        self.temp.cleanup()

    def _write(self, data: bytes) -> None:
        self.output.write(data)
        self.digest.update(data)

    def add(self, trace: Trace) -> None:
        if self.trace_count:
            self._write(b",")
        self._write(trace.model_dump_json().encode())
        self.trace_count += 1
        self.span_count += len(trace.spans)
        if self.source_digest is None:
            self.source_digest = trace.source_digest
        elif trace.source_digest != self.source_digest:
            self.mixed_source_digests = True

    def finish(self, footer: TraceCorpus) -> None:
        if footer.traces:
            raise ValueError("stream footer must not contain traces")
        suffix = footer.model_dump_json().split('"traces":[]', 1)[1]
        self._write(("]" + suffix).encode())
        self.output.close()
        self.footer = footer
        self.artifact_id = "corpus-" + self.digest.hexdigest()[:16]

    def commit(
        self, *, source_path: str, problem_count: int, report: dict | None
    ) -> ArtifactEnvelope:
        corpus = self.footer
        target = self.store._dir(self.artifact_id)
        if target.exists():
            with (
                (target / "corpus.json").open("rb") as existing,
                (self.directory / "corpus.json").open("rb") as staged,
            ):
                while True:
                    left, right = existing.read(1024 * 1024), staged.read(1024 * 1024)
                    if left != right:
                        raise ArtifactConflict(
                            f"artifact {self.artifact_id} already exists with different content"
                        )
                    if not left:
                        break
            return self.store.read_envelope(self.artifact_id)
        self.store._archive_source(self.directory, source_path, corpus)
        manifest = self.directory / "source-manifest.json"
        if self.source_digest is not None and manifest.exists() and Path(source_path).is_file():
            if json.loads(manifest.read_text())[0]["source_sha256"] != self.source_digest:
                raise ValueError(f"source file changed after ingest: {source_path}")
        version, commit, dirty = code_version()
        envelope = ArtifactEnvelope(
            artifact_id=self.artifact_id,
            created_at=datetime.now(UTC).isoformat(),
            source_path=source_path,
            source=corpus.source,
            trace_count=self.trace_count,
            span_count=self.span_count,
            issue_count=len(corpus.issues),
            derivation_version=corpus.workflow.derivation_version if corpus.workflow else None,
            bandits_version=version,
            git_commit=commit,
            git_dirty=dirty,
            problem_count=problem_count,
            redaction_count=sum(i.kind == "redaction" for i in corpus.issues),
        )
        _atomic_write(self.directory / "envelope.json", envelope.model_dump_json().encode())
        self.store._write_report(self.directory, report)
        self.store._artifacts_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.directory, target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return self.commit(source_path=source_path, problem_count=problem_count, report=report)
        return envelope


class DerivedEnvelope(Contract):
    """Header for an artifact derived from another one.

    Kept deliberately ignorant of what it wraps: the store persists opaque JSON so
    that adding an analysis, verifier, or export type never reaches back into it.
    """

    schema_version: int = 1
    artifact_id: str
    kind: str
    """What produced this, e.g. 'analysis'."""

    parent_artifact_id: str
    """The artifact this was derived from. Never empty — lineage is the point."""

    created_at: str
    summary: dict[str, int] = {}


class DerivedStore:
    """Derived artifacts, stored beside corpora rather than among them.

    A separate directory so that :meth:`ArtifactStore.list` keeps returning
    corpora only, and so an analysis can never be mistaken for source evidence.
    """

    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._derived_dir = Path(project_dir) / "derived"

    def _dir(self, artifact_id: str) -> Path:
        return self._derived_dir / artifact_id

    def write(
        self,
        artifact_id: str,
        *,
        kind: str,
        parent_artifact_id: str,
        payload: bytes,
        summary: dict[str, int] | None = None,
    ) -> DerivedEnvelope:
        artifact_dir = self._dir(artifact_id)
        if artifact_dir.exists():
            existing = (artifact_dir / "payload.json").read_bytes()
            if existing != payload:
                raise ArtifactConflict(
                    f"derived artifact {artifact_id} already exists with different content"
                )
            return self.read_envelope(artifact_id)

        artifact_dir.mkdir(parents=True)
        envelope = DerivedEnvelope(
            artifact_id=artifact_id,
            kind=kind,
            parent_artifact_id=parent_artifact_id,
            created_at=datetime.now(UTC).isoformat(),
            summary=summary or {},
        )
        _atomic_write(artifact_dir / "payload.json", payload)
        _atomic_write(artifact_dir / "envelope.json", envelope.model_dump_json().encode("utf-8"))
        return envelope

    def read_payload(self, artifact_id: str) -> bytes:
        return (self._dir(artifact_id) / "payload.json").read_bytes()

    def read_envelope(self, artifact_id: str) -> DerivedEnvelope:
        return DerivedEnvelope.model_validate_json(
            (self._dir(artifact_id) / "envelope.json").read_bytes()
        )

    def list(self, *, kind: str | None = None) -> list[DerivedEnvelope]:
        if not self._derived_dir.exists():
            return []
        envelopes = [
            self.read_envelope(entry.name)
            for entry in self._derived_dir.iterdir()
            if entry.is_dir()
        ]
        if kind is not None:
            envelopes = [e for e in envelopes if e.kind == kind]
        return sorted(envelopes, key=lambda e: e.created_at, reverse=True)
