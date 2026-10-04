"""An append-only record of every model call the pipeline makes.

A stage records its conclusions. What produced them — the request, the reply,
what it cost, how long it took, which attempts failed first — was discarded at
each call site independently, so a verdict could be read and never checked, a
bill could not be attributed, and two runs that disagreed offered nothing to
compare.

The record lives at the transport boundary rather than in each stage, because
there is one place every provider request passes through and five places that
would otherwise each invent their own shape. A stage names what it was doing;
the transport knows everything else.

Headers are dropped rather than filtered: an allowlist that grows a new field
on a provider's next release leaks it, and nothing here needs them. That is a
narrow guarantee and worth stating narrowly — it is not the same as "no secret
can reach this file". Prompts, responses, request fields and exception messages
are all recorded verbatim, and any of them can carry a credential that some
other layer put there.

What is guaranteed: no request header, and therefore no ``Authorization``, is
written. What is not: the prompt is kept, so a corpus carrying personal data
produces a ledger carrying it too. The file belongs wherever the corpus does,
and there is no redacted export path yet.

Identity: every row carries a unique ``event_id``. A ``stage_id`` is shared by
the rows inside one stage, and a ``logical_call_id`` ties a call's retries to
the call. One row, one id, always.
"""

from __future__ import annotations

import contextvars
import fcntl
import functools
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENV_PATH = "BANDITS_LEDGER"
"""Explicit destination override. Model-driven CLI commands default to their project ledger."""

_local = threading.local()

_context_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "bandits_ledger_context", default=None
)
"""Stage/call context. A context variable rather than thread-local state, so a
worker started with ``contextvars.copy_context()`` — DSPy's batched subcalls —
records under the chunk and iteration that issued it instead of under nothing."""

_failure_lock = threading.Lock()
_failure: list[LedgerWriteError] = []


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _context() -> dict[str, Any]:
    return _context_var.get() or {}


def raise_if_failed() -> None:
    """Re-raise a required write that failed anywhere in this run.

    A write failure inside a sandbox tool or a worker thread is caught by the
    code that called it — DSPy turns a tool exception into REPL output — so the
    failure is remembered here and raised at the next boundary that cannot
    swallow it: the next model call, REPL step or invocation end. Cleared when
    a recorded command starts and when it ends.
    """
    with _failure_lock:
        if _failure:
            raise LedgerWriteError(str(_failure[0]))


def clear_failure() -> None:
    with _failure_lock:
        _failure.clear()


_ENV_STRICT = "BANDITS_LEDGER_STRICT"
"""Set alongside the path to make a lost record fail the run.

Model-driven CLI commands require recording and fail if a record cannot be
written. Other callers may enable strict recording explicitly.
"""


class LedgerWriteError(RuntimeError):
    """A record could not be written while the ledger was required."""


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[None]:
    """Name what the calls inside this block were for.

    Nested rather than passed down: the transport cannot know it is serving a
    family audit, and threading a label through five signatures to tell it
    would put a logging concern in every function between.
    """
    previous = _context()
    merged = {
        **previous,
        **{key: value for key, value in fields.items() if value is not None},
        "stage": name,
        # Identifies the block, not the rows inside it. Every row still gets an
        # `event_id` of its own at write time; naming this one `event_id` made
        # a dozen calls share an identity and left no way to reference one.
        "stage_id": uuid.uuid4().hex[:16],
        "parent_stage_id": previous.get("stage_id"),
    }
    token = _context_var.set(merged)
    try:
        yield
    finally:
        _context_var.reset(token)


def enabled() -> bool:
    return bool(_path())


def _path() -> str | None:
    return getattr(_local, "ledger_path", None) or os.environ.get(_ENV_PATH)


def project_recording(function):
    """Record model-driven CLI runs in their project, without an opt-in flag."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        previous = getattr(_local, "ledger_path", None)
        required = getattr(_local, "ledger_required", False)
        project = Path(kwargs.get("project", Path.cwd()))
        _local.ledger_path = os.environ.get(_ENV_PATH) or str(project / ".bandits" / "ledger.jsonl")
        _local.ledger_required = True
        old_path, old_strict = os.environ.get(_ENV_PATH), os.environ.get(_ENV_STRICT)
        old_fsync = os.environ.get(_ENV_FSYNC)
        # Sub-LM batches use worker threads; the destination must reach them too.
        os.environ[_ENV_PATH] = _local.ledger_path
        os.environ[_ENV_STRICT] = "1"
        # A recorded command's rows are made durable one by one, unless the
        # caller set a policy of their own.
        os.environ.setdefault(_ENV_FSYNC, "1")
        clear_failure()
        try:
            record({"event_type": "recording_started", "command": function.__name__})
            return function(*args, **kwargs)
        finally:
            # A failure belongs to the run that hit it, never to the next one.
            clear_failure()
            _local.ledger_path = previous
            _local.ledger_required = required
            for key, value in (
                (_ENV_PATH, old_path),
                (_ENV_STRICT, old_strict),
                (_ENV_FSYNC, old_fsync),
            ):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return wrapped


def strict() -> bool:
    return (
        getattr(_local, "ledger_required", False) or bool(os.environ.get(_ENV_STRICT))
    ) and enabled()


LEDGER_SCHEMA = "bandits-ledger/2"
"""2: large strings stored once as content-addressed blobs; rows carry ``pid``/``seq``."""

_ENV_BLOB_MIN = "BANDITS_LEDGER_BLOB_MIN"
"""Strings at least this many characters long are stored as blobs (default 8192;
0 keeps every value inline)."""

_ENV_FSYNC = "BANDITS_LEDGER_FSYNC"
"""Set to fsync the ledger after every row. Blobs are always fsynced before the
row that references them, so a reference never points at a file that a crash
could lose while the row survived."""

_DEFAULT_BLOB_MIN = 8192
_write_lock = threading.Lock()
_seq_lock = threading.Lock()
_seq = [0]


class LedgerCorrupt(ValueError):
    """A ledger row or blob reference could not be read back as written."""


def _blob_min() -> int:
    raw = os.environ.get(_ENV_BLOB_MIN)
    try:
        return _DEFAULT_BLOB_MIN if raw is None else max(0, int(raw))
    except ValueError:
        return _DEFAULT_BLOB_MIN


def blob_dir(ledger_path: str | Path) -> Path:
    """Where a ledger's blobs live: beside it, named after it."""
    path = Path(ledger_path)
    return path.with_name(path.name + ".blobs")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_dir(directory: Path) -> None:
    """Create ``directory`` and any missing parents, each durably.

    A new directory exists after power loss only once its own parent's entry
    has been synced, so every directory created here has its parent fsynced.
    """
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    for created in reversed(missing):
        created.mkdir(exist_ok=True)
        _fsync_dir(created.parent)


_BLOB_ENCODING = ("utf-8", "surrogatepass")
"""Lone surrogates (malformed text a source or model can produce) are stored
byte-exact rather than failing the write; reading reverses the same encoding."""


def _store_blob(directory: Path, text: str) -> str:
    """Write ``text`` once under its sha256, durably, and return the digest."""
    data = text.encode(*_BLOB_ENCODING)
    digest = hashlib.sha256(data).hexdigest()
    target = directory / digest
    if target.exists():
        return digest
    _ensure_dir(directory)
    tmp = directory / f".{digest}.{os.getpid()}.{threading.get_ident()}.tmp"
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    _fsync_dir(directory)
    return digest


def _externalize(value: Any, directory: Path, minimum: int) -> Any:
    """Replace long strings, at any depth, with verified blob references."""
    if minimum <= 0:
        return value
    if isinstance(value, str):
        if len(value) < minimum:
            return value
        return {"$blob": f"sha256:{_store_blob(directory, value)}", "length": len(value)}
    if isinstance(value, dict):
        return {key: _externalize(item, directory, minimum) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_externalize(item, directory, minimum) for item in value]
    return value


def _next_seq() -> int:
    with _seq_lock:
        _seq[0] += 1
        return _seq[0]


def record(event: dict[str, Any]) -> None:
    """Append one event, stamped with an identity unique to this row.

    Best effort by default: a run that died because its logging could not write
    would trade the thing being recorded for the record of it. Under
    ``BANDITS_LEDGER_STRICT`` the trade goes the other way and the failure is
    raised, because a formal experiment missing records is not a partial
    result — it is one that cannot be checked. Every failure counts: a value
    that cannot be serialized fails the same way as a full disk.

    Large strings are stored once as blobs (see ``BANDITS_LEDGER_BLOB_MIN``) and
    restored by :func:`read_events`; nothing is truncated. ``seq`` is assigned
    under the write lock, so within one process it follows file order exactly;
    rows from several processes are kept whole by ``flock`` and told apart by
    ``pid``.
    """
    path = _path()
    if not path:
        return
    try:
        plain = json.loads(json.dumps(event, default=str))
        target = Path(path)
        _ensure_dir(target.parent)
        payload = _externalize(plain, blob_dir(target), _blob_min())
        row = {
            "recorded_at": _now(),
            "schema": LEDGER_SCHEMA,
            "pid": os.getpid(),
            **_context(),
            **payload,
            # Last, so the row's identity is the ledger's to assign. A caller
            # passing `event_id` would otherwise silently collide two rows,
            # which is the failure this field exists to make impossible.
            "event_id": uuid.uuid4().hex,
        }
        with _write_lock:
            created = not target.exists()
            row["seq"] = _next_seq()
            line = json.dumps(row, default=str) + "\n"
            with target.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                    if os.environ.get(_ENV_FSYNC):
                        os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if created:
                # The new file's directory entry, so the ledger itself survives.
                _fsync_dir(target.parent)
    except Exception as exc:  # noqa: BLE001 - any lost row is a recording failure
        if strict():
            error = LedgerWriteError(
                f"the ledger could not be written to {path!r}: "
                f"{type(exc).__name__}: {exc}. "
                "The run is recording incompletely and BANDITS_LEDGER_STRICT is set."
            )
            with _failure_lock:
                if not _failure:
                    _failure.append(error)
            raise error from exc
        # Loud even when not fatal. A silent drop is how a run finishes looking
        # complete while missing the records it was started to produce.
        print(  # noqa: T201 - deliberate: stderr must carry this past a rich console
            f"bandits: WARNING lost a ledger event, {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _restore(value: Any, directory: Path) -> Any:
    if isinstance(value, dict):
        reference = value.get("$blob")
        if isinstance(reference, str) and set(value) == {"$blob", "length"}:
            digest = reference.removeprefix("sha256:")
            try:
                data = (directory / digest).read_bytes()
            except OSError as exc:
                raise LedgerCorrupt(f"blob {digest} is missing: {exc}") from exc
            if hashlib.sha256(data).hexdigest() != digest:
                raise LedgerCorrupt(f"blob {digest} does not match its digest")
            text = data.decode(*_BLOB_ENCODING)
            if len(text) != value["length"]:
                raise LedgerCorrupt(f"blob {digest} has the wrong length")
            return text
        return {key: _restore(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item, directory) for item in value]
    return value


def read_events(path: str | Path, *, resolve: bool = True) -> list[dict[str, Any]]:
    """Every row of a ledger, with blob references verified and restored.

    A line that is not JSON — a write cut short by a crash — raises
    :class:`LedgerCorrupt` naming its line, unless it is the last line, which
    is reported as ``{"event_type": "truncated_row", ...}`` because an
    interrupted append is the one corruption a crash is expected to leave.
    """
    target = Path(path)
    lines = target.read_text(encoding="utf-8").splitlines()
    directory = blob_dir(target)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            if number == len(lines):
                rows.append({"event_type": "truncated_row", "line": number, "text": line})
                continue
            raise LedgerCorrupt(f"line {number} of {target} is not JSON: {exc}") from exc
        rows.append(_restore(row, directory) if resolve else row)
    return rows


@contextmanager
def model_call(*, provider: str, model: str, request: Any) -> Iterator[dict[str, Any]]:
    """Record one *logical* provider request, whether or not it succeeds.

    Logical, not physical, and the distinction is load-bearing: this wraps
    ``request_with_retry``, so a 429 followed by a success is one event here
    and two requests on the wire. The physical attempts appear as their own
    ``retry`` events carrying this call's ``logical_call_id``; the number of
    requests actually made is that count plus one, never this event alone.

    Instrumenting each ``urlopen`` instead would make the count exact and would
    put the record below the layer that knows what the call was for. It stays
    here until something needs per-attempt bodies.

    A failed call is the one most worth having: it still consumed a rate-limit
    budget and may have cost tokens, and its absence is what made a rerun that
    behaved differently impossible to explain.
    """
    slot: dict[str, Any] = {}
    started = time.monotonic()
    started_at = _now()
    status = "success"
    error: dict[str, Any] | None = None
    call_id = uuid.uuid4().hex[:16]
    previous = _context()
    # Published so the retries firing inside this block can name the call they
    # belong to without it being threaded through the transport's signature.
    token = _context_var.set({**previous, "logical_call_id": call_id})
    try:
        yield slot
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised untouched
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        code = _http_status(exc)
        if code is not None:
            error["http_status"] = code
        raise
    finally:
        response = slot.get("response")
        # Restoring the context gets its own `finally`. Under strict mode
        # `record` raises, and a raise here would leave this call's
        # `logical_call_id` on the context for whatever ran next to inherit,
        # attributing later retries to a call that had already finished.
        try:
            record(
                {
                    "event_type": "model_call",
                    "logical_call_id": call_id,
                    "provider": provider,
                    "model": model,
                    "request": request,
                    "response": response,
                    "usage": (
                        (response or {}).get("usage") if isinstance(response, dict) else None
                    ),
                    "status": status,
                    "error": error,
                    "started_at": started_at,
                    "finished_at": _now(),
                    "duration_seconds": round(time.monotonic() - started, 4),
                }
            )
        finally:
            _context_var.reset(token)


def record_attempt(*, attempt: int, error: Exception, delay: float) -> None:
    """One failed physical request inside a retried logical call.

    Kept separate from the call it belongs to: a call that succeeded on its
    third attempt looks identical to one that succeeded immediately, except in
    latency and in the rate-limit budget it spent getting there. The
    ``logical_call_id`` inherited from the enclosing ``model_call`` is what
    reattaches them.
    """
    event: dict[str, Any] = {
        "event_type": "retry",
        "attempt": attempt,
        "attempt_id": uuid.uuid4().hex[:16],
        "error_type": type(error).__name__,
        "message": str(error),
        "computed_delay_seconds": round(delay, 3),
    }
    from bandits.transport import response_header

    code = _http_status(error)
    if code is not None:
        event["http_status"] = code
    # The only header worth keeping: it is the server stating a number the
    # backoff then obeys, so a delay that looks wrong can be traced to it.
    asked = response_header(error, "Retry-After")
    if asked:
        event["retry_after"] = asked
    record(event)


def _http_status(error: BaseException) -> int | None:
    from bandits.transport import http_status

    return http_status(error)
