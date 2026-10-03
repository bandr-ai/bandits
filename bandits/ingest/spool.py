"""Private disk grouping for interleaved exports; never opens user-supplied databases."""

from __future__ import annotations

import pickle
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from bandits.traces import Trace

if TYPE_CHECKING:
    from bandits.ingest.otlp_standard import _Decoded


class TraceSpool:
    """Keep source order and first-copy semantics with a bounded SQLite cache."""

    def __init__(self, directory: Path) -> None:
        self.db = sqlite3.connect(directory / "traces.sqlite")
        self.db.execute("PRAGMA cache_size=-1024")
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute(
            "CREATE TABLE spans (trace TEXT, span TEXT, position INTEGER, value BLOB, PRIMARY KEY(trace, span))"
        )
        self.db.execute("CREATE TABLE traces (trace TEXT PRIMARY KEY, position INTEGER)")
        self.db.execute("CREATE TABLE normalized (trace TEXT PRIMARY KEY, value BLOB)")

    def close(self) -> None:
        self.db.close()

    def add(self, decoded: _Decoded) -> bool:
        self.db.execute(
            "INSERT OR IGNORE INTO traces VALUES (?, ?)", (decoded.trace_id, decoded.index)
        )
        result = self.db.execute(
            "INSERT OR IGNORE INTO spans VALUES (?, ?, ?, ?)",
            (
                decoded.trace_id,
                decoded.span_id,
                decoded.index,
                pickle.dumps(decoded, protocol=pickle.HIGHEST_PROTOCOL),
            ),
        )
        return result.rowcount == 1

    def items(self) -> Iterator[tuple[str, dict[str, _Decoded]]]:
        self.db.commit()
        for (trace_id,) in self.db.execute("SELECT trace FROM traces ORDER BY position"):
            yield (
                trace_id,
                {
                    span_id: pickle.loads(value)
                    for span_id, value in self.db.execute(
                        "SELECT span, value FROM spans WHERE trace=? ORDER BY position", (trace_id,)
                    )
                },
            )

    def save_trace(self, trace: Trace) -> None:
        self.db.execute(
            "INSERT INTO normalized VALUES (?, ?)",
            (trace.trace_id, trace.model_dump_json().encode()),
        )

    def sorted_traces(self) -> Iterator[Trace]:
        self.db.commit()
        for (value,) in self.db.execute("SELECT value FROM normalized ORDER BY trace"):
            yield Trace.model_validate_json(value)
