"""A folder of one-document JSON exports, joined into one JSONL file.

The native readers take one file, and a record's pointer names a line of it.
An export that writes one document per file (a FailproofAI session, a Langfuse
trace) is joined one document per line, so the folder reads as one corpus and
every pointer still resolves in the one archived file. Each line's original
file and its sha256 are listed, so a record can be traced to the file it came
from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def bundle_json_documents(folder: Path, output: Path) -> list[dict[str, Any]]:
    """Write every ``.json`` file under *folder* (sorted by path) to *output*,
    one document per line, and return ``{"line", "path", "sha256"}`` for each.

    A document already on one line is copied byte for byte; a pretty-printed
    one, or one behind a byte-order mark, is re-encoded compactly (same keys,
    order and values). A file that is not one JSON object or array is refused,
    naming the file.
    """
    files = sorted(p for p in folder.rglob("*.json") if p.is_file())
    if not files:
        raise ValueError(f"no .json files under {folder}")
    if any(p.is_file() for p in folder.rglob("*.jsonl")):
        raise ValueError(f"{folder} holds .jsonl files as well; ingest those on their own")
    listing: list[dict[str, Any]] = []
    with output.open("wb") as stream:
        for line, file in enumerate(files, start=1):
            data = file.read_bytes()
            try:
                document = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{file} is not one JSON document: {exc}") from exc
            if not isinstance(document, (dict, list)):
                raise ValueError(f"{file} holds a JSON {type(document).__name__}, not a document")
            body = data.strip()
            if b"\n" in body or b"\r" in body or body.startswith(b"\xef\xbb\xbf"):
                body = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
            stream.write(body + b"\n")
            listing.append(
                {
                    "line": line,
                    "path": str(file.relative_to(folder)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return listing
