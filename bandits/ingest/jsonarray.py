"""Read a top-level JSON array one element at a time, keeping its exact bytes.

A whole-file ``json.loads`` of a multi-GB export holds the bytes, the text and
every parsed object at once. This holds one element plus a read buffer. Each
element comes with the raw text before it (``[``, commas, whitespace), so a
caller can rewrite elements and copy everything else verbatim.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterator
from typing import IO, Any

_CHUNK = 1 << 20
_WHITESPACE = " \t\n\r"
_DECODER = json.JSONDecoder()


def iter_array(
    stream: IO[bytes], digest: Any = None, *, chunk: int = _CHUNK
) -> Iterator[tuple[bytes, bytes | None, int]]:
    """Yield ``(gap, element, line)`` for each element of the array in *stream*.

    ``gap`` is the raw bytes since the previous element and ``line`` the
    1-based line the element starts on. A last ``(tail, None, line)`` carries
    the closing bracket and what follows it, so the yielded bytes concatenate
    to the whole file. *digest*, if given, is updated with every byte read.
    Raises ``ValueError`` unless *stream* holds exactly one JSON array.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    text, pos, eof = "", 0, False

    def more(size: int = chunk) -> None:
        nonlocal text, eof
        data = stream.read(size)
        if digest is not None:
            digest.update(data)
        eof = not data
        text += decoder.decode(data, final=eof)

    def skip_whitespace() -> None:
        nonlocal pos
        while True:
            while pos < len(text) and text[pos] in _WHITESPACE:
                pos += 1
            if pos < len(text) or eof:
                return
            more()

    more()
    skip_whitespace()
    if not text.startswith("[", pos):
        raise ValueError("expected a JSON array")
    gap_start, line = 0, 1
    pos += 1
    skip_whitespace()
    if text.startswith("]", pos):
        pos += 1
    else:
        while True:
            skip_whitespace()
            size = chunk
            while True:
                try:
                    _, end = _DECODER.raw_decode(text, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                else:
                    # A value ending exactly at the buffer's end may be cut short.
                    if end < len(text) or eof:
                        break
                more(size)
                size *= 2
            gap = text[gap_start:pos]
            line += gap.count("\n")
            element = text[pos:end]
            yield gap.encode(), element.encode(), line
            line += element.count("\n")
            pos = gap_start = end
            # Drop consumed text once it is most of the buffer: amortized, not
            # a copy per element.
            if pos > chunk and pos * 2 > len(text):
                text, pos, gap_start = text[pos:], 0, 0
            skip_whitespace()
            if text.startswith(",", pos):
                pos += 1
                continue
            if text.startswith("]", pos):
                pos += 1
                break
            raise ValueError("expected ',' or ']' after a JSON array element")
    while True:
        if text[pos:].strip(_WHITESPACE):
            raise ValueError("extra data after the JSON array")
        if eof:
            break
        more()
    yield text[gap_start:].encode(), None, line
