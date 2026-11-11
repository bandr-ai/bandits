# Parsing

`parse.py` exposes `parse(paths) -> Iterator[Trajectory | ParseError]` (`l2h schema`).

- `paths` is every file under the logs folder. Group across files if a session spans several.
- `source_refs`: `{"file": <relative to logs>, "offset": <record ordinal>}`. The ordinal is the one in
  ids from `l2h sample` (JSONL: n-th non-blank line; JSON array: index; CSV: row; other text: 0).
  Use `key` instead of `offset` only when there is no ordinal; those files are then excluded from the
  dropped-record count, so prefer offsets.
- Yield `ParseError(file, error, offset)` for records you cannot parse, with a *stable* message
  (no ids inside it) so errors group by cause.
- Anything you do not understand: keep it under `meta`. Unknown values stay `None`.
- `ended`: how the logged run really ended. Look for cut-offs, errors, missing final responses.
- `l2h run parse` reports raw records that landed in no trajectory. Zero is not required, but every
  nonzero count needs an explanation in `source_notes.md` or a decision.
- Parsing is cached on parse.py + file sizes/mtimes. `--no-cache` to force.
