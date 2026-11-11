# Parsing

`parse.py` exposes `parse(paths) -> Iterator[Trajectory | ParseError]` (`l2h schema`).

A parsed trajectory is an **ATIF** trajectory (Harbor's Agent Trajectory Interchange Format, ATIF-v1.8;
older versions are accepted) plus a small l2h block in `extra["l2h"]`. Build it with
`Trajectory.from_atif(doc, id=, source_refs=, ended=, meta=, attachments=)`.

- ATIF `doc`: `agent` (`name`, `version`; use `"unknown"` for what the logs do not record, `model_name`
  if known), `steps` (non-empty; `step_id` is numbered for you), `subagent_trajectories` if the logs have
  delegated agents (each a full ATIF trajectory with a `trajectory_id`, referenced from an observation
  result by `subagent_trajectory_ref`).
- A step: `source` (`system`|`user`|`agent`), `message` (text, or a list of `{type: text|image, ...}`
  parts), optional `timestamp`, `reasoning_content`, `model_name`. Agent steps carry
  `tool_calls: [{tool_call_id, function_name, arguments}]`; the results go in the same step's
  `observation.results: [{source_call_id, content}]`, where `source_call_id` is the call's
  `tool_call_id`. Do not invent timestamps, ids or results the logs lack; leave them out.
  Anything ATIF has no field for goes in the step's or document's `extra`.
- l2h block: `id` (unique), `source_refs`, `ended`, `meta` (what you do not understand, unknown values
  `None`), `attachments`.
- `source_refs`: `{"file": <relative to logs>, "offset": <record ordinal>}`. The ordinal is the one in
  ids from `l2h sample` (JSONL: n-th non-blank line; JSON array: index; CSV: row; other text: 0).
  Use `key` instead of `offset` only when there is no ordinal; those files are then excluded from the
  dropped-record count, so prefer offsets.
- `ended`: how the logged run really ended. Look for cut-offs, errors, missing final responses.
- Each trajectory is validated: the basics always, and Harbor's own ATIF model when `harbor` is
  importable. A violation is a parse error grouped by cause (e.g. `not ATIF: agent: Field required`).
- Yield `ParseError(file, error, offset)` for records you cannot parse, with a *stable* message
  (no ids inside it) so errors group by cause.
- `l2h run parse` reports raw records that landed in no trajectory. Zero is not required, but every
  nonzero count needs an explanation in `source_notes.md` or a decision.
- Parsing is cached on parse.py + file sizes/mtimes. `--no-cache` to force.

In checks and task types, `t` has: `t["steps"]`, `t["id"]`, `t["meta"]`, `t["ended"]`, `t["source_refs"]`,
`t["attachments"]` (also `t.meta`, ...); `t.calls()` yields every tool call as
`Call(step, id, name, arguments, result)` with `.output` (the matched result text, `None` if
unanswered); `t.final_message` (text of the last agent step if it made no tool call, else `None`);
`t.subagents` (same accessors); `message_text(step["message"])` for list-form messages. In
`l2h query`/`sample`/`show --field`, `r` has the ATIF keys plus `id`, `meta`, `ended`, `source_refs`
(`r.steps`, `r.meta.model`).

## Logs that already are ATIF

Harbor job outputs (`agent/trajectory.json`) need no conversion; wrap them:

```python
import json, os
from pathlib import Path
from l2h import ParseError, Trajectory


def parse(paths):
    root = Path(
        os.path.commonpath([p.parent for p in paths])
    )  # file refs are relative to the logs folder
    for p in paths:
        if p.name != "trajectory.json":
            continue
        ref = p.relative_to(root).as_posix()
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            yield ParseError(ref, "not json", 0)
            continue
        yield Trajectory.from_atif(
            doc,
            id=doc.get("session_id") or p.parent.parent.name,
            source_refs=[{"file": ref, "offset": 0}],
            ended="unknown",
        )
```

Set `ended` from what the job result says (a clean exit is not success).
