"""Read the synthetic failed revenue job without guessing its success."""

import json

from mint import ParseError, Trajectory


def parse(paths):
    for path in paths:
        if path.suffix != ".jsonl":
            continue
        for offset, line in enumerate(path.read_text().splitlines()):
            try:
                record = json.loads(line)
                yield Trajectory.from_atif(
                    {
                        "agent": {"name": "synthetic-revenue-agent", "version": "1"},
                        "steps": [
                            {"source": "user", "message": record["request"]},
                            {"source": "agent", "message": record["answer"]},
                        ],
                    },
                    id=record["id"],
                    source_refs=[{"file": path.name, "offset": offset}],
                    ended=record["ended"],
                    meta={"starting_file": record["starting_file"]},
                    attachments=[{"name": "sales.csv", "text": record["starting_file"]}],
                )
            except (ValueError, KeyError) as error:
                yield ParseError(path.name, str(error), offset)
