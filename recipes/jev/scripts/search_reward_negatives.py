"""Make explicit negative training examples from external search tasks.

Only task IDs assigned to the training split are used. The held-out audit and
hand-authored attack probes are never included here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recipes.jev.scripts.search_reward_data import OPTIONS

UNRELATED = (
    "football scores this week",
    "discount shoes near me",
    "today's weather forecast",
    "popular movie trailers",
    "restaurant menus downtown",
)


def build(step: dict, variant: str, unrelated_query: str = "") -> dict:
    if variant == "repeat":
        previous = [{"tool": step["tool"], "action": step["action"], "observation": step["observation"]}]
        action = f"{step['tool']} {step['action']}"
        observation = step["observation"]
    else:
        previous = step["previous"]
        action = f'browser.search {{"query":{json.dumps(unrelated_query)}}}'
        observation = f"Search results about {unrelated_query}, unrelated to the research question."
    state = (
        f"Research question: {step['question']}\n"
        f"Previous steps: {json.dumps(previous, ensure_ascii=False)}\n"
        f"Current action: {action}\nObserved tool result: {observation}"
    )
    return {
        "id": f"{step['id']}-{variant}",
        "group_id": f"openresearcher-{step['qid']}",
        "split": "train",
        "state": state,
        "question": "What did this observed search action accomplish?",
        "options": OPTIONS,
        "target": "negative",
        "label_source": "programmatic_attack_synthetic_v1",
        "source": "OpenResearcher-derived synthetic negative",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=40)
    args = parser.parse_args()
    by_qid = {}
    for line in args.source.read_text().splitlines():
        step = json.loads(line)
        if step["split"] == "train":
            by_qid.setdefault(step["qid"], step)
    selected = [by_qid[qid] for qid in sorted(by_qid)[: args.tasks]]
    examples = []
    for i, step in enumerate(selected):
        examples.append(build(step, "repeat"))
        choices = UNRELATED[i % len(UNRELATED) :] + UNRELATED[: i % len(UNRELATED)]
        unrelated = next(
            query for query in choices
            if not any(
                topic in step["question"].casefold() and topic in query
                for topic in ("football", "shoe", "weather", "movie", "restaurant")
            )
        )
        examples.append(build(step, "off-topic", unrelated))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples))


if __name__ == "__main__":
    main()
