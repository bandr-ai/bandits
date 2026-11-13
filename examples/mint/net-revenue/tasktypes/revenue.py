"""Rebuild the original revenue job and check the signed total."""

import csv
import io
from decimal import Decimal

from mint import Answer, BuildResult

SPEC = {
    "unit": "episode",
    "env": "static",
    "job": "net-revenue",
    "check": "numeric",
    "gold": "computed",
    "hidden": [],
    "applies_when": "the synthetic net revenue job",
}
VERSION = 1


def applies(trajectory, findings):
    return True


def group_key(trajectory):
    return trajectory["id"]


def build(group, out):
    trajectory = group[0]
    starting_file = trajectory["meta"]["starting_file"]
    amounts = [Decimal(row["amount"]) for row in csv.DictReader(io.StringIO(starting_file))]
    expected = sum(amounts)
    wrong = sum(abs(value) for value in amounts)
    return BuildResult(
        task_id="net-revenue",
        instruction=trajectory["steps"][0]["message"],
        evidence={"sales.csv": starting_file},
        answer=Answer(expected=str(expected), check="numeric"),
        gold_type="computed",
        grading_reason="Recompute the signed sum from the complete supplied CSV; refunds are negative.",
        oracle_source="Answer oracle from the recomputed signed total; not an independent agent solve.",
        wrong_answers=[str(wrong)],
        alternatives=[f"{expected}.0"],
        assumptions=["Synthetic example uses Mint's default Python environment."],
    )
