"""Build the whole requested sum job, merging repeated attempts."""

from mint import Answer, BuildResult

SPEC = {
    "unit": "episode",
    "env": "static",
    "job": "sum",
    "check": "numeric",
    "gold": "computed",
    "hidden": [],
    "applies_when": "each synthetic sum attempt",
}
VERSION = 1


def applies(trajectory, findings):
    return True


def group_key(trajectory):
    return trajectory["meta"]["instance"]


def build(group, out):
    first = group[0]
    numbers = first["meta"]["numbers"]
    # Ground truth is recomputed from the supplied file, never copied from an attempt.
    total = sum(int(line) for line in numbers.splitlines())
    return BuildResult(
        task_id=first["meta"]["instance"],
        instruction=first["steps"][0]["message"],
        evidence={"numbers.txt": numbers},
        answer=Answer(expected=str(total), check="numeric"),
        gold_type="computed",
        grading_reason="Sum of the integers in the complete synthetic starting file.",
        oracle_source="Generated answer oracle from the recomputed sum; tests wiring only.",
        wrong_answers=[str(total - 1)],
        alternatives=[f"{total}.0"],
        assumptions=["Synthetic example uses Mint's default Python environment."],
    )
