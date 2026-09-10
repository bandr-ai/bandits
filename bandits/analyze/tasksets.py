"""Identify and persist a task set, whichever stage produced it.

Backend-independent on purpose. A task set built by semantic mining and one
built by any future grouping are the same artifact once they exist, and the
code that stores them should not know which stage wrote it.
"""

from __future__ import annotations

import hashlib

from bandits.analyze.models import TaskSet
from bandits.store import DerivedEnvelope, DerivedStore


def compute_task_set_id(task_set: TaskSet) -> str:
    digest = hashlib.sha256(task_set.model_dump_json().encode("utf-8")).hexdigest()
    return f"taskset-{digest[:16]}"


def save_task_set(task_set: TaskSet, store: DerivedStore) -> DerivedEnvelope:
    return store.write(
        compute_task_set_id(task_set),
        kind="taskset",
        parent_artifact_id=task_set.analysis_id,
        payload=task_set.model_dump_json().encode("utf-8"),
        summary={
            "families": len(task_set.families),
            "selected": len(task_set.selected),
            "missing_slots": len(task_set.missing_slots),
        },
    )


def load_task_set(task_set_id: str, store: DerivedStore) -> TaskSet:
    return TaskSet.model_validate_json(store.read_payload(task_set_id))
