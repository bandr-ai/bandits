from __future__ import annotations

import json

from bandits.decide.importer import (
    import_jevbench,
    import_jsonl,
    save_imported_dataset,
)
from bandits.store import DerivedStore


def _row(**kwargs) -> str:
    base = {"question": "which fruit", "options": {"a": "apple", "b": "banana"}, "target": "a"}
    base.update(kwargs)
    return json.dumps(base)


def test_hard_and_soft_targets_round_trip() -> None:
    lines = [
        _row(),
        _row(
            question="legal holding",
            options={"0": "x", "1": "y", "2": "z"},
            target={"0": 0.7, "1": 0.2, "2": 0.1},
        ),
    ]
    dataset = import_jsonl("\n".join(lines), source_file="demo.jsonl")

    assert dataset.counts.examples == 2
    assert dataset.counts.quarantined == 0
    assert dataset.decision_schema is None
    hard, soft = dataset.examples
    assert hard.target.kind == "hard"
    assert hard.target.probabilities["a"] == 1.0
    assert soft.target.kind == "soft"
    assert abs(sum(soft.target.probabilities.values()) - 1.0) < 1e-9


def test_invalid_json_line_is_quarantined_with_line_number() -> None:
    lines = [_row(), "not json at all"]
    dataset = import_jsonl("\n".join(lines), source_file="demo.jsonl")

    assert dataset.counts.examples == 1
    assert dataset.counts.quarantined == 1
    assert dataset.quarantined[0].source_line == 2
    assert dataset.quarantined[0].source_file == "demo.jsonl"
    assert "invalid JSON" in dataset.quarantined[0].reasons[0]


def test_missing_target_option_is_quarantined() -> None:
    bad = json.dumps({"question": "q", "options": {"a": "x", "b": "y"}, "target": "c"})
    dataset = import_jsonl(bad, source_file="demo.jsonl")

    assert dataset.counts.examples == 0
    assert dataset.counts.quarantined == 1
    assert "not among this row's options" in dataset.quarantined[0].reasons[0]


def test_empty_option_is_quarantined() -> None:
    bad = json.dumps({"question": "q", "options": {"a": "", "b": "y"}, "target": "b"})
    dataset = import_jsonl(bad, source_file="demo.jsonl")

    assert dataset.counts.quarantined == 1
    assert "empty option" in dataset.quarantined[0].reasons[0]


def test_more_than_26_options_is_quarantined() -> None:
    options = {chr(97 + i): f"opt {i}" for i in range(27)}
    bad = json.dumps({"question": "q", "options": options, "target": "a"})
    dataset = import_jsonl(bad, source_file="demo.jsonl")

    assert dataset.counts.quarantined == 1
    assert "at most 26" in dataset.quarantined[0].reasons[0]


def test_probabilities_not_summing_to_one_is_quarantined() -> None:
    bad = json.dumps(
        {
            "question": "q",
            "options": {"a": "x", "b": "y"},
            "target": {"a": 0.5, "b": 0.2},
        }
    )
    dataset = import_jsonl(bad, source_file="demo.jsonl")

    assert dataset.counts.quarantined == 1


def test_reimporting_identical_content_yields_the_same_dataset_id() -> None:
    from bandits.decide.importer import compute_import_dataset_id

    text = "\n".join([_row(), _row(question="q2")])
    first = import_jsonl(text, source_file="demo.jsonl")
    second = import_jsonl(text, source_file="demo.jsonl")

    assert compute_import_dataset_id(first) == compute_import_dataset_id(second)


def test_split_assignment_never_separates_rows_sharing_a_group_id() -> None:
    lines = [_row(group_id="g1") for _ in range(20)]
    dataset = import_jsonl("\n".join(lines), source_file="demo.jsonl")

    splits = {e.split for e in dataset.examples}
    assert len(splits) == 1


def test_explicit_split_is_respected() -> None:
    dataset = import_jsonl(_row(split="test"), source_file="demo.jsonl")
    assert dataset.examples[0].split == "test"


def test_invalid_explicit_split_is_quarantined() -> None:
    dataset = import_jsonl(_row(split="bogus"), source_file="demo.jsonl")
    assert dataset.counts.quarantined == 1
    assert "invalid split" in dataset.quarantined[0].reasons[0]


def test_source_and_license_are_kept_per_row() -> None:
    dataset = import_jsonl(
        _row(source="zeroshot/twitter-financial-news-topic", license="MIT"),
        source_file="demo.jsonl",
    )
    row = dataset.examples[0]
    assert row.source == "zeroshot/twitter-financial-news-topic"
    assert row.license == "MIT"


def test_dataset_level_source_fills_in_when_row_omits_it() -> None:
    dataset = import_jsonl(_row(), source_file="demo.jsonl", dataset_source="my-dataset")
    assert dataset.examples[0].source == "my-dataset"


def _real_jevbench_choice_item(**overrides) -> dict:
    """Shaped exactly like a real row in jevbench/datasets/public/easy.jsonl."""
    item = {
        "expected": "track_order",
        "family": "intent",
        "group": None,
        "id": "easy-intent-00",
        "labels": ["track_order", "cancel_order", "change_address", "report_damage", "billing_question"],
        "provenance": {
            "exclude_reason": None,
            "label_basis": "Answer named or stated explicitly in the text",
            "license": "MIT",
            "source": "JevBench v1.1 easy tier, original authored item",
        },
        "question": {
            "criteria": {
                "billing_question": "Asks about a charge, invoice or payment",
                "cancel_order": "Wants to cancel an order",
                "change_address": "Wants to change the delivery address",
                "report_damage": "Received an item that is broken or damaged",
                "track_order": "Wants to know where an order is or when it arrives",
            },
            "instructions": "Which intent does the user's message express?",
            "type": "choice",
        },
        "split": "public",
        "state": "Where is my package? I ordered it last week and it still hasn't arrived.",
    }
    item.update(overrides)
    return item


def test_jevbench_importer_maps_real_item_shape() -> None:
    dataset = import_jevbench([_real_jevbench_choice_item()], source_file="jevbench.jsonl")

    assert dataset.counts.examples == 1
    assert dataset.counts.quarantined == 0
    row = dataset.examples[0]
    assert row.question == "Which intent does the user's message express?"
    assert row.options["track_order"] == "Wants to know where an order is or when it arrives"
    assert set(row.options) == {
        "track_order", "cancel_order", "change_address", "report_damage", "billing_question",
    }
    assert row.target.kind == "hard"
    assert row.target.probabilities["track_order"] == 1.0
    assert row.source == "JevBench v1.1 easy tier, original authored item"
    assert row.license == "MIT"
    assert row.label_source == "jevbench"


def test_jevbench_importer_uses_group_falling_back_to_family() -> None:
    with_group = import_jevbench(
        [_real_jevbench_choice_item(group="case-group-1")], source_file="jevbench.jsonl"
    )
    assert with_group.examples[0].group_id == "case-group-1"

    without_group = import_jevbench(
        [_real_jevbench_choice_item(group=None, family="intent")], source_file="jevbench.jsonl"
    )
    assert without_group.examples[0].group_id == "intent"


def test_jevbench_importer_quarantines_non_choice_items() -> None:
    noul_item = _real_jevbench_choice_item(
        id="hard-noul-00",
        question={"type": "noul", "instructions": "is this spam?", "criteria": {"true": "spam", "false": "not spam"}},
        expected="false",
        labels=["true", "false"],
    )
    dataset = import_jevbench([noul_item, _real_jevbench_choice_item()], source_file="jevbench.jsonl")

    assert dataset.counts.examples == 1
    assert dataset.counts.quarantined == 1
    assert "not a choice item" in dataset.quarantined[0].reasons[0]
    assert "noul" in dataset.quarantined[0].reasons[0]


def test_jevbench_importer_never_uses_labels_as_target() -> None:
    """labels is the option DISPLAY ORDER, never a target -- expected is the
    only field that names the correct answer."""
    item = _real_jevbench_choice_item(expected="cancel_order")
    dataset = import_jevbench([item], source_file="jevbench.jsonl")

    row = dataset.examples[0]
    assert row.target.probabilities["cancel_order"] == 1.0
    assert row.target.probabilities["track_order"] == 0.0


def test_renaming_the_source_file_does_not_reshuffle_splits_or_ids() -> None:
    """Split and decision_id must come from row content, never the file path
    or line number -- a rename or reordered file must reproduce identical
    output. Uses rows without explicit split/group so they exercise the
    hashed path, and reorders lines too."""
    lines = [_row(question=f"q{i}") for i in range(10)]
    text = "\n".join(lines)
    reordered_text = "\n".join(reversed(lines))

    original = import_jsonl(text, source_file="demo.jsonl")
    renamed = import_jsonl(text, source_file="renamed/elsewhere.jsonl")
    reordered = import_jsonl(reordered_text, source_file="demo.jsonl")

    original_by_question = {e.question: (e.split, e.decision_id) for e in original.examples}
    renamed_by_question = {e.question: (e.split, e.decision_id) for e in renamed.examples}
    reordered_by_question = {e.question: (e.split, e.decision_id) for e in reordered.examples}

    assert original_by_question == renamed_by_question
    assert original_by_question == reordered_by_question


def test_group_split_conflict_quarantines_the_whole_group_not_the_dataset() -> None:
    """Rows sharing a group_id but disagreeing on split (one explicit, one
    hashed to something else, or two different explicit splits) must not
    crash the import -- every row in that group is quarantined instead."""
    lines = [
        _row(question="q1", group_id="g1", split="train"),
        _row(question="q2", group_id="g1", split="test"),
        _row(question="q3", group_id="g2"),  # untouched, unrelated group
    ]
    dataset = import_jsonl("\n".join(lines), source_file="demo.jsonl")

    assert dataset.counts.examples == 1
    assert dataset.examples[0].question == "q3"
    assert dataset.counts.quarantined == 2
    assert all("conflicting splits" in q.reasons[0] for q in dataset.quarantined)


def test_import_round_trips_through_store(tmp_path) -> None:
    dataset = import_jsonl(_row(), source_file="demo.jsonl")
    store = DerivedStore(tmp_path)
    envelope = save_imported_dataset(dataset, store, source_file="demo.jsonl")

    loaded = store.read_payload(envelope.artifact_id)
    from bandits.decide.dataset import DecisionDataset

    round_tripped = DecisionDataset.model_validate_json(loaded)
    assert round_tripped == dataset
