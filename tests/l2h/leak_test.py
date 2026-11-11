"""Leak scan: catches answers in instruction.md / environment/, spares legitimate evidence."""

from pathlib import Path

from l2h.leak import scan

FINAL = "The invoice total is 1,284.50 euros because the discount applied after tax on line three."


def make(tmp: Path, instruction: str, env_files: dict, tests_files: dict | None = None) -> Path:
    (tmp / "environment/data").mkdir(parents=True)
    (tmp / "instruction.md").write_text(instruction)
    for k, v in env_files.items():
        p = tmp / "environment/data" / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(v)
    for k, v in (tests_files or {"expected.json": "{}"}).items():
        (tmp / "tests").mkdir(exist_ok=True)
        (tmp / "tests" / k).write_text(v)
    return tmp


def kinds(leaks):
    return {(x.where, x.kind) for x in leaks}


def test_clean_task_has_no_leaks(tmp_path):
    d = make(
        tmp_path,
        "Find the total in /app/invoice.txt.",
        {"invoice.txt": "line 3: discount after tax\nsum due"},
    )
    assert scan(d, ["1,284.50 euros total", FINAL]) == []


def test_answer_in_instruction_is_caught(tmp_path):
    d = make(tmp_path, f"Find the total. Hint: {FINAL}", {"a.txt": "x"})
    assert ("instruction.md", "hidden_string") in kinds(scan(d, [FINAL]))


def test_answer_in_environment_file_is_caught(tmp_path):
    d = make(tmp_path, "Find it.", {"notes/log.txt": f"earlier agent said:\n  {FINAL.upper()}"})
    assert ("environment/data/notes/log.txt", "hidden_string") in kinds(scan(d, [FINAL]))


def test_legit_evidence_sharing_words_with_answer_is_not_flagged(tmp_path):
    ev = "invoice total lines: 1) 500.00 2) 700.00 3) 84.50. Discount applied after tax on line three? unclear."
    d = make(tmp_path, "Compute the invoice total in euros.", {"invoice.txt": ev})
    assert scan(d, [FINAL, "1284.50"]) == []


def test_short_answers_only_flag_in_labelled_context(tmp_path):
    d = make(
        tmp_path,
        "How many items are there?",
        {"data.txt": "items: 42 widgets, 7 gadgets\ntotal rows 42"},
    )
    assert scan(d, ["42"]) == []
    d2 = make(tmp_path / "b", "How many items are there?", {"data.txt": "Expected answer: 42"})
    assert kinds(scan(d2, ["42"])) == {("environment/data/data.txt", "labelled_answer")}


def test_answer_like_file_name_and_allowlist(tmp_path):
    d = make(tmp_path, "x", {"expected_output.txt": "whatever"})
    assert ("environment/data/expected_output.txt", "named_like_answer") in kinds(scan(d, []))
    assert scan(d, [], allow=["environment/data/expected_output.txt"]) == []


def test_copy_of_tests_file_in_environment(tmp_path):
    d = make(
        tmp_path,
        "x",
        {"harmless_name.json": '{"expected": 1}'},
        {"expected.json": '{"expected": 1}'},
    )
    assert ("environment/data/harmless_name.json", "copy_of_hidden_file") in kinds(scan(d, []))


def test_long_verbatim_run_is_caught(tmp_path):
    final = " ".join(f"word{i}" for i in range(60))
    d = make(
        tmp_path,
        "x",
        {"quote.txt": "prefix " + " ".join(f"word{i}" for i in range(15, 45)) + " suffix"},
    )
    assert ("environment/data/quote.txt", "verbatim_run") in kinds(scan(d, [final]))


LABELS = ["account_suspension", "refund_requested", "billing_dispute"]
LISTING = "Classify the ticket as one of: account_suspension, refund_requested, billing_dispute."


def test_label_in_allowed_set_listing_is_not_a_leak(tmp_path):
    d = make(tmp_path, LISTING, {"rubric.md": "- billing_dispute: ...\n- account_suspension: ..."})
    assert scan(d, ["account_suspension"], vocab=LABELS) == []
    # without the vocabulary the same label is (still) flagged
    assert ("instruction.md", "hidden_string") in kinds(scan(d, ["account_suspension"]))
    # a bulleted rubric or a sentence mentioning the label alone is judged per occurrence
    d2 = make(tmp_path / "b", "Pick a label.", {"r.md": "- account_suspension: locked\n- other: x"})
    assert scan(d2, ["account_suspension"], vocab=LABELS) == []


def test_real_leaks_still_flagged_with_a_label_vocabulary(tmp_path):
    d = make(
        tmp_path,
        LISTING + "\nThe answer: account_suspension.\n" + FINAL,
        {"gold.txt": "x"},
        {"expected.json": '{"expected": ["account_suspension"]}'},
    )
    (d / "environment/data/copy.json").write_text('{"expected": ["account_suspension"]}')
    k = kinds(scan(d, ["account_suspension", FINAL], vocab=LABELS))
    assert ("instruction.md", "hidden_string") in k  # "The answer: X" and the final response
    assert ("environment/data/copy.json", "copy_of_hidden_file") in k
    assert ("environment/data/gold.txt", "named_like_answer") in k


def test_lone_label_outside_a_listing_is_still_a_leak(tmp_path):
    d = make(tmp_path, "Classify the ticket.", {"t.txt": "tagged account_suspension by agent"})
    assert ("environment/data/t.txt", "hidden_string") in kinds(
        scan(d, ["account_suspension"], vocab=LABELS)
    )


def test_short_label_after_answer_label_is_flagged_even_in_a_listing(tmp_path):
    d = make(tmp_path, "Pick one of: yes, no, maybe.\nExpected answer: maybe", {"a.txt": "x"})
    assert ("instruction.md", "labelled_answer") in kinds(
        scan(d, ["maybe"], vocab=["yes", "no", "maybe"])
    )
