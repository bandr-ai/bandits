from recipes.jev.scripts.search_reward_data import extract, split_for
from recipes.jev.scripts.search_reward_negatives import build


def _message(role, text, *, recipient=None):
    return {"role": role, "recipient": recipient, "content": [{"text": text}]}


def test_extract_uses_observed_browser_turns_and_excludes_gold_answer():
    row = {
        "qid": 42,
        "question": "Which year was the bridge opened?",
        "answer": "SECRET GOLD ANSWER",
        "messages": [
            _message("assistant", "thinking"),
            _message("assistant", '{"query":"bridge opening"}', recipient="browser.search"),
            _message("tool", "Result: a bridge page"),
            _message("assistant", "I know the answer now"),
            _message("assistant", '{"id":"page"}', recipient="browser.open"),
        ],
    }
    steps = extract(row, max_steps=8)
    assert len(steps) == 1
    assert steps[0]["action"] == '{"query":"bridge opening"}'
    assert steps[0]["observation"] == "Result: a bridge page"
    assert steps[0]["split"] == split_for(42)
    assert "SECRET GOLD ANSWER" not in str(steps)


def test_split_keeps_all_steps_of_one_task_together():
    assert split_for(42) == split_for(42)
    assert split_for(42) in {"train", "dev", "calibration", "test"}


def test_synthetic_repeat_is_train_only_and_exposes_the_repeat():
    step = {
        "id": "external-42-3", "qid": 42, "question": "Find the archive year",
        "tool": "browser.search", "action": '{"query":"archive"}',
        "observation": "Same two documents", "previous": [],
    }
    example = build(step, "repeat")
    assert example["split"] == "train"
    assert example["target"] == "negative"
    assert example["state"].count("Same two documents") == 2
