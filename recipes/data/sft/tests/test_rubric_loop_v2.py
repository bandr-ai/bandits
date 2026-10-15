"""v2 rubric: two domain slots filled from one induction answer, plus the lint; offline, no model calls."""
import pytest

from bandits_sft.rubric_judge import build_prompt
from bandits_sft.rubric_loop import SLOTS_V2, assemble_v2, generic_v2, lint, parse_slots, prompt_section

ANSWER = "\n".join(f"<{k}>\n{k.upper()} TEXT\n</{k}>" for k in SLOTS_V2)


def test_parse_slots_reads_both():
    assert parse_slots(ANSWER) == {k: f"{k.upper()} TEXT" for k in SLOTS_V2}


def test_parse_slots_rejects_a_missing_block():
    with pytest.raises(ValueError, match="answer_rules"):
        parse_slots("<consistency_map>x</consistency_map>")


def test_assemble_fills_each_slot_once_and_drops_comments():
    r = assemble_v2(parse_slots(ANSWER))
    for k in SLOTS_V2:
        assert r.count(f"{k.upper()} TEXT") == 1 and "{" + k + "}" not in r
    assert "<!--" not in r and r.count("## Output") == 1 and "{examples}" not in r


def test_v2_judge_keeps_its_output_and_uses_a_neutral_tag():
    system, user = build_prompt({"system": "s", "messages": []}, assemble_v2(parse_slots(ANSWER)))
    assert system.count("## Output") == 1 and '"critique"' in system
    assert "<agent_system_prompt>" in user


def test_v1_rubric_is_unchanged():
    system, user = build_prompt({"system": "s", "messages": []}, "# rubric without output")
    assert system.count("## Output") == 1 and '"reason"' in system and "<agent_system_prompt>" in user


def test_prompts_have_their_placeholders_and_no_examples_slot():
    p = prompt_section("INDUCE", "induce_rubric_prompt_v2.md")
    for k in ("generic_judge", "system_prompts", "tool_definitions", "seeds"):
        assert "{" + k + "}" in p
    g = generic_v2()
    assert "{consistency_map}" in g and "{answer_rules}" in g and "{examples}" not in g


def test_lint_flags_copied_rows_but_not_quoted_rules():
    prompt = "conclude render stuck when console errors and no pending network and a spinner keeps persisting on screen"
    slots = {"consistency_map": "m",
             "answer_rules": "| situation | answer |\n|---|---|\n"
                             "| console errors and no pending network and a spinner keeps persisting | stuck |\n"
                             "| blank | null |\n"
                             '- "conclude render stuck when console errors and no pending network and a spinner" (prompt:x)\n'
                             "unverified row\nUnknowns: none"}
    r = lint(slots, [prompt])
    assert len(r["copied_table_rows"]) == 1 and "spinner" in r["copied_table_rows"][0]
    assert r["unverified_mentions"] == 1 and r["unknowns_sections"] == 1 and not r["over_length"]


def test_v2_texts_name_no_domain():
    """The shipped generic texts must not assume tools, a domain or an answer format."""
    text = (generic_v2() + prompt_section("INDUCE", "induce_rubric_prompt_v2.md")
            + prompt_section("REVISE", "induce_rubric_prompt_v2.md")).lower()
    for word in ("refund", "airline", "flight", "patient", "invoice", "cancel_order"):
        assert word not in text, word
