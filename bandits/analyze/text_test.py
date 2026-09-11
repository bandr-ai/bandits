"""request_signature: same named value, same action words, not just one or the other."""

from __future__ import annotations

from bandits.analyze.text import request_signature


def test_a_paraphrase_naming_the_same_value_has_the_same_signature() -> None:
    a = request_signature("refund order 7741")
    b = request_signature("please issue a refund for order 7741")

    assert a == b


def test_the_same_value_with_a_different_action_has_a_different_signature() -> None:
    refund = request_signature("refund order 7741")
    cancel = request_signature("cancel order 7741")

    assert refund != cancel, "sharing a value is not sharing a request"


def test_different_values_never_share_a_signature() -> None:
    a = request_signature("send the note referencing owe_list.csv")
    b = request_signature("send the note referencing debt_list.csv")

    assert a != b


def test_a_quoted_value_is_the_named_parameter_not_a_content_word() -> None:
    a = request_signature('send the note "pay before Friday"')
    b = request_signature('please send the note "pay before Friday"')

    assert a == b
    parameters, _ = a
    assert parameters == ("pay before friday",)


def test_a_request_with_no_named_value_still_has_a_signature() -> None:
    parameters, content_words = request_signature("list my playlists")

    assert parameters == ()
    assert content_words == frozenset({"list", "playlists"})
