from bandits.ingest.health import _common_string_field


def test_request_field_must_be_text_in_every_run() -> None:
    runs = [{"query": "a", "email": "x"}, {"query": "b", "email": "y"}]
    assert _common_string_field(runs, "input", ("question", "query")) == "input.query"
    assert _common_string_field([*runs, {"email": "z"}], "input", ("query",)) is None
    assert _common_string_field([{"query": ""}], "input", ("query",)) is None
    assert _common_string_field([{"query": 3}], "input", ("query",)) is None


def test_whole_text_input_is_the_request() -> None:
    assert _common_string_field(["ask", "tell"], "input", ("query",)) == "input"
    assert _common_string_field(["ask", {"query": "q"}], "input", ("query",)) is None
    assert _common_string_field([], "input", ("query",)) is None


def test_priority_order_decides_between_fields() -> None:
    runs = [{"text": "t", "query": "q"}]
    assert _common_string_field(runs, "input", ("query", "text")) == "input.query"
