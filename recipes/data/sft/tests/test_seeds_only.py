"""Seeds-only: report helpers read any agent's traces without domain config (tool-using JSON agents and free-text ones)."""
import json

from simia_plus.analyzer import answer_labels, decision_cells, outcomes


def _trace(final: str, tool: str | None = None) -> dict:
    msgs = [{"role": "user", "content": "q"}]
    if tool:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "probe", "arguments": {}}]},
                 {"role": "tool", "name": "probe", "tool_call_id": "1", "content": tool}]
    msgs.append({"role": "assistant", "content": final})
    return {"system": "s", "messages": msgs}


def test_json_answer_labels_skip_prose_and_numbers():
    ans = {"verdict": "same_page", "cause": None, "ok": True, "confidence": 0.8, "detail": "a long sentence here"}
    assert answer_labels(ans) == (("cause", None), ("ok", True), ("verdict", "same_page"))
    assert answer_labels(ans, ["verdict"]) == (("verdict", "same_page"),)


def test_free_text_answer_has_no_labels():
    t = _trace("Check your Workday inbox first.")
    assert answer_labels(None) == ()
    assert outcomes([t], []) == {}
    assert decision_cells(t)[-1][2] == ("text",)


def test_cells_read_tool_results_with_any_prefix():
    t = _trace(json.dumps({"verdict": "absent"}), tool='[r:1] {"status": "ok", "count": 0, "nodes": []}')
    cells = decision_cells(t)
    assert cells[-1][1] == (("probe", (("status", "ok"),)),)
    assert cells[-1][2] == ("answer", (("verdict", "absent"),))
