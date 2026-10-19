import json

import httpx
import pytest

from bandits.rank.client import RankClient, RankError, Receipts

EP = {"id": "e1", "run_id": "r1", "phase": "full", "attempt": 0, "task": {"id": "t1", "input": "x"}}


def _client(handler):
    return RankClient(
        "https://x.test/", "bnd_key", http=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_loop_waits_then_returns_episodes_until_done():
    replies = iter(
        [
            httpx.Response(202, json={"detail": "wait"}),
            httpx.Response(200, json=EP),
            httpx.Response(204),
        ]
    )
    client = _client(lambda request: next(replies))
    ep = client.next_episode("r1", poll=0)
    assert ep.id == "e1" and ep.task["input"] == "x"
    assert client.headers(ep) == {"Authorization": "Bearer bnd_key", "x-bandr-episode": "e1"}
    assert client.next_episode("r1", poll=0) is None


def test_report_uses_reef_shaped_fields():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), url=str(request.url))
        return httpx.Response(200, json={"ok": True})

    client = _client(handler)
    ep = _client(lambda r: httpx.Response(200, json=EP)).next_episode("r1")
    client.report(ep, score=1, references=["rec1"], feedback="done")
    assert seen["url"] == "https://x.test/rank/report"
    assert (
        seen["score"] == 1 and seen["references"] == ["rec1"] and seen["metadata"] == {"task": "t1"}
    )


def test_errors_carry_the_servers_reason():
    client = _client(lambda r: httpx.Response(409, json={"detail": "checks_failed"}))
    with pytest.raises(RankError) as e:
        client.next_episode("r1")
    assert e.value.detail == "checks_failed"


def test_receipts_are_collected_from_responses():
    got = Receipts()
    got.add(httpx.Response(200, headers={"x-bandr-record-id": "rec1"}))
    got.add(httpx.Response(200))
    assert got.ids == ["rec1"]
