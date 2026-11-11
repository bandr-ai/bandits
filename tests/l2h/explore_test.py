"""sample / show / query: seeded, no silent truncation, raw logs untouched."""

import json


def ids(res):
    return [i["id"] for i in res["data"]["items"]]


def test_sampling_is_seeded_and_stratified(env):
    env.init()
    a = env.run("sample", "-n", "5", "--json", "--dry-run")[3]
    b = env.run("sample", "-n", "5", "--json", "--dry-run")[3]
    assert ids(a) == ids(b)
    shapes = {i["shape"] for i in a["data"]["items"]}
    assert (
        len(shapes) >= 3
    )  # qa shape, qa-without-prompt, heartbeat/text/unparsed: varied, not first-N
    assert any(i["file"] == "notes.txt" for i in a["data"]["items"])  # other file type is covered
    cfg = json.loads((env.ws / "l2h.json").read_text())
    cfg["seed"] = 12345
    (env.ws / "l2h.json").write_text(json.dumps(cfg))
    c = env.run("sample", "-n", "7", "--json", "--dry-run")[3]
    assert len(ids(c)) == 7
    assert a["data"]["total_records"] == 9


def test_unseen_skips_shown_records_and_exhausts(env):
    env.init()
    seen = set()
    for _ in range(3):
        res = env.run("sample", "-n", "4", "--unseen", "--json")[3]
        new = set(ids(res))
        assert not (new & seen)
        seen |= new
    assert len(seen) == 9
    assert env.run("sample", "-n", "4", "--unseen", "--json")[3]["data"]["items"] == []


def test_sample_matching_returns_fresh_fails_and_passes(env):
    env.build_pipeline()
    res = env.run("sample", "--matching", "has_answer", "-n", "2", "--json")[3]["data"]
    assert (
        res["population"] == {"fail": 1, "pass": 3, "unknown": 0} or res["population"]["fail"] == 1
    )
    labels = {i["check_result"] for i in res["items"]}
    assert labels == {"fail", "pass"}
    res2 = env.run("sample", "--matching", "has_answer", "-n", "2", "--unseen", "--json")[3]["data"]
    assert not ({i["id"] for i in res["items"]} & {i["id"] for i in res2["items"]})
    assert env.run("sample", "--matching", "nope")[0] == 1


def test_show_truncates_loudly_and_full_shows_all(env):
    env.init()
    big = "x" * 2000
    (env.logs / "big.jsonl").write_text(
        json.dumps({"id": 1, "blob": big, "items": list(range(80))}) + "\n"
    )
    code, out, _, res = env.run("show", "big.jsonl#0", "--json")
    rec = res["data"]["record"]
    assert (
        "CUT: showed 500 of 2000 chars" in rec["blob"]
        and "l2h show big.jsonl#0 --field blob --full" in rec["blob"]
    )
    assert "CUT: showed 50 of 80 items" in rec["items"][-1]
    full = env.run("show", "big.jsonl#0", "--full", "--json")[3]["data"]["record"]
    assert full["blob"] == big and len(full["items"]) == 80
    assert (
        env.run("show", "big.jsonl#0", "--field", "blob", "--full", "--json")[3]["data"]["record"]
        == big
    )
    code, _, _, res = env.run("show", "big.jsonl#99", "--json")
    assert code == 1 and res["error"]["fix"]
    assert env.run("show", "big.jsonl#0", "--field", "nope.x")[0] == 1


def test_show_attachments_materialises_copies(env):
    env.build_pipeline()
    res = env.run("show", "q1", "--attachments", "--json")[3]["data"]
    p = res["attachments"][0]["path"]
    assert open(p).read() == "17\n25" and "/out/attachments/" in p
    assert env.run("show", "a.jsonl#0", "--attachments")[0] == 1  # raw record has no attachments


def test_query_count_group_grep_and_no_silent_cut(env):
    env.build_pipeline()
    r = env.run("query", "True", "--count", "--json")[3]["data"]
    assert r["source"] == "trajectories" and r["matched"] == 5 and "ids" not in r
    r = env.run("query", "True", "--group-by", "ended", "--json")[3]["data"]
    assert r["groups"] == {"done": 4, "error": 1}  # q4 is an error; others done
    r = env.run("query", "len(r.events) == 2 and r.meta.answer", "--json", "--limit", "2")[3][
        "data"
    ]
    assert r["matched"] == 4 and r["shown"] == 2 and r["cut"] is True
    _, out, _, _ = env.run("query", "True", "--limit", "2")
    assert "CUT: showed 2 of 5" in out
    r = env.run("query", "True", "--grep", "BLOCK", "--count", "--json")[3]["data"]
    assert r["matched"] == 1
    r = env.run("query", "True", "--raw", "--count", "--json")[3]["data"]
    assert r["source"] == "raw" and r["matched"] == 9
    r = env.run("query", "'kind' in r and r['kind'] == 'qa'", "--raw", "--count", "--json")[3][
        "data"
    ]
    assert r["matched"] == 6
    r = env.run("query", "r.nope.deeper > 1", "--count", "--json")[3]["data"]
    assert r["expression_errors"]  # reported, not swallowed
    assert env.run("query", "1 +", "--json")[0] == 2


def test_query_group_by_expression(env):
    env.build_pipeline()
    r = env.run("query", "True", "--group-by", "len(r.meta['nums'])", "--json")[3]["data"]
    assert sum(r["groups"].values()) == 5
