"""Evidence helper: candidate reads/writes/commands/sources, conservative completeness,
on two trajectory shapes (shell-driven coding agent; structured file/API tools)."""

from mint import evidence
from mint.models import Trajectory


def step(i, calls, results=None, source="agent"):
    s = {"step_id": i, "source": source, "message": ""}
    if calls:
        s["tool_calls"] = [
            {"tool_call_id": f"c{i}_{j}", "function_name": n, "arguments": a}
            for j, (n, a) in enumerate(calls)
        ]
        s["observation"] = {
            "results": [
                {"source_call_id": f"c{i}_{j}", "content": r} for j, r in enumerate(results or [])
            ]
        }
    return s


SHELL = Trajectory(
    {
        "steps": [
            {"step_id": 1, "source": "user", "message": "build the tool from data.txt"},
            step(2, [("bash", {"command": "cat data.txt && head -n 5 big.log"})], ["a\nb", "x"]),
            step(
                3,
                [("bash", {"command": "wget https://example.org/src.tar.gz && tar xf src.tar.gz"})],
                [""],
            ),
            step(
                4,
                [("bash", {"command": "python gen.py > out.txt; echo ok | tee -a log.txt"})],
                ["ok"],
            ),
            step(5, [("bash", {"command": "cat out.txt 2>/dev/null"})], ["result"]),
            step(6, [("bash", {"command": "cat huge.csv"})], ["1,2\n... [truncated 900 lines]"]),
        ]
    }
)

TOOLS = Trajectory(
    {
        "steps": [
            {"step_id": 1, "source": "user", "message": "fix the config"},
            step(2, [("read_file", {"path": "conf.yaml"})], ["k: v"]),
            step(3, [("view", {"path": "notes.md", "view_range": [1, 20]})], ["..."]),
            step(4, [("edit_file", {"path": "conf.yaml", "old": "v", "new": "w"})], ["ok"]),
            step(5, [("write_file", {"file_path": "report.md", "content": "done"})], ["ok"]),
            step(6, [("web_fetch", {"url": "https://docs.example.org/x"})], ["page"]),
            step(7, [("query_db", {"sql": "select 1"})], ["1"]),
        ],
        "subagent_trajectories": [
            {"steps": [step(1, [("read_file", {"path": "sub.txt"})], ["s"])]}
        ],
    }
)


def paths(items):
    return [(i["path"], i.get("complete")) for i in items]


def test_shell_reads_writes_sources_and_completeness():
    ev = evidence.extract(SHELL)
    reads = paths(ev["reads"])
    assert ("data.txt", None) in reads  # a full cat is still unproven: unknown, not True
    assert ("big.log", False) in reads  # head is partial
    assert ("huge.csv", False) in reads  # output shows truncation
    assert {"out.txt", "log.txt"} <= {p for p, _ in paths(ev["writes"])}
    assert "/dev/null" not in {p for p, _ in paths(ev["writes"])}
    assert ev["sources"][0]["url"] == "https://example.org/src.tar.gz"
    assert len(ev["commands"]) == 5
    assert all(i["step"] and i["call"] for k in evidence.KINDS for i in ev[k])
    assert not any(i.get("complete") is True for i in ev["reads"])


def test_starting_files_exclude_outputs_read_back():
    s = evidence.summary(evidence.extract(SHELL))
    start = {f["path"] for f in s["starting_files"]}
    assert {"data.txt", "big.log", "huge.csv"} <= start
    assert "out.txt" not in start  # written at step 4 before being read at step 5
    assert "out.txt" in s["outputs"]


def test_structured_tools_and_unknowns_and_subagents():
    ev = evidence.extract(TOOLS)
    assert ("conf.yaml", None) in paths(ev["reads"])
    assert ("notes.md", False) in paths(ev["reads"])  # view_range = partial
    writes = {i["path"]: i for i in ev["writes"]}
    assert writes["conf.yaml"]["note"] == "edit" and writes["conf.yaml"]["complete"] is False
    assert "report.md" in writes
    assert ev["sources"][0]["url"] == "https://docs.example.org/x"
    assert [u["tool"] for u in ev["unknown"]] == ["query_db"]  # never guessed at
    assert any(i["step"] == "sub0/1" and i["path"] == "sub.txt" for i in ev["reads"])
    s = evidence.summary(ev)
    assert s["unknown_tools"] == ["query_db"]
    assert "conf.yaml" in {f["path"] for f in s["starting_files"]}  # read before edited


def test_custom_interpreter_is_used_before_built_ins():
    def db(call):
        if call.name == "query_db":
            return [{"kind": "reads", "path": "db:" + call.arguments["sql"], "complete": True}]
        return None

    evidence.register("db", db)
    try:
        ev = evidence.extract(TOOLS)
        assert ("db:select 1", True) in paths(ev["reads"])
        assert ev["unknown"] == []
    finally:
        evidence._REGISTRY[:] = [e for e in evidence._REGISTRY if e[0] != "db"]


def test_broken_interpreter_does_not_hide_the_call():
    evidence.register("boom", lambda call: 1 / 0)
    try:
        ev = evidence.extract(TOOLS)
        assert ev["unknown"] and "boom failed" in ev["unknown"][0]["note"]
    finally:
        evidence._REGISTRY[:] = [e for e in evidence._REGISTRY if e[0] != "boom"]


def test_a_file_written_and_read_in_one_command_is_not_a_starting_file():
    t = Trajectory(
        {
            "steps": [
                step(1, [("bash", {"command": "python gen.py > out.txt && cat out.txt"})], ["42"])
            ]
        }
    )
    s = evidence.summary(evidence.extract(t))
    assert "out.txt" not in {f["path"] for f in s["starting_files"]}
    assert "out.txt" in s["outputs"]


def test_files_an_opaque_command_may_have_created_are_not_assumed_pre_existing():
    t = Trajectory(
        {
            "steps": [
                step(1, [("bash", {"command": "cat README.md"})], ["hi"]),
                step(2, [("bash", {"command": "tar xf src.tar.gz"})], [""]),
                step(3, [("bash", {"command": "cat src/main.c"})], ["int main"]),
                step(4, [("bash", {"command": "make && cat build.log"})], ["ok"]),
            ]
        }
    )
    files = {f["path"]: f for f in evidence.starting_files(evidence.extract(t))}
    assert files["README.md"]["pre_existing"] is True  # nothing ran before that could create it
    assert files["src/main.c"]["pre_existing"] is None  # tar may have created it
    assert files["build.log"]["pre_existing"] is None  # make may have, in the same command
