"""Anonymous telemetry: milestone commands only, no paths, no network in the suite."""

import json
import os
import socket
import sys
from pathlib import Path

import pytest

from mint import cli, telemetry
from mint.cli import main
from mint.models import ContractError
from mint.output import Result
from mint.telemetry import NOTICE, bucket

KEY = "phc_test_key_not_real"
COMMON = {
    "mint_version",
    "python_version",
    "os",
    "command",
    "duration_ms",
    "exit",
    "$process_person_profile",
    "$geoip_disable",
}


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def record(url, body, timeout):
        calls.append((url, json.loads(body), timeout))

    monkeypatch.setattr("mint.telemetry.post", record)
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", KEY)
    monkeypatch.delenv("MINT_POSTHOG_HOST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("MINT_TELEMETRY_DISABLED", raising=False)
    return calls


def _cfg() -> Path:
    return Path(os.environ["XDG_CONFIG_HOME"]) / "mint" / "telemetry.json"


def _open(monkeypatch):
    """Pytest sets PYTEST_CURRENT_TEST again at the start of the call phase."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def test_bucket_edges():
    assert bucket(0) == "0"
    assert bucket(1) == "1"
    assert bucket(-3) == "0"
    assert bucket(9) == "2-9"
    assert bucket(10) == "10-99"
    assert bucket(100) == "100-999"
    assert bucket(1000) == "1000+"


def test_closed_gates_do_not_send_or_create_a_file(monkeypatch, env, sent):
    _open(monkeypatch)
    cases = [
        {"MINT_POSTHOG_API_KEY": ""},
        {"MINT_POSTHOG_API_KEY": "phx_personal"},
        {"CI": "true"},
        {"DO_NOT_TRACK": "1"},
        {"MINT_TELEMETRY_DISABLED": "1"},
    ]
    for overrides in cases:
        sent.clear()
        for key, value in overrides.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        code, _, _, _ = env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")
        assert code == 0, overrides
        assert sent == [], overrides
        assert not _cfg().exists(), overrides
        if env.ws.exists():
            # a later case needs a fresh workspace folder
            pass
        monkeypatch.setenv("MINT_POSTHOG_API_KEY", KEY)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.delenv("MINT_TELEMETRY_DISABLED", raising=False)


def test_pytest_gate_keeps_the_exploding_stub_quiet(monkeypatch, env):
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", KEY)
    code, _, _, _ = env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")
    assert code == 0
    assert not _cfg().exists()


def test_init_payload_has_no_paths(env, sent, monkeypatch):
    _open(monkeypatch)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    logs = str(env.logs)
    code, out, err, _ = env.run(
        "init", logs, str(env.ws), "--allow-missing-runtime", "--json"
    )
    assert code == 0
    assert json.loads(out)["ok"] is True
    assert len(sent) == 1
    url, payload, timeout = sent[0]
    assert url == "https://us.i.posthog.com/i/v0/e/"
    assert timeout == 0.2
    assert payload["api_key"] == KEY
    assert payload["event"] == "command_finished"
    props = payload["properties"]
    assert COMMON <= set(props)
    assert set(props) <= COMMON | {
        "docker_present",
        "harbor_present",
        "allow_missing_runtime",
        "dry_run",
    }
    assert props["command"] == "init"
    assert props["exit"] == "ok"
    assert props["allow_missing_runtime"] is True
    assert props["dry_run"] is False
    assert props["$process_person_profile"] is False
    assert props["$geoip_disable"] is True
    assert isinstance(props["docker_present"], bool)
    blob = json.dumps(payload)
    assert logs not in blob
    assert str(env.ws) not in blob
    assert socket.gethostname() not in blob
    assert NOTICE in err
    assert "anonymous usage" not in out


def test_notice_once_on_a_tty_and_report_when_piped(env, sent, monkeypatch):
    _open(monkeypatch)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")[0] == 0
    err = env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")[2]
    assert NOTICE not in err
    assert len(sent) == 2
    assert sent[0][1]["distinct_id"] == sent[1][1]["distinct_id"]
    assert sent[0][1]["distinct_id"] != socket.gethostname()

    _cfg().unlink()
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    code, out, err, _ = env.run("run", "parse")
    assert code == 1  # no parser yet
    assert NOTICE not in err
    assert sent[-1][1]["properties"]["command"] == "run"
    assert sent[-1][1]["properties"]["stage"] == "parse"
    assert "docker_present" not in sent[-1][1]["properties"]

    code, out, err, _ = env.run("report")
    assert NOTICE in err
    assert NOTICE not in out
    report = env.ws / "report.md"
    if report.exists():
        assert "anonymous usage" not in report.read_text()
    # a second report does not repeat it
    assert NOTICE not in env.run("report")[2]


def test_host_override_and_failed_post_keeps_exit_code(env, monkeypatch):
    _open(monkeypatch)
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", KEY)
    monkeypatch.setenv("MINT_POSTHOG_HOST", "https://example.test/")

    def explode(url, body, timeout):
        assert url == "https://example.test/i/v0/e/"
        assert timeout == 0.2
        raise OSError("offline")

    monkeypatch.setattr("mint.telemetry.post", explode)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    code, out, err, _ = env.run(
        "init", str(env.logs), str(env.ws), "--allow-missing-runtime", "--json"
    )
    assert code == 0
    assert json.loads(out)["command"] == "init"
    assert NOTICE in err
    assert "anonymous usage" not in out


def test_persisted_off_and_corrupt_config(env, sent, monkeypatch):
    _open(monkeypatch)
    path = _cfg()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"anonymous_id": "keep-me", "enabled": False, "notice_shown": False}))
    code, _, _, _ = env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")
    assert code == 0 and sent == []
    assert json.loads(path.read_text())["enabled"] is False

    assert main(["telemetry", "on", "--json"]) == 0
    assert json.loads(path.read_text())["enabled"] is True
    assert json.loads(path.read_text())["anonymous_id"] == "keep-me"
    sent.clear()
    assert env.run("status")[0] == 0
    assert sent == []

    path.write_text("{not json")
    before = path.read_text()
    sent.clear()
    assert env.run("status")[0] == 0
    assert sent == []
    assert path.read_text() == before


def test_read_commands_help_and_version_do_not_send(env, sent, monkeypatch):
    _open(monkeypatch)
    env.build_pipeline()
    sent.clear()
    for cmd in (
        ["status"],
        ["sample"],
        ["show", "missing"],
        ["query", "True", "--count"],
        ["schema"],
        ["docs"],
        ["task", "show", "missing"],
        ["telemetry", "status"],
    ):
        env.run(*cmd)
        assert sent == [], cmd
    assert main(["--help"]) == 0
    assert main(["--version"]) == 0
    assert sent == []


def _json(argv, capsys):
    capsys.readouterr()
    code = main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err, json.loads(cap.out)


def test_telemetry_status_without_a_workspace_and_dry_run_still_persists(sent, capsys, monkeypatch):
    _open(monkeypatch)
    code, out, _, res = _json(["telemetry", "status", "--json"], capsys)
    assert code == 0 and res["data"]["reason"] == "on"
    assert res["data"]["effective"] is True
    assert "anonymous_id" not in out
    assert KEY not in out
    assert res["next"] == ["mint telemetry off"]
    assert not _cfg().exists()

    code, _, _, res = _json(["telemetry", "off", "--dry-run", "--json"], capsys)
    assert code == 0 and res["data"]["persisted"] == "off"
    assert res["data"]["effective"] is False
    assert res["data"]["reason"] == "persisted off"
    assert _cfg().exists()
    _, _, _, res = _json(["telemetry", "status", "--json"], capsys)
    assert res["next"] == ["mint telemetry on"]


def test_status_reports_the_earlier_gate(monkeypatch, capsys):
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", "")
    monkeypatch.delenv("POSTHOG_PROJECT_TOKEN", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    code, _, _, res = _json(["telemetry", "status", "--json"], capsys)
    assert code == 0 and res["data"]["reason"] == "no API key"
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", "phx_nope")
    _, _, _, res = _json(["telemetry", "status", "--json"], capsys)
    assert res["data"]["reason"] == "API key is not a project key"


def test_decide_and_task_try_drop_text(env, sent, monkeypatch):
    _open(monkeypatch)
    env.build_pipeline()
    env.run("run", "generate")
    sent.clear()
    code, _, _, _ = env.run("decide", "keep it", "--why", "SECRETWHY")
    assert code == 0 and len(sent) == 1
    blob = json.dumps(sent[0][1])
    assert "SECRETWHY" not in blob
    assert sent[0][1]["properties"]["command"] == "decide"
    assert set(sent[0][1]["properties"]) == COMMON
    sent.clear()
    env.run("task", "try", "t_q1", "--answer", "SECRETANSWER")
    blob = json.dumps(sent[-1][1])
    assert "SECRETANSWER" not in blob
    assert "t_q1" not in blob
    assert sent[-1][1]["properties"]["command"] == "task try"


def test_validate_buckets_and_exit_class(env, sent, monkeypatch):
    _open(monkeypatch)
    env.init()
    rows = [{"status": "validated", "task_id": "SECRET_TASK"}]
    rows += [{"status": "blocked", "task_id": f"b{i}"} for i in range(10)]
    rows += [{"status": "exec_failed", "task_id": f"e{i}"} for i in range(1000)]
    rows += [{"status": "judged", "task_id": "j"}]
    env.write("out/results.jsonl", "".join(json.dumps(r) + "\n" for r in rows))

    def fake(a, ws):
        return Result(
            {"runner": "local", "levels": [2, 1], "jobs": 4, "task_id": "SECRET_TASK"},
            "failed",
            exit=1,
        )

    monkeypatch.setattr(cli, "cmd_validate", fake)
    code, _, _, _ = env.run("validate", "--judge", "--runner", "auto")
    assert code == 1
    props = sent[-1][1]["properties"]
    assert props["command"] == "validate"
    assert props["exit"] == "error"
    assert props["judge"] is True
    assert props["runner"] == "local"
    assert props["levels"] == "1,2"
    assert props["validated"] == "1"
    assert props["generated"] == "0"
    assert props["blocked"] == "10-99"
    assert props["exec_failed"] == "1000+"
    blob = json.dumps(sent[-1][1])
    assert "SECRET_TASK" not in blob
    assert "judged" not in props
    assert "jobs" not in props


def test_contract_and_interrupt(env, sent, monkeypatch):
    _open(monkeypatch)
    env.init()

    def contract(a, ws):
        raise ContractError("bad source_ref SECRETPATH")

    monkeypatch.setattr(cli, "cmd_run", contract)
    code, _, err, _ = env.run("run", "checks")
    assert code == 1 and "SECRETPATH" in err
    props = sent[-1][1]["properties"]
    assert props["exit"] == "contract" and props["stage"] == "checks"
    assert "SECRETPATH" not in json.dumps(sent[-1][1]["properties"])

    def interrupt(a, ws):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "cmd_new", interrupt)
    code, _, _, _ = env.run("new", "parser")
    assert code == 130
    assert sent[-1][1]["properties"]["exit"] == "interrupted"
    assert sent[-1][1]["properties"]["command"] == "new"


def test_judge_calibrate_error_has_no_message(env, sent, monkeypatch):
    for name in ("MINT_JUDGE_BASE_URL", "MINT_JUDGE_API_KEY", "MINT_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    _open(monkeypatch)
    env.init()
    code, _, err, _ = env.run("judge-calibrate")
    assert code == 1
    props = sent[-1][1]["properties"]
    assert props["command"] == "judge-calibrate"
    assert props["exit"] == "error"
    assert "MINT_JUDGE" not in json.dumps(props)
    assert "needs" not in json.dumps(props)


def test_validate_usage_still_records_judge(env, sent, monkeypatch):
    _open(monkeypatch)
    env.init()
    code, _, _, _ = env.run("validate", "--level", "9", "--judge")
    assert code == 2
    props = sent[-1][1]["properties"]
    assert props["command"] == "validate"
    assert props["exit"] == "usage"
    assert props["judge"] is True
    assert "runner" not in props
    assert "levels" not in props


def test_unset_env_uses_embedded_project_key(env, sent, monkeypatch):
    _open(monkeypatch)
    monkeypatch.delenv("MINT_POSTHOG_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_TOKEN", raising=False)
    code, _, _, _ = env.run("init", str(env.logs), str(env.ws), "--allow-missing-runtime")
    assert code == 0 and len(sent) == 1
    sent_key = sent[0][1]["api_key"]
    assert sent_key.startswith("phc_") and sent_key != KEY
    assert sent_key == telemetry.project_key()


def test_skill_carries_the_notice():
    skill = Path(cli.__file__).parent.joinpath("SKILL.md").read_text()
    assert NOTICE in skill
    assert "| Telemetry |" in skill
