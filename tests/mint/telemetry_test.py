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

REAL_POST = telemetry.post
KEY = "phc_test_key_not_real"
COMMON = {
    "schema_version",
    "dry_run",
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
    code, out, err, _ = env.run("init", logs, str(env.ws), "--allow-missing-runtime", "--json")
    assert code == 0
    assert json.loads(out)["ok"] is True
    assert len(sent) == 1
    url, payload, timeout = sent[0]
    assert url == "https://us.i.posthog.com/i/v0/e/"
    assert timeout == 2.0
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


def test_host_override_and_failed_post_keeps_exit_code(env, monkeypatch, sent):
    _open(monkeypatch)
    monkeypatch.setenv("MINT_POSTHOG_API_KEY", KEY)
    monkeypatch.setenv("MINT_POSTHOG_HOST", "https://example.test/")

    attempts = []

    def explode(url, body, timeout):
        attempts.append((url, timeout))
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
    assert attempts == [("https://example.test/i/v0/e/", 2.0)]


def test_persisted_off_and_corrupt_config(env, sent, monkeypatch):
    _open(monkeypatch)
    path = _cfg()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"anonymous_id": "keep-me", "enabled": False, "notice_shown": False})
    )
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
    assert set(sent[0][1]["properties"]) == COMMON | {"kind"}
    sent.clear()
    env.run("task", "try", "t_q1", "--answer", "SECRETANSWER")
    blob = json.dumps(sent[-1][1])
    assert "SECRETANSWER" not in blob
    assert "t_q1" not in blob
    assert sent[-1][1]["properties"]["command"] == "task try"


def test_validate_buckets_and_exit_class(env, sent, monkeypatch):
    _open(monkeypatch)
    env.init()

    def fake(a, ws):
        return Result(
            {
                "runner": "local",
                "levels": [2, 1],
                "jobs": 4,
                "task_id": "SECRET_TASK",
                "level1": {"checked": 1, "failed": 0},
                "types": {
                    "SECRET_TYPE": {
                        "sampled": ["SECRET_TASK"],
                        "passed": [],
                        "failed": ["SECRET_TASK"],
                        "infra_error": [],
                        "resumed": [],
                    }
                },
                "any_failure": True,
                "any_infra_error": False,
            },
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
    assert props["sampled"] == "1"
    assert props["passed"] == "0"
    assert props["failed"] == "1"
    assert props["infra_error"] == "0"
    assert props["level1_checked"] == "1"
    assert props["any_failure"] is True
    assert props["failure_phase"] == "execute"
    assert props["error_type"] == "operation"
    assert "validated" not in props
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


def test_real_stage_summaries_expose_partial_failures(env, sent, monkeypatch):
    _open(monkeypatch)
    env.build_pipeline()
    # The fixture includes a deliberately crashing check. CLI success does not
    # mean every check ran successfully.
    props = sent[-1][1]["properties"]
    assert props["stage"] == "checks" and props["exit"] == "ok"
    assert props["outputs"] != "0" and props["errors"] != "0"
    assert env.run("run", "generate")[0] == 0
    props = sent[-1][1]["properties"]
    assert props["stage"] == "generate" and props["exit"] == "ok"
    assert props["outputs"] != "0" and props["queued"] != "0"
    assert "generator bug" not in json.dumps(sent)
    assert "check bug" not in json.dumps(sent)


def test_scoped_validation_dry_run_uses_current_summary(env, sent, monkeypatch, local_runner):
    _open(monkeypatch)
    env.build_pipeline()
    env.run("run", "generate")
    before = (env.ws / "out/results.jsonl").read_bytes()
    code, _, _, _ = env.run(
        "validate", "--runner", "local", "--task", "t_q1", "--level", "1", "--dry-run"
    )
    assert code == 0
    props = sent[-1][1]["properties"]
    assert props["dry_run"] is True
    assert props["requested_runner"] == props["runner"] == "local"
    assert props["level1_checked"] == "1"
    assert "sampled" not in props  # Level 2 did not run.
    assert (env.ws / "out/results.jsonl").read_bytes() == before
    assert "t_q1" not in json.dumps(sent[-1][1])


def test_task_try_verifier_fail_is_separate_from_command_success(env, sent, monkeypatch):
    _open(monkeypatch)
    env.build_pipeline()
    env.run("run", "generate")
    assert env.run("task", "try", "t_q1", "--answer", "99999")[0] == 0
    props = sent[-1][1]["properties"]
    assert props["exit"] == "ok" and props["verifier_outcome"] == "fail"
    assert "99999" not in json.dumps(props)


def test_early_failure_keeps_requested_validation_context(env, sent, monkeypatch):
    _open(monkeypatch)
    env.init()
    assert env.run("validate", "--runner", "local", "--level", "3,1")[0] == 1
    props = sent[-1][1]["properties"]
    assert props["levels"] == "1,3" and props["requested_runner"] == "local"
    assert props["failure_phase"] == "execute" and props["error_type"] == "operation"
    assert "runner" not in props and "level1_checked" not in props


def test_bad_extra_values_cannot_leak_text_or_drop_valid_fields(sent, monkeypatch):
    _open(monkeypatch)
    telemetry.emit(
        "validate",
        "error",
        1,
        {
            "dry_run": True,
            "levels": "SECRET",
            "passed": "SECRET",
            "runner": ["SECRET"],
            "requested_runner": "auto",
            "sampled": "2-9",
            "error_type": "SECRET",
            "task_id": "SECRET",
            "failure_phase": "execute",
        },
    )
    props = sent[-1][1]["properties"]
    assert "SECRET" not in json.dumps(props)
    assert props["dry_run"] is True and props["sampled"] == "2-9"
    assert props["requested_runner"] == "auto"
    assert props["failure_phase"] == "execute"


def test_real_transport_posts_json_to_loopback_only(monkeypatch):
    # Exercise urllib, not the capture stub, without sending production events.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    self.headers.get_content_type(),
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": 1}')

        def log_message(self, *_args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 2
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            REAL_POST(
                f"http://127.0.0.1:{server.server_port}/i/v0/e/",
                json.dumps({"event": "command_finished"}).encode(),
                2,
            )
        finally:
            thread.join(timeout=3)
    assert received == [("/i/v0/e/", "application/json", {"event": "command_finished"})]


@pytest.mark.parametrize("levels", [None, "1,2", 2])
def test_malformed_summary_levels_preserve_safe_context(env, sent, monkeypatch, levels):
    _open(monkeypatch)
    env.init()
    monkeypatch.setattr(
        cli, "cmd_validate", lambda *_a: Result({"levels": levels, "types": {}}, "failed", exit=1)
    )
    assert env.run("validate", "--runner", "local", "--judge", "--dry-run")[0] == 1
    props = sent[-1][1]["properties"]
    assert props["judge"] is True and props["dry_run"] is True
    assert props["requested_runner"] == "local"
    assert props["failure_phase"] == "execute" and props["error_type"] == "operation"
    assert "sampled" not in props


def test_render_operation_error_has_render_phase(env, sent, monkeypatch):
    from mint.output import MINTError

    _open(monkeypatch)
    env.init()

    def fail_render(*_args):
        raise MINTError("SECRET rendering error", "retry")

    monkeypatch.setattr(cli, "render", fail_render)
    assert env.run("report")[0] == 1
    props = sent[-1][1]["properties"]
    assert props["failure_phase"] == "render"
    assert props["error_type"] == "operation"
    assert "SECRET" not in json.dumps(props)
