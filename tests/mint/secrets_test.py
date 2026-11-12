"""Secret scan (level 1) and binary evidence. Every credential below is made up."""

import json

from mint import secrets

from .conftest import TASKTYPE_PY
from .validate_test import rows

FAKE = {
    "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
    "sk": "sk-" + "a1b2c3d4e5f6a1b2c3d4e5f6",
    "gh": "ghp_" + "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
    "bearer": "Authorization: Bearer " + "abcdef0123456789abcdef",
    "pem": "-----BEGIN RSA PRIVATE KEY-----",
    "pw": "password: Tr0ub4dor&3x",
    "tok": 'token = "q9w8e7r6t5y4"',
    "url": "postgres://svc:s3cr3tPass@db.internal/app",
}


def kinds(text):
    return {k for _, k, _ in secrets.scan_text(text)}


def test_detects_common_secret_shapes_and_redacts():
    for text in FAKE.values():
        assert secrets.scan_text(text), text
    hits = secrets.scan_text(FAKE["sk"])
    assert FAKE["sk"] not in json.dumps(hits) and "redacted" in hits[0][2]
    assert "private_key" in kinds(FAKE["pem"]) and "url_credentials" in kinds(FAKE["url"])


def test_no_false_alarm_on_ordinary_text():
    ok = [
        "the password: required",
        "password = <your password>",
        "token: ${API_TOKEN}",
        "tokens: 4096 per request",
        "see https://example.com/docs and http://user@host/path",
        "secret: null",
        "Bearer token auth is used",
    ]
    for t in ok:
        assert not secrets.scan_text(t), t


def test_env_style_names_and_prose_credentials_are_caught():
    caught = [
        "DB_PASSWORD=" + "hunter2hunter2",
        "AWS_SECRET_ACCESS_KEY=" + "wJalrXUtnFEMI/K7MDENG",
        "export GITHUB_TOKEN=" + "abc123def456",
        "Enter " + "X7kq2zz9" + " in the Password field",
    ]
    for t in caught:
        assert secrets.scan_text(t), t
    assert "credential_in_prose" in kinds(caught[-1])
    spared = [
        "password = getpass.getpass()",
        "DB_PASSWORD=${DB_PASSWORD}",
        "api_key = settings.API_KEY",
        "tokenizer: bert-base-uncased",
        "max_tokens = 409600",
        "token_count: 1234567",
        "TOKEN_URL=https://auth.example.com/oauth/token",
        "secretary: JaneDoe1",
        "Enter your password in the password field",
        "Enter the code shown in the token field",
        "enter 2024-01-05 in the date field",
    ]
    for t in spared:
        assert not secrets.scan_text(t), t


def test_secret_in_environment_blocks_task_and_allow_list_is_logged(env):
    env.build_pipeline()
    leaky = TASKTYPE_PY.replace(
        'evidence={"nums.txt": nums}',
        f'evidence={{"nums.txt": nums, "conf.env": "{FAKE["aws"]}\\n"}}',
    )
    env.write("tasktypes/addnums.py", leaky)
    code, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"blocked": 3}
    r = rows(env)["t_q1"]
    assert r["gates"]["secrets"] == "fail" and r["status"] == "blocked"
    msg = " ".join(r["problems"])
    assert "environment/data/conf.env" in msg and "aws_access_key" in msg
    assert FAKE["aws"] not in json.dumps(r)
    # explicit allow-list: the file is skipped, and the allowance is recorded in the row
    allowed = leaky.replace(
        "answer=Answer(", 'secret_allow=["environment/data/conf.env"], answer=Answer('
    )
    env.write("tasktypes/addnums.py", allowed)
    env.run("run", "generate")
    r = rows(env)["t_q1"]
    assert r["gates"]["secrets"] == "pass" and r["status"] == "generated"
    assert r["allow"]["secrets"] == ["environment/data/conf.env"]


def test_validate_rescans_secrets_added_after_generate(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    p = env.ws / "out/tasks/addnums/t_q1/environment/data/nums.txt"
    p.write_text(p.read_text() + "\n" + FAKE["gh"] + "\n")
    env.run("validate", "--level", "2", "--sample-size", "0")
    r = rows(env)["t_q1"]
    assert r["gates"]["secrets"] == "fail" and r["status"] == "blocked"


def test_binary_evidence_is_packaged_and_skipped_by_scans(env):
    env.build_pipeline()
    png = b"\x89PNG\r\n\x1a\n\x00\x00" + FAKE["aws"].encode() + bytes(range(256))
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace(
            'evidence={"nums.txt": nums}', 'evidence={"nums.txt": nums, "img/pic.png": PNG}'
        ).replace("SPEC =", f"PNG = {png!r}\nSPEC =", 1),
    )
    code, _, _, res = env.run("run", "generate", "--json")
    assert res["data"]["by_status"] == {"generated": 3}, res["data"]
    assert (env.ws / "out/tasks/addnums/t_q1/environment/data/img/pic.png").read_bytes() == png
    code, _, _, res = env.run("validate", "--sample-size", "3", "--json")
    assert code == 0
    assert {r["status"] for r in rows(env).values()} == {"validated"}
    code, out, _, _ = env.run("task", "show", "t_q1")
    assert code == 0 and "<binary" in out


def test_bad_evidence_type_is_a_contract_error(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace('evidence={"nums.txt": nums}', 'evidence={"nums.txt": 5}'),
    )
    _, _, _, res = env.run("run", "generate", "--json")
    assert "must be str or bytes" in (env.ws / "out/queue.jsonl").read_text()
