from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.redact import SECRETS_ONLY_RULESET, redact_source
from bandits.store import ArtifactStore

runner = CliRunner()


def _otlp(task: str, attributes: str = "") -> str:
    return (
        '{"trace_id":"t","span_id":"s","parent_span_id":null,"name":"gpt",'
        '"start_time":"2026-01-01T00:00:00Z","end_time":"2026-01-01T00:00:01Z",'
        f'"attributes":{{"gen_ai.operation.name":"chat","task":"{task}"{attributes}}}}}\n'
    )


def test_redaction_marks_corpus_and_preserves_original_digest(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    secret = "sk-exampleSecret123456789"
    source.write_text(_otlp(f"Use {secret}"))

    corpus = load_corpus(source, "otlp")

    assert secret not in corpus.model_dump_json()
    assert "[REDACTED:openai_api_key]" in corpus.model_dump_json()
    assert any(issue.kind == "redaction" for issue in corpus.issues)
    assert corpus.traces[0].source_digest == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cli_never_persists_nested_named_secret(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    secret = "super-secret-value"
    source.write_text(_otlp("Safe task", f',"metadata":{{"password":"{secret}"}}'))
    project = tmp_path / "project"

    result = runner.invoke(
        app, ["ingest", str(source), "--source", "otlp", "--project", str(project)]
    )

    assert result.exit_code == 0
    persisted = b"".join(path.read_bytes() for path in project.rglob("*") if path.is_file())
    assert secret.encode() not in persisted
    store = ArtifactStore(project / ".bandits")
    artifact_id = store.list()[0].artifact_id
    assert any(issue.kind == "redaction" for issue in store.read(artifact_id).issues)


def test_location_survives_an_earlier_multiline_replacement(tmp_path: Path) -> None:
    """A private key collapsing ten lines must not shift what comes after it."""
    source = tmp_path / "trace.jsonl"
    key = "-----BEGIN PRIVATE KEY-----\n" + "MIIBVgIBADANBgkq\n" * 8 + "-----END PRIVATE KEY-----"
    source.write_text(f"{key}\nfiller\ncontact alice@realmail.io\n")

    issues = redact_source(source).issues

    assert [i.detail for i in issues] == [
        "redacted detected private_key",
        "redacted detected email_address",
    ]
    assert issues[0].location.endswith(":1")
    assert issues[1].location.endswith(":12")


def test_secret_containing_a_space_is_fully_redacted(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_text('{"password": "hunter two rest"}')

    assert redact_source(source).data == b'{"password": "[REDACTED:named_secret]"}'


def test_overlapping_rules_replace_a_value_once(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_text('{"api_key": "sk-aaaaaaaaaaaaaaaaaaaa"}')

    result = redact_source(source)

    assert result.data.count(b"[REDACTED:") == 1
    assert b"sk-aaaa" not in result.data


def test_secrets_only_ruleset_keeps_a_task_identifying_email(tmp_path: Path) -> None:
    """An email is often the task's own identifier; removing it can void the task."""
    source = tmp_path / "trace.jsonl"
    source.write_text(_otlp("Refund the order for alice@realmail.io"))

    corpus = load_corpus(source, "otlp", SECRETS_ONLY_RULESET)

    assert corpus.traces[0].task == "Refund the order for alice@realmail.io"
    assert corpus.redaction_ruleset == "secrets-only-v1"
    assert load_corpus(source, "otlp").redaction_ruleset == "default-v1"


def test_ingest_records_the_ruleset_that_produced_the_corpus(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_text(_otlp("Refund the order for alice@realmail.io"))
    project = tmp_path / "project"

    result = runner.invoke(
        app,
        # fmt: off
        [
            "ingest",
            str(source),
            "--source",
            "otlp",
            "--redaction",
            "secrets-only-v1",
            "--project",
            str(project),
        ],
        # fmt: on
    )

    assert result.exit_code == 0
    assert "redaction:   secrets-only-v1" in result.stdout
    store = ArtifactStore(project / ".bandits")
    assert store.read(store.list()[0].artifact_id).redaction_ruleset == "secrets-only-v1"


def test_unknown_ruleset_exits_nonzero(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_text(_otlp("Safe task"))

    result = runner.invoke(
        app,
        # fmt: off
        [
            "ingest",
            str(source),
            "--source",
            "otlp",
            "--redaction",
            "nope",
            "--project",
            str(tmp_path / "p"),
        ],
        # fmt: on
    )

    assert result.exit_code == 1


def test_redaction_never_orphans_a_json_escape(tmp_path: Path) -> None:
    """`\\n@pytest.mark.x` offers `n@pytest.mark.x` as a well-formed address.

    Consuming its leading `n` leaves the backslash glued to the marker, which is
    an invalid escape, and the whole record is lost rather than merely redacted.
    Measured on a real session log: ten records destroyed this way.
    """
    source = tmp_path / "trace.jsonl"
    source.write_text(json.dumps({"text": "assert x == 1\n\n@pytest.mark.parametrize(\n"}) + "\n")

    result = redact_source(source)

    json.loads(result.data.decode())
    assert b"@pytest.mark.parametrize" in result.data


@pytest.mark.parametrize(
    "address",
    ["git@github.com", "noreply@anthropic.com", "agent@example.com", "root@localhost"],
)
def test_role_addresses_and_reserved_domains_are_not_personal(tmp_path: Path, address: str) -> None:
    """Redacting a git remote destroys a real fact and protects nobody."""
    source = tmp_path / "trace.jsonl"
    source.write_text(_otlp(f"Push to {address}"))

    assert address.encode() in redact_source(source).data


def test_a_real_address_is_still_redacted(tmp_path: Path) -> None:
    source = tmp_path / "trace.jsonl"
    source.write_text(_otlp("Mail alice.smith@realmail.io about it"))

    result = redact_source(source)

    assert b"alice.smith@realmail.io" not in result.data
    assert b"[REDACTED:email_address]" in result.data


def test_a_long_dotted_run_does_not_backtrack(tmp_path: Path) -> None:
    """This pattern once spent 85 seconds on a 20 MB file before matching nothing."""
    source = tmp_path / "trace.jsonl"
    source.write_text("a" + ".a" * 4000 + "@" + "b" * 200 + "\n")

    started = time.monotonic()
    redact_source(source)

    assert time.monotonic() - started < 2.0
