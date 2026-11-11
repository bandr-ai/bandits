"""Write and structurally check Harbor task packages.

Conventions (checked against harbor-framework/harbor, schema_version 1.4):
  task.toml, instruction.md, environment/Dockerfile, tests/test.sh,
  solution/solve.sh. Separate verifier: tests/Dockerfile bakes /tests and the
  solver's answer dir is declared in top-level `artifacts`.
"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from importlib import resources
from pathlib import Path

from .models import Answer, BuildResult, ContractError, safe_id

ANSWER_DIR = "/app/answer"
SCHEMA_VERSION = "1.4"
ORG = "l2h"
REQUIRED = [
    "task.toml",
    "instruction.md",
    "tests/test.sh",
    "tests/expected.json",
    "solution/solve.sh",
]


def grade_source() -> str:
    return resources.files("l2h").joinpath("templates/grade.py").read_text()


def _toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string escapes are valid TOML basic strings


def _toml_list(items) -> str:
    return "[" + ", ".join(_toml_str(str(i)) for i in items) + "]"


def task_toml(b: BuildResult, ttype: str, generator: str, sources: list[str]) -> str:
    name = f"{ORG}/{ttype}-{b.task_id}".lower()
    lines = [
        f'schema_version = "{SCHEMA_VERSION}"',
        f"artifacts = [{_toml_str(ANSWER_DIR)}]",
        "",
        "[task]",
        f"name = {_toml_str(name)}",
        f"description = {_toml_str(b.description or f'{ttype} task {b.task_id}')}",
        "authors = []",
        f"keywords = {_toml_list(['l2h', ttype, *b.tags])}",
        "",
        "[metadata]",
        f"difficulty = {_toml_str(b.difficulty)}",
        'category = "log-derived"',
        f"tags = {_toml_list(b.tags)}",
        "",
        "# Provenance lives here, never inside the agent-visible environment.",
        "[metadata.l2h]",
        f"type = {_toml_str(ttype)}",
        f"generator = {_toml_str(generator)}",
        f"gold = {_toml_str(b.gold_type)}",
        f"sources = {_toml_list(sources)}",
        f"assumptions = {_toml_list(b.assumptions)}",
        "",
        "[verifier]",
        "timeout_sec = 120.0",
        'environment_mode = "separate"',
        "",
        "[verifier.environment]",
        'network_mode = "no-network"',
        "",
        "[agent]",
        "timeout_sec = 900.0",
        "",
        "[environment]",
        "build_timeout_sec = 600.0",
        "cpus = 1",
        "memory_mb = 2048",
        "storage_mb = 10240",
        "",
    ]
    return "\n".join(lines)


TEST_SH = """#!/bin/bash
# l2h verifier: deterministic, no LLM. Reward goes to /logs/verifier/reward.txt.
TESTS_DIR="${L2H_TESTS_DIR:-/tests}"
ANSWER_DIR="${L2H_ANSWER_DIR:-/app/answer}"
REWARD_DIR="${L2H_REWARD_DIR:-/logs/verifier}"
mkdir -p "$REWARD_DIR"
python3 "$TESTS_DIR/grade.py" --expected "$TESTS_DIR/expected.json" \\
  --answer-dir "$ANSWER_DIR" --reward "$REWARD_DIR/reward.txt"
"""

TESTS_DOCKERFILE = """FROM python:3.12-slim

COPY --chmod=755 . /tests/
"""


def solve_sh(ans: Answer) -> str:
    return (
        "#!/bin/bash\n"
        "# Oracle: writes the logged answer. It proves the verifier is wired up,\n"
        "# not that the answer is derivable from the inputs.\n"
        'ANSWER_DIR="${L2H_ANSWER_DIR:-/app/answer}"\n'
        'mkdir -p "$ANSWER_DIR"\n'
        f"printf '%s\\n' {shlex.quote(ans.canonical())} > \"$ANSWER_DIR/{ans.file}\"\n"
    )


def env_dockerfile(has_data: bool) -> str:
    s = f"FROM ubuntu:24.04\nWORKDIR /app\nRUN mkdir -p {ANSWER_DIR}\n"
    if has_data:
        s += "COPY data/ /app/\n"
    return s


def materialize(
    b: BuildResult, dest: Path, ttype: str, generator: str, sources: list[str]
) -> list[str]:
    """Write the task package to dest. Returns relative paths written."""
    safe_id(b.task_id)
    if b.gold_type not in ("verified", "solved", "imitation", "none"):
        raise ContractError(f"gold_type {b.gold_type!r} is not valid")
    files: dict[str, str | bytes] = {}
    if b.answer is None:
        raise ContractError("BuildResult.answer is required: the verifier needs an expected answer")
    b.answer.validate()
    ans_path = f"{ANSWER_DIR}/{b.answer.file}"

    if b.instruction is not None:
        text = b.instruction.rstrip() + "\n"
        if ans_path not in text:
            text += f"\nWrite your final answer to `{ans_path}` (just the answer, nothing else).\n"
        files["instruction.md"] = text
    for k, v in b.evidence.items():
        files[f"environment/data/{k}"] = v
    files.update(b.files)  # generator overrides win

    files.setdefault("task.toml", task_toml(b, ttype, generator, sources))
    files.setdefault("tests/expected.json", json.dumps(b.answer.to_dict(), indent=2) + "\n")
    files.setdefault("tests/grade.py", grade_source())
    files.setdefault("tests/test.sh", TEST_SH)
    files.setdefault("tests/Dockerfile", TESTS_DOCKERFILE)
    files.setdefault("solution/solve.sh", solve_sh(b.answer))
    has_data = any(k.startswith("environment/data/") for k in files)
    files.setdefault("environment/Dockerfile", env_dockerfile(has_data))

    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    written = []
    for relp in sorted(files):
        p = (dest / relp).resolve()
        if dest.resolve() not in p.parents:
            raise ContractError(f"file path {relp!r} escapes the task dir")
        p.parent.mkdir(parents=True, exist_ok=True)
        v = files[relp]
        p.write_bytes(v if isinstance(v, bytes) else v.encode())
        if relp.endswith(".sh"):
            p.chmod(0o755)
        written.append(relp)
    return written


def structure_check(task_dir: Path) -> list[str]:
    """Level-1 structure gate. Returns a list of problems (empty = pass)."""
    problems: list[str] = []
    for r in REQUIRED:
        if not (task_dir / r).is_file():
            problems.append(f"missing {r}")
    if not (task_dir / "environment").is_dir():
        problems.append("missing environment/")
    cfg = {}
    if (task_dir / "task.toml").is_file():
        try:
            cfg = tomllib.loads((task_dir / "task.toml").read_text())
        except tomllib.TOMLDecodeError as e:
            problems.append(f"task.toml is not valid TOML: {e}")
    if cfg:
        if not re.match(
            r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$", cfg.get("task", {}).get("name", "")
        ):
            problems.append("task.toml [task].name must be org/name")
        mode = cfg.get("verifier", {}).get("environment_mode")
        if mode == "separate" and not cfg.get("artifacts"):
            problems.append(
                "separate verifier but no top-level `artifacts`: the solver's answer would never reach it"
            )
        if (
            mode == "separate"
            and not (task_dir / "tests/Dockerfile").is_file()
            and not any(
                (task_dir / "environment" / n).exists()
                for n in ("Dockerfile", "docker-compose.yaml")
            )
        ):
            problems.append("no image definition for the verifier (tests/Dockerfile)")
    env = task_dir / "environment"
    if (
        env.is_dir()
        and not (env / "Dockerfile").is_file()
        and not (env / "docker-compose.yaml").is_file()
    ):
        if "docker_image" not in cfg.get("environment", {}):
            problems.append("environment/ needs a Dockerfile (or docker_image in task.toml)")
    df = env / "Dockerfile"
    if df.is_file():
        for line in df.read_text().splitlines():
            m = re.match(r"\s*(COPY|ADD)\s+(.*)", line, re.I)
            if not m:
                continue
            toks = [t for t in shlex.split(m.group(2)) if not t.startswith("--")]
            for src in toks[:-1]:
                if "://" in src or "$" in src or any(c in src for c in "*?["):
                    continue
                if not (env / src).exists():
                    problems.append(
                        f"Dockerfile {m.group(1)} source {src!r} does not exist in environment/"
                    )
    ts = task_dir / "tests/test.sh"
    if ts.is_file() and "/logs/verifier/reward" not in ts.read_text():
        problems.append("tests/test.sh never mentions /logs/verifier/reward.txt")
    exp = task_dir / "tests/expected.json"
    if exp.is_file():
        try:
            spec = json.loads(exp.read_text())
            Answer(
                **{k: spec[k] for k in ("expected", "check", "file", "normalize") if k in spec}
            ).validate()
            instr = (
                (task_dir / "instruction.md").read_text()
                if (task_dir / "instruction.md").is_file()
                else ""
            )
            if f"{ANSWER_DIR}/{spec.get('file', 'answer.txt')}" not in instr:
                problems.append("instruction.md never says where to write the answer")
        except (ValueError, TypeError, KeyError, ContractError) as e:
            problems.append(f"tests/expected.json is inconsistent: {e}")
    if (task_dir / "instruction.md").is_file() and not (
        task_dir / "instruction.md"
    ).read_text().strip():
        problems.append("instruction.md is empty")
    return problems
