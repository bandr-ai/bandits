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

from .models import (
    ENV_SOURCES,
    GOLD_TYPES,
    GRADING_METHODS,
    ORACLE_KINDS,
    Answer,
    BuildResult,
    ContractError,
    normalize_gaps,
    normalize_outcome,
    safe_id,
)

ANSWER_DIR = "/app/answer"
SCHEMA_VERSION = "1.4"
ORG = "l2h"
REQUIRED = ["task.toml", "instruction.md", "tests/test.sh"]


def grade_source() -> str:
    return resources.files("l2h").joinpath("templates/grade.py").read_text()


def _toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string escapes are valid TOML basic strings


def _toml_list(items) -> str:
    return "[" + ", ".join(_toml_str(str(i)) for i in items) + "]"


def task_toml(
    b: BuildResult,
    ttype: str,
    generator: str,
    sources: list[str],
    *,
    separate: bool = True,
    spec: dict | None = None,
) -> str:
    """separate=True: answer-file task, verifier in its own image reading /app/answer.
    separate=False: custom verifier, runs in the solver's environment (checks its end state)."""
    name = f"{ORG}/{ttype}-{b.task_id}".lower()
    spec = spec or {}
    gaps = [f"{g['affects']}: {g['what']}" for g in b.gaps]
    lines = [f'schema_version = "{SCHEMA_VERSION}"']
    if separate:
        lines.append(f"artifacts = [{_toml_str(ANSWER_DIR)}]")
    lines += [
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
        f"job = {_toml_str(str(spec.get('job', '')))}",
        f"task_scope = {_toml_str(str(spec.get('task_scope', '')))}",
        f"oracle = {_toml_str(b.oracle_kind or '')}",
        f"env_source = {_toml_str(b.env_source or '')}",
        f"grading_method = {_toml_str(b.grading_method or '')}",
        f"grading_reason = {_toml_str(b.grading_reason)}",
        f"gaps = {_toml_list(gaps)}",
        f"sources = {_toml_list(sources)}",
        f"assumptions = {_toml_list(b.assumptions)}",
        "",
        "[verifier]",
        "timeout_sec = 120.0",
    ]
    if separate:
        lines += [
            'environment_mode = "separate"',
            "",
            "[verifier.environment]",
            'network_mode = "no-network"',
        ]
    lines += [
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


ORACLE_NOTE = {
    "verified": "the gold answer, independently verified for this task's scope",
    "computed": "the answer recomputed from the task inputs",
    "independent": "the expected answer, checked independently of the logged run",
    "solved": "the answer a fresh solve produced and the verifier accepted",
    "imitation": "the logged answer (imitation gold, not independently checked)",
    "negative": "a valid answer (one of the alternatives; the known-wrong ones are forbidden)",
    "none": "the expected answer",
}


def solve_sh(ans: Answer, value: str | None = None, gold_type: str = "imitation") -> str:
    value = ans.canonical() if value is None else value
    note = ORACLE_NOTE.get(gold_type, "the expected answer")
    caveat = (
        "It proves the verifier is wired up."
        if gold_type in ("verified", "computed", "solved")
        else "It proves the verifier is wired up, not that the answer is derivable from the inputs."
    )
    return (
        "#!/bin/bash\n"
        f"# Oracle ({gold_type} gold): writes {note}.\n"
        f"# {caveat}\n"
        'ANSWER_DIR="${L2H_ANSWER_DIR:-/app/answer}"\n'
        'mkdir -p "$ANSWER_DIR"\n'
        f"printf '%s\\n' {shlex.quote(value)} > \"$ANSWER_DIR\"/{shlex.quote(ans.file)}\n"
    )


def env_dockerfile(has_data: bool) -> str:
    s = f"FROM ubuntu:24.04\nWORKDIR /app\nRUN mkdir -p {ANSWER_DIR}\n"
    if has_data:
        s += "COPY data/ /app/\n"
    return s


def resolve(b: BuildResult) -> None:
    """Check a BuildResult's grading/oracle/environment contract and fill in what l2h infers.

    Mutates b: wrong_answers/alternatives normalised, gaps normalised, oracle_kind /
    env_source / grading_method resolved. Raises ContractError on anything inconsistent.
    """
    safe_id(b.task_id)
    if b.gold_type not in GOLD_TYPES:
        raise ContractError(f"gold_type {b.gold_type!r} must be one of {GOLD_TYPES}")
    custom = b.answer is None  # no answer file: the generator's own verifier grades the end state
    if custom and "tests/test.sh" not in b.files:
        raise ContractError(
            "BuildResult needs answer=Answer(...) (answer-file verifier) or its own "
            "files['tests/test.sh'] (custom verifier)"
        )
    b.wrong_answers = [normalize_outcome(w) for w in b.wrong_answers]
    b.alternatives = [normalize_outcome(a) for a in b.alternatives]
    b.gaps = normalize_gaps(b.gaps)
    if b.answer is not None:
        b.answer.validate()
        if (b.gold_type == "negative") != (b.answer.check == "forbidden"):
            raise ContractError("gold_type 'negative' and answer.check 'forbidden' go together")
    if b.answer is None and any(isinstance(x, str) for x in b.wrong_answers + b.alternatives):
        raise ContractError(
            "plain-string wrong answers/alternatives need an answer file; "
            "with a custom verifier use Outcome(script=...)"
        )

    has_solve = "solution/solve.sh" in b.files
    kind = b.oracle_kind or ("reference" if has_solve else "answer" if b.answer else "none")
    if kind not in ORACLE_KINDS:
        raise ContractError(f"oracle_kind {kind!r} must be one of {ORACLE_KINDS}")
    if kind == "none" and has_solve:
        raise ContractError("oracle_kind 'none' but files contain solution/solve.sh")
    if kind == "answer" and b.answer is None and not has_solve:
        raise ContractError("oracle_kind 'answer' needs an answer (or your own solve.sh)")
    if kind in ("replay", "fresh", "reference") and not has_solve:
        raise ContractError(f"oracle_kind {kind!r} needs files['solution/solve.sh']")
    b.oracle_kind = kind

    own_env = any(
        k in b.files for k in ("environment/Dockerfile", "environment/docker-compose.yaml")
    )
    env = b.env_source or ("rebuilt" if own_env else "evidence")
    if env not in ENV_SOURCES:
        raise ContractError(f"env_source {env!r} must be one of {ENV_SOURCES}")
    if env == "partial" and not any(g["affects"] != "none" for g in b.gaps):
        raise ContractError("env_source 'partial' needs gaps saying what is missing")
    b.env_source = env

    method = b.grading_method or ("behavioral" if custom else b.answer.check)
    if method not in GRADING_METHODS:
        raise ContractError(f"grading_method {method!r} must be one of {GRADING_METHODS}")
    b.grading_method = method


def materialize(
    b: BuildResult,
    dest: Path,
    ttype: str,
    generator: str,
    sources: list[str],
    spec: dict | None = None,
) -> list[str]:
    """Write the task package to dest. Returns relative paths written."""
    resolve(b)
    custom = b.answer is None
    files: dict[str, str | bytes] = {}
    oracle_value = None
    if b.gold_type == "negative":
        alts = [a for a in b.alternatives if isinstance(a, str)]
        if not alts:
            raise ContractError(
                "a negative task needs `alternatives`: valid answers (the oracle writes the first)"
            )
        oracle_value = alts[0]
        if oracle_value.strip() in [str(x).strip() for x in b.answer.expected]:
            raise ContractError("the oracle's valid answer must differ from the known-wrong ones")
        if b.answer.choices and oracle_value.strip() not in [c.strip() for c in b.answer.choices]:
            raise ContractError("the oracle's valid answer must be one of answer.choices")

    if b.instruction is not None:
        text = b.instruction.rstrip() + "\n"
        if b.answer is not None and not custom:
            ans_path = f"{ANSWER_DIR}/{b.answer.file}"
            if ans_path not in text:
                text += (
                    f"\nWrite your final answer to `{ans_path}` (just the answer, nothing else).\n"
                )
        files["instruction.md"] = text
    for k, v in b.evidence.items():
        if not isinstance(v, str | bytes):
            raise ContractError(f"evidence {k!r} must be str or bytes, got {type(v).__name__}")
        files[f"environment/data/{k}"] = v
    files.update(b.files)  # generator overrides win

    files.setdefault(
        "task.toml", task_toml(b, ttype, generator, sources, separate=not custom, spec=spec)
    )
    if b.answer is not None:
        files.setdefault("tests/expected.json", json.dumps(b.answer.to_dict(), indent=2) + "\n")
    if not custom:
        files.setdefault("tests/grade.py", grade_source())
        files.setdefault("tests/test.sh", TEST_SH)
        files.setdefault("tests/Dockerfile", TESTS_DOCKERFILE)
    if b.oracle_kind == "answer":
        files.setdefault("solution/solve.sh", solve_sh(b.answer, oracle_value, b.gold_type))
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
    test_sh = task_dir / "tests/test.sh"
    if (
        test_sh.is_file()
        and "expected.json" in test_sh.read_text(errors="replace")
        and not (task_dir / "tests/expected.json").is_file()
    ):
        problems.append("missing tests/expected.json (tests/test.sh reads it)")
    oracle = cfg.get("metadata", {}).get("l2h", {}).get("oracle", "")
    if oracle != "none" and not (task_dir / "solution/solve.sh").is_file():
        problems.append("missing solution/solve.sh (or set oracle_kind='none')")
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
                **{
                    k: spec[k]
                    for k in ("expected", "check", "file", "normalize", "pattern", "choices")
                    if k in spec
                }
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
