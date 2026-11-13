#!/usr/bin/env bash
# Install a built wheel into a fresh environment and ingest a fixture away
# from the checkout, the way a user would after `pip install bandr`.
#
#   scripts/smoke_wheel.sh dist/bandr-*.whl
set -euo pipefail

wheel="$(realpath "$1")"
fixture="$(realpath "$(dirname "$0")/../tests/fixtures/traces.otlp.jsonl")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel"

cd "$work"
"$work/venv/bin/python" - "$fixture" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

import bandits
from bandits.ingest import load_corpus
from bandits.store import ArtifactStore

assert bandits.__version__ == importlib.metadata.version("bandr"), bandits.__version__
corpus = load_corpus(Path(sys.argv[1]), source="otlp")
assert corpus.traces, "no traces read from the fixture"
store = ArtifactStore(Path("project"))
envelope = store.write(corpus, source_path=sys.argv[1])
assert [e.artifact_id for e in store.list()] == [envelope.artifact_id]
assert len(store.read(envelope.artifact_id).traces) == len(corpus.traces)
print(f"wheel smoke test passed: bandits {bandits.__version__}, {len(corpus.traces)} traces")
PY

# mint ships its skill, guides and grader inside the wheel and runs from the entry point.
"$work/venv/bin/mint" docs --topic harbor >/dev/null
"$work/venv/bin/mint" schema --json >/dev/null
echo "wheel smoke test passed: mint"


# Skill installation must work from wheel resources, outside a Mint workspace.
"$work/venv/bin/mint" --version
"$work/venv/bin/mint" skill install --project
"$work/venv/bin/mint" skill install --project
"$work/venv/bin/python" - <<'PYCODE'
from importlib import resources
from pathlib import Path
assert Path(".claude/skills/mint/SKILL.md").read_bytes() == resources.files("mint").joinpath("SKILL.md").read_bytes()
PYCODE
