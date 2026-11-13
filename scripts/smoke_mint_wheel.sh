#!/usr/bin/env bash
# Install the standalone bandr-mint wheel into a fresh environment and run it
# away from the checkout, the way a user would after `pip install bandr-mint`.
#
#   scripts/smoke_mint_wheel.sh dist-mint/bandr_mint-*.whl
set -euo pipefail

wheel="$(realpath "$1")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel"

cd "$work"
"$work/venv/bin/python" - <<'PY'
import importlib.metadata
import importlib.util
from importlib import resources

import mint

dist = importlib.metadata.distribution("bandr-mint")
assert mint.__version__ == dist.version, (mint.__version__, dist.version)
# Mint only: no ingestion library and no runtime dependencies.
assert importlib.util.find_spec("bandits") is None, "bandr-mint must not ship bandits"
assert not dist.requires, dist.requires
# The skill, guides and grader templates are package resources.
root = resources.files("mint")
for name in ("SKILL.md", "guides/harbor.md", "templates/grade.py"):
    assert root.joinpath(name).is_file(), name
print(f"wheel smoke test passed: bandr-mint {dist.version}")
PY

"$work/venv/bin/mint" --version
"$work/venv/bin/mint" docs --topic harbor >/dev/null
"$work/venv/bin/mint" schema --json >/dev/null
"$work/venv/bin/mint" skill install --project
"$work/venv/bin/python" - <<'PY'
from importlib import resources
from pathlib import Path

installed = Path(".claude/skills/mint/SKILL.md").read_bytes()
assert installed == resources.files("mint").joinpath("SKILL.md").read_bytes()
PY
echo "wheel smoke test passed: bandr-mint skill install"
