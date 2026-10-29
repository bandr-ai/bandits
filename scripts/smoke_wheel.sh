#!/usr/bin/env bash
# Install a built wheel into a fresh environment and run it away from the
# checkout, the way a user would after `pip install bandits-cli`.
#
#   scripts/smoke_wheel.sh dist/bandits_cli-*.whl
set -euo pipefail

wheel="$(realpath "$1")"
fixture="$(realpath "$(dirname "$0")/../tests/fixtures/traces.otlp.jsonl")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel"
bandits="$work/venv/bin/bandits"

cd "$work"
expected="$("$work/venv/bin/python" -c 'import importlib.metadata as m; print(m.version("bandits-cli"))')"
"$bandits" --version | grep -qx "bandits $expected"
"$bandits" --help >/dev/null
"$bandits" ingest "$fixture" --source otlp | tee ingest.log
corpus="$(sed -n 's/^artifact_id: //p' ingest.log)"
test -n "$corpus"
"$bandits" show "$corpus" >/dev/null
"$bandits" list | grep -q "$corpus"
echo "wheel smoke test passed: bandits $expected"
