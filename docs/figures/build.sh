#!/usr/bin/env bash
# Rebuild the README figures: every fig*.tex here becomes a PDF and a PNG in docs/assets.
# Needs a LaTeX engine (tectonic by default; TECTONIC=/path/to/tectonic to override)
# and pdftocairo (poppler-utils).
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${TECTONIC:-tectonic}"
OUT=../assets
mkdir -p "$OUT"
for tex in fig*.tex; do
  name="${tex%.tex}"
  "$TECTONIC" --chatter minimal --outdir "$OUT" "$tex"
  pdftocairo -png -singlefile -r "${DPI:-400}" "$OUT/$name.pdf" "$OUT/$name"
done
