#!/usr/bin/env bash
# The first real-model run (#95), in one command on a fresh GPU box.
#
#   1. install the recipe with the train extra
#   2. gather the Bandits projects (corpora + judge runs) into one project
#   3. build one dataset from the judge runs (split by trace, merged)
#   4. smoke-test each model (tokenizer, letter mass, loss falls) and stop on failure
#   5. `jev run` for each model: untrained, train, calibrate, trained, report
#   6. pack the reports and artifacts into one tarball to copy back
#
# Run from the repo root. Configuration is by environment variable:
#
#   PROJECTS     space-separated Bandits project dirs, each holding .bandits (required)
#   JUDGE_RUNS   space-separated turn-judge run ids inside them (required)
#   MODELS       default: "Qwen/Qwen3.5-4B-Base Qwen/Qwen3.5-4B"
#   SEED         default: 1
#   GPU_USD_PER_HOUR                         the box's hourly price, for the cost columns (optional)
#   LEDGERS                                  space-separated judge ledgers, to price the verifier (optional)
#   INPUT_USD_PER_MTOK, OUTPUT_USD_PER_MTOK  the judge model's prices, required with LEDGERS
#   FAST_KERNELS=1                           also install flash-linear-attention and causal-conv1d
#   HELD_OUT_RUNS  judge run ids (a subset of JUDGE_RUNS) for the unseen-source check: a second
#                  run per model trains without them and scores them as their own report section
#   DEVICE, DTYPE  default: cuda, bfloat16 (DEVICE=cpu DTYPE=float32 with a tiny model rehearses
#                  the whole script without a GPU)
#   SKIP_INSTALL=1 reuse the recipe's existing environment as is
#   OUT          default: runs/jev-<UTC timestamp>
#
# Example:
#   PROJECTS="work/trail-ns work/tau2-ns" \
#   JUDGE_RUNS="turn-judge-0044b8f5c041155c turn-judge-b83480800ad90a25 turn-judge-d450cf34d97b27a2" \
#   GPU_USD_PER_HOUR=1.2 recipes/jev/scripts/gpu_run.sh
set -euo pipefail

: "${PROJECTS:?set PROJECTS to the Bandits project dirs}"
: "${JUDGE_RUNS:?set JUDGE_RUNS to the turn-judge run ids}"
MODELS="${MODELS:-Qwen/Qwen3.5-4B-Base Qwen/Qwen3.5-4B}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OUT="$(realpath -m "${OUT:-runs/jev-$(date -u +%Y%m%dT%H%M%SZ)}")"
RECIPE="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$OUT/project"

log() { printf '\n== %s\n' "$*"; }

if [[ "$DEVICE" == cuda* ]]; then
  log "GPU"
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  log "install"
  (cd "$RECIPE" && uv sync --extra dev --extra train)
fi
if [[ "${FAST_KERNELS:-0}" == "1" ]]; then
  (cd "$RECIPE" && uv pip install flash-linear-attention causal-conv1d) \
    || echo "fast kernels did not install; continuing with the slower reference implementation"
fi
jev() { (cd "$RECIPE" && uv run --no-sync jev "$@"); }
# Run a jev command, show its output, and print only the dataset id it made.
# (Not `tee /dev/stderr`: that reopens a redirected log file and truncates it.)
dataset_id_of() {
  local out
  out="$(jev "$@")"
  printf '%s\n' "$out" >&2
  printf '%s\n' "$out" | awk '/^decision_dataset_id:/ {print $2}'
}

log "gather projects into $PROJECT"
mkdir -p "$PROJECT/.bandits"
for p in $PROJECTS; do
  cp -rn "$(realpath "$p")/.bandits/." "$PROJECT/.bandits/"
done

log "dataset"
datasets=()
for run in $JUDGE_RUNS; do
  id="$(dataset_id_of dataset "$run" --project "$PROJECT")"
  datasets+=("$id")
done
if (( ${#datasets[@]} > 1 )); then
  DATASET="$(dataset_id_of merge "${datasets[@]}" --project "$PROJECT")"
else
  DATASET="${datasets[0]}"
fi
[[ -n "$DATASET" ]] || { echo "could not read the dataset id" >&2; exit 1; }
echo "dataset: $DATASET"

TRAIN_WITHOUT_HELD_OUT=""
if [[ -n "${HELD_OUT_RUNS:-}" ]]; then
  kept=()
  for i in "${!datasets[@]}"; do
    run="$(echo $JUDGE_RUNS | cut -d' ' -f$((i + 1)))"
    [[ " $HELD_OUT_RUNS " == *" $run "* ]] || kept+=("${datasets[$i]}")
  done
  if (( ${#kept[@]} > 1 )); then
    TRAIN_WITHOUT_HELD_OUT="$(dataset_id_of merge "${kept[@]}" --project "$PROJECT")"
  else
    TRAIN_WITHOUT_HELD_OUT="${kept[0]}"
  fi
  echo "dataset without held-out sources: $TRAIN_WITHOUT_HELD_OUT"
fi

price_args=()
if [[ -n "${LEDGERS:-}" ]]; then
  : "${INPUT_USD_PER_MTOK:?set INPUT_USD_PER_MTOK with LEDGERS}"
  : "${OUTPUT_USD_PER_MTOK:?set OUTPUT_USD_PER_MTOK with LEDGERS}"
  for l in $LEDGERS; do price_args+=(--ledger "$(realpath "$l")"); done
  price_args+=(--input-usd-per-mtok "$INPUT_USD_PER_MTOK" --output-usd-per-mtok "$OUTPUT_USD_PER_MTOK")
fi
if [[ -n "${GPU_USD_PER_HOUR:-}" ]]; then
  price_args+=(--gpu-usd-per-hour "$GPU_USD_PER_HOUR")
fi

for model in $MODELS; do
  revision="$(curl -fsS "https://huggingface.co/api/models/$model" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
  name="${model//\//__}"
  log "smoke: $model@$revision"
  (cd "$RECIPE" && uv run --no-sync python scripts/smoke.py \
      --project "$PROJECT" --dataset "$DATASET" --model "$model" --revision "$revision" \
      --device "$DEVICE" --dtype "$DTYPE") \
    | tee "$OUT/smoke-$name.json"

  log "jev run: $model@$revision"
  jev run "$DATASET" --model "$model" --revision "$revision" --seed "$SEED" \
    --checkpoint-dir "$OUT/checkpoints/$name" --output "$OUT/reports/$name" \
    --two-order --allow-test ${price_args[@]+"${price_args[@]}"} --device "$DEVICE" --dtype "$DTYPE" --project "$PROJECT" \
    2>&1 | tee "$OUT/run-$name.log"

  if [[ -n "$TRAIN_WITHOUT_HELD_OUT" ]]; then
    held_args=()
    for run in $HELD_OUT_RUNS; do held_args+=(--held-out "$run"); done
    log "jev run without ${HELD_OUT_RUNS}: $model@$revision"
    jev run "$TRAIN_WITHOUT_HELD_OUT" "${held_args[@]}" --allow-test --model "$model" --revision "$revision" --seed "$SEED" \
      --checkpoint-dir "$OUT/checkpoints/$name-held-out" --output "$OUT/reports/$name-held-out" \
      ${price_args[@]+"${price_args[@]}"} --device "$DEVICE" --dtype "$DTYPE" --project "$PROJECT" \
      2>&1 | tee "$OUT/run-$name-held-out.log"
  fi
done

log "pack"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" \
  "$(basename "$OUT")/reports" "$(basename "$OUT")/project/.bandits/derived" \
  $(cd "$(dirname "$OUT")" && ls "$(basename "$OUT")"/*.json "$(basename "$OUT")"/*.log 2>/dev/null)
echo "reports: $OUT/reports"
echo "copy back: $OUT.tar.gz (reports, derived artifacts, smoke results, logs; no checkpoints)"
