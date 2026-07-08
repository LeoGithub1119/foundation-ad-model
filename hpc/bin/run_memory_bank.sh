#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"

RUN_TAG="${RUN_TAG:-${EXP_NAME:-dino_memory_bank}}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/outputs/$RUN_TAG}"
MODEL_PATH="${MODEL_PATH:-$WORK_DIR/models/$(basename "$MODEL_ID")}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_DIR/datasets/VisA}"
SPLIT_CSV="${SPLIT_CSV:-split_csv/2cls_highshot.csv}"

mkdir -p "$OUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[MEMORY_BANK] EXP_NAME=${EXP_NAME:-}"
echo "[MEMORY_BANK] MEMORY_DATASET=${MEMORY_DATASET:-visa}"
echo "[MEMORY_BANK] MODEL_PATH=$MODEL_PATH"
echo "[MEMORY_BANK] DATASET_ROOT=$DATASET_ROOT"
echo "[MEMORY_BANK] OUT_DIR=$OUT_DIR"

args=(
  --dataset "${MEMORY_DATASET:-visa}"
  --dataset-root "$DATASET_ROOT"
  --split-csv "$SPLIT_CSV"
  --model-path "$MODEL_PATH"
  --output-dir "$OUT_DIR"
  --batch-size "${BATCH_SIZE:-16}"
  --num-workers "${NUM_WORKERS:-4}"
  --max-bank-patches "${MAX_BANK_PATCHES:-20000}"
  --top-k "${TOP_K:-6}"
  --eval-size "${HEATMAP_EVAL_SIZE:-224}"
  --hist-bins "${HIST_BINS:-4096}"
  --distance-chunk-size "${DISTANCE_CHUNK_SIZE:-1024}"
  --seed "${SEED:-42}"
)

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
  args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi

if [ -n "${MAX_TEST_SAMPLES:-}" ]; then
  args+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

if [ "${AMP:-0}" = "1" ]; then
  args+=(--amp)
fi

python -m dino_ad.eval_memory_bank "${args[@]}"
