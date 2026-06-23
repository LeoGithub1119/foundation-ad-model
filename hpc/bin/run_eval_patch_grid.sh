#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"

RUN_TAG="${RUN_TAG:-${EXP_NAME:-dino_visa_a0_linear}}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/outputs/$RUN_TAG}"
MODEL_PATH="${MODEL_PATH:-$WORK_DIR/models/$(basename "$MODEL_ID")}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_DIR/datasets/VisA}"
SPLIT_CSV="${SPLIT_CSV:-split_csv/2cls_highshot.csv}"
CHECKPOINT="${CHECKPOINT:-$OUT_DIR/best_head.pt}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

mkdir -p "$OUT_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[PATCH_GRID_EVAL] EXP_NAME=${EXP_NAME:-}"
echo "[PATCH_GRID_EVAL] MODEL_PATH=$MODEL_PATH"
echo "[PATCH_GRID_EVAL] DATASET_ROOT=$DATASET_ROOT"
echo "[PATCH_GRID_EVAL] SPLIT_CSV=$SPLIT_CSV"
echo "[PATCH_GRID_EVAL] CHECKPOINT=$CHECKPOINT"
echo "[PATCH_GRID_EVAL] OUT_DIR=$OUT_DIR"
echo "[PATCH_GRID_EVAL] EVAL_SPLIT=$EVAL_SPLIT"

args=(
  --dataset-root "$DATASET_ROOT"
  --split-csv "$SPLIT_CSV"
  --model-path "$MODEL_PATH"
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUT_DIR"
  --split "$EVAL_SPLIT"
  --batch-size "${BATCH_SIZE:-16}"
  --num-workers "${NUM_WORKERS:-4}"
)

if [ -n "${MAX_EVAL_SAMPLES:-}" ]; then
  args+=(--max-samples "$MAX_EVAL_SAMPLES")
fi

if [ -n "${MAX_EVAL_SAMPLES_PER_LABEL:-}" ]; then
  args+=(--max-samples-per-label "$MAX_EVAL_SAMPLES_PER_LABEL")
fi

if [ "${AMP:-0}" = "1" ]; then
  args+=(--amp)
fi

if [ "${WRITE_PATCH_ROWS:-0}" = "1" ]; then
  args+=(--write-rows)
fi

python -m dino_ad.eval_visa_patch_grid "${args[@]}"
