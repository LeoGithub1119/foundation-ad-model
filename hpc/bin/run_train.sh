#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"

RUN_TAG="${RUN_TAG:-${EXP_NAME:-train}}"
OUT_DIR="$WORK_DIR/outputs/$RUN_TAG"
mkdir -p "$OUT_DIR"

MODEL_PATH="${MODEL_PATH:-$WORK_DIR/models/$(basename "$MODEL_ID")}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_DIR/datasets/VisA}"
SPLIT_CSV="${SPLIT_CSV:-split_csv/1cls.csv}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[RUN] EXP_NAME=${EXP_NAME:-}"
echo "[RUN] MODEL_PATH=$MODEL_PATH"
echo "[RUN] DATASET_ROOT=$DATASET_ROOT"
echo "[RUN] SPLIT_CSV=$SPLIT_CSV"
echo "[RUN] OUT_DIR=$OUT_DIR"

args=(
  --dataset-root "$DATASET_ROOT"
  --split-csv "$SPLIT_CSV"
  --model-path "$MODEL_PATH"
  --output-dir "$OUT_DIR"
  --head "${HEAD:-linear}"
  --feature "${FEATURE:-cls}"
  --epochs "${EPOCHS:-5}"
  --batch-size "${BATCH_SIZE:-16}"
  --num-workers "${NUM_WORKERS:-4}"
  --lr "${LR:-1e-3}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --seed "${SEED:-42}"
)

if [ "${AMP:-0}" = "1" ]; then
  args+=(--amp)
fi

if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
  args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi

if [ -n "${MAX_TEST_SAMPLES:-}" ]; then
  args+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

if [ -n "${MAX_TRAIN_SAMPLES_PER_LABEL:-}" ]; then
  args+=(--max-train-samples-per-label "$MAX_TRAIN_SAMPLES_PER_LABEL")
fi

if [ -n "${MAX_TEST_SAMPLES_PER_LABEL:-}" ]; then
  args+=(--max-test-samples-per-label "$MAX_TEST_SAMPLES_PER_LABEL")
fi

python -m dino_ad.train_visa "${args[@]}"
