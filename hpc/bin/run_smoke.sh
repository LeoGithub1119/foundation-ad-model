#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
source "$SCRIPT_DIR/activate.sh"

MODEL_PATH="${MODEL_PATH:-$WORK_DIR/models/$(basename "$MODEL_ID")}"
DATASET_ROOT="${DATASET_ROOT:-$WORK_DIR/datasets/VisA}"
SMOKE_IMAGE="${SMOKE_IMAGE:-$DATASET_ROOT/candle/Data/Images/Normal/0000.JPG}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

echo "[SMOKE] MODEL_PATH=$MODEL_PATH"
echo "[SMOKE] SMOKE_IMAGE=$SMOKE_IMAGE"

python "$PROJECT_ROOT/scripts/dinov3_feature_smoke.py" \
  --model-path "$MODEL_PATH" \
  --image "$SMOKE_IMAGE"
