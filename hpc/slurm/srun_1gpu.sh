#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
srun --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}" -p normal --gres=gpu:1 -c 8 -t 0:20:00 --pty bash
