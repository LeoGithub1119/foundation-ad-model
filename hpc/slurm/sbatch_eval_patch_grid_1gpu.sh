#!/usr/bin/env bash
#SBATCH -J dino-eval-patch-grid
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --account=mst114553
#SBATCH -o /work/foobarbaz911/dino/logs/dino_eval_patch_grid_%j.out
#SBATCH -e /work/foobarbaz911/dino/logs/dino_eval_patch_grid_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

./hpc/bin/run_eval_patch_grid.sh
