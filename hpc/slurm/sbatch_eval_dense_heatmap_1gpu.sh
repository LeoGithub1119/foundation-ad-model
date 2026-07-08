#!/usr/bin/env bash
#SBATCH -J dino-eval-dense
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH --account=mst114553
#SBATCH -o /work/foobarbaz911/dino/logs/dino_eval_dense_%j.out
#SBATCH -e /work/foobarbaz911/dino/logs/dino_eval_dense_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

./hpc/bin/run_eval_dense_heatmap.sh
