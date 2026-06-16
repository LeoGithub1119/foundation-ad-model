#!/usr/bin/env bash
#SBATCH -J dino_train
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o dino_train_%j.out
#SBATCH -e dino_train_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

echo "[SBATCH] JOBID=${SLURM_JOB_ID}"
echo "[SBATCH] REPO_ROOT=${REPO_ROOT}"

bash "${REPO_ROOT}/hpc/bin/run_train.sh"
