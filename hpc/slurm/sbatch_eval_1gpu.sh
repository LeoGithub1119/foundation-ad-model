#!/usr/bin/env bash
#SBATCH -J dino_eval
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --account=mst114553
#SBATCH -o /work/foobarbaz911/dino/logs/dino_eval_%j.out
#SBATCH -e /work/foobarbaz911/dino/logs/dino_eval_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$REPO_ROOT" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

echo "[SBATCH] JOBID=${SLURM_JOB_ID}"
echo "[SBATCH] REPO_ROOT=${REPO_ROOT}"

bash "${REPO_ROOT}/hpc/bin/run_eval.sh"
