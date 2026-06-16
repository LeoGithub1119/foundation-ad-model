#!/usr/bin/env bash
set -euo pipefail
salloc -p normal -N 1 --gres=gpu:1 --cpus-per-task=4 -t 00:20:00 --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
