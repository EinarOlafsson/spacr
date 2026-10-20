#!/usr/bin/env bash
set -euo pipefail
exec 8>/mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock
flock -n 8
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock -n 9
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
set +e
/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015/tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/api-manifest-relative-gap-census-20261015/read.py > /mnt/wd4tb/scratch/api-manifest-relative-gap-census-20261015/read.log 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > /mnt/wd4tb/scratch/api-manifest-relative-gap-census-20261015/read.exit
exit "$TASK_EXIT"
