#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock taskset -c 0 tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python /mnt/wd4tb/scratch/ci-callable-boundary-audit-20261015/driver.py > /mnt/wd4tb/scratch/ci-callable-boundary-audit-20261015/run.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-callable-boundary-audit-20261015/run.exit
exit "$status"
