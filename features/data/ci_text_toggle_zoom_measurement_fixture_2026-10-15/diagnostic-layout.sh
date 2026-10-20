#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/ci-text-toggle-zoom-baseline-20261015/diagnostic-layout.py > /mnt/wd4tb/scratch/ci-text-toggle-zoom-baseline-20261015/diagnostic-layout.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-text-toggle-zoom-baseline-20261015/diagnostic-layout.exit
exit "$status"
