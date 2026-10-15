#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
set +e
/usr/bin/time -v tools/run_capped.sh 2G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/audit-selected-api.py guarded > /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/normal-selected-guarded.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/normal-selected-guarded.exit
exit "$result"
