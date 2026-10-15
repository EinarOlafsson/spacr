#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
set +e
/usr/bin/time -v flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/audit-selected-api.py actual > /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/normal-selected-actual.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015/normal-selected-actual.exit
exit "$result"
