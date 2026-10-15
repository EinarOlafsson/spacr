#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 2G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n692-current-state-reconciliation-20261015/check-scoped-api.py > /mnt/wd4tb/scratch/n692-current-state-reconciliation-20261015/scoped-api.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n692-current-state-reconciliation-20261015/scoped-api.exit
exit "$result"
