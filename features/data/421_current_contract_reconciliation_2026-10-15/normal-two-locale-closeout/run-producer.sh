#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
set +e
/usr/bin/time -v tools/run_capped.sh 2G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n421-current-contract-audit-20261015/produce-two.py > /mnt/wd4tb/scratch/n421-current-contract-audit-20261015/normal-producer.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/n421-current-contract-audit-20261015/normal-producer.exit
exit "$result"
