#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015 || exit 90
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
tools/run_capped.sh 2G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/produce-three-tab-api.py > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-producer.log 2>&1
rc=$?
printf '%s\n' "$rc" > /mnt/wd4tb/scratch/n468-current-contract-audit-20261015/three-tab-producer.exit
exit "$rc"
