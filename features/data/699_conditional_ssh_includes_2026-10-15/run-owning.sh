#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n699-conditional-includes-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_n699_conditional_ssh_includes.py tests/test_ssh_discovery_mfa_699.py > /mnt/wd4tb/scratch/n699-conditional-include-20261015/owning.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n699-conditional-include-20261015/owning.exit
exit "$status"
