#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n690-current-contract-audit-20261015/normal-noop-and-audit.py > /mnt/wd4tb/scratch/n690-current-contract-audit-20261015/normal-noop-selected-audit.log 2>&1
TASK_RC=$?
echo "$TASK_RC" > /mnt/wd4tb/scratch/n690-current-contract-audit-20261015/normal-noop-selected-audit.exit
exit "$TASK_RC"
