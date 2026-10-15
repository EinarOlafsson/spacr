#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
tools/run_capped.sh 2G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n698-model-scale-owner-proposal-20261015/validate.py > /mnt/wd4tb/scratch/n698-model-scale-owner-proposal-20261015/validate.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-model-scale-owner-proposal-20261015/validate.exit
exit "$status"
