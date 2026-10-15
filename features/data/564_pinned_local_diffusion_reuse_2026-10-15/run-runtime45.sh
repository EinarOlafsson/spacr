#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/scoped-runtime45.py --write > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/runtime45.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015/runtime45.exit
exit "$status"
