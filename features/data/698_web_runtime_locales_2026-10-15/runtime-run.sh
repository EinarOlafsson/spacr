#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015
set +e
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/n698-web-localization-20261015/scoped-runtime.py --write > /mnt/wd4tb/scratch/n698-web-localization-20261015/runtime.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-web-localization-20261015/runtime.exit
exit "$status"
