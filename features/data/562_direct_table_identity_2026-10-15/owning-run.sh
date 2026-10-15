#!/bin/bash
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /mnt/wd4tb/spacr-worktrees/codex-f562-direct-table-identity-20261015
set +e
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_multiple_instance_learning.py -k direct_mil --tb=short --junitxml=/mnt/wd4tb/scratch/f562-direct-table-identity-20261015/after.xml > /mnt/wd4tb/scratch/f562-direct-table-identity-20261015/after.log 2>&1
status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/f562-direct-table-identity-20261015/after.exit
exit "$status"
