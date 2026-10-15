#!/bin/bash
cd /mnt/wd4tb/spacr-worktrees/codex-f562-db-cardinality-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/f562-db-counterexample-20261015.py > /mnt/wd4tb/scratch/f562-db-cardinality-20261015/counterexample.log 2>&1
printf '%s\n' "$?" > /mnt/wd4tb/scratch/f562-db-cardinality-20261015/counterexample.exit
