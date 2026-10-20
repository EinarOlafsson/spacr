#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
ulimit -c 0
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock tools/run_capped.sh 6G /home/olafsson/anaconda3/bin/python /mnt/wd4tb/scratch/n441-normal-runtime-closure-20261015/read-only-preflight.py /mnt/wd4tb/scratch/n441-normal-runtime-closure-20261015/preflight-config.FINAL.json > /mnt/wd4tb/scratch/n441-normal-runtime-closure-20261015/actual-read-only.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n441-normal-runtime-closure-20261015/actual-read-only.exit
exit "$status"
