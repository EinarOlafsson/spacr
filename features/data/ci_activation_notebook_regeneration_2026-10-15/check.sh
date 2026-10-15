#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python tools/build_notebook_settings.py --check > /mnt/wd4tb/scratch/ci-notebook-settings-current-20261015/check.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-notebook-settings-current-20261015/check.exit
exit "$status"
