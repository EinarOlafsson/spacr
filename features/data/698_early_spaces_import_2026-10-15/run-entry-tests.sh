#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n698-device-context-cache-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 3G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_web_space_entry_import_order_n698.py > /mnt/wd4tb/scratch/n698-device-context-cache-20261015/entry-final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n698-device-context-cache-20261015/entry-final.exit
exit "$status"
