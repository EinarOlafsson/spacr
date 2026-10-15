#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c 0 tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_prefixed_backend_rows_offer_their_backend.py::test_the_zoo_buttons_ask_for_the_prefixed_kinds tests/qt/test_cellpose_dino_rows_offer_their_backend.py::test_the_preview_zoo_button_asks_for_dino_rows tests/qt/test_cellpose_dino_rows_offer_their_backend.py::test_use_writes_the_cellpose_dino_setting > /mnt/wd4tb/scratch/ci-prefixed-dino-picker-fixtures-20261015/baseline.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-prefixed-dino-picker-fixtures-20261015/baseline.exit
exit "$status"
