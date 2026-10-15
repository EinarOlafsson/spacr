#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c 0 tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_docs_media_budget.py::test_every_lesson_keeps_its_video_and_its_poster tests/test_docs_media_budget.py::test_narration_is_the_stable_mobile_clock > /mnt/wd4tb/scratch/ci-current-published-media-selector-20261015/baseline.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-current-published-media-selector-20261015/baseline.exit
exit "$status"
