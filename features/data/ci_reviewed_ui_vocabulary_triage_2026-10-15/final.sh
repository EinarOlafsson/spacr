#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_i18n_reviewed_ui.py::test_reviewed_ui_vocabulary_is_complete_and_pinned > /mnt/wd4tb/scratch/ci-reviewed-ui-vocabulary-20261015/final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-reviewed-ui-vocabulary-20261015/final.exit
exit "$status"
