#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen
/home/olafsson/anaconda3/envs/spacr/bin/python -m mypy --cache-dir=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015/mypy-cache tests/test_ci_suite_classification.py > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/baseline.log 2>&1
TASK_RC=$?
echo "$TASK_RC" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/baseline.exit
exit "$TASK_RC"
