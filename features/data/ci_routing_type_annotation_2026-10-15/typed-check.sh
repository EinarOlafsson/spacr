#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
/home/olafsson/anaconda3/envs/spacr/bin/python -m mypy --python-executable=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015/tool-runtime/bin/python --cache-dir=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015/mypy-typed-cache tests/test_ci_suite_classification.py > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-mypy-typed.log 2>&1
TASK_RC=$?
echo "$TASK_RC" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-mypy-typed.exit
exit "$TASK_RC"
