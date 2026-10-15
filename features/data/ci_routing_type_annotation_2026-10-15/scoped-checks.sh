#!/bin/bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
export MYPYPATH=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015/tool-stubs
/home/olafsson/anaconda3/envs/spacr/bin/python -m mypy --cache-dir=/mnt/wd4tb/scratch/ci-lint-38699600639-20261015/mypy-cache tests/test_ci_suite_classification.py > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-mypy.log 2>&1
echo "$?" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-mypy.exit
/home/olafsson/.local/bin/ruff check --no-cache tests/test_ci_suite_classification.py > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-ruff.log 2>&1
echo "$?" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-ruff.exit
/home/olafsson/anaconda3/envs/spacr/bin/python -m xenon --max-absolute B --max-modules B --max-average A tests/test_ci_suite_classification.py > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-xenon.log 2>&1
echo "$?" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/final-xenon.exit
