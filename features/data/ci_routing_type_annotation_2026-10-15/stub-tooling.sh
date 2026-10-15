#!/bin/bash
set -u
/home/olafsson/anaconda3/envs/spacr/bin/python -m pip install --disable-pip-version-check --no-deps --no-cache-dir --target /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/tool-stubs types-PyYAML==6.0.12.20260906 > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/stub-tooling.log 2>&1
TASK_RC=$?
echo "$TASK_RC" > /mnt/wd4tb/scratch/ci-lint-38699600639-20261015/stub-tooling.exit
exit "$TASK_RC"
