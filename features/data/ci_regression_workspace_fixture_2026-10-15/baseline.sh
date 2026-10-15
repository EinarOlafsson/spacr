#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_regression_results_uncovered_paths.py::test_a_workspace_state_keeps_only_the_runs_it_can_read > /mnt/wd4tb/scratch/ci-regression-workspace-fixture-20261015/baseline.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-regression-workspace-fixture-20261015/baseline.exit
exit "$status"
