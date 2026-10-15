#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 6G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/test_n702_verified_picker_model_pins.py::test_cancellation_before_between_or_after_reads_never_issues_a_pin > /mnt/wd4tb/scratch/n702-picker-pin-provenance-20261015/cancel-final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/n702-picker-pin-provenance-20261015/cancel-final.exit
exit "$status"
