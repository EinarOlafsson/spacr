#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_CPU=$(/home/olafsson/anaconda3/bin/python -c 'import os;print(min(os.sched_getaffinity(0)))')
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c "$TASK_CPU" tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_built_nested_helper_api.py::test_empty_rollout_builds_without_helpers_and_keeps_private_api_hidden tests/test_built_nested_helper_api.py::test_changing_only_the_rollout_invalidates_sphinx_incremental_state tests/test_built_nested_helper_api.py::test_real_google_style_helpers_render_both_parameter_contracts tests/test_built_nested_helper_api.py::test_tabular_and_palette_helpers_render_without_exposing_private_parents > /mnt/wd4tb/scratch/ci-nested-helper-fixture-20261015/final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-nested-helper-fixture-20261015/final.exit
exit "$status"
