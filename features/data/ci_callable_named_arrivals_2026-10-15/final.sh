#!/bin/bash
set -euo pipefail
ulimit -c 0
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
status=0
flock /mnt/wd4tb/scratch/home-b-coordination-20261014/heavy.lock taskset -c 0 tools/run_capped.sh 8G /home/olafsson/anaconda3/bin/python -m pytest -q tests/test_docstring_correctness.py::test_public_callable_inventory_is_source_derived_not_docstring_derived tests/test_docstring_correctness.py::test_callable_boundary_is_cross_checked_with_i18n_extractor > /mnt/wd4tb/scratch/ci-callable-arrival-wiring-proposal-20261015/final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-callable-arrival-wiring-proposal-20261015/final.exit
exit "$status"
