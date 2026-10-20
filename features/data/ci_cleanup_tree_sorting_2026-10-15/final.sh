#!/bin/bash
set -euo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
status=0
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock taskset -c 0 tools/run_capped.sh 8G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_every_table_asks_for_the_sorting_contract.py::test_every_view_in_the_source_asks_for_the_sorting_contract tests/qt/test_every_table_asks_for_the_sorting_contract.py::test_nothing_builds_a_bare_table_cell_any_more tests/qt/test_n703_optional_cleanup_dialog.py > /mnt/wd4tb/scratch/ci-cleanup-tree-sorting-20261015/final.log 2>&1 || status=$?
printf '%s\n' "$status" > /mnt/wd4tb/scratch/ci-cleanup-tree-sorting-20261015/final.exit
exit "$status"
