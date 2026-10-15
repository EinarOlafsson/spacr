#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_cov_w2_6_shortcuts.py::test_the_map_is_the_window_wide_keys_plus_the_per_screen_ones tests/test_documentation_i18n.py::test_window_wide_shortcut_keys_keep_keyboard_sense > /mnt/wd4tb/scratch/ci-shortcut-contracts-20261015/after.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-shortcut-contracts-20261015/after.exit
exit "$result"
