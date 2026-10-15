#!/usr/bin/env bash
set -uo pipefail
cd /mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=/mnt/wd4tb/scratch/n702-cli-pins-20261014
ci_cpu_index=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 8G taskset -c "$ci_cpu_index" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q tests/qt/test_fold_strip_uncovered_paths.py::test_the_registry_answers_a_folded_button_with_name_sentence_and_stage tests/qt/test_cov_wf_qt_app.py::test_a_theme_refresh_skips_screens_that_cannot_hide_immature_apps > /mnt/wd4tb/scratch/ci-registry-theme-contracts-20261015/fixed.log 2>&1
result=$?
printf '%s\n' "$result" > /mnt/wd4tb/scratch/ci-registry-theme-contracts-20261015/fixed.exit
exit "$result"
