#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/n421-legacy-output-i18n-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 6G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p pytest_timeout tests/qt/test_n421_legacy_measurement_output_translations.py::test_scan_failure_writes_are_localized_without_changing_scan_flow -q --tb=short --junitxml="$TASK_SCRATCH/baseline-scan-corrected.xml" --basetemp="$TASK_SCRATCH/baseline-scan-corrected-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/baseline-scan-corrected.log" 2>&1
TASK_RC=$?
printf '%s\n' "$TASK_RC" > "$TASK_SCRATCH/baseline-scan-corrected.exit"
exit "$TASK_RC"
