#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-aliased-user-data-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/ci-historical-progress-branding-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TMPDIR=$TASK_SCRATCH/tmp XDG_CONFIG_HOME=$TASK_SCRATCH/config XDG_CACHE_HOME=$TASK_SCRATCH/cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
flock /mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock tools/run_capped.sh 4G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/test_the_name_is_spaCR.py::test_no_file_mis_cases_the_name -q --tb=short --junitxml="$TASK_SCRATCH/baseline.xml" --basetemp="$TASK_SCRATCH/baseline-tmp" -o cache_dir="$TASK_SCRATCH/pytest-cache" > "$TASK_SCRATCH/baseline.log" 2>&1
TASK_RC=$?
printf '%s\n' "$TASK_RC" > "$TASK_SCRATCH/baseline.exit"
exit "$TASK_RC"
