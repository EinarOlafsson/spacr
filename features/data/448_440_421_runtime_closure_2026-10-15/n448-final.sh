#!/usr/bin/env bash
set -u
cd /mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015
TASK_SCRATCH=/mnt/wd4tb/scratch/n448-n440-n421-runtime-20261015
export CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTEST_QT_API=pyside6 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SPACR_NO_GL=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=/mnt/wd4tb/scratch/codex-home-release-1515-20261013/a744-dark-opaque-p1-20261014/qt612-runtime/site-packages
export TMPDIR=$TASK_SCRATCH/consumer-tmp XDG_CONFIG_HOME=$TASK_SCRATCH/consumer-config XDG_CACHE_HOME=$TASK_SCRATCH/consumer-cache PYTHONPYCACHEPREFIX=$TASK_SCRATCH/consumer-pycache
mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
TASK_CPU=$(/home/olafsson/anaconda3/envs/spacr/bin/python -c 'import os; print(min(os.sched_getaffinity(0)))')
exec 9>/mnt/wd4tb/scratch/original-home-ci-20261015/heavy.lock
flock 9
tools/run_capped.sh 4G taskset -c "$TASK_CPU" /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -p pytestqt.plugin -p pytest_timeout tests/qt/test_n448_curation_status_locales.py -q --tb=short --junitxml="$TASK_SCRATCH/n448-final.xml" --basetemp="$TASK_SCRATCH/n448-final-pytest-tmp" -o cache_dir="$TASK_SCRATCH/n448-final-pytest-cache" > "$TASK_SCRATCH/n448-final.log" 2>&1
TASK_RC=$?
printf '%s\n' "$TASK_RC" > "$TASK_SCRATCH/n448-final.exit"
exit "$TASK_RC"
